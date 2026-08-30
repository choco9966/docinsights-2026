"""Codex-assisted visual transcription for an OCR silver reference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .records import deterministic_content_hash, read_jsonl, write_jsonl
from .render import render_pdf

REFERENCE_KIND = "codex-assisted-silver"
ENGINE = "codex-assisted-visual-transcription"
SCHEMA_VERSION = "1.0"
EXPECTED_BLOCK_IDS = tuple(f"b{number:02d}" for number in range(1, 24))
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_MODEL_CONFIG = ('model_reasoning_effort="high"',)
TASK_ID_PATTERN = re.compile(r"^task_[0-9]{6}$")
DISABLED_CODEX_FEATURES = (
    "shell_tool",
    "unified_exec",
    "code_mode_host",
    "apps",
    "browser_use",
    "in_app_browser",
)
PROMPT = """Transcribe the two attached document page images by vision only.
Return every content block in reading order, exactly b01 through b23.
For each block, omit the leading bNN marker from text and normalize whitespace to single spaces.
Keep visible numbers, signs, decimals, currency symbols, units, and punctuation exactly.
Exclude page footers, watermarks, training-copy marks, and other text outside b01 through b23.
Do not reason, solve any problem, infer an answer, identify evidence, or add commentary.
Return only the JSON object required by the supplied output schema.
"""


def run_codex_reference(
    input_path: str | Path,
    output_path: str | Path,
    *,
    documents_root: str | Path | None = None,
    raw_dir: str | Path = "artifacts/ocr/codex-reference-raw",
    schema_path: str | Path | None = None,
    codex_executable: str = "codex",
    poppler_executable: str = "pdftoppm",
    model: str = DEFAULT_MODEL,
    model_config: Sequence[str] = DEFAULT_MODEL_CONFIG,
    dpi: int = 200,
    timeout_seconds: float = 300.0,
    workers: int = 1,
    retry_failed: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> int:
    """Create fail-closed Codex-assisted silver transcriptions from a manifest."""
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between one and four")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    source = Path(input_path)
    destination = Path(output_path)
    schema = (
        Path(schema_path)
        if schema_path is not None
        else Path(__file__).parents[2] / "schemas" / "codex-transcription-response-v1.schema.json"
    )
    schema_bytes = schema.read_bytes()
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    prompt_sha256 = hashlib.sha256(PROMPT.encode()).hexdigest()
    raw_destination = Path(raw_dir)
    root = Path(documents_root).resolve() if documents_root else source.parent.resolve()
    codex_version = _codex_version(codex_executable)
    codex_identity = _executable_identity(codex_executable)
    renderer_identity = _executable_identity(poppler_executable)
    checkpoint = _checkpoint_path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Codex reference input and output paths must differ")
    resume_source = checkpoint if resume and checkpoint.exists() else destination
    existing = _records_by_id(resume_source) if resume and resume_source.exists() else {}

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in read_jsonl(source):
        instance_id = _required_string(task, "instance_id")
        if TASK_ID_PATTERN.fullmatch(instance_id) is None:
            raise ValueError(f"invalid DocSem instance_id: {instance_id}")
        if instance_id in seen:
            raise ValueError(f"duplicate instance_id in Codex reference input: {instance_id}")
        seen.add(instance_id)
        pdf_value = Path(_required_string(task, "document_pdf"))
        pdf_path = (pdf_value if pdf_value.is_absolute() else root / pdf_value).resolve()
        try:
            pdf_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"document_pdf escapes documents_root for {instance_id}") from exc
        input_sha256 = _sha256_file(pdf_path)
        expected_sha256 = task.get("input_pdf_sha256")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError(f"invalid input_pdf_sha256 for {instance_id}")
            if expected_sha256 != input_sha256:
                raise ValueError(f"input PDF SHA-256 mismatch for {instance_id}")
        fingerprint = deterministic_content_hash(
            {
                "input_pdf_sha256": input_sha256,
                "model": model,
                "model_config": list(model_config),
                "codex_version": codex_version,
                "prompt_sha256": prompt_sha256,
                "output_schema_sha256": schema_sha256,
                "dpi": dpi,
                "codex_executable_identity": codex_identity,
                "renderer_executable_identity": renderer_identity,
                "disabled_codex_features": list(DISABLED_CODEX_FEATURES),
            }
        )
        prior = existing.get(instance_id)
        if prior is not None:
            prior_provenance = prior.get("provenance")
            if not isinstance(prior_provenance, Mapping):
                raise ValueError(f"resume record has invalid provenance for {instance_id}")
            if prior_provenance.get("run_fingerprint") != fingerprint:
                raise ValueError(f"resume fingerprint mismatch for {instance_id}")
            if prior.get("status") != "failed" or not retry_failed:
                continue
        tasks.append(
            {
                "instance_id": instance_id,
                "pdf_path": pdf_path,
                "input_pdf_sha256": input_sha256,
                "split": task.get("split"),
                "split_seed": task.get("split_seed"),
                "run_fingerprint": fingerprint,
            }
        )
    if resume and retry_failed:
        tasks.sort(key=lambda task: task["instance_id"] in existing)
    if limit is not None:
        tasks = tasks[:limit]

    def transcribe(task: Mapping[str, Any]) -> dict[str, Any]:
        return _transcribe_one(
            task,
            raw_dir=raw_destination,
            schema_bytes=schema_bytes,
            schema_sha256=schema_sha256,
            prompt_sha256=prompt_sha256,
            codex_executable=codex_executable,
            codex_version=codex_version,
            codex_identity=codex_identity,
            poppler_executable=poppler_executable,
            renderer_identity=renderer_identity,
            model=model,
            model_config=tuple(model_config),
            dpi=dpi,
            timeout_seconds=timeout_seconds,
        )

    current = dict(existing)
    if not resume:
        _atomic_write_jsonl(checkpoint, ())
    elif not checkpoint.exists():
        _atomic_write_jsonl(checkpoint, (current[key] for key in sorted(current)))

    records = map(transcribe, tasks)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = executor.map(transcribe, tasks)
            _checkpoint_results(checkpoint, current, records)
    else:
        _checkpoint_results(checkpoint, current, records)
    checkpoint.replace(destination)
    return len(tasks)


def _transcribe_one(
    task: Mapping[str, Any],
    *,
    raw_dir: Path,
    schema_bytes: bytes,
    schema_sha256: str,
    prompt_sha256: str,
    codex_executable: str,
    codex_version: str,
    codex_identity: Mapping[str, str],
    poppler_executable: str,
    renderer_identity: Mapping[str, str],
    model: str,
    model_config: tuple[str, ...],
    dpi: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    instance_id = str(task["instance_id"])
    started = time.perf_counter()
    provenance = {
        "reference_kind": REFERENCE_KIND,
        "input_pdf_sha256": task["input_pdf_sha256"],
        "split": task.get("split"),
        "split_seed": task.get("split_seed"),
        "model": model,
        "codex_cli_version": codex_version,
        "codex_executable_identity": dict(codex_identity),
        "model_config": list(model_config),
        "prompt_sha256": prompt_sha256,
        "output_schema_sha256": schema_sha256,
        "dpi": dpi,
        "renderer": "poppler-pdftoppm",
        "renderer_executable_identity": dict(renderer_identity),
        "disabled_codex_features": list(DISABLED_CODEX_FEATURES),
        "run_fingerprint": task["run_fingerprint"],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="docinsights-codex-render-") as render_tmp:
            rendered = render_pdf(
                Path(str(task["pdf_path"])),
                render_tmp,
                dpi=dpi,
                executable=poppler_executable,
                timeout_seconds=timeout_seconds,
            )
            if len(rendered) != 2:
                raise ValueError(f"Codex reference requires exactly two pages, got {len(rendered)}")
            with tempfile.TemporaryDirectory(prefix="docinsights-codex-job-") as job_tmp:
                job = Path(job_tmp)
                images = tuple(job / f"page-{number}.png" for number in (1, 2))
                for source, target in zip(rendered, images, strict=True):
                    shutil.copyfile(source, target)
                job_schema = job / "response.schema.json"
                job_schema.write_bytes(schema_bytes)
                response_path = job / "response.json"
                image_hashes = [
                    {"page_number": number, "sha256": _sha256_file(path)}
                    for number, path in enumerate(images, 1)
                ]
                provenance["input_image_sha256"] = image_hashes
                argv = _codex_argv(
                    codex_executable=codex_executable,
                    job_dir=job,
                    images=images,
                    schema_path=job_schema,
                    response_path=response_path,
                    model=model,
                    model_config=model_config,
                )
                completed = subprocess.run(
                    argv,
                    input=PROMPT,
                    cwd=job,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                raw_response = response_path.read_text(encoding="utf-8")
                blocks = _validated_blocks(raw_response)
                _write_raw_response(raw_dir, instance_id, raw_response, completed)
        provenance["raw_response_sha256"] = hashlib.sha256(raw_response.encode()).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "reference_kind": REFERENCE_KIND,
            "instance_id": instance_id,
            "blocks": blocks,
            "engine": ENGINE,
            "provenance": provenance,
            "timing": {"total_seconds": time.perf_counter() - started},
            "status": "ok",
        }
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "reference_kind": REFERENCE_KIND,
            "instance_id": instance_id,
            "blocks": [],
            "engine": ENGINE,
            "provenance": provenance,
            "timing": {"total_seconds": time.perf_counter() - started},
            "status": "failed",
            "error_kind": _error_kind(exc),
            "error": _failure_diagnostics(exc),
        }


def _failure_diagnostics(exc: BaseException) -> str:
    details = [str(exc)]
    for stream_name in ("stderr", "stdout"):
        stream = getattr(exc, stream_name, None)
        if isinstance(stream, bytes):
            stream = stream.decode(errors="replace")
        if isinstance(stream, str) and stream.strip():
            details.append(f"{stream_name}: {stream.strip()[:4000]}")
    return "\n".join(details)


def _codex_argv(
    *,
    codex_executable: str,
    job_dir: Path,
    images: tuple[Path, Path],
    schema_path: Path,
    response_path: Path,
    model: str,
    model_config: tuple[str, ...],
) -> list[str]:
    argv = [
        codex_executable,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--model",
        model,
    ]
    for value in model_config:
        argv.extend(("--config", value))
    for feature in DISABLED_CODEX_FEATURES:
        argv.extend(("--disable", feature))
    argv.extend(
        (
            "--cd",
            str(job_dir),
            "--image",
            str(images[0]),
            "--image",
            str(images[1]),
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "-",
        )
    )
    return argv


def _validated_blocks(raw_response: str) -> list[dict[str, Any]]:
    try:
        response = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex response is not valid JSON") from exc
    if not isinstance(response, dict) or set(response) != {"blocks"}:
        raise ValueError("Codex response must contain only blocks")
    raw_blocks = response["blocks"]
    if not isinstance(raw_blocks, list):
        raise ValueError("Codex response blocks must be an array")
    blocks: list[dict[str, Any]] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, dict) or set(raw_block) != {"block_id", "text"}:
            raise ValueError("each Codex block must contain only block_id and text")
        block_id = raw_block["block_id"]
        text = raw_block["text"]
        if not isinstance(block_id, str) or not isinstance(text, str) or not text:
            raise ValueError("Codex block_id and text must be non-empty strings")
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("Codex block text must contain visible characters")
        blocks.append({"block_id": block_id, "text": normalized})
    block_ids = tuple(block["block_id"] for block in blocks)
    if block_ids != EXPECTED_BLOCK_IDS:
        raise ValueError("Codex response must contain ordered unique blocks b01 through b23")
    return blocks


def _checkpoint_results(
    checkpoint: Path,
    current: dict[str, dict[str, Any]],
    records: Any,
) -> None:
    for record in records:
        current[record["instance_id"]] = record
        _atomic_write_jsonl(checkpoint, (current[key] for key in sorted(current)))


def _checkpoint_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.checkpoint")


def _atomic_write_jsonl(path: Path, records: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        write_jsonl(temporary_path, records)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_raw_response(
    raw_dir: Path, instance_id: str, raw_response: str, completed: subprocess.CompletedProcess[str]
) -> None:
    if TASK_ID_PATTERN.fullmatch(instance_id) is None:
        raise ValueError(f"invalid DocSem instance_id: {instance_id}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(raw_dir / f"{instance_id}.json", raw_response)
    _atomic_write_text(raw_dir / f"{instance_id}.stderr.txt", completed.stderr)


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(value, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        instance_id = _required_string(record, "instance_id")
        if instance_id in records:
            raise ValueError(f"duplicate instance_id in Codex reference output: {instance_id}")
        records[instance_id] = record
    return records


def _codex_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True, timeout=10
    )
    return completed.stdout.strip()


def _executable_identity(executable: str) -> dict[str, str]:
    resolved = shutil.which(executable)
    path = Path(resolved if resolved is not None else executable).resolve()
    return {
        "name": path.name,
        "kind": "sha256",
        "sha256": _sha256_file(path),
    }


def _required_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _error_kind(error: BaseException) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, subprocess.CalledProcessError):
        return "subprocess_error"
    if isinstance(error, FileNotFoundError):
        return "file_not_found"
    if isinstance(error, ValueError):
        return "validation_error"
    if isinstance(error, OSError):
        return "os_error"
    return "runtime_error"
