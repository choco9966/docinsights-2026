#!/usr/bin/env python3
"""Build two-reader, image-only consensus-silver OCR benchmark references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = REPOSITORY_ROOT / "artifacts" / "solution-records" / "issue24" / "ocr-benchmark-v2"
TRANSCRIPTION_PROMPT = (
    "Act only as an image transcription reader for a consensus-silver OCR benchmark. Four images "
    "are supplied in order: one full page followed by three non-overlapping, full-width core "
    "regions from top to bottom. Transcribe ordinary document text and headings; exclude diagonal "
    "background watermark text. For each core, return only lines wholly visible inside that core. "
    "Exclude every line cut by a core boundary and count those exclusions. Mark uncertain true "
    "whenever any included text, heading identifier, or numeric literal is unreadable or disputed; "
    "represent unreadable text explicitly rather than guessing. A visible ID is a literal block "
    "label without its delimiter colon; preserve its case and internal punctuation. Numeric "
    "literals come only from body text, exclude IDs, and preserve signs, decimals, comma grouping, "
    "fractions, and percent signs exactly as written. Do not solve arithmetic. Do not use OCR, "
    "document names, "
    "questions, answers, labels, engine outputs, or another reader's response. Return only the "
    "required JSON object with regions in core-1, core-2, core-3 order."
)
OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["regions"],
    "properties": {
        "regions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "region_id",
                    "lines",
                    "visible_ids",
                    "numeric_literals",
                    "uncertain",
                    "boundary_exclusions",
                ],
                "properties": {
                    "region_id": {"type": "string", "enum": ["core-1", "core-2", "core-3"]},
                    "lines": {"type": "array", "items": {"type": "string"}},
                    "visible_ids": {"type": "array", "items": {"type": "string"}},
                    "numeric_literals": {"type": "array", "items": {"type": "string"}},
                    "uncertain": {"type": "boolean"},
                    "boundary_exclusions": {"type": "integer", "minimum": 0},
                },
            },
        }
    },
}
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "in_app_browser",
    "code_mode_host",
    "hooks",
    "skill_mcp_dependency_install",
    "multi_agent",
    "image_generation",
    "memories",
    "goals",
)
CHECKER_CONFIG = {
    "model": "gpt-5.6-sol",
    "model_reasoning_effort": "high",
    "sandbox": "read-only",
    "web_search": "disabled",
    "disabled_features": list(DISABLED_FEATURES),
    "timeout_seconds": 300,
    "runtime_attempts": 2,
    "readers_per_page": 2,
    "max_concurrent_calls": 4,
    "method": "codex-cli-image-only-consensus-silver-v1",
}
REFERENCE_PROTOCOL = {
    "version": "consensus-silver-derived-literals-v3",
    "normalization": "NFC plus whitespace collapse only",
    "numeric_literals": "derived from normalized reader lines with scorer NUMBER tokenizer",
    "visible_ids": "derived from normalized reader lines with scorer HEADING tokenizer",
    "empty_body": "eligible when both clear normalized bodies are exactly empty",
}
_SAFE_PAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Kept byte-for-byte equivalent to scripts/score_ocr_benchmark.py. Reference and
# candidate literals must be tokenized by the same frozen rule.
_NUMBER = re.compile(
    r"(?<![\w.])[+−-]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][+−-]?\d+)?"
    r"(?:/[+−-]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][+−-]?\d+)?)?%?(?!\w)"
)
_HEADING = re.compile(r"^\s*(\S+?):(?=\s|$)")


class BenchmarkHarnessError(ValueError):
    """The benchmark reference run violated a frozen input or artifact contract."""


BenchmarkInvoker = Callable[..., Mapping[str, Any]]


def run_benchmark(
    *,
    manifest_path: Path | str,
    output_root: Path | str,
    limit: int | None,
    resume: bool,
    invoke: BenchmarkInvoker | None = None,
) -> dict[str, Any]:
    """Run two independent image-only readers and write one consensus file per page."""
    source_manifest_path = _regular_file(manifest_path, "corpus manifest")
    manifest_sha256 = _sha256_file(source_manifest_path)
    manifest = _load_json(source_manifest_path, "corpus manifest")
    pages = _validate_manifest(manifest)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise BenchmarkHarnessError("limit must be a positive integer")
    selected = pages[:limit] if limit is not None else pages

    destination = _ensure_private_directory(output_root)
    runtime_config = _runtime_config() if invoke is None else dict(CHECKER_CONFIG)
    run_manifest = {
        "schema_version": "consensus-silver-run-v3",
        "corpus_manifest_path": str(source_manifest_path),
        "corpus_manifest_sha256": manifest_sha256,
        "checker_config": runtime_config,
        "checker_config_sha256": _canonical_sha256(runtime_config),
        "prompt_sha256": _sha256_bytes(TRANSCRIPTION_PROMPT.encode("utf-8")),
        "output_schema_sha256": _canonical_sha256(OUTPUT_SCHEMA),
        "reference_protocol": REFERENCE_PROTOCOL,
        "reference_protocol_sha256": _canonical_sha256(REFERENCE_PROTOCOL),
        "host_normalization_code_sha256": _sha256_file(Path(__file__).resolve()),
        "page_ids": [page["benchmark_page_id"] for page in pages],
    }
    _prepare_run_manifest(destination / "run-manifest.json", run_manifest, resume)

    pending: list[tuple[dict[str, Any], Path, tuple[Path, ...]]] = []
    resumed_pages = 0
    resumed_completed_pages = 0
    resumed_runtime_failed_pages = 0
    for page in selected:
        page_dir = destination / "references" / page["benchmark_page_id"]
        if resume and (page_dir / "state.json").is_file():
            status = _verify_completed_page(page_dir, manifest_sha256, page)
            resumed_pages += 1
            if status == "complete":
                resumed_completed_pages += 1
            else:
                resumed_runtime_failed_pages += 1
            continue
        if page_dir.exists():
            raise BenchmarkHarnessError(
                f"incomplete or unverified page artifacts exist: {page['benchmark_page_id']}"
            )
        images = _prepare_page_images(page, page_dir)
        pending.append((page, page_dir, images))

    reader_results: dict[str, dict[int, dict[str, Any]]] = {
        page["benchmark_page_id"]: {} for page, _, _ in pending
    }
    jobs = {}
    completed_pages = 0
    runtime_failed_pages = 0
    with ThreadPoolExecutor(max_workers=CHECKER_CONFIG["max_concurrent_calls"]) as executor:
        for page, page_dir, images in pending:
            for reader_id in (1, 2):
                future = executor.submit(
                    _run_reader,
                    page_dir=page_dir,
                    reader_id=reader_id,
                    images=images,
                    checker_config=runtime_config,
                    invoke=invoke or _invoke_codex,
                )
                jobs[future] = (page, page_dir, reader_id)
        for future in as_completed(jobs):
            page, page_dir, reader_id = jobs[future]
            page_id = page["benchmark_page_id"]
            reader_results[page_id][reader_id] = future.result()
            if len(reader_results[page_id]) == 2:
                results = [reader_results[page_id][item] for item in (1, 2)]
                status = _write_page_consensus(
                    page=page,
                    page_dir=page_dir,
                    manifest_sha256=manifest_sha256,
                    readers=results,
                )
                if status == "complete":
                    completed_pages += 1
                else:
                    runtime_failed_pages += 1

    summary = {
        "schema_version": "consensus-silver-summary-v3",
        "manifest_sha256": manifest_sha256,
        "selected_pages": len(selected),
        "completed_pages": completed_pages + resumed_completed_pages,
        "runtime_failed_pages": runtime_failed_pages + resumed_runtime_failed_pages,
        "resumed_pages": resumed_pages,
    }
    _write_json(destination / "summary.json", summary)
    return summary


def _write_page_consensus(
    *,
    page: dict[str, Any],
    page_dir: Path,
    manifest_sha256: str,
    readers: list[dict[str, Any]],
) -> str:
    consensus = _consensus_page(page, manifest_sha256, readers)
    consensus_path = page_dir / "consensus.json"
    _write_json(consensus_path, consensus)
    artifacts = _directory_artifacts(page_dir, excluded={page_dir / "state.json"})
    state = {
        "benchmark_page_id": page["benchmark_page_id"],
        "status": consensus["status"],
        "manifest_sha256": manifest_sha256,
        "page_source_sha256": _canonical_sha256(page),
        "consensus_sha256": _sha256_file(consensus_path),
        "artifacts": artifacts,
    }
    _write_json(page_dir / "state.json", state)
    return consensus["status"]


def _validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        raise BenchmarkHarnessError("corpus manifest must be an object")
    if manifest.get("schema_version") != "issue24-ocr-benchmark-v1":
        raise BenchmarkHarnessError("unsupported corpus manifest schema")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or manifest.get("page_count") != len(pages) or not pages:
        raise BenchmarkHarnessError("corpus manifest page count is invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(pages):
        if not isinstance(raw, Mapping):
            raise BenchmarkHarnessError(f"manifest page {index} must be an object")
        page = dict(raw)
        page_id = page.get("benchmark_page_id")
        if not isinstance(page_id, str) or not _SAFE_PAGE_ID.fullmatch(page_id) or page_id in seen:
            raise BenchmarkHarnessError("manifest benchmark_page_id is unsafe or duplicated")
        seen.add(page_id)
        for field in ("page_number", "width", "height"):
            value = page.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BenchmarkHarnessError(f"manifest {page_id} has invalid {field}")
        for path_field, hash_field in (
            ("source_pdf", "source_pdf_sha256"),
            ("rendered_image", "rendered_image_sha256"),
        ):
            path = _regular_file(page.get(path_field), f"manifest {path_field}")
            expected = _required_sha256(page.get(hash_field), f"manifest {hash_field}")
            if _sha256_file(path) != expected:
                raise BenchmarkHarnessError(f"manifest {path_field} hash mismatch")
            page[path_field] = str(path)
        cores = page.get("scoring_cores")
        if not isinstance(cores, list) or len(cores) != 3:
            raise BenchmarkHarnessError("each benchmark page must have three scoring cores")
        expected_tops = [0, page["height"] // 3, (2 * page["height"]) // 3]
        expected_bottoms = [page["height"] // 3, (2 * page["height"]) // 3, page["height"]]
        normalized_cores = []
        for core_index, raw_core in enumerate(cores, start=1):
            if not isinstance(raw_core, Mapping) or raw_core.get("core_index") != core_index:
                raise BenchmarkHarnessError("scoring cores must be ordered 1 through 3")
            core = dict(raw_core)
            expected_box = {
                "left": 0,
                "top": expected_tops[core_index - 1],
                "right": page["width"],
                "bottom": expected_bottoms[core_index - 1],
            }
            if core.get("crop_box") != expected_box:
                raise BenchmarkHarnessError("scoring core bounds are not exact full-width thirds")
            core_path = _regular_file(core.get("image_path"), "manifest scoring core")
            core_sha256 = _required_sha256(core.get("image_sha256"), "core image_sha256")
            if _sha256_file(core_path) != core_sha256:
                raise BenchmarkHarnessError("manifest scoring core hash mismatch")
            core["image_path"] = str(core_path)
            normalized_cores.append(core)
        page["scoring_cores"] = normalized_cores
        normalized.append(page)
    return normalized


def _prepare_page_images(page: dict[str, Any], page_dir: Path) -> tuple[Path, ...]:
    _ensure_private_directory(page_dir / "images")
    page_path = Path(page["rendered_image"])
    try:
        with Image.open(page_path) as source:
            source.load()
            if source.size != (page["width"], page["height"]):
                raise BenchmarkHarnessError("rendered page dimensions differ from manifest")
            core_paths = []
            for core in page["scoring_cores"]:
                box = core["crop_box"]
                expected = source.crop((box["left"], box["top"], box["right"], box["bottom"]))
                canonical_path = Path(core["image_path"])
                with Image.open(canonical_path) as canonical:
                    canonical.load()
                    if (
                        canonical.size != expected.size
                        or ImageChops.difference(
                            expected.convert("RGB"), canonical.convert("RGB")
                        ).getbbox()
                        is not None
                    ):
                        raise BenchmarkHarnessError("manifest scoring core pixels differ from page")
                output_path = page_dir / "images" / f"core-{core['core_index']}.png"
                _write_private(output_path, canonical_path.read_bytes())
                if _sha256_file(output_path) != core["image_sha256"]:
                    raise BenchmarkHarnessError("copied scoring core hash mismatch")
                core_paths.append(output_path.resolve())
    except BenchmarkHarnessError:
        raise
    except (OSError, ValueError) as error:
        raise BenchmarkHarnessError("could not inspect benchmark page images") from error
    image_binding = {
        "source_pdf_sha256": page["source_pdf_sha256"],
        "page_image": {"path": str(page_path), "sha256": page["rendered_image_sha256"]},
        "cores": [
            {
                "region_id": f"core-{core['core_index']}",
                "bounds": core["crop_box"],
                "path": str(core_paths[index]),
                "sha256": core["image_sha256"],
            }
            for index, core in enumerate(page["scoring_cores"])
        ],
    }
    _write_json(page_dir / "images.json", image_binding)
    return (page_path.resolve(), *core_paths)


def _run_reader(
    *,
    page_dir: Path,
    reader_id: int,
    images: tuple[Path, ...],
    checker_config: dict[str, Any],
    invoke: BenchmarkInvoker,
) -> dict[str, Any]:
    reader_dir = _ensure_private_directory(page_dir / "readers" / f"reader-{reader_id}")
    attempts = []
    for attempt_number in range(1, CHECKER_CONFIG["runtime_attempts"] + 1):
        attempt_dir = _ensure_private_directory(reader_dir / f"attempt-{attempt_number}")
        _write_private(attempt_dir / "prompt.txt", TRANSCRIPTION_PROMPT)
        _write_json(attempt_dir / "schema.json", OUTPUT_SCHEMA)
        _write_json(
            attempt_dir / "input-images.json",
            [{"path": str(path), "sha256": _sha256_file(path)} for path in images],
        )
        started = time.monotonic()
        try:
            raw_result = invoke(
                prompt=TRANSCRIPTION_PROMPT,
                images=images,
                output_schema=OUTPUT_SCHEMA,
                output_dir=attempt_dir,
                checker_config=checker_config,
            )
            result = _invocation_result(raw_result, attempt_dir)
            normalized = _normalize_response(result["response"])
            elapsed = result["elapsed_seconds"]
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "succeeded",
                    "elapsed_seconds": elapsed,
                    "error": None,
                    "response_sha256": result["raw_response_sha256"],
                    "artifacts": result["artifacts"],
                }
            )
            reader = {
                "run_id": reader_id,
                "status": "succeeded",
                "normalized": normalized,
                "response_sha256": result["raw_response_sha256"],
                "attempts": attempts,
            }
            _write_json(reader_dir / "reader.json", reader)
            return reader
        except Exception as error:  # noqa: BLE001 - preserve isolated runtime failure
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "runtime_failed",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": f"{type(error).__name__}: {error}",
                    "response_sha256": None,
                    "artifacts": _directory_artifacts(attempt_dir),
                }
            )
    reader = {
        "run_id": reader_id,
        "status": "runtime_failed",
        "normalized": None,
        "response_sha256": None,
        "attempts": attempts,
    }
    _write_json(reader_dir / "reader.json", reader)
    return reader


def _invocation_result(value: Any, attempt_dir: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkHarnessError("reader invocation result must be an object")
    required = {
        "status",
        "response",
        "raw_response_path",
        "artifact_paths",
        "elapsed_seconds",
        "error",
    }
    if set(value) != required or value["status"] != "succeeded":
        raise BenchmarkHarnessError(str(value.get("error", "reader runtime failure")))
    raw_path = _contained_file(value["raw_response_path"], attempt_dir)
    artifact_paths = value["artifact_paths"]
    if not isinstance(artifact_paths, Sequence) or isinstance(artifact_paths, str | bytes):
        raise BenchmarkHarnessError("reader artifact_paths must be a sequence")
    artifacts = []
    seen = set()
    for raw_artifact in artifact_paths:
        path = _contained_file(raw_artifact, attempt_dir)
        if path in seen:
            raise BenchmarkHarnessError("reader artifact paths must be unique")
        path.chmod(0o600)
        seen.add(path)
        artifacts.append(_artifact(path))
    if raw_path not in seen:
        raise BenchmarkHarnessError("raw response is absent from invocation artifacts")
    raw_bytes = raw_path.read_bytes()
    try:
        parsed = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkHarnessError("raw reader response is invalid JSON") from error
    if parsed != value["response"]:
        raise BenchmarkHarnessError("raw reader response differs from parsed response")
    elapsed = value["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, int | float) or elapsed < 0:
        raise BenchmarkHarnessError("reader elapsed_seconds is invalid")
    return {
        "response": parsed,
        "raw_response_sha256": _sha256_bytes(raw_bytes),
        "elapsed_seconds": elapsed,
        "artifacts": artifacts,
    }


def _normalize_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"regions"}:
        raise BenchmarkHarnessError("reader response has invalid fields")
    regions = value["regions"]
    if not isinstance(regions, list) or len(regions) != 3:
        raise BenchmarkHarnessError("reader response must contain exactly three regions")
    normalized = []
    for index, raw in enumerate(regions, start=1):
        fields = {
            "region_id",
            "lines",
            "visible_ids",
            "numeric_literals",
            "uncertain",
            "boundary_exclusions",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise BenchmarkHarnessError("reader region has invalid fields")
        if raw["region_id"] != f"core-{index}":
            raise BenchmarkHarnessError("reader regions are not in canonical core order")
        lines = _text_list(raw["lines"], "lines", allow_empty=True)
        observed_visible_ids = _text_list(raw["visible_ids"], "visible_ids", allow_empty=False)
        observed_numeric_literals = _text_list(
            raw["numeric_literals"], "numeric_literals", allow_empty=False
        )
        derived_visible_ids, derived_numeric_literals = _extract_literals(lines)
        uncertain = raw["uncertain"]
        exclusions = raw["boundary_exclusions"]
        if not isinstance(uncertain, bool):
            raise BenchmarkHarnessError("region uncertain must be boolean")
        if isinstance(exclusions, bool) or not isinstance(exclusions, int) or exclusions < 0:
            raise BenchmarkHarnessError("boundary_exclusions must be a nonnegative integer")
        normalized.append(
            {
                "region_id": raw["region_id"],
                "lines": lines,
                "body_text": "\n".join(lines),
                "visible_ids": derived_visible_ids,
                "visible_ids_observed": observed_visible_ids,
                "numeric_literals": derived_numeric_literals,
                "numeric_literals_observed": observed_numeric_literals,
                "uncertain": uncertain,
                "boundary_exclusions": exclusions,
            }
        )
    return {"regions": normalized}


def _text_list(value: Any, field: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise BenchmarkHarnessError(f"{field} must be a list of strings")
    normalized = [_normalize_text(item) for item in value]
    if not allow_empty and any(not item for item in normalized):
        raise BenchmarkHarnessError(f"{field} entries must be non-empty")
    return normalized


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _extract_literals(lines: list[str]) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    numbers: list[str] = []
    for line in lines:
        body = line
        heading = _HEADING.match(body)
        if heading:
            identifiers.append(heading.group(1))
            body = body[heading.end() :]
        numbers.extend(_NUMBER.findall(body))
    return identifiers, numbers


def _consensus_page(
    page: dict[str, Any], manifest_sha256: str, readers: list[dict[str, Any]]
) -> dict[str, Any]:
    regions = []
    all_succeeded = all(reader["status"] == "succeeded" for reader in readers)
    for index, core in enumerate(page["scoring_cores"], start=1):
        reader_views = []
        for reader in readers:
            normalized = (
                reader["normalized"]["regions"][index - 1]
                if reader["normalized"] is not None
                else None
            )
            reader_views.append(
                {
                    "run_id": reader["run_id"],
                    "status": reader["status"],
                    "normalized": normalized,
                    "response_sha256": reader["response_sha256"],
                    "attempts": reader["attempts"],
                }
            )
        succeeded = [view["normalized"] for view in reader_views if view["normalized"] is not None]
        both_clear = len(succeeded) == 2 and not any(item["uncertain"] for item in succeeded)
        bodies = [item["body_text"] for item in succeeded]
        ids = [item["visible_ids"] for item in succeeded]
        numbers = [item["numeric_literals"] for item in succeeded]
        body_eligible = both_clear and bodies[0] == bodies[1]
        ids_eligible = both_clear and Counter(ids[0]) == Counter(ids[1])
        numbers_eligible = both_clear and Counter(numbers[0]) == Counter(numbers[1])
        regions.append(
            {
                "region_id": f"core-{index}",
                "bounds": core["crop_box"],
                "readers": reader_views,
                "consensus": {
                    "body": {
                        "eligible": body_eligible,
                        "value": bodies[0] if body_eligible else None,
                    },
                    "visible_ids": {
                        "eligible": ids_eligible,
                        "value": ids[0] if ids_eligible else None,
                    },
                    "numeric_literals": {
                        "eligible": numbers_eligible,
                        "value": numbers[0] if numbers_eligible else None,
                    },
                },
                "disagreements": {
                    "body": bodies,
                    "visible_ids": ids,
                    "numeric_literals": numbers,
                    "uncertain": [item["uncertain"] for item in succeeded],
                    "boundary_exclusions": [item["boundary_exclusions"] for item in succeeded],
                },
            }
        )
    return {
        "schema_version": "consensus-silver-v3",
        "label": "consensus-silver",
        "benchmark_page_id": page["benchmark_page_id"],
        "manifest_sha256": manifest_sha256,
        "source": {
            "pdf_sha256": page["source_pdf_sha256"],
            "page_number": page["page_number"],
            "page_image_sha256": page["rendered_image_sha256"],
            "width": page["width"],
            "height": page["height"],
        },
        "regions": regions,
        "status": "complete" if all_succeeded else "runtime_failed",
    }


def _runtime_config() -> dict[str, Any]:
    executable_value = shutil.which("codex")
    if executable_value is None:
        raise BenchmarkHarnessError("Codex CLI is unavailable")
    executable = Path(executable_value).resolve()
    version = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, timeout=30, check=False
    )
    observed_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or not observed_version:
        raise BenchmarkHarnessError("could not identify Codex CLI version")
    return {
        **CHECKER_CONFIG,
        "codex_executable_path": str(executable),
        "codex_executable_sha256": _sha256_file(executable),
        "codex_observed_version": observed_version,
    }


def _invoke_codex(
    *,
    prompt: str,
    images: Sequence[Path],
    output_schema: dict[str, Any],
    output_dir: Path,
    checker_config: dict[str, Any],
) -> dict[str, Any]:
    executable = _regular_file(checker_config["codex_executable_path"], "Codex executable")
    if _sha256_file(executable) != checker_config["codex_executable_sha256"]:
        raise BenchmarkHarnessError("Codex executable hash changed")
    schema_path = output_dir / "schema.json"
    response_path = output_dir / "response.json"
    argv = [
        str(executable),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--strict-config",
        "--json",
        "--model",
        "gpt-5.6-sol",
        "--cd",
        str(output_dir),
        "--config",
        'web_search="disabled"',
        "--config",
        'model_reasoning_effort="high"',
    ]
    for feature in DISABLED_FEATURES:
        argv.extend(("--config", f"features.{feature}=false"))
    for image in images:
        argv.extend(("--image", str(image)))
    argv.extend(
        (
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(response_path),
            "-",
        )
    )
    _write_json(output_dir / "command.json", {"argv": argv, "config": checker_config})
    started = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 - frozen executable and argv
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(
            input=prompt, timeout=checker_config["timeout_seconds"]
        )
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        elapsed = time.monotonic() - started
        _write_private(output_dir / "events.jsonl", stdout or "")
        _write_private(output_dir / "stderr.txt", stderr or "")
        return {
            "status": "runtime_failed",
            "response": None,
            "raw_response_path": None,
            "artifact_paths": [output_dir / "events.jsonl", output_dir / "stderr.txt"],
            "elapsed_seconds": elapsed,
            "error": "timeout",
        }
    elapsed = time.monotonic() - started
    events_path, stderr_path = output_dir / "events.jsonl", output_dir / "stderr.txt"
    _write_private(events_path, stdout or "")
    _write_private(stderr_path, stderr or "")
    if process.returncode != 0 or not _turn_completed(stdout or ""):
        return {
            "status": "runtime_failed",
            "response": None,
            "raw_response_path": None,
            "artifact_paths": [events_path, stderr_path],
            "elapsed_seconds": elapsed,
            "error": f"exit {process.returncode} or incomplete event stream",
        }
    if not response_path.is_file():
        raise BenchmarkHarnessError("Codex reader response file is missing")
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkHarnessError("Codex reader response is invalid JSON") from error
    artifacts = [
        output_dir / "prompt.txt",
        schema_path,
        output_dir / "input-images.json",
        output_dir / "command.json",
        events_path,
        stderr_path,
        response_path,
    ]
    return {
        "status": "succeeded",
        "response": response,
        "raw_response_path": response_path,
        "artifact_paths": artifacts,
        "elapsed_seconds": elapsed,
        "error": None,
    }


def _turn_completed(events: str) -> bool:
    completed = False
    for line in events.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchmarkHarnessError("Codex emitted invalid JSONL") from error
        if not isinstance(event, Mapping):
            raise BenchmarkHarnessError("Codex emitted a non-object event")
        if event.get("type") == "turn.completed":
            completed = True
    return completed


def _prepare_run_manifest(path: Path, expected: dict[str, Any], resume: bool) -> None:
    if path.exists():
        if not resume:
            raise BenchmarkHarnessError("benchmark output exists; use --resume")
        if _load_json(path, "run manifest") != expected:
            raise BenchmarkHarnessError("resume refused: corpus manifest or config changed")
        return
    if resume:
        raise BenchmarkHarnessError("resume refused: run manifest is missing")
    _write_json(path, expected)


def _verify_completed_page(page_dir: Path, manifest_sha256: str, page: dict[str, Any]) -> str:
    state = _load_json(page_dir / "state.json", "page state")
    if (
        state.get("benchmark_page_id") != page["benchmark_page_id"]
        or state.get("manifest_sha256") != manifest_sha256
        or state.get("page_source_sha256") != _canonical_sha256(page)
    ):
        raise BenchmarkHarnessError("resume page binding changed")
    consensus_path = page_dir / "consensus.json"
    if not consensus_path.is_file() or state.get("consensus_sha256") != _sha256_file(
        consensus_path
    ):
        raise BenchmarkHarnessError("resume consensus hash mismatch or tamper detected")
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, list):
        raise BenchmarkHarnessError("resume artifact inventory is invalid")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise BenchmarkHarnessError("resume artifact inventory is malformed")
        path = _regular_file(artifact.get("path"), "resume artifact")
        if _sha256_file(path) != artifact.get("sha256"):
            raise BenchmarkHarnessError("resume artifact hash mismatch or tamper detected")
    status = state.get("status")
    if status not in {"complete", "runtime_failed"}:
        raise BenchmarkHarnessError("resume page status is invalid")
    return status


def _directory_artifacts(path: Path, excluded: set[Path] | None = None) -> list[dict[str, str]]:
    excluded_resolved = {item.resolve() for item in (excluded or set())}
    return [
        _artifact(item.resolve())
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.is_symlink() and item.resolve() not in excluded_resolved
    ]


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _contained_file(value: Any, root: Path) -> Path:
    path = _regular_file(value, "reader artifact")
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise BenchmarkHarnessError("reader artifact escapes its attempt directory") from error
    return path


def _regular_file(value: Any, field: str) -> Path:
    if not isinstance(value, str | Path):
        raise BenchmarkHarnessError(f"{field} path is invalid")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise BenchmarkHarnessError(f"{field} must be a regular non-symlink file")
    return path.resolve()


def _ensure_private_directory(path: Path | str) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    return destination.resolve()


def _write_private(path: Path, content: str | bytes) -> None:
    _ensure_private_directory(path.parent)
    payload = content.encode("utf-8") if isinstance(content, str) else content
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write_private(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _load_json(path: Path, field: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkHarnessError(f"could not read {field}") from error


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _required_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BenchmarkHarnessError(f"{field} must be a lowercase SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output_root.resolve()
    try:
        output.relative_to(PRIVATE_ROOT.resolve())
    except ValueError as error:
        raise BenchmarkHarnessError(
            f"--output-root must be inside {PRIVATE_ROOT.resolve()}"
        ) from error
    summary = run_benchmark(
        manifest_path=args.manifest,
        output_root=output,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
