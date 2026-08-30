"""Fail-closed verification for Codex-assisted OCR reference artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .codex_reference import (
    DISABLED_CODEX_FEATURES,
    ENGINE,
    EXPECTED_BLOCK_IDS,
    PROMPT,
    REFERENCE_KIND,
    SCHEMA_VERSION,
)
from .records import deterministic_content_hash, read_jsonl
from .render import render_pdf

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TASK_ID = re.compile(r"task_[0-9]{6}")
_PROVENANCE_HASH_FIELDS = (
    "input_pdf_sha256",
    "prompt_sha256",
    "output_schema_sha256",
    "raw_response_sha256",
    "run_fingerprint",
)
_FORBIDDEN_RESULT_FIELDS = frozenset({"answer", "candidate_answers", "evidence", "labels"})
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "reference_kind",
        "instance_id",
        "blocks",
        "engine",
        "provenance",
        "timing",
        "status",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "reference_kind",
        "input_pdf_sha256",
        "split",
        "split_seed",
        "model",
        "codex_cli_version",
        "codex_executable_identity",
        "model_config",
        "prompt_sha256",
        "output_schema_sha256",
        "dpi",
        "renderer",
        "renderer_executable_identity",
        "disabled_codex_features",
        "run_fingerprint",
        "input_image_sha256",
        "raw_response_sha256",
    }
)


def verify_codex_reference(
    tasks_path: str | Path,
    output_path: str | Path,
    raw_dir: str | Path,
    *,
    documents_root: str | Path | None = None,
    schema_path: str | Path | None = None,
    poppler_executable: str = "pdftoppm",
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Verify a complete Codex reference run and return a label-free summary."""
    tasks_source = Path(tasks_path)
    output_source = Path(output_path)
    raw_source = Path(raw_dir)
    root = Path(documents_root).resolve() if documents_root else tasks_source.parent.resolve()
    schema = (
        Path(schema_path)
        if schema_path is not None
        else Path(__file__).parents[2] / "schemas" / "codex-transcription-response-v1.schema.json"
    )
    prompt_sha256 = hashlib.sha256(PROMPT.encode()).hexdigest()
    schema_sha256 = _sha256_file(schema)

    tasks = _records_by_id(tasks_source, kind="tasks manifest")
    records = _records_by_id(output_source, kind="Codex reference output")
    expected_ids = set(tasks)
    actual_ids = set(records)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(f"Codex reference coverage mismatch: missing={missing}, extra={extra}")

    raw_ids = _raw_artifact_ids(raw_source, ".json")
    stderr_ids = _raw_artifact_ids(raw_source, ".stderr.txt")
    _require_exact_ids("raw responses", expected_ids, raw_ids)
    _require_exact_ids("stderr sidecars", expected_ids, stderr_ids)
    _require_exact_raw_files(raw_source, expected_ids)

    for instance_id in sorted(tasks):
        _verify_record(
            instance_id,
            tasks[instance_id],
            records[instance_id],
            root=root,
            raw_dir=raw_source,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
            poppler_executable=poppler_executable,
            timeout_seconds=timeout_seconds,
        )

    return {
        "schema_version": "1.0",
        "valid": True,
        "tasks_path": str(tasks_source.resolve()),
        "output_path": str(output_source.resolve()),
        "raw_dir": str(raw_source.resolve()),
        "expected_count": len(tasks),
        "record_count": len(records),
        "ok_count": len(records),
        "raw_response_count": len(raw_ids),
        "stderr_sidecar_count": len(stderr_ids),
        "missing_count": 0,
        "duplicate_count": 0,
        "extra_count": 0,
        "failed_count": 0,
        "tasks_sha256": _sha256_file(tasks_source),
        "output_sha256": _sha256_file(output_source),
        "raw_response_aggregate_sha256": _artifact_aggregate_sha256(
            raw_source, raw_ids, ".json"
        ),
        "stderr_sidecar_aggregate_sha256": _artifact_aggregate_sha256(
            raw_source, stderr_ids, ".stderr.txt"
        ),
    }


def _records_by_id(path: Path, *, kind: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        instance_id = _required_string(record, "instance_id", kind)
        if _TASK_ID.fullmatch(instance_id) is None:
            raise ValueError(f"invalid instance_id in {kind}: {instance_id}")
        if instance_id in records:
            raise ValueError(f"duplicate instance_id in {kind}: {instance_id}")
        records[instance_id] = record
    return records


def _verify_record(
    instance_id: str,
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    root: Path,
    raw_dir: Path,
    prompt_sha256: str,
    schema_sha256: str,
    poppler_executable: str,
    timeout_seconds: float,
) -> None:
    forbidden = sorted(_FORBIDDEN_RESULT_FIELDS.intersection(record))
    if forbidden:
        raise ValueError(f"forbidden fields in Codex record for {instance_id}: {forbidden}")
    if set(record) != _RECORD_FIELDS:
        raise ValueError(f"invalid Codex record shape for {instance_id}")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"invalid schema_version for {instance_id}")
    if record.get("reference_kind") != REFERENCE_KIND:
        raise ValueError(f"invalid reference_kind for {instance_id}")
    if record.get("engine") != ENGINE:
        raise ValueError(f"invalid engine for {instance_id}")
    if record.get("status") != "ok":
        raise ValueError(f"Codex record status is not ok for {instance_id}")
    timing = record.get("timing")
    if (
        not isinstance(timing, Mapping)
        or set(timing) != {"total_seconds"}
        or not isinstance(timing["total_seconds"], (int, float))
        or isinstance(timing["total_seconds"], bool)
        or timing["total_seconds"] < 0
    ):
        raise ValueError(f"invalid timing for {instance_id}")

    blocks = _validated_blocks(
        record.get("blocks"), instance_id, source="record", canonicalize_text=False
    )
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"invalid provenance for {instance_id}")
    if set(provenance) != _PROVENANCE_FIELDS:
        missing_fields = sorted(_PROVENANCE_FIELDS - provenance.keys())
        extra_fields = sorted(provenance.keys() - _PROVENANCE_FIELDS)
        raise ValueError(
            f"invalid provenance fields for {instance_id}: "
            f"missing={missing_fields}, extra={extra_fields}"
        )
    if provenance.get("reference_kind") != REFERENCE_KIND:
        raise ValueError(f"invalid provenance reference_kind for {instance_id}")
    for task_field in ("split", "split_seed"):
        if provenance.get(task_field) != task.get(task_field):
            raise ValueError(f"provenance {task_field} mismatch for {instance_id}")
    for field in _PROVENANCE_HASH_FIELDS:
        _required_sha256(provenance, field, instance_id)

    pdf_value = Path(_required_string(task, "document_pdf", "tasks manifest"))
    pdf_path = (pdf_value if pdf_value.is_absolute() else root / pdf_value).resolve()
    try:
        pdf_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"document_pdf escapes documents_root for {instance_id}") from exc
    if provenance["input_pdf_sha256"] != _sha256_file(pdf_path):
        raise ValueError(f"input PDF SHA-256 mismatch for {instance_id}")

    model = provenance.get("model")
    codex_version = provenance.get("codex_cli_version")
    model_config = provenance.get("model_config")
    dpi = provenance.get("dpi")
    renderer = provenance.get("renderer")
    disabled_features = provenance.get("disabled_codex_features")
    if not isinstance(model, str) or not model:
        raise ValueError(f"invalid model for {instance_id}")
    if not isinstance(codex_version, str) or not codex_version:
        raise ValueError(f"invalid codex_cli_version for {instance_id}")
    if not isinstance(model_config, list) or not all(
        isinstance(value, str) for value in model_config
    ):
        raise ValueError(f"invalid model_config for {instance_id}")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError(f"invalid dpi for {instance_id}")
    if renderer != "poppler-pdftoppm":
        raise ValueError(f"invalid renderer for {instance_id}")
    if disabled_features != list(DISABLED_CODEX_FEATURES):
        raise ValueError(f"invalid disabled_codex_features for {instance_id}")
    if provenance["prompt_sha256"] != prompt_sha256:
        raise ValueError(f"prompt SHA-256 mismatch for {instance_id}")
    if provenance["output_schema_sha256"] != schema_sha256:
        raise ValueError(f"output schema SHA-256 mismatch for {instance_id}")

    image_hashes = provenance.get("input_image_sha256")
    if not isinstance(image_hashes, list) or len(image_hashes) != 2:
        raise ValueError(f"input_image_sha256 must contain exactly two pages for {instance_id}")
    for expected_page, image_hash in enumerate(image_hashes, 1):
        if not isinstance(image_hash, Mapping) or set(image_hash) != {"page_number", "sha256"}:
            raise ValueError(f"invalid input image hash entry for {instance_id}")
        if image_hash["page_number"] != expected_page:
            raise ValueError(f"input image hashes must be ordered pages 1 and 2 for {instance_id}")
        _required_sha256(image_hash, "sha256", instance_id)

    for identity_field in ("codex_executable_identity", "renderer_executable_identity"):
        identity = provenance.get(identity_field)
        if not isinstance(identity, Mapping) or set(identity) != {"name", "kind", "sha256"}:
            raise ValueError(f"invalid {identity_field} for {instance_id}")
        _required_sha256(identity, "sha256", instance_id)
        if not isinstance(identity.get("name"), str) or not identity["name"]:
            raise ValueError(f"invalid {identity_field} name for {instance_id}")
        if identity.get("kind") != "sha256":
            raise ValueError(f"invalid {identity_field} kind for {instance_id}")

    expected_fingerprint = deterministic_content_hash(
        {
            "input_pdf_sha256": provenance["input_pdf_sha256"],
            "model": model,
            "model_config": model_config,
            "codex_version": codex_version,
            "prompt_sha256": prompt_sha256,
            "output_schema_sha256": schema_sha256,
            "dpi": dpi,
            "codex_executable_identity": provenance["codex_executable_identity"],
            "renderer_executable_identity": provenance["renderer_executable_identity"],
            "disabled_codex_features": list(DISABLED_CODEX_FEATURES),
        }
    )
    if provenance["run_fingerprint"] != expected_fingerprint:
        raise ValueError(f"run fingerprint mismatch for {instance_id}")

    with tempfile.TemporaryDirectory(prefix="docinsights-codex-verify-") as render_dir:
        rendered = render_pdf(
            pdf_path,
            render_dir,
            dpi=dpi,
            executable=poppler_executable,
            timeout_seconds=timeout_seconds,
        )
        if len(rendered) != 2:
            raise ValueError(
                f"Codex reference requires exactly two rendered pages for {instance_id}"
            )
        for expected_page, image_path in enumerate(rendered, 1):
            if image_hashes[expected_page - 1]["sha256"] != _sha256_file(image_path):
                raise ValueError(
                    f"rendered page SHA-256 mismatch for {instance_id} page {expected_page}"
                )

    raw_path = raw_dir / f"{instance_id}.json"
    stderr_path = raw_dir / f"{instance_id}.stderr.txt"
    if not raw_path.is_file():
        raise ValueError(f"missing raw response for {instance_id}")
    if not stderr_path.is_file():
        raise ValueError(f"missing stderr sidecar for {instance_id}")
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != provenance["raw_response_sha256"]:
        raise ValueError(f"raw response SHA-256 mismatch for {instance_id}")
    try:
        raw_response = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"raw response is not valid UTF-8 JSON for {instance_id}") from exc
    if not isinstance(raw_response, Mapping) or set(raw_response) != {"blocks"}:
        raise ValueError(f"raw response must contain only blocks for {instance_id}")
    raw_blocks = _validated_blocks(
        raw_response["blocks"],
        instance_id,
        source="raw response",
        canonicalize_text=True,
    )
    if raw_blocks != blocks:
        raise ValueError(f"raw response blocks do not match record for {instance_id}")


def _validated_blocks(
    value: Any,
    instance_id: str,
    *,
    source: str,
    canonicalize_text: bool,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_BLOCK_IDS):
        raise ValueError(f"{source} must contain exactly 23 blocks for {instance_id}")
    blocks: list[dict[str, str]] = []
    for block in value:
        if not isinstance(block, Mapping) or set(block) != {"block_id", "text"}:
            raise ValueError(f"invalid block shape in {source} for {instance_id}")
        block_id = block["block_id"]
        text = block["text"]
        if not isinstance(block_id, str) or not isinstance(text, str) or not text:
            raise ValueError(f"invalid block content in {source} for {instance_id}")
        canonical_text = " ".join(text.split())
        if not canonical_text:
            raise ValueError(f"invalid block content in {source} for {instance_id}")
        if not canonicalize_text and text != canonical_text:
            raise ValueError(f"non-canonical block text in {source} for {instance_id}")
        blocks.append({"block_id": block_id, "text": canonical_text})
    if tuple(block["block_id"] for block in blocks) != EXPECTED_BLOCK_IDS:
        raise ValueError(f"{source} blocks must be ordered b01 through b23 for {instance_id}")
    return blocks


def _raw_artifact_ids(raw_dir: Path, suffix: str) -> set[str]:
    ids: set[str] = set()
    for path in raw_dir.glob(f"task_*{suffix}"):
        name = path.name.removesuffix(suffix)
        if name in ids:
            raise ValueError(f"duplicate raw artifact for {name}")
        ids.add(name)
    return ids


def _require_exact_ids(kind: str, expected: set[str], actual: set[str]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{kind} coverage mismatch: missing={missing}, extra={extra}")


def _require_exact_raw_files(raw_dir: Path, expected_ids: set[str]) -> None:
    expected_names = {
        name
        for instance_id in expected_ids
        for name in (f"{instance_id}.json", f"{instance_id}.stderr.txt")
    }
    actual_names = {path.name for path in raw_dir.iterdir() if path.is_file()}
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise ValueError(f"unexpected files in raw directory: {unexpected}")


def _artifact_aggregate_sha256(raw_dir: Path, ids: set[str], suffix: str) -> str:
    manifest = [
        {
            "path": f"{instance_id}{suffix}",
            "sha256": _sha256_file(raw_dir / f"{instance_id}{suffix}"),
        }
        for instance_id in sorted(ids)
    ]
    return deterministic_content_hash(manifest)


def _required_string(record: Mapping[str, Any], key: str, kind: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} requires non-empty {key}")
    return value


def _required_sha256(record: Mapping[str, Any], key: str, instance_id: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid {key} SHA-256 for {instance_id}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
