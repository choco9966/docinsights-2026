"""JSONL serialization for Qwen-facing OCR benchmark records."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .models import Document

SCHEMA_VERSION = "1.0"


def success_record(
    *,
    instance_id: str,
    user_query: str,
    document: Document,
    provenance: Mapping[str, Any],
    timing: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "user_query": user_query,
        "pages": [
            {
                "page_number": page.number,
                "width": page.width,
                "height": page.height,
                "coordinate_system": "pixel_top_left",
            }
            for page in document.pages
        ],
        "blocks": [
            {
                "block_id": block.block_id,
                "text": block.text,
                "page_numbers": list(block.page_numbers),
                "lines": [
                    {
                        "page_number": line.page_number,
                        "text": line.text,
                        "bbox": (
                            {
                                "left": line.bbox.left,
                                "top": line.bbox.top,
                                "width": line.bbox.width,
                                "height": line.bbox.height,
                            }
                            if line.bbox is not None
                            else None
                        ),
                        "confidence": line.confidence,
                        "confidence_kind": (
                            provenance.get("confidence_kind") or "unspecified_confidence_0_to_1"
                        ),
                    }
                    for line in block.lines
                ],
            }
            for block in document.blocks
        ],
        "engine": document.engine,
        "provenance": dict(provenance),
        "timing": dict(timing),
        "status": "ok",
    }


def failure_record(
    *,
    instance_id: str,
    user_query: str,
    engine: str,
    provenance: Mapping[str, Any],
    timing: Mapping[str, float],
    error: BaseException | str,
) -> dict[str, Any]:
    """Return a fail-closed record: failed OCR never leaks partial blocks."""
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "user_query": user_query,
        "blocks": [],
        "engine": engine,
        "provenance": dict(provenance),
        "timing": dict(timing),
        "status": "failed",
        "error_kind": _error_kind(error),
        "error": str(error),
    }


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith("\n"):
                    raise ValueError(
                        f"truncated JSONL record at line {line_number} in {path}"
                    ) from exc
                raise ValueError(f"invalid JSONL record at line {line_number} in {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield value


def write_jsonl(
    path: str | Path, records: Iterable[Mapping[str, Any]], *, append: bool = False
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a" if append else "w", encoding="utf-8") as handle:
        for record in records:
            forbidden = {"answer", "evidence"}.intersection(record)
            if forbidden:
                raise ValueError(f"Qwen OCR records must not contain: {sorted(forbidden)}")
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def deterministic_content_hash(value: Any) -> str:
    """Hash JSON-compatible content independently of mapping insertion order."""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def canonical_ocr_content(record: Mapping[str, Any]) -> dict[str, Any]:
    """Select stable OCR semantics, excluding timing, queries, and machine-local paths."""
    provenance = record.get("provenance")
    stable_provenance: dict[str, Any] = {}
    if isinstance(provenance, Mapping):
        for key in (
            "dpi",
            "ocr_engine",
            "language",
            "renderer",
            "confidence_kind",
            "coordinate_system",
            "ocr_executable_identity",
            "renderer_executable_identity",
        ):
            if key in provenance:
                stable_provenance[key] = provenance[key]
        options = provenance.get("ocr_options")
        if isinstance(options, Mapping):
            stable_provenance["ocr_options"] = {
                key: value
                for key, value in options.items()
                if "executable" not in key and "path" not in key
            }
    canonical = {
        "schema_version": record.get("schema_version"),
        "instance_id": record.get("instance_id"),
        "status": record.get("status"),
        "engine": record.get("engine"),
        "pages": record.get("pages", []),
        "blocks": record.get("blocks", []),
        "provenance": stable_provenance,
    }
    if record.get("status") == "failed":
        canonical["error_kind"] = record.get("error_kind")
    return canonical


def ocr_record_hash(record: Mapping[str, Any]) -> str:
    """Hash one record's stable OCR semantics."""
    return deterministic_content_hash(canonical_ocr_content(record))


def aggregate_ocr_hash(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic per-record and aggregate hashes sorted by instance ID."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("OCR hash input requires a non-empty instance_id")
        if instance_id in seen:
            raise ValueError(f"duplicate instance_id in OCR hash input: {instance_id}")
        seen.add(instance_id)
        entries.append({"instance_id": instance_id, "content_hash": ocr_record_hash(record)})
    entries.sort(key=lambda entry: entry["instance_id"])
    return {
        "record_count": len(entries),
        "aggregate_hash": deterministic_content_hash(entries),
        "records": entries,
    }


def _error_kind(error: BaseException | str) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, FileNotFoundError):
        return "file_not_found"
    if isinstance(error, subprocess.CalledProcessError):
        return "subprocess_error"
    if isinstance(error, ValueError):
        return "validation_error"
    if isinstance(error, OSError):
        return "os_error"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    return "unknown_error"
