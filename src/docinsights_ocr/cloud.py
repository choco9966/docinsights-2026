"""Deterministic cloud input bundling, manifest sharding, and result merging."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .records import aggregate_ocr_hash, deterministic_content_hash, read_jsonl, write_jsonl

CLOUD_SHARD_SEED = "docinsights-2026-ocr-cloud-shard-v1"
_RESULT_NAME = re.compile(r"(?:^|.*-)shard-(?P<index>\d+)-of-(?P<count>\d+)\.jsonl$")
_RUNTIME_NAME = re.compile(
    r"runtime-shard-(?P<index>\d+)-of-(?P<count>\d+)\.json$"
)
_BLOCK_ID = re.compile(r"b[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_CLOUD_MANIFEST_FIELDS = frozenset(
    {
        "instance_id",
        "user_query",
        "document_pdf",
        "input_pdf_sha256",
        "split",
        "split_seed",
    }
)
_OCR_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "instance_id",
        "user_query",
        "pages",
        "blocks",
        "engine",
        "provenance",
        "timing",
        "status",
        "error",
        "error_kind",
    }
)
_ERROR_KINDS = frozenset(
    {
        "timeout",
        "file_not_found",
        "subprocess_error",
        "validation_error",
        "os_error",
        "runtime_error",
        "unknown_error",
    }
)
_RUNTIME_PACKAGE_NAMES = frozenset(
    {"paddlepaddle", "paddleocr", "paddlex", "huggingface-hub"}
)


def shard_assignments(
    records: Mapping[str, Mapping[str, Any]],
    shard_count: int,
    *,
    seed: str = CLOUD_SHARD_SEED,
) -> dict[str, int]:
    """Return balanced assignments independent of manifest order, queries, and labels."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least one")
    ranked: list[tuple[str, str]] = []
    for instance_id, record in records.items():
        if not instance_id:
            raise ValueError("instance_id must not be empty")
        input_sha256 = _required_string(record, "input_pdf_sha256")
        key = hashlib.sha256(f"{seed}\0{instance_id}\0{input_sha256}".encode()).hexdigest()
        ranked.append((key, instance_id))
    return {instance_id: rank % shard_count for rank, (_, instance_id) in enumerate(sorted(ranked))}


def split_manifest(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    shard_count: int,
    seed: str = CLOUD_SHARD_SEED,
) -> dict[str, Any]:
    """Write idempotent manifest shards and a deterministic shard plan."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least one")
    source = Path(input_path)
    records = _records_by_id(source, "cloud shard manifest")
    _canonical_cloud_manifest_bytes(records)
    assignments = shard_assignments(records, shard_count, seed=seed)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for instance_id in sorted(records):
        buckets[assignments[instance_id]].append(records[instance_id])

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {_manifest_name(index, shard_count) for index in range(shard_count)} | {
        "shard-plan.json"
    }
    stale = sorted(
        path.name
        for path in destination.iterdir()
        if path.is_file()
        and (path.name == "shard-plan.json" or "-shard-" in path.name)
        and path.name not in expected_names
    )
    if stale:
        raise ValueError(f"stale shard files in {destination}: {stale}")

    shard_rows: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        path = destination / _manifest_name(index, shard_count)
        _write_jsonl_idempotently(path, bucket)
        shard_rows.append(
            {
                "shard_index": index,
                "record_count": len(bucket),
                "manifest_file": path.name,
                "manifest_sha256": _sha256_file(path),
                "instance_ids": [record["instance_id"] for record in bucket],
            }
        )

    plan = {
        "schema_version": "1.0",
        "seed": seed,
        "shard_count": shard_count,
        "source_manifest": source.name,
        "source_manifest_sha256": _sha256_file(source),
        "source_record_count": len(records),
        "assignment_hash": deterministic_content_hash(
            {instance_id: assignments[instance_id] for instance_id in sorted(records)}
        ),
        "shards": shard_rows,
    }
    _write_json_idempotently(destination / "shard-plan.json", plan)
    return plan


def merge_shards(
    manifest_path: str | Path,
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    runtime_paths: Sequence[str | Path],
    seed: str = CLOUD_SHARD_SEED,
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Merge one verified runtime cohort, publishing only canonical records by default."""
    if not shard_paths:
        raise ValueError("at least one shard result is required")
    expected = _records_by_id(manifest_path, "merge manifest")
    canonical_manifest = _canonical_cloud_manifest_bytes(expected)
    manifest_sha256 = _sha256_bytes(canonical_manifest)
    parsed_paths = [_parse_result_path(path) for path in shard_paths]
    shard_counts = {count for _, count, _ in parsed_paths}
    if len(shard_counts) != 1:
        raise ValueError("shard result filenames disagree on shard count")
    shard_count = shard_counts.pop()
    assignments = shard_assignments(expected, shard_count, seed=seed)
    indexes = [index for index, _, _ in parsed_paths]
    if sorted(indexes) != list(range(shard_count)):
        raise ValueError(
            f"shard indexes must be exactly 0..{shard_count - 1}; received {sorted(indexes)}"
        )
    runtime_by_index = _runtime_paths_by_index(runtime_paths, shard_count)

    merged: dict[str, dict[str, Any]] = {}
    configuration: dict[str, Any] | None = None
    cohort: dict[str, Any] | None = None
    for index, _, path in sorted(parsed_paths):
        expected_shard_sha256 = _sha256_bytes(
            _shard_manifest_bytes(expected, assignments, index)
        )
        runtime = _validated_runtime(
            runtime_by_index[index],
            result_path=path,
            shard_index=index,
            shard_count=shard_count,
            manifest_sha256=manifest_sha256,
            shard_manifest_sha256=expected_shard_sha256,
        )
        current_cohort = _runtime_cohort_signature(runtime)
        if cohort is None:
            cohort = current_cohort
        elif current_cohort != cohort:
            raise ValueError(f"runtime cohort mismatch for shard {index}")
        shard_records = list(read_jsonl(path))
        failed_count = sum(record.get("status") == "failed" for record in shard_records)
        if runtime["record_count"] != len(shard_records):
            raise ValueError(f"runtime record_count mismatch for shard {index}")
        if runtime["failed_count"] != failed_count:
            raise ValueError(f"runtime failed_count mismatch for shard {index}")
        for record in shard_records:
            _validate_ocr_record(record)
            instance_id = _required_string(record, "instance_id")
            if instance_id in merged:
                raise ValueError(f"duplicate instance_id across shard results: {instance_id}")
            expected_record = expected.get(instance_id)
            if expected_record is None:
                raise ValueError(f"unexpected instance_id in shard results: {instance_id}")
            assigned = assignments[instance_id]
            if assigned != index:
                raise ValueError(
                    f"instance_id {instance_id} belongs to shard {assigned}, not shard {index}"
                )
            _validate_result_against_manifest(record, expected_record, runtime)
            current_configuration = _configuration_signature(record)
            if configuration is None:
                configuration = current_configuration
            elif current_configuration != configuration:
                raise ValueError(f"OCR configuration mismatch in shard record {instance_id}")
            merged[instance_id] = record

    missing = sorted(expected.keys() - merged.keys())
    if missing:
        raise ValueError(f"missing shard result instance_ids: {missing}")
    ordered = [merged[instance_id] for instance_id in sorted(merged)]
    failed = [record["instance_id"] for record in ordered if record["status"] != "ok"]
    if fail_closed and failed:
        raise ValueError(f"canonical merge rejects failed OCR records: {failed}")
    _atomic_write_jsonl(Path(output_path), ordered)
    digest = aggregate_ocr_hash(ordered)
    return {
        "schema_version": "1.0",
        "seed": seed,
        "shard_count": shard_count,
        "record_count": len(ordered),
        "ok_count": sum(record.get("status") == "ok" for record in ordered),
        "failed_count": sum(record.get("status") == "failed" for record in ordered),
        "aggregate_hash": digest["aggregate_hash"],
        "output_sha256": _sha256_file(Path(output_path)),
        "configuration": configuration,
        "runtime_cohort": cohort,
    }


def pack_cloud_input(
    manifest_path: str | Path,
    documents_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a deterministic, label-free archive for Colab or Kaggle."""
    manifest = Path(manifest_path)
    root = Path(documents_root).resolve()
    records = _records_by_id(manifest, "cloud bundle manifest")
    canonical_manifest = _canonical_cloud_manifest_bytes(records)
    files_by_archive: dict[PurePosixPath, tuple[Path, str]] = {}
    for instance_id in sorted(records):
        record = records[instance_id]
        relative = _safe_relative_path(_required_string(record, "document_pdf"))
        source = (root / Path(relative.as_posix())).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"document escapes documents_root: {relative}")
        expected_sha256 = _required_string(record, "input_pdf_sha256")
        actual_sha256 = _sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"input PDF SHA-256 mismatch for {instance_id}")
        archive_path = PurePosixPath("bundle/documents-root") / relative
        previous = files_by_archive.get(archive_path)
        if previous is not None and previous[1] != actual_sha256:
            raise ValueError(f"conflicting document archive path: {archive_path}")
        files_by_archive[archive_path] = (source, actual_sha256)
    files = [
        (source, archive_path, digest)
        for archive_path, (source, digest) in sorted(files_by_archive.items())
    ]

    bundle_manifest = {
        "schema_version": "1.0",
        "record_count": len(records),
        "manifest_sha256": _sha256_bytes(canonical_manifest),
        "documents": [
            {"archive_path": archive.as_posix(), "sha256": digest} for _, archive, digest in files
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            _add_bytes(archive, "bundle/manifest.jsonl", canonical_manifest)
            _add_bytes(
                archive,
                "bundle/bundle.json",
                (
                    json.dumps(bundle_manifest, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode(),
            )
            for source, archive_path, _ in files:
                _add_bytes(archive, archive_path.as_posix(), source.read_bytes())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema_version": bundle_manifest["schema_version"],
        "record_count": bundle_manifest["record_count"],
        "manifest_sha256": bundle_manifest["manifest_sha256"],
        "document_manifest_hash": deterministic_content_hash(bundle_manifest["documents"]),
        "archive": str(destination),
        "archive_sha256": _sha256_file(destination),
        "archive_bytes": destination.stat().st_size,
    }


def _runtime_paths_by_index(
    runtime_paths: Sequence[str | Path], shard_count: int
) -> dict[int, Path]:
    if not runtime_paths:
        raise ValueError("one runtime sidecar per shard result is required")
    parsed = [_parse_runtime_path(path) for path in runtime_paths]
    counts = {count for _, count, _ in parsed}
    if counts != {shard_count}:
        raise ValueError(
            "runtime sidecar filenames must use shard count "
            f"{shard_count}; received {sorted(counts)}"
        )
    indexes = [index for index, _, _ in parsed]
    if sorted(indexes) != list(range(shard_count)):
        raise ValueError(
            "runtime sidecar indexes must be exactly "
            f"0..{shard_count - 1}; received {sorted(indexes)}"
        )
    return {index: path for index, _, path in parsed}


def _validated_runtime(
    runtime_path: Path,
    *,
    result_path: Path,
    shard_index: int,
    shard_count: int,
    manifest_sha256: str,
    shard_manifest_sha256: str,
) -> dict[str, Any]:
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid runtime sidecar: {runtime_path}") from exc
    if not isinstance(runtime, dict):
        raise ValueError(f"runtime sidecar must be a JSON object: {runtime_path}")
    if runtime.get("schema_version") != "1.0":
        raise ValueError(f"unsupported runtime sidecar schema: {runtime_path}")
    if runtime.get("shard_index") != shard_index or isinstance(
        runtime.get("shard_index"), bool
    ):
        raise ValueError(f"runtime shard_index mismatch for shard {shard_index}")
    if runtime.get("shard_count") != shard_count or isinstance(
        runtime.get("shard_count"), bool
    ):
        raise ValueError(f"runtime shard_count mismatch for shard {shard_index}")
    _required_string(runtime, "session_fingerprint")
    for key in (
        "platform_role",
        "platform",
        "machine",
        "python",
    ):
        _required_string(runtime, key)
    repository_sha = _required_git_commit(runtime, "repository_sha")
    if runtime.get("pipeline_revision") != repository_sha:
        raise ValueError(f"runtime pipeline_revision mismatch for shard {shard_index}")
    if not isinstance(runtime.get("repository_dirty"), bool):
        raise ValueError(f"runtime repository_dirty must be boolean for shard {shard_index}")
    if runtime["repository_dirty"]:
        raise ValueError(f"canonical merge requires a clean repository for shard {shard_index}")
    for key in (
        "bundle_sha256",
        "manifest_sha256",
        "shard_manifest_sha256",
        "result_sha256",
    ):
        _required_sha256(runtime, key)
    if runtime["manifest_sha256"] != manifest_sha256:
        raise ValueError(f"full manifest SHA-256 mismatch for shard {shard_index}")
    if runtime["shard_manifest_sha256"] != shard_manifest_sha256:
        raise ValueError(f"shard manifest SHA-256 mismatch for shard {shard_index}")
    if runtime["result_sha256"] != _sha256_file(result_path):
        raise ValueError(f"result SHA-256 mismatch for shard {shard_index}")
    timeout_seconds = runtime.get("timeout_seconds")
    if timeout_seconds is not None and (
        not _is_nonnegative_number(timeout_seconds) or timeout_seconds == 0
    ):
        raise ValueError(
            f"runtime timeout_seconds must be positive or null for shard {shard_index}"
        )
    for key in ("record_count", "failed_count"):
        if not _is_nonnegative_integer(runtime.get(key)):
            raise ValueError(
                f"runtime {key} must be a non-negative integer for shard {shard_index}"
            )
    if runtime["failed_count"] > runtime["record_count"]:
        raise ValueError(f"runtime failed_count exceeds record_count for shard {shard_index}")
    runtime_version = runtime.get("runtime_version")
    if not isinstance(runtime_version, Mapping) or any(
        not isinstance(key, str) or not key or value is not None and not isinstance(value, str)
        for key, value in runtime_version.items()
    ):
        raise ValueError(
            "runtime_version must contain string or null values "
            f"for shard {shard_index}"
        )
    for model_key in ("detector", "recognizer"):
        model = runtime.get(model_key)
        if not isinstance(model, Mapping):
            raise ValueError(f"runtime {model_key} must be an object for shard {shard_index}")
        _required_string(model, "repo")
        _required_git_commit(model, "revision")
    packages = runtime.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError(f"runtime packages must be an object for shard {shard_index}")
    missing_packages = sorted(_RUNTIME_PACKAGE_NAMES - packages.keys())
    if missing_packages:
        raise ValueError(
            f"runtime packages missing for shard {shard_index}: {missing_packages}"
        )
    for name, version in packages.items():
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError(
                "runtime package versions must be non-empty strings "
                f"for shard {shard_index}"
            )
    return runtime


def _runtime_cohort_signature(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": runtime["schema_version"],
        "platform_role": runtime["platform_role"],
        "platform": runtime["platform"],
        "machine": runtime["machine"],
        "python": runtime["python"],
        "repository_sha": runtime["repository_sha"],
        "repository_dirty": runtime["repository_dirty"],
        "pipeline_revision": runtime["pipeline_revision"],
        "bundle_sha256": runtime["bundle_sha256"],
        "manifest_sha256": runtime["manifest_sha256"],
        "timeout_seconds": runtime["timeout_seconds"],
        "runtime_version": dict(sorted(runtime["runtime_version"].items())),
        "detector": {
            "repo": runtime["detector"]["repo"],
            "revision": runtime["detector"]["revision"],
        },
        "recognizer": {
            "repo": runtime["recognizer"]["repo"],
            "revision": runtime["recognizer"]["revision"],
        },
        "packages": dict(sorted(runtime["packages"].items())),
    }


def _configuration_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("shard result provenance must be an object")
    options = provenance.get("ocr_options")
    if not isinstance(options, Mapping):
        raise ValueError("shard result provenance.ocr_options must be an object")
    stable_options = {
        key: value
        for key, value in options.items()
        if "path" not in key and "executable" not in key
    }
    return {
        "engine": record.get("engine"),
        "dpi": provenance.get("dpi"),
        "language": provenance.get("language"),
        "renderer": provenance.get("renderer"),
        "confidence_kind": provenance.get("confidence_kind"),
        "coordinate_system": provenance.get("coordinate_system"),
        "ocr_options": stable_options,
        "ocr_executable_identity": provenance.get("ocr_executable_identity"),
        "renderer_executable_identity": provenance.get("renderer_executable_identity"),
    }


def _validate_result_against_manifest(
    record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    forbidden = {
        "answer",
        "candidate_answers",
        "evidence",
        "labels",
        "portal_score",
        "portal_scores",
    }.intersection(record)
    if forbidden:
        raise ValueError(f"OCR shard result must not contain: {sorted(forbidden)}")
    instance_id = _required_string(record, "instance_id")
    if record.get("user_query") != manifest_record.get("user_query"):
        raise ValueError(f"user_query mismatch for {instance_id}")
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"invalid provenance for {instance_id}")
    if provenance.get("input_pdf_sha256") != manifest_record.get("input_pdf_sha256"):
        raise ValueError(f"input PDF SHA-256 mismatch in result for {instance_id}")
    for key in ("split", "split_seed"):
        if provenance.get(key) != manifest_record.get(key):
            raise ValueError(f"{key} mismatch in result for {instance_id}")
    pipeline_revision = runtime["repository_sha"]
    if provenance.get("pipeline_revision") != pipeline_revision:
        raise ValueError(f"pipeline revision mismatch in result for {instance_id}")
    if provenance.get("timeout_seconds") != runtime["timeout_seconds"]:
        raise ValueError(f"timeout mismatch in result for {instance_id}")
    run_fingerprint = provenance.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or _SHA256.fullmatch(run_fingerprint) is None:
        raise ValueError(f"invalid run fingerprint in result for {instance_id}")
    options = provenance.get("ocr_options")
    if not isinstance(options, Mapping):
        raise ValueError(f"invalid OCR options in result for {instance_id}")
    expected_options = {
        "pipeline_revision": pipeline_revision,
        "detection_model_repo": runtime["detector"]["repo"],
        "detection_model_revision": runtime["detector"]["revision"],
        "recognition_model_repo": runtime["recognizer"]["repo"],
        "recognition_model_revision": runtime["recognizer"]["revision"],
        "paddlepaddle_version": runtime["packages"]["paddlepaddle"],
        "paddleocr_version": runtime["packages"]["paddleocr"],
        "paddlex_version": runtime["packages"]["paddlex"],
    }
    for key, expected_value in expected_options.items():
        if options.get(key) != expected_value:
            raise ValueError(f"{key} mismatch in result for {instance_id}")


def _validate_ocr_record(record: Mapping[str, Any]) -> None:
    unexpected = sorted(record.keys() - _OCR_TOP_LEVEL_FIELDS)
    if unexpected:
        raise ValueError(f"OCR record contains unknown top-level fields: {unexpected}")
    required = {
        "schema_version",
        "instance_id",
        "user_query",
        "blocks",
        "engine",
        "provenance",
        "timing",
        "status",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"OCR record missing required fields: {missing}")
    if record["schema_version"] != "1.0":
        raise ValueError("OCR record schema_version must be '1.0'")
    _required_string(record, "instance_id")
    _required_string(record, "user_query")
    _required_string(record, "engine")
    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("OCR record provenance must be an object")
    timing = record["timing"]
    if not isinstance(timing, Mapping):
        raise ValueError("OCR record timing must be an object")
    if any(not _is_nonnegative_number(value) for value in timing.values()):
        raise ValueError("OCR record timing values must be finite non-negative numbers")
    blocks = record["blocks"]
    if not isinstance(blocks, list):
        raise ValueError("OCR record blocks must be an array")
    for block in blocks:
        _validate_block(block)
    block_ids = [block["block_id"] for block in blocks]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("OCR record block_id values must be unique")
    block_numbers = [int(block_id[1:]) for block_id in block_ids]
    if block_numbers != sorted(block_numbers):
        raise ValueError("OCR record blocks must be ordered by block_id")
    status = record["status"]
    if not isinstance(status, str) or status not in {"ok", "failed"}:
        raise ValueError(f"OCR record has invalid status: {status!r}")
    if "error" in record and (
        not isinstance(record["error"], str) or not record["error"]
    ):
        raise ValueError("OCR record error must be a non-empty string")
    if "error_kind" in record:
        error_kind = record["error_kind"]
        if not isinstance(error_kind, str) or error_kind not in _ERROR_KINDS:
            raise ValueError(f"OCR record has invalid error_kind: {error_kind!r}")
    if "pages" in record:
        pages = record["pages"]
        if not isinstance(pages, list) or not pages:
            raise ValueError("OCR record pages must be a non-empty array")
        for page in pages:
            _validate_page(page)
        page_numbers = [page["page_number"] for page in pages]
        if page_numbers != sorted(set(page_numbers)):
            raise ValueError("OCR record pages must have ordered unique page numbers")
        page_number_set = set(page_numbers)
        for block in blocks:
            if not set(block["page_numbers"]).issubset(page_number_set):
                raise ValueError("OCR block references an undeclared page")
            if any(
                line["page_number"] not in block["page_numbers"]
                for line in block["lines"]
            ):
                raise ValueError("OCR line page must belong to its block")
    if status == "ok":
        if "pages" not in record:
            raise ValueError("successful OCR record requires pages")
        if not blocks:
            raise ValueError("successful OCR record requires at least one block")
        if "error" in record or "error_kind" in record:
            raise ValueError("successful OCR record must not contain an error")
    if status == "failed":
        if "error" not in record or "error_kind" not in record:
            raise ValueError("failed OCR record requires error and error_kind")
        if blocks:
            raise ValueError("failed OCR record must not contain blocks")


def _validate_page(page: Any) -> None:
    fields = {"page_number", "width", "height", "coordinate_system"}
    if not isinstance(page, Mapping) or set(page) != fields:
        raise ValueError("OCR page must contain exactly the schema fields")
    if not _is_positive_integer(page["page_number"]):
        raise ValueError("OCR page_number must be a positive integer")
    for key in ("width", "height"):
        if page[key] is not None and not _is_positive_integer(page[key]):
            raise ValueError(f"OCR page {key} must be null or a positive integer")
    if page["coordinate_system"] != "pixel_top_left":
        raise ValueError("OCR page coordinate_system must be 'pixel_top_left'")


def _validate_block(block: Any) -> None:
    fields = {"block_id", "text", "page_numbers", "lines"}
    if not isinstance(block, Mapping) or set(block) != fields:
        raise ValueError("OCR block must contain exactly the schema fields")
    if not isinstance(block["block_id"], str) or _BLOCK_ID.fullmatch(block["block_id"]) is None:
        raise ValueError("OCR block_id must match ^b[0-9]+$")
    if not isinstance(block["text"], str):
        raise ValueError("OCR block text must be a string")
    page_numbers = block["page_numbers"]
    if (
        not isinstance(page_numbers, list)
        or not page_numbers
        or any(not _is_positive_integer(value) for value in page_numbers)
        or len(page_numbers) != len(set(page_numbers))
    ):
        raise ValueError("OCR block page_numbers must be non-empty unique positive integers")
    lines = block["lines"]
    if not isinstance(lines, list) or not lines:
        raise ValueError("OCR block lines must be a non-empty array")
    for line in lines:
        _validate_line(line)


def _validate_line(line: Any) -> None:
    fields = {"page_number", "text", "bbox", "confidence", "confidence_kind"}
    if not isinstance(line, Mapping) or set(line) != fields:
        raise ValueError("OCR line must contain exactly the schema fields")
    if not _is_positive_integer(line["page_number"]):
        raise ValueError("OCR line page_number must be a positive integer")
    if not isinstance(line["text"], str):
        raise ValueError("OCR line text must be a string")
    confidence = line["confidence"]
    if confidence is not None and (
        not _is_nonnegative_number(confidence) or confidence > 1
    ):
        raise ValueError("OCR line confidence must be null or between 0 and 1")
    if not isinstance(line["confidence_kind"], str) or not line["confidence_kind"]:
        raise ValueError("OCR line confidence_kind must be a non-empty string")
    bbox = line["bbox"]
    if bbox is None:
        return
    fields = {"left", "top", "width", "height"}
    if not isinstance(bbox, Mapping) or set(bbox) != fields:
        raise ValueError("OCR bbox must contain exactly the schema fields")
    if any(not _is_nonnegative_integer(bbox[key]) for key in fields):
        raise ValueError("OCR bbox values must be non-negative integers")


def _canonical_cloud_manifest_bytes(records: Mapping[str, Mapping[str, Any]]) -> bytes:
    rows: list[bytes] = []
    for instance_id in sorted(records):
        record = records[instance_id]
        _validate_cloud_manifest_record(record)
        rows.append(
            (
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
    return b"".join(rows)


def _validate_cloud_manifest_record(record: Mapping[str, Any]) -> None:
    unexpected = sorted(record.keys() - _CLOUD_MANIFEST_FIELDS)
    if unexpected:
        raise ValueError(f"cloud bundle manifest contains fields outside allowlist: {unexpected}")
    missing = sorted(_CLOUD_MANIFEST_FIELDS - record.keys())
    if missing:
        raise ValueError(f"cloud bundle manifest missing required fields: {missing}")
    for key in ("instance_id", "user_query", "document_pdf", "split", "split_seed"):
        _required_string(record, key)
    _required_sha256(record, "input_pdf_sha256")


def _shard_manifest_bytes(
    records: Mapping[str, Mapping[str, Any]], assignments: Mapping[str, int], shard_index: int
) -> bytes:
    buffer = io.StringIO()
    for instance_id in sorted(records):
        if assignments[instance_id] == shard_index:
            buffer.write(
                json.dumps(dict(records[instance_id]), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return buffer.getvalue().encode()


def _required_sha256(record: Mapping[str, Any], key: str) -> str:
    value = _required_string(record, key)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"record requires a lowercase SHA-256 string {key!r}")
    return value


def _required_git_commit(record: Mapping[str, Any], key: str) -> str:
    value = _required_string(record, key)
    if _GIT_COMMIT.fullmatch(value) is None:
        raise ValueError(f"record requires a lowercase 40-character Git commit {key!r}")
    return value


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _parse_result_path(path: str | Path) -> tuple[int, int, Path]:
    parsed = Path(path)
    match = _RESULT_NAME.fullmatch(parsed.name)
    if match is None:
        raise ValueError(
            f"shard result filename must end in shard-<index>-of-<count>.jsonl: {parsed.name}"
        )
    index = int(match.group("index"))
    count = int(match.group("count"))
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid shard result filename: {parsed.name}")
    return index, count, parsed


def _parse_runtime_path(path: str | Path) -> tuple[int, int, Path]:
    parsed = Path(path)
    match = _RUNTIME_NAME.fullmatch(parsed.name)
    if match is None:
        raise ValueError(
            "runtime sidecar filename must be "
            f"runtime-shard-<index>-of-<count>.json: {parsed.name}"
        )
    index = int(match.group("index"))
    count = int(match.group("count"))
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid runtime sidecar filename: {parsed.name}")
    return index, count, parsed


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"document_pdf must be a safe relative path: {value}")
    return path


def _manifest_name(index: int, count: int) -> str:
    width = max(2, len(str(count - 1)))
    return f"manifest-shard-{index:0{width}d}-of-{count:0{width}d}.jsonl"


def _records_by_id(path: str | Path, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        instance_id = _required_string(record, "instance_id")
        if instance_id in records:
            raise ValueError(f"duplicate instance_id in {label}: {instance_id}")
        records[instance_id] = record
    return records


def _required_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record requires a non-empty string {key!r}")
    return value


def _write_jsonl_idempotently(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    rows = list(records)
    buffer = io.StringIO()
    for record in rows:
        buffer.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    expected = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing shard file differs from requested plan: {path}")
        return
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(path)


def _write_json_idempotently(path: Path, value: Mapping[str, Any]) -> None:
    expected = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing shard plan differs from requested plan: {path}")
        return
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        write_jsonl(temporary, records)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _add_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(content))
