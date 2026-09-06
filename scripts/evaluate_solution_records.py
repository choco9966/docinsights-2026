#!/usr/bin/env python3
"""Evaluate a completed frozen solution split against an allowed post-freeze reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from docinsights_analysis.solution_records import (
    SolutionRecordError,
    evaluate_records,
    write_private_atomic,
)

_GENERATION_FILES = ("solutions.jsonl", "solutions.md", "submission.jsonl")
_REFERENCE_KIND = {
    "train": "train-public-labels",
    "validation": "validation-v24-reference",
}
_PINNED_REFERENCE_METADATA = {
    "train": {
        "size": 62030,
        "git_blob_sha1": "686429d03b2d4ba5fe4fe6b07398feef7d0cd884",
    },
    "validation": {
        "sha256": "265d89696e1216f887ea3722c13bf83b3ac5c03292c7460db9afc8732328ec13"
    },
}
_SHA256_HEX_LENGTH = 64


class EvaluationError(ValueError):
    """The split cannot be evaluated without violating the post-freeze contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = b"blob " + str(len(value)).encode("ascii") + b"\0"
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _validate_reference_pin(split: str, value: bytes) -> None:
    pin = _PINNED_REFERENCE_METADATA.get(split)
    if pin is None:
        return
    if "sha256" in pin and _sha256_bytes(value) != pin["sha256"]:
        raise EvaluationError(f"{split} reference does not match pinned reference metadata")
    if (
        "size" in pin
        and "git_blob_sha1" in pin
        and (len(value) != pin["size"] or _git_blob_sha1(value) != pin["git_blob_sha1"])
    ):
        raise EvaluationError(f"{split} reference does not match pinned reference metadata")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise EvaluationError(f"cannot read frozen generation file {path}: {error}") from error
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid {description}: {error}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"invalid {description}: expected a JSON object")
    return value


def _parse_jsonl(raw: bytes, description: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvaluationError(f"invalid {description}: expected UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise EvaluationError(f"invalid {description}: blank line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"invalid {description} at line {line_number}: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise EvaluationError(
                f"invalid {description} at line {line_number}: expected an object"
            )
        rows.append(row)
    return rows


def _unique_ids(rows: list[dict[str, Any]], description: str) -> list[str]:
    identifiers = [row.get("instance_id") for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in identifiers)
        or len(identifiers) != len(set(identifiers))
    ):
        raise EvaluationError(f"{description} must have unique ID coverage")
    return [value for value in identifiers if isinstance(value, str)]


def _write_evaluation_once(path: Path, evaluation: dict[str, Any]) -> None:
    serialized = json.dumps(
        evaluation, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        write_private_atomic(path, serialized, overwrite=False)
    except FileExistsError as error:
        raise EvaluationError(
            "evaluation.json already exists; frozen evaluation is immutable"
        ) from error
    except SolutionRecordError as error:
        raise EvaluationError(str(error)) from error


def _verify_frozen_export(
    split_dir: Path, expected_split: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    split_dir = split_dir.resolve()
    manifest_path = split_dir / "manifest.json"
    run_manifest_path = split_dir / "run-manifest.json"
    completion_path = split_dir / "complete.json"
    manifest = _load_json(manifest_path, "frozen export manifest")
    split = manifest.get("split")
    if split not in {"train", "validation", "heldout"}:
        raise EvaluationError("frozen export manifest has an invalid split")
    if expected_split is not None and split != expected_split:
        raise EvaluationError(f"evaluation split is {split}, expected {expected_split}")
    declared_ids = manifest.get("instance_ids")
    total = manifest.get("total")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("private") is not True
        or not isinstance(total, int)
        or total < 1
        or not isinstance(declared_ids, list)
        or len(declared_ids) != total
        or any(not isinstance(value, str) or not value for value in declared_ids)
        or len(declared_ids) != len(set(declared_ids))
    ):
        raise EvaluationError("frozen export manifest does not prove unique full coverage")

    file_manifest = manifest.get("files")
    if not isinstance(file_manifest, dict) or set(file_manifest) != set(_GENERATION_FILES):
        raise EvaluationError("frozen export manifest has incomplete file hashes")
    generation_file_hashes: dict[str, str] = {}
    for name in _GENERATION_FILES:
        declaration = file_manifest[name]
        expected_hash = declaration.get("sha256") if isinstance(declaration, dict) else None
        actual_hash = _sha256_file(split_dir / name)
        if expected_hash != actual_hash:
            raise EvaluationError(f"frozen generation file hash mismatch: {name}")
        generation_file_hashes[name] = actual_hash

    try:
        solution_bytes = (split_dir / "solutions.jsonl").read_bytes()
        submission_bytes = (split_dir / "submission.jsonl").read_bytes()
    except OSError as error:
        raise EvaluationError(f"cannot read frozen generation records: {error}") from error
    records = _parse_jsonl(solution_bytes, "solutions.jsonl")
    submissions = _parse_jsonl(submission_bytes, "submission.jsonl")
    record_ids = _unique_ids(records, "solutions.jsonl")
    submission_ids = _unique_ids(submissions, "submission.jsonl")
    if record_ids != declared_ids or submission_ids != declared_ids:
        raise EvaluationError("frozen generation does not have unique full coverage")
    for record, submission in zip(records, submissions, strict=True):
        if record.get("split") != split:
            raise EvaluationError("frozen generation contains a mismatched split")
        projection = {
            "instance_id": record.get("instance_id"),
            "answer": record.get("answer"),
            "evidence": record.get("evidence"),
        }
        if submission != projection:
            raise EvaluationError("submission.jsonl does not match the frozen solution projection")

    run_manifest = _load_json(run_manifest_path, "run manifest")
    tasks = run_manifest.get("tasks")
    config = run_manifest.get("command_config")
    if (
        run_manifest.get("split") != split
        or run_manifest.get("official_task_count") != total
        or not isinstance(tasks, list)
        or len(tasks) != total
        or run_manifest.get("input_manifest_sha256") != _sha256_bytes(_canonical(tasks))
        or not isinstance(config, dict)
        or run_manifest.get("command_config_sha256") != _sha256_bytes(_canonical(config))
    ):
        raise EvaluationError("run manifest does not bind the completed frozen generation")
    task_ids = [
        entry.get("task", {}).get("instance_id") if isinstance(entry, dict) else None
        for entry in tasks
    ]
    if (
        any(not isinstance(value, str) or not value for value in task_ids)
        or len(task_ids) != len(set(task_ids))
        or set(task_ids) != set(declared_ids)
    ):
        raise EvaluationError("run manifest does not have unique full coverage")

    completion = _load_json(completion_path, "completion marker")
    if (
        completion.get("status") != "generation_complete"
        or completion.get("split") != split
        or completion.get("total") != total
        or completion.get("input_manifest_sha256")
        != run_manifest["input_manifest_sha256"]
        or completion.get("command_config_sha256")
        != run_manifest["command_config_sha256"]
        or completion.get("record_ids_sha256") != _sha256_bytes(_canonical(task_ids))
        or completion.get("export_manifest_sha256") != _sha256_file(manifest_path)
    ):
        raise EvaluationError("completion marker does not bind full frozen coverage")

    generation = {
        "manifest_sha256": _sha256_file(manifest_path),
        "complete_sha256": _sha256_file(completion_path),
        "run_manifest_sha256": _sha256_file(run_manifest_path),
        "files": generation_file_hashes,
        "instance_ids_sha256": _sha256_bytes(_canonical(declared_ids)),
    }
    return generation, records, declared_ids


def evaluate_split(
    split_dir: Path, reference_path: Path, reference_kind: str
) -> dict[str, Any]:
    """Evaluate one completed export, reading references only after generation verification."""
    split_dir = split_dir.resolve()
    generation, records, instance_ids = _verify_frozen_export(split_dir)
    split = records[0]["split"]
    if split == "heldout":
        raise EvaluationError("heldout reference labels are unavailable")
    required_kind = _REFERENCE_KIND[split]
    if reference_kind != required_kind:
        raise EvaluationError(
            f"{split} requires reference kind {required_kind}, got {reference_kind}"
        )
    evaluation_path = split_dir / "evaluation.json"
    if evaluation_path.exists():
        raise EvaluationError("evaluation.json already exists; frozen evaluation is immutable")

    reference_path = reference_path.resolve()
    if reference_path == split_dir or split_dir in reference_path.parents:
        raise EvaluationError("reference must be outside the frozen split directory")
    try:
        reference_bytes = reference_path.read_bytes()
    except OSError as error:
        raise EvaluationError(f"cannot read {reference_kind} reference: {error}") from error
    reference_hash = _sha256_bytes(reference_bytes)
    _validate_reference_pin(split, reference_bytes)
    references = _parse_jsonl(reference_bytes, f"{reference_kind} reference")
    try:
        comparison = evaluate_records(records, references, reference_kind)
    except SolutionRecordError as error:
        raise EvaluationError(str(error)) from error

    generation_after, _, ids_after = _verify_frozen_export(split_dir, split)
    if generation_after != generation or ids_after != instance_ids:
        raise EvaluationError("frozen generation changed while the reference was being evaluated")
    evaluation = {
        "schema_version": 1,
        "private": True,
        "status": "evaluated",
        "split": split,
        "total": len(instance_ids),
        "generation": generation,
        "reference": {
            "kind": reference_kind,
            "path": str(reference_path),
            "sha256": reference_hash,
            "size": len(reference_bytes),
            "git_blob_sha1": _git_blob_sha1(reference_bytes),
            "total": len(references),
        },
        "comparisons": comparison["comparisons"],
        "aggregate": comparison["aggregate"],
    }
    _write_evaluation_once(evaluation_path, evaluation)
    return evaluation


def record_unavailable_evaluation(split_dir: Path) -> dict[str, Any]:
    """Record Held-out generation coverage without reading or inventing labels."""
    split_dir = split_dir.resolve()
    generation, records, instance_ids = _verify_frozen_export(split_dir)
    if any(record.get("split") != "heldout" for record in records):
        raise EvaluationError("no-reference evaluation is only valid for heldout")
    evaluation_path = split_dir / "evaluation.json"
    if evaluation_path.exists():
        raise EvaluationError("evaluation.json already exists; frozen evaluation is immutable")
    generation_after, _, ids_after = _verify_frozen_export(split_dir, "heldout")
    if generation_after != generation or ids_after != instance_ids:
        raise EvaluationError("frozen generation changed while recording coverage")
    reference_kind = "heldout-labels-unavailable"
    evaluation = {
        "schema_version": 1,
        "private": True,
        "status": "reference_unavailable",
        "split": "heldout",
        "total": len(instance_ids),
        "generation": generation,
        "reference": {
            "available": False,
            "kind": reference_kind,
            "path": None,
            "sha256": None,
            "total": 0,
        },
        "comparisons": [],
        "aggregate": {
            "reference_kind": reference_kind,
            "total": len(instance_ids),
            "reference_coverage": 0,
            "answer_matches": None,
            "evidence_matches": None,
        },
    }
    _write_evaluation_once(evaluation_path, evaluation)
    return evaluation


def verify_evaluation_binding(split_dir: Path, expected_split: str) -> dict[str, Any]:
    """Verify evaluation.json still binds the current generation and reference bytes."""
    if expected_split not in {*_REFERENCE_KIND, "heldout"}:
        raise EvaluationError(f"unsupported evaluation split: {expected_split}")
    split_dir = split_dir.resolve()
    generation, records, instance_ids = _verify_frozen_export(split_dir, expected_split)
    evaluation_path = split_dir / "evaluation.json"
    evaluation = _load_json(evaluation_path, "evaluation binding")
    if evaluation_path.stat().st_mode & 0o077:
        raise EvaluationError("evaluation.json is not private")
    if (
        evaluation.get("schema_version") != 1
        or evaluation.get("private") is not True
        or evaluation.get("split") != expected_split
        or evaluation.get("total") != len(instance_ids)
        or evaluation.get("generation") != generation
    ):
        raise EvaluationError("evaluation does not bind the current frozen generation")

    if expected_split == "heldout":
        reference_kind = "heldout-labels-unavailable"
        expected_reference = {
            "available": False,
            "kind": reference_kind,
            "path": None,
            "sha256": None,
            "total": 0,
        }
        expected_aggregate = {
            "reference_kind": reference_kind,
            "total": len(instance_ids),
            "reference_coverage": 0,
            "answer_matches": None,
            "evidence_matches": None,
        }
        if (
            evaluation.get("status") != "reference_unavailable"
            or evaluation.get("reference") != expected_reference
            or evaluation.get("comparisons") != []
            or evaluation.get("aggregate") != expected_aggregate
        ):
            raise EvaluationError("heldout evaluation invents unavailable reference results")
        return evaluation

    if evaluation.get("status") != "evaluated":
        raise EvaluationError("evaluation does not have evaluated status")

    reference = evaluation.get("reference")
    if not isinstance(reference, dict) or reference.get("kind") != _REFERENCE_KIND[expected_split]:
        raise EvaluationError("evaluation has the wrong reference kind")
    reference_path_value = reference.get("path")
    expected_reference_hash = reference.get("sha256")
    if (
        not isinstance(reference_path_value, str)
        or not isinstance(expected_reference_hash, str)
        or len(expected_reference_hash) != _SHA256_HEX_LENGTH
    ):
        raise EvaluationError("evaluation has an invalid reference binding")
    reference_path = Path(reference_path_value)
    try:
        reference_bytes = reference_path.read_bytes()
    except OSError as error:
        raise EvaluationError(f"cannot read bound reference: {error}") from error
    if _sha256_bytes(reference_bytes) != expected_reference_hash:
        raise EvaluationError("reference SHA-256 no longer matches evaluation")
    if (
        reference.get("size") != len(reference_bytes)
        or reference.get("git_blob_sha1") != _git_blob_sha1(reference_bytes)
    ):
        raise EvaluationError("reference metadata no longer matches evaluation")
    _validate_reference_pin(expected_split, reference_bytes)

    comparisons = evaluation.get("comparisons")
    if not isinstance(comparisons, list) or _unique_ids(comparisons, "evaluation comparisons") != (
        instance_ids
    ):
        raise EvaluationError("evaluation comparisons do not have unique full coverage")
    if any(
        set(row) != {"instance_id", "answer_match", "evidence_match"}
        or not isinstance(row["answer_match"], bool)
        or not isinstance(row["evidence_match"], bool)
        for row in comparisons
    ):
        raise EvaluationError("evaluation comparisons contain invalid match values")
    references = _parse_jsonl(reference_bytes, "bound reference")
    try:
        recomputed = evaluate_records(records, references, _REFERENCE_KIND[expected_split])
    except SolutionRecordError as error:
        raise EvaluationError(str(error)) from error
    if (
        comparisons != recomputed["comparisons"]
        or evaluation.get("aggregate") != recomputed["aggregate"]
        or reference.get("total") != len(instance_ids)
    ):
        raise EvaluationError("evaluation aggregate does not match full comparison coverage")
    return evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", type=Path)
    source.add_argument("--no-reference", action="store_true")
    parser.add_argument(
        "--reference-kind",
        choices=tuple(_REFERENCE_KIND.values()),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.no_reference:
            if args.reference_kind is not None:
                raise EvaluationError("--reference-kind cannot be used with --no-reference")
            evaluation = record_unavailable_evaluation(args.split_dir)
        else:
            if args.reference_kind is None:
                raise EvaluationError("--reference-kind is required with --reference")
            evaluation = evaluate_split(args.split_dir, args.reference, args.reference_kind)
    except EvaluationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(evaluation["aggregate"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
