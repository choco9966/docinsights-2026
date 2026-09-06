import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_solution_records.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_solution_records", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
EvaluationError = _MODULE.EvaluationError
evaluate_split = _MODULE.evaluate_split
record_unavailable_evaluation = _MODULE.record_unavailable_evaluation
verify_evaluation_binding = _MODULE.verify_evaluation_binding


@pytest.fixture(autouse=True)
def _isolate_reference_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_PINNED_REFERENCE_METADATA",
        {key: dict(value) for key, value in _MODULE._PINNED_REFERENCE_METADATA.items()},
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _frozen_split(tmp_path: Path, split: str = "train") -> tuple[Path, Path]:
    split_dir = tmp_path / split
    split_dir.mkdir(parents=True)
    source_artifact = split_dir / "jobs" / "source-proof.json"
    source_artifact.parent.mkdir()
    source_artifact.write_text("frozen source proof\n", encoding="utf-8")
    source_checks = [
        {
            "artifacts": [
                {
                    "kind": "source-proof",
                    "path": str(source_artifact),
                    "sha256": _sha(source_artifact),
                }
            ]
        }
    ]
    records = [
        {
            "instance_id": "a",
            "split": split,
            "answer": "12.5",
            "evidence": ["E:1", "B"],
            "source_status": "fully_grounded",
            "provenance": {"source_checks": source_checks},
        },
        {
            "instance_id": "b",
            "split": split,
            "answer": "text",
            "evidence": [None],
            "source_status": "evidence_unresolved",
            "provenance": {"source_checks": source_checks},
        },
    ]
    submissions = [{"instance_id": "a", "answer": "12.5", "evidence": ["E:1", "B"]}]
    _write_jsonl(split_dir / "grounded-submission.jsonl", submissions)
    if split == "heldout":
        submissions.append({"instance_id": "b", "answer": None, "evidence": []})
    _write_jsonl(split_dir / "solutions.jsonl", records)
    (split_dir / "solutions.md").write_text("# frozen solutions\n", encoding="utf-8")
    if split == "heldout":
        _write_jsonl(split_dir / "submission.jsonl", submissions)
    generation_files = (
        "solutions.jsonl",
        "solutions.md",
        "grounded-submission.jsonl",
        *(("submission.jsonl",) if split == "heldout" else ()),
    )
    _write_json(
        split_dir / "manifest.json",
        {
            "schema_version": 2,
            "private": True,
            "split": split,
            "total": 2,
            "instance_ids": ["a", "b"],
            "coverage_complete": True,
            "answer_coverage": 2,
            "fully_grounded_count": 1,
            "evidence_unresolved_count": 1,
            "runtime_failed_count": 0,
            "submission_ready": split == "heldout",
            "submission_mode": (
                "heldout_abstentions" if split == "heldout" else "blocked_unresolved"
            ),
            "submission_count": len(submissions) if split == "heldout" else 0,
            "grounded_submission_count": 1,
            "submission_abstention_count": 1 if split == "heldout" else 0,
            "submission_exclusions": (
                []
                if split == "heldout"
                else [{"instance_id": "b", "reason": "evidence_unresolved"}]
            ),
            "files": {name: {"sha256": _sha(split_dir / name)} for name in generation_files},
        },
    )
    tasks = [{"task": {"instance_id": value}} for value in ("a", "b")]
    config = {"model": "fixture"}
    _write_json(
        split_dir / "run-manifest.json",
        {
            "split": split,
            "official_task_count": 2,
            "tasks": tasks,
            "input_manifest_sha256": hashlib.sha256(_canonical(tasks)).hexdigest(),
            "command_config": config,
            "command_config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        },
    )
    _write_json(
        split_dir / "complete.json",
        {
            "status": "generation_complete",
            "coverage_complete": True,
            "answer_coverage": 2,
            "split": split,
            "total": 2,
            "input_manifest_sha256": hashlib.sha256(_canonical(tasks)).hexdigest(),
            "command_config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
            "record_ids_sha256": hashlib.sha256(_canonical(["a", "b"])).hexdigest(),
            "export_manifest_sha256": _sha(split_dir / "manifest.json"),
            "fully_grounded_count": 1,
            "evidence_unresolved_count": 1,
            "runtime_failed_count": 0,
            "submission_ready": split == "heldout",
            "submission_mode": (
                "heldout_abstentions" if split == "heldout" else "blocked_unresolved"
            ),
            "submission_count": len(submissions) if split == "heldout" else 0,
            "grounded_submission_count": 1,
            "submission_abstention_count": 1 if split == "heldout" else 0,
        },
    )
    reference = tmp_path / f"{split}-reference.jsonl"
    _write_jsonl(
        reference,
        [
            {"instance_id": "a", "answer": "12.50", "evidence": ["B", "E:1"]},
            {"instance_id": "b", "answer": "different", "evidence": ["C"]},
        ],
    )
    reference_bytes = reference.read_bytes()
    if split == "train":
        _MODULE._PINNED_REFERENCE_METADATA[split] = {
            "size": len(reference_bytes),
            "git_blob_sha1": _MODULE._git_blob_sha1(reference_bytes),
        }
    elif split == "validation":
        _MODULE._PINNED_REFERENCE_METADATA[split] = {"sha256": _sha(reference)}
    return split_dir, reference


def test_evaluate_split_freezes_before_reading_reference(tmp_path: Path) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    (split_dir / "solutions.jsonl").write_text("tampered\n", encoding="utf-8")
    reference.unlink()

    with pytest.raises(EvaluationError, match="frozen generation.*hash"):
        evaluate_split(split_dir, reference, "train-public-labels")

    assert not (split_dir / "evaluation.json").exists()


def test_evaluate_split_verifies_source_artifacts_before_reference(tmp_path: Path) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    (split_dir / "jobs" / "source-proof.json").write_text("tampered\n", encoding="utf-8")
    reference.unlink()

    with pytest.raises(EvaluationError, match="source artifact hash mismatch"):
        evaluate_split(split_dir, reference, "train-public-labels")

    assert not (split_dir / "evaluation.json").exists()


@pytest.mark.parametrize("mutation", ["status", "export_hash"])
def test_evaluate_split_requires_runner_completion_binding_before_reference(
    tmp_path: Path, mutation: str
) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    completion_path = split_dir / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if mutation == "status":
        completion["status"] = "partial"
    else:
        completion["export_manifest_sha256"] = "0" * 64
    _write_json(completion_path, completion)
    reference.unlink()

    with pytest.raises(EvaluationError, match="completion marker"):
        evaluate_split(split_dir, reference, "train-public-labels")


def test_evaluate_split_rejects_generation_and_reference_coverage_gaps(
    tmp_path: Path,
) -> None:
    split_dir, reference = _frozen_split(tmp_path / "generation")
    rows = [json.loads(line) for line in (split_dir / "solutions.jsonl").read_text().splitlines()]
    rows[1]["instance_id"] = "a"
    _write_jsonl(split_dir / "solutions.jsonl", rows)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    manifest["files"]["solutions.jsonl"]["sha256"] = _sha(split_dir / "solutions.jsonl")
    _write_json(split_dir / "manifest.json", manifest)

    with pytest.raises(EvaluationError, match="unique.*coverage"):
        evaluate_split(split_dir, reference, "train-public-labels")

    split_dir, reference = _frozen_split(tmp_path / "reference")
    _write_jsonl(reference, [{"instance_id": "a", "answer": "12.5", "evidence": ["B"]}])
    reference_bytes = reference.read_bytes()
    _MODULE._PINNED_REFERENCE_METADATA["train"] = {
        "size": len(reference_bytes),
        "git_blob_sha1": _MODULE._git_blob_sha1(reference_bytes),
    }
    with pytest.raises(EvaluationError, match="missing reference"):
        evaluate_split(split_dir, reference, "train-public-labels")


def test_evaluate_split_writes_private_bound_evaluation_without_mutating_generation(
    tmp_path: Path,
) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    generated_names = (
        "manifest.json",
        "run-manifest.json",
        "complete.json",
        "solutions.jsonl",
        "solutions.md",
        "grounded-submission.jsonl",
    )
    before = {name: (split_dir / name).read_bytes() for name in generated_names}

    evaluation = evaluate_split(split_dir, reference, "train-public-labels")

    assert evaluation["aggregate"] == {
        "reference_kind": "train-public-labels",
        "total": 2,
        "answer_matches": 1,
        "evidence_assessable": 1,
        "evidence_matches": 1,
        "evidence_unresolved": 1,
    }
    assert evaluation["generation"]["manifest_sha256"] == _sha(split_dir / "manifest.json")
    assert evaluation["reference"] == {
        "kind": "train-public-labels",
        "path": str(reference.resolve()),
        "sha256": _sha(reference),
        "size": len(reference.read_bytes()),
        "git_blob_sha1": _MODULE._git_blob_sha1(reference.read_bytes()),
        "total": 2,
    }
    assert {name: (split_dir / name).read_bytes() for name in generated_names} == before
    assert (split_dir / "evaluation.json").stat().st_mode & 0o077 == 0
    assert verify_evaluation_binding(split_dir, "train") == evaluation
    with pytest.raises(EvaluationError, match="already exists"):
        evaluate_split(split_dir, reference, "train-public-labels")


def test_evaluate_split_rejects_dangling_evaluation_symlink(tmp_path: Path) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    outside = tmp_path / "outside.json"
    (split_dir / "evaluation.json").symlink_to(outside)

    with pytest.raises(EvaluationError, match="symbolic link"):
        evaluate_split(split_dir, reference, "train-public-labels")

    assert not outside.exists()


def test_verify_evaluation_binding_rejects_reference_tamper(tmp_path: Path) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    evaluate_split(split_dir, reference, "train-public-labels")
    reference.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="reference SHA-256"):
        verify_evaluation_binding(split_dir, "train")


def test_evaluate_split_rejects_reference_that_does_not_match_split_pin(
    tmp_path: Path,
) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    reference.write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="pinned reference metadata"):
        evaluate_split(split_dir, reference, "train-public-labels")


def test_verify_evaluation_binding_recomputes_coherently_tampered_matches(
    tmp_path: Path,
) -> None:
    split_dir, reference = _frozen_split(tmp_path)
    evaluate_split(split_dir, reference, "train-public-labels")
    evaluation_path = split_dir / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["comparisons"][0]["answer_match"] = False
    evaluation["aggregate"]["answer_matches"] = 0
    _write_json(evaluation_path, evaluation)
    evaluation_path.chmod(0o600)

    with pytest.raises(EvaluationError, match="does not match full comparison"):
        verify_evaluation_binding(split_dir, "train")


def test_validation_uses_v24_reference_kind_and_rejects_generation_as_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir, reference = _frozen_split(tmp_path / "validation", "validation")
    monkeypatch.setitem(
        _MODULE._PINNED_REFERENCE_METADATA, "validation", {"sha256": _sha(reference)}
    )
    evaluation = evaluate_split(split_dir, reference, "validation-v24-reference")
    assert evaluation["reference"]["kind"] == "validation-v24-reference"

    other_dir, _ = _frozen_split(tmp_path / "self-reference")
    with pytest.raises(EvaluationError, match="outside the frozen split"):
        evaluate_split(
            other_dir,
            other_dir / "submission.jsonl",
            "train-public-labels",
        )


def test_heldout_has_no_reference_evaluation(tmp_path: Path) -> None:
    split_dir, reference = _frozen_split(tmp_path, "heldout")

    with pytest.raises(EvaluationError, match="heldout.*unavailable"):
        evaluate_split(split_dir, reference, "train-public-labels")
    assert not (split_dir / "evaluation.json").exists()


def test_heldout_writes_no_reference_evaluation_without_opening_labels(
    tmp_path: Path,
) -> None:
    split_dir, reference = _frozen_split(tmp_path, "heldout")
    reference.unlink()
    generated = {
        name: (split_dir / name).read_bytes()
        for name in (
            "manifest.json",
            "run-manifest.json",
            "complete.json",
            "solutions.jsonl",
            "solutions.md",
            "grounded-submission.jsonl",
            "submission.jsonl",
        )
    }

    evaluation = record_unavailable_evaluation(split_dir)

    assert evaluation["status"] == "reference_unavailable"
    assert evaluation["reference"] == {
        "available": False,
        "kind": "heldout-labels-unavailable",
        "path": None,
        "sha256": None,
        "total": 0,
    }
    assert evaluation["comparisons"] == []
    assert evaluation["aggregate"] == {
        "reference_kind": "heldout-labels-unavailable",
        "total": 2,
        "reference_coverage": 0,
        "answer_matches": None,
        "evidence_matches": None,
        "fully_grounded_count": 1,
        "evidence_unresolved_count": 1,
    }
    assert {
        name: (split_dir / name).read_bytes() for name in generated
    } == generated
    assert verify_evaluation_binding(split_dir, "heldout") == evaluation
    assert (split_dir / "evaluation.json").stat().st_mode & 0o077 == 0

    with pytest.raises(EvaluationError, match="only valid for heldout"):
        train_dir, _ = _frozen_split(tmp_path / "train")
        record_unavailable_evaluation(train_dir)
