import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from docinsights_ambiguity.audit import (
    DISCLAIMER,
    _compare_benchmark,
    _expand_review_shard,
    _mark_label_carryover,
    _semantic_answer,
    _validate_record,
    _validate_review_decision,
    build_audit,
    build_blind_screen,
    screen_question,
    validate_artifacts,
    validate_schema_contract,
    write_audit,
    write_blind_screen,
)
from docinsights_ambiguity.cli import main

_HASH = "a" * 64


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_evidence(question_sha256: str = _HASH) -> dict[str, object]:
    return {
        "benchmark_block_ids": ["b01"],
        "recovered_block_id": "b01",
        "recovered_pages": [1],
        "recovered_query_sha256": question_sha256,
        "recovered_block_text_sha256": _HASH,
        "document_pdf_sha256": _HASH,
        "codex_reference_file_sha256": _HASH,
        "tasks_manifest_file_sha256": _HASH,
        "benchmark_blocks_present_in_reference": True,
        "benchmark_matches_recovered_block": True,
    }


def _comparison(question: str, *, category: str = "actual_content_difference") -> dict[str, str]:
    return {"instance_id": "task_000001", "recovered_query": question, "category": category}


@pytest.mark.parametrize(
    ("question", "signal"),
    [
        ("How many more than 4 are there?", "comparison_language"),
        (
            "Alice and Bob received 2 and 3 items, respectively. What did she receive?",
            "role_language",
        ),
        ("How many remain after 3 are removed, unless 2 return?", "condition_language"),
        ("There are 4 boxes with 3 bags each and 2 spare bags. How many?", "cardinality_cue"),
        ("After the first 2, how many are remaining for the last trip?", "quantifier_language"),
        ("What was the third value after the first value?", "ordinal_language"),
        ("At a rate per hour, it covers 3 miles every 4 minutes. How long?", "multiple_rate_cues"),
        ("How many remain? Then question two: how much costs?", "multiple_question_marks"),
        ("Although 9 is unrelated, how many are in 3 groups?", "distractor_language"),
        ("What is the difference between negative 3 and positive 2?", "sign_sensitive_language"),
        ("Assuming a constant rate, how long does it take?", "assumption_language"),
    ],
)
def test_weak_lexical_cues_are_signals_not_confirmed_tags(question: str, signal: str) -> None:
    result = screen_question("task_000001", _comparison(question))
    assert signal in {item["code"] for item in result["auto_signals"]}


def test_confirmed_structural_failures_get_tags_and_exact_taxonomy() -> None:
    rate = screen_question(
        "task_000003",
        _comparison(
            "A fog moves at 42 city blocks per hour. "
            "Each 10-minute interval covers 3 blocks. How long?"
        ),
    )
    target = screen_question(
        "task_000027",
        _comparison(
            "Ava rolls an eight-sided die, but since it has fewer sides, "
            "use a twelve-sided die instead for comparison."
        ),
    )
    assert rate["surface_integrity"] == "awkward_but_parseable"
    assert rate["semantic_determinacy"] == "multiple_plausible"
    assert "rate_unit_relation_corruption" in rate["issue_tags"]
    assert "target_or_subquestion_drift" in target["issue_tags"]


def test_multiple_question_marks_alone_are_not_confirmed_target_drift() -> None:
    result = screen_question(
        "task_000029",
        _comparison(
            "What is event A? What is event B? "
            "What is the difference between these two probabilities?"
        ),
    )
    assert "target_or_subquestion_drift" not in result["issue_tags"]
    assert "multiple_question_marks" in {signal["code"] for signal in result["auto_signals"]}


def test_ocr_uncertainty_and_blind_layer_do_not_accept_a_label() -> None:
    result = screen_question("task_000001", _comparison("How many are left?", category="ocr"))
    assert any(signal["code"] == "ocr_normalization_used" for signal in result["auto_signals"])
    assert "extraction_or_ocr_uncertain" not in result["issue_tags"]
    assert "benchmark_answer" not in result
    assert list(inspect.signature(screen_question).parameters) == ["instance_id", "comparison"]


@pytest.mark.parametrize(
    ("question", "answer", "method"),
    [
        (
            "A 10-foot whale has 8 fish, each measuring 3 inches. What percentage is that?",
            "20",
            "percent_of_length",
        ),
        (
            "A mixture uses a ratio of 7 parts coffee to 13 parts water "
            "and 120 teaspoons in total.",
            "42",
            "ratio_share",
        ),
        (
            "It takes 176 minutes to cover every 35 miles and the town is 70 miles wide.",
            "352",
            "constant_rate_coverage",
        ),
    ],
)
def test_semantic_answer_rules_are_label_independent(
    question: str, answer: str, method: str
) -> None:
    result = _semantic_answer(question)
    assert result.value == answer
    assert result.method == method


def test_label_carryover_is_only_suspected_after_a_computed_mismatch() -> None:
    def record(instance_id: str, feet: int) -> dict[str, object]:
        question = (
            f"A {feet}-foot whale has 8 fish, each measuring 3 inches. What percentage is that?"
        )
        blind = screen_question(instance_id, _comparison(question))
        return {
            "instance_id": instance_id,
            "benchmark_answer": "999",
            "axes": {"benchmark_alignment": "semantic_conflict"},
            "blind_question_screen": blind,
            "issue_tags": [],
            "auto_signals": [],
            "severity": "S4",
            "benchmark_risk": "high",
            "review_required": True,
        }

    records = [record("task_000001", 10), record("task_000002", 20)]
    _mark_label_carryover(records)
    assert "label_template_carryover_suspected" not in records[0]["issue_tags"]
    assert "label_template_carryover_suspected" in records[1]["issue_tags"]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_end_to_end_coverage_schema_and_disclaimer(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    labels = tmp_path / "labels.jsonl"
    reference = tmp_path / "reference.jsonl"
    comparison = tmp_path / "comparison.jsonl"
    output = tmp_path / "tags.jsonl"
    summary_path = tmp_path / "summary.json"
    ids = ["task_000001", "task_000002"]
    _write_jsonl(tasks, [{"instance_id": key, "user_query": "generic"} for key in ids])
    _write_jsonl(
        labels,
        [{"instance_id": key, "answer": "20", "evidence": ["b01"]} for key in ids],
    )
    _write_jsonl(
        reference,
        [
            {
                "instance_id": key,
                "blocks": [{"block_id": "b01", "text": "Evidence text."}],
                "provenance": {"input_pdf_sha256": _HASH},
            }
            for key in ids
        ],
    )
    tasks_sha256 = _file_sha256(tasks)
    reference_sha256 = _file_sha256(reference)
    _write_jsonl(
        comparison,
        [
            {
                "instance_id": key,
                "category": "actual_content_difference",
                "recovered_query": (
                    "A 10-foot whale has 8 fish, each measuring 3 inches. What percentage is that?"
                ),
                "evidence_block_id": "b01",
                "evidence_pages": [1],
                "pdf_sha256": _HASH,
                "source": {
                    "document_pdf_sha256": _HASH,
                    "codex_reference_sha256": reference_sha256,
                    "tasks_manifest_sha256": tasks_sha256,
                },
            }
            for key in ids
        ],
    )

    blind_path = tmp_path / "blind.jsonl"
    blind = build_blind_screen(tasks, comparison)
    write_blind_screen(blind, blind_path)
    result = build_audit(tasks, labels, reference, comparison, blind_path, expected_count=2)
    write_audit(result, output, summary_path)
    validation = validate_artifacts(tasks, output, summary_path, expected_count=2)

    assert validation["passed"] is True
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first["screening_statement"] == DISCLAIMER
    assert first["semantic_answer"]["value"] == "20"
    assert first["benchmark_answer"] == "20"
    assert first["axes"]["benchmark_alignment"] == "aligned"
    assert "benchmark_answer" not in first["blind_question_screen"]
    assert first["evidence"]["codex_reference_file_sha256"] == reference_sha256
    assert first["evidence"]["tasks_manifest_file_sha256"] == tasks_sha256

    changed = output.read_text(encoding="utf-8").splitlines()
    changed_first = json.loads(changed[0])
    changed_first["auto_signals"].append({"code": "tampered", "detail": "stale summary"})
    changed[0] = json.dumps(changed_first, sort_keys=True)
    output.write_text("\n".join(changed) + "\n", encoding="utf-8")
    stale_validation = validate_artifacts(tasks, output, summary_path, expected_count=2)
    assert stale_validation["passed"] is False
    assert "summary output SHA-256 mismatch" in stale_validation["errors"]
    assert (
        main(
            [
                "validate",
                "--tasks",
                str(tasks),
                "--output",
                str(output),
                "--summary",
                str(summary_path),
                "--expected-count",
                "2",
            ]
        )
        == 1
    )

    truncated = dict(result)
    truncated["records"] = result["records"][:1]
    truncated_summary = write_audit(
        truncated, tmp_path / "truncated.jsonl", tmp_path / "truncated-summary.json"
    )
    assert truncated_summary["validation"]["passed"] is False
    assert "record count 1 != 2" in truncated_summary["validation"]["errors"]


def test_review_required_is_decision_specific_for_s2() -> None:
    review = {
        "reviewer": "agent-test",
        "review_kind": "agent_text_audit_no_pdf_visual_review",
        "surface_integrity": "awkward_but_parseable",
        "semantic_determinacy": "unique_with_convention",
        "benchmark_alignment": "label_unverifiable",
        "issue_tags": ["implicit_default_assumption"],
        "severity": "S2",
        "review_required": False,
        "rationale": "The convention is explicit enough to recover without adjudication.",
        "confidence": 0.95,
    }
    record = {
        "schema_version": "1.0",
        "audit_kind": "docsem-train-ambiguity-screen",
        "screening_statement": DISCLAIMER,
        "adjudication_status": "automated_screened_agent_text_audited_human_adjudication_pending",
        "screening_classification": "flagged",
        "instance_id": "task_000001",
        "split": "train",
        "blind_question_screen": {"question_sha256": _HASH},
        "axes": {
            "surface_integrity": "awkward_but_parseable",
            "semantic_determinacy": "unique_with_convention",
            "benchmark_alignment": "label_unverifiable",
        },
        "semantic_answer": {"value": None, "status": "not_computed", "method": None},
        "benchmark_answer": "1",
        "issue_tags": ["implicit_default_assumption"],
        "severity": "S2",
        "benchmark_risk": "medium",
        "auto_signals": [],
        "evidence": _valid_evidence(),
        "agent_reviews": [review],
        "review_required": False,
    }
    assert _validate_record(record) == []
    record["severity"] = "S3"
    review["severity"] = "S3"
    assert "S3-S4 records must require adjudication" in _validate_record(record)


def test_review_shard_rejects_range_count_mismatch() -> None:
    shard = {
        "reviewer": "agent-test",
        "first_instance_id": "task_000001",
        "last_instance_id": "task_000002",
        "expected_record_count": 3,
        "pdf_visual_review_performed": False,
        "default_decision": {
            "surface_integrity": "intact",
            "semantic_determinacy": "unique_explicit",
            "benchmark_alignment": "label_unverifiable",
            "issue_tags": ["clean"],
            "severity": "S0",
            "review_required": False,
            "rationale": "No explicit ambiguity was found.",
            "confidence": 0.9,
        },
        "override_groups": [],
        "expected_override_count": 0,
    }
    with pytest.raises(ValueError, match="expected_record_count"):
        _expand_review_shard(shard)


def test_audit_cli_returns_nonzero_when_generated_validation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "docinsights_ambiguity.cli.build_blind_screen", lambda *_args: {"records": []}
    )
    monkeypatch.setattr("docinsights_ambiguity.cli.write_blind_screen", lambda *_args: {})
    monkeypatch.setattr(
        "docinsights_ambiguity.cli.build_audit", lambda *_args, **_kwargs: {"records": []}
    )
    monkeypatch.setattr(
        "docinsights_ambiguity.cli.write_audit",
        lambda *_args: {"validation": {"passed": False}},
    )
    paths = [str(tmp_path / name) for name in ("tasks", "labels", "reference", "comparison")]
    code = main(
        [
            "audit",
            "--tasks",
            paths[0],
            "--labels",
            paths[1],
            "--reference",
            paths[2],
            "--query-comparison",
            paths[3],
            "--blind-output",
            str(tmp_path / "blind.jsonl"),
            "--output",
            str(tmp_path / "tags.jsonl"),
            "--summary",
            str(tmp_path / "summary.json"),
        ]
    )
    assert code == 1


def test_agent_semantic_answer_is_promoted_and_rechecks_alignment() -> None:
    blind = screen_question("task_000001", _comparison("How much remains?"))
    review = {
        "surface_integrity": "intact",
        "semantic_determinacy": "unique_explicit",
        "benchmark_alignment": "aligned",
        "issue_tags": ["implicit_default_assumption"],
        "severity": "S1",
        "review_required": False,
        "rationale": "Direct arithmetic yields one.",
        "confidence": 0.99,
        "semantic_answer": {"value": "1", "method": "agent_text_arithmetic"},
    }
    result = _compare_benchmark(
        {"instance_id": "task_000001"},
        {"instance_id": "task_000001", "answer": "1", "evidence": ["b01"]},
        {
            "instance_id": "task_000001",
            "blocks": [{"block_id": "b01", "text": "Evidence text."}],
            "provenance": {"input_pdf_sha256": _HASH},
        },
        {
            "instance_id": "task_000001",
            "evidence_block_id": "b01",
            "evidence_pages": [1],
            "pdf_sha256": _HASH,
            "source": {
                "document_pdf_sha256": _HASH,
                "codex_reference_sha256": _HASH,
                "tasks_manifest_sha256": _HASH,
            },
        },
        blind,
        review,
    )
    assert result["semantic_answer"] == {
        "value": "1",
        "status": "computed",
        "method": "agent_text_arithmetic",
    }
    assert result["axes"]["benchmark_alignment"] == "aligned"
    assert result["blind_question_screen"]["semantic_answer"]["value"] is None


def test_agent_axes_do_not_mutate_the_persisted_blind_decision() -> None:
    blind = screen_question("task_000001", _comparison("How much remains?"))
    persisted = json.loads(json.dumps(blind))
    review = {
        "surface_integrity": "corrupted",
        "semantic_determinacy": "underdetermined",
        "benchmark_alignment": "label_unverifiable",
        "issue_tags": ["cardinality_operand_gap"],
        "severity": "S4",
        "review_required": True,
        "rationale": "A required operand is absent.",
        "confidence": 0.99,
    }
    result = _compare_benchmark(
        {"instance_id": "task_000001"},
        {"instance_id": "task_000001", "answer": "1", "evidence": ["b01"]},
        {
            "instance_id": "task_000001",
            "blocks": [{"block_id": "b01", "text": "Evidence text."}],
            "provenance": {"input_pdf_sha256": _HASH},
        },
        {
            "instance_id": "task_000001",
            "evidence_block_id": "b01",
            "evidence_pages": [1],
            "pdf_sha256": _HASH,
            "source": {
                "document_pdf_sha256": _HASH,
                "codex_reference_sha256": _HASH,
                "tasks_manifest_sha256": _HASH,
            },
        },
        blind,
        review,
    )
    assert result["blind_question_screen"] == persisted
    assert result["axes"]["surface_integrity"] == "corrupted"
    assert result["axes"]["semantic_determinacy"] == "underdetermined"


def test_proof_bearing_review_alignment_requires_semantic_answer() -> None:
    decision = {
        "surface_integrity": "intact",
        "semantic_determinacy": "unique_explicit",
        "benchmark_alignment": "aligned",
        "issue_tags": ["implicit_default_assumption"],
        "severity": "S1",
        "review_required": False,
        "rationale": "The label appears to match.",
        "confidence": 0.9,
    }
    with pytest.raises(ValueError, match="requires semantic_answer"):
        _validate_review_decision(decision, allow_clean=False)


def test_schema_contract_rejects_empty_schema() -> None:
    assert validate_schema_contract({})
    schema = json.loads(
        Path("schemas/docsem-train-ambiguity-v1.schema.json").read_text(encoding="utf-8")
    )
    assert validate_schema_contract(schema) == []
    del schema["properties"]["evidence"]["properties"]["document_pdf_sha256"]
    errors = validate_schema_contract(schema)
    assert "schema evidence properties do not match the provenance contract" in errors


def test_mission_validator_rejects_duplicate_query_comparison(tmp_path: Path) -> None:
    source = Path("artifacts/ambiguity/inputs/codex-train-query-comparison.jsonl")
    duplicate = tmp_path / "query-comparison-duplicate.jsonl"
    shutil.copyfile(source, duplicate)
    first_line = source.open(encoding="utf-8").readline()
    with duplicate.open("a", encoding="utf-8") as handle:
        handle.write(first_line)
    command = [
        sys.executable,
        ".omx/specs/autoresearch-docsem-ambiguity/validate_mission.py",
        "--tasks",
        "artifacts/ambiguity/inputs/train-tasks.jsonl",
        "--labels",
        "artifacts/ambiguity/inputs/train-labels.jsonl",
        "--reference",
        "artifacts/ambiguity/inputs/codex-train-reference.jsonl",
        "--query-comparison",
        str(duplicate),
        "--blind",
        "artifacts/ambiguity/train-ambiguity-blind.jsonl",
        "--output",
        "artifacts/ambiguity/train-ambiguity-tags.jsonl",
        "--summary",
        "artifacts/ambiguity/train-ambiguity-summary.json",
        "--schema",
        "schemas/docsem-train-ambiguity-v1.schema.json",
        "--state",
        ".omx/state/autoresearch-docsem-ambiguity/autoresearch-state.json",
        "--result",
        str(tmp_path / "result.json"),
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert any("query comparison duplicate instance_ids" in error for error in result["errors"])
