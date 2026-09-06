import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from docinsights_analysis.solution_records import (
    SolutionRecordError,
    evaluate_records,
    export_records,
    finalize_solution,
    input_digest,
    primary_record_digest,
    validate_primary_solution,
    validate_solution,
)


def task(instance_id: str = "task_1") -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "user_query": "What is revenue?",
        "document_pdf": f"documents/{instance_id}.pdf",
    }


def pages() -> list[dict[str, object]]:
    return [{"page_number": 1, "text": "Revenue anchor 12.5 million"}]


def primary(instance_id: str = "task_1") -> dict[str, object]:
    public_pages = pages()
    return {
        "instance_id": instance_id,
        "split": "heldout",
        "question": "What is revenue?",
        "solution": {
            "summary": "The source reports revenue in millions.",
            "calculations": [{"expression": "10 + 2.5", "result": "12.5"}],
        },
        "answer": "12.5",
        "evidence_regions": [{"page": 1, "ocr_anchor": "anchor 12.5"}],
        "source_pages": public_pages,
        "provenance": {
            "pdf_sha256": "a" * 64,
            "input_sha256": input_digest(task(instance_id), public_pages),
            "config_sha256": "b" * 64,
            "output_sha256": "c" * 64,
            "model": "primary-model",
            "method": "primary-solver",
        },
        "uncertainties": [],
    }


def artifact(kind: str, digest: str = "d" * 64) -> dict[str, str]:
    return {"kind": kind, "path": f"private/{kind}", "sha256": digest}


def source_check(
    normalized_primary: dict[str, object], *, unresolved: bool = False
) -> dict[str, object]:
    region = normalized_primary["evidence_regions"][0]  # type: ignore[index]
    observation = {
        "heading_line": None if unresolved else "section:1: Revenue",
        "body_text": "Revenue was 12.5 million.",
        "heading_legibility": "ambiguous" if unresolved else "clear",
        "body_legibility": "clear",
        "contains_anchor_region": True,
    }
    evidence = {
        "id": None if unresolved else "section:1",
        "page": 1,
        "quote": "Revenue was 12.5 million.",
    }
    result: dict[str, object] = {
        "region_index": 0,
        "page": 1,
        "ocr_anchor": region["ocr_anchor"],  # type: ignore[index]
        "ocr_anchor_sha256": hashlib.sha256(
            str(region["ocr_anchor"]).encode()  # type: ignore[index]
        ).hexdigest(),
        "primary_record_sha256": primary_record_digest(normalized_primary),
        "primary_response_sha256": normalized_primary["provenance"]["output_sha256"],  # type: ignore[index]
        "status": "evidence_unresolved" if unresolved else "fully_grounded",
        "evidence": evidence,
        "observed_candidates": [observation],
        "source_uncertainties": ["Heading ID is ambiguous."] if unresolved else [],
        "model": "checker-model",
        "method": "blind-pixel-transcription",
        "anchor_bbox": {"left": 1, "top": 2, "right": 3, "bottom": 4},
        "context_bbox": {"left": 0, "top": 0, "right": 10, "bottom": 10},
        "anchor_crop_bbox": {"left": 1, "top": 2, "right": 3, "bottom": 4},
        "artifacts": [artifact("invocation")],
    }
    for field in (
        "pdf_sha256",
        "renderer_sha256",
        "checker_executable_sha256",
        "selector_sha256",
        "prompt_sha256",
        "output_schema_sha256",
        "checker_config_sha256",
        "page_image_sha256",
        "ocr_image_sha256",
        "context_crop_sha256",
        "anchor_crop_sha256",
        "raw_response_sha256",
    ):
        result[field] = (
            normalized_primary["provenance"]["pdf_sha256"]  # type: ignore[index]
            if field == "pdf_sha256"
            else "e" * 64
        )
    artifact_hashes = {
        "pdf_artifact": "pdf_sha256",
        "renderer_artifact": "renderer_sha256",
        "page_image_artifact": "page_image_sha256",
        "context_crop_artifact": "context_crop_sha256",
        "anchor_crop_artifact": "anchor_crop_sha256",
        "checker_executable_artifact": "checker_executable_sha256",
        "raw_response_artifact": "raw_response_sha256",
    }
    for field, hash_field in artifact_hashes.items():
        result[field] = artifact(field, str(result[hash_field]))
    return result


def finalized(*, unresolved: bool = False) -> tuple[dict[str, object], list[dict[str, object]]]:
    normalized = validate_primary_solution(primary(), task(), pages())
    checks = [source_check(normalized, unresolved=unresolved)]
    return finalize_solution(normalized, task(), pages(), source_checks=checks), checks


def test_primary_requires_unique_exact_ocr_anchor_and_output_hash() -> None:
    normalized = validate_primary_solution(primary(), task(), pages())
    assert normalized["evidence_regions"] == [{"page": 1, "ocr_anchor": "anchor 12.5"}]

    duplicate = primary()
    duplicate_pages = [{"page_number": 1, "text": "anchor 12.5 then anchor 12.5"}]
    duplicate["source_pages"] = duplicate_pages
    duplicate["provenance"]["input_sha256"] = input_digest(task(), duplicate_pages)  # type: ignore[index]
    with pytest.raises(SolutionRecordError, match="exactly once"):
        validate_primary_solution(duplicate, task(), duplicate_pages)


def test_primary_direct_read_requires_unitless_numeric_answer() -> None:
    direct = primary()
    direct["solution"]["calculations"] = []  # type: ignore[index]
    direct["answer"] = "12.5"
    assert validate_primary_solution(direct, task(), pages())["answer"] == "12.5"

    direct["answer"] = "12.5 million"
    direct["uncertainties"] = ["The source displays the value in millions."]
    with pytest.raises(SolutionRecordError, match="answer must be a numeric string"):
        validate_primary_solution(direct, task(), pages())


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("round(1 / 3 * 100, 2)", "33.33"),
        ("round(2.345, 2)", "2.35"),
        ("round(-2.345, 2)", "-2.35"),
    ],
)
def test_primary_supports_explicit_half_up_rounding(expression: str, result: str) -> None:
    rounded = primary()
    rounded["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": expression, "result": result}
    ]
    rounded["answer"] = result

    normalized = validate_primary_solution(rounded, task(), pages())

    assert normalized["solution"]["calculations"] == [  # type: ignore[index]
        {"expression": expression, "result": result}
    ]
    assert normalized["answer"] == result


def test_explicit_rounding_requires_exact_declared_result_and_answer() -> None:
    rounded = primary()
    rounded["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": "round(1 / 3 * 100, 2)", "result": "33.34"}
    ]
    rounded["answer"] = "33.34"

    with pytest.raises(SolutionRecordError, match="calculation result mismatch"):
        validate_primary_solution(rounded, task(), pages())

    rounded["solution"]["calculations"][0]["result"] = "33.33"  # type: ignore[index]
    with pytest.raises(SolutionRecordError, match="answer does not match"):
        validate_primary_solution(rounded, task(), pages())


@pytest.mark.parametrize(
    "expression",
    [
        "round(1 / 3, 13)",
        "round(1 / 3, 1.5)",
        "round(1 / 0, 2)",
        "round(round(1 / 3, 3), 2)",
        "abs(1)",
        "__import__('os').system('true')",
    ],
)
def test_explicit_rounding_rejects_invalid_precision_and_hostile_calls(
    expression: str,
) -> None:
    rounded = primary()
    rounded["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": expression, "result": "0"}
    ]
    rounded["answer"] = "0"

    with pytest.raises(SolutionRecordError, match="arithmetic expression"):
        validate_primary_solution(rounded, task(), pages())


def test_unrounded_arithmetic_remains_strict() -> None:
    unrounded = primary()
    unrounded["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": "1 / 3", "result": "0.3333333333333333333333333333"}
    ]
    unrounded["answer"] = "0.3333333333333333333333333333"
    with pytest.raises(SolutionRecordError, match="unsupported precision"):
        validate_primary_solution(unrounded, task(), pages())

    for expression in ("5 // 2", "5 % 2"):
        unrounded["solution"]["calculations"] = [  # type: ignore[index]
            {"expression": expression, "result": "2"}
        ]
        unrounded["answer"] = "2"
        with pytest.raises(SolutionRecordError, match="invalid arithmetic expression"):
            validate_primary_solution(unrounded, task(), pages())


def test_grounded_source_check_derives_final_id_and_quote_from_pixels() -> None:
    record, checks = finalized()
    assert record["source_status"] == "fully_grounded"
    assert record["evidence"] == ["section:1"]
    assert record["evidence_details"] == [
        {"id": "section:1", "page": 1, "quote": "Revenue was 12.5 million."}
    ]
    assert validate_solution(record, task(), pages(), source_checks=checks) == record


def test_source_checks_are_independent_and_bind_immutable_primary() -> None:
    record, checks = finalized()
    with pytest.raises(SolutionRecordError, match="independently supplied"):
        validate_solution(record, task(), pages())

    changed = deepcopy(record)
    changed["solution"]["summary"] = "A changed post-check explanation."  # type: ignore[index]
    with pytest.raises(SolutionRecordError, match="primary region"):
        validate_solution(changed, task(), pages(), source_checks=checks)

    fabricated = deepcopy(record)
    fabricated["provenance"]["source_checks"] = []  # type: ignore[index]
    with pytest.raises(SolutionRecordError, match="independently supplied"):
        validate_solution(fabricated, task(), pages(), source_checks=checks)


def test_unresolved_source_keeps_readable_quote_null_id_and_separate_uncertainty() -> None:
    record, checks = finalized(unresolved=True)
    assert record["source_status"] == "evidence_unresolved"
    assert record["evidence"] == []
    assert record["evidence_details"][0]["quote"] == "Revenue was 12.5 million."  # type: ignore[index]
    assert record["source_uncertainties"] == [
        {"region_index": 0, "message": "Heading ID is ambiguous."}
    ]
    assert record["answer"] == "12.5"
    assert validate_solution(record, task(), pages(), source_checks=checks) == record


def test_runtime_failure_is_not_a_terminal_source_check() -> None:
    normalized = validate_primary_solution(primary(), task(), pages())
    failed = source_check(normalized)
    failed["status"] = "runtime_failed"
    with pytest.raises(SolutionRecordError, match="terminal"):
        finalize_solution(normalized, task(), pages(), source_checks=[failed])


def test_export_keeps_all_solutions_but_submits_only_grounded(tmp_path: Path) -> None:
    grounded, grounded_checks = finalized()
    unresolved_primary = primary("task_2")
    unresolved_primary["question"] = task("task_2")["user_query"]
    normalized = validate_primary_solution(unresolved_primary, task("task_2"), pages())
    unresolved_checks = [source_check(normalized, unresolved=True)]
    unresolved = finalize_solution(
        normalized, task("task_2"), pages(), source_checks=unresolved_checks
    )

    manifest = export_records(
        [unresolved, grounded],
        [task(), task("task_2")],
        tmp_path / "private",
        source_pages_by_id={"task_1": pages(), "task_2": pages()},
        expected_split="heldout",
        source_checks_by_id={"task_1": grounded_checks, "task_2": unresolved_checks},
    )
    assert manifest["coverage_complete"] is True
    assert manifest["answer_coverage"] == 2
    assert manifest["fully_grounded_count"] == 1
    assert manifest["evidence_unresolved_count"] == 1
    assert manifest["runtime_failed_count"] == 0
    assert manifest["submission_ready"] is False
    assert manifest["submission_mode"] == "blocked_unresolved"
    assert manifest["submission_count"] == 0
    assert manifest["grounded_submission_count"] == 1
    assert manifest["submission_abstention_count"] == 0
    assert manifest["submission_exclusions"] == [
        {"instance_id": "task_2", "reason": "evidence_unresolved"}
    ]
    solution_rows = (tmp_path / "private" / "solutions.jsonl").read_text().splitlines()
    submission_rows = (
        tmp_path / "private" / "grounded-submission.jsonl"
    ).read_text().splitlines()
    assert len(solution_rows) == 2
    assert [json.loads(row) for row in submission_rows] == [
        {"instance_id": "task_1", "answer": "12.5", "evidence": ["section:1"]}
    ]
    assert not (tmp_path / "private" / "submission.jsonl").exists()
    markdown = (tmp_path / "private" / "solutions.md").read_text()
    assert "evidence_unresolved" in markdown
    assert "Heading ID is ambiguous." in markdown


def test_heldout_abstention_projection_requires_explicit_mode(tmp_path: Path) -> None:
    record, checks = finalized(unresolved=True)
    manifest = export_records(
        [record],
        [task()],
        tmp_path / "private",
        source_pages_by_id={"task_1": pages()},
        expected_split="heldout",
        source_checks_by_id={"task_1": checks},
        submission_mode="abstain_unresolved",
    )
    assert manifest["submission_ready"] is True
    assert manifest["submission_mode"] == "heldout_abstentions"
    assert manifest["submission_abstention_count"] == 1
    assert json.loads((tmp_path / "private" / "submission.jsonl").read_text()) == {
        "instance_id": "task_1",
        "answer": None,
        "evidence": [],
    }

    frozen_submission = (tmp_path / "private" / "submission.jsonl").read_bytes()
    frozen_manifest = (tmp_path / "private" / "manifest.json").read_bytes()
    with pytest.raises(SolutionRecordError, match="stale submission"):
        export_records(
            [record],
            [task()],
            tmp_path / "private",
            source_pages_by_id={"task_1": pages()},
            expected_split="heldout",
            source_checks_by_id={"task_1": checks},
        )
    assert (tmp_path / "private" / "submission.jsonl").read_bytes() == frozen_submission
    assert (tmp_path / "private" / "manifest.json").read_bytes() == frozen_manifest


def test_evaluation_compares_all_answers_and_marks_unresolved_evidence_unassessable() -> None:
    grounded, _ = finalized()
    unresolved, _ = finalized(unresolved=True)
    unresolved["instance_id"] = "task_2"
    result = evaluate_records(
        [grounded, unresolved],
        [
            {"instance_id": "task_1", "answer": "12.50", "evidence": ["section:1"]},
            {"instance_id": "task_2", "answer": "12.5", "evidence": ["X"]},
        ],
        "reference",
    )
    assert result["comparisons"][1] == {
        "instance_id": "task_2",
        "answer_match": True,
        "evidence_assessable": False,
        "evidence_match": None,
    }
    assert result["aggregate"] == {
        "reference_kind": "reference",
        "total": 2,
        "answer_matches": 2,
        "evidence_assessable": 1,
        "evidence_matches": 1,
        "evidence_unresolved": 1,
    }
