import hashlib
import json
from pathlib import Path

import pytest

from docinsights_analysis.solution_records import (
    SolutionRecordError,
    evaluate_records,
    export_records,
    input_digest,
    validate_solution,
    visual_evidence_candidates,
)


def task(instance_id: str = "task_000001") -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "user_query": "What is total revenue?",
        "document_pdf": f"documents/{instance_id}.pdf",
    }


def pages() -> list[dict[str, object]]:
    return [
        {
            "page_number": 1,
            "text": "A-1!: Revenue was 12.5 million.\nContinued evidence text.\n\n"
            "table.total/2024: Costs were 2.5 million.",
        },
        {"page_number": 2, "text": "note#3: The report uses USD millions."},
    ]


def record(instance_id: str = "task_000001") -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "split": "heldout",
        "question": "What is total revenue?",
        "solution": {
            "summary": "The revenue block reports the requested total.",
            "calculations": [{"expression": "10 + 2.5", "result": "12.5"}],
        },
        "answer": "12.5",
        "evidence": ["A-1!", "note#3"],
        "evidence_details": [
            {"id": "A-1!", "page": 1, "quote": "Revenue was 12.5 million."},
            {"id": "note#3", "page": 2, "quote": "The report uses USD millions."},
        ],
        "source_pages": pages(),
        "provenance": {
            "pdf_sha256": "a" * 64,
            "input_sha256": input_digest(task(instance_id), pages()),
            "config_sha256": "b" * 64,
            "model": "fixture-model",
            "method": "unit-test",
        },
        "uncertainties": [],
    }


def replace_pages(raw: dict[str, object], replacement: list[dict[str, object]]) -> None:
    raw["source_pages"] = replacement
    provenance = raw["provenance"]
    assert isinstance(provenance, dict)
    provenance["input_sha256"] = input_digest(task(str(raw["instance_id"])), replacement)


def visual_alias_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    raw = record()
    alias_pages = [{"page_number": 1, "text": "RO5: The exact amount is 7."}]
    raw["answer"] = "7"
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": "3 + 4", "result": "7"}
    ]
    raw["evidence"] = ["R05"]
    raw["evidence_details"] = [
        {"id": "R05", "page": 1, "quote": "The exact amount is 7."}
    ]
    replace_pages(raw, alias_pages)
    return raw, alias_pages


def visual_proof(candidate: dict[str, object]) -> dict[str, object]:
    quote = str(candidate["quote"])
    return {
        "id": candidate["id"],
        "visible_id": candidate["id"],
        "ocr_id": candidate["ocr_id"],
        "page": candidate["page"],
        "heading_line_index": candidate["heading_line_index"],
        "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
        "page_image_sha256": "c" * 64,
        "crop_sha256": "d" * 64,
        "clear": True,
    }


def test_validate_solution_normalizes_arbitrary_multiple_evidence_ids() -> None:
    raw = record()
    raw["solution"]["summary"] = "  The revenue block reports the requested total.  "  # type: ignore[index]

    normalized = validate_solution(raw, task(), pages())

    assert normalized["question"] == "What is total revenue?"
    assert normalized["solution"]["summary"] == (
        "The revenue block reports the requested total."
    )
    assert normalized["evidence"] == ["A-1!", "note#3"]
    assert normalized["source_pages"] == pages()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(instance_id="task_999999"), "instance_id"),
        (lambda value: value.update(question="A different question"), "question"),
        (
            lambda value: value["evidence_details"][0].update(page=2),
            "evidence block",
        ),
        (
            lambda value: value["evidence_details"][0].update(
                quote="Costs were 2.5 million."
            ),
            "evidence block",
        ),
        (
            lambda value: value["solution"]["calculations"][0].update(result="13"),
            "calculation result",
        ),
        (
            lambda value: value["solution"]["calculations"][0].update(
                expression="__import__('os').getcwd()"
            ),
            "arithmetic expression",
        ),
        (lambda value: value.update(answer="99"), "final answer"),
    ],
)
def test_validate_solution_rejects_mismatches(mutate, message: str) -> None:
    raw = record()
    mutate(raw)

    with pytest.raises(SolutionRecordError, match=message):
        validate_solution(raw, task(), pages())


def test_validate_solution_requires_matching_unique_evidence_details() -> None:
    raw = record()
    raw["evidence"] = ["A-1!", "A-1!"]
    with pytest.raises(SolutionRecordError, match="duplicate evidence"):
        validate_solution(raw, task(), pages())

    raw = record()
    raw["evidence_details"] = raw["evidence_details"][:1]  # type: ignore[index]
    with pytest.raises(SolutionRecordError, match="evidence_details.*match evidence"):
        validate_solution(raw, task(), pages())


def test_validate_solution_rejects_fabricated_source_pages() -> None:
    raw = record()
    raw["source_pages"] = [{"page_number": 1, "text": "fabricated"}]

    with pytest.raises(SolutionRecordError, match="source_pages"):
        validate_solution(raw, task(), pages())


def test_validate_solution_supports_internal_colons_in_evidence_id() -> None:
    raw = record()
    colon_pages = [{"page_number": 1, "text": "section:1: Exact cited sentence."}]
    raw["evidence"] = ["section:1"]
    raw["evidence_details"] = [
        {"id": "section:1", "page": 1, "quote": "Exact cited sentence."}
    ]
    replace_pages(raw, colon_pages)

    normalized = validate_solution(raw, task(), colon_pages)

    assert normalized["evidence"] == ["section:1"]


def test_validate_solution_does_not_leak_quote_from_following_uncited_block() -> None:
    raw = record()
    block_pages = [
        {
            "page_number": 1,
            "text": "cited: Cited sentence.\nuncited: Secret sentence from another block.",
        }
    ]
    raw["evidence"] = ["cited"]
    raw["evidence_details"] = [
        {"id": "cited", "page": 1, "quote": "Secret sentence from another block."}
    ]
    replace_pages(raw, block_pages)

    with pytest.raises(SolutionRecordError, match="evidence block"):
        validate_solution(raw, task(), block_pages)


def test_visual_evidence_candidates_returns_unique_heading_only_alias() -> None:
    raw, alias_pages = visual_alias_fixture()

    assert visual_evidence_candidates(raw, alias_pages) == [
        {
            "id": "R05",
            "ocr_id": "RO5",
            "page": 1,
            "quote": "The exact amount is 7.",
            "heading_line_index": 0,
        }
    ]


def test_visual_evidence_candidates_rejects_ambiguous_or_existing_solver_id() -> None:
    raw, _ = visual_alias_fixture()
    ambiguous_pages = [
        {
            "page_number": 1,
            "text": "RO5: The exact amount is 7.\nX-2: The exact amount is 7.",
        }
    ]
    replace_pages(raw, ambiguous_pages)
    assert visual_evidence_candidates(raw, ambiguous_pages) == []

    existing_pages = [
        {"page_number": 1, "text": "RO5: The exact amount is 7."},
        {"page_number": 2, "text": "R05: A different block."},
    ]
    replace_pages(raw, existing_pages)
    assert visual_evidence_candidates(raw, existing_pages) == []


def test_validate_solution_requires_independent_exact_clear_visual_alias_proof() -> None:
    raw, alias_pages = visual_alias_fixture()
    candidate = visual_evidence_candidates(raw, alias_pages)[0]
    proof = visual_proof(candidate)
    raw["provenance"]["visual_id_checks"] = [proof]  # type: ignore[index]

    with pytest.raises(SolutionRecordError, match="independently supplied"):
        validate_solution(raw, task(), alias_pages)

    for field, value in (("visible_id", "R0S"), ("clear", False), ("page", True)):
        invalid = dict(proof)
        invalid[field] = value
        raw["provenance"]["visual_id_checks"] = [invalid]  # type: ignore[index]
        with pytest.raises(SolutionRecordError, match="visual ID check"):
            validate_solution(raw, task(), alias_pages, visual_id_checks=[invalid])

    invalid_hash = {**proof, "artifact": {"sha256": "not-a-hash"}}
    raw["provenance"]["visual_id_checks"] = [invalid_hash]  # type: ignore[index]
    with pytest.raises(SolutionRecordError, match="SHA-256"):
        validate_solution(raw, task(), alias_pages, visual_id_checks=[invalid_hash])


def test_validate_solution_accepts_visual_heading_alias_without_repairing_body() -> None:
    raw, alias_pages = visual_alias_fixture()
    candidate = visual_evidence_candidates(raw, alias_pages)[0]
    proof = visual_proof(candidate)
    raw["provenance"]["visual_id_checks"] = [proof]  # type: ignore[index]

    normalized = validate_solution(
        raw,
        task(),
        alias_pages,
        visual_id_checks=[proof],
    )

    assert normalized["evidence"] == ["R05"]
    assert normalized["source_pages"] == alias_pages
    assert normalized["evidence_details"][0]["quote"] == "The exact amount is 7."
    assert normalized["provenance"]["visual_id_checks"] == [proof]

    raw["evidence_details"][0]["quote"] = "The exact amount is seven."  # type: ignore[index]
    bad_proof = visual_proof(
        {
            **candidate,
            "quote": "The exact amount is seven.",
        }
    )
    raw["provenance"]["visual_id_checks"] = [bad_proof]  # type: ignore[index]
    with pytest.raises(SolutionRecordError, match="evidence block"):
        validate_solution(raw, task(), alias_pages, visual_id_checks=[bad_proof])


def test_validate_solution_preserves_large_decimal_literal_precision() -> None:
    raw = record()
    value = "0.123456789012345678901234567890"
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": f"{value} + 0", "result": value}
    ]
    raw["answer"] = value

    validate_solution(raw, task(), pages())


def test_validate_solution_rejects_more_than_50_significant_digits() -> None:
    raw = record()
    literal = "1." + "1" * 50
    rounded = "1." + "1" * 49
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": f"{literal} + 0", "result": rounded}
    ]
    raw["answer"] = rounded

    with pytest.raises(SolutionRecordError, match="50 significant digits"):
        validate_solution(raw, task(), pages())

    raw = record()
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": "1 + 0", "result": "1." + "0" * 50}
    ]
    raw["answer"] = "1"
    with pytest.raises(SolutionRecordError, match="50 significant digits"):
        validate_solution(raw, task(), pages())


def test_validate_solution_accepts_exact_30_digit_unary_negative() -> None:
    raw = record()
    value = "123456789012345678901234567890"
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": f"-{value}", "result": f"-{value}"}
    ]
    raw["answer"] = f"-{value}"

    validate_solution(raw, task(), pages())


def test_validate_solution_rejects_inexact_repeating_division() -> None:
    raw = record()
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": "1 / 3", "result": "0.333333333333333333333333333333"}
    ]
    raw["answer"] = "0.333333333333333333333333333333"

    with pytest.raises(SolutionRecordError, match="unsupported precision"):
        validate_solution(raw, task(), pages())


def test_validate_solution_rejects_decimal_exponent_outside_bound() -> None:
    raw = record()
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": "1e-101 + 0", "result": "1e-101"}
    ]
    raw["answer"] = "1e-101"

    with pytest.raises(SolutionRecordError, match="exponent outside"):
        validate_solution(raw, task(), pages())


def test_validate_solution_rejects_nonnumeric_answer_with_calculations() -> None:
    raw = record()
    raw["answer"] = "seven"

    with pytest.raises(SolutionRecordError, match="numeric.*calculations"):
        validate_solution(raw, task(), pages())


def test_validate_solution_preserves_supported_indeterminate_answer() -> None:
    raw = record()
    raw["solution"]["calculations"] = []  # type: ignore[index]
    raw["answer"] = "not uniquely determined"
    raw["uncertainties"] = ["The cited pages do not identify a unique total."]

    normalized = validate_solution(raw, task(), pages())

    assert normalized["answer"] == "not uniquely determined"
    assert normalized["solution"]["calculations"] == []
    assert normalized["uncertainties"] == [
        "The cited pages do not identify a unique total."
    ]


def test_validate_solution_requires_uncertainty_for_nonnumeric_answer() -> None:
    raw = record()
    raw["solution"]["calculations"] = []  # type: ignore[index]
    raw["answer"] = "not uniquely determined"

    with pytest.raises(SolutionRecordError, match="explicit uncertainty"):
        validate_solution(raw, task(), pages())


@pytest.mark.parametrize("operator", ["//", "%", "**"])
def test_validate_solution_rejects_unsupported_arithmetic_operators(operator: str) -> None:
    raw = record()
    raw["solution"]["calculations"] = [  # type: ignore[index]
        {"expression": f"5 {operator} 2", "result": "1"}
    ]
    raw["answer"] = "1"

    with pytest.raises(SolutionRecordError, match="arithmetic expression"):
        validate_solution(raw, task(), pages())


def test_export_records_writes_private_complete_outputs(tmp_path: Path) -> None:
    second = record("task_000002")
    tasks = [task(), task("task_000002")]
    manifest = export_records(
        [second, record()],
        tasks,
        tmp_path / "private",
        source_pages_by_id={item["instance_id"]: pages() for item in tasks},
        expected_split="heldout",
    )

    output_dir = tmp_path / "private"
    assert manifest["total"] == 2
    assert manifest["private"] is True
    assert set(manifest["files"]) == {
        "solutions.jsonl",
        "solutions.md",
        "submission.jsonl",
    }
    rows = [
        json.loads(line)
        for line in (output_dir / "solutions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["instance_id"] for row in rows] == ["task_000001", "task_000002"]
    submission = [
        json.loads(line)
        for line in (output_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert set(submission[0]) == {"instance_id", "answer", "evidence"}
    markdown = (output_dir / "solutions.md").read_text(encoding="utf-8")
    assert "## task_000001" in markdown
    assert all(
        heading in markdown
        for heading in (
            "질문 (Question)",
            "풀이 (Solution)",
            "계산식 (Calculations)",
            "정답 (Answer)",
            "Evidence",
            "불확실성 (Uncertainties)",
        )
    )
    assert (output_dir.stat().st_mode & 0o077) == 0
    assert all((output_dir / name).stat().st_mode & 0o077 == 0 for name in manifest["files"])
    assert not list(output_dir.glob(".*.tmp-*"))


def test_export_records_requires_independent_visual_checks_by_instance(
    tmp_path: Path,
) -> None:
    raw, alias_pages = visual_alias_fixture()
    candidate = visual_evidence_candidates(raw, alias_pages)[0]
    proof = visual_proof(candidate)
    raw["provenance"]["visual_id_checks"] = [proof]  # type: ignore[index]

    with pytest.raises(SolutionRecordError, match="independently supplied"):
        export_records(
            [raw],
            [task()],
            tmp_path / "missing-check",
            source_pages_by_id={"task_000001": alias_pages},
            expected_split="heldout",
        )

    export_records(
        [raw],
        [task()],
        tmp_path / "verified",
        source_pages_by_id={"task_000001": alias_pages},
        expected_split="heldout",
        visual_checks_by_id={"task_000001": [proof]},
    )

    with pytest.raises(SolutionRecordError, match="unknown visual checks"):
        export_records(
            [raw],
            [task()],
            tmp_path / "unknown-check",
            source_pages_by_id={"task_000001": alias_pages},
            expected_split="heldout",
            visual_checks_by_id={
                "task_000001": [proof],
                "task_999999": [proof],
            },
        )


def test_export_records_rejects_file_symlink_without_touching_outside_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text("sentinel\n", encoding="utf-8")
    output_dir = tmp_path / "private"
    output_dir.mkdir()
    (output_dir / "solutions.jsonl").symlink_to(outside)

    with pytest.raises(SolutionRecordError, match="symbolic link"):
        export_records(
            [record()],
            [task()],
            output_dir,
            source_pages_by_id={"task_000001": pages()},
            expected_split="heldout",
        )

    assert outside.read_text(encoding="utf-8") == "sentinel\n"


def test_export_records_rejects_output_under_symlinked_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SolutionRecordError, match="symbolic link"):
        export_records(
            [record()],
            [task()],
            linked / "private",
            source_pages_by_id={"task_000001": pages()},
            expected_split="heldout",
        )

    assert not (outside / "private").exists()


@pytest.mark.parametrize(
    ("records", "tasks", "message"),
    [
        ([record(), record()], [task()], "duplicate record"),
        ([record()], [task(), task("task_000002")], "missing record"),
        ([record(), record("task_999999")], [task()], "unknown record"),
        ([record()], [task(), task()], "duplicate task"),
    ],
)
def test_export_records_rejects_duplicate_missing_and_unknown_ids(
    tmp_path: Path,
    records: list[dict[str, object]],
    tasks: list[dict[str, str]],
    message: str,
) -> None:
    source_pages_by_id = {item["instance_id"]: pages() for item in tasks}
    with pytest.raises(SolutionRecordError, match=message):
        export_records(
            records,
            tasks,
            tmp_path / "output",
            source_pages_by_id=source_pages_by_id,
            expected_split="heldout",
        )


def test_export_records_requires_independent_source_coverage_and_expected_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(SolutionRecordError, match="missing source pages"):
        export_records(
            [record()],
            [task()],
            tmp_path / "missing-source",
            source_pages_by_id={},
            expected_split="heldout",
        )

    raw = record()
    raw["split"] = "train"
    with pytest.raises(SolutionRecordError, match="expected split"):
        export_records(
            [raw],
            [task()],
            tmp_path / "wrong-split",
            source_pages_by_id={"task_000001": pages()},
            expected_split="heldout",
        )


def test_validate_solution_requires_verifiable_provenance() -> None:
    raw = record()
    raw["provenance"] = {
        "pdf_sha256": "a" * 64,
        "input_sha256": "0" * 64,
        "config_sha256": "b" * 64,
        "model": "fixture-model",
        "method": "unit-test",
    }
    with pytest.raises(SolutionRecordError, match="input_sha256"):
        validate_solution(raw, task(), pages())

    for missing in ("pdf_sha256", "input_sha256", "config_sha256", "model", "method"):
        raw = record()
        del raw["provenance"][missing]  # type: ignore[index]
        with pytest.raises(SolutionRecordError, match=missing):
            validate_solution(raw, task(), pages())


def test_input_digest_uses_only_official_task_fields() -> None:
    enriched_task = {**task(), "answer": "must-not-affect-input", "other": 1}

    assert input_digest(enriched_task, pages()) == input_digest(task(), pages())


def test_evaluate_records_returns_only_comparisons_and_aggregate() -> None:
    evaluation = evaluate_records(
        [record()],
        [{"instance_id": "task_000001", "answer": "12.5", "evidence": ["A-1!"]}],
        "prior-validation-submission",
    )

    assert set(evaluation) == {"comparisons", "aggregate"}
    assert evaluation["aggregate"] == {
        "reference_kind": "prior-validation-submission",
        "total": 1,
        "answer_matches": 1,
        "evidence_matches": 0,
    }
    assert set(evaluation["comparisons"][0]) == {
        "instance_id",
        "answer_match",
        "evidence_match",
    }
    serialized = json.dumps(evaluation)
    assert "question" not in serialized
    assert "solution" not in serialized
    assert "source_pages" not in serialized


def test_evaluate_records_requires_exact_unique_coverage() -> None:
    with pytest.raises(SolutionRecordError, match="missing reference"):
        evaluate_records([record()], [], "reference")

    with pytest.raises(SolutionRecordError, match="duplicate reference"):
        evaluate_records(
            [record()],
            [
                {"instance_id": "task_000001", "answer": "12.5", "evidence": []},
                {"instance_id": "task_000001", "answer": "12.5", "evidence": []},
            ],
            "reference",
        )


def test_evaluate_records_normalizes_numeric_text_and_evidence_order() -> None:
    numeric = evaluate_records(
        [record()],
        [
            {
                "instance_id": "task_000001",
                "answer": "12.50",
                "evidence": ["note#3", "A-1!"],
            }
        ],
        "reference",
    )
    assert numeric["comparisons"][0] == {
        "instance_id": "task_000001",
        "answer_match": True,
        "evidence_match": True,
    }

    raw = record()
    raw["answer"] = "  Net Income  "
    raw["solution"]["calculations"] = []  # type: ignore[index]
    text = evaluate_records(
        [raw],
        [
            {
                "instance_id": "task_000001",
                "answer": "net income",
                "evidence": ["A-1!", "note#3"],
            }
        ],
        "reference",
    )
    assert text["comparisons"][0]["answer_match"] is True


@pytest.mark.parametrize(
    "bad_values",
    [
        {"answer": None, "evidence": ["A-1!"]},
        {"answer": "", "evidence": ["A-1!"]},
        {"answer": "12.5", "evidence": None},
        {"answer": "12.5", "evidence": []},
        {"answer": "12.5", "evidence": ["A-1!", "A-1!"]},
        {"answer": "12.5", "evidence": [""]},
    ],
)
def test_evaluate_records_rejects_malformed_comparison_values(
    bad_values: dict[str, object],
) -> None:
    reference = {"instance_id": "task_000001", **bad_values}

    with pytest.raises(SolutionRecordError, match="reference.*(answer|evidence)"):
        evaluate_records([record()], [reference], "reference")
