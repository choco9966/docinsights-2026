import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from docinsights_ocr.cli import build_parser
from docinsights_ocr.codex_query_compare import (
    LEAD_INS,
    _category,
    _character_diff,
    _extract_pdf_pages,
    _pdf_blocks,
    _scenario_matches,
    compare_codex_queries,
    write_codex_query_comparison,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, tuple[str, ...]]]:
    cases = [
        ("task_000001", "validation", "Compute 12 kg.", "Compute 12 kg."),
        ("task_000002", "validation", "Line one\nLine two", "Line one   Line two"),
        ("task_000003", "test", "What is 12 kg?", "What is 12 kg"),
        ("task_000004", "test", "Value 500 kg", "Value SOO kg"),
        ("task_000005", "test", "Value 600 kg", "Value 500 kg"),
        ("task_000006", "test", "Not recoverable", "No generated scenario here"),
    ]
    tasks: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    pages: dict[str, tuple[str, ...]] = {}
    for instance_id, split, query, recovered in cases:
        pdf = tmp_path / f"{instance_id}.pdf"
        pdf.write_bytes(f"pdf:{instance_id}".encode())
        tasks.append(
            {
                "instance_id": instance_id,
                "document_pdf": pdf.name,
                "user_query": query,
                "split": split,
                "labels": {"must": "not be used"},
            }
        )
        lead_in = LEAD_INS[(int(instance_id[-1]) - 1) % len(LEAD_INS)]
        reference_text = (
            f"preface {lead_in} {recovered}" if instance_id != "task_000006" else recovered
        )
        references.append(
            {
                "instance_id": instance_id,
                "status": "ok",
                "blocks": [{"block_id": "b12", "text": reference_text}],
                "provenance": {"input_pdf_sha256": _sha256(pdf)},
                "answer": "must not be used",
            }
        )
        pdf_recovered = "Value 500 kg" if instance_id == "task_000004" else recovered
        if instance_id == "task_000006":
            pages[pdf.name] = ("b12 No generated scenario here\n",)
        else:
            pdf_lead_in = "  ".join(lead_in.swapcase().split())
            pages[pdf.name] = (
                f"b11 preceding block\nb12 preface {pdf_lead_in} {pdf_recovered}\n",
                "b13 following block\n",
            )
    tasks_path = tmp_path / "tasks.jsonl"
    references_path = tmp_path / "reference.jsonl"
    _write_jsonl(tasks_path, tasks)
    _write_jsonl(references_path, references)
    return tasks_path, references_path, pages


def test_compare_recovers_by_lead_in_and_classifies_without_selection_labels(
    tmp_path: Path,
) -> None:
    tasks, references, pages = _artifacts(tmp_path)

    def extract(pdf_path: Path, **_kwargs: object) -> tuple[str, ...]:
        return pages[pdf_path.name]

    with patch("docinsights_ocr.codex_query_compare._extract_pdf_pages", side_effect=extract):
        comparison = compare_codex_queries(tasks, references, documents_root=tmp_path)

    records = {record["instance_id"]: record for record in comparison["records"]}
    assert records["task_000001"]["category"] == "exact"
    assert records["task_000002"]["category"] == "line_break_or_whitespace"
    assert records["task_000003"]["category"] == "punctuation"
    assert records["task_000004"]["category"] == "ocr"
    assert records["task_000005"]["category"] == "actual_content_difference"
    assert records["task_000006"]["category"] == "undetermined"
    assert records["task_000001"]["evidence_block_id"] == "b12"
    assert records["task_000001"]["evidence_pages"] == [1]
    assert records["task_000004"]["pdf_recovered_query"] == "Value 500 kg"
    assert records["task_000004"]["diff"]
    assert records["task_000002"]["diff"] == []

    summary = comparison["summary"]
    assert summary["counts"] == {
        "exact": 1,
        "normalized": 1,
        "mismatch": 3,
        "undetermined": 1,
    }
    assert summary["splits"]["validation"]["instance_ids"]["normalized"] == [
        "task_000002"
    ]
    assert comparison["sources"]["tasks_manifest"]["sha256"] == _sha256(tasks)
    assert comparison["sources"]["codex_reference_output"]["sha256"] == _sha256(
        references
    )
    serialized = json.dumps(comparison)
    assert "must not be used" not in serialized


def test_pdf_block_continuation_records_all_evidence_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "continued.pdf"
    pdf.write_bytes(b"pdf")
    tasks = tmp_path / "tasks.jsonl"
    references = tmp_path / "reference.jsonl"
    query = "First amount 10 kg. Second amount 20 kg?"
    _write_jsonl(
        tasks,
        [{"instance_id": "task_000001", "document_pdf": pdf.name, "user_query": query}],
    )
    _write_jsonl(
        references,
        [
            {
                "instance_id": "task_000001",
                "status": "ok",
                "blocks": [{"block_id": "b09", "text": f"{LEAD_INS[0]} {query}"}],
                "provenance": {"input_pdf_sha256": _sha256(pdf)},
            }
        ],
    )
    page_texts = (
        f"b09 {LEAD_INS[0]} First amount 10 kg.\n",
        "Second amount 20 kg?\nb10 Next block\n",
    )
    with patch(
        "docinsights_ocr.codex_query_compare._extract_pdf_pages", return_value=page_texts
    ):
        result = compare_codex_queries(tasks, references, documents_root=tmp_path)

    record = result["records"][0]
    assert record["comparison_status"] == "exact"
    assert record["evidence_pages"] == [1, 2]
    assert record["pdf_recovered_query"] == "First amount 10 kg.\nSecond amount 20 kg?"


def test_write_outputs_jsonl_and_markdown_with_hashes_and_mismatch_details(
    tmp_path: Path,
) -> None:
    tasks, references, pages = _artifacts(tmp_path)
    with patch(
        "docinsights_ocr.codex_query_compare._extract_pdf_pages",
        side_effect=lambda path, **_kwargs: pages[path.name],
    ):
        comparison = compare_codex_queries(tasks, references, documents_root=tmp_path)
    jsonl_path = tmp_path / "reports" / "query-comparison.jsonl"
    markdown_path = tmp_path / "reports" / "query-comparison.md"

    manifest = write_codex_query_comparison(comparison, jsonl_path, markdown_path)

    assert len(jsonl_path.read_text().splitlines()) == 6
    report = markdown_path.read_text(encoding="utf-8")
    assert "# Codex Query Comparison" in report
    assert "task_000004" in report
    assert "Value SOO kg" in report
    assert "Character diff" in report
    assert "## Undetermined" in report
    assert "reference_lead_in_matches_0" in report
    assert "Not recoverable" in report
    assert str(tasks.resolve()) in report
    assert _sha256(tasks) in report
    assert manifest["outputs"]["jsonl"]["sha256"] == _sha256(jsonl_path)
    assert manifest["outputs"]["markdown"]["sha256"] == _sha256(markdown_path)


def test_pdftotext_is_invoked_one_page_at_a_time_until_page_range_ends(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        page = int(argv[argv.index("-f") + 1])
        if page <= 2:
            return subprocess.CompletedProcess(argv, 0, f"page {page}", "")
        return subprocess.CompletedProcess(argv, 99, "", "Wrong page range")

    with patch("docinsights_ocr.codex_query_compare.subprocess.run", side_effect=run):
        assert _extract_pdf_pages(pdf, executable="pdftotext-test", timeout_seconds=2) == (
            "page 1",
            "page 2",
        )

    assert [argv[argv.index("-f") + 1] for argv in calls] == ["1", "2", "3"]
    assert all(argv[argv.index("-f") + 1] == argv[argv.index("-l") + 1] for argv in calls)


def test_image_only_pdf_falls_back_to_rendered_two_page_tesseract_ocr(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "image-only.pdf"
    pdf.write_bytes(b"pdf")
    images = (tmp_path / "page-1.png", tmp_path / "page-2.png")
    for image in images:
        image.write_bytes(b"png")
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[0] == "pdftotext-test":
            page = int(argv[argv.index("-f") + 1])
            if page <= 2:
                return subprocess.CompletedProcess(argv, 0, "\f", "")
            return subprocess.CompletedProcess(argv, 99, "", "Wrong page range")
        page_text = "page one OCR" if argv[1].endswith("page-1.png") else "page two OCR"
        return subprocess.CompletedProcess(argv, 0, page_text, "")

    with (
        patch("docinsights_ocr.codex_query_compare.subprocess.run", side_effect=run),
        patch(
            "docinsights_ocr.codex_query_compare.render_pdf", return_value=images
        ) as renderer,
    ):
        pages = _extract_pdf_pages(
            pdf,
            executable="pdftotext-test",
            renderer_executable="pdftoppm-test",
            tesseract_executable="tesseract-test",
            fallback_dpi=150,
            timeout_seconds=2,
        )

    assert pages == ("page one OCR", "page two OCR")
    renderer.assert_called_once_with(
        pdf,
        renderer.call_args.args[1],
        dpi=150,
        executable="pdftoppm-test",
        timeout_seconds=2,
    )
    tesseract_calls = [argv for argv in calls if argv[0] == "tesseract-test"]
    assert len(tesseract_calls) == 2
    assert all(argv[2:] == ["stdout", "-l", "eng", "--psm", "6"] for argv in tesseract_calls)


def test_image_only_fallback_drives_pdf_recovery_and_ocr_category(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    tasks = tmp_path / "tasks.jsonl"
    references = tmp_path / "reference.jsonl"
    _write_jsonl(
        tasks,
        [
            {
                "instance_id": "task_000001",
                "document_pdf": pdf.name,
                "user_query": "Value 500 kg",
            }
        ],
    )
    _write_jsonl(
        references,
        [
            {
                "instance_id": "task_000001",
                "status": "ok",
                "blocks": [
                    {"block_id": "b12", "text": f"{LEAD_INS[0]} Value SOO kg"}
                ],
                "provenance": {"input_pdf_sha256": _sha256(pdf)},
            }
        ],
    )
    images = (tmp_path / "page-1.png", tmp_path / "page-2.png")
    for image in images:
        image.write_bytes(b"png")

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[0] == "pdftotext-test":
            page = int(argv[argv.index("-f") + 1])
            if page <= 2:
                return subprocess.CompletedProcess(argv, 0, "\f", "")
            return subprocess.CompletedProcess(argv, 99, "", "Wrong page range")
        text = (
            f"b12 {LEAD_INS[0]} Value 500 kg\n"
            if argv[1].endswith("page-1.png")
            else "b13 End\n"
        )
        return subprocess.CompletedProcess(argv, 0, text, "")

    with (
        patch("docinsights_ocr.codex_query_compare.subprocess.run", side_effect=run),
        patch("docinsights_ocr.codex_query_compare.render_pdf", return_value=images),
    ):
        comparison = compare_codex_queries(
            tasks,
            references,
            documents_root=tmp_path,
            pdftotext_executable="pdftotext-test",
            renderer_executable="pdftoppm-test",
            tesseract_executable="tesseract-test",
        )

    record = comparison["records"][0]
    assert record["category"] == "ocr"
    assert record["pdf_recovered_query"] == "Value 500 kg"
    assert record["evidence_pages"] == [1]


def test_image_only_fallback_requires_exactly_two_rendered_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    image = tmp_path / "page-1.png"
    image.write_bytes(b"png")

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        page = int(argv[argv.index("-f") + 1])
        if page <= 2:
            return subprocess.CompletedProcess(argv, 0, "\f", "")
        return subprocess.CompletedProcess(argv, 99, "", "Wrong page range")

    with (
        patch("docinsights_ocr.codex_query_compare.subprocess.run", side_effect=run),
        patch("docinsights_ocr.codex_query_compare.render_pdf", return_value=(image,)),
        pytest.raises(ValueError, match="exactly two PDF pages"),
    ):
        _extract_pdf_pages(pdf, executable="pdftotext-test", timeout_seconds=2)


def test_pdf_block_marker_canonicalizes_ocr_o_without_changing_block_text() -> None:
    blocks = _pdf_blocks(
        ("bO6: Value O0 kg and code bO7 remain unchanged.\nbO09 Scenario\nb10 Next\n",)
    )

    assert blocks == [
        {
            "block_id": "b06",
            "text": "Value O0 kg and code bO7 remain unchanged.",
            "page_numbers": [1],
        },
        {"block_id": "b09", "text": "Scenario", "page_numbers": [1]},
        {"block_id": "b10", "text": "Next", "page_numbers": [1]},
    ]


def test_workers_parallelize_documents_and_preserve_manifest_order(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    references = tmp_path / "reference.jsonl"
    task_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for number in range(4):
        instance_id = f"task_{number:06d}"
        pdf = tmp_path / f"{instance_id}.pdf"
        pdf.write_bytes(instance_id.encode())
        task_rows.append(
            {"instance_id": instance_id, "document_pdf": pdf.name, "user_query": "q"}
        )
        reference_rows.append({"instance_id": instance_id})
    _write_jsonl(tasks, task_rows)
    _write_jsonl(references, reference_rows)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def compare_one(instance_id: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01 * (4 - int(instance_id[-1])))
        with lock:
            active -= 1
        return {
            "instance_id": instance_id,
            "split": None,
            "comparison_status": "exact",
            "category": "exact",
        }

    with patch("docinsights_ocr.codex_query_compare._compare_one", side_effect=compare_one):
        comparison = compare_codex_queries(tasks, references, documents_root=tmp_path, workers=4)

    assert max_active > 1
    assert [record["instance_id"] for record in comparison["records"]] == [
        row["instance_id"] for row in task_rows
    ]


@pytest.mark.parametrize("workers", [0, 5])
def test_workers_must_be_between_one_and_four(
    tmp_path: Path, workers: int
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    references = tmp_path / "reference.jsonl"
    tasks.write_text("", encoding="utf-8")
    references.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="workers must be between one and four"):
        compare_codex_queries(tasks, references, workers=workers)


def test_comparison_primitives_preserve_units_numbers_and_symbols() -> None:
    assert _category("10 kg", "10  kg", "10 kg") == "line_break_or_whitespace"
    assert _category("$10/kg?", "$10/kg", "$10/kg") == "punctuation"
    assert _category("10 kg", "10 lb", "10 lb") == "actual_content_difference"
    assert _category("10 kg", "1O kg", "10 kg") == "ocr"
    assert _character_diff("10 kg", "1O kg") == [
        {
            "operation": "replace",
            "user_query": {"start": 1, "end": 2, "text": "0"},
            "recovered_query": {"start": 1, "end": 2, "text": "O"},
        }
    ]


def test_split_name_can_be_explicit_when_manifest_rows_have_no_split(tmp_path: Path) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    tasks = tmp_path / "tasks.jsonl"
    references = tmp_path / "reference.jsonl"
    query = "How many items?"
    _write_jsonl(
        tasks,
        [{"instance_id": "task_000001", "document_pdf": pdf.name, "user_query": query}],
    )
    _write_jsonl(
        references,
        [
            {
                "instance_id": "task_000001",
                "status": "ok",
                "blocks": [{"block_id": "b09", "text": f"{LEAD_INS[0]} {query}"}],
                "provenance": {"input_pdf_sha256": _sha256(pdf)},
            }
        ],
    )
    with patch(
        "docinsights_ocr.codex_query_compare._extract_pdf_pages",
        return_value=(f"b09 {LEAD_INS[0]} {query}\n",),
    ):
        comparison = compare_codex_queries(
            tasks,
            references,
            documents_root=tmp_path,
            split_name="validation",
        )

    assert comparison["sources"]["split"] == "validation"
    assert comparison["records"][0]["split"] == "validation"
    assert comparison["summary"]["splits"]["validation"]["total_count"] == 1


def test_codex_query_compare_cli_surface() -> None:
    args = build_parser().parse_args(
        [
            "codex-query-compare",
            "tasks.jsonl",
            "reference.jsonl",
            "comparison.jsonl",
            "comparison.md",
            "--split-name",
            "validation",
        ]
    )

    assert args.split_name == "validation"
    assert args.pdftotext_executable == "pdftotext"
    assert args.renderer_executable == "pdftoppm"
    assert args.tesseract_executable == "tesseract"
    assert args.fallback_dpi == 200
    assert args.workers == 1


def test_lead_in_matching_is_case_whitespace_robust_and_fail_closed() -> None:
    robust = "  ".join(LEAD_INS[0].swapcase().split())
    assert _scenario_matches([{"block_id": "b07", "text": f"{robust} Query 10 kg?"}]) == [
        {"block_id": "b07", "query": "Query 10 kg?", "page_numbers": []}
    ]
    assert _scenario_matches([{"block_id": "b07", "text": LEAD_INS[0]}]) == []
    matches = _scenario_matches(
        [
            {
                "block_id": "b07",
                "text": f"{LEAD_INS[0]} First? {LEAD_INS[1]} Second?",
            }
        ]
    )
    assert len(matches) == 2


def test_lead_in_matching_accepts_tesseract_rn_as_m_in_fixed_phrase_only() -> None:
    lead_in = LEAD_INS[2].replace("afternoon", "aftemoon")

    assert _scenario_matches(
        [{"block_id": "b12", "text": f"{lead_in} Query aftemoon value?"}]
    ) == [
        {
            "block_id": "b12",
            "query": "Query aftemoon value?",
            "page_numbers": [],
        }
    ]


def test_coverage_and_failed_pdf_extraction_fail_closed(tmp_path: Path) -> None:
    tasks, references, _ = _artifacts(tmp_path)
    rows = [json.loads(line) for line in references.read_text().splitlines()]
    _write_jsonl(references, rows[:-1])
    with pytest.raises(ValueError, match="coverage mismatch"):
        compare_codex_queries(tasks, references, documents_root=tmp_path)

    _write_jsonl(references, rows)
    with patch(
        "docinsights_ocr.codex_query_compare._extract_pdf_pages",
        side_effect=FileNotFoundError("pdftotext"),
    ):
        comparison = compare_codex_queries(tasks, references, documents_root=tmp_path)
    record = comparison["records"][0]
    assert record["comparison_status"] == "undetermined"
    assert record["undetermined_reason"] == "pdf_extraction_failed:FileNotFoundError"


def test_comparison_rejects_stale_reference_pdf_binding(tmp_path: Path) -> None:
    tasks, references, _ = _artifacts(tmp_path)
    rows = [json.loads(line) for line in references.read_text().splitlines()]
    rows[0]["provenance"]["input_pdf_sha256"] = "0" * 64
    _write_jsonl(references, rows)

    with pytest.raises(ValueError, match="reference PDF SHA-256 mismatch"):
        compare_codex_queries(tasks, references, documents_root=tmp_path)


def test_write_rejects_source_output_collision_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    tasks, references, pages = _artifacts(tmp_path)
    with patch(
        "docinsights_ocr.codex_query_compare._extract_pdf_pages",
        side_effect=lambda path, **_kwargs: pages[path.name],
    ):
        comparison = compare_codex_queries(tasks, references, documents_root=tmp_path)

    with pytest.raises(ValueError, match="collide with source paths"):
        write_codex_query_comparison(comparison, tasks, tmp_path / "report.md")

    jsonl_path = tmp_path / "report.jsonl"
    markdown_path = tmp_path / "report.md"
    write_codex_query_comparison(comparison, jsonl_path, markdown_path)
    assert jsonl_path.is_file()
    assert markdown_path.is_file()
    assert not list(tmp_path.glob(".*.tmp"))
