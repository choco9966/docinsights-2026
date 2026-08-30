import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

import docinsights_ocr.benchmark as benchmark_module
from docinsights_ocr.benchmark import compare, hash_run, prepare, run
from docinsights_ocr.cli import build_parser
from docinsights_ocr.models import BoundingBox, Line, Page
from docinsights_ocr.records import (
    aggregate_ocr_hash,
    deterministic_content_hash,
    failure_record,
    ocr_record_hash,
    read_jsonl,
    write_jsonl,
)


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_supports_limit_and_resume(tmp_path: Path) -> None:
    for number in range(3):
        (tmp_path / f"{number}.pdf").write_bytes(f"pdf-{number}".encode())
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            json.dumps(
                {"instance_id": f"i{number}", "user_query": "q", "document_pdf": f"{number}.pdf"}
            )
            for number in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"

    assert prepare(tasks, output, limit=2) == 2
    assert prepare(tasks, output, resume=True, limit=1) == 1
    records = _read(output)
    assert [row["instance_id"] for row in records] == ["i0", "i1", "i2"]
    assert all(len(row["input_pdf_sha256"]) == 64 for row in records)
    assert all(row["split"] == "ocr_dev" for row in records)
    assert len({row["split_seed"] for row in records}) == 1


def test_prepare_assigns_exactly_thirty_query_free_deterministic_dev_items(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    rows = []
    for number in range(35):
        pdf = tmp_path / f"{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        rows.append(
            {
                "instance_id": f"i{number:02}",
                "user_query": f"secret-{number}",
                "document_pdf": pdf.name,
            }
        )
    tasks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    prepare(tasks, first)
    rows.reverse()
    for row in rows:
        row["user_query"] = "changed query"
    tasks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    prepare(tasks, second)

    first_splits = {row["instance_id"]: row["split"] for row in _read(first)}
    second_splits = {row["instance_id"]: row["split"] for row in _read(second)}
    assert sum(split == "ocr_dev" for split in first_splits.values()) == 30
    assert first_splits == second_splits


def test_prepare_sorts_ids_writes_relative_paths_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (documents / name).write_bytes(name.encode())
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "b", "user_query": "q", "document_pdf": "b.pdf"}),
                json.dumps({"instance_id": "a", "user_query": "q", "document_pdf": "a.pdf"}),
            ]
        )
        + "\n"
    )
    output = tmp_path / "manifest.jsonl"

    prepare(tasks, output, documents_root=documents)

    records = _read(output)
    assert [record["instance_id"] for record in records] == ["a", "b"]
    assert [record["document_pdf"] for record in records] == ["a.pdf", "b.pdf"]
    tasks.write_text(
        "\n".join(
            [
                tasks.read_text(),
                json.dumps({"instance_id": "a", "user_query": "other", "document_pdf": "a.pdf"}),
            ]
        )
    )
    with pytest.raises(ValueError, match="duplicate instance_id"):
        prepare(tasks, output, documents_root=documents)


@pytest.mark.parametrize("field", ["user_query", "input_pdf_sha256", "split_seed"])
def test_prepare_resume_rejects_stale_completed_metadata(tmp_path: Path, field: str) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "instance_id": "i1",
                "user_query": "current",
                "document_pdf": pdf.name,
            }
        )
        + "\n"
    )
    output = tmp_path / "manifest.jsonl"
    prepare(tasks, output)
    record = _read(output)[0]
    record[field] = "stale"
    write_jsonl(output, [record])

    with pytest.raises(ValueError, match=f"{field} changed"):
        prepare(tasks, output, resume=True)


def test_run_emits_qwen_record_without_answer_or_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    manifest.write_text(
        json.dumps({"instance_id": "i1", "user_query": "q", "document_pdf": str(pdf)}) + "\n"
    )
    output = tmp_path / "ocr.jsonl"
    image = tmp_path / "page-1.png"

    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(image,)),
        patch(
            "docinsights_ocr.benchmark.TesseractEngine.recognize",
            return_value=Page(
                1, (Line(1, "b01 Revenue 100", bbox=BoundingBox(10, 20, 30, 40), confidence=0.9),)
            ),
        ),
    ):
        assert run(manifest, output) == 1

    record = _read(output)[0]
    assert record["status"] == "ok"
    assert record["schema_version"] == "1.0"
    assert record["pages"] == [
        {
            "page_number": 1,
            "width": None,
            "height": None,
            "coordinate_system": "pixel_top_left",
        }
    ]
    assert record["blocks"] == [
        {
            "block_id": "b01",
            "page_numbers": [1],
            "text": "Revenue 100",
            "lines": [
                {
                    "page_number": 1,
                    "text": "b01 Revenue 100",
                    "bbox": {"left": 10, "top": 20, "width": 30, "height": 40},
                    "confidence": 0.9,
                    "confidence_kind": "mean_word_confidence_0_to_1",
                }
            ],
        }
    ]
    assert "answer" not in record and "evidence" not in record
    assert set(record) >= {"instance_id", "user_query", "blocks", "engine", "provenance", "timing"}


def test_run_fails_closed_without_markers(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    manifest.write_text(
        json.dumps({"instance_id": "i1", "user_query": "q", "document_pdf": str(pdf)}) + "\n"
    )
    output = tmp_path / "ocr.jsonl"

    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(tmp_path / "page.png",)),
        patch(
            "docinsights_ocr.benchmark.TesseractEngine.recognize",
            return_value=Page(1, (Line(1, "unmarked text"),)),
        ),
    ):
        run(manifest, output)

    record = _read(output)[0]
    assert record["status"] == "failed"
    assert record["blocks"] == []
    assert record["schema_version"] == "1.0"
    assert record["error_kind"] == "validation_error"


def test_run_selects_apple_vision_without_passing_query_to_engine(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    manifest.write_text(
        json.dumps(
            {
                "instance_id": "i1",
                "user_query": "must not reach OCR",
                "document_pdf": str(pdf),
                "input_pdf_sha256": hashlib.sha256(b"").hexdigest(),
                "split": "ocr_dev",
            }
        )
        + "\n"
    )
    output = tmp_path / "ocr.jsonl"
    image = tmp_path / "page.png"

    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(image,)),
        patch(
            "docinsights_ocr.benchmark.AppleVisionEngine.recognize",
            return_value=Page(1, (Line(1, "b01 Apple text"),)),
        ) as recognize,
    ):
        run(manifest, output, engine="apple-vision", apple_vision_executable="/opt/apple-ocr")

    recognize.assert_called_once_with(image, page_number=1)
    record = _read(output)[0]
    assert record["engine"] == "apple-vision"
    assert record["user_query"] == "must not reach OCR"
    assert record["provenance"]["input_pdf_sha256"] == hashlib.sha256(b"").hexdigest()
    assert record["provenance"]["ocr_engine"] == "apple-vision"
    assert record["provenance"]["ocr_options"]["recognition_mode"] == "accurate"
    assert record["provenance"]["ocr_options"]["executable"] == "/opt/apple-ocr"
    assert record["provenance"]["ocr_executable_identity"] == {
        "name": "apple-ocr",
        "kind": "command_name",
        "sha256": None,
    }
    assert "renderer_executable_identity" in record["provenance"]


def test_run_selects_paddleocr_with_pinned_model_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    manifest.write_text(
        json.dumps(
            {
                "instance_id": "i1",
                "user_query": "must not reach OCR",
                "document_pdf": str(pdf),
                "input_pdf_sha256": hashlib.sha256(b"").hexdigest(),
                "split": "ocr_eval",
                "split_seed": "test",
            }
        )
        + "\n"
    )
    output = tmp_path / "ocr.jsonl"
    image = tmp_path / "page.png"

    class FakePaddleEngine:
        name = "paddleocr-ppocrv5-mobile"
        language = "eng"
        confidence_kind = "paddleocr_line_confidence_0_to_1"
        options = {
            "detection_model_revision": "det-revision",
            "recognition_model_revision": "rec-revision",
        }
        executable_identity = {
            "name": "paddleocr-python-api",
            "kind": "python_packages_and_model_trees",
            "sha256": "identity",
        }

        def recognize(self, image_path: Path, *, page_number: int) -> Page:
            assert image_path == image
            return Page(page_number, (Line(page_number, "b01 Paddle text"),))

    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(image,)),
        patch(
            "docinsights_ocr.benchmark.PaddleOCREngine", return_value=FakePaddleEngine()
        ) as engine,
    ):
        run(
            manifest,
            output,
            engine="paddleocr",
            paddle_detection_model_dir="/models/det",
            paddle_recognition_model_dir="/models/rec",
            paddle_detection_model_revision="det-revision",
            paddle_recognition_model_revision="rec-revision",
        )

    engine.assert_called_once_with(
        detection_model_dir="/models/det",
        recognition_model_dir="/models/rec",
        detection_model_revision="det-revision",
        recognition_model_revision="rec-revision",
        enable_mkldnn=False,
    )
    record = _read(output)[0]
    assert record["engine"] == "paddleocr-ppocrv5-mobile"
    assert record["provenance"]["ocr_options"]["detection_model_revision"] == "det-revision"
    assert record["provenance"]["ocr_executable_identity"]["sha256"] == "identity"


def test_run_paddleocr_requires_local_model_directories(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    with pytest.raises(ValueError, match="requires pinned"):
        run(manifest, tmp_path / "out.jsonl", engine="paddleocr")


def test_run_fails_closed_on_subprocess_error(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    manifest.write_text(
        json.dumps({"instance_id": "i1", "user_query": "q", "document_pdf": str(pdf)}) + "\n"
    )
    output = tmp_path / "ocr.jsonl"

    with patch(
        "docinsights_ocr.benchmark.render_pdf",
        side_effect=subprocess.CalledProcessError(1, ["pdftoppm"]),
    ):
        run(manifest, output)

    record = _read(output)[0]
    assert record["status"] == "failed"
    assert record["blocks"] == []
    assert record["error_kind"] == "subprocess_error"


def test_failure_record_never_contains_partial_blocks() -> None:
    record = failure_record(
        instance_id="i1",
        user_query="q",
        engine="test",
        provenance={},
        timing={},
        error="bad",
    )
    assert record["blocks"] == []
    assert record["error_kind"] == "unknown_error"
    assert "answer" not in record and "evidence" not in record


def test_failure_record_classifies_timeout_and_content_hash_is_deterministic() -> None:
    record = failure_record(
        instance_id="i1",
        user_query="q",
        engine="test",
        provenance={},
        timing={},
        error=subprocess.TimeoutExpired(["ocr"], 120),
    )
    assert record["error_kind"] == "timeout"
    assert deterministic_content_hash({"b": 2, "a": 1}) == deterministic_content_hash(
        {"a": 1, "b": 2}
    )


def test_ocr_content_hash_ignores_timing_and_machine_paths_but_tracks_text_order() -> None:
    record = {
        "schema_version": "1.0",
        "instance_id": "i1",
        "user_query": "query",
        "status": "ok",
        "engine": "tesseract-tsv",
        "pages": [
            {
                "page_number": 1,
                "width": 100,
                "height": 200,
                "coordinate_system": "pixel_top_left",
            }
        ],
        "blocks": [
            {"block_id": "b01", "text": "Alpha", "page_numbers": [1], "lines": []},
            {"block_id": "b02", "text": "Beta", "page_numbers": [1], "lines": []},
        ],
        "provenance": {
            "document_pdf": "/machine-a/private/doc.pdf",
            "ocr_engine": "tesseract-tsv",
            "renderer": "poppler-pdftoppm",
            "ocr_options": {"page_segmentation_mode": 6, "executable": "/opt/a/tesseract"},
            "ocr_executable_identity": {"name": "tesseract", "kind": "sha256", "sha256": "abc"},
        },
        "timing": {"total_seconds": 1.0},
    }
    changed_environment = json.loads(json.dumps(record))
    changed_environment["timing"]["total_seconds"] = 999.0
    changed_environment["provenance"]["document_pdf"] = "/machine-b/doc.pdf"
    changed_environment["provenance"]["ocr_options"]["executable"] = "/usr/bin/tesseract"
    changed_text = json.loads(json.dumps(record))
    changed_text["blocks"][0]["text"] = "Changed"
    changed_order = json.loads(json.dumps(record))
    changed_order["blocks"].reverse()

    original_hash = ocr_record_hash(record)
    assert ocr_record_hash(changed_environment) == original_hash
    assert ocr_record_hash(changed_text) != original_hash
    assert ocr_record_hash(changed_order) != original_hash


def test_aggregate_hash_and_cli_surface_are_deterministic(tmp_path: Path) -> None:
    records = [
        {"instance_id": "b", "status": "failed", "engine": "x", "error_kind": "timeout"},
        {"instance_id": "a", "status": "ok", "engine": "x", "blocks": []},
    ]
    forward = aggregate_ocr_hash(records)
    reverse = aggregate_ocr_hash(reversed(records))
    assert forward == reverse
    path = tmp_path / "run.jsonl"
    write_jsonl(path, records)
    assert hash_run(path) == forward
    args = build_parser().parse_args(["hash", str(path)])
    assert args.command == "hash" and args.input == str(path)


def test_compare_aligns_blocks_and_reports_missing_instances(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    predicted = tmp_path / "predicted.jsonl"
    write_jsonl(
        reference,
        [
            {"instance_id": "i1", "blocks": [{"block_id": "b01", "text": "$100 total"}]},
            {"instance_id": "i2", "blocks": []},
        ],
    )
    write_jsonl(
        predicted,
        [
            {"instance_id": "i1", "blocks": [{"block_id": "b01", "text": "$101 total"}]},
        ],
    )

    result = compare(reference, predicted)

    assert result["instances"] == 2
    assert result["missing_predictions"] == ["i2"]
    assert result["mean_block_f1"] == 1.0


def test_compare_penalizes_missing_blocks_and_instances_and_counts_status(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    predicted = tmp_path / "predicted.jsonl"
    write_jsonl(
        reference,
        [
            {
                "instance_id": "i1",
                "status": "ok",
                "blocks": [
                    {"block_id": "b01", "text": "Revenue $100"},
                    {"block_id": "b02", "text": "Mass 5 kg"},
                ],
            },
            {
                "instance_id": "i2",
                "status": "ok",
                "blocks": [
                    {"block_id": "b03", "text": "Only reference 7%"},
                ],
            },
        ],
    )
    write_jsonl(
        predicted,
        [
            {
                "instance_id": "i1",
                "status": "failed",
                "blocks": [
                    {"block_id": "b01", "text": "Revenue $100"},
                ],
            },
        ],
    )

    result = compare(reference, predicted)
    first, second = result["details"]

    assert first["cer"] > 0 and first["wer"] > 0
    assert first["exact_tokens"]["f1"] < 1
    assert first["blocks"]["ordered_exact"] is False
    assert second["cer"] == 1 and second["predicted_status"] == "missing"
    assert result["ordered_block_exact_count"] == 0
    assert result["status_counts"]["reference"] == {"ok": 2, "failed": 0}
    assert result["status_counts"]["predicted"] == {"ok": 0, "failed": 1}


def test_compare_rejects_duplicate_instance_and_block_ids(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    predicted = tmp_path / "predicted.jsonl"
    write_jsonl(
        reference,
        [
            {"instance_id": "i1", "blocks": []},
            {"instance_id": "i1", "blocks": []},
        ],
    )
    write_jsonl(predicted, [])
    with pytest.raises(ValueError, match="duplicate instance_id"):
        compare(reference, predicted)

    write_jsonl(
        reference,
        [
            {
                "instance_id": "i1",
                "blocks": [
                    {"block_id": "b01", "text": "one"},
                    {"block_id": "B01", "text": "two"},
                ],
            }
        ],
    )
    write_jsonl(predicted, [{"instance_id": "i1", "blocks": []}])
    with pytest.raises(ValueError, match="duplicate block_id"):
        compare(reference, predicted)


def test_run_resume_limit_fingerprint_and_failed_retry(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for number in (1, 2):
        pdf = tmp_path / f"{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        rows.append({"instance_id": f"i{number}", "user_query": "q", "document_pdf": pdf.name})
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "ocr.jsonl"

    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(tmp_path / "page.png",)),
        patch(
            "docinsights_ocr.benchmark.TesseractEngine.recognize",
            return_value=Page(1, (Line(1, "b01 Good"),)),
        ) as recognize,
    ):
        assert run(manifest, output, documents_root=tmp_path, limit=1) == 1
        recognize.reset_mock()
        assert run(manifest, output, documents_root=tmp_path, limit=1, resume=True) == 1
        assert recognize.call_count == 1
    assert [record["instance_id"] for record in _read(output)] == ["i1", "i2"]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(manifest, output, documents_root=tmp_path, resume=True, page_segmentation_mode=3)

    retry_manifest = tmp_path / "retry.jsonl"
    retry_manifest.write_text(json.dumps(rows[0]) + "\n")
    retry_output = tmp_path / "retry-output.jsonl"
    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(tmp_path / "page.png",)),
        patch(
            "docinsights_ocr.benchmark.TesseractEngine.recognize",
            return_value=Page(1, (Line(1, "no marker"),)),
        ),
    ):
        run(retry_manifest, retry_output, documents_root=tmp_path)
    assert _read(retry_output)[0]["status"] == "failed"
    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(tmp_path / "page.png",)),
        patch(
            "docinsights_ocr.benchmark.TesseractEngine.recognize",
            return_value=Page(1, (Line(1, "b01 Retried"),)),
        ),
        patch(
            "docinsights_ocr.benchmark._atomic_write_jsonl",
            wraps=benchmark_module._atomic_write_jsonl,
        ) as checkpoint,
    ):
        assert (
            run(
                retry_manifest,
                retry_output,
                documents_root=tmp_path,
                resume=True,
                retry_failed=True,
            )
            == 1
        )
        assert checkpoint.call_count == 1
    retried = _read(retry_output)
    assert len(retried) == 1 and retried[0]["status"] == "ok"
    assert not retry_output.with_name(f"{retry_output.name}.retry-checkpoint").exists()


def test_retry_prioritizes_unseen_records_and_pipeline_revision_is_fingerprinted(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for number in range(12):
        pdf = tmp_path / f"{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        rows.append(
            {"instance_id": f"i{number:02}", "user_query": "q", "document_pdf": pdf.name}
        )
    write_jsonl(manifest, rows)
    output = tmp_path / "ocr.jsonl"

    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(tmp_path / "page.png",)),
        patch(
            "docinsights_ocr.benchmark.TesseractEngine.recognize",
            return_value=Page(1, (Line(1, "no marker"),)),
        ),
    ):
        assert (
            run(
                manifest,
                output,
                documents_root=tmp_path,
                retry_failed=True,
                limit=10,
                pipeline_revision="commit-a",
            )
            == 10
        )
        assert (
            run(
                manifest,
                output,
                documents_root=tmp_path,
                retry_failed=True,
                resume=True,
                limit=2,
                pipeline_revision="commit-a",
            )
            == 2
        )

    records = _read(output)
    expected_ids = {f"i{number:02}" for number in range(12)}
    assert {record["instance_id"] for record in records} == expected_ids
    assert all(record["provenance"]["pipeline_revision"] == "commit-a" for record in records)
    with pytest.raises(ValueError, match="resume fingerprint mismatch"):
        run(
            manifest,
            output,
            documents_root=tmp_path,
            retry_failed=True,
            resume=True,
            pipeline_revision="commit-b",
        )


@pytest.mark.parametrize("changed_executable", ["ocr", "renderer"])
def test_resume_rejects_same_path_when_executable_content_changes(
    tmp_path: Path,
    changed_executable: str,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "instance_id": "i1",
                "user_query": "q",
                "document_pdf": pdf.name,
            }
        )
        + "\n"
    )
    output = tmp_path / "output.jsonl"
    ocr_executable = tmp_path / "tesseract"
    renderer_executable = tmp_path / "pdftoppm"
    ocr_executable.write_bytes(b"ocr-version-1")
    renderer_executable.write_bytes(b"renderer-version-1")
    options = {
        "documents_root": tmp_path,
        "tesseract_executable": str(ocr_executable),
        "poppler_executable": str(renderer_executable),
    }

    with (
        patch("docinsights_ocr.benchmark.render_pdf", return_value=(tmp_path / "page.png",)),
        patch(
            "docinsights_ocr.benchmark.TesseractEngine.recognize",
            return_value=Page(1, (Line(1, "b01 Stable"),)),
        ),
    ):
        run(manifest, output, **options)

    target = ocr_executable if changed_executable == "ocr" else renderer_executable
    target.write_bytes(f"{changed_executable}-version-2".encode())

    with pytest.raises(ValueError, match="resume fingerprint mismatch"):
        run(manifest, output, resume=True, **options)


def test_read_jsonl_reports_truncated_last_line(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jsonl"
    path.write_text('{"instance_id": "i1"')
    with pytest.raises(ValueError, match="truncated JSONL"):
        list(read_jsonl(path))


def test_strict_schema_declares_all_success_record_shape_fields() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "docsem-ocr-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert "pages" in schema["properties"]
    assert schema["$defs"]["page"]["additionalProperties"] is False
    assert set(schema["$defs"]["page"]["required"]) == {
        "page_number",
        "width",
        "height",
        "coordinate_system",
    }
    line_schema = schema["$defs"]["line"]
    assert "confidence_kind" in line_schema["required"]
    assert line_schema["properties"]["confidence_kind"]["type"] == "string"
    success_rules = [
        rule["then"]
        for rule in schema["allOf"]
        if rule.get("if", {}).get("properties", {}).get("status", {}).get("const") == "ok"
    ]
    assert len(success_rules) == 1
    assert success_rules[0]["required"] == ["pages"]
    assert success_rules[0]["properties"]["blocks"]["minItems"] == 1
    assert success_rules[0]["not"]["anyOf"] == [
        {"required": ["error"]},
        {"required": ["error_kind"]},
    ]
    failure_rules = [
        rule["then"]["required"]
        for rule in schema["allOf"]
        if rule.get("if", {}).get("properties", {}).get("status", {}).get("const") == "failed"
    ]
    assert failure_rules == [["error", "error_kind"]]
    assert "error_kind" in schema["properties"]


def test_readme_run_commands_pin_documents_root_and_timeout() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    run_section = readme.split("uv run docinsights-ocr run \\", 1)[1]
    assert run_section.count("--documents-root data/raw/docsem") >= 2
    assert run_section.count("--timeout-seconds 120") == 2
