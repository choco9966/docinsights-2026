import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from docinsights_analysis import visual_id_checks
from docinsights_analysis.solution_records import (
    finalize_solution,
    input_digest,
    validate_primary_solution,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, list[dict], list[dict], Path, dict]:
    rendered_page = tmp_path / "expected-page.png"
    Image.new("RGB", (240, 300), "white").save(rendered_page)
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"public synthetic pdf")
    renderer_path = tmp_path / "pdftoppm"
    renderer_path.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "if '-v' in sys.argv:\n"
        "    print('fake-pdftoppm 1.0', file=sys.stderr)\n"
        "    raise SystemExit(0)\n"
        f"source = {str(rendered_page)!r}\n"
        "page = sys.argv[sys.argv.index('-f') + 1]\n"
        "shutil.copyfile(source, sys.argv[-1] + '-' + page + '.png')\n",
        encoding="utf-8",
    )
    renderer_path.chmod(0o700)
    checker_path = tmp_path / "codex"
    checker_path.write_bytes(b"synthetic checker executable")
    checker_path.chmod(0o700)

    record = {
        "instance_id": "item-1",
        "split": "heldout",
        "question": "SECRET QUERY",
        "answer": "42",
        "solution": {"summary": "SECRET SOLUTION", "calculations": []},
        "evidence_regions": [{"page": 1, "ocr_anchor": "UNIQUE LOCATOR"}],
        "source_pages": [{"page_number": 1, "text": "CORRUPTED OCR BODY\nUNIQUE LOCATOR"}],
        "provenance": {
            "pdf_sha256": _sha256(pdf_path),
            "input_sha256": "1" * 64,
            "config_sha256": "3" * 64,
            "output_sha256": "2" * 64,
            "model": "primary-model",
            "method": "primary-method",
        },
        "uncertainties": ["SECRET PRIMARY UNCERTAINTY"],
    }
    pages = [{"page_number": 1, "text": "CORRUPTED OCR BODY\nUNIQUE LOCATOR"}]
    geometry = [
        {
            "page_number": 1,
            "width": 240,
            "height": 300,
            "ocr_image_sha256": _sha256(rendered_page),
            "lines": [
                {
                    "line_index": 0,
                    "text": "CORRUPTED OCR BODY",
                    "bbox": {"left": 20, "top": 50, "width": 140, "height": 12},
                },
                {
                    "line_index": 1,
                    "text": "UNIQUE LOCATOR",
                    "bbox": {"left": 30, "top": 120, "width": 100, "height": 12},
                },
            ],
        }
    ]
    config = {
        "model": "checker-model",
        "method": "locked-source-region-v1",
        "renderer_path": str(renderer_path),
        "renderer_sha256": _sha256(renderer_path),
        "renderer_version": "fake-pdftoppm 1.0",
        "dpi": 175,
        "ocr_input_format": "png",
        "context_vertical_padding_pixels": 30,
        "anchor_padding_pixels": 4,
        "codex_executable_path": str(checker_path),
        "codex_executable_sha256": _sha256(checker_path),
    }
    return record, pages, geometry, pdf_path, config


def _invoke_with(response: dict, captured: dict, *, tamper_raw: bool = False):
    def invoke(**request):
        captured.update(request)
        raw_path = request["output_dir"] / "response.json"
        events_path = request["output_dir"] / "events.jsonl"
        raw_path.write_text(json.dumps(response), encoding="utf-8")
        events_path.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        result = {
            "response": response,
            "raw_response_path": raw_path,
            "artifact_paths": [raw_path, events_path],
            "model": "checker-model",
            "method": "locked-source-region-v1",
        }
        if tamper_raw:
            raw_path.write_text("{}", encoding="utf-8")
        return result

    return invoke


def _clear_response(heading: str = "A:B-7: ", body: str = "visible source body") -> dict:
    return {
        "observed_blocks": [
            {
                "heading_line": heading,
                "body_text": body,
                "heading_legibility": "clear",
                "body_legibility": "clear",
                "contains_anchor_region": True,
            }
        ]
    }


def _run(tmp_path: Path, response: dict):
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    captured: dict = {}
    proofs = visual_id_checks.check_source_regions(
        frozen_record=record,
        pages=pages,
        ocr_geometry=geometry,
        source_pdf_path=pdf_path,
        output_dir=tmp_path / "checks",
        checker_config=config,
        invoke=_invoke_with(response, captured),
    )
    return proofs, captured, record


def test_clear_anchor_block_creates_pixel_derived_evidence_and_blind_three_image_request(
    tmp_path: Path,
) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    original = copy.deepcopy(record)
    captured: dict = {}

    proofs = visual_id_checks.check_source_regions(
        frozen_record=record,
        pages=pages,
        ocr_geometry=geometry,
        source_pdf_path=pdf_path,
        output_dir=tmp_path / "checks",
        checker_config=config,
        invoke=_invoke_with(_clear_response(), captured),
    )

    assert record == original
    assert len(proofs) == 1
    proof = proofs[0]
    assert proof["status"] == "fully_grounded"
    assert proof["evidence"] == {
        "id": "A:B-7",
        "page": 1,
        "quote": "visible source body",
    }
    assert proof["source_uncertainties"] == []
    assert proof["primary_response_sha256"] == record["provenance"]["output_sha256"]
    assert proof["ocr_image_sha256"] == proof["page_image_sha256"]
    assert proof["anchor_bbox"] == {"left": 30, "top": 120, "right": 130, "bottom": 132}
    assert proof["context_bbox"] == {"left": 0, "top": 90, "right": 240, "bottom": 162}
    assert proof["anchor_crop_bbox"] == {"left": 26, "top": 116, "right": 134, "bottom": 136}
    assert len(captured["images"]) == 3
    page_path, context_path, anchor_path = captured["images"]
    assert page_path.suffix == context_path.suffix == anchor_path.suffix == ".png"
    assert Image.open(page_path).size == (240, 300)
    assert Image.open(context_path).size == (240, 72)
    assert Image.open(anchor_path).size == (108, 20)
    assert proof["page_image_sha256"] == _sha256(page_path)
    assert proof["context_crop_sha256"] == _sha256(context_path)
    assert proof["anchor_crop_sha256"] == _sha256(anchor_path)
    assert all(_sha256(Path(item["path"])) == item["sha256"] for item in proof["artifacts"])
    assert {item["kind"] for item in proof["artifacts"]} >= {
        "source_pdf",
        "renderer_executable",
        "page_image",
        "context_crop",
        "anchor_crop",
        "checker_executable",
        "render_command",
        "raw_response",
    }
    selector_artifact = next(
        item for item in proof["artifacts"] if item["kind"] == "selector_config"
    )
    selector = json.loads(Path(selector_artifact["path"]).read_text(encoding="utf-8"))
    assert selector["line_indices"] == [1]
    assert selector["anchor_bbox"] == proof["anchor_bbox"]
    assert proof["selector_sha256"] == selector_artifact["sha256"]

    prompt = captured["prompt"]
    for forbidden in (
        record["question"],
        record["answer"],
        record["solution"]["summary"],
        record["evidence_regions"][0]["ocr_anchor"],
        record["uncertainties"][0],
        "A:B-7",
        "visible source body",
        "CORRUPTED OCR BODY",
    ):
        assert forbidden not in prompt
    assert captured["output_schema"] == visual_id_checks.SOURCE_CHECK_SCHEMA


def test_primary_guessed_evidence_is_irrelevant_to_pixel_evidence(tmp_path: Path) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    record["answer"] = "WRONG-GUESS"
    record["solution"]["summary"] = "WRONG EXPECTED QUOTE"
    captured: dict = {}

    proof = visual_id_checks.check_source_regions(
        frozen_record=record,
        pages=pages,
        ocr_geometry=geometry,
        source_pdf_path=pdf_path,
        output_dir=tmp_path / "checks",
        checker_config=config,
        invoke=_invoke_with(_clear_response("PIXEL-9: ", "actual pixels"), captured),
    )[0]

    assert proof["evidence"] == {
        "id": "PIXEL-9",
        "page": 1,
        "quote": "actual pixels",
    }
    assert "WRONG-GUESS" not in captured["prompt"]
    assert "WRONG EXPECTED QUOTE" not in captured["prompt"]


def test_multiline_anchor_uses_ordered_line_union_without_solver_coordinates(
    tmp_path: Path,
) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    record["evidence_regions"][0]["ocr_anchor"] = "CORRUPTED OCR BODY\nUNIQUE LOCATOR"

    proof = visual_id_checks.check_source_regions(
        frozen_record=record,
        pages=pages,
        ocr_geometry=geometry,
        source_pdf_path=pdf_path,
        output_dir=tmp_path / "checks",
        checker_config=config,
        invoke=_invoke_with(_clear_response(), {}),
    )[0]

    assert proof["anchor_bbox"] == {"left": 20, "top": 50, "right": 160, "bottom": 132}
    selector_artifact = next(
        item for item in proof["artifacts"] if item["kind"] == "selector_config"
    )
    selector = json.loads(Path(selector_artifact["path"]).read_text(encoding="utf-8"))
    assert selector["line_indices"] == [0, 1]


def test_source_check_matches_final_record_contract(tmp_path: Path) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    task = {
        "instance_id": "item-1",
        "user_query": "SECRET QUERY",
        "document_pdf": "source.pdf",
    }
    record["provenance"]["input_sha256"] = input_digest(task, pages)
    primary = validate_primary_solution(record, task, pages)

    proofs = visual_id_checks.check_source_regions(
        frozen_record=primary,
        pages=pages,
        ocr_geometry=geometry,
        source_pdf_path=pdf_path,
        output_dir=tmp_path / "checks",
        checker_config=config,
        invoke=_invoke_with(_clear_response(), {}),
    )
    final = finalize_solution(primary, task, pages, source_checks=proofs)

    assert final["source_status"] == "fully_grounded"
    assert final["evidence"] == ["A:B-7"]
    assert final["evidence_details"] == [{"id": "A:B-7", "page": 1, "quote": "visible source body"}]


@pytest.mark.parametrize(
    "response, expected_quote, uncertainty_fragment",
    [
        (
            {
                "observed_blocks": [
                    {
                        "heading_line": "MAYBE-0: ",
                        "body_text": "readable body",
                        "heading_legibility": "ambiguous",
                        "body_legibility": "clear",
                        "contains_anchor_region": True,
                    }
                ]
            },
            "readable body",
            "heading",
        ),
        (
            {
                "observed_blocks": [
                    {
                        "heading_line": None,
                        "body_text": "readable body only",
                        "heading_legibility": "unreadable",
                        "body_legibility": "clear",
                        "contains_anchor_region": True,
                    }
                ]
            },
            "readable body only",
            "heading",
        ),
        (
            {
                "observed_blocks": [
                    {
                        "heading_line": "NO DELIMITER",
                        "body_text": "readable body only",
                        "heading_legibility": "clear",
                        "body_legibility": "clear",
                        "contains_anchor_region": True,
                    }
                ]
            },
            "readable body only",
            "delimiter",
        ),
    ],
)
def test_ambiguous_unreadable_or_missing_heading_returns_terminal_unresolved(
    tmp_path: Path,
    response: dict,
    expected_quote: str,
    uncertainty_fragment: str,
) -> None:
    proofs, _, _ = _run(tmp_path, response)
    proof = proofs[0]
    assert proof["status"] == "evidence_unresolved"
    assert proof["evidence"] == {"id": None, "page": 1, "quote": expected_quote}
    assert proof["source_uncertainties"]
    assert uncertainty_fragment in " ".join(proof["source_uncertainties"]).lower()


def test_only_colon_followed_by_whitespace_is_body_delimiter(tmp_path: Path) -> None:
    proofs, _, _ = _run(tmp_path, _clear_response("SEC:A-4: ", "literal body"))
    assert proofs[0]["evidence"]["id"] == "SEC:A-4"
    assert proofs[0]["evidence"]["quote"].startswith("literal body")


@pytest.mark.parametrize(
    "fault", ["ocr_body", "missing_anchor", "page_hash", "pdf_hash", "raw_hash"]
)
def test_corrupt_locator_or_hash_drift_fails_closed(tmp_path: Path, fault: str) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    captured: dict = {}
    invoke = _invoke_with(_clear_response(), captured, tamper_raw=fault == "raw_hash")
    if fault == "ocr_body":
        geometry[0]["lines"][1]["text"] = "CORRUPTED GEOMETRY"
    elif fault == "missing_anchor":
        record["evidence_regions"][0]["ocr_anchor"] = "ABSENT LOCATOR"
    elif fault == "page_hash":
        geometry[0]["ocr_image_sha256"] = "0" * 64
    elif fault == "pdf_hash":
        record["provenance"]["pdf_sha256"] = "0" * 64

    with pytest.raises(visual_id_checks.SourceRegionCheckError):
        visual_id_checks.check_source_regions(
            frozen_record=record,
            pages=pages,
            ocr_geometry=geometry,
            source_pdf_path=pdf_path,
            output_dir=tmp_path / "checks",
            checker_config=config,
            invoke=invoke,
        )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "primary-record.json",
        "prompt.txt",
        "schema.json",
        "checker-config.json",
        "selector-config.json",
    ],
)
def test_checker_callback_cannot_mutate_frozen_contract_artifacts(
    tmp_path: Path, artifact_name: str
) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    base_invoke = _invoke_with(_clear_response(), {})

    def invoke(**request):
        result = base_invoke(**request)
        (request["output_dir"].parent / artifact_name).write_text("tampered")
        return result

    with pytest.raises(visual_id_checks.SourceRegionCheckError, match="changed during"):
        visual_id_checks.check_source_regions(
            frozen_record=record,
            pages=pages,
            ocr_geometry=geometry,
            source_pdf_path=pdf_path,
            output_dir=tmp_path / "checks",
            checker_config=config,
            invoke=invoke,
        )


@pytest.mark.parametrize("target", ["primary", "config", "schema", "prompt"])
def test_checker_callback_cannot_mutate_in_memory_contract(tmp_path: Path, target: str) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    base_invoke = _invoke_with(_clear_response(), {})
    original_schema = copy.deepcopy(visual_id_checks.SOURCE_CHECK_SCHEMA)
    original_prompt = visual_id_checks._BLIND_PROMPT

    def invoke(**request):
        result = base_invoke(**request)
        if target == "primary":
            record["answer"] = "999"
        elif target == "config":
            request["checker_config"]["model"] = "mutated"
        elif target == "schema":
            visual_id_checks.SOURCE_CHECK_SCHEMA["mutated"] = True
        else:
            visual_id_checks._BLIND_PROMPT = "mutated"
        return result

    try:
        with pytest.raises(visual_id_checks.SourceRegionCheckError, match="changed during"):
            visual_id_checks.check_source_regions(
                frozen_record=record,
                pages=pages,
                ocr_geometry=geometry,
                source_pdf_path=pdf_path,
                output_dir=tmp_path / "checks",
                checker_config=config,
                invoke=invoke,
            )
    finally:
        visual_id_checks.SOURCE_CHECK_SCHEMA.clear()
        visual_id_checks.SOURCE_CHECK_SCHEMA.update(original_schema)
        visual_id_checks._BLIND_PROMPT = original_prompt


def test_anchor_must_be_unique_before_checker_invocation(tmp_path: Path) -> None:
    record, pages, geometry, pdf_path, config = _fixture(tmp_path)
    pages[0]["text"] += "\nUNIQUE LOCATOR"
    calls = 0

    def invoke(**_request):
        nonlocal calls
        calls += 1
        raise AssertionError("checker must not run for an ambiguous OCR locator")

    with pytest.raises(visual_id_checks.SourceRegionCheckError, match="exactly once"):
        visual_id_checks.check_source_regions(
            frozen_record=record,
            pages=pages,
            ocr_geometry=geometry,
            source_pdf_path=pdf_path,
            output_dir=tmp_path / "checks",
            checker_config=config,
            invoke=invoke,
        )
    assert calls == 0
