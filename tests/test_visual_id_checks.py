import copy
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from docinsights_analysis import visual_id_checks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    image_path = tmp_path / "page-0001.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    record = {
        "instance_id": "item-1",
        "question": "SECRET QUERY: compute the total",
        "answer": "SECRET ANSWER 7",
        "solution": {
            "summary": "SECRET RATIONALE: add the inputs",
            "calculations": [],
        },
        "evidence": ["VIS-7!"],
        "evidence_details": [{"id": "VIS-7!", "page": 1, "quote": "SECRET QUOTE: three plus four"}],
        "uncertainties": ["SECRET DIAGNOSTIC: OCR ID differs"],
    }
    pages = [
        {
            "page_number": 1,
            "text": "OCR-7?:\nSECRET QUOTE: three plus four\nOTHER: distractor",
        }
    ]
    geometry = [
        {
            "page_number": 1,
            "width": 200,
            "height": 100,
            "lines": [
                {
                    "line_index": 0,
                    "text": "OCR-7?:",
                    "bbox": {"left": 10, "top": 10, "width": 40, "height": 12},
                },
                {
                    "line_index": 1,
                    "text": "SECRET QUOTE: three plus four",
                    "bbox": {"left": 10, "top": 28, "width": 150, "height": 12},
                },
            ],
        }
    ]
    retained_images = [{"page_number": 1, "path": str(image_path), "sha256": _sha256(image_path)}]
    return record, pages, geometry, retained_images


def _candidate() -> dict:
    return {
        "id": "VIS-7!",
        "ocr_id": "OCR-7?",
        "page": 1,
        "quote": "SECRET QUOTE: three plus four",
        "heading_line_index": 0,
    }


def _successful_invoke(captured: dict):
    def invoke(**request):
        captured.update(request)
        invocation_dir = request["output_dir"]
        raw_path = invocation_dir / "response.json"
        events_path = invocation_dir / "events.jsonl"
        response = {
            "visible_headings": [
                {
                    "id": "VIS-7!",
                    "bbox": {"left": 11, "top": 10, "right": 49, "bottom": 22},
                    "legibility": "clear",
                }
            ]
        }
        raw_path.write_text(json.dumps(response), encoding="utf-8")
        events_path.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        return {
            "response": response,
            "raw_response_path": raw_path,
            "artifact_paths": [raw_path, events_path],
            "model": "checker-model",
            "method": "locked-vlm-v1",
        }

    return invoke


def test_clear_matching_visual_heading_creates_bound_proof_without_leaking_candidate(
    tmp_path: Path,
) -> None:
    record, pages, geometry, images = _inputs(tmp_path)
    original_record = copy.deepcopy(record)
    original_pages = copy.deepcopy(pages)
    captured: dict = {}

    proofs = visual_id_checks.run_visual_id_checks(
        frozen_record=record,
        pages=pages,
        paddle_geometry=geometry,
        retained_images=images,
        output_dir=tmp_path / "checks",
        checker_config={"crop_padding_pixels": 4, "model": "checker-model"},
        invoke=_successful_invoke(captured),
    )

    assert len(proofs) == 1
    proof = proofs[0]
    assert proof["id"] == proof["visible_id"] == "VIS-7!"
    assert proof["ocr_id"] == "OCR-7?"
    assert proof["clear"] is True
    assert proof["heading_bbox"] == {"left": 10, "top": 10, "right": 50, "bottom": 22}
    assert proof["crop_bbox"] == {"left": 6, "top": 6, "right": 54, "bottom": 26}
    assert proof["quote_sha256"] == hashlib.sha256(b"SECRET QUOTE: three plus four").hexdigest()
    assert proof["page_image_sha256"] == images[0]["sha256"]
    crop_path = Path(proof["crop_artifact"]["path"])
    assert crop_path.stat().st_mode & 0o777 == 0o600
    assert Image.open(crop_path).size == (48, 20)
    assert proof["crop_sha256"] == _sha256(crop_path)
    assert proof["raw_response_sha256"] == _sha256(Path(captured["output_dir"]) / "response.json")
    assert all(_sha256(Path(item["path"])) == item["sha256"] for item in proof["artifacts"])
    assert proof["model"] == "checker-model"
    assert proof["method"] == "locked-vlm-v1"

    prompt = captured["prompt"]
    for forbidden in (
        record["question"],
        record["answer"],
        record["solution"]["summary"],
        record["evidence_details"][0]["quote"],
        record["evidence"][0],
        "OCR-7?",
        record["uncertainties"][0],
    ):
        assert forbidden not in prompt
    assert captured["images"] == (Path(images[0]["path"]), crop_path)
    assert captured["output_schema"] == visual_id_checks.VISUAL_CHECK_SCHEMA
    assert record == original_record
    assert pages == original_pages


def test_no_candidate_does_not_invoke_visual_checker(
    tmp_path: Path,
) -> None:
    record, pages, geometry, images = _inputs(tmp_path)
    record["evidence"] = ["OCR-7?"]
    record["evidence_details"][0]["id"] = "OCR-7?"

    def forbidden_invoke(**_request):
        raise AssertionError("visual checker must not run without a contract-approved candidate")

    assert (
        visual_id_checks.run_visual_id_checks(
            frozen_record=record,
            pages=pages,
            paddle_geometry=geometry,
            retained_images=images,
            output_dir=tmp_path / "checks",
            checker_config={"crop_padding_pixels": 4},
            invoke=forbidden_invoke,
        )
        == []
    )
    assert not (tmp_path / "checks").exists()


@pytest.mark.parametrize(
    "headings, message",
    [
        (
            [
                {
                    "id": "VIS-7!",
                    "bbox": {"left": 11, "top": 10, "right": 49, "bottom": 22},
                    "legibility": "ambiguous",
                }
            ],
            "single clear heading",
        ),
        (
            [
                {
                    "id": "DIFFERENT",
                    "bbox": {"left": 11, "top": 10, "right": 49, "bottom": 22},
                    "legibility": "clear",
                }
            ],
            "does not exactly match",
        ),
        (
            [
                {
                    "id": "VIS-7!",
                    "bbox": {"left": 11, "top": 10, "right": 49, "bottom": 22},
                    "legibility": "clear",
                },
                {
                    "id": "ALSO",
                    "bbox": {"left": 12, "top": 11, "right": 48, "bottom": 21},
                    "legibility": "clear",
                },
            ],
            "uniquely overlap",
        ),
    ],
)
def test_ambiguous_or_different_heading_fails_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headings: list[dict],
    message: str,
) -> None:
    record, pages, geometry, images = _inputs(tmp_path)
    monkeypatch.setattr(visual_id_checks, "visual_evidence_candidates", lambda *_: [_candidate()])
    calls = 0

    def invoke(**request):
        nonlocal calls
        calls += 1
        raw_path = request["output_dir"] / "response.json"
        response = {"visible_headings": headings}
        raw_path.write_text(json.dumps(response), encoding="utf-8")
        return {
            "response": response,
            "raw_response_path": raw_path,
            "artifact_paths": [raw_path],
            "model": "checker-model",
            "method": "locked-vlm-v1",
        }

    with pytest.raises(visual_id_checks.VisualIDCheckError, match=message):
        visual_id_checks.run_visual_id_checks(
            frozen_record=record,
            pages=pages,
            paddle_geometry=geometry,
            retained_images=images,
            output_dir=tmp_path / "checks",
            checker_config={"crop_padding_pixels": 4},
            invoke=invoke,
        )
    assert calls == 1


@pytest.mark.parametrize("fault", ["image_hash", "bbox", "raw_hash"])
def test_visual_proof_fails_closed_on_hash_or_bbox_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    record, pages, geometry, images = _inputs(tmp_path)
    monkeypatch.setattr(visual_id_checks, "visual_evidence_candidates", lambda *_: [_candidate()])
    if fault == "image_hash":
        images[0]["sha256"] = "0" * 64
    elif fault == "bbox":
        geometry[0]["lines"][0]["bbox"]["width"] = 500

    captured: dict = {}
    invoke = _successful_invoke(captured)
    if fault == "raw_hash":
        original = invoke

        def invoke(**request):
            result = original(**request)
            Path(result["raw_response_path"]).write_text("tampered", encoding="utf-8")
            result["response"] = {
                "visible_headings": [
                    {
                        "id": "VIS-7!",
                        "bbox": {"left": 11, "top": 10, "right": 49, "bottom": 22},
                        "legibility": "clear",
                    }
                ]
            }
            return result

    with pytest.raises(visual_id_checks.VisualIDCheckError):
        visual_id_checks.run_visual_id_checks(
            frozen_record=record,
            pages=pages,
            paddle_geometry=geometry,
            retained_images=images,
            output_dir=tmp_path / "checks",
            checker_config={"crop_padding_pixels": 4},
            invoke=invoke,
        )
