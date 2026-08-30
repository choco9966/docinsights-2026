import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docinsights_ocr.paddle_ocr import PaddleOCREngine, parse_paddle_result


def _png(path: Path, *, width: int = 1000, height: int = 2000) -> Path:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height)
    )
    return path


def _payload() -> dict[str, object]:
    return {
        "res": {
            "rec_texts": ["b01: Revenue $100", "continued line"],
            "rec_scores": [0.91, 0.82],
            "rec_boxes": [[10, 20, 210, 70], [10, 80, 310, 130]],
        }
    }


def test_parse_paddle_result_converts_lines_and_pixel_boxes(tmp_path: Path) -> None:
    image = _png(tmp_path / "page.png")

    page = parse_paddle_result(_payload(), page_number=2, image_path=image)

    assert page.number == 2
    assert page.width == 1000 and page.height == 2000
    assert [line.text for line in page.lines] == ["b01: Revenue $100", "continued line"]
    assert page.lines[0].confidence == 0.91
    assert page.lines[0].bbox is not None
    assert page.lines[0].bbox.width == 200
    assert page.lines[0].bbox.height == 50


def test_parse_paddle_result_accepts_json_property_and_filters_large_watermark(
    tmp_path: Path,
) -> None:
    image = _png(tmp_path / "page.png")
    payload = _payload()
    payload["res"]["rec_texts"].append("TRAINING COPY")  # type: ignore[index,union-attr]
    payload["res"]["rec_scores"].append(0.99)  # type: ignore[index,union-attr]
    payload["res"]["rec_boxes"].append([0, 300, 900, 700])  # type: ignore[index,union-attr]
    result = SimpleNamespace(json=json.dumps(payload))

    page = parse_paddle_result(result, image_path=image)

    assert [line.text for line in page.lines] == ["b01: Revenue $100", "continued line"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rec_scores", [1.1, 0.8], "confidence"),
        ("rec_boxes", [[10, 20, 2000, 70], [10, 80, 310, 130]], "beyond"),
        ("rec_texts", ["only one"], "counts differ"),
    ],
)
def test_parse_paddle_result_rejects_invalid_output(
    tmp_path: Path, field: str, value: list[object], message: str
) -> None:
    image = _png(tmp_path / "page.png")
    payload = _payload()
    payload["res"][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        parse_paddle_result(payload, image_path=image)


def test_engine_uses_pinned_local_models_and_keeps_pipeline_warm(tmp_path: Path) -> None:
    detector = tmp_path / "detector"
    recognizer = tmp_path / "recognizer"
    for directory in (detector, recognizer):
        directory.mkdir()
        (directory / "inference.pdiparams").write_bytes(b"weights")
        (directory / "inference.yml").write_text("model: test\n")
    image = _png(tmp_path / "page.png")
    calls: list[dict[str, object]] = []

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def predict(self, path: str) -> list[dict[str, object]]:
            assert path == str(image.resolve())
            return [_payload()]

    fake_module = SimpleNamespace(PaddleOCR=FakePipeline)
    with (
        patch.dict(sys.modules, {"paddleocr": fake_module}),
        patch("docinsights_ocr.paddle_ocr._package_version", return_value="test-version"),
    ):
        engine = PaddleOCREngine(
            detection_model_dir=detector,
            recognition_model_dir=recognizer,
        )
        first = engine.recognize(image)
        second = engine.recognize(image)

    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"
    assert calls[0]["enable_mkldnn"] is False
    assert first.lines == second.lines
    assert engine.executable_identity["kind"] == "python_packages_and_model_trees"
    assert len(engine.executable_identity["sha256"]) == 64
