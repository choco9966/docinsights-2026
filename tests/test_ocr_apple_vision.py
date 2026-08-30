import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from docinsights_ocr.apple_vision import AppleVisionEngine, parse_apple_vision_json
from docinsights_ocr.blocks import reconstruct_blocks
from docinsights_ocr.cli import build_parser


def _output() -> str:
    return (
        json.dumps(
            {
                "schema_version": "1.0",
                "image_path": "/private/tmp/page.png",
                "width": 1000,
                "height": 2000,
                "elapsed_ms": 12.5,
                "observations": [
                    {
                        "text": "b01: Revenue $100",
                        "confidence": 0.875,
                        "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.05},
                    }
                ],
            }
        )
        + "\n"
    )


def test_parse_apple_vision_converts_normalized_bbox_to_pixel_top_left() -> None:
    page = parse_apple_vision_json(_output(), page_number=2)

    assert page.number == 2
    assert page.width == 1000 and page.height == 2000
    assert page.lines[0].bbox is not None
    assert page.lines[0].bbox.left == 100
    assert page.lines[0].bbox.top == 400
    assert page.lines[0].bbox.width == 300
    assert page.lines[0].bbox.height == 100
    assert page.lines[0].confidence == 0.875


def test_parse_apple_vision_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        parse_apple_vision_json(_output().replace('"1.0"', '"2.0"', 1))


def test_task_000909_training_watermark_does_not_contaminate_b23_sequence() -> None:
    payload = {
        "schema_version": "1.0",
        "image_path": "/tmp/task_000909.png",
        "width": 1000,
        "height": 2400,
        "elapsed_ms": 1,
        "observations": [
            {
                "text": "b23",
                "confidence": 0.99,
                "bbox": {"x": 0.05, "y": 0.4, "width": 0.05, "height": 0.01},
            },
            {
                "text": "Revenue was $100",
                "confidence": 0.98,
                "bbox": {"x": 0.12, "y": 0.402, "width": 0.3, "height": 0.01},
            },
            {
                "text": "TRAINING COP",
                "confidence": 0.8,
                "bbox": {"x": 0.45, "y": 0.405, "width": 0.2, "height": 0.24},
            },
            {
                "text": "b24 Next block",
                "confidence": 0.97,
                "bbox": {"x": 0.05, "y": 0.7, "width": 0.3, "height": 0.01},
            },
        ],
    }

    page = parse_apple_vision_json(json.dumps(payload))
    blocks = reconstruct_blocks((page,))

    assert [line.text for line in page.lines] == ["b23", "Revenue was $100", "b24 Next block"]
    assert [(block.block_id, block.text) for block in blocks] == [
        ("b23", "Revenue was $100"),
        ("b24", "Next block"),
    ]


def test_normal_sized_training_copy_body_text_is_preserved() -> None:
    payload = json.loads(_output())
    payload["observations"][0]["text"] = "TRAINING COPY"
    payload["observations"][0]["bbox"]["height"] = 0.01

    page = parse_apple_vision_json(json.dumps(payload))

    assert [line.text for line in page.lines] == ["TRAINING COPY"]


def test_tall_candidate_height_cannot_expand_body_row_tolerance() -> None:
    payload = {
        "schema_version": "1.0",
        "width": 1000,
        "height": 2400,
        "observations": [
            {
                "text": "b23",
                "confidence": 0.9,
                "bbox": {"x": 0.05, "y": 0.4, "width": 0.05, "height": 0.01},
            },
            {
                "text": "Revenue",
                "confidence": 0.9,
                "bbox": {"x": 0.15, "y": 0.402, "width": 0.2, "height": 0.01},
            },
            {
                "text": "ROTATED NOTE",
                "confidence": 0.8,
                "bbox": {"x": 0.01, "y": 0.405, "width": 0.2, "height": 0.24},
            },
        ],
    }

    page = parse_apple_vision_json(json.dumps(payload))

    assert [line.text for line in page.lines][:2] == ["b23", "Revenue"]


def test_parse_apple_vision_clusters_table_cells_into_reading_order() -> None:
    payload = json.loads(_output())
    payload["observations"] = [
        {
            "text": "Next check-in",
            "confidence": 1.0,
            "bbox": {"x": 0.7, "y": 0.20, "width": 0.2, "height": 0.012},
        },
        {
            "text": "On-call staff",
            "confidence": 1.0,
            "bbox": {"x": 0.4, "y": 0.20, "width": 0.2, "height": 0.012},
        },
        {
            "text": "Team",
            "confidence": 1.0,
            "bbox": {"x": 0.1, "y": 0.202, "width": 0.1, "height": 0.010},
        },
    ]

    page = parse_apple_vision_json(json.dumps(payload))

    assert [line.text for line in page.lines] == ["Team", "On-call staff", "Next check-in"]


def test_engine_invokes_configured_executable_with_resolved_image(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.touch()
    completed = subprocess.CompletedProcess([], 0, stdout=_output(), stderr="")

    with patch("docinsights_ocr.apple_vision.subprocess.run", return_value=completed) as mocked:
        page = AppleVisionEngine(
            executable="/opt/apple-ocr", mode="fast", timeout_seconds=7
        ).recognize(image)

    assert mocked.call_args.args[0] == [
        "/opt/apple-ocr",
        "--mode",
        "fast",
        "--language",
        "en-US",
        str(image.resolve()),
    ]
    assert page.lines[0].text == "b01: Revenue $100"
    assert mocked.call_args.kwargs["timeout"] == 7


def test_cli_selects_apple_vision_and_executable() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "in.jsonl",
            "out.jsonl",
            "--engine",
            "apple-vision",
            "--apple-vision-executable",
            "/opt/apple-ocr",
        ]
    )
    assert args.engine == "apple-vision"
    assert args.apple_vision_executable == "/opt/apple-ocr"
    assert args.timeout_seconds == 120.0
