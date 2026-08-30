import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from docinsights_ocr.tesseract import TesseractEngine, parse_tsv

TSV = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext\n"
    "1\t1\t0\t0\t0\t0\t0\t0\t1200\t1800\t-1\t\n"
    "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t90\tb01\n"
    "5\t1\t1\t1\t1\t2\t45\t20\t40\t10\t80\tRevenue\n"
    "5\t1\t1\t1\t2\t1\t10\t40\t20\t10\t70\t100\n"
)


def test_parse_tsv_groups_words_into_lines() -> None:
    page = parse_tsv(TSV, page_number=4)

    assert page.number == 4
    assert [line.text for line in page.lines] == ["b01 Revenue", "100"]
    assert page.lines[0].confidence == pytest.approx(0.85)
    assert page.lines[0].bbox is not None
    assert page.lines[0].bbox.width == 75
    assert page.width == 1200
    assert page.height == 1800


def test_parse_tsv_rejects_invalid_header() -> None:
    with pytest.raises(ValueError, match="header"):
        parse_tsv("text\nhello\n")


def test_parse_tsv_clamps_negative_word_confidence() -> None:
    page = parse_tsv(TSV.replace("\t90\tb01", "\t-1\tb01"))
    assert page.lines[0].confidence == 0.4


def test_engine_uses_resolved_image_path_and_configured_psm(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    image = real_dir / "page.png"
    image.touch()
    link = tmp_path / "linked.png"
    link.symlink_to(image)
    completed = subprocess.CompletedProcess([], 0, stdout=TSV, stderr="")

    with patch("docinsights_ocr.tesseract.subprocess.run", return_value=completed) as mocked:
        page = TesseractEngine(page_segmentation_mode=3, timeout_seconds=9).recognize(link)

    command = mocked.call_args.args[0]
    assert command[1] == str(image.resolve())
    assert command[command.index("--psm") + 1] == "3"
    assert mocked.call_args.kwargs["timeout"] == 9
    assert page.lines[0].text == "b01 Revenue"
