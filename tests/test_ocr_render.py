import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from docinsights_ocr.render import render_pdf


def test_render_pdf_invokes_poppler_with_deterministic_dpi_and_sorts_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "input.pdf"
    pdf.touch()

    def create_pages(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_prefix = Path(args[0][-1])  # type: ignore[index]
        output_prefix.with_name("page-10.png").touch()
        output_prefix.with_name("page-2.png").touch()
        return subprocess.CompletedProcess([], 0, "", "")

    with patch("docinsights_ocr.render.subprocess.run", side_effect=create_pages) as mocked:
        pages = render_pdf(pdf, tmp_path / "images", dpi=240)

    assert [path.name for path in pages] == ["page-2.png", "page-10.png"]
    assert mocked.call_args.args[0][1:4] == ["-png", "-r", "240"]


def test_render_pdf_passes_timeout_to_poppler(tmp_path: Path) -> None:
    pdf = tmp_path / "input.pdf"
    pdf.touch()

    with (
        patch("docinsights_ocr.render.subprocess.run", side_effect=RuntimeError) as mocked,
        pytest.raises(RuntimeError),
    ):
        render_pdf(pdf, tmp_path / "images", timeout_seconds=12.5)

    assert mocked.call_args.kwargs["timeout"] == 12.5
