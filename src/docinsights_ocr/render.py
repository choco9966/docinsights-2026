"""PDF rendering through Poppler's command-line tools."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_PAGE_NUMBER = re.compile(r"-(\d+)\.png$")


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    executable: str = "pdftoppm",
    prefix: str = "page",
    timeout_seconds: float | None = None,
) -> tuple[Path, ...]:
    """Render a PDF to deterministically named PNG files in page order."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    source = Path(pdf_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_prefix = destination / prefix
    subprocess.run(
        [executable, "-png", "-r", str(dpi), str(source), str(output_prefix)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    pages = tuple(
        sorted(
            destination.glob(f"{prefix}-*.png"),
            key=_page_number,
        )
    )
    if not pages:
        raise RuntimeError(f"Poppler produced no pages for {source}")
    return pages


def _page_number(path: Path) -> int:
    match = _PAGE_NUMBER.search(path.name)
    if match is None:
        raise ValueError(f"unexpected rendered page name: {path.name}")
    return int(match.group(1))
