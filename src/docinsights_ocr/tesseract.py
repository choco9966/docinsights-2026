"""Tesseract TSV subprocess adapter and pure TSV parser."""

from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import BoundingBox, Line, Page


@dataclass(frozen=True, slots=True)
class _Word:
    key: tuple[int, int, int, int]
    text: str
    confidence: float
    bbox: BoundingBox


def parse_tsv(tsv: str, *, page_number: int = 1) -> Page:
    """Parse Tesseract TSV words and reconstruct lines in source order."""
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t")
    required = {
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("invalid Tesseract TSV header")
    groups: dict[tuple[int, int, int, int], list[_Word]] = {}
    page_width: int | None = None
    page_height: int | None = None
    for row in reader:
        if row["level"] == "1":
            try:
                width = int(row["width"])
                height = int(row["height"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid page dimensions in Tesseract TSV") from exc
            if width <= 0 or height <= 0:
                raise ValueError("Tesseract page dimensions must be positive")
            page_width = width
            page_height = height
            continue
        if row["level"] != "5" or not row["text"].strip():
            continue
        try:
            key = tuple(int(row[name]) for name in ("page_num", "block_num", "par_num", "line_num"))
            word = _Word(
                key=key,  # type: ignore[arg-type]
                text=row["text"].strip(),
                confidence=max(0.0, min(100.0, float(row["conf"]))) / 100.0,
                bbox=BoundingBox(
                    left=int(row["left"]),
                    top=int(row["top"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid numeric field in Tesseract TSV") from exc
        groups.setdefault(key, []).append(word)

    lines: list[Line] = []
    for source_order, words in enumerate(groups.values()):
        left = min(word.bbox.left for word in words)
        top = min(word.bbox.top for word in words)
        right = max(word.bbox.right for word in words)
        bottom = max(word.bbox.bottom for word in words)
        confidence = sum(word.confidence for word in words) / len(words)
        lines.append(
            Line(
                page_number=page_number,
                text=" ".join(word.text for word in words),
                bbox=BoundingBox(left, top, right - left, bottom - top),
                confidence=confidence,
                source_order=source_order,
            )
        )
    return Page(
        number=page_number,
        lines=tuple(lines),
        width=page_width,
        height=page_height,
    )


class TesseractEngine:
    """Run Tesseract as a subprocess, avoiding a Python package dependency."""

    def __init__(
        self,
        *,
        executable: str = "tesseract",
        language: str = "eng",
        dpi: int = 300,
        page_segmentation_mode: int = 6,
        timeout_seconds: float | None = None,
    ) -> None:
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        if page_segmentation_mode < 0:
            raise ValueError("page_segmentation_mode must be non-negative")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = executable
        self.language = language
        self.dpi = dpi
        self.page_segmentation_mode = page_segmentation_mode
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "tesseract-tsv"

    @property
    def confidence_kind(self) -> str:
        return "mean_word_confidence_0_to_1"

    def recognize(self, image_path: str | Path, *, page_number: int = 1) -> Page:
        # Homebrew Tesseract can reject /tmp symlinks on macOS; use the real path.
        resolved_image = Path(image_path).resolve()
        result = subprocess.run(
            [
                self.executable,
                str(resolved_image),
                "stdout",
                "-l",
                self.language,
                "--dpi",
                str(self.dpi),
                "--psm",
                str(self.page_segmentation_mode),
                "tsv",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        return parse_tsv(result.stdout, page_number=page_number)
