"""Reconstruct DocSem blocks from OCR lines containing bNN markers."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from .models import Block, Line, Page

BLOCK_MARKER = re.compile(
    r"(?<![A-Za-z0-9])(?:[\[<(]\s*)?(b\d+)(?:\s*[\])>])?(?![A-Za-z0-9])(?:\s*[:.])?",
    re.IGNORECASE,
)
_OCR_ZERO_MARKER = re.compile(
    r"^(?P<prefix>\s*(?:[\[<(]\s*)?)(?P<b>[bB])(?P<letter>[oO])"
    r"(?P<leading_zero>0)?(?P<digit>[1-9])"
    r"(?P<suffix>(?:\s*[\])>])?\s*[:.])(?=\s|$)"
)
_OBVIOUS_ARTIFACT = re.compile(
    r"(?:"
    r"docsem\s*\|\s*training\s+copy"
    r"(?:\s*(?:[|·-]\s*)?page\s+\d+(?:\s+of\s+\d+)?)?"
    r"|page\s+\d+(?:\s+of\s+\d+)?"
    r")",
    re.IGNORECASE,
)


def reconstruct_blocks(pages: Iterable[Page]) -> tuple[Block, ...]:
    """Collect text following each bNN marker, including wrapped/page-spanning lines."""
    ordered_pages = list(filter_page_artifacts(pages))
    assembled: list[Block] = []
    current_id: str | None = None
    current_parts: list[str] = []
    current_lines: list[Line] = []
    current_pages: list[int] = []

    def flush() -> None:
        nonlocal current_id, current_parts, current_lines, current_pages
        if current_id is not None:
            assembled.append(
                Block(
                    block_id=current_id,
                    text=" ".join(part for part in current_parts if part).strip(),
                    page_numbers=tuple(dict.fromkeys(current_pages)),
                    lines=tuple(current_lines),
                )
            )
        current_id = None
        current_parts = []
        current_lines = []
        current_pages = []

    for page in ordered_pages:
        for line in sorted(page.lines, key=lambda item: item.source_order):
            marker_text = _normalize_line_start_marker(line.text)
            matches = list(BLOCK_MARKER.finditer(marker_text))
            if not matches:
                if current_id is not None:
                    current_parts.append(line.text.strip())
                    current_lines.append(line)
                    current_pages.append(page.number)
                continue
            for index, match in enumerate(matches):
                flush()
                current_id = match.group(1).lower()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(line.text)
                current_parts.append(line.text[match.end() : end].strip())
                current_lines.append(line)
                current_pages.append(page.number)
    flush()
    return tuple(assembled)


def filter_page_artifacts(
    pages: Iterable[Page],
    *,
    margin_ratio: float = 0.08,
    minimum_repetitions: int = 2,
) -> tuple[Page, ...]:
    """Remove safe page headers/footers without dropping body continuations."""
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError("margin_ratio must be between zero and 0.5")
    if minimum_repetitions < 2:
        raise ValueError("minimum_repetitions must be at least two")
    ordered = tuple(sorted(pages, key=lambda page: page.number))
    margin_occurrences: dict[str, set[int]] = defaultdict(set)
    for page in ordered:
        if page.height is None:
            continue
        for line in page.lines:
            if _is_margin_line(line, page.height, margin_ratio):
                margin_occurrences[_artifact_key(line.text)].add(page.number)
    repeated = {
        key
        for key, page_numbers in margin_occurrences.items()
        if key and len(page_numbers) >= minimum_repetitions
    }
    filtered: list[Page] = []
    for page in ordered:
        kept = tuple(
            line
            for line in page.lines
            if not _OBVIOUS_ARTIFACT.fullmatch(line.text.strip())
            and not (
                page.height is not None
                and _is_margin_line(line, page.height, margin_ratio)
                and _artifact_key(line.text) in repeated
            )
        )
        filtered.append(
            Page(
                number=page.number,
                lines=kept,
                image_path=page.image_path,
                width=page.width,
                height=page.height,
            )
        )
    return tuple(filtered)


def _is_margin_line(line: Line, page_height: int, margin_ratio: float) -> bool:
    if line.bbox is None:
        return False
    margin = page_height * margin_ratio
    return line.bbox.top <= margin or line.bbox.bottom >= page_height - margin


def _artifact_key(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return re.sub(r"\bpage\s+\d+(?:\s+of\s+\d+)?\b", "page <n>", normalized)


def _normalize_line_start_marker(text: str) -> str:
    match = _OCR_ZERO_MARKER.match(text)
    if match is None:
        return text
    if match.group("leading_zero") is not None:
        b_index = match.start("b")
        letter_index = match.start("letter")
        return f"{text[:b_index]} {text[b_index]}{text[letter_index + 1 :]}"
    index = match.start("letter")
    return f"{text[:index]}0{text[index + 1 :]}"
