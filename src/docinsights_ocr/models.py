"""Immutable data model shared by OCR engines and benchmark output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(self.left, self.top, self.width, self.height) < 0:
            raise ValueError("bounding-box values must be non-negative")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True, slots=True)
class Line:
    page_number: int
    text: str
    bbox: BoundingBox | None = None
    confidence: float | None = None
    source_order: int = 0

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be at least one")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class Page:
    number: int
    lines: tuple[Line, ...] = field(default_factory=tuple)
    image_path: Path | None = None
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("page number must be at least one")
        if any(line.page_number != self.number for line in self.lines):
            raise ValueError("all lines must belong to their containing page")


@dataclass(frozen=True, slots=True)
class Block:
    block_id: str
    text: str
    page_numbers: tuple[int, ...]
    lines: tuple[Line, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must not be empty")
        if not self.page_numbers:
            raise ValueError("a block must refer to at least one page")


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    pages: tuple[Page, ...]
    blocks: tuple[Block, ...] = field(default_factory=tuple)
    engine: str = "unknown"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if tuple(page.number for page in self.pages) != tuple(
            sorted(page.number for page in self.pages)
        ):
            raise ValueError("pages must be in ascending page order")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
