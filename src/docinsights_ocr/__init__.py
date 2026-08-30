"""Deterministic OCR benchmarking utilities for DocInsights documents."""

from .apple_vision import AppleVisionEngine, parse_apple_vision_json
from .blocks import reconstruct_blocks
from .models import Block, BoundingBox, Document, Line, Page
from .tesseract import TesseractEngine, parse_tsv

__all__ = [
    "Block",
    "BoundingBox",
    "Document",
    "Line",
    "Page",
    "TesseractEngine",
    "AppleVisionEngine",
    "parse_apple_vision_json",
    "parse_tsv",
    "reconstruct_blocks",
]
