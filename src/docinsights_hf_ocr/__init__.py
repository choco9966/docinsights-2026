"""DocSem small-model OCR evaluation helpers."""

from .metrics import block_fidelity, cer, extract_blocks, is_valid_ocr, normalize_text, wer

__all__ = [
    "block_fidelity",
    "cer",
    "extract_blocks",
    "is_valid_ocr",
    "normalize_text",
    "wer",
]
