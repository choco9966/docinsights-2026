"""Dependency-free text and document-block metrics."""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_BLOCK = re.compile(
    r"(?is)\bb(0[1-9]|1[0-9]|2[0-3])\s*:\s*(.*?)(?=\bb(?:0[1-9]|1[0-9]|2[0-3])\s*:|\Z)"
)
_TAG = re.compile(r"<[^>]+>")
_HORIZONTAL_SPACE = re.compile(r"[^\S\r\n]+")


def normalize_text(text: str) -> str:
    """Apply NFC and whitespace-only normalization without changing case or punctuation."""
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip(" ") for line in normalized.split("\n")]
    return "\n".join(lines).strip("\n")


def levenshtein(left: Sequence[object], right: Sequence[object]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_item in enumerate(right, start=1):
        current = [row]
        for column, left_item in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    reference_n = normalize_text(reference)
    hypothesis_n = normalize_text(hypothesis)
    if not reference_n:
        return 0.0 if not hypothesis_n else 1.0
    return levenshtein(reference_n, hypothesis_n) / len(reference_n)


def wer(reference: str, hypothesis: str) -> float:
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return levenshtein(reference_words, hypothesis_words) / len(reference_words)


def extract_blocks(text: str) -> list[tuple[str, str]]:
    """Extract ordered b01..b23 text from plain text, Markdown, or HTML-ish output."""
    cleaned = html.unescape(_TAG.sub(" ", text)).replace("<|user|>", "")
    return [
        (f"b{match.group(1)}", normalize_text(match.group(2))) for match in _BLOCK.finditer(cleaned)
    ]


def is_valid_ocr(texts: Sequence[str]) -> tuple[bool, str | None]:
    combined = "".join(texts).strip()
    if not combined:
        return False, "empty_output"
    if len(combined) >= 64:
        most_common = Counter(combined).most_common(1)[0][1]
        if most_common / len(combined) >= 0.95:
            return False, "degenerate_repeated_character"
    if not extract_blocks(combined):
        return False, "no_b01_b23_blocks"
    return True, None


def block_fidelity(
    reference_ids: Sequence[str], hypothesis_ids: Sequence[str]
) -> dict[str, object]:
    ref = list(reference_ids)
    hyp = list(hypothesis_ids)
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    overlap = sum((ref_counts & hyp_counts).values())
    precision = overlap / len(hyp) if hyp else 0.0
    recall = overlap / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missing = sorted((ref_counts - hyp_counts).elements())
    extra = sorted((hyp_counts - ref_counts).elements())
    duplicates = sorted(block_id for block_id, count in hyp_counts.items() if count > 1)
    common_ref_order = [block_id for block_id in ref if block_id in hyp_counts]
    common_hyp_order = [block_id for block_id in hyp if block_id in ref_counts]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "set_exact": set(ref) == set(hyp) and not duplicates,
        "ordered_exact": ref == hyp,
        "missing": missing,
        "extra": extra,
        "duplicate": duplicates,
        "reordered": common_ref_order != common_hyp_order,
    }
