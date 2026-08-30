"""Pure metric utilities for OCR text and DocSem block identifiers."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PrecisionRecallF1:
    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted: int
    reference: int


@dataclass(frozen=True, slots=True)
class Quantity:
    """An exact ordered quantitative expression with bound semantic modifiers."""

    sign: str
    value: str
    currency: str
    unit: str


@dataclass(frozen=True, slots=True)
class BlockAgreement:
    coverage: float
    precision: float
    recall: float
    f1: float
    exact: bool
    ordered_exact: bool
    reordered: bool
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    intersection: int
    predicted: int
    reference: int


_TOKEN_RE = re.compile(
    r"[$€£¥₹₩]"
    r"|(?<!\w)(?:USD|EUR|GBP|JPY|KRW)(?!\w)"
    r"|[+−-](?=\d)"
    r"|(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])"
    r"|%|‰|°[cf]"
    r"|(?<!\w)(?:bp|bps|kg|g|mg|lb|km|m|cm|mm|h|hr|s|ms|kwh|wh|mb|gb|tb)(?!\w)",
    re.IGNORECASE,
)
_CURRENCY_PATTERN = r"[$€£¥₹₩]|USD|EUR|GBP|JPY|KRW"
_UNIT_PATTERN = r"%|‰|°[cCfF]|bp|bps|kg|g|mg|lb|km|m|cm|mm|h|hr|s|ms|kwh|wh|mb|gb|tb"
_QUANTITY_RE = re.compile(
    rf"(?<![\w.])"
    rf"(?:(?P<sign_before>[+−-])\s*)?"
    rf"(?:(?P<currency_before>{_CURRENCY_PATTERN})\s*)?"
    rf"(?:(?P<sign_after>[+−-])\s*)?"
    rf"(?P<value>(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)"
    rf"(?:\s*(?P<suffix>{_CURRENCY_PATTERN}|{_UNIT_PATTERN}))?"
    rf"(?![\w.])",
    re.IGNORECASE,
)


_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")


def normalize_text(text: str) -> str:
    """Apply strict OCR normalization while preserving case and glyph semantics."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return _HORIZONTAL_WHITESPACE.sub(" ", normalized)


def relaxed_normalize_text(text: str) -> str:
    """Apply compatibility normalization for non-strict exploratory matching."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-")
    return " ".join(normalized.split())


def nfkc_whitespace_normalize_text(text: str) -> str:
    """Apply NFKC and collapse whitespace without changing case or punctuation."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(reference: Sequence[T], predicted: Sequence[T]) -> int:
    """Compute exact Levenshtein distance with a bit-parallel fast path.

    OCR characters and word tokens are hashable, so Myers' algorithm reduces the
    Python-level work from one iteration per sequence pair to one per item in the
    longer sequence. The dynamic-programming fallback preserves the public
    function's support for arbitrary (including unhashable) sequence items.
    """
    if reference == predicted:
        return 0
    if not reference:
        return len(predicted)
    if not predicted:
        return len(reference)
    if len(reference) > len(predicted):
        reference, predicted = predicted, reference

    try:
        return _myers_edit_distance(reference, predicted)
    except TypeError:
        return _dynamic_edit_distance(reference, predicted)


def edit_similarity(reference: Sequence[T], predicted: Sequence[T]) -> float:
    """Return symmetric Levenshtein similarity in the inclusive range 0..1."""
    denominator = max(len(reference), len(predicted))
    if denominator == 0:
        return 1.0
    return 1.0 - edit_distance(reference, predicted) / denominator


def _myers_edit_distance(pattern: Sequence[T], text: Sequence[T]) -> int:
    """Compute Levenshtein distance using Myers' arbitrary-width bit vectors."""
    pattern_masks: dict[T, int] = {}
    for index, item in enumerate(pattern):
        pattern_masks[item] = pattern_masks.get(item, 0) | (1 << index)

    score = len(pattern)
    highest_bit = 1 << (len(pattern) - 1)
    positive_vertical = (1 << len(pattern)) - 1
    negative_vertical = 0

    for item in text:
        equality = pattern_masks.get(item, 0)
        vertical = equality | negative_vertical
        horizontal = (
            ((equality & positive_vertical) + positive_vertical) ^ positive_vertical
        ) | equality
        positive_horizontal = negative_vertical | ~(horizontal | positive_vertical)
        negative_horizontal = positive_vertical & horizontal

        if positive_horizontal & highest_bit:
            score += 1
        elif negative_horizontal & highest_bit:
            score -= 1

        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive_vertical = negative_horizontal | ~(vertical | positive_horizontal)
        negative_vertical = positive_horizontal & vertical

    return score


def _dynamic_edit_distance(reference: Sequence[T], predicted: Sequence[T]) -> int:
    """Compatibility fallback for sequence items that cannot be dictionary keys."""
    if len(reference) < len(predicted):
        predicted, reference = reference, predicted
    previous = list(range(len(predicted) + 1))
    for ref_index, ref_item in enumerate(reference, 1):
        current = [ref_index]
        for pred_index, pred_item in enumerate(predicted, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[pred_index] + 1,
                    previous[pred_index - 1] + (ref_item != pred_item),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, predicted: str) -> float:
    ref = normalize_text(reference)
    pred = normalize_text(predicted)
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(ref, pred) / len(ref)


cer = character_error_rate


def word_error_rate(reference: str, predicted: str) -> float:
    ref = normalize_text(reference).split()
    pred = normalize_text(predicted).split()
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(ref, pred) / len(ref)


wer = word_error_rate


def extract_exact_tokens(text: str) -> tuple[str, ...]:
    """Extract diagnostic tokens; use quantities for modifier-bound scoring."""
    return tuple(
        relaxed_normalize_text(match.group(0)).replace(" ", "")
        for match in _TOKEN_RE.finditer(text)
    )


def exact_token_prf(reference: str, predicted: str) -> PrecisionRecallF1:
    reference_tokens = Counter(extract_exact_tokens(reference))
    predicted_tokens = Counter(extract_exact_tokens(predicted))
    true_positive = sum((reference_tokens & predicted_tokens).values())
    predicted_count = sum(predicted_tokens.values())
    reference_count = sum(reference_tokens.values())
    precision = _safe_ratio(
        true_positive, predicted_count, empty=1.0 if reference_count == 0 else 0.0
    )
    recall = _safe_ratio(true_positive, reference_count, empty=1.0 if predicted_count == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PrecisionRecallF1(precision, recall, f1, true_positive, predicted_count, reference_count)


numeric_token_prf = exact_token_prf


def extract_ordered_quantities(text: str) -> tuple[Quantity, ...]:
    """Extract quantities in reading order, binding sign/currency/unit to each value."""
    quantities: list[Quantity] = []
    for match in _QUANTITY_RE.finditer(normalize_text(text)):
        sign_before = match.group("sign_before") or ""
        sign_after = match.group("sign_after") or ""
        if sign_before and sign_after:
            continue
        prefix_currency = match.group("currency_before") or ""
        suffix = match.group("suffix") or ""
        suffix_is_currency = bool(suffix and re.fullmatch(_CURRENCY_PATTERN, suffix, re.IGNORECASE))
        quantities.append(
            Quantity(
                sign=sign_before or sign_after,
                value=match.group("value"),
                currency=prefix_currency or (suffix if suffix_is_currency else ""),
                unit="" if suffix_is_currency else suffix,
            )
        )
    return tuple(quantities)


def ordered_quantity_prf(reference: str, predicted: str) -> PrecisionRecallF1:
    """Score exact quantity tuples using ordered LCS matching."""
    reference_quantities = extract_ordered_quantities(reference)
    predicted_quantities = extract_ordered_quantities(predicted)
    true_positive = _lcs_length(reference_quantities, predicted_quantities)
    predicted_count = len(predicted_quantities)
    reference_count = len(reference_quantities)
    precision = _safe_ratio(
        true_positive, predicted_count, empty=1.0 if reference_count == 0 else 0.0
    )
    recall = _safe_ratio(true_positive, reference_count, empty=1.0 if predicted_count == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PrecisionRecallF1(precision, recall, f1, true_positive, predicted_count, reference_count)


def block_id_agreement(reference: Iterable[str], predicted: Iterable[str]) -> BlockAgreement:
    reference_sequence = tuple(item.casefold() for item in reference)
    predicted_sequence = tuple(item.casefold() for item in predicted)
    reference_ids = set(reference_sequence)
    predicted_ids = set(predicted_sequence)
    intersection = len(reference_ids & predicted_ids)
    recall = _safe_ratio(intersection, len(reference_ids), empty=1.0 if not predicted_ids else 0.0)
    precision = _safe_ratio(
        intersection, len(predicted_ids), empty=1.0 if not reference_ids else 0.0
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return BlockAgreement(
        coverage=recall,
        precision=precision,
        recall=recall,
        f1=f1,
        exact=reference_ids == predicted_ids,
        ordered_exact=reference_sequence == predicted_sequence,
        reordered=reference_ids == predicted_ids and reference_sequence != predicted_sequence,
        missing=tuple(item for item in reference_sequence if item not in predicted_ids),
        extra=tuple(item for item in predicted_sequence if item not in reference_ids),
        intersection=intersection,
        predicted=len(predicted_ids),
        reference=len(reference_ids),
    )


block_coverage_agreement = block_id_agreement


def _safe_ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _lcs_length(reference: Sequence[T], predicted: Sequence[T]) -> int:
    previous = [0] * (len(predicted) + 1)
    for reference_item in reference:
        current = [0]
        for index, predicted_item in enumerate(predicted, 1):
            current.append(
                previous[index - 1] + 1
                if reference_item == predicted_item
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]
