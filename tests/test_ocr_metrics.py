import random
import time
from collections.abc import Iterator

import pytest

import docinsights_ocr.metrics as metrics
from docinsights_ocr.metrics import (
    block_id_agreement,
    character_error_rate,
    edit_distance,
    exact_token_prf,
    extract_ordered_quantities,
    normalize_text,
    ordered_quantity_prf,
    relaxed_normalize_text,
    word_error_rate,
)


def _reference_edit_distance(reference: list[object], predicted: list[object]) -> int:
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


def test_text_normalization_and_error_rates() -> None:
    assert normalize_text("TOTAL\u3000Revenue\r\n— $100") == "TOTAL Revenue\n— $100"
    assert relaxed_normalize_text("  TOTAL\u3000Revenue — 100 ") == "total revenue - 100"
    assert character_error_rate("abc", "axc") == pytest.approx(1 / 3)
    assert word_error_rate("one two three", "one four") == pytest.approx(2 / 3)


def test_strict_error_rates_preserve_case_punctuation_currency_and_dash() -> None:
    assert character_error_rate("A—$1", "a-$1") > 0
    assert word_error_rate("Total: $1", "total $1") == 0.5


def test_bit_parallel_edit_distance_matches_dynamic_programming() -> None:
    randomizer = random.Random(20260830)
    alphabet = "abç€ "
    for _ in range(500):
        reference = [randomizer.choice(alphabet) for _ in range(randomizer.randrange(24))]
        predicted = [randomizer.choice(alphabet) for _ in range(randomizer.randrange(24))]
        expected = _reference_edit_distance(reference, predicted)
        assert edit_distance(reference, predicted) == expected
        assert edit_distance(predicted, reference) == expected


@pytest.mark.parametrize(
    ("reference", "predicted", "expected"),
    [
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        ("🙂🙂", "🙂🙃", 1),
        ("same", "same", 0),
    ],
)
def test_edit_distance_edge_cases(reference: str, predicted: str, expected: int) -> None:
    assert edit_distance(reference, predicted) == expected


def test_edit_distance_supports_unhashable_sequence_items() -> None:
    reference = [[1], [2], [3]]
    predicted = [[1], [4], [3], [5]]
    assert edit_distance(reference, predicted) == 2


def test_unhashable_fallback_uses_shorter_sequence_for_dp_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shorter = [[1], [2]]
    longer = [[1]] * 2_000
    range_stops: list[int] = []

    def observed_range(stop: int) -> Iterator[int]:
        range_stops.append(stop)
        return iter(range(stop))

    monkeypatch.setattr(metrics, "range", observed_range, raising=False)

    assert edit_distance(shorter, longer) == 1_999
    assert range_stops == [len(shorter) + 1]
    range_stops.clear()
    assert edit_distance(longer, shorter) == 1_999
    assert range_stops == [len(shorter) + 1]


def test_long_ocr_edit_distance_does_not_regress_to_quadratic_python_loop() -> None:
    reference = ("DocSem b09 revenue €123.45 and 54 kg.\n" * 300) + "answer"
    predicted = ("DocSem b09 revenue €123.45 and 54 kg.\n" * 300) + "anser"

    started = time.perf_counter()
    distance = edit_distance(reference, predicted)
    elapsed = time.perf_counter() - started

    assert distance == 1
    assert elapsed < 2.0


def test_numeric_currency_unit_sign_tokens_use_multiset_exact_matching() -> None:
    score = exact_token_prf("USD 1,200 +5% and 10 kg", "$1,200 5% and 10 kg")
    assert score.true_positive == 5
    assert score.reference == 7
    assert score.predicted == 6
    assert score.f1 == pytest.approx(10 / 13)


def test_block_coverage_and_agreement() -> None:
    score = block_id_agreement(["b01", "b02"], ["B02", "b03"])
    assert score.coverage == 0.5
    assert score.precision == 0.5
    assert score.f1 == 0.5
    assert score.exact is False
    assert score.ordered_exact is False
    assert score.missing == ("b01",)
    assert score.extra == ("b03",)


def test_block_agreement_detects_reordering_separately() -> None:
    score = block_id_agreement(["b01", "b02"], ["b02", "b01"])
    assert score.exact is True
    assert score.ordered_exact is False
    assert score.reordered is True
    assert score.missing == () and score.extra == ()


def test_ordered_quantity_prf_binds_sign_currency_value_and_unit() -> None:
    reference = "Loss was -$100 and mass was 5 kg"
    predicted = "Loss was $100 and mass was -5 kg"

    assert exact_token_prf(reference, predicted).f1 > 0.8
    assert ordered_quantity_prf(reference, predicted).f1 == 0.0
    quantities = extract_ordered_quantities(reference)
    assert quantities[0].sign == "-" and quantities[0].currency == "$"
    assert quantities[1].value == "5" and quantities[1].unit == "kg"


def test_ordered_quantity_prf_penalizes_reordering() -> None:
    assert ordered_quantity_prf("$1 then €2", "€2 then $1").f1 == 0.5
