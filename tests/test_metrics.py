from docinsights_hf_ocr.metrics import (
    block_aligned_error_rates,
    block_fidelity,
    cer,
    extract_blocks,
    is_valid_ocr,
    normalize_text,
    wer,
)


def test_normalization_only_changes_unicode_and_whitespace() -> None:
    assert normalize_text("e\u0301\t A!\r\nB") == "é A!\nB"
    assert normalize_text("ABC") != normalize_text("abc")


def test_cer_and_wer_are_levenshtein_rates() -> None:
    assert cer("abc", "axc") == 1 / 3
    assert wer("one two three", "one too three") == 1 / 3


def test_extract_blocks_from_html_and_order_fidelity() -> None:
    blocks = extract_blocks("<p>b01: Alpha</p><div>b03: Gamma</div><p>b02: Beta</p>")
    assert [block_id for block_id, _ in blocks] == ["b01", "b03", "b02"]
    fidelity = block_fidelity(["b01", "b02", "b03"], [block_id for block_id, _ in blocks])
    assert fidelity["set_exact"] is True
    assert fidelity["ordered_exact"] is False
    assert fidelity["reordered"] is True


def test_invalid_output_detection() -> None:
    assert is_valid_ocr(["!" * 512]) == (False, "degenerate_repeated_character")
    assert is_valid_ocr(["ordinary text without markers"])[0] is False
    assert is_valid_ocr(["b01: valid text"])[0] is True


def test_block_aligned_rates_treat_missing_as_empty() -> None:
    rates = block_aligned_error_rates([("b01", "alpha"), ("b02", "beta")], "b01: alpha")
    assert rates == (4 / 9, 1 / 2)


def test_block_aligned_rates_do_not_penalize_reordering() -> None:
    rates = block_aligned_error_rates([("b01", "alpha"), ("b02", "beta")], "b02: beta\nb01: alpha")
    assert rates == (0, 0)
    fidelity = block_fidelity(["b01", "b02"], ["b02", "b01"])
    assert fidelity["reordered"] is True


def test_block_aligned_rates_charge_duplicate_and_unrecognized_extra_content() -> None:
    reference = [("b01", "alpha")]
    duplicate = block_aligned_error_rates(reference, "b01: alpha\nb01: extra")
    unrecognized = block_aligned_error_rates(reference, "b01: alpha\nb99: extra")
    unmarked = block_aligned_error_rates(reference, "preface\nb01: alpha")
    assert duplicate == (1, 1)
    assert unrecognized == (1, 1)
    assert unmarked == (7 / 5, 1)
