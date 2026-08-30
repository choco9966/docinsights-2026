import pytest

from docinsights_ocr.blocks import reconstruct_blocks
from docinsights_ocr.models import BoundingBox, Document, Line, Page


def test_immutable_schema_and_page_validation() -> None:
    line = Line(page_number=1, text="b01 Revenue")
    page = Page(number=1, lines=(line,))
    document = Document(document_id="task-1", pages=(page,))

    with pytest.raises(AttributeError):
        document.engine = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        document.provenance["source"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="containing page"):
        Page(number=2, lines=(line,))
    with pytest.raises(ValueError, match="confidence"):
        Line(page_number=1, text="invalid", confidence=1.01)


def test_reconstructs_wrapped_blocks_in_page_order() -> None:
    pages = (
        Page(
            number=2,
            lines=(
                Line(2, "continued on page two", source_order=0),
                Line(2, "<b03> Third block", source_order=1),
            ),
        ),
        Page(
            number=1,
            lines=(
                Line(1, "header without marker", source_order=0),
                Line(1, "[B01] First wrapped", source_order=1),
                Line(1, "line 2", source_order=2),
                Line(1, "b02 Second", source_order=3),
            ),
        ),
    )

    blocks = reconstruct_blocks(pages)

    assert [block.block_id for block in blocks] == ["b01", "b02", "b03"]
    assert blocks[0].text == "First wrapped line 2"
    assert blocks[1].text == "Second continued on page two"
    assert blocks[1].page_numbers == (1, 2)


def test_multiple_markers_on_one_line_start_distinct_blocks() -> None:
    page = Page(number=1, lines=(Line(1, "b01: Alpha b02. Beta"),))
    assert [(block.block_id, block.text) for block in reconstruct_blocks((page,))] == [
        ("b01", "Alpha"),
        ("b02", "Beta"),
    ]


def test_only_corrects_punctuated_ocr_zero_markers_at_line_start() -> None:
    pages = (
        Page(
            number=1,
            lines=(
                Line(1, "b05: Normal", source_order=0),
                Line(1, "body bO5: remains literal", source_order=1),
                Line(1, "bO5 without punctuation", source_order=2),
                Line(1, "bO10: multi-digit lookalike", source_order=3),
                Line(1, "bO5.2 decimal lookalike", source_order=4),
                Line(1, "bO0: invalid zero block", source_order=5),
                Line(1, "bO00: invalid padded zero block", source_order=6),
                Line(1, "b06: Next", source_order=7),
            ),
        ),
    )

    blocks = reconstruct_blocks(pages)

    assert [block.block_id for block in blocks] == ["b05", "b06"]
    assert blocks[0].text == (
        "Normal body bO5: remains literal bO5 without punctuation "
        "bO10: multi-digit lookalike bO5.2 decimal lookalike "
        "bO0: invalid zero block bO00: invalid padded zero block"
    )
    assert blocks[0].lines[1].text == "body bO5: remains literal"
    assert blocks[0].lines[2].text == "bO5 without punctuation"
    assert blocks[0].lines[3].text == "bO10: multi-digit lookalike"
    assert blocks[0].lines[4].text == "bO5.2 decimal lookalike"
    assert blocks[0].lines[5].text == "bO0: invalid zero block"
    assert blocks[0].lines[6].text == "bO00: invalid padded zero block"


def test_recovers_consecutive_ocr_zero_markers_across_pages_without_mutating_lines() -> None:
    pages = (
        Page(
            number=1,
            lines=(
                Line(1, "  [bO5]: First amount", source_order=0),
                Line(1, "continued", source_order=1),
            ),
        ),
        Page(number=2, lines=(Line(2, "bO6. Second amount", source_order=0),)),
    )

    blocks = reconstruct_blocks(pages)

    assert [(block.block_id, block.text) for block in blocks] == [
        ("b05", "First amount continued"),
        ("b06", "Second amount"),
    ]
    assert blocks[0].page_numbers == (1,)
    assert blocks[1].page_numbers == (2,)
    assert blocks[0].lines[0].text == "  [bO5]: First amount"
    assert blocks[1].lines[0].text == "bO6. Second amount"


def test_recovers_zero_padded_ocr_marker_with_bracket_and_uppercase() -> None:
    line = Line(1, "  <BO05>: Padded amount")

    blocks = reconstruct_blocks((Page(number=1, lines=(line,)),))

    assert [(block.block_id, block.text) for block in blocks] == [("b05", "Padded amount")]
    assert blocks[0].lines[0].text == "  <BO05>: Padded amount"


def test_filters_repeated_margin_artifacts_and_obvious_footers_but_keeps_continuation() -> None:
    pages = (
        Page(
            1,
            (
                Line(1, "Annual Report", BoundingBox(10, 5, 100, 20), source_order=0),
                Line(1, "DocSEM | training copy", source_order=1),
                Line(1, "b01 Revenue increased", BoundingBox(10, 200, 200, 20), source_order=2),
                Line(
                    1,
                    "Confidential report · Page 1",
                    BoundingBox(10, 970, 250, 20),
                    source_order=3,
                ),
                Line(1, "DocSEM | training copy Page 1", source_order=4),
            ),
            width=800,
            height=1000,
        ),
        Page(
            2,
            (
                Line(2, "Annual Report", BoundingBox(10, 5, 100, 20), source_order=0),
                Line(2, "across the year", BoundingBox(10, 200, 200, 20), source_order=1),
                Line(
                    2,
                    "Confidential report · Page 2",
                    BoundingBox(10, 970, 250, 20),
                    source_order=2,
                ),
                Line(2, "DocSEM | training copy Page 2", source_order=3),
            ),
            width=800,
            height=1000,
        ),
    )

    blocks = reconstruct_blocks(pages)

    assert len(blocks) == 1
    assert blocks[0].text == "Revenue increased across the year"
    assert blocks[0].page_numbers == (1, 2)
