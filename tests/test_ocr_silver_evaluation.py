import hashlib
import json
from pathlib import Path

import pytest

from docinsights_ocr.cli import main
from docinsights_ocr.metrics import edit_similarity, nfkc_whitespace_normalize_text
from docinsights_ocr.silver_evaluation import (
    EVALUATION_KIND,
    INTERPRETATION,
    REFERENCE_KIND,
    evaluate_codex_silver,
    write_silver_evaluation,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(
        reference,
        [
            {
                "instance_id": "task_000001",
                "reference_kind": REFERENCE_KIND,
                "status": "ok",
                "engine": "codex-assisted-visual-transcription",
                "blocks": [
                    {"block_id": "b01", "text": "Revenue $100"},
                    {"block_id": "b02", "text": "Mass 5 kg"},
                ],
                "answer": "must not be copied",
            },
            {
                "instance_id": "task_000002",
                "reference_kind": REFERENCE_KIND,
                "status": "ok",
                "engine": "codex-assisted-visual-transcription",
                "blocks": [{"block_id": "b01", "text": "Ａ  B\nC"}],
            },
        ],
    )
    _write_jsonl(
        prediction,
        [
            {
                "instance_id": "task_000001",
                "status": "ok",
                "engine": "test-ocr",
                "blocks": [
                    {"block_id": "b01", "text": "Revenue $101"},
                    {"block_id": "b02", "text": "Mass 5 kg"},
                ],
                "timing": {"total_seconds": 2.0},
                "user_query": "must not be copied",
            },
            {
                "instance_id": "task_000002",
                "status": "ok",
                "engine": "test-ocr",
                "blocks": [{"block_id": "b01", "text": "A B C"}],
                "timing": {"total_seconds": 4.0},
            },
        ],
    )
    return reference, prediction


def test_nfkc_whitespace_normalization_and_symmetric_similarity() -> None:
    assert nfkc_whitespace_normalize_text("Ａ\t B\nC") == "A B C"
    assert edit_similarity("abc", "abc") == 1.0
    assert edit_similarity("", "") == 1.0
    assert edit_similarity("abc", "axc") == pytest.approx(2 / 3)
    assert edit_similarity("abc", "abcdef") == pytest.approx(0.5)


def test_evaluate_codex_silver_scores_text_blocks_tokens_and_latency(
    tmp_path: Path,
) -> None:
    reference, prediction = _artifacts(tmp_path)

    result = evaluate_codex_silver(reference, prediction)

    assert result["evaluation_kind"] == EVALUATION_KIND
    assert result["reference_kind"] == REFERENCE_KIND
    assert result["interpretation"] == INTERPRETATION
    assert result["sources"]["reference"]["sha256"] == _sha256(reference)
    assert result["sources"]["prediction"]["engine_label"] == "test-ocr"
    assert result["summary"]["instances"] == 2
    assert result["summary"]["prediction_ok"] == 2
    assert 0 < result["summary"]["micro_character_error_rate"] < 1
    assert result["summary"]["ordered_block_exact_count"] == 2
    assert result["summary"]["latency"]["mean_seconds_per_document"] == 3.0
    assert result["instances"][1]["strict_text"]["exact"] is False
    assert result["instances"][1]["compatible_text"]["exact"] is True
    assert result["primary_score"]["value"] == result["summary"]["silver_text_score"]
    serialized = json.dumps(result)
    assert "must not be copied" not in serialized


def test_evaluate_codex_silver_rejects_non_silver_duplicate_and_coverage_mismatch(
    tmp_path: Path,
) -> None:
    reference, prediction = _artifacts(tmp_path)
    rows = [json.loads(line) for line in reference.read_text().splitlines()]
    rows[0]["reference_kind"] = "human-gold"
    _write_jsonl(reference, rows)
    with pytest.raises(ValueError, match="reference_kind"):
        evaluate_codex_silver(reference, prediction)

    rows[0]["reference_kind"] = REFERENCE_KIND
    _write_jsonl(reference, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate instance_id"):
        evaluate_codex_silver(reference, prediction)

    _write_jsonl(reference, rows)
    prediction_rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    _write_jsonl(prediction, prediction_rows[:1])
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate_codex_silver(reference, prediction)


def test_write_and_cli_emit_hashed_json_and_markdown(tmp_path: Path) -> None:
    reference, prediction = _artifacts(tmp_path)
    output = tmp_path / "reports" / "score.json"
    markdown = tmp_path / "reports" / "score.md"

    assert (
        main(
            [
                "codex-silver-evaluate",
                str(reference),
                str(prediction),
                str(output),
                "--markdown",
                str(markdown),
                "--engine-label",
                "fixture-engine",
                "--reference-label",
                "logical/reference.jsonl",
                "--prediction-label",
                "logical/prediction.jsonl",
            ]
        )
        == 0
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["sources"]["prediction"]["engine_label"] == "fixture-engine"
    assert saved["sources"]["reference"]["path"] == "logical/reference.jsonl"
    assert saved["sources"]["prediction"]["path"] == "logical/prediction.jsonl"
    assert "silver agreement" in markdown.read_text(encoding="utf-8")
    manifest = write_silver_evaluation(saved, output, markdown_path=markdown)
    assert manifest["json"]["sha256"] == _sha256(output)
    assert manifest["markdown"]["sha256"] == _sha256(markdown)


@pytest.mark.parametrize("collision", ["reference", "prediction"])
def test_writer_rejects_overwriting_evaluation_sources(tmp_path: Path, collision: str) -> None:
    reference, prediction = _artifacts(tmp_path)
    result = evaluate_codex_silver(reference, prediction)
    source = reference if collision == "reference" else prediction
    original = source.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite a source"):
        write_silver_evaluation(result, source)
    with pytest.raises(ValueError, match="must not overwrite a source"):
        write_silver_evaluation(
            result,
            tmp_path / "score.json",
            markdown_path=source,
        )

    assert source.read_bytes() == original


def test_canonical_labels_remain_portable_without_weakening_source_protection(
    tmp_path: Path,
) -> None:
    reference, prediction = _artifacts(tmp_path / "one")
    result = evaluate_codex_silver(
        reference,
        prediction,
        reference_label="issue8/codex-validation-reference.jsonl",
        prediction_label="issue8/test-prediction.jsonl",
    )
    assert result["sources"]["reference"]["path"].startswith("issue8/")
    second_reference, second_prediction = _artifacts(tmp_path / "two")
    second_result = evaluate_codex_silver(
        second_reference,
        second_prediction,
        reference_label="issue8/codex-validation-reference.jsonl",
        prediction_label="issue8/test-prediction.jsonl",
    )
    assert result == second_result
    with pytest.raises(ValueError, match="must not overwrite a source"):
        write_silver_evaluation(
            result,
            reference,
            protected_source_paths=(reference, prediction),
        )
