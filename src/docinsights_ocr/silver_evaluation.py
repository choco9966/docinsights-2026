"""Score OCR output against a verified Codex-assisted silver transcription."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .metrics import (
    block_id_agreement,
    edit_distance,
    edit_similarity,
    exact_token_prf,
    nfkc_whitespace_normalize_text,
    normalize_text,
    ordered_quantity_prf,
)
from .records import read_jsonl

SCHEMA_VERSION = "1.0"
EVALUATION_KIND = "codex-silver-text-evaluation"
REFERENCE_KIND = "codex-assisted-silver"
INTERPRETATION = "silver_agreement_not_human_gold_accuracy"


def evaluate_codex_silver(
    reference_path: str | Path,
    prediction_path: str | Path,
    *,
    engine_label: str | None = None,
) -> dict[str, Any]:
    """Evaluate one complete OCR run against the verified Codex silver reference."""
    reference_source = Path(reference_path).resolve()
    prediction_source = Path(prediction_path).resolve()
    references = _records_by_id(reference_source, "silver reference")
    predictions = _records_by_id(prediction_source, "prediction")
    expected_ids = set(references)
    actual_ids = set(predictions)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(f"prediction coverage mismatch: missing={missing}, extra={extra}")

    reference_hash = _sha256_file(reference_source)
    prediction_hash = _sha256_file(prediction_source)
    rows = [
        _score_instance(instance_id, references[instance_id], predictions[instance_id])
        for instance_id in sorted(references)
    ]
    summary = _summary(rows)
    engines = sorted(
        {
            str(record.get("engine"))
            for record in predictions.values()
            if isinstance(record.get("engine"), str) and record.get("engine")
        }
    )
    label = engine_label or (engines[0] if len(engines) == 1 else "+".join(engines))
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_kind": EVALUATION_KIND,
        "reference_kind": REFERENCE_KIND,
        "interpretation": INTERPRETATION,
        "primary_score": {
            "name": "silver_text_score",
            "range": [0.0, 100.0],
            "definition": "100 * max(0, 1 - micro_character_error_rate)",
            "value": summary["silver_text_score"],
        },
        "normalization": {
            "strict": "Unicode NFC; CRLF/CR to LF; horizontal whitespace collapsed",
            "compatible": "Unicode NFKC; all whitespace collapsed; case and punctuation preserved",
            "semantic_corrections": False,
            "number_or_unit_corrections": False,
        },
        "sources": {
            "reference": {
                "path": str(reference_source),
                "sha256": reference_hash,
                "records": len(references),
            },
            "prediction": {
                "path": str(prediction_source),
                "sha256": prediction_hash,
                "records": len(predictions),
                "engine_label": label,
                "engines": engines,
            },
        },
        "summary": summary,
        "instances": rows,
    }


def write_silver_evaluation(
    result: Mapping[str, Any],
    output_path: str | Path,
    *,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically write JSON and optionally a compact Markdown scorecard."""
    output = Path(output_path).resolve()
    markdown = Path(markdown_path).resolve() if markdown_path is not None else None
    if markdown is not None and markdown == output:
        raise ValueError("JSON and Markdown output paths must differ")
    source_paths = _evaluation_source_paths(result)
    for destination in (output, markdown):
        if destination is not None and destination in source_paths:
            raise ValueError(
                f"evaluation output must not overwrite a source: {destination}"
            )
    serialized = json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True)
    payload = (serialized + "\n").encode()
    _atomic_write(output, payload)
    manifest = {
        "json": {"path": str(output), "sha256": hashlib.sha256(payload).hexdigest()}
    }
    if markdown is not None:
        markdown_payload = _markdown_scorecard(result).encode()
        _atomic_write(markdown, markdown_payload)
        manifest["markdown"] = {
            "path": str(markdown),
            "sha256": hashlib.sha256(markdown_payload).hexdigest(),
        }
    return manifest


def _score_instance(
    instance_id: str,
    reference: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    if reference.get("reference_kind") != REFERENCE_KIND:
        raise ValueError(f"reference_kind must be {REFERENCE_KIND!r} for {instance_id}")
    if reference.get("status") != "ok":
        raise ValueError(f"silver reference status must be ok for {instance_id}")
    reference_blocks = _ordered_blocks(reference, instance_id)
    predicted_blocks = _ordered_blocks(prediction, instance_id)
    reference_text = "\n".join(text for _, text in reference_blocks)
    predicted_text = "\n".join(text for _, text in predicted_blocks)
    strict_reference = normalize_text(reference_text)
    strict_prediction = normalize_text(predicted_text)
    compatible_reference = nfkc_whitespace_normalize_text(reference_text)
    compatible_prediction = nfkc_whitespace_normalize_text(predicted_text)
    reference_words = strict_reference.split()
    predicted_words = strict_prediction.split()
    character_distance = edit_distance(strict_reference, strict_prediction)
    word_distance = edit_distance(reference_words, predicted_words)
    character_error_rate = _reference_error_rate(
        character_distance, len(strict_reference), len(strict_prediction)
    )
    word_error_rate = _reference_error_rate(
        word_distance, len(reference_words), len(predicted_words)
    )
    timing = prediction.get("timing")
    total_seconds = timing.get("total_seconds") if isinstance(timing, Mapping) else None
    if not isinstance(total_seconds, (int, float)) or isinstance(total_seconds, bool):
        total_seconds = None
    blocks = block_id_agreement(
        (block_id for block_id, _ in reference_blocks),
        (block_id for block_id, _ in predicted_blocks),
    )
    return {
        "instance_id": instance_id,
        "reference_status": "ok",
        "prediction_status": prediction.get("status", "missing"),
        "strict_text": {
            "exact": strict_reference == strict_prediction,
            "reference_characters": len(strict_reference),
            "prediction_characters": len(strict_prediction),
            "edit_distance": character_distance,
            "character_error_rate": character_error_rate,
            "reference_normalized_character_accuracy": max(
                0.0, 1.0 - character_error_rate
            ),
            "symmetric_edit_similarity": edit_similarity(
                strict_reference, strict_prediction
            ),
        },
        "compatible_text": {
            "exact": compatible_reference == compatible_prediction,
            "reference_characters": len(compatible_reference),
            "prediction_characters": len(compatible_prediction),
            "edit_distance": edit_distance(compatible_reference, compatible_prediction),
            "symmetric_edit_similarity": edit_similarity(
                compatible_reference, compatible_prediction
            ),
        },
        "words": {
            "exact": reference_words == predicted_words,
            "reference_words": len(reference_words),
            "prediction_words": len(predicted_words),
            "edit_distance": word_distance,
            "word_error_rate": word_error_rate,
            "reference_normalized_word_accuracy": max(0.0, 1.0 - word_error_rate),
            "symmetric_edit_similarity": edit_similarity(
                reference_words, predicted_words
            ),
        },
        "blocks": asdict(blocks),
        "exact_tokens": asdict(exact_token_prf(reference_text, predicted_text)),
        "ordered_quantities": asdict(
            ordered_quantity_prf(reference_text, predicted_text)
        ),
        "total_seconds": float(total_seconds) if total_seconds is not None else None,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    strict = [row["strict_text"] for row in rows]
    compatible = [row["compatible_text"] for row in rows]
    words = [row["words"] for row in rows]
    blocks = [row["blocks"] for row in rows]
    character_distance = sum(item["edit_distance"] for item in strict)
    reference_characters = sum(item["reference_characters"] for item in strict)
    prediction_characters = sum(item["prediction_characters"] for item in strict)
    word_distance = sum(item["edit_distance"] for item in words)
    reference_words = sum(item["reference_words"] for item in words)
    prediction_words = sum(item["prediction_words"] for item in words)
    compatible_distance = sum(item["edit_distance"] for item in compatible)
    compatible_reference = sum(item["reference_characters"] for item in compatible)
    compatible_prediction = sum(item["prediction_characters"] for item in compatible)
    micro_cer = _reference_error_rate(
        character_distance, reference_characters, prediction_characters
    )
    micro_wer = _reference_error_rate(word_distance, reference_words, prediction_words)
    latencies = sorted(
        float(row["total_seconds"])
        for row in rows
        if isinstance(row.get("total_seconds"), (int, float))
    )
    exact_tokens = _aggregate_prf(rows, "exact_tokens")
    quantities = _aggregate_prf(rows, "ordered_quantities")
    return {
        "instances": count,
        "reference_ok": sum(row["reference_status"] == "ok" for row in rows),
        "prediction_ok": sum(row["prediction_status"] == "ok" for row in rows),
        "prediction_failed": sum(row["prediction_status"] != "ok" for row in rows),
        "strict_exact_count": sum(item["exact"] for item in strict),
        "strict_exact_rate": _ratio(sum(item["exact"] for item in strict), count),
        "compatible_exact_count": sum(item["exact"] for item in compatible),
        "compatible_exact_rate": _ratio(
            sum(item["exact"] for item in compatible), count
        ),
        "macro_character_error_rate": _mean(
            item["character_error_rate"] for item in strict
        ),
        "micro_character_error_rate": micro_cer,
        "macro_character_accuracy": _mean(
            item["reference_normalized_character_accuracy"] for item in strict
        ),
        "micro_character_accuracy": max(0.0, 1.0 - micro_cer),
        "macro_character_similarity": _mean(
            item["symmetric_edit_similarity"] for item in strict
        ),
        "micro_character_similarity": _symmetric_similarity(
            character_distance, reference_characters, prediction_characters
        ),
        "macro_word_error_rate": _mean(item["word_error_rate"] for item in words),
        "micro_word_error_rate": micro_wer,
        "micro_word_accuracy": max(0.0, 1.0 - micro_wer),
        "macro_word_similarity": _mean(
            item["symmetric_edit_similarity"] for item in words
        ),
        "micro_word_similarity": _symmetric_similarity(
            word_distance, reference_words, prediction_words
        ),
        "macro_compatible_character_similarity": _mean(
            item["symmetric_edit_similarity"] for item in compatible
        ),
        "micro_compatible_character_similarity": _symmetric_similarity(
            compatible_distance, compatible_reference, compatible_prediction
        ),
        "mean_block_f1": _mean(item["f1"] for item in blocks),
        "ordered_block_exact_count": sum(item["ordered_exact"] for item in blocks),
        "ordered_block_exact_rate": _ratio(
            sum(item["ordered_exact"] for item in blocks), count
        ),
        "exact_token_f1": exact_tokens["f1"],
        "ordered_quantity_f1": quantities["f1"],
        "silver_text_score": 100.0 * max(0.0, 1.0 - micro_cer),
        "latency": {
            "measured_instances": len(latencies),
            "mean_seconds_per_document": statistics.fmean(latencies)
            if latencies
            else None,
            "median_seconds_per_document": statistics.median(latencies)
            if latencies
            else None,
            "p95_seconds_per_document": _nearest_rank(latencies, 0.95),
            "documents_per_minute": 60.0 / statistics.fmean(latencies)
            if latencies and statistics.fmean(latencies) > 0
            else None,
        },
    }


def _aggregate_prf(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | int]:
    true_positive = sum(row[key]["true_positive"] for row in rows)
    predicted = sum(row[key]["predicted"] for row in rows)
    reference = sum(row[key]["reference"] for row in rows)
    precision = _ratio(true_positive, predicted, empty=1.0 if reference == 0 else 0.0)
    recall = _ratio(true_positive, reference, empty=1.0 if predicted == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted": predicted,
        "reference": reference,
    }


def _ordered_blocks(record: Mapping[str, Any], instance_id: str) -> list[tuple[str, str]]:
    blocks = record.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"blocks must be a list for {instance_id}")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError(f"block must be an object for {instance_id}")
        block_id = block.get("block_id")
        text = block.get("text")
        if not isinstance(block_id, str) or not block_id:
            raise ValueError(f"block_id must be non-empty for {instance_id}")
        if not isinstance(text, str):
            raise ValueError(f"block text must be a string for {instance_id}")
        canonical = block_id.casefold()
        if canonical in seen:
            raise ValueError(f"duplicate block_id {canonical!r} for {instance_id}")
        seen.add(canonical)
        result.append((canonical, text))
    return result


def _records_by_id(path: Path, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{label} requires a non-empty instance_id")
        if instance_id in result:
            raise ValueError(f"duplicate instance_id in {label}: {instance_id}")
        result[instance_id] = record
    return result


def _reference_error_rate(distance: int, reference: int, prediction: int) -> float:
    if reference == 0:
        return 0.0 if prediction == 0 else 1.0
    return distance / reference


def _symmetric_similarity(distance: int, reference: int, prediction: int) -> float:
    denominator = max(reference, prediction)
    return 1.0 if denominator == 0 else 1.0 - distance / denominator


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def _markdown_scorecard(result: Mapping[str, Any]) -> str:
    sources = result["sources"]
    summary = result["summary"]
    mean_latency = summary["latency"]["mean_seconds_per_document"]
    latency_text = "not measured" if mean_latency is None else f"{mean_latency:.4f} sec/doc"
    return "\n".join(
        [
            "# Codex Silver OCR Evaluation",
            "",
            "> 이 점수는 Codex-assisted silver agreement이며 human-gold accuracy가 아니다.",
            "",
            f"- Engine: `{sources['prediction']['engine_label']}`",
            f"- Instances: {summary['instances']}",
            f"- Silver text score: {summary['silver_text_score']:.4f} / 100",
            f"- Micro CER / WER: {summary['micro_character_error_rate']:.6f} / "
            f"{summary['micro_word_error_rate']:.6f}",
            f"- Symmetric character similarity: "
            f"{summary['micro_character_similarity']:.6f}",
            f"- NFKC+whitespace similarity: "
            f"{summary['micro_compatible_character_similarity']:.6f}",
            f"- Ordered block exact: {summary['ordered_block_exact_count']} / "
            f"{summary['instances']}",
            f"- Exact-token F1 / ordered-quantity F1: "
            f"{summary['exact_token_f1']:.6f} / {summary['ordered_quantity_f1']:.6f}",
            f"- Mean latency: {latency_text}",
            "",
            "## Sources",
            "",
            f"- Reference: `{sources['reference']['path']}` "
            f"(`{sources['reference']['sha256']}`)",
            f"- Prediction: `{sources['prediction']['path']}` "
            f"(`{sources['prediction']['sha256']}`)",
            "",
        ]
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _evaluation_source_paths(result: Mapping[str, Any]) -> set[Path]:
    sources = result.get("sources")
    if not isinstance(sources, Mapping):
        return set()
    paths: set[Path] = set()
    for source in sources.values():
        if not isinstance(source, Mapping):
            continue
        path = source.get("path")
        if isinstance(path, str) and path:
            paths.add(Path(path).resolve())
    return paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
