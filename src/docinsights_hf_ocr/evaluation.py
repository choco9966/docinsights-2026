"""Regenerate structured, CSV, and Markdown comparisons from raw evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .metrics import (
    block_aligned_error_rates,
    block_fidelity,
    extract_blocks,
    is_valid_ocr,
    normalize_text,
)

NA_NO_VALID = "NA(no_valid_output)"
NA_NO_REFERENCE = "NA(no_reference)"
NA_NOT_MEASURED = "NA(not_measured)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL file not found: {path}")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"required JSONL file is empty: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row has no non-empty {key}")
        if value in index:
            raise ValueError(f"duplicate {key} in {label}: {value}")
        index[value] = row
    return index


def query_passthrough(tasks_path: Path, joined_tasks_path: Path) -> dict[str, int]:
    source_rows = read_jsonl(tasks_path)
    joined_rows = read_jsonl(joined_tasks_path)
    sources = _unique_index(source_rows, "instance_id", "tasks")
    joined = _unique_index(joined_rows, "instance_id", "joined tasks")
    missing = sorted(sources.keys() - joined.keys())
    extra = sorted(joined.keys() - sources.keys())
    if missing or extra:
        raise ValueError(f"joined task keys differ: missing={missing}, extra={extra}")
    raw = normalized = digest = 0
    for instance_id, row in sources.items():
        source = row["user_query"]
        joined_query = joined[instance_id].get("user_query")
        if not isinstance(source, str) or not isinstance(joined_query, str):
            raise TypeError(f"user_query must be a string: {instance_id}")
        raw += source == joined_query
        normalized += normalize_text(source) == normalize_text(joined_query)
        digest += sha256_text(source) == sha256_text(joined_query)
    if raw != len(sources):
        raise ValueError("post-inference joined user_query differs from source tasks")
    return {
        "samples": len(sources),
        "raw_exact": raw,
        "normalized_exact": normalized,
        "sha256_exact": digest,
    }


def _reference_index(path: Path) -> dict[str, dict[str, Any]]:
    all_references = _unique_index(read_jsonl(path), "instance_id", "references")
    return {key: row for key, row in all_references.items() if row.get("status") == "ok"}


def _raw_texts(raw_dir: Path, result: dict[str, Any]) -> list[str]:
    paths = result.get("raw_output_paths", [])
    if result.get("success") and not paths:
        raise ValueError(f"successful result has no declared raw output: {result['name']}")
    texts = []
    byte_count = 0
    for raw_path in paths:
        candidate = raw_dir / Path(raw_path).name
        if not candidate.is_file():
            raise FileNotFoundError(f"declared raw output not found: {candidate}")
        byte_count += candidate.stat().st_size
        texts.append(candidate.read_text(encoding="utf-8"))
    if result.get("raw_output_bytes") != byte_count:
        raise ValueError(
            f"raw output byte mismatch for {result['name']}: "
            f"declared={result.get('raw_output_bytes')}, actual={byte_count}"
        )
    return texts


def _nullable_rate(value: Any, denominator: float) -> float | str:
    return value / denominator if isinstance(value, (int, float)) else NA_NOT_MEASURED


def evaluate(
    raw_results: Path,
    raw_dir: Path,
    candidates_path: Path,
    reference_path: Path,
    tasks_path: Path,
    joined_tasks_path: Path,
    environment_path: Path,
    baselines_path: Path,
    instance_id: str = "task_000909",
) -> dict[str, Any]:
    if (
        not candidates_path.is_file()
        or not environment_path.is_file()
        or not baselines_path.is_file()
    ):
        raise FileNotFoundError("required JSON input file is absent")
    candidates_data = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidate_rows = candidates_data["models"]
    candidates = _unique_index(
        [{**row, "name": row["model"].split("/")[-1]} for row in candidate_rows],
        "name",
        "candidates",
    )
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    references = _reference_index(reference_path)
    reference = references.get(instance_id)
    if reference is None:
        raise ValueError(f"fixed reference is missing or not validated: {instance_id}")
    query = query_passthrough(tasks_path, joined_tasks_path)
    rows: list[dict[str, Any]] = []
    result_rows = read_jsonl(raw_results)
    _unique_index(result_rows, "name", "raw results")
    for result in result_rows:
        name = result["name"]
        if name not in candidates:
            raise ValueError(f"raw result has no candidate: {name}")
        candidate = candidates[name]
        if result.get("repo") != candidate["model"]:
            raise ValueError(f"repository mismatch for {name}")
        if result.get("revision") != candidate["revision"]:
            raise ValueError(f"revision mismatch for {name}")
        texts = _raw_texts(raw_dir, result)
        valid, invalid_reason = (
            is_valid_ocr(texts) if result.get("success") else (False, "inference_failed")
        )
        hyp_blocks = extract_blocks("\n".join(texts)) if valid else []
        ref_blocks = reference.get("blocks", []) if reference else []
        if valid and reference:
            cer_value, wer_value = block_aligned_error_rates(
                [(block["block_id"], block["text"]) for block in ref_blocks], "\n".join(texts)
            )
            fidelity: dict[str, object] | str = block_fidelity(
                [block["block_id"] for block in ref_blocks],
                [block_id for block_id, _ in hyp_blocks],
            )
            quality_samples = 1
        elif not valid:
            cer_value = wer_value = NA_NO_VALID
            fidelity = NA_NO_VALID
            quality_samples = 0
        else:
            cer_value = wer_value = NA_NO_REFERENCE
            fidelity = NA_NO_REFERENCE
            quality_samples = 0
        latency = result.get("doc_latency_sec")
        rows.append(
            {
                "model": result["repo"],
                "revision": result["revision"],
                "params": candidate["params"],
                "weight_gib": candidate["weight_gib"],
                "license": candidate["license"],
                "device_runtime": result.get("device_runtime")
                or environment.get("device_runtime")
                or f"{environment['platform']}; {environment['gpu']}; "
                f"transformers {environment['packages']['transformers']}",
                "samples": 1,
                "quality_samples": quality_samples,
                "inference_success_rate": 1.0 if result.get("success") else 0.0,
                "valid_ocr_rate": 1.0 if valid else 0.0,
                "silver_agreement_cer": cer_value,
                "silver_agreement_wer": wer_value,
                "query_raw_exact": f"{query['raw_exact']}/{query['samples']}",
                "query_normalized_exact": f"{query['normalized_exact']}/{query['samples']}",
                "query_sha256_exact": f"{query['sha256_exact']}/{query['samples']}",
                "block_fidelity": fidelity,
                "load_sec": result.get("load_sec")
                if result.get("load_sec") is not None
                else NA_NOT_MEASURED,
                "sec_per_doc": latency if latency is not None else NA_NOT_MEASURED,
                "docs_per_min": _nullable_rate(60, latency) if latency else NA_NOT_MEASURED,
                "peak_ram_bytes": result.get("peak_process_rss_bytes", NA_NOT_MEASURED),
                "peak_vram_bytes": result.get("peak_cuda_allocated_bytes", NA_NOT_MEASURED),
                "output_bytes": result.get("raw_output_bytes", 0),
                "cost": result.get("cost", environment.get("cost", NA_NOT_MEASURED)),
                "notes": invalid_reason or "valid OCR; compared with Codex silver, not human gold",
                "error": result.get("error"),
            }
        )
    codex_total_seconds = sum(
        row.get("timing", {}).get("total_seconds", 0.0) for row in references.values()
    )
    return {
        "schema_version": "1.0",
        "comparison_scope": "one fixed DocSem validation case; no quality-winner claim",
        "raw_evidence_status": environment.get("raw_evidence_status", NA_NOT_MEASURED),
        "reference": {
            "kind": "codex-assisted-silver",
            "human_gold": False,
            "available_validated_subset": len(references),
            "total_seconds_when_present": codex_total_seconds,
            "fixed_case_seconds": reference.get("timing", {}).get("total_seconds")
            if reference
            else None,
        },
        "query_passthrough": query,
        "rows": rows,
        "baselines": _baseline_rows(baselines),
    }


def _baseline_rows(baselines: dict[str, Any]) -> list[dict[str, Any]]:
    existing = baselines["existing_ocr_operational_comparison"]
    agreement = existing["engine_agreement_not_accuracy"]
    rows = []
    for key, label in (("apple_vision", "Apple Vision"), ("tesseract_psm6", "Tesseract PSM 6")):
        value = existing[key]
        rows.append(
            {
                "row_type": "operational_baseline",
                "model": label,
                "documents": existing["documents"],
                "pages": existing["pages"],
                "blocks": existing["blocks"],
                "failures": existing["failures"],
                "total_seconds": value["total_seconds"],
                "sec_per_doc": value["seconds_per_document"],
                "peak_ram_bytes": value["peak_rss_bytes"],
                "engine_agreement_cer": agreement["cer"],
                "engine_agreement_wer": agreement["wer"],
                "engine_agreement_block_f1": agreement["block_f1"],
                "notes": "217-document operational baseline; engine agreement is not accuracy",
            }
        )
    return rows


def write_outputs(report: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    csv_path = out_dir / "comparison.csv"
    md_path = out_dir / "comparison.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for row in [*report["rows"], *report["baselines"]]:
        flat = dict(row)
        flat.setdefault("row_type", "measured_fixed_case")
        if "block_fidelity" in flat:
            flat["block_fidelity"] = json.dumps(
                flat["block_fidelity"], ensure_ascii=False, sort_keys=True
            )
        flat_rows.append(flat)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys(key for row in flat_rows for key in row))
            if flat_rows
            else [],
            lineterminator="\n",
        )
        if flat_rows:
            writer.writeheader()
            writer.writerows(flat_rows)
    headers = [
        "model",
        "revision",
        "params",
        "weight_gib",
        "license",
        "device_runtime",
        "samples",
        "quality_samples",
        "inference_success_rate",
        "valid_ocr_rate",
        "silver_agreement_cer",
        "silver_agreement_wer",
        "query_raw_exact",
        "query_normalized_exact",
        "query_sha256_exact",
        "block_fidelity",
        "load_sec",
        "sec_per_doc",
        "docs_per_min",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "output_bytes",
        "cost",
        "notes",
    ]
    lines = [
        "# DocSem 소형 OCR 비교표",
        "",
        "Codex 전사는 human gold가 아닌 silver reference다.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in report["rows"]:
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, dict):
                value = f"F1={value['f1']:.6f}; ordered={value['ordered_exact']}"
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 기존 OCR 운영 baseline (엔진 간 agreement이며 accuracy가 아님)",
            "",
            "| model | documents | pages | blocks | failures | total_seconds | sec_per_doc | peak_ram_bytes | engine_agreement_cer | engine_agreement_wer | engine_agreement_block_f1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["baselines"]:
        baseline_headers = [
            "model",
            "documents",
            "pages",
            "blocks",
            "failures",
            "total_seconds",
            "sec_per_doc",
            "peak_ram_bytes",
            "engine_agreement_cer",
            "engine_agreement_wer",
            "engine_agreement_block_f1",
        ]
        lines.append("| " + " | ".join(str(row[key]) for key in baseline_headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, csv_path, md_path]


def write_raw_csv(raw_results: Path, output: Path) -> Path:
    """Create a flat CSV view while retaining JSONL as canonical raw data."""
    rows = read_jsonl(raw_results)
    fields = sorted({key for row in rows for key in row})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    return output


def hash_paths(paths: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
