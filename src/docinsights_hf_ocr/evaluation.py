"""Regenerate structured, CSV, and Markdown comparisons from raw evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .metrics import block_fidelity, cer, extract_blocks, is_valid_ocr, normalize_text, wer

NA_NO_VALID = "NA(no_valid_output)"
NA_NO_REFERENCE = "NA(no_reference)"
NA_NOT_MEASURED = "NA(not_measured)"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def query_passthrough(tasks_path: Path) -> dict[str, int]:
    rows = read_jsonl(tasks_path)
    raw = normalized = digest = 0
    for row in rows:
        source = row["user_query"]
        joined = {"instance_id": row["instance_id"], "user_query": source}["user_query"]
        raw += source == joined
        normalized += normalize_text(source) == normalize_text(joined)
        digest += sha256_text(source) == sha256_text(joined)
    return {
        "samples": len(rows),
        "raw_exact": raw,
        "normalized_exact": normalized,
        "sha256_exact": digest,
    }


def _reference_index(path: Path) -> dict[str, dict[str, Any]]:
    return {row["instance_id"]: row for row in read_jsonl(path) if row.get("status") == "ok"}


def _raw_texts(raw_dir: Path, result: dict[str, Any]) -> list[str]:
    paths = result.get("raw_output_paths", [])
    texts = []
    for raw_path in paths:
        candidate = raw_dir / Path(raw_path).name
        if candidate.exists():
            texts.append(candidate.read_text(encoding="utf-8"))
    return texts


def _nullable_rate(value: Any, denominator: float) -> float | str:
    return value / denominator if isinstance(value, (int, float)) else NA_NOT_MEASURED


def evaluate(
    raw_results: Path,
    raw_dir: Path,
    candidates_path: Path,
    reference_path: Path,
    tasks_path: Path,
    instance_id: str = "task_000909",
) -> dict[str, Any]:
    candidates = {
        row["model"].split("/")[-1]: row
        for row in json.loads(candidates_path.read_text())["models"]
    }
    references = _reference_index(reference_path)
    reference = references.get(instance_id)
    query = query_passthrough(tasks_path)
    rows: list[dict[str, Any]] = []
    for result in read_jsonl(raw_results):
        name = result["name"]
        candidate = candidates[name]
        texts = _raw_texts(raw_dir, result)
        valid, invalid_reason = (
            is_valid_ocr(texts) if result.get("success") else (False, "inference_failed")
        )
        hyp_blocks = extract_blocks("\n".join(texts)) if valid else []
        ref_blocks = reference.get("blocks", []) if reference else []
        ref_text = "\n".join(block["text"] for block in ref_blocks)
        hyp_text = "\n".join(text for _, text in hyp_blocks)
        if valid and reference:
            cer_value: float | str = cer(ref_text, hyp_text)
            wer_value: float | str = wer(ref_text, hyp_text)
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
                "device_runtime": "Kaggle NVIDIA T4 cuda:0 / transformers 5.12.1 fp16",
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
                "cost": "free Kaggle quota",
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
    }


def write_outputs(report: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    csv_path = out_dir / "comparison.csv"
    md_path = out_dir / "comparison.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for row in report["rows"]:
        flat = dict(row)
        flat["block_fidelity"] = json.dumps(
            flat["block_fidelity"], ensure_ascii=False, sort_keys=True
        )
        flat_rows.append(flat)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(flat_rows[0]) if flat_rows else [],
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
