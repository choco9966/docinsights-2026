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
EXPECTED_SCHEMA_VERSION = "2.0"
EXPECTED_MODEL_COUNT = 5
REFERENCE_KIND = "codex-assisted-silver"
REFERENCE_ENGINE = "codex-assisted-visual-transcription"


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_passthrough(
    tasks_path: Path, joined_tasks_path: Path, raw_results_sha256: str
) -> dict[str, int | str]:
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
        if source != joined_query:
            raise ValueError("post-inference joined user_query differs from source tasks")
        raw += source == joined_query
        normalized += normalize_text(source) == normalize_text(joined_query)
        digest += sha256_text(source) == sha256_text(joined_query)
        expected_binding = {
            "kind": "raw_results_sha256",
            "sha256": raw_results_sha256,
        }
        if joined[instance_id].get("evidence_binding") != expected_binding:
            raise ValueError(f"joined task has stale or invalid evidence binding: {instance_id}")
        if set(joined[instance_id]) != {"instance_id", "user_query", "evidence_binding"}:
            raise ValueError(f"joined task has unexpected fields: {instance_id}")
    if raw != len(sources):
        raise ValueError("post-inference joined user_query differs from source tasks")
    return {
        "samples": len(sources),
        "raw_exact": raw,
        "normalized_exact": normalized,
        "sha256_exact": digest,
        "raw_results_sha256": raw_results_sha256,
    }


def _reference_index(path: Path) -> dict[str, dict[str, Any]]:
    all_references = _unique_index(read_jsonl(path), "instance_id", "references")
    accepted = {key: row for key, row in all_references.items() if row.get("status") == "ok"}
    for instance_id, row in accepted.items():
        if row.get("reference_kind") != REFERENCE_KIND or row.get("engine") != REFERENCE_ENGINE:
            raise ValueError(f"accepted reference has unsupported identity: {instance_id}")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("reference_kind") != REFERENCE_KIND:
            raise ValueError(f"accepted reference has invalid provenance: {instance_id}")
        for key in ("input_pdf_sha256", "input_image_sha256", "renderer"):
            if not provenance.get(key):
                raise ValueError(f"accepted reference provenance is missing {key}: {instance_id}")
        if not _is_sha256(provenance["input_pdf_sha256"]):
            raise ValueError(f"accepted reference has invalid PDF identity: {instance_id}")
        if not isinstance(provenance["renderer"], str):
            raise TypeError(f"accepted reference has invalid renderer identity: {instance_id}")
        image_identities = provenance["input_image_sha256"]
        if not isinstance(image_identities, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("page_number"), int)
            and _is_sha256(item.get("sha256"))
            for item in image_identities
        ):
            raise ValueError(f"accepted reference has invalid image identities: {instance_id}")
        for identity_key in ("renderer_executable_identity", "codex_executable_identity"):
            identity = provenance.get(identity_key)
            if not (
                isinstance(identity, dict)
                and identity.get("kind") == "sha256"
                and isinstance(identity.get("name"), str)
                and identity["name"]
                and _is_sha256(identity.get("sha256"))
            ):
                raise ValueError(f"accepted reference has invalid {identity_key}: {instance_id}")
    return accepted


def _validate_v2_results(
    result_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    instance_id: str,
) -> dict[str, dict[str, Any]]:
    candidates = _unique_index(
        [{**row, "name": row["model"].split("/")[-1]} for row in candidate_rows],
        "name",
        "candidates",
    )
    results = _unique_index(result_rows, "name", "raw results")
    if len(candidate_rows) != EXPECTED_MODEL_COUNT or len(result_rows) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"v2 comparison requires exactly {EXPECTED_MODEL_COUNT} models/results")
    if set(results) != set(candidates):
        raise ValueError("raw result/candidate model membership differs")
    if {result.get("model_index") for result in result_rows} != set(
        range(1, EXPECTED_MODEL_COUNT + 1)
    ):
        raise ValueError("raw results require model_index values 1 through 5 exactly once")
    for name, result in results.items():
        candidate = candidates[name]
        if result.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise ValueError(f"unsupported raw result schema for {name}")
        if result.get("instance_id") != instance_id:
            raise ValueError(f"fixed instance mismatch for {name}")
        if result.get("repo") != candidate["model"]:
            raise ValueError(f"repository mismatch for {name}")
        requested = result.get("requested_revision")
        resolved = result.get("resolved_revision")
        if requested != candidate["revision"] or resolved != candidate["revision"]:
            raise ValueError(f"requested/resolved revision mismatch for {name}")
        if "repo_metadata_error" not in result or result["repo_metadata_error"] is not None:
            raise ValueError(f"repository identity metadata failed for {name}")
        metadata = result.get("repo_file_metadata")
        if not isinstance(metadata, list) or not all(isinstance(item, dict) for item in metadata):
            raise TypeError(f"repository file metadata is missing for {name}")
        model_files = [item for item in metadata if item.get("rfilename") == "model.safetensors"]
        if len(model_files) != 1:
            raise ValueError(f"model.safetensors identity is missing or ambiguous for {name}")
        model_file = model_files[0]
        if not _is_sha256(candidate.get("weight_lfs_sha256")) or not isinstance(
            candidate.get("weight_bytes"), int
        ):
            raise ValueError(f"candidate model identity is incomplete for {name}")
        if (
            model_file.get("lfs_sha256") != candidate["weight_lfs_sha256"]
            or model_file.get("lfs_size") != candidate["weight_bytes"]
        ):
            raise ValueError(f"model.safetensors LFS identity mismatch for {name}")
        rss = result.get("peak_process_rss_bytes_parent_sampled")
        vram = result.get("peak_vram_bytes_parent_sampled")
        if type(rss) is not int or rss <= 0:
            raise TypeError(f"parent RSS sample must be an exact positive integer for {name}")
        if type(vram) is not int or vram < 0:
            raise TypeError(f"parent VRAM sample must be an exact non-negative integer for {name}")
        for error_field in ("parent_rss_sampling_error", "parent_vram_sampling_error"):
            if result.get(error_field) not in (None, ""):
                raise ValueError(f"parent resource sampling failed for {name}: {error_field}")
        success = result.get("success")
        if type(success) is not bool:
            raise TypeError(f"success must be an exact boolean for {name}")
        outputs = result.get("raw_outputs")
        digests = result.get("raw_output_sha256")
        if success:
            if result.get("status") != "succeeded":
                raise ValueError(f"successful result has inconsistent status for {name}")
            if not isinstance(outputs, list) or len(outputs) != 2:
                raise ValueError(
                    f"successful result must declare exactly two raw outputs for {name}"
                )
            if not all(isinstance(output, dict) for output in outputs):
                raise TypeError(f"raw output identity must be an object for {name}")
            if {output.get("page") for output in outputs} != {1, 2}:
                raise ValueError(f"raw output pages must be unique identities 1 and 2 for {name}")
            if not isinstance(digests, list) or digests != [item.get("sha256") for item in outputs]:
                raise ValueError(f"raw output digest projection mismatch for {name}")
            if type(result.get("raw_output_bytes")) is not int or result["raw_output_bytes"] <= 0:
                raise TypeError(f"raw output byte projection must be a positive integer for {name}")
            for output in outputs:
                if not (
                    type(output.get("page")) is int
                    and isinstance(output.get("path"), str)
                    and output["path"]
                    and type(output.get("bytes")) is int
                    and output["bytes"] > 0
                    and _is_sha256(output.get("sha256"))
                ):
                    raise ValueError(f"raw output identity is incomplete for {name}")
        else:
            if result.get("status") != "failed":
                raise ValueError(f"failed result has inconsistent status for {name}")
            if (
                outputs != []
                or digests != []
                or type(result.get("raw_output_bytes")) is not int
                or result["raw_output_bytes"] != 0
            ):
                raise ValueError(f"failed result must not declare raw outputs for {name}")
    return candidates


def _create_or_validate_joined_tasks(
    tasks_path: Path,
    joined_tasks_path: Path,
    raw_results_sha256: str,
    instance_id: str,
) -> dict[str, int | str]:
    source_rows = read_jsonl(tasks_path)
    if len(source_rows) != 1 or source_rows[0].get("instance_id") != instance_id:
        raise ValueError("source tasks must contain exactly the fixed comparison instance")
    joined_row = {
        "instance_id": instance_id,
        "user_query": source_rows[0].get("user_query"),
        "evidence_binding": {"kind": "raw_results_sha256", "sha256": raw_results_sha256},
    }
    expected = json.dumps(joined_row, ensure_ascii=False, sort_keys=True) + "\n"
    if joined_tasks_path.exists():
        if joined_tasks_path.read_text(encoding="utf-8") != expected:
            raise ValueError(
                "joined task artifact is stale, prebuilt, or bound to different evidence"
            )
    else:
        joined_tasks_path.parent.mkdir(parents=True, exist_ok=True)
        joined_tasks_path.write_text(expected, encoding="utf-8")
    return query_passthrough(tasks_path, joined_tasks_path, raw_results_sha256)


def _raw_texts(raw_dir: Path, result: dict[str, Any]) -> list[str]:
    outputs = result.get("raw_outputs")
    paths = outputs if isinstance(outputs, list) else result.get("raw_output_paths", [])
    if result.get("success") and not paths:
        raise ValueError(f"successful result has no declared raw output: {result['name']}")
    texts = []
    byte_count = 0
    for raw_output in paths:
        metadata = raw_output if isinstance(raw_output, dict) else {}
        raw_path = metadata.get("path") if metadata else raw_output
        if not isinstance(raw_path, str):
            raise TypeError(f"declared raw output path is invalid: {result['name']}")
        candidate = raw_dir / Path(raw_path).name
        if not candidate.is_file():
            raise FileNotFoundError(f"declared raw output not found: {candidate}")
        contents = candidate.read_bytes()
        byte_count += len(contents)
        if metadata.get("bytes") is not None and metadata["bytes"] != len(contents):
            raise ValueError(f"raw output byte mismatch for {candidate}")
        if metadata.get("sha256") is not None:
            actual_sha256 = hashlib.sha256(contents).hexdigest()
            if metadata["sha256"] != actual_sha256:
                raise ValueError(f"raw output SHA-256 mismatch for {candidate}")
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
    expected_reference_sha256: str,
    instance_id: str = "task_000909",
) -> dict[str, Any]:
    if (
        not candidates_path.is_file()
        or not environment_path.is_file()
        or not baselines_path.is_file()
    ):
        raise FileNotFoundError("required JSON input file is absent")
    candidates_data = json.loads(candidates_path.read_text(encoding="utf-8"))
    if candidates_data.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("unsupported candidates schema")
    candidate_rows = candidates_data["models"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    if not _is_sha256(expected_reference_sha256):
        raise ValueError("expected reference SHA-256 must be exactly 64 lowercase hex characters")
    observed_reference_sha256 = _sha256_file(reference_path)
    if observed_reference_sha256 != expected_reference_sha256:
        raise ValueError(
            "reference artifact SHA-256 mismatch: "
            f"expected={expected_reference_sha256}, observed={observed_reference_sha256}"
        )
    references = _reference_index(reference_path)
    reference = references.get(instance_id)
    if reference is None:
        raise ValueError(f"fixed reference is missing or not validated: {instance_id}")
    rows: list[dict[str, Any]] = []
    result_rows = read_jsonl(raw_results)
    candidates = _validate_v2_results(result_rows, candidate_rows, instance_id)
    texts_by_name = {result["name"]: _raw_texts(raw_dir, result) for result in result_rows}
    raw_results_sha256 = _sha256_file(raw_results)
    query = _create_or_validate_joined_tasks(
        tasks_path, joined_tasks_path, raw_results_sha256, instance_id
    )
    for result in result_rows:
        name = result["name"]
        if name not in candidates:
            raise ValueError(f"raw result has no candidate: {name}")
        candidate = candidates[name]
        if result.get("repo") != candidate["model"]:
            raise ValueError(f"repository mismatch for {name}")
        revision = result["resolved_revision"]
        texts = texts_by_name[name]
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
                "revision": revision,
                "params": candidate["params"],
                "weight_bytes": candidate.get("weight_bytes"),
                "weight_gib": candidate.get("weight_gib") or candidate["weight_bytes"] / 1024**3,
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
                "peak_ram_bytes": result["peak_process_rss_bytes_parent_sampled"],
                "peak_ram_bytes_child": result.get("peak_process_rss_bytes_child", NA_NOT_MEASURED),
                "peak_vram_bytes": result["peak_vram_bytes_parent_sampled"],
                "peak_vram_bytes_child_allocated": result.get(
                    "peak_cuda_allocated_bytes", NA_NOT_MEASURED
                ),
                "output_bytes": result.get("raw_output_bytes", 0),
                "cost": result.get("cost", environment.get("cost", NA_NOT_MEASURED)),
                "candidate_gate_outcome": candidate.get("gate_outcome", NA_NOT_MEASURED),
                "selection_status": candidate.get("selection_status", NA_NOT_MEASURED),
                "runner_gate": result.get("gate", NA_NOT_MEASURED),
                "notes": invalid_reason or "valid OCR; compared with Codex silver, not human gold",
                "error": result.get("error"),
            }
        )
    codex_total_seconds = sum(
        row.get("timing", {}).get("total_seconds", 0.0) for row in references.values()
    )
    return {
        "schema_version": "2.0",
        "comparison_scope": "one fixed DocSem validation case; no quality-winner claim",
        "raw_evidence_status": environment.get("raw_evidence_status", NA_NOT_MEASURED),
        "reference": {
            "kind": "codex-assisted-silver",
            "artifact_sha256": observed_reference_sha256,
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
        "weight_bytes",
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
        "peak_ram_bytes_child",
        "peak_vram_bytes",
        "peak_vram_bytes_child_allocated",
        "output_bytes",
        "cost",
        "candidate_gate_outcome",
        "selection_status",
        "runner_gate",
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
