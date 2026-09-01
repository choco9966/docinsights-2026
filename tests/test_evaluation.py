import copy
import hashlib
import json
from pathlib import Path

import pytest

from docinsights_hf_ocr.cli import build_parser
from docinsights_hf_ocr.evaluation import (
    evaluate,
    hash_paths,
    query_passthrough,
    read_jsonl,
    write_outputs,
    write_raw_csv,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value, ensure_ascii=False) + "\n")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tasks = _write(
        tmp_path / "tasks.jsonl", '{"instance_id":"task_000909","user_query":"Keep  this?"}\n'
    )
    joined = tmp_path / "joined.jsonl"
    refs = _write(
        tmp_path / "refs.jsonl",
        json.dumps(
            {
                "instance_id": "task_000909",
                "status": "ok",
                "reference_kind": "codex-assisted-silver",
                "engine": "codex-assisted-visual-transcription",
                "provenance": {
                    "reference_kind": "codex-assisted-silver",
                    "input_pdf_sha256": "a" * 64,
                    "input_image_sha256": [{"page_number": 1, "sha256": "b" * 64}],
                    "renderer": "pdftoppm",
                    "renderer_executable_identity": {
                        "kind": "sha256",
                        "name": "pdftoppm",
                        "sha256": "c" * 64,
                    },
                    "codex_executable_identity": {
                        "kind": "sha256",
                        "name": "codex",
                        "sha256": "d" * 64,
                    },
                },
                "blocks": [{"block_id": "b01", "text": "Alpha"}],
                "timing": {"total_seconds": 2},
            }
        )
        + "\n"
        + json.dumps(
            {
                "instance_id": "task_other",
                "status": "failed",
                "blocks": [],
                "timing": {"total_seconds": 3},
            }
        )
        + "\n",
    )
    models = [
        {
            "model": f"org/model-{index}",
            "revision": f"rev-{index}",
            "params": index,
            "weight_bytes": 100 + index,
            "weight_lfs_sha256": str(index) * 64,
            "license": "MIT",
        }
        for index in range(1, 6)
    ]
    candidates = _json(
        tmp_path / "candidates.json",
        {"schema_version": "2.0", "models": models},
    )
    environment = _json(
        tmp_path / "environment.json",
        {
            "platform": "test platform",
            "gpu": "test gpu",
            "packages": {"transformers": "1.2.3"},
            "cost": "test quota",
        },
    )
    reference_sha256 = hashlib.sha256(refs.read_bytes()).hexdigest()
    silver_dir = tmp_path / "raw/silver"
    silver_artifacts: dict[str, tuple[Path, str]] = {}
    for key, engine_label, engine, score, cer, wer, seconds in (
        ("apple", "Apple Vision", "apple-vision", 90.0, 0.1, 0.2, 5.0),
        ("tesseract", "Tesseract PSM 6", "tesseract-tsv", 95.0, 0.05, 0.1, 10.0),
    ):
        prediction_sha256 = ("e" if key == "apple" else "f") * 64
        evaluation = _json(
            silver_dir / f"{key}.json",
            {
                "schema_version": "1.0",
                "evaluation_kind": "codex-silver-text-evaluation",
                "reference_kind": "codex-assisted-silver",
                "interpretation": "silver_agreement_not_human_gold_accuracy",
                "primary_score": {"name": "silver_text_score", "value": score},
                "sources": {
                    "reference": {"sha256": reference_sha256, "records": 1},
                    "prediction": {
                        "sha256": prediction_sha256,
                        "records": 1,
                        "engine_label": engine_label,
                        "engines": [engine],
                    },
                },
                "summary": {
                    "instances": 1,
                    "reference_ok": 1,
                    "prediction_ok": 1,
                    "prediction_failed": 0,
                    "silver_text_score": score,
                    "micro_character_error_rate": cer,
                    "micro_word_error_rate": wer,
                    "mean_block_f1": 1.0,
                    "ordered_block_exact_rate": 1.0,
                    "strict_exact_rate": 0.5,
                    "latency": {
                        "measured_instances": 1,
                        "mean_seconds_per_document": seconds,
                        "documents_per_minute": 60 / seconds,
                        "p95_seconds_per_document": seconds + 1,
                    },
                },
                "instances": [
                    {
                        "instance_id": "task_000909",
                        "reference_status": "ok",
                        "prediction_status": "ok",
                    }
                ],
            },
        )
        silver_artifacts[key] = (evaluation, prediction_sha256)
    baselines = _json(
        tmp_path / "baselines.json",
        {
            "schema_version": "2.0",
            "reference": {
                "kind": "codex-assisted-silver",
                "records": 1,
                "sha256": reference_sha256,
            },
            "silver_baselines": {
                "apple_vision": {
                    "model": "Apple Vision",
                    "engine_label": "Apple Vision",
                    "engines": ["apple-vision"],
                    "revision": "fixture-apple",
                    "prediction_artifact_sha256": silver_artifacts["apple"][1],
                    "evaluation_artifact": {
                        "path": "raw/silver/apple.json",
                        "sha256": hashlib.sha256(
                            silver_artifacts["apple"][0].read_bytes()
                        ).hexdigest(),
                    },
                    "runtime": {
                        "device_runtime": "test Apple runtime",
                        "peak_ram_bytes": 100,
                        "cost": "test local",
                    },
                },
                "tesseract_psm6": {
                    "model": "Tesseract PSM 6",
                    "engine_label": "Tesseract PSM 6",
                    "engines": ["tesseract-tsv"],
                    "revision": "fixture-tesseract",
                    "prediction_artifact_sha256": silver_artifacts["tesseract"][1],
                    "evaluation_artifact": {
                        "path": "raw/silver/tesseract.json",
                        "sha256": hashlib.sha256(
                            silver_artifacts["tesseract"][0].read_bytes()
                        ).hexdigest(),
                    },
                    "runtime": {
                        "device_runtime": "test CPU runtime",
                        "peak_ram_bytes": 50,
                        "cost": "test local",
                    },
                },
            },
        },
    )
    raw_dir = tmp_path / "raw"
    raw_output = _write(raw_dir / "model-1-page-1.txt", "b01: Alpha")
    raw_output_2 = _write(raw_dir / "model-1-page-2.txt", "\n")
    output_sha = hashlib.sha256(raw_output.read_bytes()).hexdigest()
    output_sha_2 = hashlib.sha256(raw_output_2.read_bytes()).hexdigest()
    result_rows = []
    for index, model in enumerate(models, 1):
        success = index == 1
        result_rows.append(
            {
                "schema_version": "2.0",
                "instance_id": "task_000909",
                "model_index": index,
                "name": f"model-{index}",
                "repo": model["model"],
                "requested_revision": model["revision"],
                "resolved_revision": model["revision"],
                "repo_metadata_error": None,
                "repo_file_metadata": [
                    {
                        "rfilename": "model.safetensors",
                        "lfs_size": model["weight_bytes"],
                        "lfs_sha256": model["weight_lfs_sha256"],
                    }
                ],
                "success": success,
                "status": "succeeded" if success else "failed",
                "raw_outputs": [
                    {
                        "page": 1,
                        "path": "/remote/model-1-page-1.txt",
                        "bytes": raw_output.stat().st_size,
                        "sha256": output_sha,
                    },
                    {
                        "page": 2,
                        "path": "/remote/model-1-page-2.txt",
                        "bytes": raw_output_2.stat().st_size,
                        "sha256": output_sha_2,
                    },
                ]
                if success
                else [],
                "raw_output_sha256": [output_sha, output_sha_2] if success else [],
                "doc_latency_sec": 2 if success else None,
                "peak_process_rss_bytes_child": 30,
                "peak_process_rss_bytes_parent_sampled": 10 + index,
                "peak_process_rss_sampling_error": None,
                "peak_cuda_allocated_bytes": 20,
                "peak_vram_bytes_parent_sampled": 20 + index,
                "peak_vram_sampling_error": None,
                "raw_output_bytes": raw_output.stat().st_size + raw_output_2.stat().st_size
                if success
                else 0,
            }
        )
    results = _write(
        tmp_path / "results.jsonl",
        "".join(json.dumps(row) + "\n" for row in result_rows),
    )
    return {
        "raw_results": results,
        "raw_dir": raw_dir,
        "candidates_path": candidates,
        "reference_path": refs,
        "tasks_path": tasks,
        "joined_tasks_path": joined,
        "environment_path": environment,
        "baselines_path": baselines,
        "expected_reference_sha256": reference_sha256,
    }


def _evaluate(paths: dict[str, Path]):
    return evaluate(**paths)


def _result_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_result_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _write(path, "".join(json.dumps(row) + "\n" for row in rows))


def test_dynamic_reference_refresh_join_baselines_and_determinism(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = _evaluate(paths)
    assert report["reference"]["available_validated_subset"] == 1
    assert report["reference"]["total_seconds_when_present"] == 2
    assert (
        report["reference"]["artifact_sha256"]
        == hashlib.sha256(paths["reference_path"].read_bytes()).hexdigest()
    )
    assert report["query_passthrough"] == {
        "samples": 1,
        "raw_exact": 1,
        "normalized_exact": 1,
        "sha256_exact": 1,
        "raw_results_sha256": hashlib.sha256(paths["raw_results"].read_bytes()).hexdigest(),
    }
    assert report["rows"][0]["silver_agreement_cer"] == 0
    assert report["rows"][0]["device_runtime"].startswith("test platform")
    assert report["rows"][0]["cost"] == "test quota"
    assert {row["model"] for row in report["baselines"]} == {"Apple Vision", "Tesseract PSM 6"}
    apple = next(row for row in report["baselines"] if row["model"] == "Apple Vision")
    assert apple["samples"] == 1
    assert apple["silver_text_score"] == 90.0
    assert apple["silver_agreement_cer"] == 0.1
    assert apple["cohort"].endswith("(n=1)")
    first = write_outputs(report, tmp_path / "out")
    hashes_before = hash_paths(first)
    second = write_outputs(report, tmp_path / "out")
    assert hashes_before == hash_paths(second)
    assert "full_silver_baseline" in (tmp_path / "out/comparison.csv").read_text()
    raw_csv = write_raw_csv(paths["raw_results"], tmp_path / "out" / "measured-raw.csv")
    assert "org/model-1" in raw_csv.read_text(encoding="utf-8")


def test_silver_baseline_registry_accepts_additional_full_cohort_engine(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    additional = copy.deepcopy(baselines["silver_baselines"]["tesseract_psm6"])
    source_path = paths["baselines_path"].parent / additional["evaluation_artifact"]["path"]
    evaluation = json.loads(source_path.read_text(encoding="utf-8"))
    evaluation["sources"]["prediction"]["engine_label"] = "PP-OCRv5 mobile"
    evaluation["sources"]["prediction"]["engines"] = ["paddleocr-ppocrv5-mobile"]
    evaluation["sources"]["prediction"]["sha256"] = "1" * 64
    ppocr_path = source_path.with_name("ppocrv5.json")
    _json(ppocr_path, evaluation)
    additional["model"] = "PP-OCRv5 mobile"
    additional["engine_label"] = "PP-OCRv5 mobile"
    additional["engines"] = ["paddleocr-ppocrv5-mobile"]
    additional["prediction_artifact_sha256"] = "1" * 64
    additional["evaluation_artifact"] = {
        "path": "raw/silver/ppocrv5.json",
        "sha256": hashlib.sha256(ppocr_path.read_bytes()).hexdigest(),
    }
    additional["valid_ocr_records"] = 0
    audit_path = source_path.with_name("ppocrv5-audit.json")
    runtime_path = source_path.with_name("ppocrv5-runtime.json")
    _json(
        runtime_path,
        {"result_sha256": "1" * 64, "record_count": 1, "failed_count": 0},
    )
    _json(
        audit_path,
        {
            "prediction": {"sha256": "1" * 64},
            "runtime": {
                "path": "ppocrv5-runtime.json",
                "sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
                "declared_result_sha256_matches": True,
            },
            "coverage": {
                "expected_instance_ids": 1,
                "observed_instance_ids": 1,
                "unique_instance_ids": 1,
                "missing_instance_ids": [],
                "extra_instance_ids": [],
            },
            "canonical_merge": {
                "status": "failed",
                "error": "OCR record blocks must be ordered by block_id",
                "strict_valid_records": 0,
            },
            "evaluation": {
                "evaluation_sha256": additional["evaluation_artifact"]["sha256"],
                "semantic_correction_applied": False,
            },
        },
    )
    additional["ingestion_audit_artifact"] = {
        "path": "raw/silver/ppocrv5-audit.json",
        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    }
    additional["output_bytes"] = 6_325_899
    additional["row_type"] = "full_silver_diagnostic"
    additional["candidate_gate_outcome"] = "fail_strict_schema"
    additional["selection_status"] = "evaluated_diagnostic"
    additional["runner_gate"] = "raw_217_coverage_strict_merge_failed"
    additional["notes"] = "one malformed block marker; raw text scored without correction"
    baselines["silver_baselines"]["ppocrv5_mobile"] = additional
    _json(paths["baselines_path"], baselines)

    report = _evaluate(paths)

    assert [row["model"] for row in report["baselines"]] == [
        "Apple Vision",
        "Tesseract PSM 6",
        "PP-OCRv5 mobile",
    ]
    assert "PP-OCRv5 mobile" in report["comparison_scope"]
    ppocr = report["baselines"][2]
    assert ppocr["inference_success_rate"] == 1.0
    assert ppocr["valid_ocr_rate"] == 0.0
    assert ppocr["output_bytes"] == 6_325_899
    assert ppocr["candidate_gate_outcome"] == "fail_strict_schema"
    assert ppocr["selection_status"] == "evaluated_diagnostic"
    assert ppocr["runner_gate"] == "raw_217_coverage_strict_merge_failed"
    assert ppocr["notes"] == "one malformed block marker; raw text scored without correction"
    write_outputs(report, tmp_path / "out")
    assert "PP-OCRv5 mobile" in (tmp_path / "out/comparison.md").read_text(encoding="utf-8")


def test_silver_baseline_engine_identity_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    baselines["silver_baselines"]["apple_vision"]["engines"] = ["other-engine"]
    _json(paths["baselines_path"], baselines)

    with pytest.raises(ValueError, match="engine identity mismatch"):
        _evaluate(paths)


def test_silver_diagnostic_requires_bound_ingestion_audit(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    baseline = baselines["silver_baselines"]["apple_vision"]
    baseline["valid_ocr_records"] = 0
    baseline["row_type"] = "full_silver_diagnostic"
    baseline["candidate_gate_outcome"] = "fail_strict_schema"
    baseline["selection_status"] = "evaluated_diagnostic"
    _json(paths["baselines_path"], baselines)

    with pytest.raises(ValueError, match="missing ingestion audit identity"):
        _evaluate(paths)


def test_silver_diagnostic_cannot_be_published_as_operational_baseline(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    baseline = baselines["silver_baselines"]["apple_vision"]
    baseline["valid_ocr_records"] = 0
    baseline["row_type"] = "full_silver_baseline"
    baseline["candidate_gate_outcome"] = "operational_baseline"
    baseline["selection_status"] = "operational_baseline"
    _json(paths["baselines_path"], baselines)

    with pytest.raises(ValueError, match="invalid silver diagnostic workflow metadata"):
        _evaluate(paths)


def test_silver_baseline_artifacts_fail_closed_on_hash_and_source_mismatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    baselines["silver_baselines"]["apple_vision"]["evaluation_artifact"]["sha256"] = "0" * 64
    _json(paths["baselines_path"], baselines)
    with pytest.raises(ValueError, match="evaluation SHA-256 mismatch"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "source")
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    baselines["silver_baselines"]["apple_vision"]["prediction_artifact_sha256"] = "0" * 64
    _json(paths["baselines_path"], baselines)
    with pytest.raises(ValueError, match="prediction mismatch"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "coverage")
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    evaluation_artifact = baselines["silver_baselines"]["apple_vision"]["evaluation_artifact"]
    evaluation_path = paths["baselines_path"].parent / evaluation_artifact["path"]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["instances"] = []
    _json(evaluation_path, evaluation)
    evaluation_artifact["sha256"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    _json(paths["baselines_path"], baselines)
    with pytest.raises(ValueError, match="instance coverage|instance count"):
        _evaluate(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evaluation: evaluation["instances"][0].update(reference_status="failed"),
            "reference status coverage",
        ),
        (
            lambda evaluation: evaluation["summary"].update(reference_ok=0),
            "reference status coverage",
        ),
        (
            lambda evaluation: evaluation["instances"][0].update(prediction_status="failed"),
            "prediction status summary",
        ),
        (
            lambda evaluation: evaluation["summary"].update(prediction_ok=0, prediction_failed=1),
            "prediction status summary",
        ),
    ],
)
def test_silver_baseline_status_summaries_fail_closed_after_rehash(
    tmp_path: Path, mutation, message: str
) -> None:
    paths = _fixture(tmp_path)
    baselines = json.loads(paths["baselines_path"].read_text(encoding="utf-8"))
    evaluation_artifact = baselines["silver_baselines"]["apple_vision"]["evaluation_artifact"]
    evaluation_path = paths["baselines_path"].parent / evaluation_artifact["path"]
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    mutation(evaluation)
    _json(evaluation_path, evaluation)
    evaluation_artifact["sha256"] = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    _json(paths["baselines_path"], baselines)

    with pytest.raises(ValueError, match=message):
        _evaluate(paths)


def test_query_passthrough_uses_independent_join_and_validates_keys(tmp_path: Path) -> None:
    tasks = _write(tmp_path / "tasks.jsonl", '{"instance_id":"x","user_query":"A  B\\t?"}\n')
    binding = {"kind": "raw_results_sha256", "sha256": "a" * 64}
    joined = _write(
        tmp_path / "joined.jsonl",
        json.dumps(
            {"instance_id": "x", "user_query": "A  B\t?", "evidence_binding": binding},
            sort_keys=True,
        )
        + "\n",
    )
    assert query_passthrough(tasks, joined, "a" * 64)["raw_exact"] == 1

    _write(joined, '{"instance_id":"x","user_query":"changed"}\n')
    with pytest.raises(ValueError, match="differs"):
        query_passthrough(tasks, joined, "a" * 64)

    _write(joined, '{"instance_id":"y","user_query":"A  B\\t?"}\n')
    with pytest.raises(ValueError, match="keys differ"):
        query_passthrough(tasks, joined, "a" * 64)

    _write(
        joined,
        '{"instance_id":"x","user_query":"A  B\\t?"}\n'
        '{"instance_id":"x","user_query":"A  B\\t?"}\n',
    )
    with pytest.raises(ValueError, match="duplicate"):
        query_passthrough(tasks, joined, "a" * 64)


@pytest.mark.parametrize("filename", ["missing.jsonl", "empty.jsonl"])
def test_required_jsonl_fails_closed(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    if filename == "empty.jsonl":
        _write(path, "\n")
    with pytest.raises((FileNotFoundError, ValueError)):
        read_jsonl(path)


def test_evaluate_rejects_duplicate_results_missing_output_and_byte_mismatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    original = paths["raw_results"].read_text()
    _write(paths["raw_results"], original + original)
    with pytest.raises(ValueError, match="duplicate"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "missing")
    (paths["raw_dir"] / "model-1-page-1.txt").unlink()
    with pytest.raises(FileNotFoundError, match="declared raw output"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "bytes")
    rows = _result_rows(paths["raw_results"])
    rows[0]["raw_output_bytes"] += 1
    _write_result_rows(paths["raw_results"], rows)
    with pytest.raises(ValueError, match="byte mismatch"):
        _evaluate(paths)


def test_evaluate_rejects_missing_reference_and_revision_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write(
        paths["reference_path"],
        '{"instance_id":"other","status":"failed","blocks":[],"timing":{}}\n',
    )
    paths["expected_reference_sha256"] = hashlib.sha256(
        paths["reference_path"].read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="fixed reference"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "revision")
    rows = _result_rows(paths["raw_results"])
    rows[0]["resolved_revision"] = "wrong"
    _write_result_rows(paths["raw_results"], rows)
    with pytest.raises(ValueError, match="revision mismatch"):
        _evaluate(paths)


def test_evaluate_v2_uses_parent_sampled_peaks_and_validates_output_hash(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = _result_rows(paths["raw_results"])
    row = _evaluate(paths)["rows"][0]
    assert row["peak_ram_bytes"] == 11
    assert row["peak_ram_bytes_child"] == 30
    assert row["peak_vram_bytes"] == 21
    assert row["peak_vram_bytes_child_allocated"] == 20

    paths = _fixture(tmp_path / "hash")
    rows = _result_rows(paths["raw_results"])
    rows[0]["raw_outputs"][0]["sha256"] = "0" * 64
    rows[0]["raw_output_sha256"][0] = "0" * 64
    _write_result_rows(paths["raw_results"], rows)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _evaluate(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[0].update(schema_version="1.0"), "schema"),
        (lambda rows: rows[0].update(instance_id="other"), "instance"),
        (lambda rows: rows[0].update(requested_revision="wrong"), "revision"),
        (lambda rows: rows[0].update(repo_metadata_error="offline"), "metadata failed"),
        (lambda rows: rows[0].update(peak_process_rss_bytes_parent_sampled=None), "RSS"),
        (lambda rows: rows[0].update(peak_vram_bytes_parent_sampled=None), "VRAM"),
        (lambda rows: rows[0].update(peak_process_rss_bytes_parent_sampled=True), "RSS"),
        (lambda rows: rows[0].update(peak_process_rss_bytes_parent_sampled=-1), "RSS"),
        (lambda rows: rows[0].update(peak_vram_bytes_parent_sampled=True), "VRAM"),
        (lambda rows: rows[0].update(peak_vram_bytes_parent_sampled=-1), "VRAM"),
        (
            lambda rows: rows[0].update(peak_process_rss_sampling_error="sample failed"),
            "sampling identity",
        ),
        (
            lambda rows: rows[0].update(peak_vram_sampling_error="sample failed"),
            "sampling identity",
        ),
        (lambda rows: rows[0].pop("peak_process_rss_sampling_error"), "sampling identity"),
        (lambda rows: rows[0].pop("peak_vram_sampling_error"), "sampling identity"),
        (lambda rows: rows[0].update(success="true"), "exact boolean"),
        (lambda rows: rows.pop(), "exactly 5"),
    ],
)
def test_evaluate_v2_identity_and_parent_samples_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    paths = _fixture(tmp_path)
    rows = _result_rows(paths["raw_results"])
    mutation(rows)
    _write_result_rows(paths["raw_results"], rows)
    with pytest.raises((TypeError, ValueError), match=message):
        _evaluate(paths)


def test_evaluate_rejects_stale_join_and_non_codex_reference(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write(paths["joined_tasks_path"], '{"instance_id":"task_000909","user_query":"Keep  this?"}\n')
    with pytest.raises(ValueError, match="stale, prebuilt"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "reference")
    references = [json.loads(line) for line in paths["reference_path"].read_text().splitlines()]
    references[0]["engine"] = "other"
    _write_result_rows(paths["reference_path"], references)
    paths["expected_reference_sha256"] = hashlib.sha256(
        paths["reference_path"].read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="unsupported identity"):
        _evaluate(paths)


def test_evaluate_rejects_lfs_mismatch_and_outputs_on_failure(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = _result_rows(paths["raw_results"])
    rows[0]["repo_file_metadata"][0]["lfs_sha256"] = "0" * 64
    _write_result_rows(paths["raw_results"], rows)
    with pytest.raises(ValueError, match="LFS identity mismatch"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "failure-output")
    rows = _result_rows(paths["raw_results"])
    rows[1]["raw_outputs"] = [{"path": "ghost", "bytes": 1, "sha256": "0" * 64}]
    rows[1]["raw_output_sha256"] = ["0" * 64]
    rows[1]["raw_output_bytes"] = 1
    _write_result_rows(paths["raw_results"], rows)
    with pytest.raises(ValueError, match="failed result must not declare"):
        _evaluate(paths)


@pytest.mark.parametrize("page_mutation", ["missing", "duplicate", "reversed"])
def test_evaluate_rejects_missing_or_duplicate_success_page_identities(
    tmp_path: Path, page_mutation: str
) -> None:
    paths = _fixture(tmp_path)
    rows = _result_rows(paths["raw_results"])
    if page_mutation == "missing":
        rows[0]["raw_outputs"][0].pop("page")
    elif page_mutation == "duplicate":
        rows[0]["raw_outputs"][1]["page"] = 1
    else:
        rows[0]["raw_outputs"].reverse()
        rows[0]["raw_output_sha256"].reverse()
    _write_result_rows(paths["raw_results"], rows)
    with pytest.raises(ValueError, match=r"page sequence must be exactly \[1, 2\]"):
        _evaluate(paths)


def test_raw_output_failure_never_creates_or_modifies_join(tmp_path: Path) -> None:
    absent = _fixture(tmp_path / "absent")
    _write(absent["raw_dir"] / "model-1-page-1.txt", "corrupt")
    with pytest.raises(ValueError, match="byte mismatch"):
        _evaluate(absent)
    assert not absent["joined_tasks_path"].exists()

    existing = _fixture(tmp_path / "existing")
    sentinel = b"preexisting join must remain byte-exact\n"
    existing["joined_tasks_path"].write_bytes(sentinel)
    _write(existing["raw_dir"] / "model-1-page-2.txt", "corrupt")
    with pytest.raises(ValueError, match="byte mismatch"):
        _evaluate(existing)
    assert existing["joined_tasks_path"].read_bytes() == sentinel


@pytest.mark.parametrize("expected", ["bad", "A" * 64, "0" * 64])
def test_reference_artifact_hash_is_required_and_exact(tmp_path: Path, expected: str) -> None:
    paths = _fixture(tmp_path)
    paths["expected_reference_sha256"] = expected
    with pytest.raises(ValueError, match="reference.*SHA-256"):
        _evaluate(paths)


def test_generate_cli_requires_reference_sha256() -> None:
    arguments = ["generate"]
    for name in (
        "raw-results",
        "raw-dir",
        "candidates",
        "reference",
        "tasks",
        "joined-tasks",
        "environment",
        "baselines",
        "out-dir",
    ):
        arguments.extend((f"--{name}", name))
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)
    parsed = build_parser().parse_args([*arguments, "--reference-sha256", "0" * 64])
    assert parsed.reference_sha256 == "0" * 64
