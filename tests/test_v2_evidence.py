import hashlib
import json
import os
import runpy
import statistics
from pathlib import Path

import pytest

from docinsights_hf_ocr.evaluation import evaluate, write_outputs, write_raw_csv

ROOT = Path("research/ocr-small-models")
V2 = ROOT / "raw/v2"
SILVER = ROOT / "raw/silver"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_textual_bundle_matches_embedded_artifact_manifest() -> None:
    assert _sha256(V2 / "artifact-hashes.json") == (
        "47b8606af538115799eaecdfd1f6433ead64f1bc892f465d5e5b3d6323f447c7"
    )
    manifest = json.loads((V2 / "artifact-hashes.json").read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 34
    prefix = "/kaggle/working/docinsights-hf-smoke-v2/"
    omitted = set()
    checked = 0
    for artifact in manifest["artifacts"]:
        relative = artifact["path"].removeprefix(prefix)
        local = V2 / relative
        if relative.startswith("pages/"):
            omitted.add(relative)
            assert not local.exists()
            continue
        assert local.is_file(), relative
        assert local.stat().st_size == artifact["bytes"]
        assert _sha256(local) == artifact["sha256"]
        checked += 1
    assert checked == 32
    assert omitted == {
        "pages/task_000909-page-1.png",
        "pages/task_000909-page-2.png",
    }


def test_v2_results_link_to_raw_outputs_logs_revisions_and_model_files() -> None:
    rows = [json.loads(line) for line in (V2 / "results.jsonl").read_text().splitlines()]
    candidates = {
        row["model"]: row
        for row in json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))["models"]
    }
    for row in rows:
        candidate = candidates[row["repo"]]
        assert row["schema_version"] == "2.0"
        assert row["resolved_revision"] == candidate["revision"]
        model_file = next(
            item for item in row["repo_file_metadata"] if item["rfilename"] == "model.safetensors"
        )
        assert model_file["lfs_size"] == candidate["weight_bytes"]
        assert model_file["lfs_sha256"] == candidate["weight_lfs_sha256"]
        for output in row["raw_outputs"]:
            path = V2 / "raw" / Path(output["path"]).name
            assert path.stat().st_size == output["bytes"]
            assert _sha256(path) == output["sha256"]
        for stream in ("stdout", "stderr"):
            if row[f"{stream}_path"] is not None:
                path = V2 / "logs" / Path(row[f"{stream}_path"]).name
                assert path.stat().st_size == row[f"{stream}_bytes"]
                assert _sha256(path) == row[f"{stream}_sha256"]


def test_executed_runner_is_immutable_and_environment_snapshot_is_verbatim() -> None:
    environment = json.loads((V2 / "environment.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((V2 / "run-manifest.json").read_text(encoding="utf-8"))
    runner = V2 / "runner-executed.py"
    expected = "4e5be04c3afb6d487b547765a813e9737047cafa18df1882705a06b57ca728e3"
    assert _sha256(runner) == expected
    assert environment["runner"]["sha256"] == expected
    assert run_manifest["started_from_exact_runner_sha256"] == expected
    assert run_manifest["run_id"] == "task_000909-1788113789-4e5be04c3afb"
    assert Path("notebooks/docsem_hf_small_ocr_smoke_v2.py").read_bytes() == (
        runner.read_bytes() + b"\n"
    )
    snapshot = environment["pip_freeze_all_verbatim"]
    assert (V2 / "pip-freeze.txt").read_text() == snapshot
    assert (ROOT / "requirements-kaggle-v2.txt").read_text() == snapshot
    manifest = json.loads((ROOT / "manifests/environment.json").read_text())
    assert manifest["bundle_artifacts"]["zip_sha256"] == (
        "08995ebc6283c082fd9add596412a870910d2a875f63948cf1ae824939d2ec17"
    )
    assert manifest["bundle_artifacts"]["artifact_count"] == 34
    assert "environment_snapshot" in manifest
    assert "not a reconstructible lock" in manifest["environment_snapshot"]["note"]


def test_current_runner_fails_closed_and_reports_resource_columns_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path("notebooks/docsem_hf_small_ocr_smoke_v2.py")
    empty = namespace["empty_result"](namespace["SPECS"][0], 1)
    assert empty["resolved_revision"] is None
    assert empty["peak_process_rss_sampling_error"] is None
    assert empty["peak_vram_sampling_error"] is None
    report = namespace["build_report"]([empty], "test-run")
    assert "parent RSS B | child RSS B | parent VRAM B | child allocated VRAM B" in report
    monkeypatch.setattr(namespace["shutil"], "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="nvidia-smi is unavailable"):
        namespace["sampled_vram_bytes"](123)
    source = Path("notebooks/docsem_hf_small_ocr_smoke_v2.py").read_text()
    assert "repository identity metadata failed" in source
    assert '"resolved_revision": spec["revision"]' not in source


def test_selection_gate_has_four_selected_and_glm_diagnostic_rejection() -> None:
    data = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    selected = [row for row in data["models"] if row["selection_status"] == "selected"]
    assert {row["model"] for row in selected} == set(data["selected_models"])
    assert len(selected) == 4
    assert all(row["gate_outcome"] == "pass" for row in selected)
    assert all(row["trendingScore"] is None for row in data["models"])
    assert all(row["languages"] and row["document_traits"] for row in data["models"])
    assert all("install_size_bytes" in row for row in data["models"])
    assert all(
        set(row["feasibility"]) == {"cuda_gpu", "cpu", "apple_silicon"} for row in data["models"]
    )
    assert all(
        row["model_card_url"].startswith("https://huggingface.co/") for row in data["models"]
    )
    assert all(
        sum(item["score"] for item in row["rubric"].values()) == row["score_total"]
        for row in data["models"]
    )
    glm = next(row for row in data["models"] if row["model"] == "zai-org/GLM-OCR")
    assert glm["params"] == 1325258240
    assert glm["gate_outcome"] == "fail_params"
    assert glm["selection_status"] == "not_selected_diagnostic"


def test_full_silver_baselines_are_scorer_outputs_with_pinned_sources() -> None:
    expected = {
        "apple-vision-evaluation.json": {
            "evaluation_sha256": "5e7a85338f58ad766cdcc0353e5bd9e45e3a32a4394d41f73d0c20751fb32645",
            "prediction_sha256": "8d55f10f9f628cdc6744f451d1c04de5158495a6452ae123d0ff9670d1908c01",
            "prediction_label": "issue8/apple-vision-200dpi.jsonl",
            "engine_label": "Apple Vision accurate 200 DPI",
            "engines": ["apple-vision"],
            "score": 99.37773767034393,
            "cer": 0.0062226232965606745,
            "wer": 0.00894595377474789,
        },
        "tesseract-evaluation.json": {
            "evaluation_sha256": "3db904ee7e4278b101915fbb701ecf4b38025e105c5d51033290f57e52446e49",
            "prediction_sha256": "8b5db676267a0a1ab51c345798994eb5f38f4b5148728e54adbb40cf94acadaf",
            "prediction_label": "issue8/tesseract-200dpi-psm6-final.jsonl",
            "engine_label": "Tesseract eng PSM 6 200 DPI",
            "engines": ["tesseract-tsv"],
            "score": 99.94149497079819,
            "cer": 0.00058505029201817,
            "wer": 0.006029087822589881,
        },
        "ppocrv5-kaggle-evaluation.json": {
            "evaluation_sha256": "359ea3dd74f7995e2c710da80165134fad3147917587e0658d9efffa2808fb47",
            "prediction_sha256": "60e1844155e70fc5f4cea218e86be4ac2e6ca9fa35d4699fc820c568231c0fd1",
            "prediction_label": "kaggle/version-3/result-shard-00-of-01.jsonl",
            "engine_label": "PP-OCRv5 mobile (Kaggle CPU)",
            "engines": ["paddleocr-ppocrv5-mobile"],
            "score": 99.61763870863076,
            "cer": 0.0038236129136924074,
            "wer": 0.014463074363240753,
        },
    }
    for filename, values in expected.items():
        path = SILVER / filename
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        assert _sha256(path) == values["evaluation_sha256"]
        assert evaluation["interpretation"] == "silver_agreement_not_human_gold_accuracy"
        assert evaluation["sources"]["reference"] == {
            "path": "issue8/codex-validation-reference.jsonl",
            "records": 217,
            "sha256": "d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a",
        }
        assert evaluation["sources"]["prediction"]["path"] == values["prediction_label"]
        assert evaluation["sources"]["prediction"]["sha256"] == values["prediction_sha256"]
        assert evaluation["sources"]["prediction"]["engine_label"] == values["engine_label"]
        assert evaluation["sources"]["prediction"]["engines"] == values["engines"]
        assert evaluation["summary"]["instances"] == 217
        assert evaluation["summary"]["prediction_ok"] == 217
        assert evaluation["summary"]["silver_text_score"] == values["score"]
        assert evaluation["summary"]["micro_character_error_rate"] == values["cer"]
        assert evaluation["summary"]["micro_word_error_rate"] == values["wer"]

    ppocr = json.loads((SILVER / "ppocrv5-kaggle-evaluation.json").read_text())
    assert ppocr["summary"]["ordered_block_exact_count"] == 216
    assert ppocr["summary"]["exact_token_f1"] == 0.9966111471627371
    assert ppocr["summary"]["ordered_quantity_f1"] == 0.9964476021314386
    ppocr_instances = ppocr["instances"]
    character_distance = sum(row["strict_text"]["edit_distance"] for row in ppocr_instances)
    reference_characters = sum(
        row["strict_text"]["reference_characters"] for row in ppocr_instances
    )
    word_distance = sum(row["words"]["edit_distance"] for row in ppocr_instances)
    reference_words = sum(row["words"]["reference_words"] for row in ppocr_instances)
    assert ppocr["summary"]["micro_character_error_rate"] == pytest.approx(
        character_distance / reference_characters
    )
    assert ppocr["summary"]["micro_word_error_rate"] == pytest.approx(
        word_distance / reference_words
    )
    assert ppocr["summary"]["mean_block_f1"] == pytest.approx(
        statistics.fmean(row["blocks"]["f1"] for row in ppocr_instances)
    )
    assert ppocr["summary"]["ordered_block_exact_count"] == sum(
        row["blocks"]["ordered_exact"] for row in ppocr_instances
    )
    latencies = [row["total_seconds"] for row in ppocr_instances]
    assert ppocr["summary"]["latency"]["mean_seconds_per_document"] == pytest.approx(
        statistics.fmean(latencies)
    )
    runtime = SILVER / "ppocrv5-kaggle-runtime.json"
    assert _sha256(runtime) == "913b5b5e80a3e8a23f2542a23978f255ee1a4e2b93f8847965984e3bdc6d0a48"
    runtime_data = json.loads(runtime.read_text())
    assert runtime_data["record_count"] == 217
    assert runtime_data["failed_count"] == 0
    assert (
        runtime_data["result_sha256"]
        == expected["ppocrv5-kaggle-evaluation.json"]["prediction_sha256"]
    )
    audit = json.loads((SILVER / "ppocrv5-kaggle-ingestion-audit.json").read_text())
    assert audit["prediction"]["sha256"] == runtime_data["result_sha256"]
    assert audit["coverage"] == {
        "expected_instance_ids": 217,
        "observed_instance_ids": 217,
        "unique_instance_ids": 217,
        "missing_instance_ids": [],
        "extra_instance_ids": [],
    }
    assert audit["canonical_merge"] == {
        "status": "failed",
        "error": "OCR record blocks must be ordered by block_id",
        "failing_instance_id": "task_001108",
        "reference_block_id": "b09",
        "observed_block_id": "b0",
        "block_text_equal_to_reference": True,
        "strict_valid_records": 216,
    }
    assert audit["evaluation"]["semantic_correction_applied"] is False

    comparison = json.loads((ROOT / "generated/comparison.json").read_text(encoding="utf-8"))
    assert comparison["cross_cohort_quality_ranking_allowed"] is False
    assert {row["samples"] for row in comparison["rows"]} == {1}
    assert {row["samples"] for row in comparison["baselines"]} == {217}
    assert {row["quality_samples"] for row in comparison["baselines"]} == {217}
    ppocr_row = next(
        row for row in comparison["baselines"] if row["row_type"] == "full_silver_diagnostic"
    )
    assert ppocr_row["valid_ocr_rate"] == 216 / 217
    assert ppocr_row["runner_gate"].endswith("task_001108_b09_to_b0")


def test_pinned_model_revisions_parameters_oids_and_remote_code_audit() -> None:
    models = {
        row["model"]: row
        for row in json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))["models"]
    }
    expected = {
        "PaddlePaddle/PaddleOCR-VL-1.6": (
            "c5630abae1d940eafe0697512a0325494b02ab42",
            958588736,
            1917255968,
            "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db",
        ),
        "zai-org/GLM-OCR": (
            "ca5d8b3e287e52589e37c28385d9655ee4372f9d",
            1325258240,
            2650579464,
            "a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815",
        ),
        "datalab-to/surya-ocr-2": (
            "3b3d4cdf88d6928b0acdc75181b13206ea67c4a3",
            686154304,
            1372368672,
            "5755f82a997dd0b111964fa8b31cc2daef7aeb7a706bbd17d73d6a93ef3f723e",
        ),
        "ibm-granite/granite-docling-258M": (
            "982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe",
            257517120,
            515093104,
            "1cdad234deb1cde18ee6a586f849057f19851daf1fedce2e40aff791dbe46f61",
        ),
        "docling-project/SmolDocling-256M-preview": (
            "ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8",
            256484928,
            513028808,
            "cdcdf5d823c5684029c7d8e52177cf10f9034b3aba6577549cfb1a9ce36ad0a2",
        ),
    }
    for model, (revision, params, weight_bytes, oid) in expected.items():
        row = models[model]
        assert (row["revision"], row["params"], row["weight_bytes"], row["weight_lfs_sha256"]) == (
            revision,
            params,
            weight_bytes,
            oid,
        )
    paddle_audit = models["PaddlePaddle/PaddleOCR-VL-1.6"]["remote_code_audit"]
    assert paddle_audit["result"] == "reviewed_pinned_sources_runtime_failed"
    assert {item["filename"]: item["blob_id"] for item in paddle_audit["source_files"]} == {
        "configuration_paddleocr_vl.py": "a8fd139287293301b287db6dfdaac21d7ad1a236",
        "image_processing_paddleocr_vl.py": "f7e28fd3c1971331581f73d573b9c4a2a4ce7a58",
        "modeling_paddleocr_vl.py": "693782514116586458b12cfd911c88d6565f552c",
        "processing_paddleocr_vl.py": "73c3faeff201555fc7b52709848e3c669419dbb1",
    }


def test_current_inputs_reproduce_checked_in_generated_outputs(tmp_path: Path) -> None:
    reference_value = os.environ.get("ISSUE8_REFERENCE")
    reference_sha256 = os.environ.get("ISSUE8_REFERENCE_SHA256")
    if not reference_value or not reference_sha256:
        pytest.skip("set ISSUE8_REFERENCE and ISSUE8_REFERENCE_SHA256 for reproduction")
    reference = Path(reference_value)
    assert reference.is_file()
    assert _sha256(reference) == reference_sha256
    current_rows = [json.loads(line) for line in (V2 / "results.jsonl").read_text().splitlines()]
    required_sampling_fields = {
        "peak_process_rss_sampling_error",
        "peak_vram_sampling_error",
    }
    assert all(required_sampling_fields <= row.keys() for row in current_rows)
    assert all(
        row["peak_process_rss_sampling_error"] is None and row["peak_vram_sampling_error"] is None
        for row in current_rows
    )
    generated = tmp_path / "generated"
    report = evaluate(
        V2 / "results.jsonl",
        V2 / "raw",
        ROOT / "candidates.json",
        reference,
        ROOT / "manifests/source-queries.jsonl",
        generated / "joined-queries.jsonl",
        ROOT / "manifests/environment.json",
        ROOT / "baselines.json",
        reference_sha256,
    )
    outputs = write_outputs(report, generated)
    outputs.append(write_raw_csv(V2 / "results.jsonl", generated / "measured-raw.csv"))
    for output in outputs:
        assert output.read_bytes() == (ROOT / "generated" / output.name).read_bytes()
    assert (generated / "joined-queries.jsonl").read_bytes() == (
        ROOT / "generated/joined-queries.jsonl"
    ).read_bytes()
