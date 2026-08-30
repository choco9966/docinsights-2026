import hashlib
import json
import os
import runpy
from pathlib import Path

import pytest

from docinsights_hf_ocr.evaluation import evaluate, write_outputs, write_raw_csv

ROOT = Path("research/ocr-small-models")
V2 = ROOT / "raw/v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_textual_bundle_matches_embedded_artifact_manifest() -> None:
    assert _sha256(V2 / "artifact-hashes.json") == (
        "5fe83dc767facd523bed6a2b29ad0ab4b4815642fcbf064b1de0c9bee9c07b9e"
    )
    manifest = json.loads((V2 / "artifact-hashes.json").read_text(encoding="utf-8"))
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
    expected = "64706348c218f729e94430ab0fa4b33e9ec6467e41f05e665731a3a7c78644cf"
    assert _sha256(runner) == expected
    assert environment["runner"]["sha256"] == expected
    assert run_manifest["started_from_exact_runner_sha256"] == expected
    assert Path("notebooks/docsem_hf_small_ocr_smoke_v2.py").read_bytes() != runner.read_bytes()
    snapshot = environment["pip_freeze_all_verbatim"]
    assert (V2 / "pip-freeze.txt").read_text() == snapshot
    assert (ROOT / "requirements-kaggle-v2.txt").read_text() == snapshot
    manifest = json.loads((ROOT / "manifests/environment.json").read_text())
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
    if not all(required_sampling_fields <= row.keys() for row in current_rows):
        pytest.xfail("checked-in raw v2 awaits rerun with explicit sampling-error identities")
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
