import hashlib
import json
from pathlib import Path

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


def test_runner_hash_lock_and_checked_in_source_linkage() -> None:
    environment = json.loads((V2 / "environment.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((V2 / "run-manifest.json").read_text(encoding="utf-8"))
    runner = V2 / "runner-executed.py"
    expected = "64706348c218f729e94430ab0fa4b33e9ec6467e41f05e665731a3a7c78644cf"
    assert _sha256(runner) == expected
    assert environment["runner"]["sha256"] == expected
    assert run_manifest["started_from_exact_runner_sha256"] == expected
    assert (
        Path("notebooks/docsem_hf_small_ocr_smoke_v2.py").read_bytes()
        == runner.read_bytes() + b"\n"
    )
    lock = environment["pip_freeze_all_verbatim"]
    assert (V2 / "pip-freeze.txt").read_text() == lock
    assert (ROOT / "requirements-kaggle-v2.txt").read_text() == lock


def test_selection_gate_has_four_selected_and_glm_diagnostic_rejection() -> None:
    data = json.loads((ROOT / "candidates.json").read_text(encoding="utf-8"))
    selected = [row for row in data["models"] if row["selection_status"] == "selected"]
    assert {row["model"] for row in selected} == set(data["selected_models"])
    assert len(selected) == 4
    assert all(row["gate_outcome"] == "pass" for row in selected)
    assert all(row["trendingScore"] is None for row in data["models"])
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
