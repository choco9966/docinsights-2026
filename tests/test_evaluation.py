import hashlib
import json
from pathlib import Path

import pytest

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
    joined = _write(
        tmp_path / "joined.jsonl", '{"instance_id":"task_000909","user_query":"Keep  this?"}\n'
    )
    refs = _write(
        tmp_path / "refs.jsonl",
        json.dumps(
            {
                "instance_id": "task_000909",
                "status": "ok",
                "blocks": [{"block_id": "b01", "text": "Alpha"}],
                "timing": {"total_seconds": 2},
            }
        )
        + "\n"
        + json.dumps(
            {
                "instance_id": "task_other",
                "status": "ok",
                "blocks": [],
                "timing": {"total_seconds": 3},
            }
        )
        + "\n",
    )
    candidates = _json(
        tmp_path / "candidates.json",
        {
            "models": [
                {
                    "model": "org/model",
                    "revision": "abc",
                    "params": 1,
                    "weight_gib": 0.1,
                    "license": "MIT",
                }
            ]
        },
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
    baselines = _json(
        tmp_path / "baselines.json",
        {
            "existing_ocr_operational_comparison": {
                "documents": 2,
                "pages": 4,
                "blocks": 6,
                "failures": 0,
                "apple_vision": {
                    "total_seconds": 10,
                    "seconds_per_document": 5,
                    "peak_rss_bytes": 100,
                },
                "tesseract_psm6": {
                    "total_seconds": 20,
                    "seconds_per_document": 10,
                    "peak_rss_bytes": 50,
                },
                "engine_agreement_not_accuracy": {"cer": 0.1, "wer": 0.2, "block_f1": 1},
            }
        },
    )
    raw_dir = tmp_path / "raw"
    raw_output = _write(raw_dir / "model-page-1.txt", "b01: Alpha")
    results = _write(
        tmp_path / "results.jsonl",
        json.dumps(
            {
                "name": "model",
                "repo": "org/model",
                "revision": "abc",
                "success": True,
                "raw_output_paths": ["/remote/model-page-1.txt"],
                "doc_latency_sec": 2,
                "peak_process_rss_bytes": 10,
                "peak_cuda_allocated_bytes": 20,
                "raw_output_bytes": raw_output.stat().st_size,
            }
        )
        + "\n",
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
    }


def _evaluate(paths: dict[str, Path]):
    return evaluate(**paths)


def test_dynamic_reference_refresh_join_baselines_and_determinism(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = _evaluate(paths)
    assert report["reference"]["available_validated_subset"] == 2
    assert report["reference"]["total_seconds_when_present"] == 5
    assert report["query_passthrough"] == {
        "samples": 1,
        "raw_exact": 1,
        "normalized_exact": 1,
        "sha256_exact": 1,
    }
    assert report["rows"][0]["silver_agreement_cer"] == 0
    assert report["rows"][0]["device_runtime"].startswith("test platform")
    assert report["rows"][0]["cost"] == "test quota"
    assert {row["model"] for row in report["baselines"]} == {"Apple Vision", "Tesseract PSM 6"}
    first = write_outputs(report, tmp_path / "out")
    hashes_before = hash_paths(first)
    second = write_outputs(report, tmp_path / "out")
    assert hashes_before == hash_paths(second)
    assert "operational_baseline" in (tmp_path / "out/comparison.csv").read_text()
    raw_csv = write_raw_csv(paths["raw_results"], tmp_path / "out" / "measured-raw.csv")
    assert "org/model" in raw_csv.read_text(encoding="utf-8")


def test_query_passthrough_uses_independent_join_and_validates_keys(tmp_path: Path) -> None:
    tasks = _write(tmp_path / "tasks.jsonl", '{"instance_id":"x","user_query":"A  B\\t?"}\n')
    joined = _write(tmp_path / "joined.jsonl", '{"instance_id":"x","user_query":"A  B\\t?"}\n')
    assert query_passthrough(tasks, joined)["raw_exact"] == 1

    _write(joined, '{"instance_id":"x","user_query":"changed"}\n')
    with pytest.raises(ValueError, match="differs"):
        query_passthrough(tasks, joined)

    _write(joined, '{"instance_id":"y","user_query":"A  B\\t?"}\n')
    with pytest.raises(ValueError, match="keys differ"):
        query_passthrough(tasks, joined)

    _write(
        joined,
        '{"instance_id":"x","user_query":"A  B\\t?"}\n'
        '{"instance_id":"x","user_query":"A  B\\t?"}\n',
    )
    with pytest.raises(ValueError, match="duplicate"):
        query_passthrough(tasks, joined)


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
    (paths["raw_dir"] / "model-page-1.txt").unlink()
    with pytest.raises(FileNotFoundError, match="declared raw output"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "bytes")
    row = json.loads(paths["raw_results"].read_text())
    row["raw_output_bytes"] += 1
    _json(paths["raw_results"], row)
    with pytest.raises(ValueError, match="byte mismatch"):
        _evaluate(paths)


def test_evaluate_rejects_missing_reference_and_revision_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write(
        paths["reference_path"],
        '{"instance_id":"other","status":"ok","blocks":[],"timing":{}}\n',
    )
    with pytest.raises(ValueError, match="fixed reference"):
        _evaluate(paths)

    paths = _fixture(tmp_path / "revision")
    result = json.loads(paths["raw_results"].read_text())
    result["revision"] = "wrong"
    _json(paths["raw_results"], result)
    with pytest.raises(ValueError, match="revision mismatch"):
        _evaluate(paths)


def test_evaluate_v2_uses_parent_sampled_peaks_and_validates_output_hash(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = paths["raw_dir"] / "model-page-1.txt"
    result = json.loads(paths["raw_results"].read_text())
    result.pop("revision")
    result.pop("raw_output_paths")
    result["resolved_revision"] = "abc"
    result["raw_outputs"] = [
        {
            "path": "/remote/model-page-1.txt",
            "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    ]
    result["peak_process_rss_bytes_child"] = 30
    result["peak_process_rss_bytes_parent_sampled"] = 11
    result["peak_vram_bytes_parent_sampled"] = 21
    _json(paths["raw_results"], result)
    row = _evaluate(paths)["rows"][0]
    assert row["peak_ram_bytes"] == 11
    assert row["peak_ram_bytes_child"] == 30
    assert row["peak_vram_bytes"] == 21
    assert row["peak_vram_bytes_child_allocated"] == 20

    result["raw_outputs"][0]["sha256"] = "0" * 64
    _json(paths["raw_results"], result)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _evaluate(paths)
