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
    raw_output = _write(raw_dir / "model-1-page-1.txt", "b01: Alpha")
    output_sha = hashlib.sha256(raw_output.read_bytes()).hexdigest()
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
                        "path": "/remote/model-1-page-1.txt",
                        "bytes": raw_output.stat().st_size,
                        "sha256": output_sha,
                    }
                ]
                if success
                else [],
                "raw_output_sha256": [output_sha] if success else [],
                "doc_latency_sec": 2 if success else None,
                "peak_process_rss_bytes_child": 30,
                "peak_process_rss_bytes_parent_sampled": 10 + index,
                "peak_cuda_allocated_bytes": 20,
                "peak_vram_bytes_parent_sampled": 20 + index,
                "raw_output_bytes": raw_output.stat().st_size if success else 0,
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
    first = write_outputs(report, tmp_path / "out")
    hashes_before = hash_paths(first)
    second = write_outputs(report, tmp_path / "out")
    assert hashes_before == hash_paths(second)
    assert "operational_baseline" in (tmp_path / "out/comparison.csv").read_text()
    raw_csv = write_raw_csv(paths["raw_results"], tmp_path / "out" / "measured-raw.csv")
    assert "org/model-1" in raw_csv.read_text(encoding="utf-8")


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
    rows[0]["raw_output_sha256"] = ["0" * 64]
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
