import json
from pathlib import Path

from docinsights_hf_ocr.evaluation import evaluate, hash_paths, query_passthrough, write_outputs


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_dynamic_reference_refresh_query_passthrough_and_determinism(tmp_path: Path) -> None:
    tasks = _write(
        tmp_path / "tasks.jsonl", '{"instance_id":"task_000909","user_query":"Keep  this?"}\n'
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
    candidates = _write(
        tmp_path / "candidates.json",
        json.dumps(
            {"models": [{"model": "org/model", "params": 1, "weight_gib": 0.1, "license": "MIT"}]}
        ),
    )
    raw_dir = tmp_path / "raw"
    _write(raw_dir / "model-page-1.txt", "b01: Alpha")
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
                "raw_output_bytes": 10,
            }
        )
        + "\n",
    )
    report = evaluate(results, raw_dir, candidates, refs, tasks)
    assert report["reference"]["available_validated_subset"] == 2
    assert report["reference"]["total_seconds_when_present"] == 5
    assert report["query_passthrough"] == {
        "samples": 1,
        "raw_exact": 1,
        "normalized_exact": 1,
        "sha256_exact": 1,
    }
    assert report["rows"][0]["silver_agreement_cer"] == 0
    first = write_outputs(report, tmp_path / "out")
    hashes_before = hash_paths(first)
    second = write_outputs(report, tmp_path / "out")
    assert hashes_before == hash_paths(second)


def test_query_passthrough_preserves_raw_source(tmp_path: Path) -> None:
    tasks = _write(tmp_path / "tasks.jsonl", '{"instance_id":"x","user_query":"A  B\\t?"}\n')
    assert query_passthrough(tasks)["raw_exact"] == 1
