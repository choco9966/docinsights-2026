import hashlib
import json
import tarfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest

from docinsights_ocr.cli import build_parser
from docinsights_ocr.cloud import (
    merge_shards,
    pack_cloud_input,
    shard_assignments,
    split_manifest,
)
from docinsights_ocr.records import read_jsonl, write_jsonl


def _manifest_record(instance_id: str, pdf: Path) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "user_query": f"query-{instance_id}",
        "document_pdf": pdf.name,
        "input_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "split": "ocr_eval",
        "split_seed": "test",
    }


def _result(record: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "instance_id": record["instance_id"],
        "user_query": record["user_query"],
        "pages": [
            {
                "page_number": 1,
                "width": 100,
                "height": 200,
                "coordinate_system": "pixel_top_left",
            }
        ],
        "blocks": [
            {
                "block_id": "b01",
                "text": "value 10",
                "page_numbers": [1],
                "lines": [
                    {
                        "page_number": 1,
                        "text": "b01 value 10",
                        "bbox": {"left": 1, "top": 2, "width": 3, "height": 4},
                        "confidence": 0.9,
                        "confidence_kind": "test",
                    }
                ],
            }
        ],
        "engine": "test-engine",
        "provenance": {
            "input_pdf_sha256": record["input_pdf_sha256"],
            "split": record["split"],
            "split_seed": record["split_seed"],
            "dpi": 200,
            "language": "eng",
            "renderer": "test-renderer",
            "confidence_kind": "test",
            "coordinate_system": "pixel_top_left",
            "ocr_options": {
                "pipeline_revision": "1" * 40,
                "detection_model_repo": "detector",
                "detection_model_revision": "3" * 40,
                "detection_model_path": "/different/per-host/detector",
                "recognition_model_repo": "recognizer",
                "recognition_model_revision": "4" * 40,
                "recognition_model_path": "/different/per-host/recognizer",
                "paddlepaddle_version": "3.2.0",
                "paddleocr_version": "3.3.2",
                "paddlex_version": "3.3.13",
            },
            "ocr_executable_identity": {"sha256": "same-engine"},
            "renderer_executable_identity": {"sha256": "same-renderer"},
            "pipeline_revision": "1" * 40,
            "timeout_seconds": 300.0,
            "run_fingerprint": "5" * 64,
        },
        "timing": {"total_seconds": 1.0},
        "status": "ok",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_manifest_sha256(records: list[dict[str, object]]) -> str:
    content = b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        for record in sorted(records, key=lambda row: str(row["instance_id"]))
    )
    return hashlib.sha256(content).hexdigest()


def _shard_manifest_sha256(
    records: list[dict[str, object]], assignments: dict[str, int], index: int
) -> str:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in sorted(records, key=lambda row: str(row["instance_id"]))
        if assignments[str(record["instance_id"])] == index
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _write_runtime_sidecars(
    tmp_path: Path,
    records: list[dict[str, object]],
    result_paths: list[Path],
    assignments: dict[str, int],
    *,
    mutate: Callable[[int, dict[str, object]], None] | None = None,
) -> list[Path]:
    count = len(result_paths)
    manifest_sha256 = _canonical_manifest_sha256(records)
    paths: list[Path] = []
    for index, result_path in enumerate(result_paths):
        runtime: dict[str, object] = {
            "schema_version": "1.0",
            "session_fingerprint": f"session-{index}",
            "platform_role": "kaggle-cpu",
            "platform": "Linux-test",
            "machine": "x86_64",
            "python": "3.11.13",
            "repository_sha": "1" * 40,
            "repository_dirty": False,
            "pipeline_revision": "1" * 40,
            "bundle_sha256": "2" * 64,
            "manifest_sha256": manifest_sha256,
            "shard_manifest_sha256": _shard_manifest_sha256(
                records, assignments, index
            ),
            "shard_count": count,
            "shard_index": index,
            "result_sha256": _sha256(result_path),
            "timeout_seconds": 300.0,
            "runtime_version": {
                "KAGGLE_KERNEL_RUN_TYPE": "Interactive",
                "COLAB_RELEASE_TAG": None,
            },
            "detector": {"repo": "detector", "revision": "3" * 40},
            "recognizer": {"repo": "recognizer", "revision": "4" * 40},
            "packages": {
                "paddlepaddle": "3.2.0",
                "paddleocr": "3.3.2",
                "paddlex": "3.3.13",
                "huggingface-hub": "0.34.4",
            },
            "record_count": sum(1 for _ in read_jsonl(result_path)),
            "failed_count": sum(
                record.get("status") == "failed" for record in read_jsonl(result_path)
            ),
        }
        if mutate is not None:
            mutate(index, runtime)
        path = tmp_path / f"runtime-shard-{index:02}-of-{count:02}.json"
        path.write_text(json.dumps(runtime), encoding="utf-8")
        paths.append(path)
    return paths


def _single_shard_case(
    tmp_path: Path, *, result: dict[str, object] | None = None
) -> tuple[Path, list[dict[str, object]], list[Path], list[Path]]:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    record = _manifest_record("task_000001", pdf)
    records = [record]
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, records)
    result_paths = [tmp_path / "result-shard-00-of-01.jsonl"]
    write_jsonl(result_paths[0], [result or _result(record)])
    runtimes = _write_runtime_sidecars(
        tmp_path, records, result_paths, {"task_000001": 0}
    )
    return manifest, records, result_paths, runtimes


def _set_out_of_order_blocks(record: dict[str, object]) -> None:
    first = deepcopy(record["blocks"][0])  # type: ignore[index]
    second = deepcopy(first)
    first["block_id"] = "b02"
    record["blocks"] = [first, second]


def _set_undeclared_line_page(record: dict[str, object]) -> None:
    block = record["blocks"][0]  # type: ignore[index]
    block["lines"][0]["page_number"] = 999


def test_shard_assignment_is_stable_balanced_and_query_independent(tmp_path: Path) -> None:
    records = {}
    for number in range(17):
        pdf = tmp_path / f"{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        record = _manifest_record(f"task_{number:06}", pdf)
        records[str(record["instance_id"])] = record

    first = shard_assignments(records, 4)
    reordered = {
        instance_id: {**record, "user_query": "changed"}
        for instance_id, record in reversed(list(records.items()))
    }
    second = shard_assignments(reordered, 4)

    assert first == second
    sizes = [sum(index == shard for index in first.values()) for shard in range(4)]
    assert max(sizes) - min(sizes) <= 1


def test_shard_assignment_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        shard_assignments({"": {"input_pdf_sha256": "abc"}}, 8)
    with pytest.raises(ValueError, match="at least one"):
        shard_assignments({}, 0)


def test_split_manifest_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    records = []
    for number in range(9):
        pdf = tmp_path / f"task_{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        records.append(_manifest_record(f"task_{number:06}", pdf))
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, reversed(records))
    output = tmp_path / "shards"

    first = split_manifest(manifest, output, shard_count=4)
    second = split_manifest(manifest, output, shard_count=4)

    assert first == second
    assert sum(shard["record_count"] for shard in first["shards"]) == 9
    assigned = [instance_id for shard in first["shards"] for instance_id in shard["instance_ids"]]
    assert sorted(assigned) == sorted(record["instance_id"] for record in records)
    assert json.loads((output / "shard-plan.json").read_text()) == first


def test_split_manifest_rejects_label_fields(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    record = _manifest_record("task_000001", pdf)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [{**record, "target": "secret"}])

    with pytest.raises(ValueError, match="outside allowlist"):
        split_manifest(manifest, tmp_path / "shards", shard_count=1)


def test_merge_shards_requires_complete_correct_assignment(tmp_path: Path) -> None:
    records = []
    for number in range(6):
        pdf = tmp_path / f"task_{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        records.append(_manifest_record(f"task_{number:06}", pdf))
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, records)
    shard_count = 3
    assignments = shard_assignments(
        {str(record["instance_id"]): record for record in records}, shard_count
    )
    paths = [tmp_path / f"result-shard-{index:02}-of-{shard_count:02}.jsonl" for index in range(3)]
    for index, path in enumerate(paths):
        write_jsonl(
            path,
            (
                _result(record)
                for record in records
                if assignments[str(record["instance_id"])] == index
            ),
        )
    runtimes = _write_runtime_sidecars(tmp_path, records, paths, assignments)

    output = tmp_path / "merged.jsonl"
    summary = merge_shards(
        manifest,
        list(reversed(paths)),
        output,
        runtime_paths=list(reversed(runtimes)),
    )

    merged = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["instance_id"] for record in merged] == sorted(
        record["instance_id"] for record in records
    )
    assert summary["record_count"] == 6
    assert summary["ok_count"] == 6
    assert len(summary["aggregate_hash"]) == 64


def test_merge_shards_rejects_missing_duplicate_and_wrong_shard(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    record = _manifest_record("task_000001", pdf)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [record])
    correct = shard_assignments({"task_000001": record}, 2)["task_000001"]
    wrong = 1 - correct
    paths = [tmp_path / f"result-shard-{index:02}-of-02.jsonl" for index in range(2)]
    write_jsonl(paths[correct], [])
    write_jsonl(paths[wrong], [_result(record)])
    assignments = {"task_000001": correct}
    runtimes = _write_runtime_sidecars(tmp_path, [record], paths, assignments)

    with pytest.raises(ValueError, match="belongs to shard"):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )

    write_jsonl(paths[wrong], [])
    runtimes = _write_runtime_sidecars(tmp_path, [record], paths, assignments)
    with pytest.raises(ValueError, match="missing shard result"):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )

    write_jsonl(paths[correct], [_result(record), _result(record)])
    runtimes = _write_runtime_sidecars(tmp_path, [record], paths, assignments)
    with pytest.raises(ValueError, match="duplicate instance_id across shard results"):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )

    unexpected = deepcopy(_result(record))
    unexpected["instance_id"] = "task_999999"
    write_jsonl(paths[correct], [unexpected])
    runtimes = _write_runtime_sidecars(tmp_path, [record], paths, assignments)
    with pytest.raises(ValueError, match="unexpected instance_id"):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )


def test_merge_shards_requires_matching_runtime_sidecars_and_preserves_output(
    tmp_path: Path,
) -> None:
    manifest, records, paths, runtimes = _single_shard_case(tmp_path)
    output = tmp_path / "merged.jsonl"
    output.write_bytes(b"existing-canonical-output\n")

    with pytest.raises(ValueError, match="runtime sidecar"):
        merge_shards(manifest, paths, output, runtime_paths=[])
    assert output.read_bytes() == b"existing-canonical-output\n"

    runtime = json.loads(runtimes[0].read_text())
    runtime["manifest_sha256"] = "f" * 64
    runtimes[0].write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="full manifest SHA-256 mismatch"):
        merge_shards(manifest, paths, output, runtime_paths=runtimes)
    assert output.read_bytes() == b"existing-canonical-output\n"

    runtimes = _write_runtime_sidecars(
        tmp_path, records, paths, {"task_000001": 0}
    )
    paths[0].write_text(paths[0].read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result SHA-256 mismatch"):
        merge_shards(manifest, paths, output, runtime_paths=runtimes)
    assert output.read_bytes() == b"existing-canonical-output\n"


def test_merge_shards_rejects_incomplete_or_misnamed_shard_sets(tmp_path: Path) -> None:
    manifest, _, paths, runtimes = _single_shard_case(tmp_path)
    misnamed = tmp_path / "runtime-00.json"
    misnamed.write_bytes(runtimes[0].read_bytes())

    with pytest.raises(ValueError, match="runtime sidecar filename"):
        merge_shards(manifest, paths, tmp_path / "merged.jsonl", runtime_paths=[misnamed])

    duplicate_result = tmp_path / "copy-shard-00-of-01.jsonl"
    duplicate_result.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="shard indexes"):
        merge_shards(
            manifest,
            [paths[0], duplicate_result],
            tmp_path / "merged.jsonl",
            runtime_paths=runtimes,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shard_index", 1, "runtime shard_index mismatch"),
        ("shard_count", 2, "runtime shard_count mismatch"),
        ("shard_manifest_sha256", "f" * 64, "shard manifest SHA-256 mismatch"),
        ("repository_dirty", True, "requires a clean repository"),
    ],
)
def test_merge_shards_rejects_invalid_runtime_identity(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest, records, paths, _ = _single_shard_case(tmp_path)

    def mutate(_index: int, runtime: dict[str, object]) -> None:
        runtime[field] = value

    runtimes = _write_runtime_sidecars(
        tmp_path, records, paths, {"task_000001": 0}, mutate=mutate
    )
    with pytest.raises(ValueError, match=message):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )


@pytest.mark.parametrize(
    "cohort_change",
    [
        "platform_role",
        "platform",
        "machine",
        "python",
        "repository_sha",
        "bundle_sha256",
        "timeout_seconds",
        "runtime_version",
        "detector_revision",
        "recognizer_revision",
        "package_version",
    ],
)
def test_merge_shards_rejects_mixed_runtime_cohorts(
    tmp_path: Path, cohort_change: str
) -> None:
    records = []
    for number in range(2):
        pdf = tmp_path / f"doc-{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        records.append(_manifest_record(f"task_{number:06}", pdf))
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, records)
    assignments = shard_assignments(
        {str(record["instance_id"]): record for record in records}, 2
    )
    paths = [tmp_path / f"result-shard-{index:02}-of-02.jsonl" for index in range(2)]
    for index, path in enumerate(paths):
        write_jsonl(
            path,
            [
                _result(record)
                for record in records
                if assignments[str(record["instance_id"])] == index
            ],
        )

    def mutate(index: int, runtime: dict[str, object]) -> None:
        if index != 1:
            return
        if cohort_change == "detector_revision":
            runtime["detector"] = {"repo": "detector", "revision": "9" * 40}
        elif cohort_change == "recognizer_revision":
            runtime["recognizer"] = {"repo": "recognizer", "revision": "9" * 40}
        elif cohort_change == "package_version":
            runtime["packages"] = {
                **runtime["packages"],  # type: ignore[dict-item]
                "paddleocr": "9.9.9",
            }
        elif cohort_change == "bundle_sha256":
            runtime[cohort_change] = "9" * 64
        elif cohort_change == "timeout_seconds":
            runtime[cohort_change] = 600.0
        elif cohort_change == "runtime_version":
            runtime[cohort_change] = {
                "KAGGLE_KERNEL_RUN_TYPE": "Batch",
                "COLAB_RELEASE_TAG": None,
            }
        elif cohort_change == "repository_sha":
            runtime["repository_sha"] = "9" * 40
            runtime["pipeline_revision"] = "9" * 40
        else:
            runtime[cohort_change] = f"different-{cohort_change}"

    runtimes = _write_runtime_sidecars(
        tmp_path, records, paths, assignments, mutate=mutate
    )
    with pytest.raises(ValueError, match="runtime cohort mismatch"):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )


@pytest.mark.parametrize(
    ("record_field", "message"),
    [
        ("pipeline_revision", "pipeline revision mismatch"),
        ("timeout_seconds", "timeout mismatch"),
        ("detection_model_revision", "detection_model_revision mismatch"),
        ("recognition_model_revision", "recognition_model_revision mismatch"),
        ("paddleocr_version", "paddleocr_version mismatch"),
    ],
)
def test_merge_shards_binds_runtime_to_result_provenance(
    tmp_path: Path,
    record_field: str,
    message: str,
) -> None:
    manifest, records, paths, _ = _single_shard_case(tmp_path)
    result = json.loads(paths[0].read_text())
    provenance = result["provenance"]
    if record_field == "pipeline_revision":
        provenance[record_field] = "9" * 40
    elif record_field == "timeout_seconds":
        provenance[record_field] = 600.0
    else:
        provenance["ocr_options"][record_field] = "different"
    write_jsonl(paths[0], [result])
    runtimes = _write_runtime_sidecars(
        tmp_path, records, paths, {"task_000001": 0}
    )

    with pytest.raises(ValueError, match=message):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )


def test_merge_shards_rejects_runtime_result_counts(tmp_path: Path) -> None:
    manifest, records, paths, _ = _single_shard_case(tmp_path)

    def mutate(_index: int, runtime: dict[str, object]) -> None:
        runtime["record_count"] = 2

    runtimes = _write_runtime_sidecars(
        tmp_path, records, paths, {"task_000001": 0}, mutate=mutate
    )
    with pytest.raises(ValueError, match="runtime record_count mismatch"):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.update(status="bogus"),
        lambda record: record.pop("pages"),
        lambda record: record.update(pages=[]),
        lambda record: record.update(unexpected_field=True),
        lambda record: record.update(provenance=[]),
        lambda record: record.update(blocks=[{}]),
        lambda record: record.update(blocks=[]),
        lambda record: record.update(error="unexpected"),
        _set_out_of_order_blocks,
        _set_undeclared_line_page,
    ],
    ids=[
        "invalid-status",
        "missing-pages",
        "empty-pages",
        "unknown-top-level",
        "invalid-provenance",
        "invalid-block",
        "empty-success",
        "successful-error",
        "out-of-order-blocks",
        "undeclared-line-page",
    ],
)
def test_merge_shards_rejects_schema_invalid_records(
    tmp_path: Path, mutate: Callable[[dict[str, object]], object]
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    manifest_record = _manifest_record("task_000001", pdf)
    result = deepcopy(_result(manifest_record))
    mutate(result)
    manifest, _, paths, runtimes = _single_shard_case(tmp_path, result=result)

    with pytest.raises(ValueError, match="OCR"):
        merge_shards(
            manifest, paths, tmp_path / "merged.jsonl", runtime_paths=runtimes
        )


def test_merge_shards_rejects_failed_records_by_default(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    manifest_record = _manifest_record("task_000001", pdf)
    failed = deepcopy(_result(manifest_record))
    failed.pop("pages")
    failed.update(
        status="failed",
        blocks=[],
        error="OCR timed out",
        error_kind="timeout",
    )
    manifest, _, paths, runtimes = _single_shard_case(tmp_path, result=failed)
    output = tmp_path / "merged.jsonl"

    with pytest.raises(ValueError, match="canonical merge rejects failed OCR records"):
        merge_shards(manifest, paths, output, runtime_paths=runtimes)
    assert not output.exists()

    summary = merge_shards(
        manifest,
        paths,
        output,
        runtime_paths=runtimes,
        fail_closed=False,
    )
    assert summary["failed_count"] == 1


def test_pack_cloud_input_is_reproducible_and_contains_no_labels(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    first_pdf = documents / "a.pdf"
    second_pdf = documents / "b.pdf"
    first_pdf.write_bytes(b"pdf-content-a")
    second_pdf.write_bytes(b"pdf-content-b")
    records = [
        _manifest_record("task_000002", second_pdf),
        _manifest_record("task_000001", first_pdf),
    ]
    manifest = tmp_path / "manifest.jsonl"
    reordered_manifest = tmp_path / "manifest-reordered.jsonl"
    write_jsonl(manifest, records)
    write_jsonl(reordered_manifest, reversed(records))
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_summary = pack_cloud_input(manifest, documents, first)
    second_summary = pack_cloud_input(reordered_manifest, documents, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_summary["archive_sha256"] == second_summary["archive_sha256"]
    with tarfile.open(first, "r:gz") as archive:
        names = archive.getnames()
        assert names == [
            "bundle/manifest.jsonl",
            "bundle/bundle.json",
            "bundle/documents-root/a.pdf",
            "bundle/documents-root/b.pdf",
        ]
        manifest_text = archive.extractfile("bundle/manifest.jsonl").read().decode()  # type: ignore[union-attr]
        bundled_records = [json.loads(line) for line in manifest_text.splitlines()]
    assert [record["instance_id"] for record in bundled_records] == [
        "task_000001",
        "task_000002",
    ]
    assert '"answer"' not in manifest_text
    assert '"evidence"' not in manifest_text


@pytest.mark.parametrize(
    "label_alias",
    ["answer", "candidate_answers", "evidence", "labels", "gold_answer", "target"],
)
def test_pack_cloud_input_rejects_labels_and_aliases(
    tmp_path: Path, label_alias: str
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    record = _manifest_record("task_000001", pdf)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({**record, label_alias: "secret"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside allowlist"):
        pack_cloud_input(manifest, tmp_path, tmp_path / "bundle.tar.gz")


def test_pack_cloud_input_rejects_path_traversal(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"pdf")
    record = _manifest_record("task_000001", pdf)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [{**record, "document_pdf": "../doc.pdf"}])
    with pytest.raises(ValueError, match="safe relative path"):
        pack_cloud_input(manifest, tmp_path, tmp_path / "bundle.tar.gz")


def test_cloud_merge_cli_requires_runtime_sidecars() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["cloud-merge", "manifest.jsonl", "merged.jsonl", "result.jsonl"])
    args = parser.parse_args(
        [
            "cloud-merge",
            "manifest.jsonl",
            "merged.jsonl",
            "result-shard-00-of-01.jsonl",
            "--runtimes",
            "runtime-shard-00-of-01.json",
        ]
    )
    assert args.runtimes == ["runtime-shard-00-of-01.json"]
    assert args.allow_failed is False
