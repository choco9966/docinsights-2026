import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from docinsights_ocr.cli import main
from docinsights_ocr.codex_reference import (
    DISABLED_CODEX_FEATURES,
    ENGINE,
    EXPECTED_BLOCK_IDS,
    PROMPT,
    REFERENCE_KIND,
    SCHEMA_VERSION,
)
from docinsights_ocr.codex_verify import verify_codex_reference
from docinsights_ocr.records import deterministic_content_hash


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture(autouse=True)
def _rendered_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pages = (tmp_path / "rendered-1.png", tmp_path / "rendered-2.png")
    pages[0].write_bytes(b"page-1")
    pages[1].write_bytes(b"page-2")
    monkeypatch.setattr("docinsights_ocr.codex_verify.render_pdf", lambda *_args, **_kwargs: pages)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "instance_id": "task_000001",
                "document_pdf": pdf.name,
                "input_pdf_sha256": _sha256(pdf.read_bytes()),
                "split": "validation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    blocks = [{"block_id": block_id, "text": f"text {block_id}"} for block_id in EXPECTED_BLOCK_IDS]
    raw_text = json.dumps({"blocks": blocks})
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_dir.joinpath("task_000001.json").write_text(raw_text, encoding="utf-8")
    raw_dir.joinpath("task_000001.stderr.txt").write_text("diagnostic", encoding="utf-8")
    prompt_sha256 = _sha256(PROMPT.encode())
    schema_path = (
        Path(__file__).parents[1] / "schemas" / "codex-transcription-response-v1.schema.json"
    )
    schema_sha256 = _sha256(
        schema_path.read_bytes()
    )
    codex_identity = {"name": "codex", "kind": "sha256", "sha256": "6" * 64}
    renderer_identity = {"name": "pdftoppm", "kind": "sha256", "sha256": "7" * 64}
    fingerprint = deterministic_content_hash(
        {
            "input_pdf_sha256": _sha256(pdf.read_bytes()),
            "model": "gpt-test",
            "model_config": ['model_reasoning_effort="high"'],
            "codex_version": "codex-cli test",
            "prompt_sha256": prompt_sha256,
            "output_schema_sha256": schema_sha256,
            "dpi": 200,
            "codex_executable_identity": codex_identity,
            "renderer_executable_identity": renderer_identity,
            "disabled_codex_features": list(DISABLED_CODEX_FEATURES),
        }
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reference_kind": REFERENCE_KIND,
        "instance_id": "task_000001",
        "blocks": blocks,
        "engine": ENGINE,
        "provenance": {
            "reference_kind": REFERENCE_KIND,
            "input_pdf_sha256": _sha256(pdf.read_bytes()),
            "split": "validation",
            "split_seed": None,
            "model": "gpt-test",
            "codex_cli_version": "codex-cli test",
            "model_config": ['model_reasoning_effort="high"'],
            "prompt_sha256": prompt_sha256,
            "output_schema_sha256": schema_sha256,
            "raw_response_sha256": _sha256(raw_text.encode()),
            "run_fingerprint": fingerprint,
            "dpi": 200,
            "renderer": "poppler-pdftoppm",
            "disabled_codex_features": list(DISABLED_CODEX_FEATURES),
            "input_image_sha256": [
                {"page_number": 1, "sha256": _sha256(b"page-1")},
                {"page_number": 2, "sha256": _sha256(b"page-2")},
            ],
            "codex_executable_identity": codex_identity,
            "renderer_executable_identity": renderer_identity,
        },
        "timing": {"total_seconds": 1.0},
        "status": "ok",
    }
    output = tmp_path / "reference.jsonl"
    _write_output(output, record)
    return manifest, output, raw_dir, record


def _write_output(path: Path, *records: dict[str, Any]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_verify_codex_reference_checks_complete_artifacts_and_returns_safe_summary(
    tmp_path: Path,
) -> None:
    manifest, output, raw_dir, _ = _artifacts(tmp_path)

    summary = verify_codex_reference(manifest, output, raw_dir)

    assert summary["valid"] is True
    assert summary["expected_count"] == summary["record_count"] == summary["ok_count"] == 1
    assert summary["raw_response_count"] == 1
    assert summary["stderr_sidecar_count"] == 1
    assert summary["missing_count"] == summary["duplicate_count"] == 0
    assert summary["extra_count"] == summary["failed_count"] == 0
    assert summary["tasks_path"] == str(manifest.resolve())
    assert summary["output_path"] == str(output.resolve())
    assert summary["raw_dir"] == str(raw_dir.resolve())
    assert len(summary["raw_response_aggregate_sha256"]) == 64
    assert len(summary["stderr_sidecar_aggregate_sha256"]) == 64
    serialized = json.dumps(summary)
    for forbidden in ("user_query", "labels", "answer", "evidence"):
        assert forbidden not in serialized


def test_codex_verify_cli_prints_json_and_optionally_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, output, raw_dir, _ = _artifacts(tmp_path)
    report = tmp_path / "reports" / "verification.json"

    argv = ["codex-verify", str(manifest), str(output), str(raw_dir), "--report", str(report)]
    assert main(argv) == 0

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["valid"] is True


def test_verify_codex_reference_rejects_missing_duplicate_and_extra_ids(tmp_path: Path) -> None:
    manifest, output, raw_dir, record = _artifacts(tmp_path)
    output.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="coverage mismatch.*missing"):
        verify_codex_reference(manifest, output, raw_dir)

    _write_output(output, record, record)
    with pytest.raises(ValueError, match="duplicate instance_id"):
        verify_codex_reference(manifest, output, raw_dir)

    extra = deepcopy(record)
    extra["instance_id"] = "task_000002"
    _write_output(output, record, extra)
    with pytest.raises(ValueError, match="coverage mismatch.*extra"):
        verify_codex_reference(manifest, output, raw_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update(status="failed"), "status is not ok"),
        (lambda record: record.update(schema_version="2.0"), "invalid schema_version"),
        (lambda record: record.update(engine="wrong"), "invalid engine"),
        (lambda record: record.pop("timing"), "invalid Codex record shape"),
        (lambda record: record["blocks"].pop(), "exactly 23 blocks"),
        (
            lambda record: record["blocks"].__setitem__(
                0, {"block_id": "b02", "text": "out of order"}
            ),
            "ordered b01 through b23",
        ),
        (
            lambda record: record["provenance"].update(run_fingerprint="8" * 64),
            "run fingerprint mismatch",
        ),
        (
            lambda record: record["provenance"].update(prompt_sha256="8" * 64),
            "prompt SHA-256 mismatch",
        ),
        (
            lambda record: record["provenance"].update(disabled_codex_features=[]),
            "disabled_codex_features",
        ),
        (
            lambda record: record["provenance"].update(
                input_image_sha256=[{"page_number": 1, "sha256": "4" * 64}]
            ),
            "exactly two pages",
        ),
        (
            lambda record: record["provenance"]["input_image_sha256"].__setitem__(
                1, {"page_number": 1, "sha256": "5" * 64}
            ),
            "ordered pages 1 and 2",
        ),
        (
            lambda record: record["provenance"]["codex_executable_identity"].update(
                sha256="bad"
            ),
            "invalid sha256 SHA-256",
        ),
        (lambda record: record.update(answer="secret"), "forbidden fields"),
    ],
)
def test_verify_codex_reference_rejects_invalid_record_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest, output, raw_dir, record = _artifacts(tmp_path)
    mutation(record)
    _write_output(output, record)

    with pytest.raises(ValueError, match=message):
        verify_codex_reference(manifest, output, raw_dir)


def test_verify_codex_reference_binds_pdf_raw_response_and_stderr(tmp_path: Path) -> None:
    manifest, output, raw_dir, record = _artifacts(tmp_path)
    (tmp_path / "document.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input PDF SHA-256 mismatch"):
        verify_codex_reference(manifest, output, raw_dir)

    (tmp_path / "document.pdf").write_bytes(b"pdf")
    raw_path = raw_dir / "task_000001.json"
    raw_path.write_text(json.dumps({"blocks": record["blocks"]}) + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="raw response SHA-256 mismatch"):
        verify_codex_reference(manifest, output, raw_dir)

    changed_blocks = [{**record["blocks"][0], "text": "changed"}, *record["blocks"][1:]]
    raw_text = json.dumps({"blocks": changed_blocks})
    raw_path.write_text(raw_text, encoding="utf-8")
    record["provenance"]["raw_response_sha256"] = _sha256(raw_text.encode())
    _write_output(output, record)
    with pytest.raises(ValueError, match="blocks do not match"):
        verify_codex_reference(manifest, output, raw_dir)

    raw_path.write_text(json.dumps({"blocks": record["blocks"]}), encoding="utf-8")
    record["provenance"]["raw_response_sha256"] = _sha256(raw_path.read_bytes())
    _write_output(output, record)
    raw_dir.joinpath("task_000001.stderr.txt").unlink()
    with pytest.raises(ValueError, match="stderr sidecars coverage mismatch"):
        verify_codex_reference(manifest, output, raw_dir)


def test_verify_codex_reference_compares_producer_canonicalized_raw_blocks(
    tmp_path: Path,
) -> None:
    manifest, output, raw_dir, record = _artifacts(tmp_path)
    raw_blocks = deepcopy(record["blocks"])
    raw_blocks[0]["text"] = "text   b01\n"
    raw_text = json.dumps({"blocks": raw_blocks})
    raw_path = raw_dir / "task_000001.json"
    raw_path.write_text(raw_text, encoding="utf-8")
    record["provenance"]["raw_response_sha256"] = _sha256(raw_text.encode())
    _write_output(output, record)

    assert verify_codex_reference(manifest, output, raw_dir)["valid"] is True


def test_verify_codex_reference_rejects_extra_raw_artifacts(tmp_path: Path) -> None:
    manifest, output, raw_dir, _ = _artifacts(tmp_path)
    raw_dir.joinpath("task_999999.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="raw responses coverage mismatch.*extra"):
        verify_codex_reference(manifest, output, raw_dir)


def test_verify_codex_reference_rejects_non_contract_raw_file(tmp_path: Path) -> None:
    manifest, output, raw_dir, _ = _artifacts(tmp_path)
    raw_dir.joinpath("labels.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected files in raw directory.*labels"):
        verify_codex_reference(manifest, output, raw_dir)


def test_verify_codex_reference_rejects_extra_provenance_field(tmp_path: Path) -> None:
    manifest, output, raw_dir, record = _artifacts(tmp_path)
    record["provenance"]["user_query"] = "must not be retained"
    _write_output(output, record)

    with pytest.raises(ValueError, match="invalid provenance fields.*user_query"):
        verify_codex_reference(manifest, output, raw_dir)


def test_verify_codex_reference_rejects_rendered_page_count_or_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, output, raw_dir, _ = _artifacts(tmp_path)
    one_page = tmp_path / "only.png"
    one_page.write_bytes(b"page-1")
    monkeypatch.setattr(
        "docinsights_ocr.codex_verify.render_pdf", lambda *_args, **_kwargs: (one_page,)
    )
    with pytest.raises(ValueError, match="exactly two rendered pages"):
        verify_codex_reference(manifest, output, raw_dir)

    wrong_page = tmp_path / "wrong.png"
    wrong_page.write_bytes(b"wrong")
    monkeypatch.setattr(
        "docinsights_ocr.codex_verify.render_pdf",
        lambda *_args, **_kwargs: (one_page, wrong_page),
    )
    with pytest.raises(ValueError, match="rendered page SHA-256 mismatch"):
        verify_codex_reference(manifest, output, raw_dir)
