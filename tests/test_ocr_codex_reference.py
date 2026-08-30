import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from docinsights_ocr.cli import build_parser
from docinsights_ocr.codex_reference import (
    EXPECTED_BLOCK_IDS,
    PROMPT,
    _validated_blocks,
    run_codex_reference,
)


def _response(*, block_ids: tuple[str, ...] = EXPECTED_BLOCK_IDS) -> str:
    return json.dumps(
        {
            "blocks": [
                {"block_id": block_id, "text": f"  Text   for {block_id}  "}
                for block_id in block_ids
            ]
        }
    )


def _manifest(tmp_path: Path, *, count: int = 1) -> Path:
    rows = []
    for number in range(count):
        pdf = tmp_path / f"doc-{number}.pdf"
        pdf.write_bytes(f"pdf-{number}".encode())
        rows.append(
            {
                "instance_id": f"task_{number:06d}",
                "user_query": f"secret query {number}",
                "document_pdf": pdf.name,
                "split": "ocr_eval",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def _fake_codex_run(calls: list[dict[str, object]]):
    def run(argv, **kwargs):
        if argv[-1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, "codex-cli 0.test\n", "")
        cwd = Path(kwargs["cwd"])
        calls.append(
            {
                "argv": argv,
                "cwd": cwd,
                "files": sorted(path.name for path in cwd.iterdir()),
                "prompt": kwargs["input"],
            }
        )
        response_path = Path(argv[argv.index("--output-last-message") + 1])
        response_path.write_text(_response(), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, _response(), "diagnostic")

    return run


def test_codex_reference_uses_exactly_two_images_and_isolated_noninteractive_argv(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "artifacts" / "reference.jsonl"
    raw_dir = tmp_path / "artifacts" / "raw"
    rendered = (tmp_path / "rendered-1.png", tmp_path / "rendered-2.png")
    for number, image in enumerate(rendered, 1):
        image.write_bytes(f"image-{number}".encode())
    calls: list[dict[str, object]] = []

    with (
        patch("docinsights_ocr.codex_reference.render_pdf", return_value=rendered),
        patch("docinsights_ocr.codex_reference.subprocess.run", side_effect=_fake_codex_run(calls)),
    ):
        assert (
            run_codex_reference(
                manifest,
                output,
                documents_root=tmp_path,
                raw_dir=raw_dir,
                model="gpt-test",
                model_config=('model_reasoning_effort="high"',),
            )
            == 1
        )

    call = calls[0]
    argv = call["argv"]
    assert call["files"] == ["page-1.png", "page-2.png", "response.schema.json"]
    assert call["prompt"] == PROMPT
    assert "secret query" not in call["prompt"]
    for flag in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--output-schema",
    ):
        assert flag in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--model") + 1] == "gpt-test"
    assert argv.count("--image") == 2
    assert argv[-1] == "-"
    image_paths = [argv[index + 1] for index, value in enumerate(argv) if value == "--image"]
    assert all(Path(path).parent == call["cwd"] for path in image_paths)
    assert raw_dir.joinpath("task_000000.json").is_file()

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["reference_kind"] == "codex-assisted-silver"
    assert record["engine"] == "codex-assisted-visual-transcription"
    assert record["status"] == "ok"
    assert [block["block_id"] for block in record["blocks"]] == list(EXPECTED_BLOCK_IDS)
    assert record["blocks"][0]["text"] == "Text for b01"
    assert "answer" not in record and "evidence" not in record
    provenance = record["provenance"]
    assert provenance["model"] == "gpt-test"
    assert provenance["codex_cli_version"] == "codex-cli 0.test"
    assert len(provenance["prompt_sha256"]) == 64
    assert len(provenance["output_schema_sha256"]) == 64
    assert len(provenance["input_image_sha256"]) == 2


@pytest.mark.parametrize(
    "block_ids",
    [
        EXPECTED_BLOCK_IDS[:-1],
        EXPECTED_BLOCK_IDS[:-1] + ("b22",),
        ("b02", "b01") + EXPECTED_BLOCK_IDS[2:],
    ],
)
def test_codex_response_rejects_missing_duplicate_or_out_of_order_blocks(
    block_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="ordered unique blocks"):
        _validated_blocks(_response(block_ids=block_ids))


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        json.dumps({"blocks": "wrong"}),
        json.dumps({"blocks": [{"block_id": "b01", "text": "x", "answer": "x"}]}),
        json.dumps({"blocks": [{"block_id": "b01", "text": ""}]}),
        json.dumps({"blocks": [{"block_id": "b01", "text": "   "}]}),
    ],
)
def test_codex_response_rejects_malformed_or_forbidden_shapes(response: str) -> None:
    with pytest.raises(ValueError):
        _validated_blocks(response)


def test_codex_reference_fails_closed_when_page_count_or_response_is_invalid(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "artifacts" / "reference.jsonl"
    image = tmp_path / "one.png"
    image.write_bytes(b"image")

    with (
        patch("docinsights_ocr.codex_reference._codex_version", return_value="codex-cli test"),
        patch("docinsights_ocr.codex_reference.render_pdf", return_value=(image,)),
    ):
        run_codex_reference(manifest, output, documents_root=tmp_path)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["blocks"] == []
    assert record["error_kind"] == "validation_error"
    assert "answer" not in record and "evidence" not in record


def test_codex_reference_preserves_subprocess_stderr_in_failure_record(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "artifacts" / "reference.jsonl"
    rendered = (tmp_path / "rendered-1.png", tmp_path / "rendered-2.png")
    for image in rendered:
        image.write_bytes(b"image")
    failure = subprocess.CalledProcessError(
        1,
        ["codex", "exec"],
        stderr="remote service unavailable",
        output="request id 123",
    )

    with (
        patch("docinsights_ocr.codex_reference._codex_version", return_value="codex-cli test"),
        patch(
            "docinsights_ocr.codex_reference._executable_identity",
            return_value={"name": "test", "kind": "sha256", "sha256": "1" * 64},
        ),
        patch("docinsights_ocr.codex_reference.render_pdf", return_value=rendered),
        patch("docinsights_ocr.codex_reference.subprocess.run", side_effect=failure),
    ):
        run_codex_reference(manifest, output, documents_root=tmp_path)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error_kind"] == "subprocess_error"
    assert "stderr: remote service unavailable" in record["error"]
    assert "stdout: request id 123" in record["error"]


def test_codex_reference_resume_retry_replaces_only_failed_record(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=2)
    output = tmp_path / "artifacts" / "reference.jsonl"
    rendered = (tmp_path / "rendered-1.png", tmp_path / "rendered-2.png")
    for image in rendered:
        image.write_bytes(b"image")
    calls: list[dict[str, object]] = []

    with (
        patch("docinsights_ocr.codex_reference.render_pdf", return_value=rendered),
        patch("docinsights_ocr.codex_reference.subprocess.run", side_effect=_fake_codex_run(calls)),
    ):
        run_codex_reference(manifest, output, documents_root=tmp_path)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    records[1]["status"] = "failed"
    records[1]["blocks"] = []
    records[1]["error_kind"] = "validation_error"
    records[1]["error"] = "forced"
    output.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    calls.clear()

    with (
        patch("docinsights_ocr.codex_reference.render_pdf", return_value=rendered),
        patch("docinsights_ocr.codex_reference.subprocess.run", side_effect=_fake_codex_run(calls)),
    ):
        assert (
            run_codex_reference(
                manifest,
                output,
                documents_root=tmp_path,
                resume=True,
                retry_failed=True,
            )
            == 1
        )

    retried = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(retried) == 2
    assert all(record["status"] == "ok" for record in retried)
    assert len(calls) == 1


def test_codex_reference_retry_prioritizes_unseen_records(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, count=3)
    output = tmp_path / "artifacts" / "reference.jsonl"
    transcribed: list[str] = []

    def transcribe(task, **_kwargs):
        instance_id = str(task["instance_id"])
        transcribed.append(instance_id)
        return {
            "schema_version": "1.0",
            "reference_kind": "codex-assisted-silver",
            "instance_id": instance_id,
            "blocks": [],
            "engine": "codex-assisted-visual-transcription",
            "provenance": {"run_fingerprint": task["run_fingerprint"]},
            "timing": {"total_seconds": 0.0},
            "status": "failed",
            "error_kind": "validation_error",
            "error": "forced persistent failure",
        }

    with (
        patch("docinsights_ocr.codex_reference._codex_version", return_value="codex-cli test"),
        patch(
            "docinsights_ocr.codex_reference._executable_identity",
            return_value={"name": "test", "kind": "command_name", "sha256": None},
        ),
        patch("docinsights_ocr.codex_reference._transcribe_one", side_effect=transcribe),
    ):
        run_codex_reference(manifest, output, documents_root=tmp_path, limit=1)
        run_codex_reference(
            manifest,
            output,
            documents_root=tmp_path,
            resume=True,
            retry_failed=True,
            limit=1,
        )
        run_codex_reference(
            manifest,
            output,
            documents_root=tmp_path,
            resume=True,
            retry_failed=True,
            limit=1,
        )

    assert transcribed == ["task_000000", "task_000001", "task_000002"]
    assert [json.loads(line)["instance_id"] for line in output.read_text().splitlines()] == [
        "task_000000",
        "task_000001",
        "task_000002",
    ]


def test_codex_reference_schema_and_cli_defaults_are_strict() -> None:
    root = Path(__file__).parents[1]
    schema = json.loads(
        (root / "schemas" / "codex-transcription-response-v1.schema.json").read_text()
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["blocks"]["minItems"] == 23
    assert schema["properties"]["blocks"]["maxItems"] == 23
    assert schema["properties"]["blocks"]["items"]["additionalProperties"] is False

    args = build_parser().parse_args(["codex-reference", "in.jsonl", "out.jsonl"])
    assert args.workers == 1
    assert args.model == "gpt-5.6-sol"
    assert args.raw_dir.startswith("artifacts/")
