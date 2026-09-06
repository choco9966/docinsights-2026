import hashlib
import importlib.util
import json
import subprocess
import time
import unicodedata
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT = Path(__file__).parents[1] / "scripts" / "transcribe_ocr_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("transcribe_ocr_benchmark", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, *, height: int = 10) -> Path:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"synthetic public PDF")
    page_path = tmp_path / "page.png"
    Image.new("RGB", (12, height), "white").save(page_path)
    bounds = [(0, height // 3), (height // 3, (2 * height) // 3), ((2 * height) // 3, height)]
    cores = []
    with Image.open(page_path) as page:
        for index, (top, bottom) in enumerate(bounds, start=1):
            core_path = tmp_path / f"manifest-core-{index}.png"
            page.crop((0, top, 12, bottom)).save(
                core_path, format="PNG", compress_level=9, optimize=False
            )
            cores.append(
                {
                    "core_index": index,
                    "crop_box": {"left": 0, "top": top, "right": 12, "bottom": bottom},
                    "image_path": str(core_path),
                    "image_sha256": _sha256(core_path),
                    "image_bytes": core_path.stat().st_size,
                    "width": 12,
                    "height": bottom - top,
                }
            )
    manifest = {
        "schema_version": "issue24-ocr-benchmark-v1",
        "scope": "synthetic",
        "selection": {"method": "sha256-sampled"},
        "render": {"renderer": "pdftoppm", "dpi": 175, "format": "png"},
        "scoring_cores": {"count": 3, "layout": "full-width-thirds"},
        "inventory_sources": {},
        "document_count": 1,
        "page_count": 1,
        "pages": [
            {
                "benchmark_page_id": "heldout-task_1-p0001",
                "split": "heldout",
                "instance_id": "task_1",
                "page_number": 1,
                "pdf_page_count": 1,
                "source_pdf": str(source_pdf),
                "source_pdf_sha256": _sha256(source_pdf),
                "rendered_image": str(page_path),
                "rendered_image_sha256": _sha256(page_path),
                "rendered_image_bytes": page_path.stat().st_size,
                "width": 12,
                "height": height,
                "scoring_cores": cores,
            }
        ],
    }
    path = tmp_path / "corpus-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _response(*, suffix: str = "", uncertain: bool = False) -> dict:
    return {
        "regions": [
            {
                "region_id": f"core-{index}",
                "lines": [f"Caf\N{COMBINING ACUTE ACCENT} total   {index}{suffix}"],
                "visible_ids": ["SEC:A-2", "b01"],
                "numeric_literals": ["-3.5%", "1/2", "1,200"],
                "uncertain": uncertain,
                "boundary_exclusions": index - 1,
            }
            for index in range(1, 4)
        ]
    }


def _two_page_manifest(tmp_path: Path) -> Path:
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    second = dict(manifest["pages"][0])
    second["benchmark_page_id"] = "train-task_2-p0001"
    second["split"] = "train"
    second["instance_id"] = "task_2"
    manifest["pages"].append(second)
    manifest["page_count"] = 2
    manifest["document_count"] = 2
    path.write_text(json.dumps(manifest))
    return path


def _successful_invoke(calls: list[dict], responses: list[dict] | None = None):
    queued = list(responses or [_response(), _response()])

    def invoke(**request):
        calls.append(request)
        response = queued.pop(0)
        output = request["output_dir"]
        raw = output / "response.json"
        events = output / "events.jsonl"
        stderr = output / "stderr.txt"
        raw.write_text(json.dumps(response), encoding="utf-8")
        events.write_text('{"type":"turn.completed","usage":{"input_tokens":1}}\n')
        stderr.write_text("")
        return {
            "status": "succeeded",
            "response": response,
            "raw_response_path": raw,
            "artifact_paths": [raw, events, stderr],
            "elapsed_seconds": 0.1,
            "error": None,
        }

    return invoke


def test_harness_creates_exact_thirds_and_two_candidate_blind_image_only_calls(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    calls: list[dict] = []
    output = tmp_path / "private" / "ocr-benchmark-v2"

    summary = _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=_successful_invoke(calls),
    )

    assert summary["completed_pages"] == 1
    assert len(calls) == 2
    assert calls[0]["prompt"] == calls[1]["prompt"] == _MODULE.TRANSCRIPTION_PROMPT
    assert calls[0]["output_schema"] == calls[1]["output_schema"] == _MODULE.OUTPUT_SCHEMA
    for call in calls:
        assert len(call["images"]) == 4
        assert [Image.open(path).size for path in call["images"]] == [
            (12, 10),
            (12, 3),
            (12, 3),
            (12, 4),
        ]
        serialized = call["prompt"] + json.dumps(call["checker_config"])
        for forbidden in ("task_1", "heldout", "source.pdf", "benchmark_page_id"):
            assert forbidden not in serialized
    assert calls[0]["output_dir"] != calls[1]["output_dir"]
    assert all(path.stat().st_mode & 0o077 == 0 for path in output.rglob("*") if path.is_file())


def test_consensus_normalizes_only_nfc_and_whitespace_and_matches_multisets(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    second = _response()
    for region in second["regions"]:
        region["lines"] = [unicodedata.normalize("NFC", region["lines"][0])]
        region["visible_ids"].reverse()
        region["numeric_literals"].reverse()
    output = tmp_path / "private" / "ocr-benchmark-v2"

    _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=_successful_invoke([], [_response(), second]),
    )

    consensus = json.loads(
        (output / "references" / "heldout-task_1-p0001" / "consensus.json").read_text()
    )
    assert consensus["schema_version"] == "consensus-silver-v3"
    for region in consensus["regions"]:
        assert region["consensus"]["body"] == {
            "eligible": True,
            "value": unicodedata.normalize("NFC", region["readers"][0]["normalized"]["body_text"]),
        }
        assert region["consensus"]["visible_ids"]["eligible"] is True
        assert region["consensus"]["numeric_literals"]["eligible"] is True


def test_numeric_literals_are_derived_from_lines_with_scorer_tokenizer(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    response = _response()
    for region in response["regions"]:
        region["lines"] = ["A7: 15:00 .5 -.5 −.25 1e3 +2.5E-4 1/.5 50% code1.2"]
        region["visible_ids"] = ["model observation with whitespace:"]
        region["numeric_literals"] = ["reader", "observations", "are", "not", "scored"]

    output = tmp_path / "private" / "ocr-benchmark-v2"
    _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=_successful_invoke([], [response, response]),
    )

    consensus = json.loads(
        (output / "references" / "heldout-task_1-p0001" / "consensus.json").read_text()
    )
    for region in consensus["regions"]:
        normalized = region["readers"][0]["normalized"]
        assert normalized["visible_ids"] == ["A7"]
        assert normalized["visible_ids_observed"] == ["model observation with whitespace:"]
        assert normalized["numeric_literals"] == [
            "15",
            "00",
            ".5",
            "-.5",
            "−.25",
            "1e3",
            "+2.5E-4",
            "1/.5",
            "50%",
        ]
        assert normalized["numeric_literals_observed"] == [
            "reader",
            "observations",
            "are",
            "not",
            "scored",
        ]


def test_two_clear_empty_regions_are_body_eligible(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    response = _response()
    for region in response["regions"]:
        region["lines"] = []
        region["visible_ids"] = []
        region["numeric_literals"] = []
    output = tmp_path / "private" / "ocr-benchmark-v2"

    _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=_successful_invoke([], [response, response]),
    )

    consensus = json.loads(
        (output / "references" / "heldout-task_1-p0001" / "consensus.json").read_text()
    )
    for region in consensus["regions"]:
        assert region["consensus"]["body"] == {"eligible": True, "value": ""}


def test_page_is_finalized_as_soon_as_both_readers_finish(tmp_path: Path) -> None:
    manifest = _two_page_manifest(tmp_path)
    output = tmp_path / "private" / "ocr-benchmark-v2"

    def invoke(**request):
        page_id = request["output_dir"].parents[2].name
        if page_id.startswith("train-"):
            first_state = output / "references" / "heldout-task_1-p0001" / "state.json"
            deadline = time.monotonic() + 2
            while not first_state.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert first_state.is_file()
        return _successful_invoke([])(**request)

    summary = _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=invoke,
    )
    assert summary["completed_pages"] == 2


def test_uncertain_disagreement_is_preserved_without_imputation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "private" / "ocr-benchmark-v2"
    _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=_successful_invoke([], [_response(), _response(suffix=" changed", uncertain=True)]),
    )
    consensus = json.loads(
        (output / "references" / "heldout-task_1-p0001" / "consensus.json").read_text()
    )
    for region in consensus["regions"]:
        assert region["consensus"]["body"] == {"eligible": False, "value": None}
        assert region["consensus"]["visible_ids"] == {"eligible": False, "value": None}
        assert region["consensus"]["numeric_literals"] == {"eligible": False, "value": None}
        assert len(region["disagreements"]["body"]) == 2
        assert sorted(region["disagreements"]["uncertain"]) == [False, True]


def test_runtime_retry_uses_new_directory_without_prompt_feedback(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    calls: list[dict] = []
    success = _successful_invoke(calls, [_response(), _response()])
    failed_once = False

    def invoke(**request):
        nonlocal failed_once
        calls.append(request)
        if not failed_once:
            failed_once = True
            return {
                "status": "runtime_failed",
                "response": None,
                "raw_response_path": None,
                "artifact_paths": [],
                "elapsed_seconds": 0.1,
                "error": "timeout",
            }
        calls.pop()
        return success(**request)

    summary = _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=tmp_path / "private" / "ocr-benchmark-v2",
        limit=None,
        resume=False,
        invoke=invoke,
    )
    assert summary["completed_pages"] == 1
    assert len(calls) == 3
    assert len({str(call["output_dir"]) for call in calls}) == 3
    assert all(call["prompt"] == _MODULE.TRANSCRIPTION_PROMPT for call in calls)


def test_resume_reuses_verified_page_and_rejects_tampered_artifact(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "private" / "ocr-benchmark-v2"
    _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=_successful_invoke([]),
    )
    consensus = output / "references" / "heldout-task_1-p0001" / "consensus.json"
    original_sha = _sha256(consensus)

    resumed = _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=True,
        invoke=lambda **_: pytest.fail("verified page must not be reinvoked"),
    )
    assert resumed["resumed_pages"] == 1
    assert _sha256(consensus) == original_sha

    crop = output / "references" / "heldout-task_1-p0001" / "images" / "core-1.png"
    crop.write_bytes(b"tampered")
    with pytest.raises(_MODULE.BenchmarkHarnessError, match="tamper|hash|artifact"):
        _MODULE.run_benchmark(
            manifest_path=manifest,
            output_root=output,
            limit=None,
            resume=True,
            invoke=lambda **_: pytest.fail("tampered page must fail before invocation"),
        )


def test_resume_preserves_runtime_failed_status(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "private" / "ocr-benchmark-v2"

    def failed(**request):
        return {
            "status": "runtime_failed",
            "response": None,
            "raw_response_path": None,
            "artifact_paths": [],
            "elapsed_seconds": 0.1,
            "error": "timeout",
        }

    initial = _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=failed,
    )
    assert initial["completed_pages"] == 0
    assert initial["runtime_failed_pages"] == 1

    resumed = _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=True,
        invoke=lambda **_: pytest.fail("terminal runtime failure must not be reinvoked"),
    )
    assert resumed["completed_pages"] == 0
    assert resumed["runtime_failed_pages"] == 1
    assert resumed["resumed_pages"] == 1


def test_resume_rejects_changed_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "private" / "ocr-benchmark-v2"
    _MODULE.run_benchmark(
        manifest_path=manifest,
        output_root=output,
        limit=None,
        resume=False,
        invoke=_successful_invoke([]),
    )
    payload = json.loads(manifest.read_text())
    payload["scope"] = "changed"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(_MODULE.BenchmarkHarnessError, match="manifest"):
        _MODULE.run_benchmark(
            manifest_path=manifest,
            output_root=output,
            limit=None,
            resume=True,
            invoke=_successful_invoke([]),
        )


def test_cli_timeout_kills_the_entire_process_group(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "codex"
    executable.write_bytes(b"fake executable")
    executable.chmod(0o700)
    output = tmp_path / "attempt"
    output.mkdir()
    killed: list[tuple[int, int]] = []

    class Process:
        pid = 4321
        returncode = -9
        calls = 0

        def communicate(self, *, input=None, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("codex", timeout)
            return ('{"type":"turn.started"}\n', "timed out")

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(_MODULE.subprocess, "Popen", popen)
    monkeypatch.setattr(_MODULE.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    config = {
        **_MODULE.CHECKER_CONFIG,
        "codex_executable_path": str(executable),
        "codex_executable_sha256": _sha256(executable),
    }

    result = _MODULE._invoke_codex(
        prompt="blind prompt",
        images=(),
        output_schema=_MODULE.OUTPUT_SCHEMA,
        output_dir=output,
        checker_config=config,
    )

    assert result["status"] == "runtime_failed"
    assert result["error"] == "timeout"
    assert killed == [(4321, _MODULE.signal.SIGKILL)]
