import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_solution_records.py"
_SPEC = importlib.util.spec_from_file_location("run_solution_records", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_WORKER = sys.modules["solution_paddle_worker"]
PageBundle = _MODULE.PageBundle
RunnerError = _MODULE.RunnerError
run_pipeline = _MODULE.run_pipeline
validate_cli_output_root = _MODULE.validate_cli_output_root
load_pages = _MODULE.load_pages


@pytest.fixture(autouse=True)
def _isolate_official_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _MODULE,
        "OFFICIAL_SPLITS",
        {split: dict(pin) for split, pin in _MODULE.OFFICIAL_SPLITS.items()},
    )
    monkeypatch.setattr(_MODULE, "check_source_regions", _successful_source_checker)


def _pin(split: str, tasks_path: Path, count: int) -> None:
    _MODULE.OFFICIAL_SPLITS[split] = {
        "count": count,
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
    }


def _write_tasks(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _task(instance_id: str, question: str = "What is the total?") -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "user_query": question,
        "document_pdf": f"{instance_id}.pdf",
    }


def _public_inputs(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    for row in rows:
        (pdf_root / str(row["document_pdf"])).write_bytes(b"public pdf bytes")
    return _write_tasks(tmp_path / "tasks.jsonl", rows), pdf_root


def _pages(_task: dict[str, object], _pdf: Path, cache: Path) -> PageBundle:
    image = cache / "page-1.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    return PageBundle(
        pages=[{"page_number": 1, "text": "E-7: The document states 3 plus 4."}],
        images=[image],
        ocr_provenance=_MODULE._generation_config()["ocr"],
        geometry=[
            {
                "page_number": 1,
                "width": 100,
                "height": 100,
                "ocr_image_sha256": hashlib.sha256(b"png").hexdigest(),
                "lines": [
                    {
                        "line_index": 0,
                        "text": "E-7: The document states 3 plus 4.",
                        "bbox": {"left": 1, "top": 1, "width": 90, "height": 10},
                    }
                ],
            }
        ],
    )


def _successful_source_checker(**kwargs) -> list[dict[str, object]]:
    primary = kwargs["frozen_record"]
    output_dir = Path(kwargs["output_dir"])
    checker_config = kwargs["checker_config"]
    pdf_path = Path(kwargs["source_pdf_path"])
    geometry = kwargs["ocr_geometry"]
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def artifact(kind: str, path: Path, content: bytes | None = None) -> dict[str, str]:
        if content is not None:
            path.write_bytes(content)
        item = {
            "kind": kind,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        artifacts.append(item)
        return item

    pdf_artifact = artifact("source_pdf", pdf_path)
    renderer_artifact = artifact("renderer_executable", Path(checker_config["renderer_path"]))
    checker_artifact = artifact(
        "checker_executable", Path(checker_config["codex_executable_path"])
    )
    page_artifact = artifact("page_image", output_dir / "page.png", b"png")
    context_artifact = artifact("context_crop", output_dir / "context.png", b"context")
    anchor_artifact = artifact("anchor_crop", output_dir / "anchor.png", b"anchor")
    raw_artifact = artifact("checker_response", output_dir / "response.json", b"{}")
    checks = []
    for index, region in enumerate(primary["evidence_regions"]):
        quote = "The document states 3 plus 4."
        checks.append(
            {
                "region_index": index,
                "page": region["page"],
                "ocr_anchor": region["ocr_anchor"],
                "ocr_anchor_sha256": hashlib.sha256(region["ocr_anchor"].encode()).hexdigest(),
                "primary_record_sha256": _MODULE.primary_record_digest(primary),
                "primary_response_sha256": primary["provenance"]["output_sha256"],
                "status": "fully_grounded",
                "evidence": {"id": "E-7", "page": 1, "quote": quote},
                "observed_candidates": [
                    {
                        "heading_line": "E-7:",
                        "body_text": quote,
                        "heading_legibility": "clear",
                        "body_legibility": "clear",
                        "contains_anchor_region": True,
                    }
                ],
                "source_uncertainties": [],
                "pdf_sha256": pdf_artifact["sha256"],
                "renderer_sha256": renderer_artifact["sha256"],
                "checker_executable_sha256": checker_artifact["sha256"],
                "selector_sha256": "1" * 64,
                "prompt_sha256": "2" * 64,
                "output_schema_sha256": "3" * 64,
                "checker_config_sha256": hashlib.sha256(
                    json.dumps(
                        checker_config, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "page_image_sha256": page_artifact["sha256"],
                "ocr_image_sha256": geometry[0]["ocr_image_sha256"],
                "context_crop_sha256": context_artifact["sha256"],
                "anchor_crop_sha256": anchor_artifact["sha256"],
                "raw_response_sha256": raw_artifact["sha256"],
                "model": checker_config["model"],
                "method": checker_config["method"],
                "anchor_bbox": {"left": 1, "top": 1, "right": 91, "bottom": 11},
                "context_bbox": {"left": 0, "top": 0, "right": 100, "bottom": 100},
                "anchor_crop_bbox": {"left": 0, "top": 0, "right": 100, "bottom": 23},
                "pdf_artifact": pdf_artifact,
                "renderer_artifact": renderer_artifact,
                "page_image_artifact": page_artifact,
                "context_crop_artifact": context_artifact,
                "anchor_crop_artifact": anchor_artifact,
                "checker_executable_artifact": checker_artifact,
                "raw_response_artifact": raw_artifact,
                "artifacts": artifacts,
            }
        )
    return checks


def _record_heldout_evaluation(split_dir: Path) -> None:
    evaluator_path = Path(__file__).parents[1] / "scripts" / "evaluate_solution_records.py"
    spec = importlib.util.spec_from_file_location("test_solution_evaluator_helper", evaluator_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.record_unavailable_evaluation(split_dir)


def _successful_codex(calls: list[dict[str, object]]):
    def run(argv, **kwargs):
        response_path = Path(argv[argv.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "answer": "7",
                    "solution": {
                        "summary": "The cited values add to the requested total.",
                        "calculations": [{"expression": "3 + 4", "result": "7"}],
                    },
                    "evidence_regions": [
                        {"page": 1, "ocr_anchor": "The document states 3 plus 4."}
                    ],
                    "uncertainties": [],
                }
            ),
            encoding="utf-8",
        )
        calls.append({"argv": argv, "prompt": kwargs["input"]})
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                '{"type":"thread.started","thread_id":"t"}\n'
                '{"type":"turn.completed","usage":{"input_tokens":10,'
                '"cached_input_tokens":0,"output_tokens":20}}\n'
            ),
            stderr="warning retained",
        )

    return run


def test_split_order_requires_verified_prior_coverage_but_smoke_is_independent(
    tmp_path: Path,
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("v1")])
    _pin("validation", tasks, 1)
    output = tmp_path / "private"

    with pytest.raises(RunnerError, match="heldout.*complete coverage"):
        run_pipeline(
            split="validation",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )

    result = run_pipeline(
        split="validation",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        limit=1,
        smoke=True,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )

    assert result["succeeded"] == 1
    assert (output / "smoke" / "validation" / "jobs" / "v1" / "record.json").is_file()
    assert not (output / "validation" / "complete.json").exists()


def test_train_refuses_partial_heldout_even_if_marker_was_fabricated(tmp_path: Path) -> None:
    heldout_tasks, pdf_root = _public_inputs(tmp_path, [_task("h1"), _task("h2")])
    _pin("heldout", heldout_tasks, 2)
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=heldout_tasks,
        pdf_root=pdf_root,
        output_root=output,
        limit=1,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    (output / "heldout" / "complete.json").write_text("{}", encoding="utf-8")
    train_tasks = _write_tasks(tmp_path / "train.jsonl", [_task("t1")])
    _pin("train", train_tasks, 1)
    (pdf_root / "t1.pdf").write_bytes(b"train")

    with pytest.raises(RunnerError, match="heldout.*complete coverage"):
        run_pipeline(
            split="train",
            tasks_path=train_tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )


def test_resume_skips_valid_success_and_rejects_input_or_config_hash_change(
    tmp_path: Path,
) -> None:
    rows = [_task("h1")]
    tasks, pdf_root = _public_inputs(tmp_path, rows)
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"
    calls: list[dict[str, object]] = []
    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
    )
    assert len(calls) == 1

    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        resume=True,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
    )
    assert len(calls) == 1

    manifest_path = output / "heldout" / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["command_config"]["model_reasoning_effort"] = "low"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunnerError, match="command config hash"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )

    manifest["command_config"]["model_reasoning_effort"] = "high"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (pdf_root / "h1.pdf").write_bytes(b"changed")
    with pytest.raises(RunnerError, match="input manifest hash"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )


def test_failed_or_empty_completion_is_recorded_without_a_solution(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    prompts: list[str] = []

    def empty_completion(argv, **kwargs):
        prompts.append(kwargs["input"])
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":1}}\n',
            stderr="",
        )

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        page_loader=_pages,
        command_runner=empty_completion,
    )

    job = tmp_path / "private" / "heldout" / "jobs" / "h1"
    failure = json.loads((job / "error.json").read_text(encoding="utf-8"))
    assert result["split"] == "heldout"
    assert result["total"] == 1
    assert result["succeeded"] == 0
    assert result["failed"] == 1
    assert result["error_counts"] == {"empty_final_response": 1}
    assert failure["status"] == "failed"
    assert failure["error_kind"] == "empty_final_response"
    assert len(failure["attempts"]) == 2
    assert "RETRY DIAGNOSTIC" not in prompts[0]
    assert "RETRY DIAGNOSTIC" in prompts[1]
    assert "empty final response" in prompts[1]
    assert (job / "attempts" / "1" / "events.jsonl").is_file()
    assert (job / "attempts" / "2" / "events.jsonl").is_file()
    assert (job / "source" / "page-1.jpg").is_file()
    assert not (job / "record.json").exists()
    assert not (tmp_path / "private" / "heldout" / "complete.json").exists()
    failures = json.loads(
        (tmp_path / "private" / "heldout" / "failures.json").read_text(encoding="utf-8")
    )
    assert failures == {
        "failed": 1,
        "error_counts": {"empty_final_response": 1},
        "tasks": [{"instance_id": "h1", "error_kind": "empty_final_response"}],
    }
    original_attempts = {
        number: (job / "attempts" / number / "events.jsonl").read_bytes()
        for number in ("1", "2")
    }

    resumed = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        resume=True,
        page_loader=_pages,
        command_runner=empty_completion,
    )

    assert resumed["failed"] == 1
    assert len(prompts) == 2
    assert {
        number: (job / "attempts" / number / "events.jsonl").read_bytes()
        for number in ("1", "2")
    } == original_attempts


def test_jobs_are_private_and_codex_sees_only_current_public_input(tmp_path: Path) -> None:
    rows = [_task("h1", "Question one"), _task("h2", "Question two")]
    tasks, pdf_root = _public_inputs(tmp_path, rows)
    _pin("heldout", tasks, 2)
    calls: list[dict[str, object]] = []

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        limit=1,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
    )

    assert result["succeeded"] == 1
    call = calls[0]
    prompt = call["prompt"]
    assert isinstance(prompt, str)
    assert "Question one" in prompt
    assert "Question two" not in prompt
    assert "answer" not in rows[0]
    argv = call["argv"]
    assert isinstance(argv, list)
    manifest = json.loads(
        (tmp_path / "private" / "heldout" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert argv[0] == manifest["command_config"]["codex_executable_path"]
    assert manifest["command_config"]["codex_executable_sha256"]
    assert 'model_reasoning_effort="high"' in argv
    assert argv[argv.index("--model") + 1] == "gpt-6-astra"
    assert 'features.shell_tool=false' in argv
    assert argv.count("--image") == 1
    job = tmp_path / "private" / "heldout" / "jobs" / "h1"
    assert (job.stat().st_mode & 0o077) == 0
    provenance = json.loads((job / "record.json").read_text(encoding="utf-8"))["provenance"]
    assert provenance["model"] == "gpt-6-astra"
    assert provenance["model_identity_evidence"] == (
        "requested-model-plus-completed-turn-without-fallback-warning"
    )
    assert provenance["pdf_sha256"]
    assert provenance["input_sha256"]
    assert provenance["output_sha256"]
    assert provenance["page_images"][0]["sha256"]
    assert not (job / "source" / "page-1.jpg").exists()
    assert manifest["command_config"]["ocr"]["successful_page_jpegs"] == (
        "exact_regenerable_cache_v1"
    )
    assert manifest["command_config"]["ocr"]["image_cache_helper_sha256"]
    assert provenance["usage"]["input_tokens"] == 10


def test_partial_page_jpeg_cleanup_failure_preserves_verified_success_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"
    primary_calls: list[dict[str, object]] = []

    def two_pages(task, pdf, cache):
        bundle = _pages(task, pdf, cache)
        second = cache / "page-2.jpg"
        second.write_bytes(b"jpeg-2")
        bundle.pages.append({"page_number": 2, "text": "Additional public source text."})
        bundle.images.append(second)
        bundle.geometry.append(
            {
                "page_number": 2,
                "width": 100,
                "height": 100,
                "ocr_image_sha256": hashlib.sha256(b"ocr-2").hexdigest(),
                "lines": [],
            }
        )
        return bundle

    def partial_cleanup(*, page_images, **_kwargs):
        first_present = next(
            (Path(image["path"]) for image in page_images if Path(image["path"]).exists()),
            None,
        )
        if first_present is not None:
            first_present.unlink()
        raise OSError("simulated cleanup interruption")

    monkeypatch.setattr(_MODULE, "evict_verified_success_page_jpegs", partial_cleanup)
    first = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=two_pages,
        command_runner=_successful_codex(primary_calls),
    )

    job = output / "heldout" / "jobs" / "h1"
    state_before_resume = (job / "job.json").read_bytes()
    record_before_resume = (job / "record.json").read_bytes()
    warning = json.loads((job / "cache-cleanup-warning.json").read_text())
    assert first["succeeded"] == 1
    assert len(primary_calls) == 1
    assert json.loads(state_before_resume)["status"] == "succeeded"
    assert warning["status"] == "verified_success_cache_cleanup_failed"
    assert not (job / "error.json").exists()
    assert not (job / "source" / "page-1.jpg").exists()
    assert (job / "source" / "page-2.jpg").is_file()

    resumed = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        resume=True,
        page_loader=two_pages,
        command_runner=_successful_codex(primary_calls),
    )

    assert resumed["succeeded"] == 1
    assert len(primary_calls) == 1
    assert (job / "job.json").read_bytes() == state_before_resume
    assert (job / "record.json").read_bytes() == record_before_resume
    assert not (job / "source" / "page-2.jpg").exists()


@pytest.mark.parametrize(
    "rows, message",
    [
        ([_task("same"), _task("same")], "duplicate instance_id"),
        ([{"user_query": "q", "document_pdf": "x.pdf"}], "missing instance_id"),
    ],
)
def test_manifest_rejects_missing_and_duplicate_ids(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    tasks = _write_tasks(tmp_path / "tasks.jsonl", rows)
    _pin("heldout", tasks, len(rows))
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()

    with pytest.raises(RunnerError, match=message):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=tmp_path / "private",
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )


def test_official_short_manifest_cannot_complete_or_unlock_train(tmp_path: Path) -> None:
    heldout_tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    output = tmp_path / "private"

    with pytest.raises(RunnerError, match="official heldout task manifest"):
        run_pipeline(
            split="heldout",
            tasks_path=heldout_tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )

    _pin("heldout", heldout_tasks, 1)
    run_pipeline(
        split="heldout",
        tasks_path=heldout_tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    _MODULE.OFFICIAL_SPLITS["heldout"] = {"count": 2, "tasks_sha256": "0" * 64}
    train_tasks = _write_tasks(tmp_path / "train.jsonl", [_task("t1")])
    _pin("train", train_tasks, 1)
    (pdf_root / "t1.pdf").write_bytes(b"train")

    with pytest.raises(RunnerError, match="heldout.*official manifest pin"):
        run_pipeline(
            split="train",
            tasks_path=train_tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )


def test_smoke_requires_exactly_one_validation_task_and_never_completes(
    tmp_path: Path,
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("v1")])
    _pin("validation", tasks, 1)
    output = tmp_path / "private"

    with pytest.raises(RunnerError, match="smoke.*validation.*limit 1"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            smoke=True,
            limit=1,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )
    with pytest.raises(RunnerError, match="smoke.*validation.*limit 1"):
        run_pipeline(
            split="validation",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            smoke=True,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )

    run_pipeline(
        split="validation",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        smoke=True,
        limit=1,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    assert not (output / "smoke" / "validation" / "complete.json").exists()


def test_smoke_instance_selector_uses_named_official_task(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(
        tmp_path, [_task("v1", "First question"), _task("v2", "Named question")]
    )
    _pin("validation", tasks, 2)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"

    result = run_pipeline(
        split="validation",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        smoke=True,
        smoke_instance_id="v2",
        limit=1,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
    )

    assert result["total"] == 1
    assert "Named question" in str(calls[0]["prompt"])
    assert "First question" not in str(calls[0]["prompt"])
    assert (output / "smoke" / "validation" / "jobs" / "v2" / "record.json").is_file()
    assert not (output / "smoke" / "validation" / "jobs" / "v1").exists()


def test_cli_output_must_resolve_inside_gitignored_issue24_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    private_root = repository / "artifacts" / "solution-records" / "issue24"
    private_root.mkdir(parents=True)
    (repository / ".gitignore").write_text("artifacts/solution-records/\n", encoding="utf-8")
    monkeypatch.setattr(_MODULE, "REPOSITORY_ROOT", repository)

    assert validate_cli_output_root(private_root / "runs") == (private_root / "runs").resolve()
    with pytest.raises(RunnerError, match="private issue24 artifact tree"):
        validate_cli_output_root(repository / "public-results")

    outside = tmp_path / "outside"
    outside.mkdir()
    (private_root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RunnerError, match="private issue24 artifact tree"):
        validate_cli_output_root(private_root / "escape" / "runs")


def test_resume_rejects_deleted_raw_success_response(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"
    calls: list[dict[str, object]] = []
    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
    )
    (output / "heldout" / "jobs" / "h1" / "attempts" / "1" / "response.json").unlink()

    with pytest.raises(RunnerError, match="invalid or interrupted job artifacts"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_pages,
            command_runner=_successful_codex(calls),
        )
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_resume_rejects_missing_or_tampered_failed_job_input(
    tmp_path: Path, mutation: str
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"
    calls = 0

    def empty_completion(argv, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":1}}\n',
            stderr="",
        )

    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=empty_completion,
    )
    input_path = output / "heldout" / "jobs" / "h1" / "input.json"
    if mutation == "delete":
        input_path.unlink()
    else:
        frozen_input = json.loads(input_path.read_text(encoding="utf-8"))
        frozen_input["source_pages"][0]["text"] = "tampered public source"
        input_path.write_text(json.dumps(frozen_input), encoding="utf-8")

    with pytest.raises(RunnerError, match="invalid or interrupted job artifacts"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_pages,
            command_runner=empty_completion,
        )
    assert calls == 2


@pytest.mark.parametrize(
    "generation_setting", ["prompt", "prompt_layout", "retry_policy", "schema", "ocr"]
)
def test_resume_rejects_generation_contract_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generation_setting: str
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    schema_path = output / "heldout" / "schema.json"
    original_schema = schema_path.read_bytes()
    if generation_setting == "prompt":
        monkeypatch.setattr(
            _MODULE, "PROMPT_INSTRUCTIONS", _MODULE.PROMPT_INSTRUCTIONS + " changed"
        )
    elif generation_setting == "prompt_layout":
        monkeypatch.setattr(
            _MODULE, "BASE_PROMPT_TEMPLATE", _MODULE.BASE_PROMPT_TEMPLATE + " changed"
        )
    elif generation_setting == "retry_policy":
        monkeypatch.setattr(
            _MODULE, "RETRY_PROMPT_TEMPLATE", _MODULE.RETRY_PROMPT_TEMPLATE + " changed"
        )
    elif generation_setting == "schema":
        changed_schema = dict(_MODULE.OUTPUT_SCHEMA)
        changed_schema["title"] = "changed"
        monkeypatch.setattr(_MODULE, "OUTPUT_SCHEMA", changed_schema)
    else:
        changed_ocr = dict(_MODULE.OCR_CONFIG)
        changed_ocr["dpi"] = 300
        monkeypatch.setattr(_MODULE, "OCR_CONFIG", changed_ocr)

    with pytest.raises(RunnerError, match="command config hash"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )
    assert schema_path.read_bytes() == original_schema


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [("solutions.jsonl", "delete"), ("submission.jsonl", "tamper")],
)
def test_prior_split_requires_intact_complete_exports(
    tmp_path: Path, artifact: str, mutation: str
) -> None:
    heldout_tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", heldout_tasks, 1)
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=heldout_tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    artifact_path = output / "heldout" / artifact
    if mutation == "delete":
        artifact_path.unlink()
    else:
        artifact_path.write_text("tampered\n", encoding="utf-8")
    train_tasks = _write_tasks(tmp_path / "train.jsonl", [_task("t1")])
    _pin("train", train_tasks, 1)
    (pdf_root / "t1.pdf").write_bytes(b"train")

    with pytest.raises(RunnerError, match="heldout.*complete export"):
        run_pipeline(
            split="train",
            tasks_path=train_tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )


def test_split_directory_symlink_cannot_escape_private_output(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"
    output.mkdir()
    outside = tmp_path / "public"
    outside.mkdir()
    (output / "heldout").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RunnerError, match="symlink"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )
    assert list(outside.iterdir()) == []


def test_private_write_ignores_predictable_temporary_symlink(tmp_path: Path) -> None:
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched", encoding="utf-8")
    predictable = private_dir / ".result.json.tmp-12345"
    predictable.symlink_to(outside)

    _MODULE._write_private(private_dir / "result.json", "private payload")

    assert (private_dir / "result.json").read_text(encoding="utf-8") == "private payload"
    assert outside.read_text(encoding="utf-8") == "untouched"
    assert predictable.is_symlink()


def test_malformed_jsonl_event_fails_and_preserves_raw_stream(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"

    def malformed_events(argv, **_kwargs):
        response_path = Path(argv[argv.index("--output-last-message") + 1])
        response_path.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='not-json\n{"type":"turn.completed","usage":{}}\n',
            stderr="",
        )

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=malformed_events,
    )

    job = output / "heldout" / "jobs" / "h1"
    assert result["error_counts"] == {"invalid_event_stream": 1}
    assert (job / "attempts" / "1" / "events.jsonl").read_text(encoding="utf-8").startswith(
        "not-json\n"
    )
    assert not (job / "record.json").exists()


def test_pdf_mutation_during_page_loading_is_rejected(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)

    def mutating_loader(task: dict[str, object], pdf: Path, cache: Path) -> PageBundle:
        bundle = _pages(task, pdf, cache)
        pdf.write_bytes(b"changed during extraction")
        return bundle

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        page_loader=mutating_loader,
        command_runner=_successful_codex([]),
    )

    assert result["succeeded"] == 0
    assert result["error_counts"] == {"input_changed": 1}


def test_validation_requires_train_evaluation_binding(tmp_path: Path) -> None:
    heldout_tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", heldout_tasks, 1)
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=heldout_tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    _record_heldout_evaluation(output / "heldout")
    train_tasks = _write_tasks(tmp_path / "train.jsonl", [_task("t1")])
    _pin("train", train_tasks, 1)
    (pdf_root / "t1.pdf").write_bytes(b"train")
    run_pipeline(
        split="train",
        tasks_path=train_tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    validation_tasks = _write_tasks(tmp_path / "validation.jsonl", [_task("v1")])
    _pin("validation", validation_tasks, 1)
    (pdf_root / "v1.pdf").write_bytes(b"validation")

    with pytest.raises(RunnerError, match="train evaluation binding"):
        run_pipeline(
            split="validation",
            tasks_path=validation_tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )


def test_train_requires_heldout_no_reference_evaluation_binding(tmp_path: Path) -> None:
    heldout_tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", heldout_tasks, 1)
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=heldout_tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    train_tasks = _write_tasks(tmp_path / "train.jsonl", [_task("t1")])
    _pin("train", train_tasks, 1)
    (pdf_root / "t1.pdf").write_bytes(b"train")

    with pytest.raises(RunnerError, match="heldout evaluation binding"):
        run_pipeline(
            split="train",
            tasks_path=train_tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )


def test_bundle_ocr_provenance_must_equal_frozen_manifest(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []

    def mismatched_loader(task: dict[str, object], pdf: Path, cache: Path) -> PageBundle:
        bundle = _pages(task, pdf, cache)
        return PageBundle(
            pages=bundle.pages,
            images=bundle.images,
            ocr_provenance={**bundle.ocr_provenance, "dpi": 300},
        )

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        page_loader=mismatched_loader,
        command_runner=_successful_codex(calls),
    )

    assert result["succeeded"] == 0
    assert calls == []
    assert result["error_counts"] == {"runtime_error": 1}


def test_source_check_runs_only_after_primary_is_validated_and_frozen(
    tmp_path: Path,
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"

    source_calls: list[dict[str, object]] = []

    def checking_source(**kwargs):
        primary = kwargs["frozen_record"]
        attempt_dir = Path(kwargs["output_dir"]).parent
        source_calls.append(primary)
        assert (attempt_dir / "primary-response.json").is_file()
        assert (attempt_dir / "primary-record.json").is_file()
        assert "evidence" not in primary
        assert "evidence_details" not in primary
        return _successful_source_checker(**kwargs)

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
        source_checker=checking_source,
    )

    assert result["succeeded"] == 1
    assert len(calls) == 1
    assert len(source_calls) == 1
    record = json.loads(
        (output / "heldout" / "jobs" / "h1" / "record.json").read_text(encoding="utf-8")
    )
    assert record["source_status"] == "fully_grounded"
    assert record["evidence"] == ["E-7"]
    assert record["evidence_details"][0]["quote"] == "The document states 3 plus 4."


def test_unit_bearing_direct_answer_is_retried_before_source_check(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    primary_calls = 0
    source_calls = 0

    def answer_runner(argv, **_kwargs):
        nonlocal primary_calls
        primary_calls += 1
        response_path = Path(argv[argv.index("--output-last-message") + 1])
        response_path.write_text(
            json.dumps(
                {
                    "answer": "12.5 million" if primary_calls == 1 else "12.5",
                    "solution": {
                        "summary": "The source reports 12.5 million units.",
                        "calculations": [],
                    },
                    "evidence_regions": [
                        {"page": 1, "ocr_anchor": "The document states 3 plus 4."}
                    ],
                    "uncertainties": ["The unit is retained in the summary."],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv, 0, stdout='{"type":"turn.completed","usage":{}}\n', stderr=""
        )

    def counting_source(**kwargs):
        nonlocal source_calls
        source_calls += 1
        return _successful_source_checker(**kwargs)

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        page_loader=_pages,
        command_runner=answer_runner,
        source_checker=counting_source,
    )

    assert result["succeeded"] == 1
    assert primary_calls == 2
    assert source_calls == 1


def test_source_checker_runtime_failure_resumes_from_frozen_primary_without_primary_retry(
    tmp_path: Path,
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"

    source_calls = 0

    def transient_source(**kwargs):
        nonlocal source_calls
        source_calls += 1
        if source_calls == 1:
            raise _MODULE.SourceRegionCheckError("checker unavailable")
        return _successful_source_checker(**kwargs)

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
        source_checker=transient_source,
    )

    assert result["succeeded"] == 0
    assert result["error_counts"] == {"source_check_runtime_failed": 1}
    assert len(calls) == 1
    assert source_calls == 1
    job = output / "heldout" / "jobs" / "h1"
    assert (job / "attempts" / "1" / "primary-response.json").is_file()
    assert not (job / "attempts" / "2").exists()
    resumed = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        resume=True,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
        source_checker=transient_source,
    )
    assert resumed["succeeded"] == 1
    assert len(calls) == 1
    assert source_calls == 2
    assert (job / "record.json").is_file()
    assert not (job / "error.json").exists()


def test_source_checker_timeout_preserves_partial_streams_and_uses_source_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "source-check"
    timeout = subprocess.TimeoutExpired(
        cmd=["codex"],
        timeout=1,
        output='{"type":"turn.started"}\n',
        stderr="checker still running\n",
    )

    def timed_out(*_args, **_kwargs):
        raise timeout

    monkeypatch.setattr(_MODULE, "_verify_codex_runtime", lambda _config: None)

    with pytest.raises(_MODULE.SourceRegionCheckError, match="timed out"):
        _MODULE._invoke_source_checker(
            prompt="check source",
            images=[],
            output_schema={"type": "object"},
            output_dir=output_dir,
            checker_config={
                "model": "gpt-test",
                "codex_executable_path": "/tmp/codex",
                "timeout_seconds": 1,
            },
            command_runner=timed_out,
        )

    assert (output_dir / "events.jsonl").read_text() == '{"type":"turn.started"}\n'
    assert (output_dir / "stderr.txt").read_text() == "checker still running\n"


def test_isolated_command_timeout_kills_parent_and_grandchild_and_preserves_streams(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "pids.txt"
    child = tmp_path / "parent.py"
    child.write_text(
        "\n".join(
            (
                "import os, pathlib, subprocess, sys, time",
                "grandchild = subprocess.Popen("
                "[sys.executable, '-c', 'import time; time.sleep(60)'])",
                "pathlib.Path(sys.argv[1]).write_text(f'{os.getpid()} {grandchild.pid}')",
                "print('partial stdout', flush=True)",
                "print('partial stderr', file=sys.stderr, flush=True)",
                "time.sleep(60)",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _MODULE._run_isolated(
            [sys.executable, str(child), str(pid_file)],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )

    assert caught.value.stdout == "partial stdout\n"
    assert caught.value.stderr == "partial stderr\n"
    parent_pid, grandchild_pid = (int(value) for value in pid_file.read_text().split())
    deadline = time.monotonic() + 3
    states: dict[int, str] = {}
    while time.monotonic() < deadline:
        states = {
            pid: subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
            ).stdout.strip()
            for pid in (parent_pid, grandchild_pid)
        }
        if all(not state or state.startswith("Z") for state in states.values()):
            break
        time.sleep(0.05)
    assert all(not state or state.startswith("Z") for state in states.values())


def test_pipeline_uses_isolated_command_runner_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(_MODULE, "_run_isolated", _successful_codex(calls))

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        limit=1,
        page_loader=_pages,
    )

    assert result["succeeded"] == 1
    assert len(calls) == 1


def test_resolved_codex_runtime_executes_and_pins_native_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "node_modules" / "@openai" / "codex"
    wrapper = package_root / "bin" / "codex.js"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "#!/usr/bin/env node\n"
        "const PLATFORM_PACKAGE_BY_TARGET = {};\n"
        "function findCodexExecutable() {}\n"
        "const vendorRoot = 'vendor';\n"
        "const targetTriple = 'aarch64-apple-darwin';\n"
        "const codexExecutable = path.join(vendorRoot, targetTriple, 'bin',\n"
        '  process.platform === "win32" ? "codex.exe" : "codex");\n',
        encoding="utf-8",
    )
    native = (
        package_root
        / "node_modules"
        / "@openai"
        / "codex-darwin-arm64"
        / "vendor"
        / "aarch64-apple-darwin"
        / "bin"
        / "codex"
    )
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native codex")
    monkeypatch.setattr(_MODULE.shutil, "which", lambda _name: str(wrapper))
    monkeypatch.setattr(_MODULE.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_MODULE.platform, "machine", lambda: "arm64")
    calls: list[list[str]] = []

    def version(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="codex-cli 0.153.4\n", stderr="")

    monkeypatch.setattr(_MODULE.subprocess, "run", version)
    monkeypatch.setitem(_MODULE.COMMAND_CONFIG, "codex_executable", "codex")
    monkeypatch.setitem(
        _MODULE.COMMAND_CONFIG,
        "codex_wrapper_sha256",
        hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    )
    monkeypatch.setitem(
        _MODULE.COMMAND_CONFIG,
        "codex_native_sha256",
        hashlib.sha256(native.read_bytes()).hexdigest(),
    )

    resolved = _MODULE._resolved_codex_runtime()

    assert calls == [[str(native.resolve()), "--version"]]
    assert resolved["codex_executable_path"] == str(native.resolve())
    assert resolved["codex_executable_sha256"] == hashlib.sha256(b"native codex").hexdigest()
    assert resolved["codex_wrapper_path"] == str(wrapper.resolve())
    assert resolved["codex_wrapper_sha256"] == hashlib.sha256(wrapper.read_bytes()).hexdigest()


def test_production_codex_runtime_is_isolated_astra() -> None:
    resolved = _MODULE._resolved_codex_runtime()

    assert _MODULE.COMMAND_CONFIG["model"] == "gpt-6-astra"
    assert _MODULE.COMMAND_CONFIG["model_reasoning_effort"] == "high"
    assert _MODULE.SOURCE_CHECK_CONFIG["model"] == "gpt-6-astra"
    assert _MODULE.SOURCE_CHECK_CONFIG["model_reasoning_effort"] == "high"
    assert resolved["codex_observed_version"] == "codex-cli 0.153.4"
    assert resolved["codex_wrapper_path"].endswith(
        "data/issue24/codex-0.153.4/node_modules/@openai/codex/bin/codex.js"
    )
    assert resolved["codex_wrapper_sha256"] == (
        "61b0194f3bb6534439c8d26a3ed57d0805f84b884588b761795323eeb92fcf70"
    )
    assert resolved["codex_executable_sha256"] == (
        "b973d440acac501fd2594a43e7ca9ce41e0a65b9dfb28d0d7a7837c99e1261e3"
    )


def test_model_fallback_warning_is_detected() -> None:
    old_cli_events = (
        '{"type":"item.completed","item":{"type":"error","message":'
        '"Model metadata for `gpt-6-astra` not found. Defaulting to fallback metadata"}}\n'
    )

    assert _MODULE._model_fallback_warning(old_cli_events, "") is not None
    assert _MODULE._model_fallback_warning(
        '{"type":"turn.started"}\n{"type":"turn.completed","usage":{}}\n', ""
    ) is None


def test_primary_rejects_model_fallback_warning(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    successful = _successful_codex([])

    def fallback(argv, **kwargs):
        completed = successful(argv, **kwargs)
        warning = (
            '{"type":"item.completed","item":{"type":"error","message":'
            '"Model metadata for `gpt-6-astra` not found. Defaulting to fallback metadata"}}\n'
        )
        return subprocess.CompletedProcess(
            argv,
            completed.returncode,
            stdout=warning + completed.stdout,
            stderr=completed.stderr,
        )

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        limit=1,
        page_loader=_pages,
        command_runner=fallback,
    )

    error = json.loads(
        (tmp_path / "private" / "heldout" / "jobs" / "h1" / "error.json").read_text()
    )
    assert result["failed"] == 1
    assert error["error_kind"] == "model_fallback_warning"
    assert len(error["attempts"]) == 2


def test_source_checker_uses_astra_and_rejects_fallback_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[str] = []

    def fallback(argv, **_kwargs):
        observed.extend(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                '{"type":"item.completed","item":{"type":"error","message":'
                '"Model metadata for `gpt-6-astra` not found. Defaulting to fallback metadata"}}\n'
                '{"type":"turn.completed","usage":{}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(_MODULE, "_verify_codex_runtime", lambda _config: None)
    with pytest.raises(_MODULE.SourceRegionCheckError, match="fallback metadata"):
        _MODULE._invoke_source_checker(
            prompt="inspect pixels",
            images=[],
            output_schema={"type": "object"},
            output_dir=tmp_path / "checker",
            checker_config={
                **_MODULE.SOURCE_CHECK_CONFIG,
                "codex_executable_path": "/isolated/codex",
            },
            command_runner=fallback,
        )

    assert observed[observed.index("--model") + 1] == "gpt-6-astra"


def test_source_runtime_resume_rejects_tampered_frozen_primary(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"

    def failing_source(**_kwargs):
        raise _MODULE.SourceRegionCheckError("checker unavailable")

    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
        source_checker=failing_source,
    )
    primary_path = output / "heldout" / "jobs" / "h1" / "attempts" / "1" / "primary-record.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary["answer"] = "8"
    primary_path.write_text(json.dumps(primary), encoding="utf-8")

    with pytest.raises(RunnerError, match="invalid or interrupted job artifacts"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_pages,
            command_runner=_successful_codex(calls),
            source_checker=_successful_source_checker,
        )
    assert len(calls) == 1


def test_unresolved_source_check_preserves_answer_and_completes_private_coverage(
    tmp_path: Path,
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"

    def unresolved_source(**kwargs):
        checks = _successful_source_checker(**kwargs)
        check = checks[0]
        check["status"] = "evidence_unresolved"
        check["evidence"] = {
            "id": None,
            "page": 1,
            "quote": "The document states 3 plus 4.",
        }
        observations = check["observed_candidates"]
        assert isinstance(observations, list) and isinstance(observations[0], dict)
        observations[0]["heading_legibility"] = "ambiguous"
        check["source_uncertainties"] = ["The source heading is ambiguous."]
        return checks

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
        source_checker=unresolved_source,
    )

    record = json.loads(
        (output / "heldout" / "jobs" / "h1" / "record.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output / "heldout" / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((output / "heldout" / "complete.json").read_text(encoding="utf-8"))
    assert result["succeeded"] == 1
    assert record["answer"] == "7"
    assert record["source_status"] == "evidence_unresolved"
    assert record["evidence"] == []
    assert record["evidence_details"][0]["id"] is None
    assert manifest["coverage_complete"] is True
    assert manifest["submission_ready"] is False
    assert manifest["submission_mode"] == "blocked_unresolved"
    assert not (output / "heldout" / "submission.jsonl").exists()
    assert complete["coverage_complete"] is True
    assert complete["evidence_unresolved_count"] == 1


def test_resume_rejects_tampered_source_check_artifact(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex(calls),
    )
    crop = next(
        (output / "heldout" / "jobs" / "h1" / "attempts" / "1").rglob("context.png")
    )
    crop.write_bytes(b"tampered")

    with pytest.raises(RunnerError, match="invalid or interrupted job artifacts"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_pages,
            command_runner=_successful_codex(calls),
        )
    assert len(calls) == 1


@pytest.mark.parametrize("consumer", ["resume", "prior_split"])
def test_frozen_job_ocr_provenance_is_reverified(
    tmp_path: Path, consumer: str
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    job_dir = output / "heldout" / "jobs" / "h1"
    input_path = job_dir / "input.json"
    frozen_input = json.loads(input_path.read_text(encoding="utf-8"))
    frozen_input["ocr"]["dpi"] = 300
    input_path.write_text(json.dumps(frozen_input), encoding="utf-8")
    state_path = job_dir / "job.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["job_input_sha256"] = hashlib.sha256(
        json.dumps(
            frozen_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    if consumer == "resume":
        with pytest.raises(RunnerError, match="invalid or interrupted job artifacts"):
            run_pipeline(
                split="heldout",
                tasks_path=tasks,
                pdf_root=pdf_root,
                output_root=output,
                resume=True,
                page_loader=_pages,
                command_runner=_successful_codex([]),
            )
    else:
        train_tasks = _write_tasks(tmp_path / "train.jsonl", [_task("t1")])
        _pin("train", train_tasks, 1)
        (pdf_root / "t1.pdf").write_bytes(b"train")
        with pytest.raises(RunnerError, match="heldout.*complete coverage"):
            run_pipeline(
                split="train",
                tasks_path=train_tasks,
                pdf_root=pdf_root,
                output_root=output,
                page_loader=_pages,
                command_runner=_successful_codex([]),
            )


def test_paddle_loader_uses_lossless_ocr_and_retained_jpeg_in_numeric_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    render_calls: list[list[str]] = []
    ocr_calls: list[Path] = []

    def fake_render(argv, **_kwargs):
        render_calls.append(list(argv))
        _MODULE.Image.new("RGB", (200, 100), "white").save(f"{argv[-1]}-10.png")
        _MODULE.Image.new("RGB", (200, 100), "white").save(f"{argv[-1]}-2.png")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def fake_document(images):
        results = []
        for image in images:
            image_path = Path(image)
            ocr_calls.append(image_path)
            results.append(
                {
                    "texts": [
                        "TRAINING COPY",
                        f" {image_path.stem.replace('ocr-page', 'page')} ",
                    ],
                    "scores": [0.99, 0.98],
                    "boxes": [[10.2, 20.4, 80.7, 32.0], [15.0, 40.0, 90.0, 52.0]],
                    "polygons": [
                        [[10.2, 20.4], [80.7, 20.1], [80.2, 31.9], [10.0, 32.0]],
                        [[15.0, 40.0], [90.0, 40.0], [90.0, 52.0], [15.0, 52.0]],
                    ],
                }
            )
        return results

    monkeypatch.setattr(_MODULE, "_run_checked", fake_render)
    monkeypatch.setattr(_MODULE, "_paddle_document", fake_document)
    bundle = load_pages(_task("h1"), pdf, tmp_path / "split" / "jobs" / "h1" / "source")

    assert render_calls == [
        [
            _MODULE._resolved_ocr_config()["renderer_path"],
            "-png",
            "-r",
            "175",
            str(pdf),
            str(tmp_path / "split" / "jobs" / "h1" / "source" / "ocr-page"),
        ],
    ]
    assert [page["page_number"] for page in bundle.pages] == [2, 10]
    assert [page["text"] for page in bundle.pages] == [
        "TRAINING COPY\n page-2 ",
        "TRAINING COPY\n page-10 ",
    ]
    assert [path.stem for path in ocr_calls] == ["ocr-page-2", "ocr-page-10"]
    assert not any(path.exists() for path in ocr_calls)
    assert bundle.geometry[0]["lines"][0]["bbox"] == {
        "left": 10,
        "top": 20,
        "width": 71,
        "height": 12,
    }
    assert bundle.geometry[0]["lines"][0]["raw_bbox"] == [10.2, 20.4, 80.7, 32.0]
    assert bundle.geometry[0]["lines"][0]["raw_geometry"] == {
        "quadrilateral": [[10.2, 20.4], [80.7, 20.1], [80.2, 31.9], [10.0, 32.0]]
    }
    assert bundle.geometry[0]["ocr_image_sha256"]
    assert all("tesseract" not in call for call in render_calls)
    assert bundle.ocr_provenance["detector_tree_sha256"]
    assert bundle.ocr_provenance["recognizer_tree_sha256"]


def test_paddle_failure_has_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    calls: list[list[str]] = []

    def fake_render(argv, **_kwargs):
        calls.append(list(argv))
        _MODULE.Image.new("RGB", (200, 100), "white").save(f"{argv[-1]}-1.png")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def failing_document(_images):
        raise RuntimeError("PaddleOCR failed")

    monkeypatch.setattr(_MODULE, "_run_checked", fake_render)
    monkeypatch.setattr(_MODULE, "_paddle_document", failing_document)

    with pytest.raises(RunnerError, match="PaddleOCR failed"):
        load_pages(_task("h1"), pdf, tmp_path / "split" / "jobs" / "h1" / "source")
    assert all("tesseract" not in call for call in calls)


def test_paddle_production_config_matches_frozen_benchmark() -> None:
    resolved = _MODULE._resolved_ocr_config()
    engine_config = {
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_dir": resolved["detector_path"],
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_dir": resolved["recognizer_path"],
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "cpu",
        "enable_mkldnn": False,
        "cpu_threads": 2,
        "text_recognition_batch_size": 6,
    }
    assert _MODULE._paddle_engine_config(resolved) == engine_config
    assert resolved["engine_init_config_sha256"] == hashlib.sha256(
        _MODULE._canonical_bytes(engine_config)
    ).hexdigest()
    assert (
        resolved["detector_effective_limit_side_len"],
        resolved["detector_effective_limit_type"],
        resolved["detector_effective_max_side_limit"],
    ) == (64, "min", 4000)
    assert resolved["ocr_backend"] == "paddlex-ppocrv5-onnxruntime-cpu"
    assert resolved["onnxruntime_version"] == "1.23.2"
    assert resolved["onnxruntime_provider"] == "CPUExecutionProvider"
    assert resolved["onnxruntime_intra_op_num_threads"] == 4
    assert resolved["onnxruntime_inter_op_num_threads"] == 1
    assert resolved["onnxruntime_execution_mode"] == "ORT_SEQUENTIAL"
    assert resolved["onnxruntime_graph_optimization_level"] == "ORT_ENABLE_ALL"
    assert resolved["onnxruntime_module_path"].endswith("onnxruntime/__init__.py")
    assert resolved["ocr_processes"] == 1
    assert resolved["successful_page_jpegs"] == "exact_regenerable_cache_v1"
    assert resolved["image_cache_helper_sha256"] == hashlib.sha256(
        Path(resolved["image_cache_helper_path"]).read_bytes()
    ).hexdigest()
    assert resolved["detector_onnx_sha256"] == (
        "d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb"
    )
    assert resolved["recognizer_onnx_sha256"] == (
        "4212d483f00f1c8617ba143ba36731e361d8307f49b5fae830d828f64b2162a2"
    )
    worker_config = _MODULE._paddle_worker_config(resolved)
    assert worker_config["paddle"] == engine_config
    assert worker_config["onnxruntime"]["providers"] == ["CPUExecutionProvider"]


def test_paddle_pool_uses_one_spawn_worker_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakePool:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def shutdown(self, **kwargs):
            calls["shutdown"] = kwargs

    config = {
        **_MODULE.OCR_CONFIG,
        "python_path": str(Path(sys.executable).absolute()),
        "detector_path": "/models/detector",
        "recognizer_path": "/models/recognizer",
    }
    monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(_MODULE, "_OCR_POOL", None)
    monkeypatch.setattr(_MODULE, "_verify_locked_runtime", lambda _config: None)
    monkeypatch.setattr(_MODULE, "_paddle_worker_config", lambda _config: {"worker": True})

    _MODULE._start_ocr_pool(config)
    _MODULE._shutdown_ocr_pool()

    assert calls["max_workers"] == 1
    assert calls["mp_context"] is _MODULE.multiprocessing.get_context("spawn")
    assert calls["initializer"] is _MODULE.initialize_worker
    assert calls["initargs"] == ({"worker": True},)
    assert calls["shutdown"] == {"wait": False, "cancel_futures": True}


def test_worker_replaces_only_paddlex_infer_with_pinned_ort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rapid_site = tmp_path / "rapid-site"
    ort_package = rapid_site / "onnxruntime"
    ort_package.mkdir(parents=True)
    ort_init = ort_package / "__init__.py"
    ort_init.write_text("# fake", encoding="utf-8")
    det_model = tmp_path / "det.onnx"
    rec_model = tmp_path / "rec.onnx"
    det_model.write_bytes(b"det")
    rec_model.write_bytes(b"rec")
    sessions = []

    class Options:
        pass

    class Session:
        def __init__(self, model, *, sess_options, providers):
            self.model = model
            self.options = sess_options
            self.providers = providers
            sessions.append(self)

        def get_inputs(self):
            return [types.SimpleNamespace(name="x")]

        def get_providers(self):
            return self.providers

        def run(self, _outputs, feed):
            return [feed["x"]]

    fake_ort = types.SimpleNamespace(
        __file__=str(ort_init),
        __version__="1.23.2",
        SessionOptions=Options,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=Session,
    )
    pipeline = types.SimpleNamespace(
        text_det_model=types.SimpleNamespace(infer="native-det"),
        text_rec_model=types.SimpleNamespace(infer="native-rec"),
    )
    engine = types.SimpleNamespace(
        paddlex_pipeline=types.SimpleNamespace(_pipeline=pipeline)
    )
    fake_paddle = types.SimpleNamespace(PaddleOCR=lambda **_kwargs: engine)

    def import_module(name):
        return {"paddleocr": fake_paddle, "onnxruntime": fake_ort}[name]

    monkeypatch.setattr(_WORKER.importlib, "import_module", import_module)
    monkeypatch.setattr(_WORKER, "_ENGINE", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _WORKER.initialize_worker(
        {
            "paddle": {"device": "cpu"},
            "onnxruntime": {
                "site_packages": str(rapid_site),
                "version": "1.23.2",
                "module_path": str(ort_init),
                "module_sha256": hashlib.sha256(ort_init.read_bytes()).hexdigest(),
                "detector_model": str(det_model),
                "detector_sha256": hashlib.sha256(b"det").hexdigest(),
                "recognizer_model": str(rec_model),
                "recognizer_sha256": hashlib.sha256(b"rec").hexdigest(),
                "intra_op_num_threads": 4,
                "inter_op_num_threads": 1,
                "execution_mode": "ORT_SEQUENTIAL",
                "graph_optimization_level": "ORT_ENABLE_ALL",
                "providers": ["CPUExecutionProvider"],
            },
        }
    )

    assert isinstance(pipeline.text_det_model.infer, _WORKER.OrtInfer)
    assert isinstance(pipeline.text_rec_model.infer, _WORKER.OrtInfer)
    assert [session.model for session in sessions] == [str(det_model), str(rec_model)]
    assert all(session.options.intra_op_num_threads == 4 for session in sessions)
    assert all(session.providers == ["CPUExecutionProvider"] for session in sessions)
    pipeline.text_det_model.infer(x=[_WORKER.np.zeros((1, 1), dtype="float32")])


def test_paddle_document_timeout_aborts_shared_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class TimedOutFuture:
        def result(self, *, timeout):
            calls["timeout"] = timeout
            raise concurrent.futures.TimeoutError

    class FakePool:
        def submit(self, *_args):
            return TimedOutFuture()

    pool = FakePool()
    monkeypatch.setattr(_MODULE, "_OCR_POOL", pool)
    monkeypatch.setattr(
        _MODULE, "_abort_ocr_pool", lambda expected: calls.setdefault("aborted", expected)
    )

    with pytest.raises(RunnerError, match="timed out"):
        _MODULE._paddle_document([Path("one.png"), Path("two.png")])

    assert calls == {"timeout": 420, "aborted": pool}


def test_paddle_document_timeout_starts_after_ocr_capacity_is_acquired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_submitted = threading.Event()
    submit_count = 0

    class FakeFuture:
        def __init__(self, number: int) -> None:
            self.number = number

        def result(self, *, timeout):
            assert timeout == 300
            if self.number == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            return [{"document": self.number}]

    class FakePool:
        def submit(self, *_args):
            nonlocal submit_count
            submit_count += 1
            if submit_count == 2:
                second_submitted.set()
            return FakeFuture(submit_count)

    monkeypatch.setattr(_MODULE, "_OCR_POOL", FakePool())
    monkeypatch.setattr(_MODULE, "_OCR_CAPACITY_SEMAPHORE", threading.BoundedSemaphore(1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_MODULE._paddle_document, [Path("first.png")])
        assert first_started.wait(timeout=1)
        second = executor.submit(_MODULE._paddle_document, [Path("second.png")])
        assert not second_submitted.wait(timeout=0.1)
        release_first.set()
        assert first.result(timeout=2) == [{"document": 1}]
        assert second.result(timeout=2) == [{"document": 2}]

    assert submit_count == 2


def test_stale_ocr_failure_does_not_terminate_replacement_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_pool = object()
    replacement_pool = object()
    terminated: list[object] = []
    monkeypatch.setattr(_MODULE, "_OCR_POOL", replacement_pool)
    monkeypatch.setattr(
        _MODULE,
        "_terminate_ocr_pool",
        lambda pool, *, grace_seconds: terminated.append(pool),
    )

    _MODULE._abort_ocr_pool(stale_pool)

    assert _MODULE._OCR_POOL is replacement_pool
    assert terminated == []


def test_bounded_pool_teardown_kills_real_hung_spawn_worker() -> None:
    pool = _MODULE.ProcessPoolExecutor(
        max_workers=1, mp_context=_MODULE.multiprocessing.get_context("spawn")
    )
    future = pool.submit(time.sleep, 60)
    deadline = time.monotonic() + 5
    while not getattr(pool, "_processes", {}) and time.monotonic() < deadline:
        time.sleep(0.01)
    processes = list(pool._processes.values())

    started = time.monotonic()
    _MODULE._terminate_ocr_pool(pool, grace_seconds=0.2)

    assert time.monotonic() - started < 3
    assert processes and all(not process.is_alive() for process in processes)
    assert future.cancelled() or future.done()


def test_paddle_document_broken_pool_is_aborted(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _MODULE.ProcessPoolExecutor(
        max_workers=1, mp_context=_MODULE.multiprocessing.get_context("spawn")
    )
    crashed = pool.submit(os._exit, 17)
    with pytest.raises(concurrent.futures.process.BrokenProcessPool):
        crashed.result(timeout=5)
    calls: list[object] = []
    monkeypatch.setattr(_MODULE, "_OCR_POOL", pool)
    monkeypatch.setattr(_MODULE, "_abort_ocr_pool", calls.append)

    with pytest.raises(RunnerError, match="worker process failed"):
        _MODULE._paddle_document([Path("page.png")])

    assert calls == [pool]


def test_pipeline_accepts_eight_workers(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=tmp_path / "private",
        limit=1,
        workers=8,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )

    assert result["succeeded"] == 1


def test_locked_runtime_rejects_mutated_distribution_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "REPOSITORY_ROOT", tmp_path)
    site_packages = (
        tmp_path
        / "data"
        / "issue24"
        / "ppocr-env"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    member = site_packages / "demo.py"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"trusted runtime code")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(member.read_bytes()).digest())
        .decode()
        .rstrip("=")
    )
    record = site_packages / "demo-1.0.dist-info" / "RECORD"
    record.parent.mkdir()
    record.write_text(f"demo.py,sha256={digest},{member.stat().st_size}\n", encoding="utf-8")
    lock = tmp_path / "requirements" / "solution-paddle-ocr.txt"
    lock.parent.mkdir()
    lock.write_text("demo==1.0\n", encoding="utf-8")
    artifacts = {}
    for role in (
        "python",
        "renderer",
        "worker",
        "image_cache_helper",
        "detector_onnx",
        "recognizer_onnx",
        "adapter_reference",
        "conversion_setup",
        "conversion_requirements",
    ):
        artifact = tmp_path / f"{role}.bin"
        artifact.write_bytes(role.encode())
        artifacts[f"{role}_path"] = str(artifact)
        artifacts[f"{role}_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    for role in ("detector", "recognizer"):
        model = tmp_path / role
        model.mkdir()
        (model / "model.bin").write_bytes(role.encode())
        artifacts[f"{role}_path"] = str(model)
        artifacts[f"{role}_tree_sha256"] = _MODULE._directory_content_hash(model)
    external_site = tmp_path / "external-env" / "lib" / "python3.11" / "site-packages"
    external_site.mkdir(parents=True)
    external_record = external_site / "external-1.0.dist-info" / "RECORD"
    external_record.parent.mkdir()
    external_record.write_text("", encoding="utf-8")
    external_record_sha256 = hashlib.sha256(external_record.read_bytes()).hexdigest()
    external_module = external_site / "onnxruntime" / "__init__.py"
    external_module.parent.mkdir()
    external_module.write_text("# test", encoding="utf-8")
    equivalence = tmp_path / "equivalence.json"
    equivalence.write_text("{}", encoding="utf-8")
    config = {
        **artifacts,
        "environment_root": str(site_packages.parents[2]),
        "renderer_version": "pdftoppm test version",
        "runtime_lock_path": str(lock),
        "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "runtime_distributions": [
            {
                "name": "demo",
                "record_path": str(record),
                "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
            }
        ],
        "onnxruntime_site_packages": str(external_site),
        "onnxruntime_record_path": str(external_record),
        "onnxruntime_record_sha256": external_record_sha256,
        "onnxruntime_module_path": str(external_module),
        "onnxruntime_module_sha256": hashlib.sha256(external_module.read_bytes()).hexdigest(),
        "equivalence_artifacts": [
            {
                "path": str(equivalence),
                "sha256": hashlib.sha256(equivalence.read_bytes()).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="", stderr="pdftoppm test version\n"
        ),
    )

    _MODULE._verify_locked_runtime(config)
    member.write_bytes(b"mutated runtime code")
    with pytest.raises(RunnerError, match="runtime member changed"):
        _MODULE._verify_locked_runtime(config)


def test_locked_runtime_rejects_changed_renderer_binary() -> None:
    config = _MODULE._resolved_ocr_config()
    config["renderer_sha256"] = "0" * 64
    with pytest.raises(RunnerError, match="renderer artifact changed"):
        _MODULE._verify_locked_runtime(config)


def test_next_split_rejects_prior_generation_under_different_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    heldout_tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", heldout_tasks, 1)
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=heldout_tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_pages,
        command_runner=_successful_codex([]),
    )
    train_tasks = _write_tasks(tmp_path / "train.jsonl", [_task("t1")])
    _pin("train", train_tasks, 1)
    (pdf_root / "t1.pdf").write_bytes(b"train")
    monkeypatch.setattr(
        _MODULE,
        "PROMPT_INSTRUCTIONS",
        _MODULE.PROMPT_INSTRUCTIONS + " changed",
    )

    with pytest.raises(RunnerError, match="heldout generation config differs"):
        run_pipeline(
            split="train",
            tasks_path=train_tasks,
            pdf_root=pdf_root,
            output_root=output,
            page_loader=_pages,
            command_runner=_successful_codex([]),
        )
