import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_solution_records.py"
_SPEC = importlib.util.spec_from_file_location("run_solution_records", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
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
    image = cache / "page-0001.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    return PageBundle(
        pages=[{"page_number": 1, "text": "E-7: The document states 3 plus 4."}],
        images=[image],
        ocr_provenance=_MODULE._generation_config()["ocr"],
    )


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
                    "evidence": ["E-7"],
                    "evidence_details": [
                        {"id": "E-7", "page": 1, "quote": "The document states 3 plus 4."}
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


def _visual_pages(_task: dict[str, object], _pdf: Path, cache: Path) -> PageBundle:
    image = cache / "page-1.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    _MODULE.Image.new("RGB", (200, 100), "white").save(image, format="JPEG", quality=65)
    text = "E-7: The document states 3 plus 4."
    return PageBundle(
        pages=[{"page_number": 1, "text": text}],
        images=[image],
        ocr_provenance=_MODULE._generation_config()["ocr"],
        geometry=[
            {
                "page_number": 1,
                "width": 200,
                "height": 100,
                "lines": [
                    {
                        "line_index": 0,
                        "text": text,
                        "bbox": {"left": 10, "top": 10, "width": 100, "height": 20},
                    }
                ],
            }
        ],
    )


def _visual_candidate_codex(calls: list[dict[str, object]], visible_id: str):
    def run(argv, **kwargs):
        response_path = Path(argv[argv.index("--output-last-message") + 1])
        is_checker = "visual-id-checks" in str(response_path)
        if is_checker:
            response = {
                "visible_headings": [
                    {
                        "id": visible_id,
                        "bbox": {"left": 10, "top": 10, "right": 60, "bottom": 30},
                        "legibility": "clear",
                    }
                ]
            }
        else:
            response = {
                "answer": "7",
                "solution": {
                    "summary": "The cited block states the two values to add.",
                    "calculations": [{"expression": "3 + 4", "result": "7"}],
                },
                "evidence": ["E-7!"],
                "evidence_details": [
                    {
                        "id": "E-7!",
                        "page": 1,
                        "quote": "The document states 3 plus 4.",
                    }
                ],
                "uncertainties": ["OCR omitted punctuation from the evidence ID."],
            }
        response_path.write_text(json.dumps(response), encoding="utf-8")
        calls.append(
            {
                "argv": list(argv),
                "prompt": kwargs["input"],
                "checker": is_checker,
                "primary_frozen": (
                    response_path.parents[3] / "primary-response.json"
                ).is_file()
                if is_checker
                else None,
            }
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='{"type":"turn.completed","usage":{"input_tokens":10}}\n',
            stderr="",
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
    assert argv[argv.index("--model") + 1] == "gpt-5.6-sol"
    assert 'features.shell_tool=false' in argv
    assert argv.count("--image") == 1
    job = tmp_path / "private" / "heldout" / "jobs" / "h1"
    assert (job.stat().st_mode & 0o077) == 0
    provenance = json.loads((job / "record.json").read_text(encoding="utf-8"))["provenance"]
    assert provenance["model"] == "gpt-5.6-sol"
    assert provenance["pdf_sha256"]
    assert provenance["input_sha256"]
    assert provenance["output_sha256"]
    assert provenance["page_images"][0]["sha256"]
    assert provenance["usage"]["input_tokens"] == 10


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


def test_unique_missing_ocr_id_uses_one_blind_visual_check_after_primary_freeze(
    tmp_path: Path,
) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_visual_pages,
        command_runner=_visual_candidate_codex(calls, "E-7!"),
    )

    assert result["succeeded"] == 1
    assert len(calls) == 2
    checker = next(call for call in calls if call["checker"])
    assert checker["primary_frozen"] is True
    assert "What is the total?" not in str(checker["prompt"])
    assert "E-7" not in str(checker["prompt"])
    assert "3 plus 4" not in str(checker["prompt"])
    record = json.loads(
        (output / "heldout" / "jobs" / "h1" / "record.json").read_text(encoding="utf-8")
    )
    proof = record["provenance"]["visual_id_checks"][0]
    assert proof["visible_id"] == "E-7!"
    assert proof["ocr_id"] == "E-7"


def test_visual_id_failure_is_preserved_without_solver_or_checker_retry(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"

    result = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_visual_pages,
        command_runner=_visual_candidate_codex(calls, "DIFFERENT"),
    )

    assert result["succeeded"] == 0
    assert result["error_counts"] == {"needs_visual_review": 1}
    assert len(calls) == 2
    job = output / "heldout" / "jobs" / "h1"
    assert (job / "attempts" / "1" / "primary-response.json").is_file()
    assert list((job / "attempts" / "1" / "visual-id-checks").rglob("response.json"))
    assert not (job / "attempts" / "2").exists()
    resumed = run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        resume=True,
        page_loader=_visual_pages,
        command_runner=_visual_candidate_codex(calls, "E-7!"),
    )
    assert resumed["succeeded"] == 0
    assert len(calls) == 2


def test_resume_rejects_tampered_visual_proof_artifact(tmp_path: Path) -> None:
    tasks, pdf_root = _public_inputs(tmp_path, [_task("h1")])
    _pin("heldout", tasks, 1)
    calls: list[dict[str, object]] = []
    output = tmp_path / "private"
    run_pipeline(
        split="heldout",
        tasks_path=tasks,
        pdf_root=pdf_root,
        output_root=output,
        page_loader=_visual_pages,
        command_runner=_visual_candidate_codex(calls, "E-7!"),
    )
    crop = next(
        (output / "heldout" / "jobs" / "h1" / "attempts" / "1").rglob(
            "heading-crop.png"
        )
    )
    crop.write_bytes(b"tampered")

    with pytest.raises(RunnerError, match="invalid or interrupted job artifacts"):
        run_pipeline(
            split="heldout",
            tasks_path=tasks,
            pdf_root=pdf_root,
            output_root=output,
            resume=True,
            page_loader=_visual_pages,
            command_runner=_visual_candidate_codex(calls, "E-7!"),
        )
    assert len(calls) == 2


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


def test_rapidocr_loader_uses_lossless_ocr_and_retained_jpeg_in_numeric_order(
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

    class FakeImage:
        shape = (100, 200, 3)

    class FakeResult:
        boxes = [[[10.2, 20.4], [80.7, 20.1], [80.2, 31.9], [10.0, 32.0]]]
        txts: tuple[str, ...]
        scores = (0.99,)
        img = FakeImage()

    class FakeEngine:
        def __call__(self, image, *, use_cls):
            assert use_cls is False
            image_path = Path(image)
            ocr_calls.append(image_path)
            result = FakeResult()
            result.txts = (image_path.stem.replace("ocr-page", "page"),)
            return result

    monkeypatch.setattr(_MODULE, "_run_checked", fake_render)
    monkeypatch.setattr(_MODULE, "_rapidocr_engine", lambda: FakeEngine())
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
    assert [page["text"] for page in bundle.pages] == ["page-2", "page-10"]
    assert [path.stem for path in ocr_calls] == ["ocr-page-2", "ocr-page-10"]
    assert not any(path.exists() for path in ocr_calls)
    assert bundle.geometry[0]["lines"][0]["bbox"] == {
        "left": 10,
        "top": 20,
        "width": 71,
        "height": 12,
    }
    assert bundle.geometry[0]["ocr_image_sha256"]
    assert all("tesseract" not in call for call in render_calls)
    assert bundle.ocr_provenance["detector_sha256"]
    assert bundle.ocr_provenance["recognizer_sha256"]


def test_rapidocr_failure_has_no_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "document.pdf"
    pdf.write_bytes(b"pdf")
    calls: list[list[str]] = []

    def fake_render(argv, **_kwargs):
        calls.append(list(argv))
        _MODULE.Image.new("RGB", (200, 100), "white").save(f"{argv[-1]}-1.png")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    class FailingEngine:
        def __call__(self, _image, *, use_cls):
            assert use_cls is False
            raise RuntimeError("RapidOCR failed")

    monkeypatch.setattr(_MODULE, "_run_checked", fake_render)
    monkeypatch.setattr(_MODULE, "_rapidocr_engine", lambda: FailingEngine())

    with pytest.raises(RunnerError, match="RapidOCR failed"):
        load_pages(_task("h1"), pdf, tmp_path / "split" / "jobs" / "h1" / "source")
    assert all("tesseract" not in call for call in calls)


def test_locked_runtime_rejects_mutated_distribution_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "REPOSITORY_ROOT", tmp_path)
    site_packages = (
        tmp_path
        / "data"
        / "issue24"
        / "rapidocr-env"
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
    lock = tmp_path / "requirements" / "solution-ocr.txt"
    lock.parent.mkdir()
    lock.write_text("demo==1.0\n", encoding="utf-8")
    artifacts = {}
    for role in ("python", "renderer", "detector", "recognizer"):
        artifact = tmp_path / f"{role}.bin"
        artifact.write_bytes(role.encode())
        artifacts[f"{role}_path"] = str(artifact)
        artifacts[f"{role}_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config = {
        **artifacts,
        "renderer_version": "pdftoppm test version",
        "runtime_lock_path": str(lock),
        "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "runtime_distributions": [
            {
                "name": "demo",
                "record_path": str(record),
                "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
            }
        ]
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
