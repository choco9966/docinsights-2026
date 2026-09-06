#!/usr/bin/env python3
"""Generate private, auditable solution records from one public split."""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from PIL import Image
from PIL import __version__ as PILLOW_VERSION

from docinsights_analysis.solution_records import (
    SolutionRecordError,
    ensure_private_directory,
    export_records,
    finalize_solution,
    input_digest,
    primary_record_digest,
    validate_primary_solution,
    validate_solution,
    write_private_atomic,
)
from docinsights_analysis.visual_id_checks import (
    SOURCE_CHECK_SCHEMA,
    SourceRegionCheckError,
    check_source_regions,
)

_RUNNER_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _RUNNER_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _RUNNER_SCRIPTS_DIR)
from solution_image_cache import (  # noqa: E402
    ImageCacheError,
    evict_verified_success_page_jpegs,
    validate_page_jpeg_cache,
)
from solution_paddle_worker import initialize_worker, recognize_document  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ORDER = ("heldout", "train", "validation")
OFFICIAL_SPLITS = {
    "heldout": {
        "count": 1730,
        "tasks_sha256": "5fe8fbb8169b0c2b396fe155d263db36f4fa34b02a0cedd9075423b0bd3fc40d",
    },
    "train": {
        "count": 908,
        "tasks_sha256": "6d9cd9087d0c5e30bfc17c83aec30752403d4109fb93d8357f534da425969489",
    },
    "validation": {
        "count": 217,
        "tasks_sha256": "5b6f57a30f4dc8b27873162ca58434c0411fb89f4726b5f2988903344b43443a",
    },
}
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "in_app_browser",
    "code_mode_host",
    "hooks",
    "skill_mcp_dependency_install",
    "multi_agent",
    "image_generation",
    "memories",
    "goals",
)
COMMAND_CONFIG = {
    "codex_executable": "data/issue24/codex-0.153.4/node_modules/.bin/codex",
    "codex_cli_version": "0.153.4",
    "codex_wrapper_sha256": "61b0194f3bb6534439c8d26a3ed57d0805f84b884588b761795323eeb92fcf70",
    "codex_native_sha256": "b973d440acac501fd2594a43e7ca9ce41e0a65b9dfb28d0d7a7837c99e1261e3",
    "model": "gpt-6-astra",
    "model_reasoning_effort": "high",
    "model_identity_evidence": "requested-model-plus-completed-turn-without-fallback-warning",
    "backend_model_event_field": None,
    "sandbox": "read-only",
    "web_search": "disabled",
    "disabled_features": list(DISABLED_FEATURES),
    "timeout_seconds": 300,
    "maximum_attempts": 2,
}
OCR_CONFIG = {
    "renderer": "pdftoppm",
    "ocr_input_format": "png",
    "retained_image_format": "jpeg",
    "retained_image_extension": "jpg",
    "retained_jpeg_quality": 65,
    "successful_page_jpegs": "exact_regenerable_cache_v1",
    "page_jpeg_recipe": "pdftoppm-png-pillow-rgb-jpeg-v1",
    "pillow_version": PILLOW_VERSION,
    "dpi": 175,
    "minimum_free_bytes": 1_073_741_824,
    "ocr_backend": "paddlex-ppocrv5-onnxruntime-cpu",
    "text_detection_model_name": "PP-OCRv5_mobile_det",
    "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "device": "cpu",
    "enable_mkldnn": False,
    "cpu_threads": 2,
    "text_recognition_batch_size": 6,
    "onnxruntime_version": "1.23.2",
    "onnxruntime_provider": "CPUExecutionProvider",
    "onnxruntime_intra_op_num_threads": 4,
    "onnxruntime_inter_op_num_threads": 1,
    "onnxruntime_execution_mode": "ORT_SEQUENTIAL",
    "onnxruntime_graph_optimization_level": "ORT_ENABLE_ALL",
    "detector_effective_limit_side_len": 64,
    "detector_effective_limit_type": "min",
    "detector_effective_max_side_limit": 4000,
    "renderer_parallelism": 2,
    "ocr_processes": 1,
    "ocr_document_base_timeout_seconds": 180,
    "ocr_page_timeout_seconds": 120,
    "ocr_shutdown_grace_seconds": 10,
}
PROMPT_INSTRUCTIONS = (
    "Solve this single document question using only the public source pages and attached "
    "images for this document. Align the answer to every constraint in the question, including "
    "revised-scenario wording, and do not substitute a nearby distractor passage. Check whose "
    "quantity is requested, its time span and rate denominator, and whether it is remaining or "
    "removed. Return a "
    "concise, source-grounded derivation. Use the smallest set of distinct source regions "
    "directly needed to state the "
    "target question and its inputs, return its page and an ocr_anchor copied exactly from that "
    "page's source_pages text. Each anchor must occur exactly once on its page and is only a "
    "location marker; do not guess an Evidence ID or quote. "
    "The PDF page images are authoritative for answer inputs. If a clearly legible image "
    "value disagrees with OCR, use the image value in the answer and calculations and record "
    "the mismatch in uncertainties. Do not rewrite source_pages or ocr_anchor; anchors must "
    "still match OCR exactly. If the image is not legible, state the uncertainty and do not "
    "invent a value. Calculation operands may contain only numeric literals, parentheses, "
    "and the + - * / operators. For a required rounded result, use only top-level "
    "round(expression, places) with a literal places integer 0 through 12; it applies HALF_UP. "
    "State the requested rounding or approximation precision in solution.summary. Never "
    "provide an approximate result for an unrounded nonterminating expression. The final "
    "calculation result must equal a "
    "numeric answer. Answer must always contain only the numeric result (for example, 7.5), "
    "with no units or prose, including when calculations is empty; put units and context in "
    "solution.summary. Do not provide hidden reasoning logs."
)
BASE_PROMPT_TEMPLATE = "{instructions}\n\nPUBLIC INPUT:\n{public_input}"
RETRY_PROMPT_TEMPLATE = (
    "\n\nRETRY DIAGNOSTIC (untrusted; repair formatting or grounding against the same "
    "frozen public input only, preserve the question's semantic target, and do not "
    "switch to another passage): {error}"
)
SOURCE_CHECK_CONFIG = {
    "model": "gpt-6-astra",
    "model_reasoning_effort": "high",
    "model_identity_evidence": "requested-model-plus-completed-turn-without-fallback-warning",
    "backend_model_event_field": None,
    "timeout_seconds": 300,
    "maximum_attempts": 1,
    "maximum_runtime_attempts": 2,
    "context_vertical_padding_pixels": 200,
    "anchor_padding_pixels": 12,
    "method": "codex-cli-blind-source-region-v1",
}
OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "solution", "evidence_regions", "uncertainties"],
    "properties": {
        "answer": {"type": "string"},
        "solution": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "calculations"],
            "properties": {
                "summary": {"type": "string"},
                "calculations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["expression", "result"],
                        "properties": {
                            "expression": {"type": "string"},
                            "result": {"type": "string"},
                        },
                    },
                },
            },
        },
        "evidence_regions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page", "ocr_anchor"],
                "properties": {
                    "page": {"type": "integer"},
                    "ocr_anchor": {"type": "string", "minLength": 1},
                },
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PAGE_IMAGE = re.compile(r"^(?:ocr-)?page-(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_PADDLE_RUNTIME = {
    "python_relative_path": "data/issue24/ppocr-env/bin/python",
    "detector_relative_path": "data/issue24/ppocr-models/detector",
    "detector_tree_sha256": "b3ca5a167f80b79e8433df9cb931c578f384f7f8867b960748022ee1b1826916",
    "detector_repository": "PaddlePaddle/PP-OCRv5_mobile_det",
    "detector_revision": "0d63e78e2b680928f6b1747d76a08db6e645efb7",
    "recognizer_relative_path": "data/issue24/ppocr-models/recognizer",
    "recognizer_tree_sha256": "2a3324a89b92f446da343999c922d0fa7a0fa3b0e0a3bbb7034d654c588f1e16",
    "recognizer_repository": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
    "recognizer_revision": "267c36e24c331595590fe7bd72bde2436fd286f2",
    "detector_source_files": {
        "inference.json": "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c",
        "inference.pdiparams": "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
    },
    "recognizer_source_files": {
        "inference.json": "fd1b6ec722ea841a72d3ba43e527df1d1066d5d7808e0503ee3eec7265188753",
        "inference.pdiparams": "3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b",
    },
    "detector_onnx_relative_path": "data/issue24/ppocr-onnx-models/detector/inference.onnx",
    "detector_onnx_sha256": "d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb",
    "recognizer_onnx_relative_path": "data/issue24/ppocr-onnx-models/recognizer/inference.onnx",
    "recognizer_onnx_sha256": "4212d483f00f1c8617ba143ba36731e361d8307f49b5fae830d828f64b2162a2",
    "onnxruntime_site_packages_relative_path": (
        "data/issue24/rapidocr-env/lib/python3.11/site-packages"
    ),
    "onnxruntime_record_relative_path": (
        "data/issue24/rapidocr-env/lib/python3.11/site-packages/"
        "onnxruntime-1.23.2.dist-info/RECORD"
    ),
    "onnxruntime_record_sha256": (
        "e35600cfddbd37c7e0e85394a631480dd6f742c475daf5f6db8f8388637dc968"
    ),
    "onnxruntime_module_sha256": (
        "1ca44e862e236031dde5dfb8a4d217e3770fa43201dadcf0b947fd43ed16d20e"
    ),
    "adapter_reference_relative_path": (
        "artifacts/solution-records/issue24/ocr-benchmark-v2/acceleration-probe/"
        "probe_paddlex_ort.py"
    ),
    "adapter_reference_sha256": (
        "3f6d644acd02040bf107413569522c134e0f33b013a2ea21cd2d17a9b8de647d"
    ),
    "conversion_setup_relative_path": "scripts/setup_solution_paddle_onnx.py",
    "conversion_setup_sha256": (
        "78fa67ed47971e4d6507fd252b76003d98214bef043d5ba53ee5b3312707f6ff"
    ),
    "conversion_requirements_relative_path": "requirements/solution-paddle-onnx-export.txt",
    "conversion_requirements_sha256": (
        "37e194e4021d421c325577c0768d6c04177888032e9c5728979cba171bc10c42"
    ),
    "equivalence_shards": {
        "shard-0-of-2.json": "121b49c2c5ac2d2992a802d4e25eba4c607af8318e0cbe75449c79d05a44f6a3",
        "shard-1-of-2.json": "ae4c5d2e53e76e6924cdb088fe2e49e2b07e1f15d426377878e8c3f6bf029db4",
        "cpu-thread-and-pool-tuning.json": (
            "99d94144bffd8e2123443f14ff29317253a4716116e138375e9da859cc9981af"
        ),
    },
    "distributions": {
        "paddlepaddle": "3.2.0",
        "paddleocr": "3.3.2",
        "paddlex": "3.3.13",
    },
}
_RENDER_SEMAPHORE = threading.BoundedSemaphore(OCR_CONFIG["renderer_parallelism"])
_OCR_CAPACITY_SEMAPHORE = threading.BoundedSemaphore(OCR_CONFIG["ocr_processes"])
_OCR_POOL_LOCK = threading.Lock()
_OCR_POOL: ProcessPoolExecutor | None = None
_OCR_POOL_BASE_TIMEOUT_SECONDS = float(OCR_CONFIG["ocr_document_base_timeout_seconds"])
_OCR_POOL_PAGE_TIMEOUT_SECONDS = float(OCR_CONFIG["ocr_page_timeout_seconds"])
_OCR_POOL_SHUTDOWN_GRACE_SECONDS = float(OCR_CONFIG["ocr_shutdown_grace_seconds"])


class RunnerError(ValueError):
    """The split cannot be run without violating its audit contract."""


class AttemptFailure(RunnerError):
    """One Codex attempt failed with a stable public classification."""

    def __init__(self, message: str, error_kind: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


@dataclass(frozen=True)
class PageBundle:
    pages: list[dict[str, Any]]
    images: list[Path]
    ocr_provenance: dict[str, Any] = field(default_factory=dict)
    geometry: list[dict[str, Any]] = field(default_factory=list)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
PageLoader = Callable[[dict[str, Any], Path, Path], PageBundle]
SourceChecker = Callable[..., list[dict[str, Any]]]


def validate_cli_output_root(output_root: Path) -> Path:
    """Confine production CLI output to the repository's gitignored private tree."""
    repository = REPOSITORY_ROOT.resolve()
    private_root = (repository / "artifacts" / "solution-records" / "issue24").resolve()
    resolved = output_root.resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as error:
        raise RunnerError(
            "--output-root must be inside the private issue24 artifact tree"
        ) from error
    gitignore = repository / ".gitignore"
    try:
        ignored_patterns = {
            line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()
        }
    except OSError as error:
        raise RunnerError(f"cannot verify private output ignore rule: {error}") from error
    if "artifacts/solution-records/" not in ignored_patterns:
        raise RunnerError("private issue24 artifact tree is not gitignored")
    return resolved


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_dir(path: Path) -> None:
    try:
        ensure_private_directory(path)
    except SolutionRecordError as error:
        raise RunnerError(str(error)) from error


def _write_private(path: Path, content: str | bytes) -> None:
    _private_dir(path.parent)
    try:
        write_private_atomic(path, content)
    except SolutionRecordError as error:
        raise RunnerError(str(error)) from error


def _write_json(path: Path, value: Any) -> None:
    _write_private(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RunnerError(f"cannot read tasks: {error}") from error
    try:
        if path.suffix == ".json":
            loaded = json.loads(text)
            rows = loaded if isinstance(loaded, list) else loaded.get("tasks")
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (AttributeError, json.JSONDecodeError) as error:
        raise RunnerError(f"invalid tasks file: {error}") from error
    if not isinstance(rows, list) or not rows:
        raise RunnerError("tasks must contain at least one public task")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, task in enumerate(rows):
        if not isinstance(task, dict):
            raise RunnerError(f"task {index} must be an object")
        for field_name in ("instance_id", "user_query", "document_pdf"):
            if field_name not in task:
                raise RunnerError(f"task {index} missing {field_name}")
            if not isinstance(task[field_name], str) or not task[field_name].strip():
                raise RunnerError(f"task {index} has invalid {field_name}")
        instance_id = task["instance_id"]
        if not _SAFE_ID.fullmatch(instance_id) or instance_id in {".", ".."}:
            raise RunnerError(f"unsafe instance_id: {instance_id!r}")
        if instance_id in seen:
            raise RunnerError(f"duplicate instance_id: {instance_id}")
        seen.add(instance_id)
        tasks.append(task)
    return tasks


def _official_tasks_sha256(split: str, tasks_path: Path, tasks: list[dict[str, Any]]) -> str:
    pin = OFFICIAL_SPLITS[split]
    actual_sha256 = _sha256_file(tasks_path)
    if len(tasks) != pin["count"] or actual_sha256 != pin["tasks_sha256"]:
        raise RunnerError(
            f"official {split} task manifest must contain {pin['count']} tasks and have "
            f"SHA-256 {pin['tasks_sha256']}"
        )
    return actual_sha256


def _resolve_pdf(pdf_root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute():
        raise RunnerError("document_pdf must be relative to --pdf-root")
    root = pdf_root.resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise RunnerError(f"document_pdf is missing or outside --pdf-root: {relative_name}")
    return candidate


def _build_input_manifest(tasks: list[dict[str, Any]], pdf_root: Path) -> list[dict[str, Any]]:
    entries = []
    for task in tasks:
        pdf_path = _resolve_pdf(pdf_root, task["document_pdf"])
        public_task = {
            key: task[key]
            for key in ("instance_id", "user_query", "document_pdf", "source_pages")
            if key in task
        }
        entries.append(
            {
                "task": public_task,
                "pdf_path": str(pdf_path),
                "pdf_sha256": _sha256_file(pdf_path),
            }
        )
    return entries


def _run_checked(argv: Sequence[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise RunnerError(f"command timed out after {timeout}s: {argv[0]}") from error


def _run_isolated(
    argv: Sequence[str],
    *,
    input: str | None = None,
    capture_output: bool = False,
    text: bool = False,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE if input is not None else None,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=text,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        assert timeout is not None
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=list(argv), timeout=timeout, output=stdout, stderr=stderr
        ) from None
    completed = subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)
    if check and completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _directory_content_hash(path: Path) -> str:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and not any(part.startswith(".") for part in file.relative_to(path).parts)
    )
    if not files:
        raise RunnerError(f"pinned model directory is empty: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_distribution_records(
    environment_root: Path, lock_path: Path
) -> list[dict[str, str]]:
    site_package_roots = list((environment_root / "lib").glob("python*/site-packages"))
    if len(site_package_roots) != 1:
        raise RunnerError("pinned PaddleOCR environment has an unexpected layout")
    site_packages = site_package_roots[0]
    locked = []
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, version = stripped.partition("==")
        if not separator:
            raise RunnerError("PaddleOCR lock must contain exact versions")
        locked.append((name, version))
    if not locked:
        raise RunnerError("PaddleOCR lock is empty")
    for name, version in _PADDLE_RUNTIME["distributions"].items():
        if (name, version) not in locked:
            raise RunnerError(f"PaddleOCR lock is missing {name}=={version}")
    installed: dict[tuple[str, str], Path] = {}
    for metadata_path in site_packages.glob("*.dist-info/METADATA"):
        fields: dict[str, str] = {}
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if ": " in line:
                field, value = line.split(": ", 1)
                if field in {"Name", "Version"} and field not in fields:
                    fields[field] = value
            if len(fields) == 2:
                break
        if set(fields) == {"Name", "Version"}:
            normalized = re.sub(r"[-_.]+", "-", fields["Name"]).casefold()
            installed[(normalized, fields["Version"])] = metadata_path
    records = []
    for name, version in locked:
        normalized_name = re.sub(r"[-_.]+", "-", name).casefold()
        metadata_path = installed.get((normalized_name, version))
        if metadata_path is None:
            raise RunnerError(f"pinned OCR distribution is missing: {name}=={version}")
        record_path = metadata_path.parent / "RECORD"
        if not record_path.is_file():
            raise RunnerError(f"pinned OCR RECORD is missing: {name}=={version}")
        records.append(
            {
                "name": name,
                "version": version,
                "record_path": str(record_path.resolve()),
                "record_sha256": _sha256_file(record_path),
            }
        )
    return records


def _resolved_ocr_config() -> dict[str, Any]:
    resolved: dict[str, Any] = {**OCR_CONFIG, "operating_system": platform.platform()}
    environment_root = (REPOSITORY_ROOT / "data" / "issue24" / "ppocr-env").resolve()
    python_path = REPOSITORY_ROOT / str(_PADDLE_RUNTIME["python_relative_path"])
    if not python_path.is_file():
        raise RunnerError("pinned PaddleOCR Python runtime is missing")
    resolved.update(
        {
            "environment_root": str(environment_root),
            "python_path": str(python_path.absolute()),
            "python_sha256": _sha256_file(python_path),
        }
    )
    for role in ("detector", "recognizer"):
        path = REPOSITORY_ROOT / str(_PADDLE_RUNTIME[f"{role}_relative_path"])
        observed = _directory_content_hash(path)
        expected = str(_PADDLE_RUNTIME[f"{role}_tree_sha256"])
        if observed != expected:
            raise RunnerError(f"pinned PaddleOCR {role} model tree changed")
        resolved.update(
            {
                f"{role}_path": str(path.resolve()),
                f"{role}_tree_sha256": observed,
                f"{role}_repository": _PADDLE_RUNTIME[f"{role}_repository"],
                f"{role}_revision": _PADDLE_RUNTIME[f"{role}_revision"],
            }
        )
        for filename, expected_file_sha256 in _PADDLE_RUNTIME[
            f"{role}_source_files"
        ].items():
            source_file = path / filename
            if not source_file.is_file() or _sha256_file(source_file) != expected_file_sha256:
                raise RunnerError(f"pinned PaddleOCR {role} source file changed: {filename}")
        resolved[f"{role}_source_files"] = dict(_PADDLE_RUNTIME[f"{role}_source_files"])
    for role in (
        "detector_onnx",
        "recognizer_onnx",
        "adapter_reference",
        "conversion_setup",
        "conversion_requirements",
    ):
        path = REPOSITORY_ROOT / str(_PADDLE_RUNTIME[f"{role}_relative_path"])
        expected = str(_PADDLE_RUNTIME[f"{role}_sha256"])
        if not path.is_file() or _sha256_file(path) != expected:
            raise RunnerError(f"pinned OCR {role} artifact is missing or changed")
        resolved[f"{role}_path"] = str(path.resolve())
        resolved[f"{role}_sha256"] = expected
    acceleration_root = (
        REPOSITORY_ROOT
        / "artifacts"
        / "solution-records"
        / "issue24"
        / "ocr-benchmark-v2"
        / "acceleration-probe"
    )
    equivalence_artifacts = []
    for name, expected in _PADDLE_RUNTIME["equivalence_shards"].items():
        path = acceleration_root / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise RunnerError(f"pinned OCR equivalence artifact changed: {name}")
        equivalence_artifacts.append({"path": str(path.resolve()), "sha256": expected})
    onnxruntime_site_packages = REPOSITORY_ROOT / str(
        _PADDLE_RUNTIME["onnxruntime_site_packages_relative_path"]
    )
    onnxruntime_record = REPOSITORY_ROOT / str(
        _PADDLE_RUNTIME["onnxruntime_record_relative_path"]
    )
    onnxruntime_module = onnxruntime_site_packages / "onnxruntime" / "__init__.py"
    if (
        not onnxruntime_record.is_file()
        or _sha256_file(onnxruntime_record)
        != _PADDLE_RUNTIME["onnxruntime_record_sha256"]
    ):
        raise RunnerError("pinned OCR onnxruntime RECORD is missing or changed")
    if (
        not onnxruntime_module.is_file()
        or _sha256_file(onnxruntime_module) != _PADDLE_RUNTIME["onnxruntime_module_sha256"]
    ):
        raise RunnerError("pinned OCR onnxruntime module is missing or changed")
    resolved.update(
        {
            "onnxruntime_site_packages": str(onnxruntime_site_packages.resolve()),
            "onnxruntime_record_path": str(onnxruntime_record.resolve()),
            "onnxruntime_record_sha256": _PADDLE_RUNTIME["onnxruntime_record_sha256"],
            "onnxruntime_module_path": str(onnxruntime_module.resolve()),
            "onnxruntime_module_sha256": _PADDLE_RUNTIME["onnxruntime_module_sha256"],
            "converter_version": "2.1.0",
            "conversion_opset_version": 11,
            "conversion_enable_onnx_checker": True,
            "conversion_optimize_tool": "None",
            "equivalence_artifacts": equivalence_artifacts,
            "equivalence_page_count": 60,
            "equivalence_text_boxes_polygons": "exact",
            "equivalence_confidence_max_abs": 0.000017524,
            "ocr_capacity_selection": "pool1-ort-intra4-memory-bounded",
            "ocr_capacity_reference_rss_bytes": 1_805_910_016,
        }
    )
    lock_path = REPOSITORY_ROOT / "requirements" / "solution-paddle-ocr.txt"
    distributions = _installed_distribution_records(environment_root, lock_path)
    renderer_path_value = shutil.which(str(OCR_CONFIG["renderer"]))
    if renderer_path_value is None:
        raise RunnerError("pinned pdftoppm renderer is unavailable")
    renderer_path = Path(renderer_path_value)
    version = subprocess.run(
        [str(renderer_path), "-v"], capture_output=True, text=True, timeout=30, check=False
    )
    version_lines = (version.stderr or version.stdout).splitlines()
    if version.returncode != 0 or not version_lines:
        raise RunnerError("could not identify the pdftoppm renderer version")
    renderer_version = version_lines[0]
    worker_path = REPOSITORY_ROOT / "scripts" / "solution_paddle_worker.py"
    image_cache_helper_path = REPOSITORY_ROOT / "scripts" / "solution_image_cache.py"
    resolved.update(
        {
            "renderer_path": str(renderer_path.resolve()),
            "renderer_sha256": _sha256_file(renderer_path),
            "renderer_version": renderer_version,
            "runtime_distributions": distributions,
            "runtime_lock_path": str(lock_path.resolve()),
            "runtime_lock_sha256": _sha256_file(lock_path),
            "worker_path": str(worker_path.resolve()),
            "worker_sha256": _sha256_file(worker_path),
            "image_cache_helper_path": str(image_cache_helper_path.resolve()),
            "image_cache_helper_sha256": _sha256_file(image_cache_helper_path),
        }
    )
    resolved["engine_init_config_sha256"] = _sha256_bytes(
        _canonical_bytes(_paddle_engine_config(resolved))
    )
    resolved["worker_init_config_sha256"] = _sha256_bytes(
        _canonical_bytes(_paddle_worker_config(resolved))
    )
    return resolved


def _verify_locked_runtime(config: dict[str, Any]) -> None:
    environment_root = Path(config["environment_root"]).resolve()
    site_package_roots = list((environment_root / "lib").glob("python*/site-packages"))
    if len(site_package_roots) != 1:
        raise RunnerError("pinned PaddleOCR environment has an unexpected layout")
    site_packages = site_package_roots[0].resolve()
    lock_path = Path(config["runtime_lock_path"])
    if not lock_path.is_file() or _sha256_file(lock_path) != config["runtime_lock_sha256"]:
        raise RunnerError("pinned PaddleOCR runtime lock changed")
    for role in ("python", "renderer", "worker", "image_cache_helper"):
        path = Path(config[f"{role}_path"])
        if not path.is_file() or _sha256_file(path) != config[f"{role}_sha256"]:
            raise RunnerError(f"pinned OCR {role} artifact changed")
    for artifact in config["equivalence_artifacts"]:
        path = Path(artifact["path"])
        if not path.is_file() or _sha256_file(path) != artifact["sha256"]:
            raise RunnerError(f"pinned OCR equivalence artifact changed: {path.name}")
    for role in (
        "detector_onnx",
        "recognizer_onnx",
        "adapter_reference",
        "conversion_setup",
        "conversion_requirements",
    ):
        path = Path(config[f"{role}_path"])
        if not path.is_file() or _sha256_file(path) != config[f"{role}_sha256"]:
            raise RunnerError(f"pinned OCR {role} artifact changed")
    for role in ("detector", "recognizer"):
        path = Path(config[f"{role}_path"])
        if _directory_content_hash(path) != config[f"{role}_tree_sha256"]:
            raise RunnerError(f"pinned OCR {role} model tree changed")
    renderer_version = subprocess.run(
        [config["renderer_path"], "-v"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    version_lines = (renderer_version.stderr or renderer_version.stdout).splitlines()
    if (
        renderer_version.returncode != 0
        or not version_lines
        or version_lines[0] != config["renderer_version"]
    ):
        raise RunnerError("pinned OCR renderer version changed")
    for distribution in config["runtime_distributions"]:
        record_path = Path(distribution["record_path"])
        if _sha256_file(record_path) != distribution["record_sha256"]:
            raise RunnerError(f"pinned OCR RECORD changed: {distribution['name']}")
        with record_path.open(encoding="utf-8", newline="") as source:
            for relative_name, digest_field, size_field in csv.reader(source):
                if not digest_field:
                    continue
                algorithm, separator, encoded = digest_field.partition("=")
                if algorithm != "sha256" or not separator:
                    raise RunnerError("pinned OCR RECORD contains an unsupported digest")
                member = (site_packages / relative_name).resolve()
                try:
                    member.relative_to(environment_root)
                except ValueError as error:
                    raise RunnerError("pinned OCR RECORD member escapes its environment") from error
                if ".." in Path(relative_name).parts:
                    # Console-script shebangs contain the environment's absolute path and are
                    # rewritten by installers. The pinned RECORD itself still freezes the wheel
                    # inventory; importable runtime members below site-packages remain verified.
                    continue
                if not member.is_file():
                    raise RunnerError(f"pinned OCR runtime member is missing: {relative_name}")
                expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
                if _sha256_file(member) != expected or (
                    size_field and member.stat().st_size != int(size_field)
                ):
                    raise RunnerError(f"pinned OCR runtime member changed: {relative_name}")
    _verify_external_record(
        Path(config["onnxruntime_record_path"]),
        str(config["onnxruntime_record_sha256"]),
        Path(config["onnxruntime_site_packages"]),
    )
    onnxruntime_module = Path(config["onnxruntime_module_path"])
    if (
        not onnxruntime_module.is_file()
        or _sha256_file(onnxruntime_module) != config["onnxruntime_module_sha256"]
    ):
        raise RunnerError("pinned ONNX Runtime import module changed")


def _verify_external_record(record_path: Path, expected_sha256: str, site_packages: Path) -> None:
    if not record_path.is_file() or _sha256_file(record_path) != expected_sha256:
        raise RunnerError(f"pinned external OCR RECORD changed: {record_path}")
    environment_root = site_packages.parents[2]
    with record_path.open(encoding="utf-8", newline="") as source:
        for relative_name, digest_field, size_field in csv.reader(source):
            if not digest_field or ".." in Path(relative_name).parts:
                continue
            algorithm, separator, encoded = digest_field.partition("=")
            if algorithm != "sha256" or not separator:
                raise RunnerError("pinned external OCR RECORD has an unsupported digest")
            member = (site_packages / relative_name).resolve()
            try:
                member.relative_to(environment_root)
            except ValueError as error:
                raise RunnerError("pinned external OCR member escapes its environment") from error
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
            if not member.is_file() or _sha256_file(member) != expected or (
                size_field and member.stat().st_size != int(size_field)
            ):
                raise RunnerError(f"pinned external OCR runtime member changed: {relative_name}")


def _paddle_engine_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "text_detection_model_name": config["text_detection_model_name"],
        "text_detection_model_dir": config["detector_path"],
        "text_recognition_model_name": config["text_recognition_model_name"],
        "text_recognition_model_dir": config["recognizer_path"],
        "use_doc_orientation_classify": config["use_doc_orientation_classify"],
        "use_doc_unwarping": config["use_doc_unwarping"],
        "use_textline_orientation": config["use_textline_orientation"],
        "device": config["device"],
        "enable_mkldnn": config["enable_mkldnn"],
        "cpu_threads": config["cpu_threads"],
        "text_recognition_batch_size": config["text_recognition_batch_size"],
    }


def _paddle_worker_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "paddle": _paddle_engine_config(config),
        "onnxruntime": {
            "site_packages": config["onnxruntime_site_packages"],
            "version": config["onnxruntime_version"],
            "module_path": config["onnxruntime_module_path"],
            "module_sha256": config["onnxruntime_module_sha256"],
            "detector_model": config["detector_onnx_path"],
            "detector_sha256": config["detector_onnx_sha256"],
            "recognizer_model": config["recognizer_onnx_path"],
            "recognizer_sha256": config["recognizer_onnx_sha256"],
            "intra_op_num_threads": config["onnxruntime_intra_op_num_threads"],
            "inter_op_num_threads": config["onnxruntime_inter_op_num_threads"],
            "execution_mode": config["onnxruntime_execution_mode"],
            "graph_optimization_level": config["onnxruntime_graph_optimization_level"],
            "providers": [config["onnxruntime_provider"]],
        },
    }


def _start_ocr_pool(config: dict[str, Any]) -> None:
    global _OCR_POOL
    global _OCR_POOL_BASE_TIMEOUT_SECONDS
    global _OCR_POOL_PAGE_TIMEOUT_SECONDS
    global _OCR_POOL_SHUTDOWN_GRACE_SECONDS
    with _OCR_POOL_LOCK:
        if _OCR_POOL is not None:
            raise RunnerError("PaddleOCR process pool is already running")
        _verify_locked_runtime(config)
        if Path(sys.executable).absolute() != Path(config["python_path"]).absolute():
            raise RunnerError(f"PaddleOCR runner must use pinned Python: {config['python_path']}")
        _OCR_POOL_BASE_TIMEOUT_SECONDS = float(config["ocr_document_base_timeout_seconds"])
        _OCR_POOL_PAGE_TIMEOUT_SECONDS = float(config["ocr_page_timeout_seconds"])
        _OCR_POOL_SHUTDOWN_GRACE_SECONDS = float(config["ocr_shutdown_grace_seconds"])
        _OCR_POOL = ProcessPoolExecutor(
            max_workers=int(config["ocr_processes"]),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(_paddle_worker_config(config),),
        )


def _terminate_ocr_pool(pool: ProcessPoolExecutor, *, grace_seconds: float) -> None:
    processes = list((getattr(pool, "_processes", None) or {}).values())
    manager_thread = getattr(pool, "_executor_manager_thread", None)
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in alive:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [process for process in alive if process.is_alive()]
    for process in alive:
        process.kill()
    for process in alive:
        process.join(timeout=grace_seconds)
    remaining = [process for process in alive if process.is_alive()]
    if manager_thread is not None:
        manager_thread.join(timeout=grace_seconds)
    if remaining:
        pids = ", ".join(str(process.pid) for process in remaining)
        raise RunnerError(f"could not terminate PaddleOCR worker processes: {pids}")


def _abort_ocr_pool(expected: ProcessPoolExecutor) -> None:
    global _OCR_POOL
    with _OCR_POOL_LOCK:
        if _OCR_POOL is not expected:
            return
        _OCR_POOL = None
    _terminate_ocr_pool(expected, grace_seconds=_OCR_POOL_SHUTDOWN_GRACE_SECONDS)


def _shutdown_ocr_pool() -> None:
    global _OCR_POOL
    with _OCR_POOL_LOCK:
        pool, _OCR_POOL = _OCR_POOL, None
    if pool is not None:
        _terminate_ocr_pool(pool, grace_seconds=_OCR_POOL_SHUTDOWN_GRACE_SECONDS)


def _paddle_document(images: Sequence[Path]) -> list[dict[str, Any]]:
    with _OCR_CAPACITY_SEMAPHORE:
        with _OCR_POOL_LOCK:
            pool = _OCR_POOL
        if pool is None:
            raise RunnerError("PaddleOCR process pool is not running")
        timeout = _OCR_POOL_BASE_TIMEOUT_SECONDS + len(images) * _OCR_POOL_PAGE_TIMEOUT_SECONDS
        try:
            future = pool.submit(recognize_document, [str(image) for image in images])
            return future.result(timeout=timeout)
        except TimeoutError as error:
            _abort_ocr_pool(pool)
            raise RunnerError(
                f"PaddleOCR document timed out after {timeout:g}s; resume with a fresh pool"
            ) from error
        except BrokenProcessPool as error:
            _abort_ocr_pool(pool)
            raise RunnerError(
                "PaddleOCR worker process failed; resume with a fresh pool"
            ) from error


def _paddle_page(
    payload: dict[str, Any], image: Path, page_number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    texts = payload["texts"]
    scores = payload["scores"]
    boxes = payload["boxes"]
    polygons = payload["polygons"]
    if not (len(texts) == len(scores) == len(boxes) == len(polygons)):
        raise RunnerError("PaddleOCR returned misaligned geometry, text, and confidence values")
    with Image.open(image) as pixels:
        width, height = pixels.size
    lines = []
    for line_index, (text, confidence, raw_box, polygon) in enumerate(
        zip(texts, scores, boxes, polygons, strict=True)
    ):
        try:
            x0, y0, x1, y1 = (float(value) for value in raw_box)
        except (TypeError, ValueError) as error:
            raise RunnerError("PaddleOCR returned an invalid text box") from error
        clipped = [
            min(max(x0, 0.0), float(width)),
            min(max(y0, 0.0), float(height)),
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
        ]
        left, top, right, bottom = clipped
        raw_geometry = (
            {"quadrilateral": polygon}
            if polygon is not None
            else {"axis_aligned_box": [x0, y0, x1, y1]}
        )
        lines.append(
            {
                "line_index": line_index,
                "source_order": line_index,
                "text": str(text),
                "confidence": float(confidence),
                "raw_bbox": [x0, y0, x1, y1],
                "raw_geometry": raw_geometry,
                "clipped_bbox": clipped,
                "bbox": {
                    "left": math.floor(left),
                    "top": math.floor(top),
                    "width": max(0, math.ceil(right) - math.floor(left)),
                    "height": max(0, math.ceil(bottom) - math.floor(top)),
                },
            }
        )
    page = {"page_number": page_number, "text": "\n".join(line["text"] for line in lines)}
    geometry = {
        "page_number": page_number,
        "width": width,
        "height": height,
        "ocr_image_sha256": _sha256_file(image),
        "lines": lines,
    }
    geometry["source_sha256"] = _sha256_bytes(_canonical_bytes(geometry))
    return page, geometry


def _page_image_number(path: Path) -> int:
    match = _PAGE_IMAGE.fullmatch(path.name)
    if match is None:
        raise RunnerError(f"unexpected rendered page filename: {path.name}")
    return int(match.group(1))


def load_pages(task: dict[str, Any], pdf_path: Path, cache_dir: Path) -> PageBundle:
    """OCR lossless renders with pinned PaddleOCR and retain derived solver JPEGs."""
    _private_dir(cache_dir)
    if shutil.disk_usage(cache_dir).free < OCR_CONFIG["minimum_free_bytes"]:
        raise RunnerError("insufficient free disk space for retained page images")
    ocr_provenance = _resolved_ocr_config()
    prefix = cache_dir / "ocr-page"
    with _RENDER_SEMAPHORE:
        rendered = _run_checked(
            [
                str(ocr_provenance["renderer_path"]),
                f"-{OCR_CONFIG['ocr_input_format']}",
                "-r",
                str(OCR_CONFIG["dpi"]),
                str(pdf_path),
                str(prefix),
            ]
        )
    if rendered.returncode != 0:
        raise RunnerError(f"pdftoppm failed: {rendered.stderr.strip()}")
    ocr_images = sorted(cache_dir.glob("ocr-page-*.png"), key=_page_image_number)
    if not ocr_images:
        raise RunnerError("pdftoppm produced no page images")
    images: list[Path] = []
    for ocr_image in ocr_images:
        image = cache_dir / f"page-{_page_image_number(ocr_image)}.jpg"
        with Image.open(ocr_image) as pixels:
            pixels.convert("RGB").save(
                image,
                format="JPEG",
                quality=OCR_CONFIG["retained_jpeg_quality"],
            )
        image.chmod(0o600)
        images.append(image)

    supplied = task.get("source_pages")
    geometry: list[dict[str, Any]] = []
    if supplied is not None:
        if not isinstance(supplied, list) or len(supplied) != len(images):
            raise RunnerError("supplied source_pages must match the rendered PDF page count")
        pages = supplied
    else:
        pages = []
        try:
            payloads = _paddle_document(ocr_images)
        except Exception as error:  # noqa: BLE001 - normalize process-pool failures
            raise RunnerError(f"PaddleOCR failed: {error}") from error
        if len(payloads) != len(ocr_images):
            raise RunnerError("PaddleOCR returned the wrong number of pages")
        for image, payload in zip(ocr_images, payloads, strict=True):
            page_number = _page_image_number(image)
            try:
                page, page_geometry = _paddle_page(payload, image, page_number)
            except Exception as error:  # noqa: BLE001 - normalize pinned OCR failures
                raise RunnerError(f"PaddleOCR failed on page {page_number}: {error}") from error
            pages.append(page)
            geometry.append(page_geometry)
    for image in ocr_images:
        image.unlink(missing_ok=True)
    _write_json(cache_dir / "source_pages.json", pages)
    _write_json(cache_dir / "ocr_geometry.json", geometry)
    _write_json(
        cache_dir / "images.json",
        [
            {
                "page_number": _page_image_number(path),
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for path in images
        ],
    )
    return PageBundle(
        pages=pages,
        images=images,
        ocr_provenance=ocr_provenance,
        geometry=geometry,
    )


def _generation_config() -> dict[str, Any]:
    codex_runtime = _resolved_codex_runtime()
    ocr = _resolved_ocr_config()
    return {
        **COMMAND_CONFIG,
        **codex_runtime,
        "prompt_instructions_sha256": _sha256_bytes(PROMPT_INSTRUCTIONS.encode("utf-8")),
        "base_prompt_template_sha256": _sha256_bytes(BASE_PROMPT_TEMPLATE.encode("utf-8")),
        "retry_prompt_template_sha256": _sha256_bytes(RETRY_PROMPT_TEMPLATE.encode("utf-8")),
        "output_schema_sha256": _sha256_bytes(_canonical_bytes(OUTPUT_SCHEMA)),
        "ocr": ocr,
        "source_checker": _resolved_source_check_config(codex_runtime, ocr),
    }


_CODEX_NODE_TARGETS = {
    ("Darwin", "arm64"): ("@openai/codex-darwin-arm64", "aarch64-apple-darwin"),
    ("Darwin", "x86_64"): ("@openai/codex-darwin-x64", "x86_64-apple-darwin"),
}


def _native_codex_executable(wrapper: Path) -> tuple[Path, str]:
    if wrapper.name != "codex.js":
        return wrapper, "direct-executable-v1"
    try:
        wrapper_source = wrapper.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RunnerError("configured Codex wrapper cannot be inspected") from error
    required_markers = (
        "#!/usr/bin/env node",
        "PLATFORM_PACKAGE_BY_TARGET",
        "findCodexExecutable",
        "vendorRoot",
        "targetTriple",
        "const codexExecutable = path.join(",
        'process.platform === "win32"',
    )
    if not all(marker in wrapper_source for marker in required_markers):
        raise RunnerError("configured Codex wrapper layout is not recognized")
    target = _CODEX_NODE_TARGETS.get((platform.system(), platform.machine()))
    if target is None:
        raise RunnerError("configured Codex wrapper target is unsupported")
    platform_package, target_triple = target
    package_root = wrapper.parent.parent.resolve()
    installation_node_modules = package_root.parents[1]
    candidates = (
        package_root
        / "node_modules"
        / platform_package
        / "vendor"
        / target_triple
        / "bin"
        / "codex",
        package_root / "vendor" / target_triple / "bin" / "codex",
        installation_node_modules
        / platform_package
        / "vendor"
        / target_triple
        / "bin"
        / "codex",
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.is_relative_to(installation_node_modules):
            return resolved, "openai-node-wrapper-vendor-layout-v1"
    raise RunnerError("configured Codex wrapper native executable is unavailable")


def _resolved_codex_runtime() -> dict[str, str]:
    configured = Path(str(COMMAND_CONFIG["codex_executable"]))
    executable_value = (
        str((REPOSITORY_ROOT / configured).resolve())
        if configured.parent != Path(".")
        else shutil.which(str(configured))
    )
    if executable_value is None:
        raise RunnerError("configured Codex executable is unavailable")
    wrapper = Path(executable_value).resolve()
    if not wrapper.is_file():
        raise RunnerError("configured Codex executable is unavailable")
    executable, resolution_method = _native_codex_executable(wrapper)
    if _sha256_file(wrapper) != COMMAND_CONFIG["codex_wrapper_sha256"]:
        raise RunnerError("configured Codex wrapper does not match the pinned hash")
    if _sha256_file(executable) != COMMAND_CONFIG["codex_native_sha256"]:
        raise RunnerError("configured Codex native executable does not match the pinned hash")
    version = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    observed_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or observed_version != (
        f"codex-cli {COMMAND_CONFIG['codex_cli_version']}"
    ):
        raise RunnerError("configured Codex CLI version does not match the pinned version")
    return {
        "codex_executable_path": str(executable),
        "codex_executable_sha256": _sha256_file(executable),
        "codex_observed_version": observed_version,
        "codex_wrapper_path": str(wrapper),
        "codex_wrapper_sha256": _sha256_file(wrapper),
        "codex_resolution_method": resolution_method,
    }


def _verify_codex_runtime(config: dict[str, Any]) -> None:
    wrapper = Path(config["codex_wrapper_path"])
    if not wrapper.is_file() or _sha256_file(wrapper) != config["codex_wrapper_sha256"]:
        raise RunnerError("pinned Codex wrapper changed")
    executable = Path(config["codex_executable_path"])
    if not executable.is_file() or _sha256_file(executable) != config["codex_executable_sha256"]:
        raise RunnerError("pinned Codex executable changed")
    resolved_executable, resolution_method = _native_codex_executable(wrapper.resolve())
    if (
        resolved_executable != executable.resolve()
        or resolution_method != config["codex_resolution_method"]
    ):
        raise RunnerError("pinned Codex wrapper resolution changed")
    version = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    observed_version = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or observed_version != config["codex_observed_version"]:
        raise RunnerError("pinned Codex executable version changed")


def _resolved_source_check_config(
    codex_runtime: dict[str, str], ocr: dict[str, Any]
) -> dict[str, Any]:
    module_path = REPOSITORY_ROOT / "src" / "docinsights_analysis" / "visual_id_checks.py"
    if not module_path.is_file():
        raise RunnerError("source-region checker implementation is missing")
    return {
        **SOURCE_CHECK_CONFIG,
        **codex_runtime,
        "renderer_path": ocr["renderer_path"],
        "renderer_sha256": ocr["renderer_sha256"],
        "renderer_version": ocr["renderer_version"],
        "dpi": ocr["dpi"],
        "ocr_input_format": ocr["ocr_input_format"],
        "module_path": str(module_path.resolve()),
        "module_sha256": _sha256_file(module_path),
        "output_schema_sha256": _sha256_bytes(_canonical_bytes(SOURCE_CHECK_SCHEMA)),
    }


def _command_config_hash() -> str:
    return _sha256_bytes(_canonical_bytes(_generation_config()))


def _manifest_payload(
    split: str,
    entries: list[dict[str, Any]],
    tasks_path: Path,
    pdf_root: Path,
    official_tasks_sha256: str,
) -> dict[str, Any]:
    command_config = _generation_config()
    return {
        "split": split,
        "tasks_path": str(tasks_path.resolve()),
        "pdf_root": str(pdf_root.resolve()),
        "tasks": entries,
        "official_task_count": OFFICIAL_SPLITS[split]["count"],
        "official_tasks_sha256": official_tasks_sha256,
        "input_manifest_sha256": _sha256_bytes(_canonical_bytes(entries)),
        "command_config": command_config,
        "command_config_sha256": _sha256_bytes(_canonical_bytes(command_config)),
        "created_at": datetime.now(UTC).isoformat(),
        "private": True,
    }


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"invalid {description}: {error}") from error


def _attempt_artifacts(attempt_dir: Path) -> dict[str, str | None]:
    required = ("command.json", "prompt.txt", "events.jsonl", "stderr.txt")
    artifacts: dict[str, str | None] = {}
    for name in (*required, "response.json", "primary-response.json", "primary-record.json"):
        path = attempt_dir / name
        if name in required and not path.is_file():
            raise RunnerError(f"attempt artifact is missing: {path}")
        artifacts[name] = _sha256_file(path) if path.is_file() else None
    return artifacts


def _directory_artifacts(root: Path) -> list[dict[str, str]]:
    if not root.exists():
        return []
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RunnerError(f"artifact path contains a symlink: {path}")
        if path.is_file():
            artifacts.append({"path": str(path.resolve()), "sha256": _sha256_file(path)})
    return artifacts


def _verify_attempt_artifacts(job_dir: Path, attempts: Any) -> bool:
    if not isinstance(attempts, list) or not attempts:
        return False
    seen: set[int] = set()
    for attempt in attempts:
        if not isinstance(attempt, dict):
            return False
        number = attempt.get("attempt")
        artifacts = attempt.get("artifacts")
        if not isinstance(number, int) or number < 1 or number in seen:
            return False
        if not isinstance(artifacts, dict):
            return False
        seen.add(number)
        attempt_dir = job_dir / "attempts" / str(number)
        try:
            if artifacts != _attempt_artifacts(attempt_dir):
                return False
            if attempt.get("source_check_artifacts", []) != _directory_artifacts(
                attempt_dir / "source-checks"
            ):
                return False
        except RunnerError:
            return False
    return seen == set(range(1, max(seen) + 1))


def _verified_job_input(
    job_dir: Path,
    entry: dict[str, Any],
    command_config_sha256: str,
    expected_input_sha256: Any,
    expected_ocr: dict[str, Any],
    *,
    allow_missing_page_jpegs: bool = False,
) -> dict[str, Any] | None:
    input_path = job_dir / "input.json"
    if not input_path.is_file():
        return None
    job_input = _load_json(input_path, f"{job_dir.name} input")
    if expected_input_sha256 != _sha256_bytes(_canonical_bytes(job_input)):
        return None
    if job_input.get("task") != entry["task"]:
        return None
    if job_input.get("pdf_sha256") != entry["pdf_sha256"]:
        return None
    if job_input.get("pdf_path") != entry["pdf_path"]:
        return None
    if job_input.get("command_config_sha256") != command_config_sha256:
        return None
    images = job_input.get("page_images")
    if not isinstance(images, list) or not images:
        return None
    try:
        validate_page_jpeg_cache(
            source_dir=job_dir / "source",
            page_images=images,
            cache_config=expected_ocr,
            allow_missing_for_verified_success=allow_missing_page_jpegs,
        )
    except ImageCacheError:
        return None
    if not isinstance(job_input.get("source_pages"), list):
        return None
    source_pages_path = job_dir / "source" / "source_pages.json"
    images_path = job_dir / "source" / "images.json"
    geometry_path = job_dir / "source" / "ocr_geometry.json"
    if not source_pages_path.is_file() or not images_path.is_file() or not geometry_path.is_file():
        return None
    if _load_json(source_pages_path, f"{job_dir.name} source pages") != job_input["source_pages"]:
        return None
    if _load_json(images_path, f"{job_dir.name} image inventory") != images:
        return None
    geometry = job_input.get("ocr_geometry")
    if (
        not isinstance(geometry, list)
        or _load_json(geometry_path, f"{job_dir.name} OCR geometry") != geometry
    ):
        return None
    if job_input.get("ocr_geometry_sha256") != _sha256_bytes(_canonical_bytes(geometry)):
        return None
    if job_input.get("source_pages_sha256") != _sha256_bytes(
        _canonical_bytes(job_input["source_pages"])
    ):
        return None
    ocr = job_input.get("ocr")
    if not isinstance(ocr, dict) or ocr != expected_ocr:
        return None
    binary_path_value = ocr.get("binary_path")
    binary_sha256 = ocr.get("binary_sha256")
    if binary_path_value is not None or binary_sha256 is not None:
        if not isinstance(binary_path_value, str) or not isinstance(binary_sha256, str):
            return None
        binary_path = Path(binary_path_value)
        try:
            binary_path.resolve().relative_to((job_dir.parents[1] / "runtime").resolve())
        except ValueError:
            return None
        if not binary_path.is_file() or _sha256_file(binary_path) != binary_sha256:
            return None
    return job_input


def _verified_source_checks(
    proofs: Any,
    job_dir: Path,
    job_input: dict[str, Any],
    checker_config: dict[str, Any],
    primary: dict[str, Any],
    primary_response_sha256: str,
) -> list[dict[str, Any]] | None:
    if proofs is None:
        return []
    if not isinstance(proofs, list):
        return None
    expected_config_sha256 = _sha256_bytes(_canonical_bytes(checker_config))
    expected_primary_sha256 = primary_record_digest(primary)
    geometry_by_page = {
        page["page_number"]: page
        for page in job_input["ocr_geometry"]
        if isinstance(page, dict) and isinstance(page.get("page_number"), int)
    }
    source_root = (job_dir / "attempts").resolve()

    def intact(artifact: Any) -> bool:
        if not isinstance(artifact, dict) or set(artifact) != {"kind", "path", "sha256"}:
            return False
        path_value = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            return False
        path = Path(path_value)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            return False
        external_paths = {
            Path(job_input["pdf_path"]).resolve(),
            Path(checker_config["renderer_path"]).resolve(),
            Path(checker_config["codex_executable_path"]).resolve(),
        }
        if path.resolve() in external_paths:
            return True
        try:
            path.resolve().relative_to(source_root)
        except ValueError:
            return False
        return "source-checks" in path.parts

    for proof in proofs:
        if not isinstance(proof, dict):
            return None
        if proof.get("checker_config_sha256") != expected_config_sha256:
            return None
        if (
            proof.get("primary_record_sha256") != expected_primary_sha256
            or proof.get("primary_response_sha256") != primary_response_sha256
            or proof.get("pdf_sha256") != job_input["pdf_sha256"]
        ):
            return None
        geometry = geometry_by_page.get(proof.get("page"))
        if not isinstance(geometry, dict) or proof.get("ocr_image_sha256") != geometry.get(
            "ocr_image_sha256"
        ):
            return None
        if proof.get("page_image_sha256") != proof.get("ocr_image_sha256"):
            return None
        if (
            proof.get("model") != checker_config["model"]
            or proof.get("method") != checker_config["method"]
        ):
            return None
        named = {
            "pdf_artifact": "pdf_sha256",
            "renderer_artifact": "renderer_sha256",
            "page_image_artifact": "page_image_sha256",
            "context_crop_artifact": "context_crop_sha256",
            "anchor_crop_artifact": "anchor_crop_sha256",
            "checker_executable_artifact": "checker_executable_sha256",
            "raw_response_artifact": "raw_response_sha256",
        }
        for artifact_field, digest_field in named.items():
            artifact = proof.get(artifact_field)
            if (
                not isinstance(artifact, dict)
                or not intact(artifact)
                or artifact.get("sha256") != proof.get(digest_field)
            ):
                return None
        expected_external = {
            "pdf_artifact": Path(job_input["pdf_path"]).resolve(),
            "renderer_artifact": Path(checker_config["renderer_path"]).resolve(),
            "checker_executable_artifact": Path(checker_config["codex_executable_path"]).resolve(),
        }
        if any(
            Path(proof[field]["path"]).resolve() != expected_path
            for field, expected_path in expected_external.items()
        ):
            return None
        artifacts = proof.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not all(intact(artifact) for artifact in artifacts)
        ):
            return None
        if any(proof.get(field) not in artifacts for field in named):
            return None
    return proofs


def _verified_cached_record(
    job_dir: Path,
    entry: dict[str, Any],
    command_config_sha256: str,
    expected_config: dict[str, Any],
) -> dict[str, Any] | None:
    record_path = job_dir / "record.json"
    state_path = job_dir / "job.json"
    if not record_path.is_file() or not state_path.is_file():
        return None
    record = _load_json(record_path, f"{job_dir.name} record")
    state = _load_json(state_path, f"{job_dir.name} state")
    if state.get("status") != "succeeded":
        return None
    if state.get("manifest_entry_sha256") != _sha256_bytes(_canonical_bytes(entry)):
        return None
    if state.get("command_config_sha256") != command_config_sha256:
        return None
    if state.get("record_sha256") != _sha256_file(record_path):
        return None
    job_input = _verified_job_input(
        job_dir,
        entry,
        command_config_sha256,
        state.get("job_input_sha256"),
        expected_config["ocr"],
        allow_missing_page_jpegs=True,
    )
    if job_input is None:
        return None
    pages = cast(list[dict[str, Any]], job_input["source_pages"])
    attempts = state.get("attempts")
    if not _verify_attempt_artifacts(job_dir, attempts):
        return None
    successful_attempts = [attempt for attempt in attempts if attempt.get("status") == "succeeded"]
    if len(successful_attempts) != 1:
        return None
    selected_attempt = successful_attempts[0]
    response_path = job_dir / "attempts" / str(selected_attempt["attempt"]) / "response.json"
    if not response_path.is_file():
        return None
    raw_output_sha256 = _sha256_file(response_path)
    if state.get("raw_output_sha256") != raw_output_sha256:
        return None
    try:
        generated = json.loads(response_path.read_text(encoding="utf-8"))
        provenance = dict(record["provenance"])
        raw_proofs = provenance.pop("source_checks")
        primary = validate_primary_solution(
            {
                "instance_id": entry["task"]["instance_id"],
                "split": record["split"],
                "question": entry["task"]["user_query"],
                "solution": generated["solution"],
                "answer": generated["answer"],
                "evidence_regions": generated["evidence_regions"],
                "source_pages": pages,
                "provenance": provenance,
                "uncertainties": generated["uncertainties"],
            },
            entry["task"],
            pages,
        )
        source_checks = _verified_source_checks(
            raw_proofs,
            job_dir,
            job_input,
            expected_config["source_checker"],
            primary,
            raw_output_sha256,
        )
        if source_checks is None or state.get("source_checks_sha256") != _sha256_bytes(
            _canonical_bytes(source_checks)
        ):
            return None
        normalized = validate_solution(record, entry["task"], pages, source_checks=source_checks)
    except (json.JSONDecodeError, KeyError, SolutionRecordError, TypeError):
        return None
    for field_name in ("solution", "answer", "evidence_regions", "uncertainties"):
        if primary[field_name] != normalized[field_name]:
            return None
    provenance = primary["provenance"]
    if provenance.get("pdf_sha256") != entry["pdf_sha256"]:
        return None
    if provenance.get("config_sha256") != command_config_sha256:
        return None
    if provenance.get("output_sha256") != raw_output_sha256:
        return None
    if provenance.get("ocr") != expected_config["ocr"]:
        return None
    return normalized


def _evict_verified_record_page_jpegs(
    job_dir: Path, record: dict[str, Any], ocr_config: dict[str, Any]
) -> None:
    provenance = record.get("provenance")
    page_images = provenance.get("page_images") if isinstance(provenance, dict) else None
    if not isinstance(page_images, list):
        raise RunnerError("verified record does not contain a page image ledger")
    try:
        evict_verified_success_page_jpegs(
            source_dir=job_dir / "source",
            page_images=page_images,
            cache_config=ocr_config,
            success_verified=True,
        )
    except ImageCacheError as error:
        raise RunnerError(f"verified page JPEG cache could not be evicted: {error}") from error


def _postcommit_success(
    job_dir: Path,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    *,
    verified: dict[str, Any] | None = None,
) -> bool:
    record = verified or _verified_cached_record(
        job_dir,
        entry,
        manifest["command_config_sha256"],
        manifest["command_config"],
    )
    if record is None:
        raise RunnerError("new success record failed frozen verification")
    warning_path = job_dir / "cache-cleanup-warning.json"
    try:
        _evict_verified_record_page_jpegs(
            job_dir, record, manifest["command_config"]["ocr"]
        )
    except OSError as error:
        _write_json(
            warning_path,
            {
                "status": "verified_success_cache_cleanup_failed",
                "error_kind": "page_jpeg_cleanup_os_error",
                "error": str(error),
                "record_sha256": _sha256_file(job_dir / "record.json"),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        return True
    warning_path.unlink(missing_ok=True)
    return True


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"invalid {description}: {error}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise RunnerError(f"invalid {description}: rows must be objects")
    return rows


def _verify_complete_export(
    split_dir: Path,
    split: str,
    run_manifest: dict[str, Any],
    entries: list[dict[str, Any]],
) -> None:
    complete_path = split_dir / "complete.json"
    export_manifest_path = split_dir / "manifest.json"
    if not complete_path.is_file() or not export_manifest_path.is_file():
        raise RunnerError(f"{split} does not have a complete export")
    complete = _load_json(complete_path, f"{split} completion marker")
    export_manifest = _load_json(export_manifest_path, f"{split} export manifest")
    expected_ids = sorted(entry["task"]["instance_id"] for entry in entries)
    if (
        complete.get("status") != "generation_complete"
        or complete.get("coverage_complete") is not True
        or complete.get("split") != split
        or complete.get("total") != len(entries)
        or complete.get("input_manifest_sha256") != run_manifest["input_manifest_sha256"]
        or complete.get("command_config_sha256") != run_manifest["command_config_sha256"]
        or complete.get("record_ids_sha256")
        != _sha256_bytes(_canonical_bytes([entry["task"]["instance_id"] for entry in entries]))
        or complete.get("export_manifest_sha256") != _sha256_file(export_manifest_path)
    ):
        raise RunnerError(f"{split} does not have a complete export binding")
    count_fields = (
        "answer_coverage",
        "fully_grounded_count",
        "evidence_unresolved_count",
        "runtime_failed_count",
        "submission_ready",
        "submission_mode",
        "submission_count",
        "grounded_submission_count",
        "submission_abstention_count",
    )
    if any(complete.get(field) != export_manifest.get(field) for field in count_fields):
        raise RunnerError(f"{split} completion counts do not match its export")
    if (
        export_manifest.get("private") is not True
        or export_manifest.get("split") != split
        or export_manifest.get("total") != len(entries)
        or export_manifest.get("coverage_complete") is not True
        or export_manifest.get("answer_coverage") != len(entries)
        or export_manifest.get("runtime_failed_count") != 0
        or export_manifest.get("instance_ids") != expected_ids
    ):
        raise RunnerError(f"{split} does not have complete export coverage")
    files = export_manifest.get("files")
    required_files = {"solutions.jsonl", "solutions.md", "grounded-submission.jsonl"}
    if (
        not isinstance(files, dict)
        or not required_files <= set(files)
        or set(files) - (required_files | {"submission.jsonl"})
    ):
        raise RunnerError(f"{split} does not have a complete export file manifest")
    for name in files:
        path = split_dir / name
        metadata = files.get(name)
        if (
            not path.is_file()
            or not isinstance(metadata, dict)
            or metadata.get("sha256") != _sha256_file(path)
        ):
            raise RunnerError(f"{split} does not have an intact complete export: {name}")
    records = []
    for entry in entries:
        record = _verified_cached_record(
            split_dir / "jobs" / entry["task"]["instance_id"],
            entry,
            run_manifest["command_config_sha256"],
            run_manifest["command_config"],
        )
        if record is None:
            raise RunnerError(f"{split} does not have complete record coverage")
        records.append(record)
    records.sort(key=lambda record: record["instance_id"])
    if _read_jsonl(split_dir / "solutions.jsonl", f"{split} solutions export") != records:
        raise RunnerError(f"{split} solutions export does not match frozen records")
    grounded_submission = [
        {
            "instance_id": record["instance_id"],
            "answer": record["answer"],
            "evidence": record["evidence"],
        }
        for record in records
        if record["source_status"] == "fully_grounded"
    ]
    if (
        _read_jsonl(split_dir / "grounded-submission.jsonl", f"{split} grounded submission export")
        != grounded_submission
    ):
        raise RunnerError(f"{split} grounded submission does not match frozen records")
    grounded_count = len(grounded_submission)
    unresolved_ids = [
        record["instance_id"]
        for record in records
        if record["source_status"] == "evidence_unresolved"
    ]
    if (
        export_manifest.get("fully_grounded_count") != grounded_count
        or export_manifest.get("evidence_unresolved_count") != len(unresolved_ids)
        or export_manifest.get("grounded_submission_count") != grounded_count
        or export_manifest.get("submission_exclusions")
        != [
            {"instance_id": instance_id, "reason": "evidence_unresolved"}
            for instance_id in unresolved_ids
        ]
    ):
        raise RunnerError(f"{split} export status counts do not match frozen records")
    submission_path = split_dir / "submission.jsonl"
    if unresolved_ids:
        if (
            export_manifest.get("submission_ready") is not False
            or export_manifest.get("submission_mode") != "blocked_unresolved"
            or export_manifest.get("submission_count") != 0
            or export_manifest.get("submission_abstention_count") != 0
            or submission_path.exists()
        ):
            raise RunnerError(f"{split} unresolved evidence must block full submission")
    elif (
        export_manifest.get("submission_ready") is not True
        or export_manifest.get("submission_mode") != "fully_grounded"
        or export_manifest.get("submission_count") != len(records)
        or export_manifest.get("submission_abstention_count") != 0
        or _read_jsonl(submission_path, f"{split} submission export") != grounded_submission
    ):
        raise RunnerError(f"{split} submission export does not match frozen records")


def _verify_prior_split(output_root: Path, split: str) -> None:
    split_dir = output_root / split
    manifest_path = split_dir / "run-manifest.json"
    if not manifest_path.is_file():
        raise RunnerError(f"{split} does not have complete coverage")
    manifest = _load_json(manifest_path, f"{split} run manifest")
    pin = OFFICIAL_SPLITS[split]
    if (
        manifest.get("split") != split
        or manifest.get("official_task_count") != pin["count"]
        or manifest.get("official_tasks_sha256") != pin["tasks_sha256"]
    ):
        raise RunnerError(f"{split} does not match its official manifest pin")
    tasks_path = Path(manifest.get("tasks_path", ""))
    if not tasks_path.is_file() or _sha256_file(tasks_path) != pin["tasks_sha256"]:
        raise RunnerError(f"{split} does not match its official manifest pin")
    frozen_tasks = _read_tasks(tasks_path)
    entries = manifest.get("tasks")
    if not isinstance(entries, list) or manifest.get("input_manifest_sha256") != _sha256_bytes(
        _canonical_bytes(entries)
    ):
        raise RunnerError(f"{split} does not have complete coverage with a valid manifest hash")
    if len(frozen_tasks) != pin["count"] or [entry.get("task") for entry in entries] != [
        {
            key: task[key]
            for key in ("instance_id", "user_query", "document_pdf", "source_pages")
            if key in task
        }
        for task in frozen_tasks
    ]:
        raise RunnerError(f"{split} does not match its official manifest pin")
    command_config = manifest.get("command_config")
    if not isinstance(command_config, dict) or manifest.get(
        "command_config_sha256"
    ) != _sha256_bytes(_canonical_bytes(command_config)):
        raise RunnerError(f"{split} does not have complete coverage with a valid config hash")
    if command_config != _generation_config():
        raise RunnerError(f"{split} generation config differs from the current split config")
    expected_ids = [entry.get("task", {}).get("instance_id") for entry in entries]
    if any(not isinstance(value, str) for value in expected_ids) or len(expected_ids) != len(
        set(expected_ids)
    ):
        raise RunnerError(f"{split} does not have complete coverage with unique IDs")
    successful: set[str] = set()
    for entry, instance_id in zip(entries, expected_ids, strict=True):
        pdf_path = Path(entry.get("pdf_path", ""))
        if not pdf_path.is_file() or entry.get("pdf_sha256") != _sha256_file(pdf_path):
            continue
        job_dir = split_dir / "jobs" / instance_id
        if _verified_cached_record(
            job_dir,
            entry,
            manifest["command_config_sha256"],
            manifest["command_config"],
        ):
            successful.add(instance_id)
    if successful != set(expected_ids):
        raise RunnerError(f"{split} does not have complete coverage")
    _verify_complete_export(split_dir, split, manifest, entries)


def _verify_split_evaluation(split_dir: Path, split: str) -> None:
    evaluator_path = Path(__file__).with_name("evaluate_solution_records.py")
    spec = importlib.util.spec_from_file_location("solution_records_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RunnerError(f"cannot load the {split} evaluation binding verifier")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module.verify_evaluation_binding(split_dir, expected_split=split)
    except Exception as error:  # noqa: BLE001 - normalize the sibling verifier boundary
        raise RunnerError(f"{split} evaluation binding is missing or invalid: {error}") from error


def _prepare_manifest(
    split_dir: Path,
    split: str,
    entries: list[dict[str, Any]],
    tasks_path: Path,
    pdf_root: Path,
    official_tasks_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    expected = _manifest_payload(split, entries, tasks_path, pdf_root, official_tasks_sha256)
    path = split_dir / "run-manifest.json"
    if path.exists():
        if not resume:
            raise RunnerError(f"output already exists for {split}; use --resume")
        actual = _load_json(path, "run manifest")
        if (
            actual.get("command_config_sha256") != expected["command_config_sha256"]
            or actual.get("command_config") != expected["command_config"]
        ):
            raise RunnerError("resume refused: command config hash mismatch")
        if (
            actual.get("input_manifest_sha256") != expected["input_manifest_sha256"]
            or actual.get("tasks") != entries
        ):
            raise RunnerError("resume refused: input manifest hash mismatch")
        return actual
    if resume:
        raise RunnerError(f"cannot resume {split}: run manifest is missing")
    _write_json(path, expected)
    return expected


def _prompt(task: dict[str, Any], pages: list[dict[str, Any]]) -> str:
    payload = {"question": task["user_query"], "source_pages": pages}
    return BASE_PROMPT_TEMPLATE.format(
        instructions=PROMPT_INSTRUCTIONS,
        public_input=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _codex_argv(
    job_dir: Path,
    attempt_dir: Path,
    images: Sequence[Path],
    *,
    schema_path: Path | None = None,
    response_path: Path | None = None,
    model: str | None = None,
    executable: str | None = None,
) -> list[str]:
    argv = [
        executable or str(COMMAND_CONFIG["codex_executable"]),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--strict-config",
        "--json",
        "--model",
        model or str(COMMAND_CONFIG["model"]),
        "--cd",
        str(job_dir),
        "--config",
        'web_search="disabled"',
        "--config",
        'model_reasoning_effort="high"',
    ]
    for feature in DISABLED_FEATURES:
        argv.extend(("--config", f"features.{feature}=false"))
    for image in images:
        argv.extend(("--image", str(image)))
    argv.extend(
        (
            "--output-schema",
            str(schema_path or job_dir.parent.parent / "schema.json"),
            "--output-last-message",
            str(response_path or attempt_dir / "response.json"),
            "-",
        )
    )
    return argv


def _event_summary(events_text: str) -> tuple[bool, dict[str, Any] | None]:
    completed = False
    usage = None
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise AttemptFailure("Codex emitted malformed JSONL", "invalid_event_stream") from error
        if not isinstance(event, dict):
            raise AttemptFailure("Codex emitted a non-object JSONL event", "invalid_event_stream")
        if event.get("type") == "turn.completed":
            completed = True
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
    return completed, usage


def _model_fallback_warning(events_text: str, stderr: str) -> str | None:
    diagnostics = [stderr]
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") in {"error", "turn.failed"}:
            diagnostics.append(json.dumps(event, ensure_ascii=False))
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "error"
        ):
            diagnostics.append(json.dumps(item, ensure_ascii=False))
    combined = "\n".join(diagnostics).casefold()
    indicators = (
        "defaulting to fallback metadata",
        "unknown model gpt-6-astra",
        "requires a newer version of codex",
    )
    return next((indicator for indicator in indicators if indicator in combined), None)


def _invoke_source_checker(
    *,
    prompt: str,
    images: Sequence[Path],
    output_schema: dict[str, Any],
    output_dir: Path,
    checker_config: dict[str, Any],
    command_runner: CommandRunner,
) -> dict[str, Any]:
    _verify_codex_runtime(checker_config)
    _private_dir(output_dir)
    schema_path = output_dir / "schema.json"
    response_path = output_dir / "response.json"
    _write_json(schema_path, output_schema)
    _write_private(output_dir / "prompt.txt", prompt)
    argv = _codex_argv(
        output_dir,
        output_dir,
        images,
        schema_path=schema_path,
        response_path=response_path,
        model=str(checker_config["model"]),
        executable=str(checker_config["codex_executable_path"]),
    )
    _write_json(output_dir / "command.json", {"argv": argv, "config": checker_config})
    try:
        completed = command_runner(
            argv,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=checker_config["timeout_seconds"],
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else error.stderr or ""
        )
        _write_private(output_dir / "events.jsonl", stdout)
        _write_private(output_dir / "stderr.txt", stderr)
        raise SourceRegionCheckError(
            f"source checker timed out after {checker_config['timeout_seconds']}s"
        ) from error
    events = completed.stdout or ""
    stderr = completed.stderr or ""
    _write_private(output_dir / "events.jsonl", events)
    _write_private(output_dir / "stderr.txt", stderr)
    fallback_warning = _model_fallback_warning(events, stderr)
    if fallback_warning is not None:
        raise SourceRegionCheckError(
            f"source checker could not verify requested model: {fallback_warning}"
        )
    turn_completed, _ = _event_summary(events)
    if completed.returncode != 0 or not turn_completed:
        raise SourceRegionCheckError("source checker did not complete successfully")
    if not response_path.is_file() or not response_path.read_text(encoding="utf-8").strip():
        raise SourceRegionCheckError("source checker returned an empty response")
    response_path.chmod(0o600)
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SourceRegionCheckError("source checker returned invalid JSON") from error
    artifacts = [
        schema_path,
        output_dir / "prompt.txt",
        output_dir / "command.json",
        output_dir / "events.jsonl",
        output_dir / "stderr.txt",
        response_path,
    ]
    return {
        "response": response,
        "raw_response_path": response_path,
        "artifact_paths": artifacts,
        "model": checker_config["model"],
        "method": checker_config["method"],
    }


def _classify_exception(error: Exception) -> str:
    if isinstance(error, AttemptFailure):
        return error.error_kind
    if isinstance(error, SourceRegionCheckError):
        return "source_check_runtime_failed"
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, (json.JSONDecodeError, KeyError, SolutionRecordError)):
        return "invalid_response"
    return "runtime_error"


def _resume_source_check(
    *,
    entry: dict[str, Any],
    job_dir: Path,
    manifest: dict[str, Any],
    previous_error: dict[str, Any],
    job_input: dict[str, Any],
    command_runner: CommandRunner,
    source_checker: SourceChecker,
) -> bool:
    attempts = previous_error["attempts"]
    source_attempt = attempts[-1]
    attempt_number = source_attempt["attempt"]
    attempt_dir = job_dir / "attempts" / str(attempt_number)
    response_path = attempt_dir / "response.json"
    primary_path = attempt_dir / "primary-record.json"
    primary_response_path = attempt_dir / "primary-response.json"
    raw_output_sha256 = _sha256_file(response_path)
    if (
        not primary_path.is_file()
        or not primary_response_path.is_file()
        or _sha256_file(primary_response_path) != raw_output_sha256
    ):
        raise RunnerError("resume refused: frozen primary artifacts are incomplete")
    task = entry["task"]
    pages = cast(list[dict[str, Any]], job_input["source_pages"])
    primary = _load_json(primary_path, f"{task['instance_id']} frozen primary")
    try:
        normalized_primary = validate_primary_solution(primary, task, pages)
        generated = _load_json(response_path, f"{task['instance_id']} primary response")
    except SolutionRecordError as error:
        raise RunnerError("resume refused: frozen primary record is invalid") from error
    if primary != normalized_primary or any(
        generated.get(field_name) != primary[field_name]
        for field_name in ("solution", "answer", "evidence_regions", "uncertainties")
    ):
        raise RunnerError("resume refused: frozen primary differs from its raw response")
    provenance = primary["provenance"]
    if (
        provenance.get("output_sha256") != raw_output_sha256
        or provenance.get("pdf_sha256") != entry["pdf_sha256"]
        or provenance.get("config_sha256") != manifest["command_config_sha256"]
    ):
        raise RunnerError("resume refused: frozen primary provenance is invalid")

    runtime_attempt = int(previous_error.get("source_runtime_attempts", 1)) + 1
    recovery_dir = attempt_dir / "source-checks" / f"runtime-recovery-{runtime_attempt}"
    if recovery_dir.exists():
        raise RunnerError("resume refused: source recovery directory already exists")
    frozen_primary_sha256 = primary_record_digest(primary)
    try:
        source_checks = source_checker(
            frozen_record=primary,
            pages=pages,
            ocr_geometry=job_input["ocr_geometry"],
            source_pdf_path=Path(entry["pdf_path"]),
            output_dir=recovery_dir,
            checker_config=manifest["command_config"]["source_checker"],
            invoke=lambda **kwargs: _invoke_source_checker(**kwargs, command_runner=command_runner),
        )
        if primary_record_digest(primary) != frozen_primary_sha256:
            raise SourceRegionCheckError("source checker mutated the frozen primary record")
        record = finalize_solution(primary, task, pages, source_checks=source_checks)
        normalized = validate_solution(record, task, pages, source_checks=source_checks)
    except Exception as error:  # noqa: BLE001 - preserve bounded recovery evidence
        error_kind = "source_check_runtime_failed"
        previous_error["error_kind"] = error_kind
        previous_error["error"] = str(error)
        previous_error["source_runtime_attempts"] = runtime_attempt
        previous_error.setdefault("source_runtime_history", []).append(
            {
                "runtime_attempt": runtime_attempt,
                "status": "failed",
                "error_kind": error_kind,
                "error": str(error),
                "artifacts": _directory_artifacts(recovery_dir),
            }
        )
        source_attempt["source_check_artifacts"] = _directory_artifacts(
            attempt_dir / "source-checks"
        )
        previous_error["timestamp"] = datetime.now(UTC).isoformat()
        _write_json(job_dir / "error.json", previous_error)
        return False

    _write_json(job_dir / "record.json", normalized)
    recovered_attempt = dict(source_attempt)
    recovered_attempt.pop("error_kind", None)
    recovered_attempt.pop("error", None)
    recovered_attempt["status"] = "succeeded"
    recovered_attempt["source_check_artifacts"] = _directory_artifacts(
        attempt_dir / "source-checks"
    )
    recovered_attempt["source_runtime_history"] = [
        {
            "runtime_attempt": 1,
            "status": "failed",
            "error_kind": source_attempt["error_kind"],
            "error": source_attempt["error"],
        },
        *previous_error.get("source_runtime_history", []),
        {
            "runtime_attempt": runtime_attempt,
            "status": "succeeded",
            "artifacts": _directory_artifacts(recovery_dir),
        },
    ]
    state = {
        "instance_id": task["instance_id"],
        "status": "succeeded",
        "manifest_entry_sha256": _sha256_bytes(_canonical_bytes(entry)),
        "command_config_sha256": manifest["command_config_sha256"],
        "job_input_sha256": previous_error["job_input_sha256"],
        "raw_output_sha256": raw_output_sha256,
        "source_checks_sha256": _sha256_bytes(_canonical_bytes(source_checks)),
        "record_sha256": _sha256_file(job_dir / "record.json"),
        "attempts": [*attempts[:-1], recovered_attempt],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    _write_json(job_dir / "job.json", state)
    (job_dir / "error.json").unlink(missing_ok=True)
    return _postcommit_success(job_dir, entry, manifest)


def _run_one(
    split: str,
    entry: dict[str, Any],
    split_dir: Path,
    manifest: dict[str, Any],
    resume: bool,
    page_loader: PageLoader,
    command_runner: CommandRunner,
    source_checker: SourceChecker,
) -> bool:
    task = entry["task"]
    command_config = manifest["command_config"]
    instance_id = task["instance_id"]
    job_dir = split_dir / "jobs" / instance_id
    _private_dir(job_dir)
    record_path = job_dir / "record.json"
    state_path = job_dir / "job.json"
    entry_hash = _sha256_bytes(_canonical_bytes(entry))
    if resume:
        cached = _verified_cached_record(
            job_dir,
            entry,
            manifest["command_config_sha256"],
            command_config,
        )
        if cached is not None:
            return _postcommit_success(job_dir, entry, manifest, verified=cached)
        error_path = job_dir / "error.json"
        if error_path.is_file():
            previous_error = _load_json(error_path, f"{instance_id} error")
            previous_attempts = previous_error.get("attempts")
            verified_input = _verified_job_input(
                job_dir,
                entry,
                manifest["command_config_sha256"],
                previous_error.get("job_input_sha256"),
                command_config["ocr"],
            )
            if (
                previous_error.get("manifest_entry_sha256") == entry_hash
                and previous_error.get("command_config_sha256") == manifest["command_config_sha256"]
                and isinstance(previous_attempts, list)
                and (
                    len(previous_attempts) >= command_config["maximum_attempts"]
                    or previous_error.get("terminal") is True
                )
                and _verify_attempt_artifacts(job_dir, previous_attempts)
                and verified_input is not None
            ):
                if (
                    previous_error.get("error_kind") == "source_check_runtime_failed"
                    and previous_error.get("source_runtime_attempts", 1)
                    < command_config["source_checker"]["maximum_runtime_attempts"]
                ):
                    return _resume_source_check(
                        entry=entry,
                        job_dir=job_dir,
                        manifest=manifest,
                        previous_error=previous_error,
                        job_input=verified_input,
                        command_runner=command_runner,
                        source_checker=source_checker,
                    )
                return False
        if record_path.exists() or state_path.exists() or (job_dir / "attempts").exists():
            raise RunnerError(
                f"resume refused: invalid or interrupted job artifacts for {instance_id}"
            )

    try:
        pdf_path = Path(entry["pdf_path"])
        if _sha256_file(pdf_path) != entry["pdf_sha256"]:
            raise AttemptFailure("PDF changed before page extraction", "input_changed")
        bundle = page_loader(task, pdf_path, job_dir / "source")
        if _sha256_file(pdf_path) != entry["pdf_sha256"]:
            raise AttemptFailure("PDF changed during page extraction", "input_changed")
        if not bundle.images or len(bundle.images) != len(bundle.pages):
            raise RunnerError("page loader returned no images")
        if bundle.ocr_provenance != command_config["ocr"]:
            raise RunnerError("page loader OCR provenance does not match frozen run config")
        image_metadata = [
            {
                "page_number": bundle.pages[index]["page_number"],
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for index, path in enumerate(bundle.images)
        ]
        _write_json(job_dir / "source" / "source_pages.json", bundle.pages)
        _write_json(job_dir / "source" / "images.json", image_metadata)
        _write_json(job_dir / "source" / "ocr_geometry.json", bundle.geometry)
        job_input = {
            "task": task,
            "pdf_path": entry["pdf_path"],
            "pdf_sha256": entry["pdf_sha256"],
            "source_pages": bundle.pages,
            "source_pages_sha256": _sha256_bytes(_canonical_bytes(bundle.pages)),
            "page_images": image_metadata,
            "ocr_geometry": bundle.geometry,
            "ocr_geometry_sha256": _sha256_bytes(_canonical_bytes(bundle.geometry)),
            "ocr": bundle.ocr_provenance,
            "command_config_sha256": manifest["command_config_sha256"],
        }
        job_input_hash = _sha256_bytes(_canonical_bytes(job_input))
        public_input_hash = input_digest(task, bundle.pages)
        _write_json(job_dir / "input.json", job_input)
        base_prompt = _prompt(task, bundle.pages)
        _write_private(job_dir / "prompt.txt", base_prompt)
    except Exception as error:  # noqa: BLE001 - persist all per-task boundary failures
        failure = {
            "instance_id": instance_id,
            "status": "failed",
            "error_kind": _classify_exception(error),
            "error": str(error),
            "attempts": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _write_json(job_dir / "error.json", failure)
        return False

    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    success_committed = False
    for attempt_number in range(1, command_config["maximum_attempts"] + 1):
        attempt_dir = job_dir / "attempts" / str(attempt_number)
        _private_dir(attempt_dir)
        _verify_codex_runtime(command_config)
        argv = _codex_argv(
            job_dir,
            attempt_dir,
            bundle.images,
            executable=str(command_config["codex_executable_path"]),
        )
        _write_json(attempt_dir / "command.json", {"argv": argv, "config": command_config})
        prompt = base_prompt
        if last_error is not None:
            prompt += RETRY_PROMPT_TEMPLATE.format(error=last_error)
        _write_private(attempt_dir / "prompt.txt", prompt)
        try:
            completed = command_runner(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=command_config["timeout_seconds"],
                check=False,
            )
            events = completed.stdout or ""
            stderr = completed.stderr or ""
            _write_private(attempt_dir / "events.jsonl", events)
            _write_private(attempt_dir / "stderr.txt", stderr)
            response_path = attempt_dir / "response.json"
            fallback_warning = _model_fallback_warning(events, stderr)
            if fallback_warning is not None:
                raise AttemptFailure(
                    f"Codex could not verify requested model: {fallback_warning}",
                    "model_fallback_warning",
                )
            turn_completed, usage = _event_summary(events)
            if completed.returncode != 0:
                raise AttemptFailure(
                    f"codex exited with status {completed.returncode}", "codex_exit"
                )
            if not turn_completed:
                raise AttemptFailure(
                    "Codex stream did not contain turn.completed", "incomplete_completion"
                )
            if not response_path.is_file() or not response_path.read_text(encoding="utf-8").strip():
                raise AttemptFailure("empty final response", "empty_final_response")
            response_path.chmod(0o600)
            raw_response = response_path.read_bytes()
            _write_private(attempt_dir / "primary-response.json", raw_response)
            generated = json.loads(raw_response)
            primary = validate_primary_solution(
                {
                    "instance_id": instance_id,
                    "split": split,
                    "question": task["user_query"],
                    "solution": generated["solution"],
                    "answer": generated["answer"],
                    "evidence_regions": generated["evidence_regions"],
                    "source_pages": bundle.pages,
                    "provenance": {
                        "pdf_sha256": entry["pdf_sha256"],
                        "input_sha256": public_input_hash,
                        "output_sha256": _sha256_bytes(raw_response),
                        "config_sha256": manifest["command_config_sha256"],
                        "model": command_config["model"],
                        "model_identity_evidence": command_config["model_identity_evidence"],
                        "backend_model_event_field": command_config[
                            "backend_model_event_field"
                        ],
                        "method": "codex-cli-schema-v1",
                        "ocr": job_input["ocr"],
                        "attempt": attempt_number,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "page_images": image_metadata,
                        "usage": usage,
                    },
                    "uncertainties": generated["uncertainties"],
                },
                task,
                bundle.pages,
            )
            _write_json(attempt_dir / "primary-record.json", primary)
            frozen_primary_sha256 = primary_record_digest(primary)
            try:
                source_checks = source_checker(
                    frozen_record=primary,
                    pages=bundle.pages,
                    ocr_geometry=bundle.geometry,
                    source_pdf_path=pdf_path,
                    output_dir=attempt_dir / "source-checks",
                    checker_config=command_config["source_checker"],
                    invoke=lambda **kwargs: _invoke_source_checker(
                        **kwargs, command_runner=command_runner
                    ),
                )
                if primary_record_digest(primary) != frozen_primary_sha256:
                    raise SourceRegionCheckError("source checker mutated the frozen primary record")
                record = finalize_solution(primary, task, bundle.pages, source_checks=source_checks)
                normalized = validate_solution(
                    record, task, bundle.pages, source_checks=source_checks
                )
            except SourceRegionCheckError:
                raise
            except Exception as error:
                raise SourceRegionCheckError(f"source checker failed: {error}") from error
            _write_json(record_path, normalized)
            state = {
                "instance_id": instance_id,
                "status": "succeeded",
                "manifest_entry_sha256": entry_hash,
                "command_config_sha256": manifest["command_config_sha256"],
                "job_input_sha256": job_input_hash,
                "raw_output_sha256": _sha256_bytes(raw_response),
                "source_checks_sha256": _sha256_bytes(_canonical_bytes(source_checks)),
                "record_sha256": _sha256_file(record_path),
                "attempts": attempts
                + [
                    {
                        "attempt": attempt_number,
                        "status": "succeeded",
                        "usage": usage,
                        "artifacts": _attempt_artifacts(attempt_dir),
                        "source_check_artifacts": _directory_artifacts(
                            attempt_dir / "source-checks"
                        ),
                    }
                ],
                "timestamp": datetime.now(UTC).isoformat(),
            }
            _write_json(state_path, state)
            (job_dir / "error.json").unlink(missing_ok=True)
            success_committed = True
            break
        except Exception as error:  # noqa: BLE001 - one audited retry for any invalid completion
            error_kind = getattr(error, "error_kind", _classify_exception(error))
            if isinstance(error, subprocess.TimeoutExpired):
                stdout = (
                    error.stdout.decode(errors="replace")
                    if isinstance(error.stdout, bytes)
                    else error.stdout or ""
                )
                stderr = (
                    error.stderr.decode(errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr or ""
                )
                _write_private(attempt_dir / "events.jsonl", stdout)
                _write_private(attempt_dir / "stderr.txt", stderr)
            if not (attempt_dir / "events.jsonl").exists():
                _write_private(attempt_dir / "events.jsonl", b"")
            if not (attempt_dir / "stderr.txt").exists():
                _write_private(attempt_dir / "stderr.txt", b"")
            response_path = attempt_dir / "response.json"
            if response_path.exists():
                response_path.chmod(0o600)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "failed",
                    "error_kind": error_kind,
                    "error": str(error),
                    "artifacts": _attempt_artifacts(attempt_dir),
                    "source_check_artifacts": _directory_artifacts(attempt_dir / "source-checks"),
                }
            )
            last_error = error
            if isinstance(error, SourceRegionCheckError):
                break

    if success_committed:
        return _postcommit_success(job_dir, entry, manifest)

    record_path.unlink(missing_ok=True)
    failure = {
        "instance_id": instance_id,
        "status": "failed",
        "error_kind": attempts[-1]["error_kind"],
        "error": str(last_error),
        "attempts": attempts,
        "manifest_entry_sha256": entry_hash,
        "command_config_sha256": manifest["command_config_sha256"],
        "job_input_sha256": job_input_hash,
        "terminal": isinstance(last_error, SourceRegionCheckError),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if isinstance(last_error, SourceRegionCheckError):
        failure["source_runtime_attempts"] = 1
    _write_json(job_dir / "error.json", failure)
    return False


def _all_records(
    split_dir: Path,
    entries: list[dict[str, Any]],
    command_config_sha256: str,
    expected_config: dict[str, Any],
) -> list[dict[str, Any]] | None:
    records = []
    for entry in entries:
        instance_id = entry["task"]["instance_id"]
        record = _verified_cached_record(
            split_dir / "jobs" / instance_id,
            entry,
            command_config_sha256,
            expected_config,
        )
        if record is None:
            return None
        records.append(record)
    return records


def _failure_summary(split_dir: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = []
    error_counts: dict[str, int] = {}
    for entry in entries:
        instance_id = entry["task"]["instance_id"]
        error_path = split_dir / "jobs" / instance_id / "error.json"
        if not error_path.is_file():
            continue
        error = _load_json(error_path, f"{instance_id} error")
        error_kind = error.get("error_kind", "unknown")
        if not isinstance(error_kind, str):
            error_kind = "unknown"
        tasks.append({"instance_id": instance_id, "error_kind": error_kind})
        error_counts[error_kind] = error_counts.get(error_kind, 0) + 1
    return {"failed": len(tasks), "error_counts": error_counts, "tasks": tasks}


def run_pipeline(
    *,
    split: str,
    tasks_path: Path,
    pdf_root: Path,
    output_root: Path,
    limit: int | None = None,
    workers: int = 2,
    resume: bool = False,
    smoke: bool = False,
    smoke_instance_id: str | None = None,
    page_loader: PageLoader = load_pages,
    command_runner: CommandRunner | None = None,
    source_checker: SourceChecker | None = None,
) -> dict[str, Any]:
    if split not in SPLIT_ORDER:
        raise RunnerError(f"unknown split: {split}")
    if workers < 1 or workers > 8:
        raise RunnerError("workers must be between 1 and 8")
    if limit is not None and limit < 1:
        raise RunnerError("limit must be positive")
    if smoke and (split != "validation" or limit != 1):
        raise RunnerError("smoke runs require --split validation and --limit 1")
    if smoke_instance_id is not None and not smoke:
        raise RunnerError("--smoke-instance-id requires --smoke")
    command_runner = command_runner or _run_isolated
    source_checker = source_checker or check_source_regions
    output_root = output_root.resolve()
    _private_dir(output_root)
    if smoke:
        output_root = output_root / "smoke"
        _private_dir(output_root)
    elif split != "heldout":
        for prior in SPLIT_ORDER[: SPLIT_ORDER.index(split)]:
            _verify_prior_split(output_root, prior)
            _verify_split_evaluation(output_root / prior, prior)

    tasks = _read_tasks(tasks_path)
    official_tasks_sha256 = _official_tasks_sha256(split, tasks_path, tasks)
    entries = _build_input_manifest(tasks, pdf_root)
    split_dir = output_root / split
    _private_dir(split_dir)
    manifest = _prepare_manifest(
        split_dir,
        split,
        entries,
        tasks_path,
        pdf_root,
        official_tasks_sha256,
        resume=resume,
    )
    _write_json(split_dir / "schema.json", OUTPUT_SCHEMA)
    if smoke_instance_id is not None:
        selected = [entry for entry in entries if entry["task"]["instance_id"] == smoke_instance_id]
        if not selected:
            raise RunnerError("--smoke-instance-id is not present in the official manifest")
    else:
        selected = entries[:limit] if limit is not None else entries
    succeeded = 0
    uses_paddle_pool = page_loader is load_pages
    if uses_paddle_pool:
        _start_ocr_pool(manifest["command_config"]["ocr"])
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    split,
                    entry,
                    split_dir,
                    manifest,
                    resume,
                    page_loader,
                    command_runner,
                    source_checker,
                )
                for entry in selected
            ]
            for future in as_completed(futures):
                if future.result():
                    succeeded += 1
    finally:
        if uses_paddle_pool:
            _shutdown_ocr_pool()
    failed = len(selected) - succeeded
    failure_summary = _failure_summary(split_dir, selected)
    _write_json(split_dir / "failures.json", failure_summary)
    selected_records = [
        record
        for entry in selected
        if (
            record := _verified_cached_record(
                split_dir / "jobs" / entry["task"]["instance_id"],
                entry,
                manifest["command_config_sha256"],
                manifest["command_config"],
            )
        )
        is not None
    ]
    fully_grounded = sum(record["source_status"] == "fully_grounded" for record in selected_records)
    evidence_unresolved = sum(
        record["source_status"] == "evidence_unresolved" for record in selected_records
    )

    records = _all_records(
        split_dir,
        entries,
        manifest["command_config_sha256"],
        manifest["command_config"],
    )
    if not smoke and records is not None and len(records) == len(entries):
        export_manifest = export_records(
            records,
            [entry["task"] for entry in entries],
            split_dir,
            source_pages_by_id={
                record["instance_id"]: record["source_pages"] for record in records
            },
            expected_split=split,
            source_checks_by_id={
                record["instance_id"]: record["provenance"]["source_checks"] for record in records
            },
        )
        completion = {
            "status": "generation_complete",
            "coverage_complete": True,
            "split": split,
            "total": len(entries),
            "input_manifest_sha256": manifest["input_manifest_sha256"],
            "command_config_sha256": manifest["command_config_sha256"],
            "record_ids_sha256": _sha256_bytes(
                _canonical_bytes([record["instance_id"] for record in records])
            ),
            "export_manifest_sha256": _sha256_file(split_dir / "manifest.json"),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        for field in (
            "answer_coverage",
            "fully_grounded_count",
            "evidence_unresolved_count",
            "runtime_failed_count",
            "submission_ready",
            "submission_mode",
            "submission_count",
            "grounded_submission_count",
            "submission_abstention_count",
        ):
            completion[field] = export_manifest[field]
        _write_json(split_dir / "complete.json", completion)
    else:
        (split_dir / "complete.json").unlink(missing_ok=True)
    return {
        "split": split,
        "total": len(selected),
        "succeeded": succeeded,
        "failed": failed,
        "fully_grounded": fully_grounded,
        "evidence_unresolved": evidence_unresolved,
        "error_counts": failure_summary["error_counts"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLIT_ORDER)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--pdf-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-instance-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output_root = validate_cli_output_root(args.output_root)
        result = run_pipeline(
            split=args.split,
            tasks_path=args.tasks,
            pdf_root=args.pdf_root,
            output_root=output_root,
            limit=args.limit,
            workers=args.workers,
            resume=args.resume,
            smoke=args.smoke,
            smoke_instance_id=args.smoke_instance_id,
        )
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
