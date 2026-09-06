#!/usr/bin/env python3
"""Generate private, auditable solution records from one public split."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from PIL import Image

from docinsights_analysis.solution_records import (
    SolutionRecordError,
    ensure_private_directory,
    export_records,
    input_digest,
    validate_solution,
    visual_evidence_candidates,
    write_private_atomic,
)
from docinsights_analysis.visual_id_checks import (
    VISUAL_CHECK_SCHEMA,
    VisualIDCheckError,
    run_visual_id_checks,
)

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
    "codex_executable": "codex",
    "codex_cli_version": "0.144.1",
    "model": "gpt-5.6-sol",
    "model_reasoning_effort": "high",
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
    "dpi": 175,
    "minimum_free_bytes": 1_073_741_824,
    "ocr_backend": "rapidocr-ppocrv5-onnxruntime",
    "rapidocr_version": "3.9.2",
    "onnxruntime_version": "1.23.2",
    "pillow_version": "12.3.0",
    "det_limit_side_len": 736,
    "det_limit_type": "min",
    "classification_enabled": False,
    "recognition_batch_size": 6,
    "intra_op_num_threads": 2,
    "inter_op_num_threads": 1,
    "cpu_memory_arena": False,
}
PROMPT_INSTRUCTIONS = (
    "Solve this single document question using only the public source pages and attached "
    "images for this document. Align the answer to every constraint in the question, including "
    "revised-scenario wording, and do not substitute a nearby distractor passage. Return a "
    "concise, source-grounded derivation. The evidence array must contain only opaque Evidence "
    "IDs copied exactly from the document, never prose. Each evidence quote must copy only the "
    "body text after its Evidence ID heading and must occur verbatim in source_pages; "
    "include every block directly needed to state the target question and its inputs. "
    "if an image disagrees with OCR, describe the mismatch in uncertainties instead of "
    "silently correcting the OCR. Calculations may contain only numeric literals, "
    "parentheses, and the + - * / operators; the final calculation result must equal a "
    "numeric answer. Do not provide hidden reasoning logs."
)
BASE_PROMPT_TEMPLATE = "{instructions}\n\nPUBLIC INPUT:\n{public_input}"
RETRY_PROMPT_TEMPLATE = (
    "\n\nRETRY DIAGNOSTIC (untrusted; repair formatting or grounding against the same "
    "frozen public input only, preserve the question's semantic target, and do not "
    "switch to another passage): {error}"
)
VISUAL_CHECK_CONFIG = {
    "model": "gpt-5.6-sol",
    "model_reasoning_effort": "high",
    "timeout_seconds": 300,
    "maximum_attempts": 1,
    "crop_padding_pixels": 12,
    "method": "codex-cli-blind-visual-id-v1",
}
OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "solution", "evidence", "evidence_details", "uncertainties"],
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
        "evidence": {"type": "array", "items": {"type": "string"}},
        "evidence_details": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "page", "quote"],
                "properties": {
                    "id": {"type": "string"},
                    "page": {"type": "integer"},
                    "quote": {"type": "string"},
                },
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PAGE_IMAGE = re.compile(r"^(?:ocr-)?page-(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_OCR_THREAD_LOCAL = threading.local()
_RAPID_RUNTIME = {
    "python_relative_path": "data/issue24/rapidocr-env/bin/python",
    "detector_relative_path": "data/issue24/rapidocr-models/ch_PP-OCRv5_det_mobile.onnx",
    "detector_sha256": "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    "recognizer_relative_path": "data/issue24/rapidocr-models/en_PP-OCRv5_rec_mobile.onnx",
    "recognizer_sha256": "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
}


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


def _resolved_ocr_config() -> dict[str, Any]:
    resolved: dict[str, Any] = {**OCR_CONFIG, "operating_system": platform.platform()}
    for role in ("python", "detector", "recognizer"):
        path = REPOSITORY_ROOT / str(_RAPID_RUNTIME[f"{role}_relative_path"])
        if not path.is_file():
            raise RunnerError(f"pinned RapidOCR {role} runtime artifact is missing or changed")
        observed = _sha256_file(path)
        expected = _RAPID_RUNTIME.get(f"{role}_sha256")
        if expected is not None and observed != expected:
            raise RunnerError(f"pinned RapidOCR {role} runtime artifact is missing or changed")
        resolved[f"{role}_path"] = str(path.absolute())
        resolved[f"{role}_sha256"] = observed
    lock_path = REPOSITORY_ROOT / "requirements" / "solution-ocr.txt"
    distributions = _locked_distribution_records(lock_path)
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
    resolved.update(
        {
            "renderer_path": str(renderer_path.resolve()),
            "renderer_sha256": _sha256_file(renderer_path),
            "renderer_version": renderer_version,
            "runtime_lock_path": str(lock_path.resolve()),
            "runtime_lock_sha256": _sha256_file(lock_path),
            "runtime_distributions": distributions,
        }
    )
    return resolved


def _locked_distribution_records(lock_path: Path) -> list[dict[str, str]]:
    site_packages = (
        REPOSITORY_ROOT
        / "data"
        / "issue24"
        / "rapidocr-env"
        / "lib"
        / "python3.11"
        / "site-packages"
    )
    installed: dict[tuple[str, str], Path] = {}
    for metadata_path in site_packages.glob("*.dist-info/METADATA"):
        fields: dict[str, str] = {}
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                if key in {"Name", "Version"} and key not in fields:
                    fields[key] = value
            if len(fields) == 2:
                break
        if set(fields) == {"Name", "Version"}:
            installed[(fields["Name"].casefold().replace("_", "-"), fields["Version"])] = (
                metadata_path.parent / "RECORD"
            )
    records = []
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, version = stripped.partition("==")
        if not separator:
            raise RunnerError("solution OCR lock must contain exact versions")
        key = (name.casefold().replace("_", "-"), version)
        record_path = installed.get(key)
        if record_path is None or not record_path.is_file():
            raise RunnerError(f"pinned OCR distribution is missing: {stripped}")
        records.append(
            {
                "name": name,
                "version": version,
                "record_path": str(record_path.resolve()),
                "record_sha256": _sha256_file(record_path),
            }
        )
    return records


def _verify_locked_runtime(config: dict[str, Any]) -> None:
    environment_root = (
        REPOSITORY_ROOT
        / "data"
        / "issue24"
        / "rapidocr-env"
    ).resolve()
    site_packages = (
        environment_root
        / "lib"
        / "python3.11"
        / "site-packages"
    ).resolve()
    lock_path = Path(config["runtime_lock_path"])
    if not lock_path.is_file() or _sha256_file(lock_path) != config["runtime_lock_sha256"]:
        raise RunnerError("pinned OCR runtime lock changed")
    for role in ("python", "renderer", "detector", "recognizer"):
        path = Path(config[f"{role}_path"])
        if not path.is_file() or _sha256_file(path) != config[f"{role}_sha256"]:
            raise RunnerError(f"pinned OCR {role} artifact changed")
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
                if not member.is_file():
                    raise RunnerError(f"pinned OCR runtime member is missing: {relative_name}")
                expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
                if _sha256_file(member) != expected or (
                    size_field and member.stat().st_size != int(size_field)
                ):
                    raise RunnerError(f"pinned OCR runtime member changed: {relative_name}")


def _rapidocr_engine() -> Any:
    existing = getattr(_OCR_THREAD_LOCAL, "engine", None)
    if existing is not None:
        return existing
    config = _resolved_ocr_config()
    if Path(sys.executable).absolute() != Path(config["python_path"]).absolute():
        raise RunnerError(f"RapidOCR runner must use pinned Python: {config['python_path']}")
    try:
        rapidocr = importlib.import_module("rapidocr")
    except ImportError as error:
        raise RunnerError("pinned RapidOCR runtime is unavailable") from error
    params = {
        "Global.use_cls": False,
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": config["intra_op_num_threads"],
        "EngineConfig.onnxruntime.inter_op_num_threads": config["inter_op_num_threads"],
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": config["cpu_memory_arena"],
        "Det.engine_type": rapidocr.EngineType.ONNXRUNTIME,
        "Det.ocr_version": rapidocr.OCRVersion.PPOCRV5,
        "Det.lang_type": rapidocr.LangDet.CH,
        "Det.model_type": rapidocr.ModelType.MOBILE,
        "Det.model_path": config["detector_path"],
        "Det.limit_side_len": config["det_limit_side_len"],
        "Det.limit_type": config["det_limit_type"],
        "Rec.engine_type": rapidocr.EngineType.ONNXRUNTIME,
        "Rec.ocr_version": rapidocr.OCRVersion.PPOCRV5,
        "Rec.lang_type": rapidocr.LangRec.EN,
        "Rec.model_type": rapidocr.ModelType.MOBILE,
        "Rec.model_path": config["recognizer_path"],
        "Rec.rec_batch_num": config["recognition_batch_size"],
    }
    engine = rapidocr.RapidOCR(params=params)
    _OCR_THREAD_LOCAL.engine = engine
    return engine


def _rapidocr_page(
    engine: Any, image: Path, page_number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = engine(image, use_cls=False)
    raw_boxes: Any = [] if result.boxes is None else result.boxes
    tolist = getattr(raw_boxes, "tolist", None)
    boxes = cast(list[Any], tolist() if callable(tolist) else raw_boxes)
    texts = [] if result.txts is None else list(result.txts)
    scores = [] if result.scores is None else [float(value) for value in result.scores]
    if not (len(boxes) == len(texts) == len(scores)):
        raise RunnerError("RapidOCR returned misaligned boxes, text, and confidence values")
    try:
        height, width = int(result.img.shape[0]), int(result.img.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise RunnerError("RapidOCR returned invalid source image dimensions") from error
    lines = []
    for line_index, (quad, text, confidence) in enumerate(
        zip(boxes, texts, scores, strict=True)
    ):
        try:
            points = [[float(point[0]), float(point[1])] for point in quad]
        except (IndexError, TypeError, ValueError) as error:
            raise RunnerError("RapidOCR returned an invalid text quadrilateral") from error
        if len(points) != 4:
            raise RunnerError("RapidOCR text quadrilateral must contain four points")
        left = max(0, math.floor(min(point[0] for point in points)))
        top = max(0, math.floor(min(point[1] for point in points)))
        right = min(width, math.ceil(max(point[0] for point in points)))
        bottom = min(height, math.ceil(max(point[1] for point in points)))
        if left >= right or top >= bottom:
            raise RunnerError("RapidOCR returned an empty or out-of-bounds text box")
        lines.append(
            {
                "line_index": line_index,
                "text": str(text),
                "confidence": confidence,
                "quadrilateral": points,
                "bbox": {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
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
    """OCR lossless renders with pinned RapidOCR and retain derived solver JPEGs."""
    _private_dir(cache_dir)
    if shutil.disk_usage(cache_dir).free < OCR_CONFIG["minimum_free_bytes"]:
        raise RunnerError("insufficient free disk space for retained page images")
    ocr_provenance = _resolved_ocr_config()
    _verify_locked_runtime(ocr_provenance)
    prefix = cache_dir / "ocr-page"
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
        engine = _rapidocr_engine()
        for image in ocr_images:
            page_number = _page_image_number(image)
            try:
                page, page_geometry = _rapidocr_page(engine, image, page_number)
            except Exception as error:  # noqa: BLE001 - normalize pinned OCR failures
                raise RunnerError(f"RapidOCR failed on page {page_number}: {error}") from error
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
    return {
        **COMMAND_CONFIG,
        **codex_runtime,
        "prompt_instructions_sha256": _sha256_bytes(PROMPT_INSTRUCTIONS.encode("utf-8")),
        "base_prompt_template_sha256": _sha256_bytes(BASE_PROMPT_TEMPLATE.encode("utf-8")),
        "retry_prompt_template_sha256": _sha256_bytes(RETRY_PROMPT_TEMPLATE.encode("utf-8")),
        "output_schema_sha256": _sha256_bytes(_canonical_bytes(OUTPUT_SCHEMA)),
        "ocr": _resolved_ocr_config(),
        "visual_id_checker": _resolved_visual_check_config(codex_runtime),
    }


def _resolved_codex_runtime() -> dict[str, str]:
    executable_value = shutil.which(str(COMMAND_CONFIG["codex_executable"]))
    if executable_value is None:
        raise RunnerError("configured Codex executable is unavailable")
    executable = Path(executable_value).resolve()
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
    }


def _verify_codex_runtime(config: dict[str, Any]) -> None:
    executable = Path(config["codex_executable_path"])
    if not executable.is_file() or _sha256_file(executable) != config["codex_executable_sha256"]:
        raise RunnerError("pinned Codex executable changed")
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


def _resolved_visual_check_config(codex_runtime: dict[str, str]) -> dict[str, Any]:
    module_path = REPOSITORY_ROOT / "src" / "docinsights_analysis" / "visual_id_checks.py"
    if not module_path.is_file():
        raise RunnerError("visual ID checker implementation is missing")
    return {
        **VISUAL_CHECK_CONFIG,
        **codex_runtime,
        "module_path": str(module_path.resolve()),
        "module_sha256": _sha256_file(module_path),
        "output_schema_sha256": _sha256_bytes(_canonical_bytes(VISUAL_CHECK_SCHEMA)),
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
    for name in (*required, "response.json", "primary-response.json"):
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
            if attempt.get("visual_artifacts", []) != _directory_artifacts(
                attempt_dir / "visual-id-checks"
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
    if job_input.get("command_config_sha256") != command_config_sha256:
        return None
    images = job_input.get("page_images")
    if not isinstance(images, list) or not images:
        return None
    source_dir = (job_dir / "source").resolve()
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("path"), str):
            return None
        image_path = Path(image["path"])
        try:
            image_path.resolve().relative_to(source_dir)
        except ValueError:
            return None
        if not image_path.is_file() or image.get("sha256") != _sha256_file(image_path):
            return None
    if not isinstance(job_input.get("source_pages"), list):
        return None
    source_pages_path = job_dir / "source" / "source_pages.json"
    images_path = job_dir / "source" / "images.json"
    geometry_path = job_dir / "source" / "ocr_geometry.json"
    if (
        not source_pages_path.is_file()
        or not images_path.is_file()
        or not geometry_path.is_file()
    ):
        return None
    if _load_json(source_pages_path, f"{job_dir.name} source pages") != job_input[
        "source_pages"
    ]:
        return None
    if _load_json(images_path, f"{job_dir.name} image inventory") != images:
        return None
    geometry = job_input.get("ocr_geometry")
    if not isinstance(geometry, list) or _load_json(
        geometry_path, f"{job_dir.name} OCR geometry"
    ) != geometry:
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


def _verified_visual_checks(
    proofs: Any,
    job_dir: Path,
    job_input: dict[str, Any],
    checker_config: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if proofs is None:
        return []
    if not isinstance(proofs, list):
        return None
    expected_config_sha256 = _sha256_bytes(_canonical_bytes(checker_config))
    page_images = {
        image["sha256"]: Path(image["path"])
        for image in job_input["page_images"]
        if isinstance(image, dict)
        and isinstance(image.get("sha256"), str)
        and isinstance(image.get("path"), str)
    }
    visual_root = (job_dir / "attempts").resolve()

    def intact(artifact: Any, *, page_image: bool = False) -> bool:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            return False
        path_value = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            return False
        path = Path(path_value)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            return False
        if page_image:
            return digest in page_images and page_images[digest].resolve() == path.resolve()
        try:
            path.resolve().relative_to(visual_root)
        except ValueError:
            return False
        return "visual-id-checks" in path.parts

    for proof in proofs:
        if not isinstance(proof, dict):
            return None
        if proof.get("checker_config_sha256") != expected_config_sha256:
            return None
        if proof.get("model") != checker_config["model"] or proof.get(
            "method"
        ) != checker_config["method"]:
            return None
        if not intact(proof.get("page_image_artifact"), page_image=True):
            return None
        if not intact(proof.get("crop_artifact")) or not intact(
            proof.get("raw_response_artifact")
        ):
            return None
        artifacts = proof.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts or not all(
            intact(artifact) for artifact in artifacts
        ):
            return None
        if proof.get("raw_response_artifact") not in artifacts:
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
    )
    if job_input is None:
        return None
    pages = cast(list[dict[str, Any]], job_input["source_pages"])
    attempts = state.get("attempts")
    if not _verify_attempt_artifacts(job_dir, attempts):
        return None
    successful_attempts = [
        attempt for attempt in attempts if attempt.get("status") == "succeeded"
    ]
    if len(successful_attempts) != 1:
        return None
    selected_attempt = successful_attempts[0]
    response_path = job_dir / "attempts" / str(selected_attempt["attempt"]) / "response.json"
    if not response_path.is_file():
        return None
    raw_output_sha256 = _sha256_file(response_path)
    if state.get("raw_output_sha256") != raw_output_sha256:
        return None
    record_provenance = record.get("provenance") if isinstance(record, dict) else None
    raw_proofs = (
        record_provenance.get("visual_id_checks")
        if isinstance(record_provenance, dict)
        else None
    )
    visual_checks = _verified_visual_checks(
        raw_proofs, job_dir, job_input, expected_config["visual_id_checker"]
    )
    if visual_checks is None:
        return None
    if state.get("visual_id_checks_sha256") != _sha256_bytes(
        _canonical_bytes(visual_checks)
    ):
        return None
    try:
        normalized = validate_solution(
            record,
            entry["task"],
            pages,
            visual_id_checks=visual_checks or None,
        )
        generated = json.loads(response_path.read_text(encoding="utf-8"))
        raw_record = dict(normalized)
        for field in (
            "solution",
            "answer",
            "evidence",
            "evidence_details",
            "uncertainties",
        ):
            raw_record[field] = generated[field]
        normalized_raw = validate_solution(
            raw_record,
            entry["task"],
            pages,
            visual_id_checks=visual_checks or None,
        )
    except (json.JSONDecodeError, KeyError, SolutionRecordError, TypeError):
        return None
    for field in ("solution", "answer", "evidence", "evidence_details", "uncertainties"):
        if normalized_raw[field] != normalized[field]:
            return None
    provenance = normalized["provenance"]
    if provenance.get("pdf_sha256") != entry["pdf_sha256"]:
        return None
    if provenance.get("config_sha256") != command_config_sha256:
        return None
    if provenance.get("output_sha256") != raw_output_sha256:
        return None
    if provenance.get("ocr") != expected_config["ocr"]:
        return None
    return normalized


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
        or complete.get("split") != split
        or complete.get("total") != len(entries)
        or complete.get("input_manifest_sha256")
        != run_manifest["input_manifest_sha256"]
        or complete.get("command_config_sha256")
        != run_manifest["command_config_sha256"]
        or complete.get("record_ids_sha256")
        != _sha256_bytes(
            _canonical_bytes([entry["task"]["instance_id"] for entry in entries])
        )
        or complete.get("export_manifest_sha256") != _sha256_file(export_manifest_path)
    ):
        raise RunnerError(f"{split} does not have a complete export binding")
    if (
        export_manifest.get("private") is not True
        or export_manifest.get("split") != split
        or export_manifest.get("total") != len(entries)
        or export_manifest.get("instance_ids") != expected_ids
    ):
        raise RunnerError(f"{split} does not have complete export coverage")
    files = export_manifest.get("files")
    expected_files = {"solutions.jsonl", "solutions.md", "submission.jsonl"}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise RunnerError(f"{split} does not have a complete export file manifest")
    for name in expected_files:
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
    expected_submission = [
        {
            "instance_id": record["instance_id"],
            "answer": record["answer"],
            "evidence": record["evidence"],
        }
        for record in records
    ]
    if _read_jsonl(
        split_dir / "submission.jsonl", f"{split} submission export"
    ) != expected_submission:
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
    expected = _manifest_payload(
        split, entries, tasks_path, pdf_root, official_tasks_sha256
    )
    path = split_dir / "run-manifest.json"
    if path.exists():
        if not resume:
            raise RunnerError(f"output already exists for {split}; use --resume")
        actual = _load_json(path, "run manifest")
        if actual.get("command_config_sha256") != expected["command_config_sha256"] or actual.get(
            "command_config"
        ) != expected["command_config"]:
            raise RunnerError("resume refused: command config hash mismatch")
        if actual.get("input_manifest_sha256") != expected["input_manifest_sha256"] or actual.get(
            "tasks"
        ) != entries:
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


def _invoke_visual_checker(
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
    completed = command_runner(
        argv,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=checker_config["timeout_seconds"],
        check=False,
    )
    events = completed.stdout or ""
    stderr = completed.stderr or ""
    _write_private(output_dir / "events.jsonl", events)
    _write_private(output_dir / "stderr.txt", stderr)
    turn_completed, _ = _event_summary(events)
    if completed.returncode != 0 or not turn_completed:
        raise VisualIDCheckError("visual checker did not complete successfully")
    if not response_path.is_file() or not response_path.read_text(encoding="utf-8").strip():
        raise VisualIDCheckError("visual checker returned an empty response")
    response_path.chmod(0o600)
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise VisualIDCheckError("visual checker returned invalid JSON") from error
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
    if isinstance(error, VisualIDCheckError):
        return "needs_visual_review"
    message = str(error)
    if "quote does not belong to evidence block" in message:
        return "needs_visual_review"
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, (json.JSONDecodeError, KeyError, SolutionRecordError)):
        return "invalid_response"
    return "runtime_error"


def _run_one(
    split: str,
    entry: dict[str, Any],
    split_dir: Path,
    manifest: dict[str, Any],
    resume: bool,
    page_loader: PageLoader,
    command_runner: CommandRunner,
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
            return True
        error_path = job_dir / "error.json"
        if error_path.is_file():
            previous_error = _load_json(error_path, f"{instance_id} error")
            previous_attempts = previous_error.get("attempts")
            if (
                previous_error.get("manifest_entry_sha256") == entry_hash
                and previous_error.get("command_config_sha256")
                == manifest["command_config_sha256"]
                and isinstance(previous_attempts, list)
                and (
                    len(previous_attempts) >= command_config["maximum_attempts"]
                    or previous_error.get("terminal") is True
                )
                and _verify_attempt_artifacts(job_dir, previous_attempts)
                and _verified_job_input(
                    job_dir,
                    entry,
                    manifest["command_config_sha256"],
                    previous_error.get("job_input_sha256"),
                    command_config["ocr"],
                )
                is not None
            ):
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
            record = {
                "instance_id": instance_id,
                "split": split,
                "question": task["user_query"],
                "solution": generated["solution"],
                "answer": generated["answer"],
                "evidence": generated["evidence"],
                "evidence_details": generated["evidence_details"],
                "source_pages": bundle.pages,
                "provenance": {
                    "pdf_sha256": entry["pdf_sha256"],
                    "input_sha256": public_input_hash,
                    "output_sha256": _sha256_bytes(raw_response),
                    "config_sha256": manifest["command_config_sha256"],
                    "model": command_config["model"],
                    "method": "codex-cli-schema-v1",
                    "ocr": job_input["ocr"],
                    "attempt": attempt_number,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "page_images": image_metadata,
                    "usage": usage,
                },
                "uncertainties": generated["uncertainties"],
            }
            visual_checks: list[dict[str, Any]] | None = None
            if visual_evidence_candidates(record, bundle.pages):
                visual_checks = run_visual_id_checks(
                    frozen_record=record,
                    pages=bundle.pages,
                    paddle_geometry=bundle.geometry,
                    retained_images=image_metadata,
                    output_dir=attempt_dir / "visual-id-checks",
                    checker_config=command_config["visual_id_checker"],
                    invoke=lambda **kwargs: _invoke_visual_checker(
                        **kwargs, command_runner=command_runner
                    ),
                )
                record["provenance"]["visual_id_checks"] = visual_checks
            normalized = validate_solution(
                record,
                task,
                bundle.pages,
                visual_id_checks=visual_checks,
            )
            _write_json(record_path, normalized)
            state = {
                "instance_id": instance_id,
                "status": "succeeded",
                "manifest_entry_sha256": entry_hash,
                "command_config_sha256": manifest["command_config_sha256"],
                "job_input_sha256": job_input_hash,
                "raw_output_sha256": _sha256_bytes(raw_response),
                "visual_id_checks_sha256": _sha256_bytes(
                    _canonical_bytes(visual_checks or [])
                ),
                "record_sha256": _sha256_file(record_path),
                "attempts": attempts
                + [
                    {
                        "attempt": attempt_number,
                        "status": "succeeded",
                        "usage": usage,
                        "artifacts": _attempt_artifacts(attempt_dir),
                        "visual_artifacts": _directory_artifacts(
                            attempt_dir / "visual-id-checks"
                        ),
                    }
                ],
                "timestamp": datetime.now(UTC).isoformat(),
            }
            _write_json(state_path, state)
            (job_dir / "error.json").unlink(missing_ok=True)
            return True
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
                    "visual_artifacts": _directory_artifacts(
                        attempt_dir / "visual-id-checks"
                    ),
                }
            )
            last_error = error
            if isinstance(error, VisualIDCheckError):
                break

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
        "terminal": isinstance(last_error, VisualIDCheckError),
        "timestamp": datetime.now(UTC).isoformat(),
    }
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
    page_loader: PageLoader = load_pages,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    if split not in SPLIT_ORDER:
        raise RunnerError(f"unknown split: {split}")
    if workers < 1 or workers > 4:
        raise RunnerError("workers must be between 1 and 4")
    if limit is not None and limit < 1:
        raise RunnerError("limit must be positive")
    if smoke and (split != "validation" or limit != 1):
        raise RunnerError("smoke runs require --split validation and --limit 1")
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
    selected = entries[:limit] if limit is not None else entries
    succeeded = 0
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
            )
            for entry in selected
        ]
        for future in as_completed(futures):
            if future.result():
                succeeded += 1
    failed = len(selected) - succeeded
    failure_summary = _failure_summary(split_dir, selected)
    _write_json(split_dir / "failures.json", failure_summary)

    records = _all_records(
        split_dir,
        entries,
        manifest["command_config_sha256"],
        manifest["command_config"],
    )
    if not smoke and records is not None and len(records) == len(entries):
        export_records(
            records,
            [entry["task"] for entry in entries],
            split_dir,
            source_pages_by_id={
                record["instance_id"]: record["source_pages"] for record in records
            },
            expected_split=split,
            visual_checks_by_id={
                record["instance_id"]: record["provenance"]["visual_id_checks"]
                for record in records
                if "visual_id_checks" in record["provenance"]
            },
        )
        completion = {
            "status": "generation_complete",
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
        _write_json(split_dir / "complete.json", completion)
    else:
        (split_dir / "complete.json").unlink(missing_ok=True)
    return {
        "split": split,
        "total": len(selected),
        "succeeded": succeeded,
        "failed": failed,
        "error_counts": failure_summary["error_counts"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLIT_ORDER)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--pdf-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
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
        )
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
