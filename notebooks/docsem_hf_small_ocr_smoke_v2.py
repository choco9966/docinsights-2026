"""Single-cell Kaggle runner for a fixed, label-blind DocSem OCR preflight.

Paste this entire file into one Kaggle Python cell, or execute it as a script.  The
parent process renders the only approved PDF and launches every model in a fresh,
sequential child process.  The output directory is a self-describing evidence DAG;
no task, query, label, answer, or reference-evidence file is opened.

The fixed 512-token ceiling is intentionally preserved for comparison with the
original smoke.  It is a preflight ceiling, not proof that a document was completely
transcribed, so every result explicitly carries a truncation-risk marker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.0"
INSTANCE_ID = "task_000909"
EXPECTED_PDF_SHA256 = "fa54e0a898b530757c8419524552d00dcb60bb9449e92b78c3c8c6ce3d82b798"
EXPECTED_PAGE_SHA256 = (
    "8f6d64c96a7b7434ea95612b983e986838598f88d5fbf63c186b049727d9d12b",
    "26644f2ecedd9ab14976d13f68e3103797058a06a2c10bebff03d8bad6cae2cd",
)
INPUT_CANDIDATES = (
    Path(
        "/kaggle/input/datasets/chocozzz/docsem-validation-ocr-input/"
        "bundle/documents-root/val/documents/task_000909.pdf"
    ),
    Path("/kaggle/input/docsem-validation-ocr-input/task_000909.pdf"),
    Path("/kaggle/input/docsem-validation-ocr-input/documents/task_000909.pdf"),
    Path("/kaggle/input/docsem-validation-ocr-input/val/documents/task_000909.pdf"),
)
OUT = Path("/kaggle/working/docinsights-hf-smoke-v2")
MAX_NEW_TOKENS = 512
DTYPE = "float16"
DEVICE = "cuda:0"
RENDER_DPI = 200

# Kaggle supplies its CUDA-enabled torch build.  All other direct runtime packages
# are installed without dependency resolution so the executed set is exact.  Torch
# is asserted separately because the +cu128 wheel comes from Kaggle's base image.
EXPECTED_RUNTIME = {
    "torch": "2.10.0+cu128",
    "transformers": "5.12.1",
    "accelerate": "1.10.1",
    "huggingface-hub": "1.5.0",
    "psutil": "7.0.0",
    "Pillow": "10.4.0",
}
PIP_REQUIREMENTS = (
    "transformers==5.12.1",
    "accelerate==1.10.1",
    "huggingface-hub==1.5.0",
    "psutil==7.0.0",
    "Pillow==10.4.0",
)

# Add another replacement by appending one dict with this same shape.  `gate` and
# `selection_status` are evidence-only: GLM remains executable, but its known
# diagnostic gate outcome is recorded without suppressing the measured invocation.
SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "PaddleOCR-VL-1.6",
        "repo": "PaddlePaddle/PaddleOCR-VL-1.6",
        "revision": "c5630abae1d940eafe0697512a0325494b02ab42",
        "prompt": "OCR:",
        "trust_remote_code": True,
        "gate": "candidate",
        "selection_status": "existing_candidate",
        "model_class": "AutoModelForImageTextToText",
        "input_mode": "paddle_chat",
    },
    {
        "name": "GLM-OCR",
        "repo": "zai-org/GLM-OCR",
        "revision": "ca5d8b3e287e52589e37c28385d9655ee4372f9d",
        "prompt": "Text Recognition:",
        "trust_remote_code": False,
        "gate": "diagnostic_gate_fail",
        "selection_status": "not_selected_diagnostic_gate_fail",
        "model_class": "AutoModelForImageTextToText",
        "input_mode": "glm_chat",
    },
    {
        "name": "surya-ocr-2",
        "repo": "datalab-to/surya-ocr-2",
        "revision": "3b3d4cdf88d6928b0acdc75181b13206ea67c4a3",
        "prompt": (
            "OCR this image to HTML. Each block is a div with data-label and "
            "data-bbox (x0 y0 x1 y1, normalized 0-1000)."
        ),
        "trust_remote_code": False,
        "gate": "candidate",
        "selection_status": "existing_candidate",
        "model_class": "AutoModelForImageTextToText",
        "input_mode": "surya_chat",
    },
    {
        "name": "granite-docling-258M",
        "repo": "ibm-granite/granite-docling-258M",
        "revision": "982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe",
        "prompt": "Convert this page to DocTags.",
        "trust_remote_code": False,
        "gate": "candidate",
        "selection_status": "existing_candidate",
        "model_class": "AutoModelForImageTextToText",
        "input_mode": "docling_chat",
    },
    {
        "name": "SmolDocling-256M-preview",
        "repo": "docling-project/SmolDocling-256M-preview",
        "revision": "ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8",
        "prompt": "Convert this page to docling.",
        "trust_remote_code": False,
        "gate": "replacement_candidate",
        "selection_status": "replacement_candidate",
        "model_class": "AutoModelForImageTextToText",
        "input_mode": "docling_chat",
    },
)

RESULT_FIELDS = (
    "schema_version",
    "instance_id",
    "model_index",
    "name",
    "repo",
    "requested_revision",
    "resolved_revision",
    "revision_resolution",
    "repo_file_metadata",
    "repo_metadata_error",
    "prompt",
    "max_new_tokens",
    "fixed_preflight",
    "truncation_risk",
    "dtype",
    "device",
    "trust_remote_code",
    "gate",
    "selection_status",
    "model_class",
    "input_mode",
    "status",
    "success",
    "exit_code",
    "load_sec",
    "page_latency_sec",
    "doc_latency_sec",
    "child_wall_sec",
    "parent_wall_sec",
    "peak_process_rss_bytes_child",
    "peak_process_rss_bytes_parent_sampled",
    "peak_process_rss_sampling_error",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "peak_vram_bytes_parent_sampled",
    "peak_vram_sampling_error",
    "raw_outputs",
    "raw_output_bytes",
    "raw_output_sha256",
    "stdout_path",
    "stdout_bytes",
    "stdout_sha256",
    "stderr_path",
    "stderr_bytes",
    "stderr_sha256",
    "error",
    "traceback",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    indent = 2 if pretty else None
    return (json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json_bytes(value))


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def exact_executed_source() -> tuple[bytes, str, str]:
    """Return exact source bytes, acquisition mode, and the bootstrap caveat."""
    file_name = globals().get("__file__")
    if file_name and not str(file_name).startswith("<"):
        source_path = Path(str(file_name)).resolve()
        if source_path.is_file():
            return source_path.read_bytes(), "script_file", "none"
    try:
        shell = get_ipython()  # type: ignore[name-defined]  # noqa: F821
        raw = shell.history_manager.input_hist_raw[-1]
    except Exception as exc:  # pragma: no cover - only reachable in unusual notebook shells
        raise RuntimeError("cannot recover the exact executed notebook-cell source") from exc
    return (
        raw.encode("utf-8"),
        "ipython_raw_cell_history",
        (
            "IPython exposes the submitted raw cell, not the notebook JSON container; "
            "the recorded SHA covers exactly the Python source submitted for execution."
        ),
    )


def install_and_verify_runtime() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-deps",
        *PIP_REQUIREMENTS,
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    observed: dict[str, str | None] = {}
    for package in EXPECTED_RUNTIME:
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed[package] = None
    mismatches = {
        package: {"expected": expected, "observed": observed[package]}
        for package, expected in EXPECTED_RUNTIME.items()
        if observed[package] != expected
    }
    record = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "expected": EXPECTED_RUNTIME,
        "observed": observed,
        "mismatches": mismatches,
    }
    write_json(OUT / "bootstrap.json", record)
    if completed.returncode or mismatches:
        raise RuntimeError(f"pinned runtime bootstrap failed: {mismatches or completed.stderr}")
    return record


def select_fixed_pdf() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.is_file():
            observed = sha256_file(candidate)
            if observed != EXPECTED_PDF_SHA256:
                raise RuntimeError(
                    f"fixed PDF SHA mismatch: expected {EXPECTED_PDF_SHA256}, observed {observed}"
                )
            return candidate
    raise FileNotFoundError("task_000909 PDF is not mounted at either approved exact path")


def poppler_facts() -> dict[str, Any]:
    binary_text = shutil.which("pdftoppm")
    if binary_text is None:
        raise FileNotFoundError("pdftoppm is required")
    binary = Path(binary_text).resolve()
    version = subprocess.run([str(binary), "-v"], text=True, capture_output=True, check=False)
    if version.returncode:
        raise RuntimeError(f"pdftoppm -v failed: {version.stderr}")
    return {
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary),
        "version_stdout": version.stdout,
        "version_stderr": version.stderr,
        "version_exit_code": version.returncode,
        "dpi": RENDER_DPI,
    }


def render_fixed_pages(pdf: Path, poppler: dict[str, Any]) -> list[Path]:
    pages_dir = OUT / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for page_number in (1, 2):
        output = pages_dir / f"{INSTANCE_ID}-page-{page_number}.png"
        if output.exists():
            output.unlink()
        prefix = output.with_suffix("")
        command = [
            poppler["binary_path"],
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-r",
            str(RENDER_DPI),
            "-png",
            str(pdf),
            str(prefix),
        ]
        subprocess.run(command, capture_output=True, check=True)
        if not output.is_file():
            raise RuntimeError(f"Poppler did not create page {page_number}")
        paths.append(output)
    observed = tuple(sha256_file(path) for path in paths)
    if observed != EXPECTED_PAGE_SHA256:
        raise RuntimeError(
            f"fixed rendered-page SHA mismatch: expected {EXPECTED_PAGE_SHA256}, observed {observed}"
        )
    return paths


def empty_result(spec: dict[str, Any], model_index: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": INSTANCE_ID,
        "model_index": model_index,
        "name": spec["name"],
        "repo": spec["repo"],
        "requested_revision": spec["revision"],
        "resolved_revision": None,
        "revision_resolution": None,
        "repo_file_metadata": [],
        "repo_metadata_error": None,
        "prompt": spec["prompt"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "fixed_preflight": True,
        "truncation_risk": (
            "possible: max_new_tokens=512 is a fixed comparable preflight ceiling; "
            "token-limit completion does not establish full-page coverage"
        ),
        "dtype": DTYPE,
        "device": DEVICE,
        "trust_remote_code": spec["trust_remote_code"],
        "gate": spec["gate"],
        "selection_status": spec["selection_status"],
        "model_class": spec["model_class"],
        "input_mode": spec["input_mode"],
        "status": "not_started",
        "success": False,
        "exit_code": None,
        "load_sec": None,
        "page_latency_sec": [],
        "doc_latency_sec": None,
        "child_wall_sec": None,
        "parent_wall_sec": None,
        "peak_process_rss_bytes_child": None,
        "peak_process_rss_bytes_parent_sampled": None,
        "peak_process_rss_sampling_error": None,
        "peak_cuda_allocated_bytes": None,
        "peak_cuda_reserved_bytes": None,
        "peak_vram_bytes_parent_sampled": None,
        "peak_vram_sampling_error": None,
        "raw_outputs": [],
        "raw_output_bytes": 0,
        "raw_output_sha256": [],
        "stdout_path": None,
        "stdout_bytes": 0,
        "stdout_sha256": None,
        "stderr_path": None,
        "stderr_bytes": 0,
        "stderr_sha256": None,
        "error": None,
        "traceback": None,
    }


def huggingface_repo_metadata(repo: str, revision: str) -> tuple[str | None, list[dict[str, Any]]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = []
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None) or {}
        lfs_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        files.append(
            {
                "rfilename": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs_sha256": lfs_sha256,
                "lfs_size": lfs_size,
            }
        )
    return info.sha, sorted(files, key=lambda row: row["rfilename"])


def child_run(spec: dict[str, Any], model_index: int, pages: list[Path], result_path: Path) -> int:
    record = empty_result(spec, model_index)
    started = time.perf_counter()
    model = None
    processor = None
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is prohibited")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("child must see exactly one GPU via CUDA_VISIBLE_DEVICES=0")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        try:
            resolved, metadata = huggingface_repo_metadata(spec["repo"], spec["revision"])
        except Exception as exc:
            record["repo_metadata_error"] = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(
                f"repository identity metadata failed: {record['repo_metadata_error']}"
            ) from exc
        if resolved != spec["revision"]:
            raise RuntimeError(
                f"resolved revision differs from requested pin: requested={spec['revision']}, resolved={resolved}"
            )
        model_files = [row for row in metadata if row["rfilename"] == "model.safetensors"]
        if (
            len(model_files) != 1
            or not model_files[0]["lfs_sha256"]
            or model_files[0]["lfs_size"] is None
        ):
            raise RuntimeError("model.safetensors LFS identity is missing or ambiguous")
        record["resolved_revision"] = resolved
        record["revision_resolution"] = "huggingface_api_confirmed_commit"
        record["repo_file_metadata"] = metadata

        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            spec["repo"],
            revision=spec["revision"],
            trust_remote_code=spec["trust_remote_code"],
        )
        if spec["model_class"] != "AutoModelForImageTextToText":
            raise ValueError(f"unsupported model_class: {spec['model_class']}")
        model = AutoModelForImageTextToText.from_pretrained(
            spec["repo"],
            revision=spec["revision"],
            trust_remote_code=spec["trust_remote_code"],
            dtype=torch.float16,
            device_map={"": 0},
        ).eval()
        torch.cuda.synchronize(0)
        record["load_sec"] = time.perf_counter() - load_started
        config_commit = getattr(model.config, "_commit_hash", None)
        if config_commit and config_commit != record["resolved_revision"]:
            raise RuntimeError(
                f"loaded config revision differs from verified Hub revision: {config_commit}"
            )

        doc_started = time.perf_counter()
        for page_number, page in enumerate(pages, start=1):
            with Image.open(page) as opened:
                image = opened.convert("RGB")
            if spec["input_mode"] == "glm_chat":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": str(page)},
                            {"type": "text", "text": spec["prompt"]},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(DEVICE)
                inputs.pop("token_type_ids", None)
            elif spec["input_mode"] in {"paddle_chat", "surya_chat"}:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": spec["prompt"]},
                        ],
                    }
                ]
                template_kwargs: dict[str, Any] = {}
                if spec["input_mode"] == "paddle_chat":
                    template_kwargs["images_kwargs"] = {
                        "size": {
                            "shortest_edge": processor.image_processor.min_pixels,
                            "longest_edge": 1280 * 28 * 28,
                        }
                    }
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    **template_kwargs,
                ).to(DEVICE)
            elif spec["input_mode"] == "docling_chat":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": spec["prompt"]},
                        ],
                    }
                ]
                model_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=model_prompt, images=[image], return_tensors="pt").to(
                    DEVICE
                )
            else:
                raise ValueError(f"unsupported input_mode: {spec['input_mode']}")
            torch.cuda.synchronize(0)
            page_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            torch.cuda.synchronize(0)
            record["page_latency_sec"].append(time.perf_counter() - page_started)
            generated_tail = generated[:, inputs["input_ids"].shape[1] :]
            text = processor.batch_decode(
                generated_tail,
                skip_special_tokens=spec["input_mode"] == "surya_chat",
            )[0]
            if spec["input_mode"] == "docling_chat":
                text = text.lstrip()
            raw = text.encode("utf-8")
            raw_path = OUT / "raw" / f"{model_index:02d}-{spec['name']}-page-{page_number}.txt"
            write_bytes(raw_path, raw)
            record["raw_outputs"].append(
                {
                    "page": page_number,
                    "path": str(raw_path),
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )
        record["doc_latency_sec"] = time.perf_counter() - doc_started
        record["raw_output_bytes"] = sum(row["bytes"] for row in record["raw_outputs"])
        record["raw_output_sha256"] = [row["sha256"] for row in record["raw_outputs"]]
        record["status"] = "succeeded"
        record["success"] = True
    except Exception as exc:  # noqa: BLE001 - failure is a required complete evidence row
        for output in record["raw_outputs"]:
            Path(output["path"]).unlink(missing_ok=True)
        record["raw_outputs"] = []
        record["raw_output_bytes"] = 0
        record["raw_output_sha256"] = []
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        record["child_wall_sec"] = time.perf_counter() - started
        record["peak_process_rss_bytes_child"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_available():
            record["peak_cuda_allocated_bytes"] = torch_module.cuda.max_memory_allocated(0)
            record["peak_cuda_reserved_bytes"] = torch_module.cuda.max_memory_reserved(0)
        write_json(result_path, record)
        print(json.dumps({"status": record["status"], "result_path": str(result_path)}), flush=True)
        del model, processor
    return 0 if record["success"] else 1


def sampled_rss_bytes(process: Any) -> int:
    processes = [process, *process.children(recursive=True)]
    return sum(item.memory_info().rss for item in processes if item.is_running())


def sampled_vram_bytes(pid: int) -> int:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise RuntimeError("nvidia-smi is unavailable")
    completed = subprocess.run(
        [
            binary,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"nvidia-smi sampling failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    total_mib = 0
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) == 2 and columns[0] == str(pid):
            try:
                total_mib += int(columns[1])
            except ValueError as exc:
                raise RuntimeError(f"invalid nvidia-smi memory sample: {columns[1]}") from exc
    return total_mib * 1024 * 1024


def run_fresh_child(
    runner: Path, spec: dict[str, Any], model_index: int, pages: list[Path]
) -> dict[str, Any]:
    import psutil

    logs = OUT / "logs"
    results = OUT / "child-results"
    logs.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{model_index:02d}-{spec['name']}.stdout.txt"
    stderr_path = logs / f"{model_index:02d}-{spec['name']}.stderr.txt"
    result_path = results / f"{model_index:02d}-{spec['name']}.json"
    expected_raw_paths = [
        OUT / "raw" / f"{model_index:02d}-{spec['name']}-page-{page_number}.txt"
        for page_number in (1, 2)
    ]
    if result_path.exists():
        result_path.unlink()
    for raw_path in expected_raw_paths:
        if raw_path.exists():
            raw_path.unlink()
    command = [
        sys.executable,
        str(runner),
        "--child",
        "--model-index",
        str(model_index),
        "--spec-json",
        json.dumps(spec, ensure_ascii=False),
        "--result-path",
        str(result_path),
        "--pages",
        *(str(page) for page in pages),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.perf_counter()
    peak_rss = None
    peak_vram = None
    rss_sampling_errors: set[str] = set()
    vram_sampling_errors: set[str] = set()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        child = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
            cwd=str(OUT),
        )
        monitored = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss_sample = sampled_rss_bytes(monitored)
                peak_rss = rss_sample if peak_rss is None else max(peak_rss, rss_sample)
            except (OSError, psutil.Error) as exc:
                rss_sampling_errors.add(f"{type(exc).__name__}: {exc}")
            try:
                vram_sample = sampled_vram_bytes(child.pid)
                peak_vram = vram_sample if peak_vram is None else max(peak_vram, vram_sample)
            except (OSError, RuntimeError) as exc:
                vram_sampling_errors.add(f"{type(exc).__name__}: {exc}")
            time.sleep(0.1)
        exit_code = child.wait()
    wall = time.perf_counter() - started
    if peak_rss is None or peak_rss <= 0:
        rss_sampling_errors.add("no positive parent RSS sample collected")
    if peak_vram is None:
        vram_sampling_errors.add("no parent VRAM sample collected")
    if result_path.is_file():
        try:
            row = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            row = empty_result(spec, model_index)
            row.update(status="failed", error=f"invalid child result: {type(exc).__name__}: {exc}")
    else:
        row = empty_result(spec, model_index)
        row.update(status="failed", error="child exited without writing its result record")
    if exit_code and row["success"]:
        row.update(
            success=False, status="failed", error=f"child exit code {exit_code} contradicted result"
        )
    if rss_sampling_errors or vram_sampling_errors:
        sampling_error = (
            f"parent resource sampling failed: RSS={'; '.join(sorted(rss_sampling_errors)) or None}; "
            f"VRAM={'; '.join(sorted(vram_sampling_errors)) or None}"
        )
        row.update(
            success=False,
            status="failed",
            error=f"{row['error']}; {sampling_error}" if row.get("error") else sampling_error,
        )
    if not row.get("success"):
        for raw_path in expected_raw_paths:
            raw_path.unlink(missing_ok=True)
        row["raw_outputs"] = []
        row["raw_output_bytes"] = 0
        row["raw_output_sha256"] = []
    row.update(
        exit_code=exit_code,
        parent_wall_sec=wall,
        peak_process_rss_bytes_parent_sampled=peak_rss,
        peak_process_rss_sampling_error="; ".join(sorted(rss_sampling_errors)) or None,
        peak_vram_bytes_parent_sampled=peak_vram,
        peak_vram_sampling_error="; ".join(sorted(vram_sampling_errors)) or None,
        stdout_path=str(stdout_path),
        stdout_bytes=stdout_path.stat().st_size,
        stdout_sha256=sha256_file(stdout_path),
        stderr_path=str(stderr_path),
        stderr_bytes=stderr_path.stat().st_size,
        stderr_sha256=sha256_file(stderr_path),
    )
    return {field: row.get(field) for field in RESULT_FIELDS}


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def write_results(rows: list[dict[str, Any]]) -> None:
    jsonl = b"".join(json_bytes(row, pretty=False) for row in rows)
    write_bytes(OUT / "results.jsonl", jsonl)
    csv_path = OUT / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row[key]) for key in RESULT_FIELDS})


def build_report(rows: list[dict[str, Any]], run_id: str) -> str:
    lines = [
        "# Fixed DocSem OCR smoke v2",
        "",
        f"- run_id: `{run_id}`",
        f"- input: `{INSTANCE_ID}` PDF only; pages 1-2 rendered by Poppler at {RENDER_DPI} dpi",
        f"- preflight generation ceiling: `{MAX_NEW_TOKENS}` tokens per page",
        "- truncation warning: 512 generated tokens can truncate a page; this run does not claim full coverage",
        "- device policy: `CUDA_VISIBLE_DEVICES=0`, model device `cuda:0`, sequential fresh children",
        "- blind-input policy: no task, query, label, answer, or evidence source is read",
        "",
        "| model | gate | status | load s | doc s | parent RSS B | child RSS B | parent VRAM B | child allocated VRAM B | raw B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['gate']} | {row['status']} | "
            f"{row['load_sec']} | {row['doc_latency_sec']} | "
            f"{row['peak_process_rss_bytes_parent_sampled']} | "
            f"{row['peak_process_rss_bytes_child']} | "
            f"{row['peak_vram_bytes_parent_sampled']} | "
            f"{row['peak_cuda_allocated_bytes']} | {row['raw_output_bytes']} |"
        )
        if row["peak_process_rss_sampling_error"] or row["peak_vram_sampling_error"]:
            lines.append(
                f"\n`{row['name']}` sampling error: "
                f"RSS={row['peak_process_rss_sampling_error']}; "
                f"VRAM={row['peak_vram_sampling_error']}"
            )
        if row["error"]:
            lines.append(f"\n`{row['name']}` error: `{row['error']}`")
    lines.extend(
        [
            "",
            "The JSONL is authoritative. CSV is a lossless field-for-field projection; nested values are JSON strings.",
            "Child stdout/stderr and raw model text are stored verbatim and content-addressed in the artifact manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


def environment_manifest(
    runner_fact: dict[str, Any], source_mode: str, source_caveat: str, bootstrap: dict[str, Any]
) -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    package_versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    nvidia = (
        subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
        if shutil.which("nvidia-smi")
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "package_versions": dict(
            sorted(package_versions.items(), key=lambda item: item[0].lower())
        ),
        "pip_freeze_all_verbatim": freeze,
        "bootstrap": bootstrap,
        "runner": {
            **runner_fact,
            "source_acquisition_mode": source_mode,
            "bootstrap_caveat": source_caveat,
        },
        "cuda_visible_devices_for_children": "0",
        "device": DEVICE,
        "dtype": DTYPE,
        "nvidia_smi_L_stdout": nvidia.stdout if nvidia else None,
        "nvidia_smi_L_stderr": nvidia.stderr if nvidia else None,
    }


def write_artifact_hashes() -> list[dict[str, Any]]:
    manifest_path = OUT / "artifact-hashes.json"
    artifacts = []
    for path in sorted(
        item for item in OUT.glob("**/*") if item.is_file() and item != manifest_path
    ):
        artifacts.append(file_fact(path))
    write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "scope": "all files under OUT except this self-referential manifest",
            "artifacts": artifacts,
        },
    )
    return artifacts


def parent_main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source, source_mode, source_caveat = exact_executed_source()
    runner = OUT / "runner-executed.py"
    write_bytes(runner, source)
    runner_fact = file_fact(runner)
    bootstrap = install_and_verify_runtime()
    pdf = select_fixed_pdf()
    poppler = poppler_facts()
    pages = render_fixed_pages(pdf, poppler)
    run_id = f"{INSTANCE_ID}-{int(time.time())}-{runner_fact['sha256'][:12]}"

    input_manifest = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": INSTANCE_ID,
        "pdf": file_fact(pdf),
        "pages": [{"page": number, **file_fact(path)} for number, path in enumerate(pages, 1)],
        "expected_pdf_sha256": EXPECTED_PDF_SHA256,
        "expected_page_sha256": list(EXPECTED_PAGE_SHA256),
        "poppler": poppler,
    }
    write_json(OUT / "input.json", input_manifest)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "result": "pass",
        "approved_source_reads": [str(pdf)],
        "model_visible_inputs": [str(page) for page in pages],
        "model_visible_non_file_input": ["fixed per-model OCR prompt"],
        "prohibited_source_reads": ["tasks", "query", "labels", "answer", "evidence"],
        "statement": (
            "The parent selects only the exact task_000909 PDF; children receive only the two "
            "rendered page paths and fixed spec. No task/query/label/answer/evidence source path "
            "is discovered, opened, joined, or passed to a model."
        ),
    }
    write_json(OUT / "audit.json", audit)
    write_json(
        OUT / "environment.json",
        environment_manifest(runner_fact, source_mode, source_caveat, bootstrap),
    )

    rows = [run_fresh_child(runner, spec, index, pages) for index, spec in enumerate(SPECS, 1)]
    write_results(rows)
    write_bytes(OUT / "report.md", build_report(rows, run_id).encode("utf-8"))

    nodes = [
        {"id": "runner", "kind": "executed_source", **runner_fact},
        {"id": "pdf", "kind": "fixed_input", **file_fact(pdf)},
        *[
            {"id": f"page-{number}", "kind": "rendered_input", **file_fact(path)}
            for number, path in enumerate(pages, 1)
        ],
        *[
            {
                "id": f"model-{row['model_index']}",
                "kind": "fresh_child_result",
                "status": row["status"],
            }
            for row in rows
        ],
        {"id": "results-jsonl", "kind": "verbatim_records", **file_fact(OUT / "results.jsonl")},
        {"id": "results-csv", "kind": "verbatim_projection", **file_fact(OUT / "results.csv")},
        {"id": "report", "kind": "human_summary", **file_fact(OUT / "report.md")},
    ]
    edges = [
        {"from": "runner", "to": "page-1", "operation": "Poppler render"},
        {"from": "runner", "to": "page-2", "operation": "Poppler render"},
        {"from": "pdf", "to": "page-1", "operation": "page 1 at 200 dpi"},
        {"from": "pdf", "to": "page-2", "operation": "page 2 at 200 dpi"},
    ]
    for row in rows:
        child_id = f"model-{row['model_index']}"
        edges.extend(
            [
                {"from": "runner", "to": child_id, "operation": "fresh sequential child"},
                {"from": "page-1", "to": child_id, "operation": "pixel input"},
                {"from": "page-2", "to": child_id, "operation": "pixel input"},
                {"from": child_id, "to": "results-jsonl", "operation": "verbatim result row"},
            ]
        )
    edges.extend(
        [
            {"from": "results-jsonl", "to": "results-csv", "operation": "field projection"},
            {"from": "results-jsonl", "to": "report", "operation": "summary"},
        ]
    )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_from_exact_runner_sha256": runner_fact["sha256"],
        "sequential_model_order": [spec["name"] for spec in SPECS],
        "fresh_child_per_model": True,
        "dag": {"nodes": nodes, "edges": edges},
        "terminal_state": "complete_with_model_failures"
        if any(not row["success"] for row in rows)
        else "complete",
        "completed_at": utc_now(),
    }
    write_json(OUT / "run-manifest.json", run_manifest)
    artifacts = write_artifact_hashes()
    print(json.dumps({"out": str(OUT), "run_id": run_id, "artifact_count": len(artifacts)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--model-index", type=int)
    parser.add_argument("--spec-json")
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--pages", type=Path, nargs="*")
    # Notebook kernels add their own argv entries.  Child mode still validates every
    # required runner argument below, while a pasted single cell ignores kernel args.
    return parser.parse_known_args()[0]


def main() -> int:
    args = parse_args()
    if args.child:
        if (
            None in (args.model_index, args.spec_json, args.result_path)
            or len(args.pages or []) != 2
        ):
            raise SystemExit(
                "child mode requires model index, spec, result path, and exactly two pages"
            )
        return child_run(json.loads(args.spec_json), args.model_index, args.pages, args.result_path)
    parent_main()
    return 0


if __name__ == "__main__":
    _exit_code = main()
    # A successful pasted notebook cell must not end in a visible SystemExit.  The
    # materialized child script does need a non-zero process status on model failure.
    if "--child" in sys.argv:
        raise SystemExit(_exit_code)
