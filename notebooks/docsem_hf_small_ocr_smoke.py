"""Kaggle GPU script for the fixed, label-free DocSem OCR smoke case.

The repository evaluator is dependency-free; this isolated runner intentionally pins
the experimental ML stack. It reads one exact PDF and passes only rendered pixels plus
a fixed OCR instruction to each model.
"""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import subprocess
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

INSTANCE_ID = "task_000909"
EXPECTED_PDF_SHA256 = "fa54e0a898b530757c8419524552d00dcb60bb9449e92b78c3c8c6ce3d82b798"
INPUT_CANDIDATES = (
    Path("/kaggle/input/docsem-validation-ocr-input/documents/task_000909.pdf"),
    Path("/kaggle/input/docsem-validation-ocr-input/val/documents/task_000909.pdf"),
)
OUT = Path("/kaggle/working/docinsights-hf-smoke")
MAX_NEW_TOKENS = 512
SPECS = (
    (
        "PaddleOCR-VL-1.6",
        "PaddlePaddle/PaddleOCR-VL-1.6",
        "c5630abae1d940eafe0697512a0325494b02ab42",
        "OCR:",
        True,
    ),
    (
        "GLM-OCR",
        "zai-org/GLM-OCR",
        "ca5d8b3e287e52589e37c28385d9655ee4372f9d",
        "Text Recognition:",
        False,
    ),
    (
        "surya-ocr-2",
        "datalab-to/surya-ocr-2",
        "3b3d4cdf88d6928b0acdc75181b13206ea67c4a3",
        "OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).",
        False,
    ),
    (
        "granite-docling-258M",
        "ibm-granite/granite-docling-258M",
        "982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe",
        "Convert this page to DocTags.",
        False,
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_pdf() -> Path:
    for candidate in INPUT_CANDIDATES:
        if candidate.is_file():
            if sha256(candidate) != EXPECTED_PDF_SHA256:
                raise RuntimeError("fixed PDF hash mismatch")
            return candidate
    raise FileNotFoundError("fixed DocSem PDF not mounted at an approved exact path")


def render(pdf: Path) -> list[Path]:
    pages = OUT / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-r", "200", "-png", str(pdf), str(pages / INSTANCE_ID)], check=True
    )
    return sorted(pages.glob(f"{INSTANCE_ID}-*.png"))


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_model(
    name: str, repo: str, revision: str, prompt: str, remote: bool, pages: list[Path]
) -> dict[str, object]:
    record: dict[str, object] = {
        "name": name,
        "repo": repo,
        "revision": revision,
        "prompt": prompt,
        "max_new_tokens": MAX_NEW_TOKENS,
        "trust_remote_code": remote,
        "success": False,
    }
    raw_paths: list[str] = []
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        processor = AutoProcessor.from_pretrained(repo, revision=revision, trust_remote_code=remote)
        model = AutoModelForImageTextToText.from_pretrained(
            repo,
            revision=revision,
            trust_remote_code=remote,
            dtype=torch.float16,
            device_map={"": 0},
        ).eval()
        synchronize()
        record["load_sec"] = time.perf_counter() - started
        latencies = []
        infer_started = time.perf_counter()
        for page_number, page_path in enumerate(pages, start=1):
            image = Image.open(page_path).convert("RGB")
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
            synchronize()
            page_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            synchronize()
            latencies.append(time.perf_counter() - page_started)
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            output = OUT / "raw" / f"{name}-page-{page_number}.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            raw_paths.append(str(output))
        record.update(
            success=True,
            doc_latency_sec=time.perf_counter() - infer_started,
            page_latency_sec=latencies,
        )
    except Exception as exc:  # noqa: BLE001 - each model failure is measured evidence
        record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        record["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        record["peak_cuda_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        record["raw_output_paths"] = raw_paths
        record["raw_output_bytes"] = sum(Path(path).stat().st_size for path in raw_paths)
        record["command"] = (
            "direct transformers pinned from_pretrained + generate on two fixed rendered PNGs"
        )
        for value in (locals().get("model"), locals().get("processor")):
            del value
        gc.collect()
        torch.cuda.empty_cache()
    return record


def main() -> None:
    pages = render(fixed_pdf())
    records = [run_model(*spec, pages) for spec in SPECS]
    result_path = OUT / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    print(result_path)


if __name__ == "__main__":
    main()
