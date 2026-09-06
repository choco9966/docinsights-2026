#!/usr/bin/env python3
"""Prepare and run the input-only Issue 24 OCR benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "issue24-ocr-benchmark-v1"
SELECTION_SALT = f"{SCHEMA_VERSION}:"
EXCLUDED_INSTANCE_IDS = frozenset({"task_010001", "task_010002"})
SPLIT_COUNTS = {"heldout": 20, "train": 10}
RAPID_DET_SHA256 = "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"
RAPID_REC_SHA256 = "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8"
PADDLE_DET_TREE_SHA256 = "b3ca5a167f80b79e8433df9cb931c578f384f7f8867b960748022ee1b1826916"
PADDLE_REC_TREE_SHA256 = "2a3324a89b92f446da343999c922d0fa7a0fa3b0e0a3bbb7034d654c588f1e16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(instance_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}{instance_id}".encode()).hexdigest()


def page_selection_key(
    split: str, instance_id: str, pdf_sha256: str, page_number: int
) -> str:
    value = f"{SELECTION_SALT}{split}:{instance_id}:{pdf_sha256}:{page_number}"
    return hashlib.sha256(value.encode()).hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def selected_records(inventory_path: Path, split: str) -> list[dict[str, Any]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidates = [
        record
        for record in inventory["records"]
        if record["instance_id"] not in EXCLUDED_INSTANCE_IDS and int(record["pdf_pages"]) >= 2
    ]
    candidates.sort(
        key=lambda record: (selection_key(record["instance_id"]), record["instance_id"])
    )
    count = SPLIT_COUNTS[split]
    if len(candidates) < count:
        raise RuntimeError(f"{split} has only {len(candidates)} eligible public documents")
    return candidates[:count]


def render_page(pdf: Path, page_number: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="ocr-benchmark-render-", dir=destination.parent
    ) as temp:
        prefix = Path(temp) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-singlefile",
                "-r",
                "175",
                "-png",
                str(pdf),
                str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        rendered = prefix.with_suffix(".png")
        os.replace(rendered, destination)


def create_core_crops(image: Path, page_id: str, output_root: Path) -> list[dict[str, Any]]:
    from PIL import Image

    crops = []
    with Image.open(image) as source:
        boundaries = (0, source.height // 3, (source.height * 2) // 3, source.height)
        for index, (top, bottom) in enumerate(
            zip(boundaries, boundaries[1:], strict=True), start=1
        ):
            crop_path = output_root / "cores" / image.parent.name / f"{page_id}-core-{index}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            with source.crop((0, top, source.width, bottom)) as crop:
                crop.save(crop_path, format="PNG")
            crops.append(
                {
                    "core_index": index,
                    "crop_box": {
                        "left": 0,
                        "top": top,
                        "right": source.width,
                        "bottom": bottom,
                    },
                    "image_path": str(crop_path.resolve()),
                    "image_sha256": sha256_file(crop_path),
                    "image_bytes": crop_path.stat().st_size,
                    "width": source.width,
                    "height": bottom - top,
                }
            )
    return crops


def prepare(args: argparse.Namespace) -> int:
    manifest_path = args.output_root / "corpus-manifest.json"
    if manifest_path.exists():
        print(json.dumps({"manifest": str(manifest_path.resolve()), "status": "already_exists"}))
        return 0

    pages: list[dict[str, Any]] = []
    inventories = {
        split: args.inputs_root / split / "pdf-inventory.json" for split in SPLIT_COUNTS
    }
    for split, inventory_path in inventories.items():
        for source in selected_records(inventory_path, split):
            pdf = Path(source["actual_path"])
            actual_pdf_sha256 = sha256_file(pdf)
            if actual_pdf_sha256 != source["sha256"]:
                raise RuntimeError(f"public PDF digest mismatch: {pdf}")
            page_count = int(source["pdf_pages"])
            chosen_pages = sorted(
                sorted(
                    range(1, page_count + 1),
                    key=lambda page: (
                        page_selection_key(
                            split, source["instance_id"], actual_pdf_sha256, page
                        ),
                        page,
                    ),
                )[:2]
            )
            for page_number in chosen_pages:
                page_id = f"{split}-{source['instance_id']}-p{page_number:04d}"
                image = args.output_root / "pages" / split / f"{page_id}.png"
                render_page(pdf, page_number, image)
                width, height = png_dimensions(image)
                pages.append(
                    {
                        "benchmark_page_id": page_id,
                        "split": split,
                        "instance_id": source["instance_id"],
                        "page_number": page_number,
                        "pdf_page_count": page_count,
                        "source_pdf": str(pdf.resolve()),
                        "source_pdf_sha256": actual_pdf_sha256,
                        "rendered_image": str(image.resolve()),
                        "rendered_image_sha256": sha256_file(image),
                        "rendered_image_bytes": image.stat().st_size,
                        "width": width,
                        "height": height,
                        "scoring_cores": create_core_crops(image, page_id, args.output_root),
                    }
                )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "public_inputs_only": True,
            "queries_read": False,
            "labels_read": False,
            "answers_read": False,
            "references_read": False,
        },
        "selection": {
            "algorithm": "ascending SHA256('issue24-ocr-benchmark-v1:' + instance_id)",
            "page_algorithm": (
                "per selected document, take the two lowest ascending SHA256("
                "'issue24-ocr-benchmark-v1:' + split + ':' + instance_id + ':' + "
                "pdf_sha256 + ':' + decimal_page_number), then sort by page number"
            ),
            "excluded_instance_ids": sorted(EXCLUDED_INSTANCE_IDS),
            "split_document_counts": SPLIT_COUNTS,
            "pages_per_document": 2,
            "minimum_pdf_pages": 2,
        },
        "render": {
            "tool": "pdftoppm",
            "dpi": 175,
            "format": "png",
            "lossless": True,
            "command_template": (
                "pdftoppm -f PAGE -l PAGE -singlefile -r 175 -png INPUT.pdf OUTPUT_PREFIX"
            ),
        },
        "scoring_cores": {
            "count_per_page": 3,
            "layout": "full-width non-overlapping horizontal thirds",
            "boundaries": "[0, floor(H/3), floor(2*H/3), H]",
            "engine_input": False,
            "line_boundary_policy": "scorer excludes OCR lines crossing a core boundary",
        },
        "inventory_sources": {
            split: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for split, path in inventories.items()
        },
        "document_count": sum(SPLIT_COUNTS.values()),
        "page_count": len(pages),
        "pages": pages,
    }
    if manifest["page_count"] != 60:
        raise RuntimeError(f"expected 60 pages, got {manifest['page_count']}")
    atomic_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        f"{digest}  {manifest_path.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": digest,
                "documents": manifest["document_count"],
                "pages": manifest["page_count"],
                "rendered_bytes": sum(page["rendered_image_bytes"] for page in pages),
            },
            sort_keys=True,
        )
    )
    return 0


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def directory_content_hash(path: Path) -> str:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and not any(part.startswith(".") for part in file.relative_to(path).parts)
    )
    if not files:
        raise ValueError(f"model directory is empty: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def clipped_box(raw: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = raw
    return [
        min(max(x0, 0.0), float(width)),
        min(max(y0, 0.0), float(height)),
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
    ]


def output_line(
    *,
    source_order: int,
    text: str,
    confidence: float | None,
    raw_bbox: list[float],
    raw_geometry: Any,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "source_order": source_order,
        "text": text,
        "confidence": confidence,
        "bbox": clipped_box(raw_bbox, width, height),
        "raw_bbox": raw_bbox,
        "raw_geometry": raw_geometry,
    }


def base_result(
    *, manifest_path: Path, manifest_sha256: str, engine: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "issue24-ocr-candidate-v1",
        "benchmark_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
        },
        "scope": {
            "public_page_images_only": True,
            "queries_read": False,
            "labels_read": False,
            "answers_read": False,
            "references_read": False,
        },
        "engine": engine,
        "config": config,
        "config_sha256": canonical_hash(config),
        "initialization_seconds": None,
        "cold_start_seconds": None,
        "pages": [],
        "summary": None,
    }


def page_stub(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_page_id": page["benchmark_page_id"],
        "status": "failed",
        "input_image": page["rendered_image"],
        "input_image_sha256": page["rendered_image_sha256"],
        "width": page["width"],
        "height": page["height"],
        "elapsed_seconds": None,
        "engine_elapsed_seconds": None,
        "error": None,
        "lines": [],
    }


def finalize_result(result: dict[str, Any], output: Path, total_seconds: float) -> None:
    elapsed = [
        float(page["elapsed_seconds"])
        for page in result["pages"]
        if page["status"] == "succeeded" and page["elapsed_seconds"] is not None
    ]
    ordered = sorted(elapsed)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1) if ordered else 0
    result["summary"] = {
        "page_count": len(result["pages"]),
        "succeeded": sum(page["status"] == "succeeded" for page in result["pages"]),
        "failed": sum(page["status"] != "succeeded" for page in result["pages"]),
        "total_wall_seconds": total_seconds,
        "mean_page_seconds": statistics.fmean(elapsed) if elapsed else None,
        "median_page_seconds": statistics.median(elapsed) if elapsed else None,
        "p95_page_seconds": ordered[p95_index] if ordered else None,
    }
    atomic_json(output, result)


def verify_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("page_count") != 60 or len(manifest.get("pages", [])) != 60:
        raise RuntimeError("benchmark manifest must contain exactly 60 pages")
    for page in manifest["pages"]:
        image = Path(page["rendered_image"])
        if sha256_file(image) != page["rendered_image_sha256"]:
            raise RuntimeError(f"benchmark image digest mismatch: {image}")
    return manifest, digest


def rapidocr_runner(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    rapidocr = importlib.import_module("rapidocr")

    if sha256_file(args.det_model) != RAPID_DET_SHA256:
        raise RuntimeError("RapidOCR detector digest mismatch")
    if sha256_file(args.rec_model) != RAPID_REC_SHA256:
        raise RuntimeError("RapidOCR recognizer digest mismatch")
    config = {
        "Global.use_cls": False,
        "Global.log_level": "error",
        "EngineConfig.onnxruntime.intra_op_num_threads": 2,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        "Det.engine_type": "onnxruntime",
        "Det.ocr_version": "PP-OCRv5",
        "Det.lang_type": "ch",
        "Det.model_type": "mobile",
        "Det.model_path": str(args.det_model.resolve()),
        "Det.limit_side_len": 736,
        "Det.limit_type": "min",
        "Rec.engine_type": "onnxruntime",
        "Rec.ocr_version": "PP-OCRv5",
        "Rec.lang_type": "en",
        "Rec.model_type": "mobile",
        "Rec.model_path": str(args.rec_model.resolve()),
        "Rec.rec_batch_num": 6,
    }
    params = dict(config)
    params.update(
        {
            "Det.engine_type": rapidocr.EngineType.ONNXRUNTIME,
            "Det.ocr_version": rapidocr.OCRVersion.PPOCRV5,
            "Det.lang_type": rapidocr.LangDet.CH,
            "Det.model_type": rapidocr.ModelType.MOBILE,
            "Rec.engine_type": rapidocr.EngineType.ONNXRUNTIME,
            "Rec.ocr_version": rapidocr.OCRVersion.PPOCRV5,
            "Rec.lang_type": rapidocr.LangRec.EN,
            "Rec.model_type": rapidocr.ModelType.MOBILE,
        }
    )
    started = time.perf_counter()
    engine = rapidocr.RapidOCR(params=params)
    initialization = time.perf_counter() - started
    identity = {
        "name": "rapidocr-ppocrv5-onnx",
        "implementation": "RapidOCR public third-party PP-OCRv5 ONNX conversion",
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("rapidocr", "onnxruntime", "numpy", "opencv-python")
        },
        "models": {
            "detector": {
                "path": str(args.det_model.resolve()),
                "sha256": RAPID_DET_SHA256,
                "source": (
                    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
                    "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx"
                ),
            },
            "recognizer": {
                "path": str(args.rec_model.resolve()),
                "sha256": RAPID_REC_SHA256,
                "source": (
                    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
                    "onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx"
                ),
            },
        },
    }

    def recognize(page: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
        value = engine(Path(page["rendered_image"]), use_cls=False)
        boxes = [] if value.boxes is None else value.boxes.tolist()
        texts = [] if value.txts is None else list(value.txts)
        scores = [] if value.scores is None else [float(score) for score in value.scores]
        if not len(boxes) == len(texts) == len(scores):
            raise RuntimeError("RapidOCR boxes/text/scores differ in length")
        lines = []
        for index, (quad, text, confidence) in enumerate(
            zip(boxes, texts, scores, strict=True)
        ):
            points = [[float(point[0]), float(point[1])] for point in quad]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            lines.append(
                output_line(
                    source_order=index,
                    text=str(text),
                    confidence=confidence,
                    raw_bbox=[min(xs), min(ys), max(xs), max(ys)],
                    raw_geometry={"quadrilateral": points},
                    width=page["width"],
                    height=page["height"],
                )
            )
        return lines, float(value.elapse)

    return {"config": config, "identity": identity, "initialization": initialization}, recognize


def paddle_payload(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else getattr(value, "json", None)
    if payload is None:
        payload = getattr(value, "res", None)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR result is not an object")
    return payload["res"] if isinstance(payload.get("res"), dict) else payload


def as_list(value: Any, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise RuntimeError(f"PaddleOCR {field} is not a list")
    return value


def paddle_runner(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    PaddleOCR = importlib.import_module("paddleocr").PaddleOCR

    det_hash = directory_content_hash(args.det_model)
    rec_hash = directory_content_hash(args.rec_model)
    if det_hash != PADDLE_DET_TREE_SHA256:
        raise RuntimeError("native Paddle detector tree digest mismatch")
    if rec_hash != PADDLE_REC_TREE_SHA256:
        raise RuntimeError("native Paddle recognizer tree digest mismatch")
    config = {
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_dir": str(args.det_model.resolve()),
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_dir": str(args.rec_model.resolve()),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "cpu",
        "enable_mkldnn": False,
        "cpu_threads": 2,
        "text_recognition_batch_size": 6,
    }
    started = time.perf_counter()
    engine = PaddleOCR(**config)
    initialization = time.perf_counter() - started
    identity = {
        "name": "native-paddle-ppocrv5-mobile",
        "implementation": "PaddleOCR Python API with official public Paddle model trees",
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("paddlepaddle", "paddleocr", "paddlex")
        },
        "models": {
            "detector": {
                "path": str(args.det_model.resolve()),
                "tree_sha256": det_hash,
                "repository": "PaddlePaddle/PP-OCRv5_mobile_det",
                "revision": "0d63e78e2b680928f6b1747d76a08db6e645efb7",
            },
            "recognizer": {
                "path": str(args.rec_model.resolve()),
                "tree_sha256": rec_hash,
                "repository": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
                "revision": "267c36e24c331595590fe7bd72bde2436fd286f2",
            },
        },
    }

    def recognize(page: dict[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
        raw_results = list(engine.predict(page["rendered_image"]))
        if len(raw_results) != 1:
            raise RuntimeError(f"PaddleOCR returned {len(raw_results)} page results")
        payload = paddle_payload(raw_results[0])
        texts = as_list(payload.get("rec_texts"), "rec_texts")
        scores = as_list(payload.get("rec_scores"), "rec_scores")
        boxes = as_list(payload.get("rec_boxes"), "rec_boxes")
        polygons_value = payload.get("rec_polys")
        polygons = as_list(polygons_value, "rec_polys") if polygons_value is not None else boxes
        if not len(texts) == len(scores) == len(boxes) == len(polygons):
            raise RuntimeError("PaddleOCR geometry/text/score lengths differ")
        lines = []
        for index, (text, confidence, box, polygon) in enumerate(
            zip(texts, scores, boxes, polygons, strict=True)
        ):
            raw_box = [float(coordinate) for coordinate in box]
            geometry = (
                {"quadrilateral": [[float(x), float(y)] for x, y in polygon]}
                if polygon is not box
                else {"axis_aligned_box": raw_box}
            )
            lines.append(
                output_line(
                    source_order=index,
                    text=str(text),
                    confidence=float(confidence),
                    raw_bbox=raw_box,
                    raw_geometry=geometry,
                    width=page["width"],
                    height=page["height"],
                )
            )
        return lines, None

    return {"config": config, "identity": identity, "initialization": initialization}, recognize


def run_python_engine(args: argparse.Namespace, manifest: dict[str, Any], manifest_sha: str) -> int:
    factory = rapidocr_runner if args.engine == "rapidocr" else paddle_runner
    setup, recognize = factory(args)
    result = base_result(
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha,
        engine=setup["identity"],
        config=setup["config"],
    )
    result["initialization_seconds"] = setup["initialization"]
    total_started = time.perf_counter()
    for source_page in manifest["pages"]:
        page = page_stub(source_page)
        started = time.perf_counter()
        try:
            lines, engine_elapsed = recognize(source_page)
            page.update(
                {
                    "status": "succeeded",
                    "elapsed_seconds": time.perf_counter() - started,
                    "engine_elapsed_seconds": engine_elapsed,
                    "lines": lines,
                }
            )
        except Exception as error:
            page.update(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        result["pages"].append(page)
        if result["cold_start_seconds"] is None:
            result["cold_start_seconds"] = setup["initialization"] + page["elapsed_seconds"]
        atomic_json(args.output, result)
    finalize_result(result, args.output, time.perf_counter() - total_started)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


def apple_runner(args: argparse.Namespace, manifest: dict[str, Any], manifest_sha: str) -> int:
    config = {
        "recognition_level": "accurate",
        "recognition_languages": ["en-US"],
        "uses_language_correction": True,
        "orientation": "image metadata or up",
        "batch_mode": "one warm process, sequential requests",
    }
    version = subprocess.check_output(["xcrun", "swift", "--version"], text=True).strip()
    os_version = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    identity = {
        "name": "apple-vision-accurate",
        "implementation": "macOS Vision VNRecognizeTextRequest",
        "versions": {"swift": version, "macOS": os_version},
        "tool": {
            "path": str(args.apple_tool.resolve()),
            "sha256": sha256_file(args.apple_tool),
            "source_path": str(args.apple_source.resolve()),
            "source_sha256": sha256_file(args.apple_source),
        },
    }
    result = base_result(
        manifest_path=args.manifest,
        manifest_sha256=manifest_sha,
        engine=identity,
        config=config,
    )
    first_image = manifest["pages"][0]["rendered_image"]
    cold_started = time.perf_counter()
    cold = subprocess.run(
        [str(args.apple_tool), first_image],
        check=True,
        capture_output=True,
        text=True,
    )
    cold_wall = time.perf_counter() - cold_started
    cold_payload = json.loads(cold.stdout)
    result["initialization_seconds"] = max(0.0, cold_wall - cold_payload["elapsed_seconds"])
    result["cold_start_seconds"] = cold_wall

    image_paths = [page["rendered_image"] for page in manifest["pages"]]
    total_started = time.perf_counter()
    process = subprocess.run(
        [str(args.apple_tool), *image_paths],
        check=False,
        capture_output=True,
        text=True,
    )
    values = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    by_path = {str(Path(value["image_path"]).resolve()): value for value in values}
    stderr = process.stderr.strip()
    for source_page in manifest["pages"]:
        page = page_stub(source_page)
        value = by_path.get(str(Path(source_page["rendered_image"]).resolve()))
        if value is None:
            page["error"] = {
                "type": "AppleVisionMissingResult",
                "message": stderr or "tool returned no result for image",
            }
        else:
            lines = []
            for observation in value["observations"]:
                raw = observation["raw_bbox_pixels"]
                crop = observation["crop_bbox_pixels"]
                lines.append(
                    {
                        "source_order": observation["source_order"],
                        "text": observation["text"],
                        "confidence": float(observation["confidence"]),
                        "bbox": [crop["x0"], crop["y0"], crop["x1"], crop["y1"]],
                        "raw_bbox": [raw["x0"], raw["y0"], raw["x1"], raw["y1"]],
                        "raw_geometry": {
                            "vision_normalized_lower_left": observation["raw_bbox_normalized"]
                        },
                    }
                )
            page.update(
                {
                    "status": "succeeded",
                    "elapsed_seconds": float(value["elapsed_seconds"]),
                    "engine_elapsed_seconds": float(value["elapsed_seconds"]),
                    "lines": lines,
                }
            )
        result["pages"].append(page)
    result["engine"]["request_revision"] = values[0]["request_revision"] if values else None
    finalize_result(result, args.output, time.perf_counter() - total_started)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


def run_engine(args: argparse.Namespace) -> int:
    manifest, manifest_sha = verify_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.engine == "apple-vision":
        return apple_runner(args, manifest, manifest_sha)
    return run_python_engine(args, manifest, manifest_sha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="freeze and render the public sample")
    prepare_parser.add_argument("--inputs-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.set_defaults(handler=prepare)
    run_parser = subparsers.add_parser("run", help="run one engine on the frozen corpus")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument(
        "--engine", choices=("apple-vision", "rapidocr", "paddle"), required=True
    )
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--det-model", type=Path)
    run_parser.add_argument("--rec-model", type=Path)
    run_parser.add_argument("--apple-tool", type=Path)
    run_parser.add_argument("--apple-source", type=Path)
    run_parser.set_defaults(handler=run_engine)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
