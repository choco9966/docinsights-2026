"""Spawn-safe PaddleOCR worker used by the Issue 24 solution runner."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ENGINE: Any | None = None


class OrtInfer:
    def __init__(self, ort: Any, model: Path, config: dict[str, Any]) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(config["intra_op_num_threads"])
        options.inter_op_num_threads = int(config["inter_op_num_threads"])
        options.execution_mode = getattr(ort.ExecutionMode, config["execution_mode"])
        options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, config["graph_optimization_level"]
        )
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=list(config["providers"])
        )
        if self.session.get_providers() != list(config["providers"]):
            raise RuntimeError("ONNX Runtime did not select the pinned execution provider")
        self.names = [item.name for item in self.session.get_inputs()]

    def __call__(self, x: list[np.ndarray]) -> list[np.ndarray]:
        if len(x) != len(self.names):
            raise RuntimeError("PaddleX and ONNX Runtime input counts differ")
        feed = {
            name: np.ascontiguousarray(value)
            for name, value in zip(self.names, x, strict=True)
        }
        return self.session.run(None, feed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_worker(config: dict[str, Any]) -> None:
    global _ENGINE
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    paddle_config = dict(config["paddle"])
    ort_config = dict(config["onnxruntime"])
    site_packages = Path(ort_config["site_packages"]).resolve()
    site_value = str(site_packages)
    if site_value not in sys.path:
        sys.path.append(site_value)
    ort = importlib.import_module("onnxruntime")
    module_file = getattr(ort, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("ONNX Runtime does not expose its imported module path")
    module_path = Path(module_file).resolve()
    expected_module_path = Path(ort_config["module_path"]).resolve()
    if module_path != expected_module_path:
        raise RuntimeError("ONNX Runtime was imported from an unpinned location")
    if _sha256_file(module_path) != ort_config["module_sha256"]:
        raise RuntimeError("ONNX Runtime import module differs from the pinned hash")
    if ort.__version__ != ort_config["version"]:
        raise RuntimeError("ONNX Runtime version differs from the pinned version")
    models = {}
    for role in ("detector", "recognizer"):
        model = Path(ort_config[f"{role}_model"]).resolve()
        if not model.is_file() or _sha256_file(model) != ort_config[f"{role}_sha256"]:
            raise RuntimeError(f"pinned {role} ONNX model is missing or changed")
        models[role] = model
    PaddleOCR = importlib.import_module("paddleocr").PaddleOCR
    engine = PaddleOCR(**paddle_config)
    pipeline = engine.paddlex_pipeline._pipeline
    pipeline.text_det_model.infer = OrtInfer(ort, models["detector"], ort_config)
    pipeline.text_rec_model.infer = OrtInfer(ort, models["recognizer"], ort_config)
    _ENGINE = engine


def _payload(value: Any) -> dict[str, Any]:
    payload = value if isinstance(value, dict) else getattr(value, "json", None)
    if payload is None:
        payload = getattr(value, "res", None)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR result is not an object")
    return payload["res"] if isinstance(payload.get("res"), dict) else payload


def _list(value: Any, field: str) -> list[Any]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, list):
        raise RuntimeError(f"PaddleOCR {field} is not a list")
    return value


def recognize_document(image_paths: list[str]) -> list[dict[str, Any]]:
    if _ENGINE is None:
        raise RuntimeError("PaddleOCR worker is not initialized")
    pages = []
    for image_path in image_paths:
        raw_results = list(_ENGINE.predict(image_path))
        if len(raw_results) != 1:
            raise RuntimeError(f"PaddleOCR returned {len(raw_results)} page results")
        payload = _payload(raw_results[0])
        texts = [str(value) for value in _list(payload.get("rec_texts"), "rec_texts")]
        scores = [float(value) for value in _list(payload.get("rec_scores"), "rec_scores")]
        boxes = [
            [float(coordinate) for coordinate in box]
            for box in _list(payload.get("rec_boxes"), "rec_boxes")
        ]
        polygons_value = payload.get("rec_polys")
        polygons = (
            [
                [[float(x), float(y)] for x, y in polygon]
                for polygon in _list(polygons_value, "rec_polys")
            ]
            if polygons_value is not None
            else [None] * len(boxes)
        )
        if not len(texts) == len(scores) == len(boxes) == len(polygons):
            raise RuntimeError("PaddleOCR geometry/text/score lengths differ")
        pages.append(
            {"texts": texts, "scores": scores, "boxes": boxes, "polygons": polygons}
        )
    return pages
