"""Spawn-safe PaddleOCR worker used by the Issue 24 solution runner."""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

_ENGINE: Any | None = None


def initialize_worker(config: dict[str, Any]) -> None:
    global _ENGINE
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    PaddleOCR = importlib.import_module("paddleocr").PaddleOCR
    _ENGINE = PaddleOCR(**config)


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
