"""Optional PP-OCRv5 mobile adapter for portable CPU benchmark workers."""

from __future__ import annotations

import hashlib
import json
import struct
from importlib import metadata
from pathlib import Path
from typing import Any

from .models import BoundingBox, Line, Page
from .records import deterministic_content_hash

DETECTION_MODEL_REPO = "PaddlePaddle/PP-OCRv5_mobile_det"
DETECTION_MODEL_REVISION = "0d63e78e2b680928f6b1747d76a08db6e645efb7"
RECOGNITION_MODEL_REPO = "PaddlePaddle/en_PP-OCRv5_mobile_rec"
RECOGNITION_MODEL_REVISION = "267c36e24c331595590fe7bd72bde2436fd286f2"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PaddleOCREngine:
    """Keep one PaddleOCR pipeline warm while recognizing a document shard."""

    def __init__(
        self,
        *,
        detection_model_dir: str | Path,
        recognition_model_dir: str | Path,
        detection_model_revision: str = DETECTION_MODEL_REVISION,
        recognition_model_revision: str = RECOGNITION_MODEL_REVISION,
        enable_mkldnn: bool = False,
    ) -> None:
        self.detection_model_dir = _validated_model_dir(detection_model_dir)
        self.recognition_model_dir = _validated_model_dir(recognition_model_dir)
        self.detection_model_revision = detection_model_revision
        self.recognition_model_revision = recognition_model_revision
        self.enable_mkldnn = enable_mkldnn
        self.language = "eng"
        self.executable = "paddleocr-python-api"
        self.options = {
            "device": "cpu",
            "enable_mkldnn": enable_mkldnn,
            "inference_timeout_enforced": False,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "detection_model_repo": DETECTION_MODEL_REPO,
            "detection_model_revision": detection_model_revision,
            "detection_model_tree_sha256": _directory_content_hash(self.detection_model_dir),
            "detection_model_path": str(self.detection_model_dir),
            "recognition_model_repo": RECOGNITION_MODEL_REPO,
            "recognition_model_revision": recognition_model_revision,
            "recognition_model_tree_sha256": _directory_content_hash(self.recognition_model_dir),
            "recognition_model_path": str(self.recognition_model_dir),
            "paddlepaddle_version": _package_version("paddlepaddle"),
            "paddleocr_version": _package_version("paddleocr"),
            "paddlex_version": _package_version("paddlex"),
        }
        identity_payload = {
            key: value for key, value in self.options.items() if not key.endswith("_path")
        }
        self.executable_identity = {
            "name": self.executable,
            "kind": "python_packages_and_model_trees",
            "sha256": deterministic_content_hash(identity_payload),
            "details": identity_payload,
        }
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is optional; install the pinned cloud notebook dependencies"
            ) from exc
        self._ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_detection_model_dir=str(self.detection_model_dir),
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            text_recognition_model_dir=str(self.recognition_model_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=enable_mkldnn,
        )

    @property
    def name(self) -> str:
        return "paddleocr-ppocrv5-mobile"

    @property
    def confidence_kind(self) -> str:
        return "paddleocr_line_confidence_0_to_1"

    def recognize(self, image_path: str | Path, *, page_number: int = 1) -> Page:
        resolved_image = Path(image_path).resolve()
        results = list(self._ocr.predict(str(resolved_image)))
        if len(results) != 1:
            raise ValueError("PaddleOCR must return exactly one result per page")
        return parse_paddle_result(results[0], page_number=page_number, image_path=resolved_image)


def parse_paddle_result(
    result: Any,
    *,
    page_number: int = 1,
    image_path: str | Path,
) -> Page:
    """Convert one PaddleOCR 3.x result to the common pixel-coordinate page model."""
    payload = _result_payload(result)
    texts = _list_field(payload, "rec_texts")
    scores = _list_field(payload, "rec_scores")
    boxes = _list_field(payload, "rec_boxes")
    if not len(texts) == len(scores) == len(boxes):
        raise ValueError("PaddleOCR text, score, and box counts differ")
    resolved_image = Path(image_path).resolve()
    width, height = _png_dimensions(resolved_image)
    lines: list[Line] = []
    for source_order, (text, score, box) in enumerate(zip(texts, scores, boxes, strict=True)):
        if not isinstance(text, str):
            raise ValueError("PaddleOCR rec_texts must contain strings")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("PaddleOCR rec_scores must contain numbers")
        confidence = float(score)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("PaddleOCR confidence must be between zero and one")
        bbox = _parse_box(box, image_width=width, image_height=height)
        if _is_known_watermark(text, bbox=bbox, image_width=width, image_height=height):
            continue
        if text.strip():
            lines.append(
                Line(
                    page_number=page_number,
                    text=text.strip(),
                    bbox=bbox,
                    confidence=confidence,
                    source_order=source_order,
                )
            )
    return Page(
        number=page_number,
        lines=tuple(lines),
        image_path=resolved_image,
        width=width,
        height=height,
    )


def _result_payload(result: Any) -> dict[str, Any]:
    value = result
    if not isinstance(value, dict):
        value = getattr(result, "json", None)
        if value is None:
            value = getattr(result, "res", None)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid PaddleOCR JSON result") from exc
    if not isinstance(value, dict):
        raise ValueError("PaddleOCR result must be an object")
    nested = value.get("res")
    if isinstance(nested, dict):
        value = nested
    return value


def _list_field(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"PaddleOCR {key} must be a list")
    return value


def _parse_box(box: Any, *, image_width: int, image_height: int) -> BoundingBox:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("PaddleOCR rec_boxes must contain four-coordinate boxes")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in box):
        raise ValueError("PaddleOCR box coordinates must be numeric")
    left, top, right, bottom = (round(float(value)) for value in box)
    if not 0 <= left < right <= image_width or not 0 <= top < bottom <= image_height:
        raise ValueError("PaddleOCR box extends beyond the image")
    return BoundingBox(left=left, top=top, width=right - left, height=bottom - top)


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != _PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"PaddleOCR input must be a PNG image: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1 or height < 1:
        raise ValueError("PNG dimensions must be positive")
    return width, height


def _is_known_watermark(
    text: str,
    *,
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
) -> bool:
    normalized = " ".join(text.upper().split())
    known_text = normalized in {"TRAINING COP", "TRAINING COPY"}
    oversized = bbox.height >= image_height * 0.15 or bbox.width >= image_width * 0.75
    return known_text and oversized


def _validated_model_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    required = (resolved / "inference.pdiparams", resolved / "inference.yml")
    if not resolved.is_dir() or not all(file.is_file() for file in required):
        raise ValueError(f"invalid pinned PaddleOCR model directory: {resolved}")
    return resolved


def _directory_content_hash(path: Path) -> str:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and not any(part.startswith(".") for part in file.relative_to(path).parts)
    )
    if not files:
        raise ValueError(f"PaddleOCR model directory contains no model files: {path}")
    digest = hashlib.sha256()
    for file in files:
        relative = file.relative_to(path).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required optional package is not installed: {distribution}") from exc
