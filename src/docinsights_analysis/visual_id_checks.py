"""Blind visual adjudication for narrowly scoped OCR evidence-ID mismatches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .solution_records import (
    ensure_private_directory,
    visual_evidence_candidates,
    write_private_atomic,
)

VISUAL_CHECK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["visible_headings"],
    "properties": {
        "visible_headings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "bbox", "legibility"],
                "properties": {
                    "id": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["left", "top", "right", "bottom"],
                        "properties": {
                            "left": {"type": "integer"},
                            "top": {"type": "integer"},
                            "right": {"type": "integer"},
                            "bottom": {"type": "integer"},
                        },
                    },
                    "legibility": {
                        "type": "string",
                        "enum": ["clear", "ambiguous", "unreadable"],
                    },
                },
            },
        }
    },
}

_BLIND_PROMPT = (
    "Act only as a visual transcription checker. The first supplied image is a full page and "
    "the second is a crop of one heading line from that page. Inspect pixels only. Return every "
    "visible heading identifier that overlaps the crop's source line. Copy each identifier "
    "exactly as visible, give its bounding box in full-page pixel coordinates, and classify "
    "legibility as clear, ambiguous, or unreadable. Do not solve or discuss any document task, "
    "infer missing characters, explain reasoning, or add diagnostics. Return only the required "
    "JSON object."
)
_SHA256_LENGTH = 64


class VisualIDCheckError(ValueError):
    """A visual evidence-ID check could not produce a unique, audit-bound proof."""


CheckerInvoker = Callable[..., Mapping[str, Any]]


def run_visual_id_checks(
    *,
    frozen_record: Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
    paddle_geometry: Sequence[Mapping[str, Any]],
    retained_images: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    checker_config: Mapping[str, Any],
    invoke: CheckerInvoker,
) -> list[dict[str, Any]]:
    """Adjudicate only contract-approved OCR/visual ID mismatches, once per candidate.

    ``invoke`` receives keyword arguments ``prompt``, ``images``, ``output_schema``,
    ``output_dir``, and ``checker_config``. It must freeze its raw invocation artifacts before
    returning ``response``, ``raw_response_path``, ``artifact_paths``, ``model``, and ``method``.
    """
    candidates = visual_evidence_candidates(frozen_record, pages)
    if not isinstance(candidates, list):
        raise VisualIDCheckError("visual evidence candidates must be a list")
    if not candidates:
        return []

    destination = ensure_private_directory(output_dir)
    normalized_config = _json_object(checker_config, "checker_config")
    padding = normalized_config.get("crop_padding_pixels", 12)
    if isinstance(padding, bool) or not isinstance(padding, int) or not 0 <= padding <= 256:
        raise VisualIDCheckError("crop_padding_pixels must be an integer from 0 through 256")
    config_sha256 = _sha256_bytes(_canonical_bytes(normalized_config))

    page_texts = _page_text_index(pages)
    geometries = _geometry_index(paddle_geometry)
    images = _image_index(retained_images)
    proofs: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(candidates, start=1):
        normalized_candidate = _candidate(candidate)
        page_number = normalized_candidate["page"]
        if page_number not in page_texts:
            raise VisualIDCheckError("visual candidate page is absent from public OCR pages")
        geometry = geometries.get(page_number)
        image_record = images.get(page_number)
        if geometry is None or image_record is None:
            raise VisualIDCheckError("visual candidate page lacks geometry or a retained image")

        image_path = image_record["path"]
        observed_image_sha256 = _sha256_file(image_path)
        if observed_image_sha256 != image_record["sha256"]:
            raise VisualIDCheckError("retained page image SHA-256 mismatch")

        heading_bbox = _heading_bbox(
            normalized_candidate,
            page_texts[page_number],
            geometry,
        )
        width = geometry["width"]
        height = geometry["height"]
        crop_bbox = {
            "left": max(0, heading_bbox["left"] - padding),
            "top": max(0, heading_bbox["top"] - padding),
            "right": min(width, heading_bbox["right"] + padding),
            "bottom": min(height, heading_bbox["bottom"] + padding),
        }

        candidate_dir = ensure_private_directory(destination / f"candidate-{ordinal:04d}")
        crop_path = candidate_dir / "heading-crop.png"
        _write_crop(image_path, crop_bbox, width, height, crop_path)
        crop_sha256 = _sha256_file(crop_path)
        invocation_dir = ensure_private_directory(candidate_dir / "invocation")

        raw_result = invoke(
            prompt=_BLIND_PROMPT,
            images=(image_path, crop_path),
            output_schema=VISUAL_CHECK_SCHEMA,
            output_dir=invocation_dir,
            checker_config=normalized_config,
        )
        result = _invocation_result(raw_result, invocation_dir, normalized_config)
        visible = _unique_clear_heading(
            result["response"], heading_bbox, page_width=width, page_height=height
        )
        if visible["id"] != normalized_candidate["id"]:
            raise VisualIDCheckError(
                "clear visible heading does not exactly match the frozen solver ID"
            )

        quote_sha256 = _sha256_bytes(normalized_candidate["quote"].encode("utf-8"))
        proof = {
            "id": normalized_candidate["id"],
            "visible_id": visible["id"],
            "ocr_id": normalized_candidate["ocr_id"],
            "page": page_number,
            "heading_line_index": normalized_candidate["heading_line_index"],
            "quote_sha256": quote_sha256,
            "page_image_sha256": observed_image_sha256,
            "crop_sha256": crop_sha256,
            "clear": True,
            "heading_bbox": heading_bbox,
            "crop_bbox": crop_bbox,
            "visible_bbox": visible["bbox"],
            "page_image_artifact": {
                "path": str(image_path),
                "sha256": observed_image_sha256,
            },
            "crop_artifact": {"path": str(crop_path), "sha256": crop_sha256},
            "checker_config_sha256": config_sha256,
            "raw_response_sha256": result["raw_response_sha256"],
            "raw_response_artifact": result["raw_response_artifact"],
            "artifacts": result["artifacts"],
            "model": result["model"],
            "method": result["method"],
        }
        proofs.append(proof)
    return proofs


def _candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualIDCheckError("visual evidence candidate must be an object")
    required = {"id", "ocr_id", "page", "quote", "heading_line_index"}
    if set(value) != required:
        raise VisualIDCheckError("visual evidence candidate has invalid fields")
    normalized: dict[str, Any] = {}
    for field in ("id", "ocr_id", "quote"):
        item = value[field]
        if not isinstance(item, str) or not item:
            raise VisualIDCheckError(f"visual candidate {field} must be a non-empty string")
        normalized[field] = item
    for field, minimum in (("page", 1), ("heading_line_index", 0)):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise VisualIDCheckError(f"visual candidate {field} is invalid")
        normalized[field] = item
    return normalized


def _page_text_index(pages: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    indexed: dict[int, str] = {}
    for page in pages:
        if not isinstance(page, Mapping):
            raise VisualIDCheckError("public OCR page must be an object")
        number = page.get("page_number")
        text = page.get("text")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or not isinstance(text, str)
            or number in indexed
        ):
            raise VisualIDCheckError("public OCR pages are malformed or duplicated")
        indexed[number] = text
    return indexed


def _geometry_index(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise VisualIDCheckError("Paddle geometry page must be an object")
        number = row.get("page_number")
        width = row.get("width")
        height = row.get("height")
        lines = row.get("lines")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or number in indexed
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
            or not isinstance(lines, list)
        ):
            raise VisualIDCheckError("Paddle geometry pages are malformed or duplicated")
        indexed[number] = {
            "page_number": number,
            "width": width,
            "height": height,
            "lines": lines,
        }
    return indexed


def _image_index(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise VisualIDCheckError("retained image metadata must be an object")
        number = row.get("page_number")
        raw_path = row.get("path")
        digest = row.get("sha256")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or number in indexed
            or not isinstance(raw_path, (str, Path))
            or not _is_sha256(digest)
        ):
            raise VisualIDCheckError("retained image metadata is malformed or duplicated")
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise VisualIDCheckError("retained image must be a regular non-symlink file")
        indexed[number] = {"page_number": number, "path": path.resolve(), "sha256": digest}
    return indexed


def _heading_bbox(
    candidate: Mapping[str, Any], page_text: str, geometry: Mapping[str, Any]
) -> dict[str, int]:
    line_index = candidate["heading_line_index"]
    text_lines = page_text.splitlines()
    if line_index >= len(text_lines):
        raise VisualIDCheckError("candidate heading line index is outside the OCR page")
    expected_line = text_lines[line_index]
    matching = [
        line
        for line in geometry["lines"]
        if isinstance(line, Mapping) and line.get("line_index") == line_index
    ]
    if len(matching) != 1:
        raise VisualIDCheckError("candidate heading does not have unique stored OCR geometry")
    line = matching[0]
    if line.get("text") != expected_line:
        raise VisualIDCheckError("stored OCR geometry text does not match the frozen page line")
    prefix = expected_line.partition(":")[0].strip()
    if prefix != candidate["ocr_id"]:
        raise VisualIDCheckError("candidate OCR ID does not match its frozen heading line")
    return _xywh_bbox(
        line.get("bbox"),
        width=geometry["width"],
        height=geometry["height"],
    )


def _xywh_bbox(value: Any, *, width: int, height: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"left", "top", "width", "height"}:
        raise VisualIDCheckError("stored OCR heading bbox is malformed")
    coordinates = [value[name] for name in ("left", "top", "width", "height")]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in coordinates):
        raise VisualIDCheckError("stored OCR heading bbox coordinates must be integers")
    left, top, box_width, box_height = coordinates
    right = left + box_width
    bottom = top + box_height
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise VisualIDCheckError("stored OCR heading bbox extends beyond the retained image")
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _write_crop(
    image_path: Path,
    bbox: Mapping[str, int],
    expected_width: int,
    expected_height: int,
    output_path: Path,
) -> None:
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.size != (expected_width, expected_height):
                raise VisualIDCheckError("retained image dimensions differ from Paddle geometry")
            crop = image.crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
            buffer = BytesIO()
            crop.save(buffer, format="PNG", compress_level=9, optimize=False)
    except VisualIDCheckError:
        raise
    except (OSError, ValueError) as error:
        raise VisualIDCheckError("could not read or crop the retained page image") from error
    write_private_atomic(output_path, buffer.getvalue())


def _invocation_result(
    value: Any, invocation_dir: Path, checker_config: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualIDCheckError("visual checker invocation result must be an object")
    required = {"response", "raw_response_path", "artifact_paths", "model", "method"}
    if set(value) != required:
        raise VisualIDCheckError("visual checker invocation result has invalid fields")
    response = value["response"]
    raw_path = _contained_artifact(value["raw_response_path"], invocation_dir)
    artifact_paths = value["artifact_paths"]
    if not isinstance(artifact_paths, Sequence) or isinstance(artifact_paths, (str, bytes)):
        raise VisualIDCheckError("visual checker artifact_paths must be a sequence")
    artifacts: list[dict[str, str]] = []
    seen: set[Path] = set()
    for raw_artifact_path in artifact_paths:
        artifact_path = _contained_artifact(raw_artifact_path, invocation_dir)
        if artifact_path in seen:
            raise VisualIDCheckError("visual checker artifact paths must be unique")
        seen.add(artifact_path)
        artifacts.append({"path": str(artifact_path), "sha256": _sha256_file(artifact_path)})
    if raw_path not in seen:
        raise VisualIDCheckError("raw visual checker response must be listed as an artifact")
    raw_bytes = raw_path.read_bytes()
    try:
        parsed_raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VisualIDCheckError("raw visual checker response is not valid JSON") from error
    if parsed_raw != response:
        raise VisualIDCheckError("raw visual checker response does not match parsed response")

    model = value["model"]
    method = value["method"]
    if not isinstance(model, str) or not model or not isinstance(method, str) or not method:
        raise VisualIDCheckError("visual checker model and method must be non-empty strings")
    configured_model = checker_config.get("model")
    if configured_model is not None and configured_model != model:
        raise VisualIDCheckError("visual checker model differs from frozen checker config")
    raw_sha256 = _sha256_bytes(raw_bytes)
    return {
        "response": response,
        "raw_response_sha256": raw_sha256,
        "raw_response_artifact": {"path": str(raw_path), "sha256": raw_sha256},
        "artifacts": artifacts,
        "model": model,
        "method": method,
    }


def _contained_artifact(value: Any, root: Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise VisualIDCheckError("visual checker artifact path is invalid")
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise VisualIDCheckError("visual checker artifact must be a regular non-symlink file")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise VisualIDCheckError(
            "visual checker artifact escapes its invocation directory"
        ) from error
    return resolved


def _unique_clear_heading(
    response: Any, heading_bbox: Mapping[str, int], *, page_width: int, page_height: int
) -> dict[str, Any]:
    if not isinstance(response, Mapping) or set(response) != {"visible_headings"}:
        raise VisualIDCheckError("visual checker response has invalid fields")
    headings = response["visible_headings"]
    if not isinstance(headings, list):
        raise VisualIDCheckError("visible_headings must be a list")
    overlapping: list[dict[str, Any]] = []
    for heading in headings:
        normalized = _visible_heading(heading, page_width, page_height)
        if _overlaps(normalized["bbox"], heading_bbox):
            overlapping.append(normalized)
    if len(overlapping) != 1:
        raise VisualIDCheckError("visual headings do not uniquely overlap the cited OCR heading")
    visible = overlapping[0]
    if visible["legibility"] != "clear":
        raise VisualIDCheckError("visual checker did not return a single clear heading")
    return visible


def _visible_heading(value: Any, page_width: int, page_height: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"id", "bbox", "legibility"}:
        raise VisualIDCheckError("visible heading has invalid fields")
    identifier = value["id"]
    legibility = value["legibility"]
    if not isinstance(identifier, str) or not identifier:
        raise VisualIDCheckError("visible heading ID must be a non-empty string")
    if legibility not in {"clear", "ambiguous", "unreadable"}:
        raise VisualIDCheckError("visible heading legibility is invalid")
    bbox = value["bbox"]
    if not isinstance(bbox, Mapping) or set(bbox) != {"left", "top", "right", "bottom"}:
        raise VisualIDCheckError("visible heading bbox is malformed")
    coordinates = [bbox[name] for name in ("left", "top", "right", "bottom")]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in coordinates):
        raise VisualIDCheckError("visible heading bbox coordinates must be integers")
    left, top, right, bottom = coordinates
    if not 0 <= left < right <= page_width or not 0 <= top < bottom <= page_height:
        raise VisualIDCheckError("visible heading bbox extends beyond the retained image")
    return {
        "id": identifier,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
        "legibility": legibility,
    }


def _overlaps(left: Mapping[str, int], right: Mapping[str, int]) -> bool:
    return min(left["right"], right["right"]) > max(left["left"], right["left"]) and min(
        left["bottom"], right["bottom"]
    ) > max(left["top"], right["top"])


def _json_object(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualIDCheckError(f"{field} must be an object")
    normalized = dict(value)
    try:
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise VisualIDCheckError(f"{field} must contain JSON-safe values") from error
    return normalized


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )
