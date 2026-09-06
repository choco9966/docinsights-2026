"""Blind, lossless source-region transcription for frozen primary solutions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .solution_records import (
    ensure_private_directory,
    primary_record_digest,
    write_private_atomic,
)

SOURCE_CHECK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["observed_blocks"],
    "properties": {
        "observed_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "heading_line",
                    "body_text",
                    "heading_legibility",
                    "body_legibility",
                    "contains_anchor_region",
                ],
                "properties": {
                    "heading_line": {"type": ["string", "null"]},
                    "body_text": {"type": ["string", "null"]},
                    "heading_legibility": {
                        "type": "string",
                        "enum": ["clear", "ambiguous", "unreadable"],
                    },
                    "body_legibility": {
                        "type": "string",
                        "enum": ["clear", "ambiguous", "unreadable"],
                    },
                    "contains_anchor_region": {"type": "boolean"},
                },
            },
        }
    },
}

_BLIND_PROMPT = (
    "Act only as a visual source transcription checker. Three images are supplied in order: "
    "the lossless full page, a full-width context crop, and a tight crop marking the source "
    "region selected independently before this invocation. Inspect pixels only. Transcribe each "
    "visible source block in the context crop as its complete heading line and literal body text. "
    "A source block may start with an inline identifier followed by a colon, with body words on "
    "the same line and no separate typographic title. In that case, heading_line is that exact "
    "first line, including the identifier and any same-line body words; do not mark the heading "
    "missing merely because it is inline. body_text contains the literal body after the "
    "identifier delimiter, including body words on the first line and subsequent lines. "
    "For each block, report whether its heading and body are clear, ambiguous, or unreadable, and "
    "whether the block contains the region shown by the tight crop. Preserve case and punctuation. "
    "Do not solve or discuss a document question, infer unreadable characters, use OCR, report "
    "coordinates, explain reasoning, or add diagnostics. Return only the required JSON object."
)
_OBSERVED_FIELDS = {
    "heading_line",
    "body_text",
    "heading_legibility",
    "body_legibility",
    "contains_anchor_region",
}
_LEGIBILITY = {"clear", "ambiguous", "unreadable"}
_SHA256_LENGTH = 64


class SourceRegionCheckError(ValueError):
    """A source-region check violated its frozen, candidate-blind contract."""


CheckerInvoker = Callable[..., Mapping[str, Any]]


def check_source_regions(
    *,
    frozen_record: Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
    ocr_geometry: Sequence[Mapping[str, Any]],
    source_pdf_path: Path | str,
    output_dir: Path | str,
    checker_config: Mapping[str, Any],
    invoke: CheckerInvoker,
) -> list[dict[str, Any]]:
    """Transcribe each frozen OCR-located region from fresh lossless PDF pixels.

    ``invoke`` receives only the blind prompt, three image paths, schema, private output
    directory, and checker configuration. It is called once per region and must return
    ``response``, ``raw_response_path``, ``artifact_paths``, ``model``, and ``method``.
    """
    if not isinstance(frozen_record, Mapping):
        raise SourceRegionCheckError("frozen_record must be an object")
    regions = frozen_record.get("evidence_regions")
    if not isinstance(regions, list) or not regions:
        raise SourceRegionCheckError("frozen_record.evidence_regions must be non-empty")
    provenance = frozen_record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise SourceRegionCheckError("frozen_record.provenance must be an object")
    pdf_sha256 = _required_sha256(provenance.get("pdf_sha256"), "provenance.pdf_sha256")
    primary_response_sha256 = _required_sha256(
        provenance.get("output_sha256"), "provenance.output_sha256"
    )
    try:
        primary_sha256 = primary_record_digest(frozen_record)
    except (TypeError, ValueError) as error:
        raise SourceRegionCheckError("frozen_record is not a normalized primary record") from error

    page_texts = _page_text_index(pages)
    geometries = _geometry_index(ocr_geometry)
    config = _checker_config(checker_config)
    config_sha256 = _sha256_bytes(_canonical_bytes(config))
    selector_config = {
        "anchor_padding_pixels": config["anchor_padding_pixels"],
        "context_vertical_padding_pixels": config["context_vertical_padding_pixels"],
        "locator": "unique-exact-public-ocr-anchor-v1",
    }
    prompt_sha256 = _sha256_bytes(_BLIND_PROMPT.encode("utf-8"))
    schema_sha256 = _sha256_bytes(_canonical_bytes(SOURCE_CHECK_SCHEMA))

    pdf_path = _regular_file(source_pdf_path, "source PDF")
    if _sha256_file(pdf_path) != pdf_sha256:
        raise SourceRegionCheckError("source PDF SHA-256 differs from the frozen primary")
    renderer_path = _regular_file(config["renderer_path"], "renderer executable")
    renderer_sha256 = _sha256_file(renderer_path)
    if renderer_sha256 != config["renderer_sha256"]:
        raise SourceRegionCheckError("renderer executable SHA-256 mismatch")
    checker_path = _regular_file(config["codex_executable_path"], "checker executable")
    checker_executable_sha256 = _sha256_file(checker_path)
    if checker_executable_sha256 != config["codex_executable_sha256"]:
        raise SourceRegionCheckError("checker executable SHA-256 mismatch")
    _verify_renderer_version(renderer_path, config["renderer_version"])

    destination = ensure_private_directory(output_dir)
    proofs: list[dict[str, Any]] = []
    for region_index, raw_region in enumerate(regions):
        region = _region(raw_region, region_index)
        page_number = region["page"]
        if page_number not in page_texts or page_number not in geometries:
            raise SourceRegionCheckError("evidence region page lacks OCR text or geometry")
        geometry = geometries[page_number]
        anchor_bbox, line_indices = _anchor_selection(
            region["ocr_anchor"], page_texts[page_number], geometry
        )
        context_bbox = {
            "left": 0,
            "top": max(0, anchor_bbox["top"] - config["context_vertical_padding_pixels"]),
            "right": geometry["width"],
            "bottom": min(
                geometry["height"],
                anchor_bbox["bottom"] + config["context_vertical_padding_pixels"],
            ),
        }
        anchor_crop_bbox = _padded_bbox(
            anchor_bbox,
            config["anchor_padding_pixels"],
            geometry["width"],
            geometry["height"],
        )
        selector = {
            **selector_config,
            "page": page_number,
            "ocr_anchor_sha256": _sha256_bytes(region["ocr_anchor"].encode("utf-8")),
            "line_indices": line_indices,
            "anchor_bbox": anchor_bbox,
            "context_bbox": context_bbox,
            "anchor_crop_bbox": anchor_crop_bbox,
        }
        selector_sha256 = _sha256_bytes(_canonical_bytes(selector))

        region_dir = ensure_private_directory(destination / f"region-{region_index:04d}")
        primary_path = region_dir / "primary-record.json"
        prompt_path = region_dir / "prompt.txt"
        schema_path = region_dir / "schema.json"
        config_path = region_dir / "checker-config.json"
        selector_path = region_dir / "selector-config.json"
        write_private_atomic(primary_path, _canonical_bytes(frozen_record))
        write_private_atomic(prompt_path, _BLIND_PROMPT)
        write_private_atomic(schema_path, _canonical_bytes(SOURCE_CHECK_SCHEMA))
        write_private_atomic(config_path, _canonical_bytes(config))
        write_private_atomic(selector_path, _canonical_bytes(selector))

        page_path = _render_page(
            renderer_path=renderer_path,
            pdf_path=pdf_path,
            page_number=page_number,
            dpi=config["dpi"],
            image_format=config["ocr_input_format"],
            output_dir=region_dir,
        )
        page_image_sha256 = _sha256_file(page_path)
        if page_image_sha256 != geometry["ocr_image_sha256"]:
            raise SourceRegionCheckError(
                "lossless rendered page SHA-256 differs from frozen OCR input"
            )
        context_path = region_dir / "context-crop.png"
        anchor_path = region_dir / "anchor-crop.png"
        _write_crop(page_path, context_bbox, geometry["width"], geometry["height"], context_path)
        _write_crop(
            page_path,
            anchor_crop_bbox,
            geometry["width"],
            geometry["height"],
            anchor_path,
        )
        context_crop_sha256 = _sha256_file(context_path)
        anchor_crop_sha256 = _sha256_file(anchor_path)

        invocation_dir = ensure_private_directory(region_dir / "invocation")
        raw_result = invoke(
            prompt=_BLIND_PROMPT,
            images=(page_path, context_path, anchor_path),
            output_schema=SOURCE_CHECK_SCHEMA,
            output_dir=invocation_dir,
            checker_config=config,
        )
        _verify_frozen_contract(
            frozen_record=frozen_record,
            primary_sha256=primary_sha256,
            config=config,
            config_sha256=config_sha256,
            selector=selector,
            selector_sha256=selector_sha256,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
            artifact_hashes={
                primary_path: primary_sha256,
                prompt_path: prompt_sha256,
                schema_path: schema_sha256,
                config_path: config_sha256,
                selector_path: selector_sha256,
            },
        )
        invocation = _invocation_result(raw_result, invocation_dir, config)
        observations = _observed_blocks(invocation["response"])
        status, evidence, uncertainties = _terminal_result(observations, page_number)

        if _sha256_file(pdf_path) != pdf_sha256:
            raise SourceRegionCheckError("source PDF changed during source checking")
        if _sha256_file(renderer_path) != renderer_sha256:
            raise SourceRegionCheckError("renderer executable changed during source checking")
        if _sha256_file(checker_path) != checker_executable_sha256:
            raise SourceRegionCheckError("checker executable changed during source checking")

        named_artifacts = [
            _artifact("source_pdf", pdf_path),
            _artifact("renderer_executable", renderer_path),
            _artifact("primary_record", primary_path),
            _artifact("checker_prompt", prompt_path),
            _artifact("checker_schema", schema_path),
            _artifact("checker_config", config_path),
            _artifact("selector_config", selector_path),
            _artifact("render_command", region_dir / "render-command.json"),
            _artifact("page_image", page_path),
            _artifact("context_crop", context_path),
            _artifact("anchor_crop", anchor_path),
            _artifact("checker_executable", checker_path),
        ]
        all_artifacts = _unique_artifacts(named_artifacts + invocation["artifacts"])
        proof = {
            "region_index": region_index,
            "page": page_number,
            "ocr_anchor": region["ocr_anchor"],
            "ocr_anchor_sha256": _sha256_bytes(region["ocr_anchor"].encode("utf-8")),
            "primary_record_sha256": primary_sha256,
            "primary_response_sha256": primary_response_sha256,
            "status": status,
            "evidence": evidence,
            "observed_candidates": observations,
            "source_uncertainties": uncertainties,
            "pdf_sha256": pdf_sha256,
            "renderer_sha256": renderer_sha256,
            "checker_executable_sha256": checker_executable_sha256,
            "selector_sha256": selector_sha256,
            "prompt_sha256": prompt_sha256,
            "output_schema_sha256": schema_sha256,
            "checker_config_sha256": config_sha256,
            "page_image_sha256": page_image_sha256,
            "ocr_image_sha256": geometry["ocr_image_sha256"],
            "context_crop_sha256": context_crop_sha256,
            "anchor_crop_sha256": anchor_crop_sha256,
            "raw_response_sha256": invocation["raw_response_sha256"],
            "model": invocation["model"],
            "method": invocation["method"],
            "anchor_bbox": anchor_bbox,
            "context_bbox": context_bbox,
            "anchor_crop_bbox": anchor_crop_bbox,
            "pdf_artifact": _artifact("source_pdf", pdf_path),
            "renderer_artifact": _artifact("renderer_executable", renderer_path),
            "page_image_artifact": _artifact("page_image", page_path),
            "context_crop_artifact": _artifact("context_crop", context_path),
            "anchor_crop_artifact": _artifact("anchor_crop", anchor_path),
            "checker_executable_artifact": _artifact("checker_executable", checker_path),
            "raw_response_artifact": invocation["raw_response_artifact"],
            "artifacts": all_artifacts,
        }
        proofs.append(proof)
    return proofs


def _verify_frozen_contract(
    *,
    frozen_record: Mapping[str, Any],
    primary_sha256: str,
    config: Mapping[str, Any],
    config_sha256: str,
    selector: Mapping[str, Any],
    selector_sha256: str,
    prompt_sha256: str,
    schema_sha256: str,
    artifact_hashes: Mapping[Path, str],
) -> None:
    try:
        in_memory_hashes = {
            "primary record": primary_record_digest(frozen_record),
            "checker config": _sha256_bytes(_canonical_bytes(config)),
            "selector config": _sha256_bytes(_canonical_bytes(selector)),
            "checker prompt": _sha256_bytes(_BLIND_PROMPT.encode("utf-8")),
            "checker schema": _sha256_bytes(_canonical_bytes(SOURCE_CHECK_SCHEMA)),
        }
    except (TypeError, ValueError) as error:
        raise SourceRegionCheckError("frozen checker input changed during invocation") from error
    expected = {
        "primary record": primary_sha256,
        "checker config": config_sha256,
        "selector config": selector_sha256,
        "checker prompt": prompt_sha256,
        "checker schema": schema_sha256,
    }
    for name, digest in in_memory_hashes.items():
        if digest != expected[name]:
            raise SourceRegionCheckError(f"{name} changed during checker invocation")
    for path, digest in artifact_hashes.items():
        if _sha256_file(path) != digest:
            raise SourceRegionCheckError(f"{path.name} changed during checker invocation")


def _checker_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = _json_object(value, "checker_config")
    required = {
        "model",
        "method",
        "renderer_path",
        "renderer_sha256",
        "renderer_version",
        "dpi",
        "ocr_input_format",
        "codex_executable_path",
        "codex_executable_sha256",
    }
    missing = sorted(required - set(config))
    if missing:
        raise SourceRegionCheckError(f"checker_config missing fields: {', '.join(missing)}")
    for field in ("model", "method", "renderer_path", "renderer_version", "codex_executable_path"):
        if not isinstance(config[field], str) or not config[field]:
            raise SourceRegionCheckError(f"checker_config.{field} must be non-empty text")
    for field in ("renderer_sha256", "codex_executable_sha256"):
        config[field] = _required_sha256(config[field], f"checker_config.{field}")
    if config["dpi"] != 175 or config["ocr_input_format"] != "png":
        raise SourceRegionCheckError("source checker requires frozen 175 dpi PNG rendering")
    for field, default, maximum in (
        ("context_vertical_padding_pixels", 200, 4096),
        ("anchor_padding_pixels", 12, 256),
    ):
        item = config.get(field, default)
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= maximum:
            raise SourceRegionCheckError(f"checker_config.{field} is invalid")
        config[field] = item
    return config


def _page_text_index(pages: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    indexed: dict[int, str] = {}
    for page in pages:
        if not isinstance(page, Mapping):
            raise SourceRegionCheckError("public OCR page must be an object")
        number, text = page.get("page_number"), page.get("text")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or number in indexed
            or not isinstance(text, str)
        ):
            raise SourceRegionCheckError("public OCR pages are malformed or duplicated")
        indexed[number] = text
    return indexed


def _geometry_index(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SourceRegionCheckError("OCR geometry page must be an object")
        number, width, height, lines = (
            row.get("page_number"),
            row.get("width"),
            row.get("height"),
            row.get("lines"),
        )
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
            raise SourceRegionCheckError("OCR geometry pages are malformed or duplicated")
        digest = _required_sha256(row.get("ocr_image_sha256"), "ocr_image_sha256")
        indexed[number] = {
            "page_number": number,
            "width": width,
            "height": height,
            "ocr_image_sha256": digest,
            "lines": lines,
        }
    return indexed


def _region(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"page", "ocr_anchor"}:
        raise SourceRegionCheckError(f"evidence_regions[{index}] has invalid fields")
    page, anchor = value["page"], value["ocr_anchor"]
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise SourceRegionCheckError(f"evidence_regions[{index}].page is invalid")
    if not isinstance(anchor, str) or not anchor.strip():
        raise SourceRegionCheckError(f"evidence_regions[{index}].ocr_anchor is invalid")
    return {"page": page, "ocr_anchor": anchor}


def _anchor_selection(
    anchor: str, page_text: str, geometry: Mapping[str, Any]
) -> tuple[dict[str, int], list[int]]:
    if page_text.count(anchor) != 1:
        raise SourceRegionCheckError("OCR anchor must occur exactly once on its cited page")
    start = page_text.index(anchor)
    end = start + len(anchor)
    text_lines = page_text.splitlines()
    selected_indices: list[int] = []
    offset = 0
    for index, line_text in enumerate(text_lines):
        line_start, line_end = offset, offset + len(line_text)
        if max(start, line_start) < min(end, line_end):
            selected_indices.append(index)
        offset = line_end + 1
    if not selected_indices:
        raise SourceRegionCheckError("OCR anchor does not cover a text line")

    by_index: dict[int, Mapping[str, Any]] = {}
    for line in geometry["lines"]:
        if not isinstance(line, Mapping):
            raise SourceRegionCheckError("OCR geometry line must be an object")
        line_index = line.get("line_index")
        if (
            isinstance(line_index, bool)
            or not isinstance(line_index, int)
            or line_index in by_index
        ):
            raise SourceRegionCheckError("OCR geometry line indices are malformed or duplicated")
        by_index[line_index] = line
    boxes = []
    for index in selected_indices:
        line = by_index.get(index)
        if line is None or line.get("text") != text_lines[index]:
            raise SourceRegionCheckError("OCR geometry text does not match the frozen anchor line")
        boxes.append(_xywh_bbox(line.get("bbox"), geometry["width"], geometry["height"]))
    return (
        {
            "left": min(box["left"] for box in boxes),
            "top": min(box["top"] for box in boxes),
            "right": max(box["right"] for box in boxes),
            "bottom": max(box["bottom"] for box in boxes),
        },
        selected_indices,
    )


def _xywh_bbox(value: Any, width: int, height: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"left", "top", "width", "height"}:
        raise SourceRegionCheckError("stored OCR bbox is malformed")
    coordinates = [value[field] for field in ("left", "top", "width", "height")]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in coordinates):
        raise SourceRegionCheckError("stored OCR bbox coordinates must be integers")
    left, top, box_width, box_height = coordinates
    right, bottom = left + box_width, top + box_height
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise SourceRegionCheckError("stored OCR bbox extends beyond the lossless page")
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _padded_bbox(bbox: Mapping[str, int], padding: int, width: int, height: int) -> dict[str, int]:
    return {
        "left": max(0, bbox["left"] - padding),
        "top": max(0, bbox["top"] - padding),
        "right": min(width, bbox["right"] + padding),
        "bottom": min(height, bbox["bottom"] + padding),
    }


def _verify_renderer_version(path: Path, expected: str) -> None:
    try:
        result = subprocess.run(
            [str(path), "-v"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SourceRegionCheckError("could not execute the frozen renderer") from error
    lines = (result.stderr or result.stdout).splitlines()
    if result.returncode != 0 or not lines or lines[0] != expected:
        raise SourceRegionCheckError("renderer version differs from frozen configuration")


def _render_page(
    *,
    renderer_path: Path,
    pdf_path: Path,
    page_number: int,
    dpi: int,
    image_format: str,
    output_dir: Path,
) -> Path:
    prefix = output_dir / "lossless-page"
    argv = [
        str(renderer_path),
        f"-{image_format}",
        "-r",
        str(dpi),
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        str(pdf_path),
        str(prefix),
    ]
    write_private_atomic(output_dir / "render-command.json", _canonical_bytes({"argv": argv}))
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SourceRegionCheckError("lossless page renderer failed") from error
    if result.returncode != 0:
        raise SourceRegionCheckError("lossless page renderer returned a failure")
    rendered = list(output_dir.glob(f"lossless-page-*.{image_format}"))
    if len(rendered) != 1:
        raise SourceRegionCheckError("lossless renderer did not produce exactly one cited page")
    path = rendered[0]
    if path.is_symlink() or not path.is_file():
        raise SourceRegionCheckError("lossless rendered page is not a regular file")
    path.chmod(0o600)
    return path.resolve()


def _write_crop(
    image_path: Path,
    bbox: Mapping[str, int],
    width: int,
    height: int,
    output_path: Path,
) -> None:
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.size != (width, height):
                raise SourceRegionCheckError("lossless page dimensions differ from OCR geometry")
            crop = image.crop((bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]))
            buffer = BytesIO()
            crop.save(buffer, format="PNG", compress_level=9, optimize=False)
    except SourceRegionCheckError:
        raise
    except (OSError, ValueError) as error:
        raise SourceRegionCheckError("could not read or crop the lossless page") from error
    write_private_atomic(output_path, buffer.getvalue())


def _invocation_result(
    value: Any, invocation_dir: Path, checker_config: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceRegionCheckError("checker invocation result must be an object")
    required = {"response", "raw_response_path", "artifact_paths", "model", "method"}
    if set(value) != required:
        raise SourceRegionCheckError("checker invocation result has invalid fields")
    raw_path = _contained_artifact(value["raw_response_path"], invocation_dir)
    paths = value["artifact_paths"]
    if not isinstance(paths, Sequence) or isinstance(paths, str | bytes):
        raise SourceRegionCheckError("checker artifact_paths must be a sequence")
    artifacts: list[dict[str, str]] = []
    seen: set[Path] = set()
    for item in paths:
        path = _contained_artifact(item, invocation_dir)
        if path in seen:
            raise SourceRegionCheckError("checker artifact paths must be unique")
        seen.add(path)
        kind = "raw_response" if path == raw_path else f"checker_artifact:{path.name}"
        artifacts.append(_artifact(kind, path))
    if raw_path not in seen:
        raise SourceRegionCheckError("raw checker response must be listed as an artifact")
    raw_bytes = raw_path.read_bytes()
    try:
        parsed = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceRegionCheckError("raw checker response is not valid JSON") from error
    if parsed != value["response"]:
        raise SourceRegionCheckError("raw checker response differs from parsed response")
    model, method = value["model"], value["method"]
    if not isinstance(model, str) or not model or not isinstance(method, str) or not method:
        raise SourceRegionCheckError("checker model and method must be non-empty")
    if model != checker_config["model"] or method != checker_config["method"]:
        raise SourceRegionCheckError("checker model or method differs from frozen config")
    raw_sha256 = _sha256_bytes(raw_bytes)
    return {
        "response": value["response"],
        "raw_response_sha256": raw_sha256,
        "raw_response_artifact": _artifact("raw_response", raw_path),
        "artifacts": artifacts,
        "model": model,
        "method": method,
    }


def _observed_blocks(response: Any) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != {"observed_blocks"}:
        raise SourceRegionCheckError("checker response has invalid fields")
    blocks = response["observed_blocks"]
    if not isinstance(blocks, list):
        raise SourceRegionCheckError("observed_blocks must be a list")
    normalized: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping) or set(block) != _OBSERVED_FIELDS:
            raise SourceRegionCheckError("observed source block has invalid fields")
        heading, body = block["heading_line"], block["body_text"]
        if heading is not None and not isinstance(heading, str):
            raise SourceRegionCheckError("observed heading_line must be text or null")
        if body is not None and not isinstance(body, str):
            raise SourceRegionCheckError("observed body_text must be text or null")
        for field in ("heading_legibility", "body_legibility"):
            if block[field] not in _LEGIBILITY:
                raise SourceRegionCheckError(f"observed {field} is invalid")
        if not isinstance(block["contains_anchor_region"], bool):
            raise SourceRegionCheckError("observed anchor membership must be boolean")
        normalized.append(dict(block))
    return normalized


def _terminal_result(
    observations: list[dict[str, Any]], page_number: int
) -> tuple[str, dict[str, Any], list[str]]:
    anchored = [item for item in observations if item["contains_anchor_region"]]
    if len(anchored) == 1:
        block = anchored[0]
        evidence_id = _heading_id(block["heading_line"] or "")
        body = block["body_text"]
        if (
            block["heading_legibility"] == "clear"
            and block["body_legibility"] == "clear"
            and evidence_id is not None
            and isinstance(body, str)
            and body.strip()
        ):
            return (
                "fully_grounded",
                {"id": evidence_id, "page": page_number, "quote": body},
                [],
            )

    uncertainties: list[str] = []
    if not anchored:
        uncertainties.append("Checker found no source block containing the selected region.")
    elif len(anchored) > 1:
        uncertainties.append("Checker found multiple source blocks containing the selected region.")
    else:
        block = anchored[0]
        if block["heading_legibility"] != "clear":
            uncertainties.append(
                f"Source heading is {block['heading_legibility']} in the lossless pixels."
            )
        if block["body_legibility"] != "clear":
            uncertainties.append(
                f"Source body is {block['body_legibility']} in the lossless pixels."
            )
        if (
            block["heading_legibility"] == "clear"
            and _heading_id(block["heading_line"] or "") is None
        ):
            uncertainties.append("Source heading lacks a clear colon-before-whitespace delimiter.")
        if block["body_legibility"] == "clear" and not (
            isinstance(block["body_text"], str) and block["body_text"].strip()
        ):
            uncertainties.append("Source body has no readable literal transcription.")
    readable = {
        item["body_text"]
        for item in anchored
        if item["body_legibility"] == "clear"
        and isinstance(item["body_text"], str)
        and item["body_text"].strip()
    }
    quote = next(iter(readable)) if len(readable) == 1 else None
    if not uncertainties:
        uncertainties.append("Source evidence could not be uniquely grounded from the pixels.")
    return (
        "evidence_unresolved",
        {"id": None, "page": page_number, "quote": quote},
        uncertainties,
    )


def _heading_id(heading_line: str) -> str | None:
    for index, character in enumerate(heading_line):
        if (
            character == ":"
            and index > 0
            and (index + 1 == len(heading_line) or heading_line[index + 1].isspace())
        ):
            candidate = heading_line[:index]
            if candidate and not any(character.isspace() for character in candidate):
                return candidate
    return None


def _contained_artifact(value: Any, root: Path) -> Path:
    path = _regular_file(value, "checker artifact")
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise SourceRegionCheckError("checker artifact escapes its invocation directory") from error
    return path


def _regular_file(value: Any, field: str) -> Path:
    if not isinstance(value, str | Path):
        raise SourceRegionCheckError(f"{field} path is invalid")
    raw = Path(value)
    if raw.is_symlink() or not raw.is_file():
        raise SourceRegionCheckError(f"{field} must be a regular non-symlink file")
    return raw.resolve()


def _artifact(kind: str, path: Path) -> dict[str, str]:
    return {"kind": kind, "path": str(path), "sha256": _sha256_file(path)}


def _unique_artifacts(values: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: list[dict[str, str]] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value["path"]).resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(value)
    return unique


def _json_object(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceRegionCheckError(f"{field} must be an object")
    normalized = dict(value)
    try:
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SourceRegionCheckError(f"{field} must contain JSON-safe values") from error
    return normalized


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _required_sha256(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise SourceRegionCheckError(f"{field} must be a lowercase SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )
