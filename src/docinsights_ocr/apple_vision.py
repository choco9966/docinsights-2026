"""Adapter for the repository's Apple Vision OCR executable."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .models import BoundingBox, Line, Page


@dataclass(frozen=True, slots=True)
class _Observation:
    text: str
    confidence: float
    bbox: BoundingBox


@dataclass(slots=True)
class _Row:
    anchor_top: float
    observations: list[_Observation] = field(default_factory=list)


class AppleVisionEngine:
    """Run Apple Vision OCR and convert its JSON output to the common schema."""

    def __init__(
        self,
        *,
        executable: str | Path = "tools/apple_vision_ocr.swift",
        language: str = "en-US",
        mode: str = "accurate",
        timeout_seconds: float | None = None,
    ) -> None:
        if mode not in {"accurate", "fast"}:
            raise ValueError("mode must be 'accurate' or 'fast'")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = str(executable)
        self.language = language
        self.mode = mode
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "apple-vision"

    @property
    def confidence_kind(self) -> str:
        return "vision_observation_confidence_0_to_1"

    def recognize(self, image_path: str | Path, *, page_number: int = 1) -> Page:
        resolved_image = Path(image_path).resolve()
        result = subprocess.run(
            [
                self.executable,
                "--mode",
                self.mode,
                "--language",
                self.language,
                str(resolved_image),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        return parse_apple_vision_json(
            result.stdout,
            page_number=page_number,
            image_path=resolved_image,
        )


def parse_apple_vision_json(
    output: str,
    *,
    page_number: int = 1,
    image_path: str | Path | None = None,
) -> Page:
    """Parse one JSON-line result from ``apple_vision_ocr.swift``."""
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Apple Vision output must contain exactly one JSON object")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid Apple Vision JSON output") from exc
    if not isinstance(payload, dict):
        raise ValueError("Apple Vision output must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported Apple Vision schema_version")
    if "error" in payload:
        raise ValueError(f"Apple Vision returned an error: {payload['error']}")
    width = _positive_int(payload, "width")
    height = _positive_int(payload, "height")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Apple Vision observations must be a list")

    parsed_observations: list[_Observation] = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("Apple Vision observation must be an object")
        text = observation.get("text")
        confidence = observation.get("confidence")
        bbox = observation.get("bbox")
        if not isinstance(text, str) or not isinstance(confidence, (int, float)):
            raise ValueError("Apple Vision observation has invalid text or confidence")
        if not isinstance(bbox, dict):
            raise ValueError("Apple Vision observation bbox must be an object")
        x = _unit_float(bbox, "x")
        y = _unit_float(bbox, "y")
        box_width = _unit_float(bbox, "width")
        box_height = _unit_float(bbox, "height")
        if x + box_width > 1.000001 or y + box_height > 1.000001:
            raise ValueError("Apple Vision bbox extends beyond the image")
        parsed_observations.append(
            _Observation(
                text=text,
                confidence=float(confidence),
                bbox=BoundingBox(
                    left=round(x * width),
                    top=round(y * height),
                    width=round(box_width * width),
                    height=round(box_height * height),
                ),
            )
        )
    visible_observations = [
        observation
        for observation in parsed_observations
        if not _is_known_watermark(observation, width=width, height=height)
    ]
    parsed_lines = tuple(
        Line(
            page_number=page_number,
            text=observation.text,
            bbox=observation.bbox,
            confidence=observation.confidence,
            source_order=source_order,
        )
        for source_order, observation in enumerate(_reading_order(visible_observations))
    )
    resolved_path = Path(image_path).resolve() if image_path is not None else None
    return Page(
        number=page_number,
        lines=parsed_lines,
        image_path=resolved_path,
        width=width,
        height=height,
    )


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Apple Vision {key} must be a positive integer")
    return value


def _unit_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Apple Vision bbox {key} must be numeric")
    converted = float(value)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"Apple Vision bbox {key} must be between zero and one")
    return converted


def _reading_order(observations: list[_Observation]) -> tuple[_Observation, ...]:
    """Cluster slightly misaligned table cells into rows before sorting left-to-right."""
    rows: list[_Row] = []
    for observation in sorted(observations, key=lambda item: (item.bbox.top, item.bbox.left)):
        compatible: list[tuple[float, _Row]] = []
        for row in rows:
            typical_height = median(item.bbox.height for item in row.observations)
            tolerance = min(max(typical_height * 0.5, 2.0), 24.0)
            height_ratio = max(typical_height, observation.bbox.height) / max(
                min(typical_height, observation.bbox.height), 1
            )
            distance = abs(observation.bbox.top - row.anchor_top)
            if height_ratio <= 2.5 and distance <= tolerance:
                compatible.append((distance, row))
        if compatible:
            row = min(compatible, key=lambda candidate: candidate[0])[1]
            row.observations.append(observation)
            row.anchor_top = sum(item.bbox.top for item in row.observations) / len(row.observations)
        else:
            rows.append(
                _Row(
                    anchor_top=float(observation.bbox.top),
                    observations=[observation],
                )
            )
    return tuple(
        observation
        for row in sorted(rows, key=lambda item: item.anchor_top)
        for observation in sorted(row.observations, key=lambda item: item.bbox.left)
    )


def _is_known_watermark(
    observation: _Observation,
    *,
    width: int,
    height: int,
) -> bool:
    """Remove only the known oversized training-copy furniture observation."""
    normalized = " ".join(observation.text.upper().split())
    known_text = normalized in {"TRAINING COP", "TRAINING COPY"}
    oversized = observation.bbox.height >= height * 0.15 or observation.bbox.width >= width * 0.75
    return known_text and oversized
