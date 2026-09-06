#!/usr/bin/env python3
"""Export or verify the exact PaddleOCR ONNX models used by Issue #24.

The native inputs are downloaded independently from their pinned public
Hugging Face revisions::

    python scripts/setup_solution_paddle_ocr.py \
        --model-dir data/issue24/ppocr-models

Create a conversion-only Python 3.11 environment, install
``requirements/solution-paddle-onnx-export.txt``, then run::

    python scripts/setup_solution_paddle_onnx.py

Existing outputs are checksum-verified and never overwritten. A new export is
published only when it is byte-identical to the output used by the 60-page
quality gate. The known hashes came from the gated 2026-09-07 conversion; that
expensive conversion was not rerun when this reproducibility script was added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PADDLE2ONNX_VERSION = "2.1.0"
OPSET_VERSION = 11


@dataclass(frozen=True)
class ExportSpec:
    role: str
    repository: str
    revision: str
    source_json_sha256: str
    source_params_sha256: str
    output_sha256: str


EXPORTS = (
    ExportSpec(
        role="detector",
        repository="PaddlePaddle/PP-OCRv5_mobile_det",
        revision="0d63e78e2b680928f6b1747d76a08db6e645efb7",
        source_json_sha256=("05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c"),
        source_params_sha256=("afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b"),
        output_sha256=("d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb"),
    ),
    ExportSpec(
        role="recognizer",
        repository="PaddlePaddle/en_PP-OCRv5_mobile_rec",
        revision="267c36e24c331595590fe7bd72bde2436fd286f2",
        source_json_sha256=("fd1b6ec722ea841a72d3ba43e527df1d1066d5d7808e0503ee3eec7265188753"),
        source_params_sha256=("3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b"),
        output_sha256=("4212d483f00f1c8617ba143ba36731e361d8307f49b5fae830d828f64b2162a2"),
    ),
)


class ExportSetupError(RuntimeError):
    """Raised when a pinned source, converter, or output cannot be verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise ExportSetupError(f"required {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ExportSetupError(
            f"{label} digest mismatch for {path}: expected {expected}, got {actual}"
        )


def _verified_output(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    if not path.is_file():
        raise ExportSetupError(f"ONNX destination is not a regular file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ExportSetupError(
            f"existing ONNX digest mismatch; refusing to replace {path}: "
            f"expected {expected}, got {actual}"
        )
    return True


def _resolve_converter(converter: Path) -> Path:
    if converter.is_file():
        return converter.resolve()
    discovered = shutil.which(str(converter))
    if discovered is None:
        raise ExportSetupError(f"paddle2onnx executable is unavailable: {converter}")
    return Path(discovered).resolve()


def _converter_version(converter: Path) -> str:
    completed = subprocess.run(
        [str(converter), "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    text = "\n".join((completed.stdout, completed.stderr))
    if completed.returncode != 0:
        raise ExportSetupError(f"could not identify paddle2onnx version: {text.strip()}")
    versions = re.findall(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", text)
    if PADDLE2ONNX_VERSION not in versions:
        observed = ", ".join(versions) if versions else text.strip() or "unknown"
        raise ExportSetupError(
            f"paddle2onnx version mismatch: expected {PADDLE2ONNX_VERSION}, observed {observed}"
        )
    return PADDLE2ONNX_VERSION


def _command(converter: Path, source: Path, output: Path) -> list[str]:
    return [
        str(converter),
        "--model_dir",
        str(source.resolve()),
        "--model_filename",
        "inference.json",
        "--params_filename",
        "inference.pdiparams",
        "--save_file",
        str(output.resolve()),
        "--opset_version",
        str(OPSET_VERSION),
        "--enable_onnx_checker",
        "True",
        "--optimize_tool",
        "None",
    ]


def _convert(converter: Path, source: Path, destination: Path, expected: str) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary_dir:
        temporary = Path(temporary_dir) / destination.name
        command = _command(converter, source, temporary)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ExportSetupError(f"paddle2onnx failed for {source.name}: {detail[-2000:]}")
        if not temporary.is_file():
            raise ExportSetupError(f"paddle2onnx did not create the expected output: {temporary}")
        actual = sha256_file(temporary)
        if actual != expected:
            raise ExportSetupError(
                "fresh ONNX export did not match the gated digest for "
                f"{source.name}: expected {expected}, got {actual}; output was not published"
            )
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _verified_output(destination, expected)
        return [str(destination) if item == str(temporary.resolve()) else item for item in command]


def prepare_exports(
    source_root: Path,
    output_root: Path,
    *,
    converter: Path,
    verify_only: bool,
    specs: Iterable[ExportSpec] = EXPORTS,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    missing: list[tuple[ExportSpec, Path, Path]] = []
    for spec in specs:
        source = source_root / spec.role
        source_json = source / "inference.json"
        source_params = source / "inference.pdiparams"
        _verify_file(source_json, spec.source_json_sha256, label="source model graph")
        _verify_file(source_params, spec.source_params_sha256, label="source model parameters")
        destination = output_root / spec.role / "inference.onnx"
        record: dict[str, object] = {
            "role": spec.role,
            "repository": spec.repository,
            "revision": spec.revision,
            "source": {
                "model_dir": str(source.resolve()),
                "inference_json_sha256": spec.source_json_sha256,
                "inference_pdiparams_sha256": spec.source_params_sha256,
            },
            "output_path": str(destination.resolve()),
            "output_sha256": spec.output_sha256,
        }
        if _verified_output(destination, spec.output_sha256):
            record["status"] = "verified_existing"
        elif verify_only:
            raise ExportSetupError(f"required ONNX export is missing: {destination}")
        else:
            missing.append((spec, source, destination))
        records.append(record)

    converter_record: dict[str, object] = {
        "required_version": PADDLE2ONNX_VERSION,
        "path": None,
        "version": None,
    }
    if missing:
        resolved_converter = _resolve_converter(converter)
        version = _converter_version(resolved_converter)
        converter_record.update(path=str(resolved_converter), version=version)
        records_by_role = {str(record["role"]): record for record in records}
        for spec, source, destination in missing:
            command = _convert(resolved_converter, source, destination, spec.output_sha256)
            records_by_role[spec.role].update(status="converted", command=command)

    return {
        "schema_version": "issue24-paddle-onnx-export-v1",
        "source_model_root": str(source_root.resolve()),
        "output_model_root": str(output_root.resolve()),
        "converter": converter_record,
        "settings": {
            "opset_version": OPSET_VERSION,
            "enable_onnx_checker": True,
            "optimize_tool": "None",
        },
        "fresh_byte_identical_reexport_verified_this_run": any(
            record.get("status") == "converted" for record in records
        ),
        "reproducibility_limitation": (
            "The expected output hashes came from the gated 2026-09-07 export. "
            "No fresh expensive conversion was run when this setup script was introduced; "
            "future exports are published only if byte-identical."
        ),
        "exports": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=Path("data/issue24/ppocr-models"),
        help="root containing detector/ and recognizer/ native Paddle models",
    )
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        default=Path("data/issue24/ppocr-onnx-models"),
        help="root for the gated detector/ and recognizer/ ONNX models",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("paddle2onnx"),
        help="paddle2onnx 2.1.0 executable from the conversion-only environment",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify pinned sources and outputs without invoking paddle2onnx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = prepare_exports(
            args.source_model_dir,
            args.output_model_dir,
            converter=args.converter,
            verify_only=args.verify_only,
        )
    except (OSError, subprocess.SubprocessError, ExportSetupError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
