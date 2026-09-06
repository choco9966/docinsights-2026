#!/usr/bin/env python3
"""Download and verify the pinned public RapidOCR PP-OCRv5 ONNX models.

Runtime setup::

    python3 -m venv .venv-solution-ocr
    .venv-solution-ocr/bin/pip install -r requirements/solution-ocr.txt
    .venv-solution-ocr/bin/python scripts/setup_solution_ocr.py \
        --model-dir data/issue24/rapidocr-models

Existing verified files are reused. An existing file with the wrong digest is
never replaced; the command fails and reports the mismatch. Downloads use no
credentials and become visible at the destination only after verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_MODEL_BYTES = 64 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 3


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    sha256: str


MODELS = (
    ModelSpec(
        filename="ch_PP-OCRv5_det_mobile.onnx",
        url=(
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
            "onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx"
        ),
        sha256="4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    ),
    ModelSpec(
        filename="en_PP-OCRv5_rec_mobile.onnx",
        url=(
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
            "onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx"
        ),
        sha256="c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
    ),
)


class ModelSetupError(RuntimeError):
    """Raised when a pinned model cannot be safely prepared."""


class BinaryReadable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tls_context() -> ssl.SSLContext:
    """Use the installed certifi bundle when the host Python lacks CA roots."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def verify_existing(path: Path, spec: ModelSpec) -> dict[str, object] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ModelSetupError(f"model destination is not a regular file: {path}")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ModelSetupError(
            f"existing model digest mismatch; refusing to replace {path}: "
            f"expected {spec.sha256}, got {actual}"
        )
    return model_record(path, spec, status="verified_existing")


def copy_response(response: BinaryReadable, destination: Path) -> None:
    copied = 0
    with destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > MAX_MODEL_BYTES:
                raise ModelSetupError(
                    f"download exceeded {MAX_MODEL_BYTES} bytes for {destination.name}"
                )
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())


def download_once(spec: ModelSpec, temporary: Path, context: ssl.SSLContext) -> None:
    request = Request(spec.url, headers={"User-Agent": "docinsights-solution-ocr/1"})
    with urlopen(request, timeout=60, context=context) as response:
        copy_response(response, temporary)


def model_record(path: Path, spec: ModelSpec, *, status: str) -> dict[str, object]:
    return {
        "filename": spec.filename,
        "path": str(path.resolve()),
        "url": spec.url,
        "sha256": spec.sha256,
        "bytes": path.stat().st_size,
        "status": status,
    }


def ensure_model(
    model_dir: Path,
    spec: ModelSpec,
    *,
    verify_only: bool,
    context: ssl.SSLContext,
) -> dict[str, object]:
    destination = model_dir / spec.filename
    existing = verify_existing(destination, spec)
    if existing is not None:
        return existing
    if verify_only:
        raise ModelSetupError(f"required model is missing: {destination}")

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=model_dir,
                prefix=f".{spec.filename}.",
                suffix=".partial",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            download_once(spec, temporary_path, context)
            actual = sha256_file(temporary_path)
            if actual != spec.sha256:
                raise ModelSetupError(
                    f"downloaded model digest mismatch for {spec.filename}: "
                    f"expected {spec.sha256}, got {actual}"
                )

            try:
                os.link(temporary_path, destination)
            except FileExistsError as error:
                existing = verify_existing(destination, spec)
                if existing is None:
                    raise ModelSetupError(
                        f"model appeared but cannot be verified: {destination}"
                    ) from error
                return existing
            return model_record(destination, spec, status="downloaded")
        except (HTTPError, URLError, TimeoutError, OSError, ModelSetupError) as error:
            last_error = error
            if attempt == DOWNLOAD_ATTEMPTS:
                break
            time.sleep(2 ** attempt)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    raise ModelSetupError(
        f"failed to download {spec.filename} after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="directory in which to verify or download the two pinned ONNX model files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing files and fail if either model is missing; do not use the network",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    context = tls_context()
    records = [
        ensure_model(
            args.model_dir,
            spec,
            verify_only=args.verify_only,
            context=context,
        )
        for spec in MODELS
    ]
    payload = {"model_dir": str(args.model_dir.resolve()), "models": records}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModelSetupError as error:
        raise SystemExit(f"error: {error}") from error
