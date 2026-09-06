#!/usr/bin/env python3
"""Prepare the exact public Native Paddle model files used by Issue #24.

Reuses the checksum-first, non-overwriting downloader from setup_solution_ocr.
This command does not install packages or download task data/labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from setup_solution_ocr import ModelSetupError, ModelSpec, ensure_model, tls_context


class ModelGroup(TypedDict):
    repository: str
    revision: str
    files: dict[str, str]


MODEL_GROUPS: dict[str, ModelGroup] = {
    "detector": {
        "repository": "PaddlePaddle/PP-OCRv5_mobile_det",
        "revision": "0d63e78e2b680928f6b1747d76a08db6e645efb7",
        "files": {
            ".gitattributes": "a3a43406012ee8c7b9b38dfab1b11610f33e8245cd864a9a9302d042616a859e",
            "README.md": "4cc20ad6d41af86b3ce9885ffb0956e152574a2eb14179aeb07fd2d3956161ca",
            "config.json": "7ac1c33f377ba58561f4b89d3180b6add2b7c5c60a4edac90fa4e0ceccdc6665",
            "inference.json": "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c",
            "inference.pdiparams": (
                "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b"
            ),
            "inference.yml": "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699",
        },
    },
    "recognizer": {
        "repository": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
        "revision": "267c36e24c331595590fe7bd72bde2436fd286f2",
        "files": {
            ".gitattributes": "a3a43406012ee8c7b9b38dfab1b11610f33e8245cd864a9a9302d042616a859e",
            "README.md": "4c1cfd6e103b0966fe97505b5254cfa35a931d47d7effca97a9db47fb57dd699",
            "config.json": "16103ac291a82fca50f0bb000d62a4b8168f6042f03c89c44cf32f241d1eb1f2",
            "inference.json": "fd1b6ec722ea841a72d3ba43e527df1d1066d5d7808e0503ee3eec7265188753",
            "inference.pdiparams": (
                "3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b"
            ),
            "inference.yml": "27e91d0582f40168aa218303c76e184bc78fa7a5d105aad0cfbad8458b441067",
        },
    },
}


def prepare_models(model_dir: Path, *, verify_only: bool) -> dict[str, object]:
    context = tls_context()
    records = []
    for role, group in MODEL_GROUPS.items():
        destination = model_dir / role
        destination.mkdir(parents=True, exist_ok=True)
        for filename, digest in group["files"].items():
            spec = ModelSpec(
                filename=filename,
                url=f"https://huggingface.co/{group['repository']}/resolve/{group['revision']}/{filename}",
                sha256=digest,
            )
            records.append(
                {
                    "role": role,
                    **ensure_model(destination, spec, verify_only=verify_only, context=context),
                }
            )
    return {"schema_version": "issue24-native-paddle-models-v1", "models": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare_models(args.model_dir, verify_only=args.verify_only)
    except (OSError, ModelSetupError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
