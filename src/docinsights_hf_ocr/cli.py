from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate, hash_paths, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docinsights-hf-ocr")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="regenerate JSON/CSV/Markdown comparison")
    for name in ("raw-results", "raw-dir", "candidates", "reference", "tasks", "out-dir"):
        generate.add_argument(f"--{name}", type=Path, required=True)
    digest = commands.add_parser("hash", help="print stable SHA-256 values")
    digest.add_argument("paths", type=Path, nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hash":
        print(json.dumps(hash_paths(args.paths), indent=2, sort_keys=True))
        return 0
    report = evaluate(
        args.raw_results,
        args.raw_dir,
        args.candidates,
        args.reference,
        args.tasks,
    )
    for path in write_outputs(report, args.out_dir):
        print(path)
    return 0
