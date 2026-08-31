"""Command-line interface for the training ambiguity screen."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .audit import (
    build_audit,
    build_blind_screen,
    validate_artifacts,
    write_audit,
    write_blind_screen,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m docinsights_ambiguity",
        description="Run or validate the DocSem training ambiguity screen.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="screen all records and write artifacts")
    audit.add_argument("--tasks", type=Path, required=True)
    audit.add_argument("--labels", type=Path, required=True)
    audit.add_argument("--reference", type=Path, required=True)
    audit.add_argument("--query-comparison", type=Path, required=True)
    audit.add_argument("--blind-output", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--summary", type=Path, required=True)
    audit.add_argument("--review-shard", type=Path, action="append", default=[])
    audit.add_argument("--expected-count", type=int, default=908)

    validate = subparsers.add_parser("validate", help="validate generated audit artifacts")
    validate.add_argument("--tasks", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--summary", type=Path, required=True)
    validate.add_argument("--expected-count", type=int, default=908)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        blind = build_blind_screen(args.tasks, args.query_comparison)
        write_blind_screen(blind, args.blind_output)
        result = build_audit(
            args.tasks,
            args.labels,
            args.reference,
            args.query_comparison,
            args.blind_output,
            review_shards=args.review_shard,
            expected_count=args.expected_count,
        )
        summary = write_audit(result, args.output, args.summary)
        print(json.dumps(summary["validation"], sort_keys=True))
        return 0 if summary["validation"]["passed"] else 1
    validation = validate_artifacts(
        args.tasks, args.output, args.summary, expected_count=args.expected_count
    )
    print(json.dumps(validation, sort_keys=True))
    return 0 if validation["passed"] else 1
