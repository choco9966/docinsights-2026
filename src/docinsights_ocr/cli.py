"""Command-line entry points for OCR benchmark preparation and evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .benchmark import compare, hash_run, prepare, run, write_comparison
from .cloud import merge_shards, pack_cloud_input, split_manifest
from .codex_query_compare import compare_codex_queries, write_codex_query_comparison
from .codex_reference import DEFAULT_MODEL, run_codex_reference
from .codex_verify import verify_codex_reference
from .paddle_ocr import DETECTION_MODEL_REVISION, RECOGNITION_MODEL_REVISION
from .silver_evaluation import evaluate_codex_silver, write_silver_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docinsights-ocr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("tasks")
    prepare_parser.add_argument("output")
    prepare_parser.add_argument("--documents-root")
    _add_batch_options(prepare_parser)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("input")
    run_parser.add_argument("output")
    run_parser.add_argument(
        "--engine", choices=("tesseract", "apple-vision", "paddleocr"), default="tesseract"
    )
    run_parser.add_argument("--dpi", type=int, default=300)
    run_parser.add_argument("--language", default="eng")
    run_parser.add_argument("--poppler-executable", default="pdftoppm")
    run_parser.add_argument("--tesseract-executable", default="tesseract")
    run_parser.add_argument("--page-segmentation-mode", type=int, default=6)
    run_parser.add_argument("--apple-vision-executable", default="tools/apple_vision_ocr.swift")
    run_parser.add_argument("--apple-vision-mode", choices=("accurate", "fast"), default="accurate")
    run_parser.add_argument("--paddle-detection-model-dir")
    run_parser.add_argument("--paddle-recognition-model-dir")
    run_parser.add_argument("--paddle-detection-model-revision", default=DETECTION_MODEL_REVISION)
    run_parser.add_argument(
        "--paddle-recognition-model-revision", default=RECOGNITION_MODEL_REVISION
    )
    run_parser.add_argument("--paddle-enable-mkldnn", action="store_true")
    run_parser.add_argument("--pipeline-revision")
    run_parser.add_argument("--documents-root")
    run_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Renderer/subprocess timeout; in-process Paddle predict is not interrupted",
    )
    run_parser.add_argument("--retry-failed", action="store_true")
    _add_batch_options(run_parser)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("reference")
    compare_parser.add_argument("predicted")
    compare_parser.add_argument("--output")

    hash_parser = subparsers.add_parser("hash")
    hash_parser.add_argument("input")
    hash_parser.add_argument("--output")

    codex_parser = subparsers.add_parser("codex-reference")
    codex_parser.add_argument("input")
    codex_parser.add_argument("output")
    codex_parser.add_argument("--documents-root")
    codex_parser.add_argument("--raw-dir", default="artifacts/ocr/codex-reference-raw")
    codex_parser.add_argument("--schema")
    codex_parser.add_argument("--codex-executable", default="codex")
    codex_parser.add_argument("--poppler-executable", default="pdftoppm")
    codex_parser.add_argument("--model", default=DEFAULT_MODEL)
    codex_parser.add_argument("--model-config", action="append", default=[])
    codex_parser.add_argument("--dpi", type=int, default=200)
    codex_parser.add_argument("--timeout-seconds", type=float, default=300.0)
    codex_parser.add_argument("--workers", type=int, default=1)
    codex_parser.add_argument("--retry-failed", action="store_true")
    _add_batch_options(codex_parser)

    codex_verify_parser = subparsers.add_parser("codex-verify")
    codex_verify_parser.add_argument("tasks")
    codex_verify_parser.add_argument("output")
    codex_verify_parser.add_argument("raw_dir")
    codex_verify_parser.add_argument("--documents-root")
    codex_verify_parser.add_argument("--schema")
    codex_verify_parser.add_argument("--poppler-executable", default="pdftoppm")
    codex_verify_parser.add_argument("--timeout-seconds", type=float, default=120.0)
    codex_verify_parser.add_argument("--report")

    codex_query_parser = subparsers.add_parser("codex-query-compare")
    codex_query_parser.add_argument("tasks")
    codex_query_parser.add_argument("reference")
    codex_query_parser.add_argument("jsonl_output")
    codex_query_parser.add_argument("markdown_output")
    codex_query_parser.add_argument("--documents-root")
    codex_query_parser.add_argument("--split-name")
    codex_query_parser.add_argument("--pdftotext-executable", default="pdftotext")
    codex_query_parser.add_argument("--renderer-executable", default="pdftoppm")
    codex_query_parser.add_argument("--tesseract-executable", default="tesseract")
    codex_query_parser.add_argument("--fallback-dpi", type=int, default=200)
    codex_query_parser.add_argument("--workers", type=int, default=1)
    codex_query_parser.add_argument("--timeout-seconds", type=float, default=30.0)

    silver_parser = subparsers.add_parser("codex-silver-evaluate")
    silver_parser.add_argument("reference")
    silver_parser.add_argument("prediction")
    silver_parser.add_argument("output")
    silver_parser.add_argument("--markdown")
    silver_parser.add_argument("--engine-label")
    silver_parser.add_argument("--reference-label")
    silver_parser.add_argument("--prediction-label")

    shard_parser = subparsers.add_parser("cloud-shard")
    shard_parser.add_argument("input")
    shard_parser.add_argument("output_dir")
    shard_parser.add_argument("--shard-count", type=int, required=True)
    shard_parser.add_argument("--seed")

    merge_parser = subparsers.add_parser("cloud-merge")
    merge_parser.add_argument("manifest")
    merge_parser.add_argument("output")
    merge_parser.add_argument("shards", nargs="+")
    merge_parser.add_argument("--runtimes", nargs="+", required=True)
    merge_parser.add_argument("--allow-failed", action="store_true")
    merge_parser.add_argument("--seed")
    merge_parser.add_argument("--report")

    pack_parser = subparsers.add_parser("cloud-pack")
    pack_parser.add_argument("manifest")
    pack_parser.add_argument("documents_root")
    pack_parser.add_argument("output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.tasks,
            args.output,
            documents_root=args.documents_root,
            limit=args.limit,
            resume=args.resume,
        )
    elif args.command == "run":
        run(
            args.input,
            args.output,
            dpi=args.dpi,
            language=args.language,
            engine=args.engine,
            poppler_executable=args.poppler_executable,
            tesseract_executable=args.tesseract_executable,
            page_segmentation_mode=args.page_segmentation_mode,
            apple_vision_executable=args.apple_vision_executable,
            apple_vision_mode=args.apple_vision_mode,
            paddle_detection_model_dir=args.paddle_detection_model_dir,
            paddle_recognition_model_dir=args.paddle_recognition_model_dir,
            paddle_detection_model_revision=args.paddle_detection_model_revision,
            paddle_recognition_model_revision=args.paddle_recognition_model_revision,
            paddle_enable_mkldnn=args.paddle_enable_mkldnn,
            pipeline_revision=args.pipeline_revision,
            documents_root=args.documents_root,
            timeout_seconds=args.timeout_seconds,
            retry_failed=args.retry_failed,
            limit=args.limit,
            resume=args.resume,
        )
    elif args.command == "codex-reference":
        run_codex_reference(
            args.input,
            args.output,
            documents_root=args.documents_root,
            raw_dir=args.raw_dir,
            schema_path=args.schema,
            codex_executable=args.codex_executable,
            poppler_executable=args.poppler_executable,
            model=args.model,
            model_config=args.model_config or ('model_reasoning_effort="high"',),
            dpi=args.dpi,
            timeout_seconds=args.timeout_seconds,
            workers=args.workers,
            retry_failed=args.retry_failed,
            limit=args.limit,
            resume=args.resume,
        )
    elif args.command == "codex-verify":
        result = verify_codex_reference(
            args.tasks,
            args.output,
            args.raw_dir,
            documents_root=args.documents_root,
            schema_path=args.schema,
            poppler_executable=args.poppler_executable,
            timeout_seconds=args.timeout_seconds,
        )
        if args.report:
            write_comparison(args.report, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "codex-query-compare":
        comparison = compare_codex_queries(
            args.tasks,
            args.reference,
            documents_root=args.documents_root,
            split_name=args.split_name,
            pdftotext_executable=args.pdftotext_executable,
            renderer_executable=args.renderer_executable,
            tesseract_executable=args.tesseract_executable,
            fallback_dpi=args.fallback_dpi,
            workers=args.workers,
            timeout_seconds=args.timeout_seconds,
        )
        result = write_codex_query_comparison(
            comparison,
            args.jsonl_output,
            args.markdown_output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "codex-silver-evaluate":
        evaluation = evaluate_codex_silver(
            args.reference,
            args.prediction,
            engine_label=args.engine_label,
            reference_label=args.reference_label,
            prediction_label=args.prediction_label,
        )
        outputs = write_silver_evaluation(
            evaluation,
            args.output,
            markdown_path=args.markdown,
            protected_source_paths=(args.reference, args.prediction),
        )
        print(
            json.dumps(
                {"outputs": outputs, "summary": evaluation["summary"]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "cloud-shard":
        options = {"shard_count": args.shard_count}
        if args.seed is not None:
            options["seed"] = args.seed
        result = split_manifest(args.input, args.output_dir, **options)
        summary = {key: value for key, value in result.items() if key != "shards"}
        summary["shard_record_counts"] = [shard["record_count"] for shard in result["shards"]]
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "cloud-merge":
        options = {}
        if args.seed is not None:
            options["seed"] = args.seed
        result = merge_shards(
            args.manifest,
            args.shards,
            args.output,
            runtime_paths=args.runtimes,
            fail_closed=not args.allow_failed,
            **options,
        )
        if args.report:
            write_comparison(args.report, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "cloud-pack":
        result = pack_cloud_input(args.manifest, args.documents_root, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "compare":
        result = compare(args.reference, args.predicted)
        if args.output:
            write_comparison(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        result = hash_run(args.input)
        if args.output:
            write_comparison(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_batch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
