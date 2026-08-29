import argparse
from collections.abc import Sequence
from pathlib import Path

from docinsights_analysis.consensus import ReviewValidationError, compare_review_passes
from docinsights_analysis.constants import DATASET_REVISION, DEFAULT_DATA_DIR
from docinsights_analysis.download import download_dataset
from docinsights_analysis.submission import (
    SubmissionValidationError,
    validate_submission,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocInsights 2026 DocSem 데이터 도구")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="DocSem 공개 데이터 다운로드")
    download_parser.add_argument(
        "--output", type=Path, default=DEFAULT_DATA_DIR, help="데이터 저장 디렉터리"
    )
    download_parser.add_argument(
        "--revision", default=DATASET_REVISION, help="Hugging Face dataset revision"
    )
    download_parser.add_argument(
        "--manifests-only",
        action="store_true",
        help="PDF를 제외하고 JSONL과 안내 파일만 다운로드",
    )

    validate_parser = subparsers.add_parser(
        "validate-submission", help="DocSem 제출 JSONL 사전 검증"
    )
    validate_parser.add_argument("submission", type=Path, help="검증할 제출 JSONL")
    validate_parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_DATA_DIR / "val" / "tasks.jsonl",
        help="제출 대상 tasks.jsonl",
    )

    consensus_parser = subparsers.add_parser(
        "compare-reviews", help="독립 검수 결과의 전원 일치 여부 비교"
    )
    consensus_parser.add_argument(
        "passes", nargs="+", type=Path, help="독립 검수 JSONL 3개 이상"
    )
    consensus_parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_DATA_DIR / "val" / "tasks.jsonl",
        help="검수 대상 tasks.jsonl",
    )
    consensus_parser.add_argument(
        "--consensus",
        type=Path,
        default=Path("artifacts/docsem_validation/consensus.jsonl"),
        help="전원 일치 제출 후보 JSONL",
    )
    consensus_parser.add_argument(
        "--disagreements",
        type=Path,
        default=Path("artifacts/docsem_validation/disagreements.jsonl"),
        help="불일치 검수 기록 JSONL",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        downloaded_path = download_dataset(
            args.output, revision=args.revision, include_pdfs=not args.manifests_only
        )
        print(f"다운로드 완료: {downloaded_path}")
        return 0

    if args.command == "validate-submission":
        try:
            summary = validate_submission(args.submission, args.tasks)
        except SubmissionValidationError as error:
            print(f"제출 파일 검증 실패:\n{error}")
            return 2
        print(f"제출 파일 검증 완료: {summary.total}개 인스턴스")
        return 0

    if args.command == "compare-reviews":
        try:
            summary = compare_review_passes(
                args.passes,
                args.tasks,
                consensus_path=args.consensus,
                disagreements_path=args.disagreements,
            )
        except ReviewValidationError as error:
            print(f"독립 검수 결과 비교 실패:\n{error}")
            return 2
        print(
            "독립 검수 결과 비교 완료: "
            f"전체 {summary.total}개, 전원 일치 {summary.unanimous}개, "
            f"재검토 {summary.disagreements}개"
        )
        return 0

    raise AssertionError(f"지원하지 않는 명령: {args.command}")
