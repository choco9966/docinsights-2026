import argparse
from collections.abc import Sequence
from pathlib import Path

from docinsights_analysis.blind_review import (
    BlindReviewError,
    compare_blind_review,
    export_blind_review,
    export_blind_subset,
    export_qa_review,
    merge_blind_reviews,
)
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
    consensus_parser.add_argument("passes", nargs="+", type=Path, help="독립 검수 JSONL 3개 이상")
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

    export_parser = subparsers.add_parser(
        "export-blind-review", help="기존 정답을 숨긴 외부 LLM 검수 Q/A 패킷 생성"
    )
    export_parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_DATA_DIR / "val" / "tasks.jsonl",
        help="검수 대상 tasks.jsonl",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/docsem_validation/claude_blind"),
        help="Q/A 패킷 출력 디렉터리",
    )
    export_parser.add_argument("--batch-size", type=int, default=7)
    export_parser.add_argument("--workers", type=int, default=6)

    subset_parser = subparsers.add_parser(
        "export-blind-subset", help="선택된 항목만 정답 없는 외부 LLM 검수 Q/A로 분리"
    )
    subset_parser.add_argument(
        "--questions",
        type=Path,
        default=Path("artifacts/docsem_validation/claude_blind/questions.jsonl"),
        help="전체 블라인드 질문 JSONL",
    )
    subset_parser.add_argument(
        "--selection",
        type=Path,
        default=Path("artifacts/docsem_validation/codex_blind/comparison/needs_review.jsonl"),
        help="남길 instance_id가 들어 있는 JSONL",
    )
    subset_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/docsem_validation/claude_blind_unresolved"),
        help="선별 Q/A 패킷 출력 디렉터리",
    )
    subset_parser.add_argument("--batch-size", type=int, default=7)
    subset_parser.add_argument(
        "--expected-count",
        type=int,
        required=True,
        help="선택 파일이 반드시 포함해야 하는 instance 수",
    )

    qa_review_parser = subparsers.add_parser(
        "export-qa-review",
        help="PDF 없이 문제와 현재 답만 담은 외부 LLM 검산 패킷 생성",
    )
    qa_review_parser.add_argument(
        "--review",
        type=Path,
        default=Path("artifacts/docsem_validation/codex_blind/review.jsonl"),
        help="문제 문장이 들어 있는 전수 검수 JSONL",
    )
    qa_review_parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/submissions/v7.jsonl"),
        help="검증할 현재 답변 JSONL",
    )
    qa_review_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/docsem_validation/claude_qa_review"),
        help="Q/A 검산 패킷 출력 디렉터리",
    )
    qa_review_parser.add_argument("--batch-size", type=int, default=7)
    qa_review_parser.add_argument(
        "--expected-count",
        type=int,
        default=217,
        help="두 입력에 반드시 있어야 하는 전체 instance 수",
    )

    blind_compare_parser = subparsers.add_parser(
        "compare-blind-review", help="외부 LLM 블라인드 풀이와 기준 제출 비교"
    )
    blind_compare_parser.add_argument("review", type=Path)
    blind_compare_parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/submissions/v7.jsonl")
    )
    blind_compare_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/docsem_validation/claude_blind/comparison"),
    )
    blind_compare_parser.add_argument("--minimum-confidence", type=float, default=0.95)
    blind_compare_parser.add_argument(
        "--portal-confirmations",
        type=Path,
        help="Git에서 제외한 포털 확정 JSON 산출물",
    )
    blind_compare_parser.add_argument(
        "--portal-confirmations-sha256",
        help="포털 확정 산출물의 기대 SHA-256",
    )

    blind_merge_parser = subparsers.add_parser(
        "merge-blind-reviews", help="여러 블라인드 검수 lane JSONL 병합"
    )
    blind_merge_parser.add_argument("reviews", nargs="+", type=Path)
    blind_merge_parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_DATA_DIR / "val" / "tasks.jsonl",
    )
    blind_merge_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/docsem_validation/claude_blind/review.jsonl"),
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

    if args.command == "export-blind-review":
        try:
            summary = export_blind_review(
                args.tasks,
                args.output,
                batch_size=args.batch_size,
                workers=args.workers,
            )
        except BlindReviewError as error:
            print(f"블라인드 검수 Q/A 생성 실패:\n{error}")
            return 2
        print(
            f"블라인드 검수 Q/A 생성 완료: {summary.total}개, "
            f"{summary.batches}개 배치, {summary.output_dir}"
        )
        return 0

    if args.command == "export-blind-subset":
        try:
            summary = export_blind_subset(
                args.questions,
                args.selection,
                args.output,
                batch_size=args.batch_size,
                expected_count=args.expected_count,
            )
        except BlindReviewError as error:
            print(f"선별 블라인드 검수 Q/A 생성 실패:\n{error}")
            return 2
        print(
            f"선별 블라인드 검수 Q/A 생성 완료: {summary.total}개, "
            f"{summary.batches}개 배치, {summary.output_dir}"
        )
        return 0

    if args.command == "export-qa-review":
        try:
            summary = export_qa_review(
                args.review,
                args.baseline,
                args.output,
                batch_size=args.batch_size,
                expected_count=args.expected_count,
            )
        except BlindReviewError as error:
            print(f"Q/A 풀이 검증 패킷 생성 실패:\n{error}")
            return 2
        print(
            f"Q/A 풀이 검증 패킷 생성 완료: {summary.total}개, "
            f"{summary.batches}개 배치, {summary.output_dir}"
        )
        return 0

    if args.command == "compare-blind-review":
        try:
            summary = compare_blind_review(
                args.review,
                args.baseline,
                args.output,
                minimum_confidence=args.minimum_confidence,
                portal_confirmations_path=args.portal_confirmations,
                portal_confirmations_sha256=args.portal_confirmations_sha256,
            )
        except BlindReviewError as error:
            print(f"블라인드 검수 비교 실패:\n{error}")
            return 2
        print(
            f"블라인드 검수 비교 완료: 전체 {summary.total}개, "
            f"확정 일치 {summary.confirmed}개, 안전 후보 {summary.candidates}개, "
            f"재검토 {summary.needs_review}개, 포털 확정 제외 "
            f"{summary.excluded_portal_confirmed}개, 포털 기준 충돌 "
            f"{summary.portal_conflicts}개"
        )
        return 0

    if args.command == "merge-blind-reviews":
        try:
            summary = merge_blind_reviews(args.reviews, args.tasks, args.output)
        except BlindReviewError as error:
            print(f"블라인드 검수 병합 실패:\n{error}")
            return 2
        print(f"블라인드 검수 병합 완료: {summary.total}개, {summary.output_path}")
        return 0

    raise AssertionError(f"지원하지 않는 명령: {args.command}")
