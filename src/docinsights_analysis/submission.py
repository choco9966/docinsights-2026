import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = frozenset({"instance_id", "answer", "evidence"})
BLOCK_ID_PATTERN = re.compile(r"^b\d{2}$")
FINAL_ANSWER_PATTERN = re.compile(r"^\S+$")


class SubmissionValidationError(ValueError):
    """공식 제출 전에 수정해야 하는 JSONL 오류."""


@dataclass(frozen=True)
class SubmissionSummary:
    total: int
    instance_ids: frozenset[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            raise SubmissionValidationError(f"{path}:{line_number}: 빈 줄이 있습니다")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise SubmissionValidationError(
                f"{path}:{line_number}: 올바른 JSON 객체가 아닙니다: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise SubmissionValidationError(f"{path}:{line_number}: 각 줄은 JSON 객체여야 합니다")
        rows.append(row)
    return rows


def _expected_instance_ids(tasks_path: Path) -> frozenset[str]:
    rows = _read_jsonl(tasks_path)
    identifiers: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise SubmissionValidationError(
                f"{tasks_path}:{line_number}: instance_id가 올바르지 않습니다"
            )
        identifiers.append(instance_id)
    duplicates = sorted(
        identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
    )
    if duplicates:
        raise SubmissionValidationError(f"{tasks_path}: 중복 instance_id: {', '.join(duplicates)}")
    return frozenset(identifiers)


def _validate_row(row: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    fields = frozenset(row)
    missing_fields = sorted(REQUIRED_FIELDS - fields)
    extra_fields = sorted(fields - REQUIRED_FIELDS)
    if missing_fields:
        errors.append(f"{line_number}행: 필수 필드 누락: {', '.join(missing_fields)}")
    if extra_fields:
        errors.append(f"{line_number}행: 허용되지 않은 필드: {', '.join(extra_fields)}")

    instance_id = row.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        errors.append(f"{line_number}행: instance_id는 비어 있지 않은 문자열이어야 합니다")

    answer = row.get("answer")
    if not isinstance(answer, str) or not FINAL_ANSWER_PATTERN.fullmatch(answer):
        errors.append(f"{line_number}행: answer는 공백 없는 최종 답 문자열이어야 합니다")

    evidence = row.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append(f"{line_number}행: evidence는 비어 있지 않은 목록이어야 합니다")
    elif any(
        not isinstance(block_id, str) or not BLOCK_ID_PATTERN.fullmatch(block_id)
        for block_id in evidence
    ):
        errors.append(f"{line_number}행: evidence에 올바른 블록 ID가 아닙니다")
    elif len(evidence) != len(set(evidence)):
        errors.append(f"{line_number}행: evidence에 중복 블록 ID가 있습니다")
    return errors


def validate_submission(submission_path: Path, tasks_path: Path) -> SubmissionSummary:
    """제출 JSONL의 행 스키마와 대상 task ID의 완전성을 검증한다."""
    expected_ids = _expected_instance_ids(tasks_path)
    rows = _read_jsonl(submission_path)
    errors: list[str] = []
    identifiers: list[str] = []

    for line_number, row in enumerate(rows, start=1):
        errors.extend(_validate_row(row, line_number))
        instance_id = row.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            identifiers.append(instance_id)

    submitted_ids = frozenset(identifiers)
    duplicate_ids = sorted(
        identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
    )
    missing_ids = sorted(expected_ids - submitted_ids)
    unknown_ids = sorted(submitted_ids - expected_ids)
    if duplicate_ids:
        errors.append(f"중복 instance_id: {', '.join(duplicate_ids)}")
    if missing_ids:
        errors.append(f"누락 instance_id: {', '.join(missing_ids)}")
    if unknown_ids:
        errors.append(f"알 수 없는 instance_id: {', '.join(unknown_ids)}")
    if errors:
        raise SubmissionValidationError("\n".join(errors))

    return SubmissionSummary(total=len(rows), instance_ids=submitted_ids)
