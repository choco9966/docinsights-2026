import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docinsights_analysis.submission import validate_submission_row

REVIEW_FIELDS = frozenset(
    {"run_id", "instance_id", "answer", "evidence", "rationale", "confidence"}
)
CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


class ReviewValidationError(ValueError):
    """독립 검수 결과를 합의에 사용할 수 없을 때 발생하는 오류."""


@dataclass(frozen=True)
class ConsensusSummary:
    total: int
    unanimous: int
    disagreements: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ReviewValidationError(
                f"{path}:{line_number}: 올바른 JSON이 아닙니다: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise ReviewValidationError(f"{path}:{line_number}: JSON 객체가 아닙니다")
        rows.append(row)
    return rows


def _task_ids(tasks_path: Path) -> list[str]:
    identifiers: list[str] = []
    for line_number, row in enumerate(_read_jsonl(tasks_path), start=1):
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ReviewValidationError(
                f"{tasks_path}:{line_number}: instance_id가 올바르지 않습니다"
            )
        identifiers.append(instance_id)
    if len(identifiers) != len(set(identifiers)):
        raise ReviewValidationError(f"{tasks_path}: 중복 instance_id가 있습니다")
    return identifiers


def _validate_review_row(path: Path, line_number: int, row: dict[str, Any]) -> None:
    missing = sorted(REVIEW_FIELDS - frozenset(row))
    extra = sorted(frozenset(row) - REVIEW_FIELDS)
    if missing:
        raise ReviewValidationError(f"{path}:{line_number}: 필수 필드 누락: {', '.join(missing)}")
    if extra:
        raise ReviewValidationError(f"{path}:{line_number}: 허용되지 않은 필드: {', '.join(extra)}")
    run_id = row["run_id"]
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ReviewValidationError(f"{path}:{line_number}: run_id가 올바르지 않습니다")
    submission_row = {field: row[field] for field in ("instance_id", "answer", "evidence")}
    submission_errors = validate_submission_row(submission_row, line_number)
    if submission_errors:
        raise ReviewValidationError(f"{path}: " + "; ".join(submission_errors))
    if not isinstance(row["rationale"], str) or not row["rationale"].strip():
        raise ReviewValidationError(f"{path}:{line_number}: rationale이 비어 있습니다")
    confidence = row["confidence"]
    valid_numeric_confidence = (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0 <= confidence <= 1
    )
    if confidence not in CONFIDENCE_VALUES and not valid_numeric_confidence:
        raise ReviewValidationError(
            f"{path}:{line_number}: confidence는 high, medium, low 또는 0~1 수치여야 합니다"
        )


def _load_pass(path: Path, expected_ids: frozenset[str]) -> tuple[str, dict[str, dict[str, Any]]]:
    reviews: dict[str, dict[str, Any]] = {}
    run_ids: set[str] = set()
    for line_number, row in enumerate(_read_jsonl(path), start=1):
        _validate_review_row(path, line_number, row)
        run_ids.add(row["run_id"])
        instance_id = row["instance_id"]
        if instance_id in reviews:
            raise ReviewValidationError(f"{path}: 중복 instance_id: {instance_id}")
        reviews[instance_id] = row
    submitted_ids = frozenset(reviews)
    missing = sorted(expected_ids - submitted_ids)
    unknown = sorted(submitted_ids - expected_ids)
    if missing:
        raise ReviewValidationError(f"{path}: 누락 instance_id: {', '.join(missing)}")
    if unknown:
        raise ReviewValidationError(f"{path}: 알 수 없는 instance_id: {', '.join(unknown)}")
    if len(run_ids) != 1:
        raise ReviewValidationError(f"{path}: 한 pass에는 하나의 run_id만 있어야 합니다")
    return run_ids.pop(), reviews


def _decision_key(review: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return review["answer"], tuple(sorted(review["evidence"]))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _same_file_or_path(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def compare_review_passes(
    pass_paths: list[Path],
    tasks_path: Path,
    *,
    consensus_path: Path,
    disagreements_path: Path,
) -> ConsensusSummary:
    """세 개 이상의 독립 검수 결과에서 전원 일치 항목만 제출 후보로 만든다."""
    if len(pass_paths) < 3:
        raise ReviewValidationError("독립 검수 결과가 최소 3개 필요합니다")
    if any(
        _same_file_or_path(first, second)
        for index, first in enumerate(pass_paths)
        for second in pass_paths[index + 1 :]
    ):
        raise ReviewValidationError("같은 검수 파일을 독립 pass로 중복 사용할 수 없습니다")
    if _same_file_or_path(consensus_path, disagreements_path):
        raise ReviewValidationError("합의와 불일치 출력 경로는 서로 달라야 합니다")
    protected_inputs = [*pass_paths, tasks_path]
    if any(
        _same_file_or_path(output_path, input_path)
        for output_path in (consensus_path, disagreements_path)
        for input_path in protected_inputs
    ):
        raise ReviewValidationError("출력 경로가 검수 또는 task 입력 경로와 겹칩니다")
    ordered_ids = _task_ids(tasks_path)
    expected_ids = frozenset(ordered_ids)
    loaded_passes = [_load_pass(path, expected_ids) for path in pass_paths]
    run_ids = [run_id for run_id, _ in loaded_passes]
    if len(run_ids) != len(set(run_ids)):
        raise ReviewValidationError("같은 run_id를 독립 pass로 중복 사용할 수 없습니다")
    passes = [review_pass for _, review_pass in loaded_passes]
    consensus: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []

    for instance_id in ordered_ids:
        reviews = [review_pass[instance_id] for review_pass in passes]
        decisions = {_decision_key(review) for review in reviews}
        if len(decisions) == 1:
            consensus.append(
                {
                    "instance_id": instance_id,
                    "answer": reviews[0]["answer"],
                    "evidence": sorted(reviews[0]["evidence"]),
                }
            )
        else:
            disagreements.append({"instance_id": instance_id, "reviews": reviews})

    _write_jsonl(consensus_path, consensus)
    _write_jsonl(disagreements_path, disagreements)
    return ConsensusSummary(
        total=len(ordered_ids),
        unanimous=len(consensus),
        disagreements=len(disagreements),
    )
