import json
from pathlib import Path

import pytest

from docinsights_analysis.submission import (
    SubmissionValidationError,
    validate_submission,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_submission_accepts_complete_canonical_jsonl(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    submission_path = tmp_path / "submission.jsonl"
    write_jsonl(
        tasks_path,
        [
            {"instance_id": "task_000001", "user_query": "q1", "document_pdf": "1.pdf"},
            {"instance_id": "task_000002", "user_query": "q2", "document_pdf": "2.pdf"},
        ],
    )
    write_jsonl(
        submission_path,
        [
            {"instance_id": "task_000002", "answer": "20", "evidence": ["b06"]},
            {"instance_id": "task_000001", "answer": "10", "evidence": ["b10"]},
        ],
    )

    summary = validate_submission(submission_path, tasks_path)

    assert summary.total == 2
    assert summary.instance_ids == frozenset({"task_000001", "task_000002"})


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"instance_id": "task_000001", "answer": "10", "evidence": []},
            "evidence는 비어 있지 않은 목록이어야 합니다",
        ),
        (
            {"instance_id": "task_000001", "answer": "정답은 10입니다", "evidence": ["b10"]},
            "answer는 공백 없는 최종 답 문자열이어야 합니다",
        ),
        (
            {"instance_id": "task_000001", "answer": "10", "evidence": ["B10"]},
            "올바른 블록 ID가 아닙니다",
        ),
        (
            {
                "instance_id": "task_000001",
                "answer": "10",
                "evidence": ["b10"],
                "reasoning": "내부 기록",
            },
            "허용되지 않은 필드",
        ),
    ],
)
def test_validate_submission_rejects_invalid_rows(
    tmp_path: Path, row: dict[str, object], message: str
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    submission_path = tmp_path / "submission.jsonl"
    write_jsonl(
        tasks_path,
        [{"instance_id": "task_000001", "user_query": "q", "document_pdf": "1.pdf"}],
    )
    write_jsonl(submission_path, [row])

    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(submission_path, tasks_path)


def test_validate_submission_reports_duplicate_missing_and_unknown_ids(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    submission_path = tmp_path / "submission.jsonl"
    write_jsonl(
        tasks_path,
        [
            {"instance_id": "task_000001", "user_query": "q1", "document_pdf": "1.pdf"},
            {"instance_id": "task_000002", "user_query": "q2", "document_pdf": "2.pdf"},
        ],
    )
    write_jsonl(
        submission_path,
        [
            {"instance_id": "task_000001", "answer": "10", "evidence": ["b10"]},
            {"instance_id": "task_000001", "answer": "10", "evidence": ["b10"]},
            {"instance_id": "task_999999", "answer": "1", "evidence": ["b06"]},
        ],
    )

    with pytest.raises(SubmissionValidationError) as error:
        validate_submission(submission_path, tasks_path)

    message = str(error.value)
    assert "중복 instance_id: task_000001" in message
    assert "누락 instance_id: task_000002" in message
    assert "알 수 없는 instance_id: task_999999" in message
