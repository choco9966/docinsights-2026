import json
from pathlib import Path

import pytest

from docinsights_analysis.consensus import ReviewValidationError, compare_review_passes


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def review(
    instance_id: str,
    answer: str,
    evidence: list[str],
    confidence: str = "high",
    *,
    run_id: str = "review-run-1",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "instance_id": instance_id,
        "answer": answer,
        "evidence": evidence,
        "rationale": "1 + 1 = 2",
        "confidence": confidence,
    }


def test_compare_review_passes_writes_only_unanimous_predictions(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    consensus_path = tmp_path / "consensus.jsonl"
    disagreements_path = tmp_path / "disagreements.jsonl"
    write_jsonl(
        tasks_path,
        [
            {"instance_id": "task_000001"},
            {"instance_id": "task_000002"},
        ],
    )
    write_jsonl(
        pass_paths[0],
        [
            review("task_000001", "2", ["b06"], run_id="review-run-1"),
            review("task_000002", "3", ["b07"], run_id="review-run-1"),
        ],
    )
    write_jsonl(
        pass_paths[1],
        [
            review("task_000001", "2", ["b06"], run_id="review-run-2"),
            review("task_000002", "4", ["b07"], run_id="review-run-2"),
        ],
    )
    write_jsonl(
        pass_paths[2],
        [
            review("task_000001", "2", ["b06"], run_id="review-run-3"),
            review("task_000002", "3", ["b07"], run_id="review-run-3"),
        ],
    )

    summary = compare_review_passes(
        pass_paths,
        tasks_path,
        consensus_path=consensus_path,
        disagreements_path=disagreements_path,
    )

    assert summary.total == 2
    assert summary.unanimous == 1
    assert summary.disagreements == 1
    assert json.loads(consensus_path.read_text().strip()) == {
        "instance_id": "task_000001",
        "answer": "2",
        "evidence": ["b06"],
    }
    disagreement = json.loads(disagreements_path.read_text().strip())
    assert disagreement["instance_id"] == "task_000002"
    assert [item["answer"] for item in disagreement["reviews"]] == ["3", "4", "3"]


def test_compare_review_passes_treats_evidence_as_a_set(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    write_jsonl(tasks_path, [{"instance_id": "task_000001"}])
    write_jsonl(
        pass_paths[0],
        [review("task_000001", "2", ["b06", "b07"], run_id="review-run-1")],
    )
    write_jsonl(
        pass_paths[1],
        [review("task_000001", "2", ["b07", "b06"], run_id="review-run-2")],
    )
    write_jsonl(
        pass_paths[2],
        [review("task_000001", "2", ["b06", "b07"], run_id="review-run-3")],
    )

    summary = compare_review_passes(
        pass_paths,
        tasks_path,
        consensus_path=tmp_path / "consensus.jsonl",
        disagreements_path=tmp_path / "disagreements.jsonl",
    )

    assert summary.unanimous == 1


def test_compare_review_passes_accepts_numeric_confidence(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    write_jsonl(tasks_path, [{"instance_id": "task_000001"}])
    for number, path in enumerate(pass_paths, start=1):
        numeric_review = review("task_000001", "2", ["b06"], run_id=f"review-run-{number}")
        numeric_review["confidence"] = 0.99
        write_jsonl(path, [numeric_review])

    summary = compare_review_passes(
        pass_paths,
        tasks_path,
        consensus_path=tmp_path / "consensus.jsonl",
        disagreements_path=tmp_path / "disagreements.jsonl",
    )

    assert summary.unanimous == 1


def test_compare_review_passes_rejects_incomplete_pass(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    write_jsonl(
        tasks_path,
        [{"instance_id": "task_000001"}, {"instance_id": "task_000002"}],
    )
    for number, path in enumerate(pass_paths, start=1):
        write_jsonl(
            path,
            [review("task_000001", "2", ["b06"], run_id=f"review-run-{number}")],
        )

    with pytest.raises(ReviewValidationError, match="누락 instance_id: task_000002"):
        compare_review_passes(
            pass_paths,
            tasks_path,
            consensus_path=tmp_path / "consensus.jsonl",
            disagreements_path=tmp_path / "disagreements.jsonl",
        )


def test_compare_review_passes_rejects_duplicate_resolved_path(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    first_path = tmp_path / "pass1.jsonl"
    alias_path = tmp_path / "pass-alias.jsonl"
    third_path = tmp_path / "pass3.jsonl"
    write_jsonl(tasks_path, [{"instance_id": "task_000001"}])
    write_jsonl(
        first_path,
        [review("task_000001", "2", ["b06"], run_id="review-run-1")],
    )
    alias_path.symlink_to(first_path)
    write_jsonl(
        third_path,
        [review("task_000001", "2", ["b06"], run_id="review-run-3")],
    )

    with pytest.raises(ReviewValidationError, match="파일 경로"):
        compare_review_passes(
            [first_path, alias_path, third_path],
            tasks_path,
            consensus_path=tmp_path / "consensus.jsonl",
            disagreements_path=tmp_path / "disagreements.jsonl",
        )


def test_compare_review_passes_rejects_duplicate_run_id(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    write_jsonl(tasks_path, [{"instance_id": "task_000001"}])
    for path in pass_paths:
        write_jsonl(
            path,
            [review("task_000001", "2", ["b06"], run_id="reused-run-id")],
        )

    with pytest.raises(ReviewValidationError, match="run_id"):
        compare_review_passes(
            pass_paths,
            tasks_path,
            consensus_path=tmp_path / "consensus.jsonl",
            disagreements_path=tmp_path / "disagreements.jsonl",
        )


@pytest.mark.parametrize(
    ("answer", "evidence", "message"),
    [(" 2", ["b06"], "answer"), ("2", ["b06", "b06"], "중복 블록")],
)
def test_compare_review_passes_reuses_submission_row_validation(
    tmp_path: Path, answer: str, evidence: list[str], message: str
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    write_jsonl(tasks_path, [{"instance_id": "task_000001"}])
    for number, path in enumerate(pass_paths, start=1):
        write_jsonl(
            path,
            [
                review(
                    "task_000001",
                    answer,
                    evidence,
                    run_id=f"review-run-{number}",
                )
            ],
        )

    with pytest.raises(ReviewValidationError, match=message):
        compare_review_passes(
            pass_paths,
            tasks_path,
            consensus_path=tmp_path / "consensus.jsonl",
            disagreements_path=tmp_path / "disagreements.jsonl",
        )


def test_compare_review_passes_rejects_output_input_collision(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    write_jsonl(tasks_path, [{"instance_id": "task_000001"}])
    for number, path in enumerate(pass_paths, start=1):
        write_jsonl(
            path,
            [review("task_000001", "2", ["b06"], run_id=f"review-run-{number}")],
        )

    with pytest.raises(ReviewValidationError, match="입력 경로"):
        compare_review_passes(
            pass_paths,
            tasks_path,
            consensus_path=pass_paths[0],
            disagreements_path=tmp_path / "disagreements.jsonl",
        )
