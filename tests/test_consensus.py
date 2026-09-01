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
    instance_id: str, answer: str, evidence: list[str], confidence: str = "high"
) -> dict[str, object]:
    return {
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
        [review("task_000001", "2", ["b06"]), review("task_000002", "3", ["b07"])],
    )
    write_jsonl(
        pass_paths[1],
        [review("task_000001", "2", ["b06"]), review("task_000002", "4", ["b07"])],
    )
    write_jsonl(
        pass_paths[2],
        [review("task_000001", "2", ["b06"]), review("task_000002", "3", ["b07"])],
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
    write_jsonl(pass_paths[0], [review("task_000001", "2", ["b06", "b07"])])
    write_jsonl(pass_paths[1], [review("task_000001", "2", ["b07", "b06"])])
    write_jsonl(pass_paths[2], [review("task_000001", "2", ["b06", "b07"])])

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
    numeric_review = review("task_000001", "2", ["b06"])
    numeric_review["confidence"] = 0.99
    for path in pass_paths:
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
    for path in pass_paths:
        write_jsonl(path, [review("task_000001", "2", ["b06"])])

    with pytest.raises(ReviewValidationError, match="누락 instance_id: task_000002"):
        compare_review_passes(
            pass_paths,
            tasks_path,
            consensus_path=tmp_path / "consensus.jsonl",
            disagreements_path=tmp_path / "disagreements.jsonl",
        )
