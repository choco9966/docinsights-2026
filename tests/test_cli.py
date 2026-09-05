from pathlib import Path

import huggingface_hub

from docinsights_analysis import cli
from docinsights_analysis.blind_review import ComparisonSummary, ExportSummary
from docinsights_analysis.consensus import ConsensusSummary
from docinsights_analysis.submission import SubmissionSummary


def test_download_command_uses_latest_release_by_default(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    exit_code = cli.main(["download", "--output", str(tmp_path), "--manifests-only"])

    assert exit_code == 0
    assert captured["revision"] == "e6c9c75bea7575a64279072dcdf0f6050fef9e9f"
    assert captured["local_dir"] == str(tmp_path.resolve())
    assert not any(
        pattern.endswith(".pdf") or "documents" in pattern
        for pattern in captured["allow_patterns"]
    )


def test_download_command_preserves_explicit_historical_revision(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    historical_revision = "b171c5ad488f0c8c50df05951a5b288ea50e9501"

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    exit_code = cli.main(
        [
            "download",
            "--output",
            str(tmp_path),
            "--revision",
            historical_revision,
            "--manifests-only",
        ]
    )

    assert exit_code == 0
    assert captured["revision"] == historical_revision


def test_validate_submission_command_reports_success(monkeypatch, tmp_path: Path, capsys) -> None:
    submission_path = tmp_path / "submission.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    submission_path.touch()
    tasks_path.touch()

    def fake_validate(submission: Path, tasks: Path) -> SubmissionSummary:
        assert submission == submission_path
        assert tasks == tasks_path
        return SubmissionSummary(total=217, instance_ids=frozenset())

    monkeypatch.setattr(cli, "validate_submission", fake_validate)

    exit_code = cli.main(["validate-submission", str(submission_path), "--tasks", str(tasks_path)])

    assert exit_code == 0
    assert "217개 인스턴스" in capsys.readouterr().out


def test_compare_reviews_command_reports_consensus(monkeypatch, tmp_path: Path, capsys) -> None:
    pass_paths = [tmp_path / f"pass{number}.jsonl" for number in range(1, 4)]
    tasks_path = tmp_path / "tasks.jsonl"

    def fake_compare(passes, tasks, *, consensus_path, disagreements_path):
        assert passes == pass_paths
        assert tasks == tasks_path
        return ConsensusSummary(total=217, unanimous=200, disagreements=17)

    monkeypatch.setattr(cli, "compare_review_passes", fake_compare)

    exit_code = cli.main(
        [
            "compare-reviews",
            *(str(path) for path in pass_paths),
            "--tasks",
            str(tasks_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "전원 일치 200개" in output
    assert "재검토 17개" in output


def test_export_blind_review_command_reports_output(monkeypatch, tmp_path: Path, capsys) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "packets"

    def fake_export(tasks, output, *, batch_size, workers):
        assert tasks == tasks_path
        assert output == output_dir
        assert batch_size == 7
        assert workers == 6
        return ExportSummary(total=217, batches=31, output_dir=output_dir)

    monkeypatch.setattr(cli, "export_blind_review", fake_export)

    exit_code = cli.main(
        [
            "export-blind-review",
            "--tasks",
            str(tasks_path),
            "--output",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert "217개" in capsys.readouterr().out


def test_export_blind_subset_command_reports_output(monkeypatch, tmp_path: Path, capsys) -> None:
    questions_path = tmp_path / "questions.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    output_dir = tmp_path / "subset"

    def fake_export(questions, selection, output, *, batch_size, expected_count):
        assert questions == questions_path
        assert selection == selection_path
        assert output == output_dir
        assert batch_size == 5
        assert expected_count == 87
        return ExportSummary(total=87, batches=18, output_dir=output_dir)

    monkeypatch.setattr(cli, "export_blind_subset", fake_export)

    exit_code = cli.main(
        [
            "export-blind-subset",
            "--questions",
            str(questions_path),
            "--selection",
            str(selection_path),
            "--output",
            str(output_dir),
            "--batch-size",
            "5",
            "--expected-count",
            "87",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "87개" in output
    assert "18개 배치" in output


def test_export_qa_review_command_reports_output(monkeypatch, tmp_path: Path, capsys) -> None:
    review_path = tmp_path / "review.jsonl"
    baseline_path = tmp_path / "v7.jsonl"
    output_dir = tmp_path / "qa"

    def fake_export(review, baseline, output, *, batch_size, expected_count):
        assert review == review_path
        assert baseline == baseline_path
        assert output == output_dir
        assert batch_size == 5
        assert expected_count == 217
        return ExportSummary(total=217, batches=44, output_dir=output_dir)

    monkeypatch.setattr(cli, "export_qa_review", fake_export)

    exit_code = cli.main(
        [
            "export-qa-review",
            "--review",
            str(review_path),
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_dir),
            "--batch-size",
            "5",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "217개" in output
    assert "44개 배치" in output


def test_compare_blind_review_command_reports_candidates(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    review_path = tmp_path / "review.jsonl"
    baseline_path = tmp_path / "v7.jsonl"
    output_dir = tmp_path / "comparison"

    confirmations_path = tmp_path / "portal-confirmations.json"
    confirmations_sha256 = "0" * 64

    def fake_compare(
        review,
        baseline,
        output,
        *,
        minimum_confidence,
        portal_confirmations_path,
        portal_confirmations_sha256,
    ):
        assert review == review_path
        assert baseline == baseline_path
        assert output == output_dir
        assert minimum_confidence == 0.95
        assert portal_confirmations_path == confirmations_path
        assert portal_confirmations_sha256 == confirmations_sha256
        return ComparisonSummary(
            total=217,
            confirmed=200,
            candidates=1,
            needs_review=3,
            excluded_portal_confirmed=2,
            portal_conflicts=0,
        )

    monkeypatch.setattr(cli, "compare_blind_review", fake_compare)

    exit_code = cli.main(
        [
            "compare-blind-review",
            str(review_path),
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_dir),
            "--portal-confirmations",
            str(confirmations_path),
            "--portal-confirmations-sha256",
            confirmations_sha256,
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "안전 후보 1개" in output
    assert "포털 확정 제외 2개" in output
