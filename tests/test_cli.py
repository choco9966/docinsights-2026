from pathlib import Path

from docinsights_analysis import cli
from docinsights_analysis.submission import SubmissionSummary


def test_download_command_forwards_manifest_option(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_download(output: Path, *, revision: str, include_pdfs: bool) -> Path:
        captured.update(output=output, revision=revision, include_pdfs=include_pdfs)
        return output

    monkeypatch.setattr(cli, "download_dataset", fake_download)

    exit_code = cli.main(["download", "--output", str(tmp_path), "--manifests-only"])

    assert exit_code == 0
    assert captured["output"] == tmp_path
    assert captured["include_pdfs"] is False


def test_validate_submission_command_reports_success(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    submission_path = tmp_path / "submission.jsonl"
    tasks_path = tmp_path / "tasks.jsonl"
    submission_path.touch()
    tasks_path.touch()

    def fake_validate(submission: Path, tasks: Path) -> SubmissionSummary:
        assert submission == submission_path
        assert tasks == tasks_path
        return SubmissionSummary(total=217, instance_ids=frozenset())

    monkeypatch.setattr(cli, "validate_submission", fake_validate)

    exit_code = cli.main(
        ["validate-submission", str(submission_path), "--tasks", str(tasks_path)]
    )

    assert exit_code == 0
    assert "217개 인스턴스" in capsys.readouterr().out
