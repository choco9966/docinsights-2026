from pathlib import Path

from docinsights_analysis import cli


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
