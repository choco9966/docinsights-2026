from pathlib import Path

from docinsights_analysis.constants import DATASET_REPO_ID, DATASET_REVISION
from docinsights_analysis.download import download_dataset


def test_download_dataset_uses_pinned_revision_and_manifest_filter(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_downloader(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    result = download_dataset(tmp_path, include_pdfs=False, downloader=fake_downloader)

    assert result == tmp_path
    assert captured["repo_id"] == DATASET_REPO_ID
    assert captured["repo_type"] == "dataset"
    assert captured["revision"] == DATASET_REVISION
    assert captured["local_dir"] == str(tmp_path.resolve())
    assert "train/tasks.jsonl" in captured["allow_patterns"]
    assert not any(
        pattern.endswith(".pdf") or "documents" in pattern for pattern in captured["allow_patterns"]
    )
