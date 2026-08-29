from collections.abc import Callable
from pathlib import Path

from docinsights_analysis.constants import DATASET_REPO_ID, DATASET_REVISION

SnapshotDownloader = Callable[..., str]

FULL_DATA_PATTERNS = (
    "README.md",
    "INSTRUCTIONS.md",
    "LICENSE.txt",
    "examples/**",
    "train/**",
    "val/**",
)

MANIFEST_PATTERNS = (
    "README.md",
    "INSTRUCTIONS.md",
    "LICENSE.txt",
    "examples/**",
    "train/tasks.jsonl",
    "train/labels.jsonl",
    "val/tasks.jsonl",
)


def download_dataset(
    output_dir: Path,
    *,
    revision: str = DATASET_REVISION,
    include_pdfs: bool = True,
    downloader: SnapshotDownloader | None = None,
) -> Path:
    """고정 revision의 DocSem 공개 데이터를 지정 디렉터리에 다운로드한다."""
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = FULL_DATA_PATTERNS if include_pdfs else MANIFEST_PATTERNS
    downloaded_path = downloader(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=list(patterns),
    )
    return Path(downloaded_path)
