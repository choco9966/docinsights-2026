import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_validation_material_does_not_embed_concrete_rows() -> None:
    for relative_path in (
        "experiments/submissions.md",
        "prompts/claude_validation_review.md",
    ):
        contents = (ROOT / relative_path).read_text(encoding="utf-8")
        assert re.search(r"task_\d{6}", contents) is None

    implementation = (ROOT / "src/docinsights_analysis/blind_review.py").read_text(encoding="utf-8")
    assert "PORTAL_CONFIRMED_ROWS" not in implementation
    assert "V7_SHA256" not in implementation


def test_private_validation_artifact_paths_are_gitignored() -> None:
    for relative_path in (
        "artifacts/submissions/private-check.jsonl",
        "artifacts/docsem_validation/portal-confirmations.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", relative_path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0
