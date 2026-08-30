import ast
import json
import re
from pathlib import Path


def test_cloud_notebook_is_clean_compilable_and_immutably_pinned() -> None:
    root = Path(__file__).parents[1]
    path = root / "notebooks" / "ocr" / "cloud_cpu_ppocrv5.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in code_cells)

    assert len(code_cells) == 6
    for index, cell in enumerate(code_cells, 1):
        ast.parse("".join(cell["source"]), filename=f"{path}#cell-{index}")
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []

    repository_sha = re.search(r'EXPECTED_REPO_SHA = "([0-9a-f]{40})"', source)
    assert repository_sha is not None
    assert repository_sha.group(1) != "PIN_AFTER_IMPLEMENTATION_COMMIT"
    bundle_sha = re.search(r'EXPECTED_BUNDLE_SHA256 = "([0-9a-f]{64})"', source)
    manifest_sha = re.search(r'EXPECTED_MANIFEST_SHA256 = "([0-9a-f]{64})"', source)
    assert bundle_sha is not None
    assert bundle_sha.group(1) == "9fb35e81feead385fedd3a5bd66ca780ca2aaee5b92b2247f75114cfae642967"
    assert manifest_sha is not None
    assert manifest_sha.group(1) == (
        "08bb8ef1948bdbb69ceddfc669d31adf7002707cdd149937b04615dae0eb2d3b"
    )
    assert '"paddlepaddle==3.2.0"' in source
    assert '"paddleocr==3.3.2"' in source
    assert '"paddlex==3.3.13"' in source
    assert '"huggingface-hub==0.34.4"' in source
    assert 'DET_REVISION = "0d63e78e2b680928f6b1747d76a08db6e645efb7"' in source
    assert 'REC_REVISION = "267c36e24c331595590fe7bd72bde2436fd286f2"' in source


def test_cloud_notebook_enforces_resume_and_runtime_contracts() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "ocr" / "cloud_cpu_ppocrv5.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for required in (
        '"--pipeline-revision"',
        '"--retry-failed"',
        '"--resume"',
        "MAX_RETRY_ROUNDS = 2",
        "session_fingerprint",
        "Checkpoint belongs to another VM or software cohort",
        "No checkpoint progress",
        "Shard has failed OCR records after bounded retries",
        '"pipeline_revision": git_sha',
        '"timeout_seconds": 300.0',
        '"record_count": len(records)',
        '"failed_count": len(FINAL_FAILED)',
    ):
        assert required in source
    assert "BATCH_SIZE" not in source
