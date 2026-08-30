import ast
from pathlib import Path


def test_kaggle_inference_does_not_read_forbidden_sources() -> None:
    notebook = Path("notebooks/docsem_hf_small_ocr_smoke.py")
    if not notebook.exists():
        return
    source = notebook.read_text(encoding="utf-8")
    ast.parse(source)
    forbidden_fragments = (
        "labels/",
        "answer.json",
        "evidence.json",
        "tasks.jsonl",
        "os.walk",
        ".rglob(",
    )
    assert all(fragment not in source for fragment in forbidden_fragments)
    assert "task_000909.pdf" in source
