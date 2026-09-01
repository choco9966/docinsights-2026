import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/01_docsem_data_analysis.ipynb")


def test_notebook_stores_successful_execution_results_and_code_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert notebook["nbformat"] == 4
    assert code_cells
    image_outputs = 0
    for cell in code_cells:
        assert isinstance(cell["execution_count"], int)
        assert all(output["output_type"] != "error" for output in cell["outputs"])
        image_outputs += sum("image/png" in output.get("data", {}) for output in cell["outputs"])
        compile("".join(cell["source"]), f"{NOTEBOOK_PATH}:{cell['id']}", "exec")

    assert image_outputs >= 3


def test_notebook_covers_query_answer_evidence_and_page_images() -> None:
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    for required_term in (
        "user_query",
        "answer",
        "evidence",
        "inspect_instance",
        "compare_sample",
        "get_pixmap",
        "get_text",
        "worked-example",
        "b10",
        "10 foot",
    ):
        assert required_term in notebook_text
