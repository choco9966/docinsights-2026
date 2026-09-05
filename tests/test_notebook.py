import ast
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


def _code_cells_by_id(notebook: dict[str, object]) -> dict[str, str]:
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def _assert_shared_download_revision(code_by_id: dict[str, str]) -> None:
    setup = ast.parse(code_by_id["setup"])
    constants_imports = {
        alias.name
        for node in setup.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "docinsights_analysis.constants"
        for alias in node.names
    }
    assert {"DATASET_REPO_ID", "DATASET_REVISION"} <= constants_imports

    assignments = {
        target.id: node.value.id
        for node in setup.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["REPO_ID"] == "DATASET_REPO_ID"
    assert assignments["REVISION"] == "DATASET_REVISION"

    expected_calls = {
        "load-manifests": "snapshot_download",
        "instance-functions": "hf_hub_download",
    }
    for cell_id, function_name in expected_calls.items():
        tree = ast.parse(code_by_id[cell_id])
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function_name
        ]
        assert len(calls) == 1
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        assert isinstance(keywords["repo_id"], ast.Name)
        assert keywords["repo_id"].id == "REPO_ID"
        assert isinstance(keywords["revision"], ast.Name)
        assert keywords["revision"].id == "REVISION"


def test_notebook_download_paths_use_shared_release_pin() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    _assert_shared_download_revision(_code_cells_by_id(notebook))
