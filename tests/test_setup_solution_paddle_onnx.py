import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "setup_solution_paddle_onnx.py"
_SPEC = importlib.util.spec_from_file_location("setup_solution_paddle_onnx", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture_spec(source_json: bytes, source_params: bytes, output: bytes):
    return _MODULE.ExportSpec(
        role="detector",
        repository="example/public-model",
        revision="1" * 40,
        source_json_sha256=_digest(source_json),
        source_params_sha256=_digest(source_params),
        output_sha256=_digest(output),
    )


def _write_source(root: Path, source_json: bytes, source_params: bytes) -> None:
    model = root / "detector"
    model.mkdir(parents=True)
    (model / "inference.json").write_bytes(source_json)
    (model / "inference.pdiparams").write_bytes(source_params)


def _write_converter(path: Path, output: bytes) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('paddle2onnx version: 2.1.0')\n"
        "    raise SystemExit(0)\n"
        "args = sys.argv[1:]\n"
        "save = pathlib.Path(args[args.index('--save_file') + 1])\n"
        f"save.write_bytes({output!r})\n"
        "save.with_suffix('.argv.json').write_text(json.dumps(args))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_prepare_exports_runs_exact_unoptimized_conversion(tmp_path: Path) -> None:
    source_json = b"public graph"
    source_params = b"public weights"
    output = b"expected onnx"
    source_root = tmp_path / "native"
    output_root = tmp_path / "onnx"
    _write_source(source_root, source_json, source_params)
    converter = tmp_path / "paddle2onnx"
    _write_converter(converter, output)

    report = _MODULE.prepare_exports(
        source_root,
        output_root,
        converter=converter,
        verify_only=False,
        specs=(_fixture_spec(source_json, source_params, output),),
    )

    destination = output_root / "detector" / "inference.onnx"
    assert destination.read_bytes() == output
    assert report["converter"]["version"] == "2.1.0"
    assert report["settings"] == {
        "opset_version": 11,
        "enable_onnx_checker": True,
        "optimize_tool": "None",
    }
    assert report["exports"][0]["status"] == "converted"
    command = report["exports"][0]["command"]
    assert command[command.index("--model_filename") + 1] == "inference.json"
    assert command[command.index("--params_filename") + 1] == "inference.pdiparams"
    assert command[command.index("--opset_version") + 1] == "11"
    assert command[command.index("--enable_onnx_checker") + 1] == "True"
    assert command[command.index("--optimize_tool") + 1] == "None"


def test_prepare_exports_rejects_stale_output_without_replacing_it(tmp_path: Path) -> None:
    source_json = b"public graph"
    source_params = b"public weights"
    expected = b"expected onnx"
    source_root = tmp_path / "native"
    output_root = tmp_path / "onnx"
    _write_source(source_root, source_json, source_params)
    destination = output_root / "detector" / "inference.onnx"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale")

    with pytest.raises(_MODULE.ExportSetupError, match="refusing to replace"):
        _MODULE.prepare_exports(
            source_root,
            output_root,
            converter=tmp_path / "missing-converter",
            verify_only=False,
            specs=(_fixture_spec(source_json, source_params, expected),),
        )

    assert destination.read_bytes() == b"stale"


def test_prepare_exports_does_not_publish_unexpected_conversion(tmp_path: Path) -> None:
    source_json = b"public graph"
    source_params = b"public weights"
    source_root = tmp_path / "native"
    output_root = tmp_path / "onnx"
    _write_source(source_root, source_json, source_params)
    converter = tmp_path / "paddle2onnx"
    _write_converter(converter, b"different output")

    with pytest.raises(_MODULE.ExportSetupError, match="did not match the gated digest"):
        _MODULE.prepare_exports(
            source_root,
            output_root,
            converter=converter,
            verify_only=False,
            specs=(_fixture_spec(source_json, source_params, b"expected onnx"),),
        )

    assert not (output_root / "detector" / "inference.onnx").exists()


def test_verify_only_requires_the_gated_output(tmp_path: Path) -> None:
    source_json = b"public graph"
    source_params = b"public weights"
    source_root = tmp_path / "native"
    _write_source(source_root, source_json, source_params)

    with pytest.raises(_MODULE.ExportSetupError, match="required ONNX export is missing"):
        _MODULE.prepare_exports(
            source_root,
            tmp_path / "onnx",
            converter=tmp_path / "unused",
            verify_only=True,
            specs=(_fixture_spec(source_json, source_params, b"expected onnx"),),
        )
