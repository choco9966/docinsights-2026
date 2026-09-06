from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from PIL import __version__ as pillow_version

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.solution_image_cache import (  # noqa: E402
    ImageCacheError,
    evict_verified_success_page_jpegs,
    rematerialize_page_jpegs,
    validate_page_jpeg_cache,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger(source: Path, *paths: Path) -> list[dict[str, object]]:
    return [
        {"page_number": index, "path": str(path.resolve()), "sha256": _sha(path)}
        for index, path in enumerate(paths, 1)
    ]


def _policy() -> dict[str, object]:
    return {"successful_page_jpegs": "exact_regenerable_cache_v1"}


def _runtime_config(renderer: Path) -> dict[str, object]:
    version = subprocess.run([str(renderer), "-v"], capture_output=True, text=True, check=False)
    renderer_version = (version.stderr or version.stdout).splitlines()[0]
    distribution = importlib.metadata.distribution("Pillow")
    record = Path(distribution._path) / "RECORD"  # type: ignore[attr-defined]
    return {
        **_policy(),
        "page_jpeg_recipe": "pdftoppm-png-pillow-rgb-jpeg-v1",
        "renderer_path": str(renderer.resolve()),
        "renderer_sha256": _sha(renderer),
        "renderer_version": renderer_version,
        "dpi": 175,
        "ocr_input_format": "png",
        "retained_image_format": "jpeg",
        "retained_image_extension": "jpg",
        "retained_jpeg_quality": 65,
        "pillow_version": pillow_version,
        "runtime_distributions": [
            {
                "name": "pillow",
                "version": pillow_version,
                "record_path": str(record.resolve()),
                "record_sha256": _sha(record),
            }
        ],
    }


def test_validation_rejects_wrong_bytes_symlinks_and_unapproved_missing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "page-1.jpg"
    image.write_bytes(b"expected")
    ledger = _ledger(source, image)

    assert validate_page_jpeg_cache(
        source_dir=source, page_images=ledger, cache_config=_policy()
    ) == (image.resolve(),)

    image.write_bytes(b"changed")
    with pytest.raises(ImageCacheError, match="hash"):
        validate_page_jpeg_cache(
            source_dir=source,
            page_images=ledger,
            cache_config=_policy(),
            allow_missing_for_verified_success=True,
        )

    image.unlink()
    image.symlink_to(tmp_path / "absent.jpg")
    with pytest.raises(ImageCacheError, match="symlink"):
        validate_page_jpeg_cache(
            source_dir=source,
            page_images=ledger,
            cache_config=_policy(),
            allow_missing_for_verified_success=True,
        )

    image.unlink()
    with pytest.raises(ImageCacheError, match="missing"):
        validate_page_jpeg_cache(source_dir=source, page_images=ledger, cache_config=_policy())
    assert validate_page_jpeg_cache(
        source_dir=source,
        page_images=ledger,
        cache_config=_policy(),
        allow_missing_for_verified_success=True,
    ) == (image.resolve(),)
    with pytest.raises(ImageCacheError, match="policy"):
        validate_page_jpeg_cache(
            source_dir=source,
            page_images=ledger,
            cache_config={"successful_page_jpegs": "retain"},
            allow_missing_for_verified_success=True,
        )


def test_validation_rejects_paths_outside_the_ordered_page_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "page-1.jpg"
    outside.write_bytes(b"outside")
    with pytest.raises(ImageCacheError, match="direct child"):
        validate_page_jpeg_cache(
            source_dir=source,
            page_images=_ledger(source, outside),
            cache_config=_policy(),
        )

    second = source / "page-2.jpg"
    first = source / "page-1.jpg"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    reversed_ledger = [
        {"page_number": 2, "path": str(second), "sha256": _sha(second)},
        {"page_number": 1, "path": str(first), "sha256": _sha(first)},
    ]
    with pytest.raises(ImageCacheError, match="ordered"):
        validate_page_jpeg_cache(
            source_dir=source, page_images=reversed_ledger, cache_config=_policy()
        )


def test_eviction_requires_verified_success_and_deletes_only_ledger_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    allowed = source / "page-1.jpg"
    extra = source / "page-99.jpg"
    allowed.write_bytes(b"allowed")
    extra.write_bytes(b"extra")
    ledger = _ledger(source, allowed)

    with pytest.raises(ImageCacheError, match="verified success"):
        evict_verified_success_page_jpegs(
            source_dir=source,
            page_images=ledger,
            cache_config=_policy(),
            success_verified=False,
        )
    assert allowed.exists() and extra.exists()

    assert evict_verified_success_page_jpegs(
        source_dir=source,
        page_images=ledger,
        cache_config=_policy(),
        success_verified=True,
    ) == (allowed.resolve(),)
    assert not allowed.exists()
    assert extra.read_bytes() == b"extra"
    assert (
        evict_verified_success_page_jpegs(
            source_dir=source,
            page_images=ledger,
            cache_config=_policy(),
            success_verified=True,
        )
        == ()
    )


def test_rematerialization_checks_present_mismatch_before_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "page-1.jpg"
    image.write_bytes(b"expected")
    ledger = _ledger(source, image)
    image.write_bytes(b"tampered")
    called = False

    def forbidden_run(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("renderer must not run")

    monkeypatch.setattr(subprocess, "run", forbidden_run)
    with (
        pytest.raises(ImageCacheError, match="hash"),
        rematerialize_page_jpegs(
            source_dir=source,
            pdf_path=tmp_path / "missing.pdf",
            pdf_sha256="0" * 64,
            page_images=ledger,
            cache_config=_policy(),
            temp_parent=tmp_path / "temp",
        ),
    ):
        pass
    assert called is False


def test_same_pdf_rematerializes_byte_identically_in_separate_processes(tmp_path: Path) -> None:
    renderer_name = shutil.which("pdftoppm")
    assert renderer_name is not None
    renderer = Path(renderer_name).resolve()
    config = _runtime_config(renderer)
    pdf = tmp_path / "sample.pdf"
    pages = [Image.new("RGB", (180, 120), color) for color in ("white", "lightblue")]
    pages[0].save(pdf, "PDF", save_all=True, append_images=pages[1:], resolution=72)

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    subprocess.run(
        [str(renderer), "-png", "-r", "175", str(pdf), str(baseline / "ocr-page")],
        check=True,
        capture_output=True,
        text=True,
    )
    source = tmp_path / "source"
    source.mkdir()
    expected_paths = []
    for index, png in enumerate(sorted(baseline.glob("ocr-page-*.png")), 1):
        target = source / f"page-{index}.jpg"
        with Image.open(png) as pixels:
            pixels.convert("RGB").save(target, format="JPEG", quality=65)
        expected_paths.append(target)
    ledger = _ledger(source, *expected_paths)
    evict_verified_success_page_jpegs(
        source_dir=source,
        page_images=ledger,
        cache_config=config,
        success_verified=True,
    )

    config_path = tmp_path / "config.json"
    ledger_path = tmp_path / "ledger.json"
    config_path.write_text(json.dumps(config))
    ledger_path.write_text(json.dumps(ledger))
    helper_root = Path(__file__).parents[1] / "scripts"
    script = """
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from solution_image_cache import rematerialize_page_jpegs
config = json.loads(Path(sys.argv[2]).read_text())
ledger = json.loads(Path(sys.argv[3]).read_text())
with rematerialize_page_jpegs(
    source_dir=Path(sys.argv[4]),
    pdf_path=Path(sys.argv[5]),
    pdf_sha256=sys.argv[6],
    page_images=ledger,
    cache_config=config,
    temp_parent=Path(sys.argv[7]),
) as paths:
    print(json.dumps([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]))
"""
    outputs = []
    for run in ("one", "two"):
        temp_parent = tmp_path / f"temp-{run}"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(helper_root),
                str(config_path),
                str(ledger_path),
                str(source),
                str(pdf),
                _sha(pdf),
                str(temp_parent),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(json.loads(completed.stdout))
        assert not list(temp_parent.iterdir())
    expected = [str(entry["sha256"]) for entry in ledger]
    assert outputs == [expected, expected]


def test_rematerialization_rejects_pdf_renderer_and_pillow_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    missing = source / "page-1.jpg"
    ledger = [{"page_number": 1, "path": str(missing), "sha256": "0" * 64}]
    renderer_name = shutil.which("pdftoppm")
    assert renderer_name is not None
    renderer = Path(renderer_name).resolve()
    config = _runtime_config(renderer)
    pdf = tmp_path / "sample.pdf"
    Image.new("RGB", (50, 50), "white").save(pdf, "PDF")

    with (
        pytest.raises(ImageCacheError, match="PDF hash"),
        rematerialize_page_jpegs(
            source_dir=source,
            pdf_path=pdf,
            pdf_sha256="f" * 64,
            page_images=ledger,
            cache_config=config,
            temp_parent=tmp_path / "temp-pdf",
        ),
    ):
        pass

    bad_renderer = {**config, "renderer_sha256": "f" * 64}
    with (
        pytest.raises(ImageCacheError, match="renderer"),
        rematerialize_page_jpegs(
            source_dir=source,
            pdf_path=pdf,
            pdf_sha256=_sha(pdf),
            page_images=ledger,
            cache_config=bad_renderer,
            temp_parent=tmp_path / "temp-renderer",
        ),
    ):
        pass

    pillow_record = dict(config["runtime_distributions"][0])  # type: ignore[index]
    pillow_record["record_sha256"] = "f" * 64
    bad_pillow = {**config, "runtime_distributions": [pillow_record]}
    with (
        pytest.raises(ImageCacheError, match="Pillow RECORD"),
        rematerialize_page_jpegs(
            source_dir=source,
            pdf_path=pdf,
            pdf_sha256=_sha(pdf),
            page_images=ledger,
            cache_config=bad_pillow,
            temp_parent=tmp_path / "temp-pillow",
        ),
    ):
        pass


def test_rematerialization_verifies_hashed_pillow_record_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    ledger = [
        {
            "page_number": 1,
            "path": str(source / "page-1.jpg"),
            "sha256": "0" * 64,
        }
    ]
    renderer_name = shutil.which("pdftoppm")
    assert renderer_name is not None
    renderer = Path(renderer_name).resolve()
    config = _runtime_config(renderer)
    pdf = tmp_path / "sample.pdf"
    Image.new("RGB", (50, 50), "white").save(pdf, "PDF")

    site_packages = tmp_path / "site-packages"
    package = site_packages / "PIL"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    member = package / "member.py"
    init.write_text("# frozen package\n")
    member.write_text("frozen = True\n")
    dist_info = site_packages / f"pillow-{pillow_version}.dist-info"
    dist_info.mkdir()

    def record_row(path: Path) -> str:
        digest = base64.urlsafe_b64encode(bytes.fromhex(_sha(path))).decode().rstrip("=")
        return f"{path.relative_to(site_packages).as_posix()},sha256={digest},{path.stat().st_size}"

    record = dist_info / "RECORD"
    record.write_text(f"{record_row(init)}\n{record_row(member)}\n")
    config["runtime_distributions"] = [
        {
            "name": "pillow",
            "version": pillow_version,
            "record_path": str(record),
            "record_sha256": _sha(record),
        }
    ]
    monkeypatch.setattr("scripts.solution_image_cache.PIL.__file__", str(init))
    member.write_text("frozen = False\n")

    with (
        pytest.raises(ImageCacheError, match="member changed"),
        rematerialize_page_jpegs(
            source_dir=source,
            pdf_path=pdf,
            pdf_sha256=_sha(pdf),
            page_images=ledger,
            cache_config=config,
            temp_parent=tmp_path / "temp-member",
        ),
    ):
        pass
