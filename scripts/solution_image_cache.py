"""Exact regeneration and eviction controls for successful solver-page JPEG caches."""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import PIL
from PIL import Image

_CACHE_POLICY = "exact_regenerable_cache_v1"
_RECIPE = "pdftoppm-png-pillow-rgb-jpeg-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ImageCacheError(ValueError):
    """The derived page-image cache cannot satisfy its frozen integrity contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_policy(cache_config: Mapping[str, Any]) -> None:
    if cache_config.get("successful_page_jpegs") != _CACHE_POLICY:
        raise ImageCacheError("page JPEG cache policy is not exact_regenerable_cache_v1")


def _ledger_paths(
    source_dir: Path, page_images: Sequence[Mapping[str, Any]]
) -> tuple[tuple[Path, str], ...]:
    if source_dir.is_symlink():
        raise ImageCacheError("source directory must not be a symlink")
    if not source_dir.is_dir():
        raise ImageCacheError("source directory is missing")
    if not page_images:
        raise ImageCacheError("page image ledger is empty")
    source = source_dir.resolve()
    parsed: list[tuple[Path, str]] = []
    page_numbers: list[int] = []
    for entry in page_images:
        if not isinstance(entry, Mapping) or set(entry) != {"page_number", "path", "sha256"}:
            raise ImageCacheError("page image ledger entry has an invalid schema")
        page_number = entry["page_number"]
        path_value = entry["path"]
        expected = entry["sha256"]
        if type(page_number) is not int or page_number < 1:
            raise ImageCacheError("page image ledger page_number is invalid")
        if (
            not isinstance(path_value, str)
            or not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
        ):
            raise ImageCacheError("page image ledger path or SHA-256 is invalid")
        path = Path(path_value)
        if path.parent.resolve() != source or path.name != f"page-{page_number}.jpg":
            raise ImageCacheError(
                "page image path must be a direct child named source/page-<n>.jpg"
            )
        page_numbers.append(page_number)
        parsed.append((path, expected))
    if page_numbers != list(range(1, len(page_numbers) + 1)):
        raise ImageCacheError("page image ledger must be ordered and contiguous from page 1")
    return tuple(parsed)


def validate_page_jpeg_cache(
    *,
    source_dir: Path,
    page_images: Sequence[Mapping[str, Any]],
    cache_config: Mapping[str, Any],
    allow_missing_for_verified_success: bool = False,
) -> tuple[Path, ...]:
    """Validate ledger paths and every present byte; optionally allow success-only misses."""

    _require_policy(cache_config)
    entries = _ledger_paths(source_dir, page_images)
    paths = []
    for path, expected in entries:
        if path.is_symlink():
            raise ImageCacheError(f"page image is a symlink: {path}")
        if not path.exists():
            if allow_missing_for_verified_success:
                paths.append(path.resolve())
                continue
            raise ImageCacheError(f"page image is missing: {path}")
        if not path.is_file():
            raise ImageCacheError(f"page image is not a regular file: {path}")
        if _sha256(path) != expected:
            raise ImageCacheError(f"page image hash mismatch: {path}")
        paths.append(path.resolve())
    return tuple(paths)


def evict_verified_success_page_jpegs(
    *,
    source_dir: Path,
    page_images: Sequence[Mapping[str, Any]],
    cache_config: Mapping[str, Any],
    success_verified: bool,
) -> tuple[Path, ...]:
    """Delete only ledger-listed JPEGs after the caller verifies a complete success."""

    _require_policy(cache_config)
    if success_verified is not True:
        raise ImageCacheError("page JPEG eviction requires a verified success record")
    entries = _ledger_paths(source_dir, page_images)
    validate_page_jpeg_cache(
        source_dir=source_dir,
        page_images=page_images,
        cache_config=cache_config,
        allow_missing_for_verified_success=True,
    )
    deleted: list[Path] = []
    # Recheck the complete allowlist before deleting any member.
    for path, expected in entries:
        if path.is_symlink():
            raise ImageCacheError(f"page image is a symlink: {path}")
        if path.exists() and (not path.is_file() or _sha256(path) != expected):
            raise ImageCacheError(f"page image hash changed before eviction: {path}")
    for path, _expected in entries:
        if path.exists():
            os.unlink(path)
            deleted.append(path.resolve())
    return tuple(deleted)


def _verify_pillow_runtime(cache_config: Mapping[str, Any]) -> None:
    expected_version = cache_config.get("pillow_version")
    if not isinstance(expected_version, str) or PIL.__version__ != expected_version:
        raise ImageCacheError("active Pillow version does not match frozen config")
    distributions = cache_config.get("runtime_distributions")
    if not isinstance(distributions, list):
        raise ImageCacheError("frozen runtime distributions are missing")
    pillow_records = [
        item
        for item in distributions
        if isinstance(item, Mapping)
        and re.sub(r"[-_.]+", "-", str(item.get("name", ""))).casefold() == "pillow"
    ]
    if len(pillow_records) != 1:
        raise ImageCacheError("frozen runtime must contain exactly one Pillow RECORD")
    record_info = pillow_records[0]
    if record_info.get("version") != expected_version:
        raise ImageCacheError("Pillow RECORD version does not match frozen config")
    record_path_value = record_info.get("record_path")
    record_sha256 = record_info.get("record_sha256")
    if not isinstance(record_path_value, str) or not isinstance(record_sha256, str):
        raise ImageCacheError("Pillow RECORD binding is invalid")
    record_path = Path(record_path_value)
    if (
        record_path.is_symlink()
        or not record_path.is_file()
        or _sha256(record_path) != record_sha256
    ):
        raise ImageCacheError("Pillow RECORD is missing or changed")
    site_packages = record_path.parent.parent.resolve()
    try:
        Path(PIL.__file__).resolve().relative_to(site_packages)
    except ValueError as error:
        raise ImageCacheError("active Pillow import is outside the frozen distribution") from error
    with record_path.open(encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if len(row) != 3:
                raise ImageCacheError("Pillow RECORD row is malformed")
            relative_name, digest_field, size_field = row
            if not digest_field:
                continue
            algorithm, separator, encoded = digest_field.partition("=")
            if algorithm != "sha256" or not separator:
                raise ImageCacheError("Pillow RECORD contains an unsupported digest")
            relative = Path(relative_name)
            if ".." in relative.parts:
                continue
            member = (site_packages / relative).resolve()
            try:
                member.relative_to(site_packages)
            except ValueError as error:
                raise ImageCacheError("Pillow RECORD member escapes site-packages") from error
            if member.is_symlink() or not member.is_file():
                raise ImageCacheError(f"Pillow RECORD member is missing: {relative_name}")
            try:
                expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
                expected_size = int(size_field) if size_field else None
            except (ValueError, TypeError) as error:
                raise ImageCacheError("Pillow RECORD member metadata is invalid") from error
            if _sha256(member) != expected or (
                expected_size is not None and member.stat().st_size != expected_size
            ):
                raise ImageCacheError(f"Pillow RECORD member changed: {relative_name}")


def _verify_runtime(cache_config: Mapping[str, Any]) -> tuple[Path, int, int]:
    _require_policy(cache_config)
    expected = {
        "page_jpeg_recipe": _RECIPE,
        "ocr_input_format": "png",
        "retained_image_format": "jpeg",
        "retained_image_extension": "jpg",
    }
    if any(cache_config.get(key) != value for key, value in expected.items()):
        raise ImageCacheError("frozen page JPEG rendering recipe is unsupported")
    dpi = cache_config.get("dpi")
    quality = cache_config.get("retained_jpeg_quality")
    if type(dpi) is not int or dpi < 1 or type(quality) is not int or not 0 <= quality <= 100:
        raise ImageCacheError("frozen page JPEG numeric settings are invalid")
    renderer_value = cache_config.get("renderer_path")
    renderer_sha = cache_config.get("renderer_sha256")
    renderer_version = cache_config.get("renderer_version")
    if not all(
        isinstance(value, str) for value in (renderer_value, renderer_sha, renderer_version)
    ):
        raise ImageCacheError("frozen renderer binding is invalid")
    renderer = Path(str(renderer_value))
    if renderer.is_symlink() or not renderer.is_file() or _sha256(renderer) != renderer_sha:
        raise ImageCacheError("pinned renderer is missing or changed")
    completed = subprocess.run(
        [str(renderer), "-v"], capture_output=True, text=True, timeout=30, check=False
    )
    lines = (completed.stderr or completed.stdout).splitlines()
    if completed.returncode != 0 or not lines or lines[0] != renderer_version:
        raise ImageCacheError("pinned renderer version changed")
    _verify_pillow_runtime(cache_config)
    return renderer, dpi, quality


def _rendered_page_number(path: Path) -> int:
    match = re.fullmatch(r"ocr-page-(\d+)\.png", path.name)
    if match is None:
        raise ImageCacheError(f"unexpected renderer output: {path.name}")
    return int(match.group(1))


@contextmanager
def rematerialize_page_jpegs(
    *,
    source_dir: Path,
    pdf_path: Path,
    pdf_sha256: str,
    page_images: Sequence[Mapping[str, Any]],
    cache_config: Mapping[str, Any],
    temp_parent: Path,
) -> Iterator[tuple[Path, ...]]:
    """Yield an exact, whole ordered JPEG set in a private disposable directory."""

    entries = _ledger_paths(source_dir, page_images)
    validate_page_jpeg_cache(
        source_dir=source_dir,
        page_images=page_images,
        cache_config=cache_config,
        allow_missing_for_verified_success=True,
    )
    if pdf_path.is_symlink() or not pdf_path.is_file():
        raise ImageCacheError("source PDF is missing or is a symlink")
    if not _SHA256.fullmatch(pdf_sha256) or _sha256(pdf_path) != pdf_sha256:
        raise ImageCacheError("source PDF hash mismatch")
    renderer, dpi, quality = _verify_runtime(cache_config)
    if temp_parent.is_symlink():
        raise ImageCacheError("temporary parent must not be a symlink")
    temp_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(temp_parent, 0o700)
    with tempfile.TemporaryDirectory(prefix="page-jpeg-cache-", dir=temp_parent) as temporary:
        work = Path(temporary)
        os.chmod(work, 0o700)
        prefix = work / "ocr-page"
        completed = subprocess.run(
            [str(renderer), "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            raise ImageCacheError(f"pdftoppm failed: {completed.stderr.strip()}")
        pngs = sorted(work.glob("ocr-page-*.png"), key=_rendered_page_number)
        expected_pages = list(range(1, len(entries) + 1))
        if [_rendered_page_number(path) for path in pngs] != expected_pages:
            raise ImageCacheError("renderer page set does not match the ordered ledger")
        materialized: list[Path] = []
        for page_number, (png, (_source_path, expected_sha)) in enumerate(
            zip(pngs, entries, strict=True), 1
        ):
            output = work / f"page-{page_number}.jpg"
            with Image.open(png) as pixels:
                pixels.convert("RGB").save(output, format="JPEG", quality=quality)
            os.chmod(output, 0o600)
            if _sha256(output) != expected_sha:
                raise ImageCacheError(
                    f"rematerialized page image hash mismatch: page {page_number}"
                )
            materialized.append(output)
        yield tuple(materialized)
