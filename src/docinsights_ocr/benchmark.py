"""Prepare, execute, and compare OCR benchmark JSONL files."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .apple_vision import AppleVisionEngine
from .blocks import reconstruct_blocks
from .metrics import (
    block_id_agreement,
    character_error_rate,
    exact_token_prf,
    ordered_quantity_prf,
    word_error_rate,
)
from .models import Document
from .paddle_ocr import (
    DETECTION_MODEL_REVISION,
    RECOGNITION_MODEL_REVISION,
    PaddleOCREngine,
)
from .records import (
    aggregate_ocr_hash,
    deterministic_content_hash,
    failure_record,
    read_jsonl,
    success_record,
    write_jsonl,
)
from .render import render_pdf
from .tesseract import TesseractEngine

SPLIT_SEED = "docinsights-2026-ocr-validation-v1"
OCR_DEV_SIZE = 30


def prepare(
    tasks_path: str | Path,
    output_path: str | Path,
    *,
    documents_root: str | Path | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> int:
    """Create a deterministic OCR work manifest from DocSem task JSONL."""
    existing = (
        _records_by_id(output_path, "resume manifest")
        if resume and Path(output_path).exists()
        else {}
    )
    root = Path(documents_root) if documents_root is not None else Path(tasks_path).parent
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for task in read_jsonl(tasks_path):
        instance_id = _required_string(task, "instance_id")
        if instance_id in seen_ids:
            raise ValueError(f"duplicate instance_id in manifest input: {instance_id}")
        seen_ids.add(instance_id)
        document_pdf = _required_string(task, "document_pdf")
        pdf_path = (root / document_pdf).resolve()
        pdf_sha256 = _sha256_file(pdf_path)
        candidates.append(
            {
                "instance_id": instance_id,
                "user_query": _required_string(task, "user_query"),
                "document_pdf": pdf_path.relative_to(root.resolve()).as_posix(),
                "input_pdf_sha256": pdf_sha256,
                "_split_key": hashlib.sha256(
                    f"{SPLIT_SEED}\0{instance_id}\0{pdf_sha256}".encode()
                ).hexdigest(),
            }
        )
    dev_ids = {
        record["instance_id"]
        for record in sorted(
            candidates, key=lambda item: (item["_split_key"], item["instance_id"])
        )[:OCR_DEV_SIZE]
    }
    current: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["instance_id"]):
        finalized = {key: value for key, value in candidate.items() if key != "_split_key"}
        finalized["split"] = "ocr_dev" if candidate["instance_id"] in dev_ids else "ocr_eval"
        finalized["split_seed"] = SPLIT_SEED
        current.append(finalized)
    current_by_id = {record["instance_id"]: record for record in current}
    for instance_id, completed_record in existing.items():
        current_record = current_by_id.get(instance_id)
        if current_record is None:
            raise ValueError(f"stale resume manifest instance_id: {instance_id}")
        _validate_completed_manifest_record(completed_record, current_record)
    pending = (record for record in current if record["instance_id"] not in existing)
    records = list(_limited(pending, limit))
    write_jsonl(output_path, records, append=resume)
    return len(records)


def run(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 300,
    language: str = "eng",
    engine: str = "tesseract",
    poppler_executable: str = "pdftoppm",
    tesseract_executable: str = "tesseract",
    page_segmentation_mode: int = 6,
    apple_vision_executable: str | Path = "tools/apple_vision_ocr.swift",
    apple_vision_mode: str = "accurate",
    paddle_detection_model_dir: str | Path | None = None,
    paddle_recognition_model_dir: str | Path | None = None,
    paddle_detection_model_revision: str = DETECTION_MODEL_REVISION,
    paddle_recognition_model_revision: str = RECOGNITION_MODEL_REVISION,
    paddle_enable_mkldnn: bool = False,
    pipeline_revision: str | None = None,
    documents_root: str | Path | None = None,
    timeout_seconds: float | None = 120.0,
    retry_failed: bool = False,
    limit: int | None = None,
    resume: bool = False,
) -> int:
    """OCR document images, then attach ``user_query`` only when assembling output."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if pipeline_revision is not None and not pipeline_revision.strip():
        raise ValueError("pipeline_revision must not be empty")
    checkpoint_path = _retry_checkpoint_path(output_path)
    if retry_failed and not resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    resume_path = (
        checkpoint_path
        if retry_failed and resume and checkpoint_path.exists()
        else Path(output_path)
    )
    existing = (
        _records_by_id(resume_path, "resume output") if resume and resume_path.exists() else {}
    )
    if engine == "tesseract":
        engine_adapter = TesseractEngine(
            executable=tesseract_executable,
            language=language,
            dpi=dpi,
            page_segmentation_mode=page_segmentation_mode,
            timeout_seconds=timeout_seconds,
        )
    elif engine == "apple-vision":
        apple_language = "en-US" if language == "eng" else language
        engine_adapter = AppleVisionEngine(
            executable=apple_vision_executable,
            language=apple_language,
            mode=apple_vision_mode,
            timeout_seconds=timeout_seconds,
        )
    elif engine == "paddleocr":
        if paddle_detection_model_dir is None or paddle_recognition_model_dir is None:
            raise ValueError(
                "paddleocr requires pinned detection and recognition model directories"
            )
        engine_adapter = PaddleOCREngine(
            detection_model_dir=paddle_detection_model_dir,
            recognition_model_dir=paddle_recognition_model_dir,
            detection_model_revision=paddle_detection_model_revision,
            recognition_model_revision=paddle_recognition_model_revision,
            enable_mkldnn=paddle_enable_mkldnn,
        )
    else:
        raise ValueError("engine must be 'tesseract', 'apple-vision', or 'paddleocr'")
    ocr_executable_identity = getattr(engine_adapter, "executable_identity", None)
    if ocr_executable_identity is None:
        ocr_executable_identity = _executable_identity(engine_adapter.executable)
    renderer_executable_identity = _executable_identity(poppler_executable)
    if not resume:
        write_jsonl(output_path, [])
    deferred_records: dict[str, dict[str, Any]] = {}
    deferred_order: list[str] = []
    tasks = list(read_jsonl(input_path))
    seen_input_ids: set[str] = set()
    for task in tasks:
        instance_id = _required_string(task, "instance_id")
        if instance_id in seen_input_ids:
            raise ValueError(f"duplicate instance_id in run input: {instance_id}")
        seen_input_ids.add(instance_id)
    if retry_failed and resume:
        tasks.sort(
            key=lambda task: (
                0
                if task["instance_id"] not in existing
                else 1
                if existing[task["instance_id"]].get("status") == "failed"
                else 2
            )
        )
    written = 0
    input_root = (
        Path(documents_root).resolve()
        if documents_root is not None
        else Path(input_path).parent.resolve()
    )
    for task in tasks:
        instance_id = _required_string(task, "instance_id")
        user_query = _required_string(task, "user_query")
        manifest_pdf = Path(_required_string(task, "document_pdf"))
        pdf_path = manifest_pdf if manifest_pdf.is_absolute() else input_root / manifest_pdf
        pdf_path = pdf_path.resolve()
        expected_sha256 = task.get("input_pdf_sha256")
        input_error: OSError | ValueError | None = None
        try:
            actual_sha256 = _sha256_file(pdf_path)
        except OSError as exc:
            input_error = exc
            actual_sha256 = expected_sha256 if isinstance(expected_sha256, str) else "unavailable"
        if expected_sha256 is not None and expected_sha256 != actual_sha256:
            input_error = ValueError(f"input PDF SHA-256 mismatch for {instance_id}")
        if isinstance(engine_adapter, TesseractEngine):
            options = {
                "page_segmentation_mode": engine_adapter.page_segmentation_mode,
                "executable": engine_adapter.executable,
                "renderer_executable": poppler_executable,
            }
        elif isinstance(engine_adapter, AppleVisionEngine):
            options = {
                "recognition_mode": engine_adapter.mode,
                "executable": engine_adapter.executable,
                "renderer_executable": poppler_executable,
            }
        else:
            options = {
                **engine_adapter.options,
                "renderer_executable": poppler_executable,
            }
        if pipeline_revision is not None:
            options["pipeline_revision"] = pipeline_revision
        run_fingerprint = _run_fingerprint(
            engine=engine_adapter.name,
            dpi=dpi,
            language=engine_adapter.language,
            options=options,
            input_sha256=actual_sha256,
            timeout_seconds=timeout_seconds,
            ocr_executable_identity=ocr_executable_identity,
            renderer_executable_identity=renderer_executable_identity,
        )
        prior = existing.get(instance_id)
        if prior is not None:
            prior_provenance = prior.get("provenance")
            if not isinstance(prior_provenance, dict):
                raise ValueError(f"resume record has invalid provenance for {instance_id}")
            prior_fingerprint = prior_provenance.get("run_fingerprint")
            if prior_fingerprint != run_fingerprint:
                raise ValueError(f"resume fingerprint mismatch for {instance_id}")
            if prior.get("status") != "failed" or not retry_failed:
                continue
        if limit is not None and written >= limit:
            break
        started = time.perf_counter()
        provenance = {
            "document_pdf": str(pdf_path),
            "input_pdf_sha256": actual_sha256,
            "split": task.get("split"),
            "split_seed": task.get("split_seed"),
            "dpi": dpi,
            "ocr_engine": engine_adapter.name,
            "ocr_options": options,
            "language": engine_adapter.language,
            "renderer": "poppler-pdftoppm",
            "confidence_kind": engine_adapter.confidence_kind,
            "coordinate_system": "pixel_top_left",
            "run_fingerprint": run_fingerprint,
            "ocr_executable_identity": ocr_executable_identity,
            "renderer_executable_identity": renderer_executable_identity,
            "pipeline_revision": pipeline_revision,
            "timeout_seconds": timeout_seconds,
        }
        try:
            if input_error is not None:
                raise input_error
            with tempfile.TemporaryDirectory(prefix="docinsights-ocr-") as temporary:
                render_started = time.perf_counter()
                images = render_pdf(
                    pdf_path,
                    temporary,
                    dpi=dpi,
                    executable=poppler_executable,
                    timeout_seconds=timeout_seconds,
                )
                rendered = time.perf_counter()
                pages = tuple(
                    engine_adapter.recognize(image, page_number=page_number)
                    for page_number, image in enumerate(images, 1)
                )
                recognized = time.perf_counter()
            blocks = reconstruct_blocks(pages)
            if not blocks:
                raise ValueError("OCR output contained no bNN block markers")
            document = Document(
                document_id=instance_id,
                pages=pages,
                blocks=blocks,
                engine=engine_adapter.name,
                provenance=provenance,
            )
            record = success_record(
                instance_id=instance_id,
                user_query=user_query,
                document=document,
                provenance=provenance,
                timing={
                    "render_seconds": rendered - render_started,
                    "ocr_seconds": recognized - rendered,
                    "total_seconds": time.perf_counter() - started,
                },
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            record = failure_record(
                instance_id=instance_id,
                user_query=user_query,
                engine=engine_adapter.name,
                provenance=provenance,
                timing={"total_seconds": time.perf_counter() - started},
                error=exc,
            )
        if retry_failed:
            deferred_records[instance_id] = record
            deferred_order.append(instance_id)
            _atomic_write_jsonl(
                checkpoint_path,
                _merged_retry_records(existing, deferred_records, deferred_order),
            )
        else:
            write_jsonl(output_path, [record], append=True)
        written += 1
    if retry_failed and checkpoint_path.exists():
        checkpoint_path.replace(output_path)
    return written


def compare(reference_path: str | Path, predicted_path: str | Path) -> dict[str, Any]:
    """Compare OCR records by aligned instance and block IDs/text."""
    references = _records_by_id(reference_path, "reference")
    predictions = _records_by_id(predicted_path, "prediction")
    rows: list[dict[str, Any]] = []
    for instance_id in sorted(references.keys() | predictions.keys()):
        reference_record = references.get(instance_id)
        predicted_record = predictions.get(instance_id)
        reference_blocks = _blocks_by_id(reference_record, instance_id) if reference_record else {}
        predicted_blocks = _blocks_by_id(predicted_record, instance_id) if predicted_record else {}
        agreement = block_id_agreement(reference_blocks, predicted_blocks)
        reference_text = "\n".join(reference_blocks.values())
        predicted_text = "\n".join(predicted_blocks.values())
        rows.append(
            {
                "instance_id": instance_id,
                "reference_status": (
                    reference_record.get("status", "missing") if reference_record else "missing"
                ),
                "predicted_status": (
                    predicted_record.get("status", "missing") if predicted_record else "missing"
                ),
                "cer": character_error_rate(reference_text, predicted_text),
                "wer": word_error_rate(reference_text, predicted_text),
                "exact_tokens": asdict(exact_token_prf(reference_text, predicted_text)),
                "ordered_quantities": asdict(ordered_quantity_prf(reference_text, predicted_text)),
                "blocks": asdict(agreement),
            }
        )
    count = len(rows)
    return {
        "instances": count,
        "missing_predictions": sorted(references.keys() - predictions.keys()),
        "unexpected_predictions": sorted(predictions.keys() - references.keys()),
        "mean_cer": sum(row["cer"] for row in rows) / count if count else 0.0,
        "mean_wer": sum(row["wer"] for row in rows) / count if count else 0.0,
        "mean_block_f1": sum(row["blocks"]["f1"] for row in rows) / count if count else 0.0,
        "mean_exact_token_f1": sum(row["exact_tokens"]["f1"] for row in rows) / count
        if count
        else 0.0,
        "mean_ordered_quantity_f1": sum(row["ordered_quantities"]["f1"] for row in rows) / count
        if count
        else 0.0,
        "ordered_block_exact_count": sum(row["blocks"]["ordered_exact"] for row in rows),
        "exact_token_metric_role": "diagnostic",
        "reference_ok_count": sum(record.get("status") == "ok" for record in references.values()),
        "reference_failed_count": sum(
            record.get("status") == "failed" for record in references.values()
        ),
        "predicted_ok_count": sum(record.get("status") == "ok" for record in predictions.values()),
        "predicted_failed_count": sum(
            record.get("status") == "failed" for record in predictions.values()
        ),
        "status_counts": {
            "reference": {
                "ok": sum(record.get("status") == "ok" for record in references.values()),
                "failed": sum(record.get("status") == "failed" for record in references.values()),
            },
            "predicted": {
                "ok": sum(record.get("status") == "ok" for record in predictions.values()),
                "failed": sum(record.get("status") == "failed" for record in predictions.values()),
            },
        },
        "details": rows,
    }


def write_comparison(path: str | Path, result: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def hash_run(path: str | Path) -> dict[str, Any]:
    """Hash stable OCR content for a JSONL run."""
    return aggregate_ocr_hash(read_jsonl(path))


def _limited(records: Iterable[dict[str, Any]], limit: int | None) -> Iterable[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        yield record


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record requires a non-empty string {key!r}")
    return value


def _blocks_by_id(record: dict[str, Any], instance_id: str) -> dict[str, str]:
    blocks = record.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a list")
    result: dict[str, str] = {}
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("each block must be an object")
        block_id = _required_string(block, "block_id").casefold()
        if block_id in result:
            raise ValueError(f"duplicate block_id {block_id!r} in instance {instance_id}")
        result[block_id] = _required_string(block, "text")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _records_by_id(path: str | Path, label: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        instance_id = _required_string(record, "instance_id")
        if instance_id in records:
            raise ValueError(f"duplicate instance_id in {label}: {instance_id}")
        records[instance_id] = record
    return records


def _run_fingerprint(
    *,
    engine: str,
    dpi: int,
    language: str,
    options: dict[str, Any],
    input_sha256: str,
    timeout_seconds: float | None,
    ocr_executable_identity: dict[str, Any],
    renderer_executable_identity: dict[str, Any],
) -> str:
    payload = {
        "engine": engine,
        "dpi": dpi,
        "language": language,
        "options": options,
        "input_sha256": input_sha256,
        "timeout_seconds": timeout_seconds,
        "ocr_executable_identity": ocr_executable_identity,
        "renderer_executable_identity": renderer_executable_identity,
    }
    return deterministic_content_hash(payload)


def _validate_completed_manifest_record(completed: dict[str, Any], current: dict[str, Any]) -> None:
    for field in (
        "user_query",
        "document_pdf",
        "input_pdf_sha256",
        "split",
        "split_seed",
    ):
        if completed.get(field) != current.get(field):
            raise ValueError(f"stale resume manifest for {current['instance_id']}: {field} changed")


def _retry_checkpoint_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.name}.retry-checkpoint")


def _merged_retry_records(
    existing: dict[str, dict[str, Any]],
    replacements: dict[str, dict[str, Any]],
    replacement_order: list[str],
) -> list[dict[str, Any]]:
    merged = [replacements.get(instance_id, record) for instance_id, record in existing.items()]
    merged.extend(
        replacements[instance_id]
        for instance_id in replacement_order
        if instance_id not in existing
    )
    return merged


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    write_jsonl(temporary, records)
    temporary.replace(path)


def _executable_identity(executable: str | Path) -> dict[str, Any]:
    requested = Path(executable)
    resolved: Path | None = requested.resolve() if requested.is_file() else None
    if resolved is None:
        discovered = shutil.which(str(executable))
        resolved = Path(discovered).resolve() if discovered is not None else None
    return {
        "name": requested.name,
        "kind": "sha256" if resolved is not None else "command_name",
        "sha256": _sha256_file(resolved) if resolved is not None else None,
    }
