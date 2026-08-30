"""Label-free recovery and comparison of generated quantitative queries."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .records import read_jsonl
from .render import render_pdf

SCHEMA_VERSION = "1.0"
COMPARISON_KIND = "codex-query-comparison"
LEAD_INS = (
    "The planning desk recorded the following quantitative matter for consideration.",
    "the coordination team added the following scenario to its working notes.",
    "The group agreed that the following matter should be resolved before the afternoon briefing.",
)
_TASK_ID = re.compile(r"task_[0-9]{6}")
_BLOCK_MARKER = re.compile(r"^\s*(b(?:[0-9o]{2}|o[0-9]{2}))\b[ \t]*:?\s*(.*)$", re.IGNORECASE)
_LEAD_IN_TOKEN_PATTERNS = {"afternoon": r"afte(?:rnoon|moon)"}
_LEAD_IN_PATTERNS = tuple(
    re.compile(
        r"\s+".join(
            _LEAD_IN_TOKEN_PATTERNS.get(token.lower(), re.escape(token))
            for token in lead_in.split()
        ),
        re.IGNORECASE,
    )
    for lead_in in LEAD_INS
)
_OUTCOMES = ("exact", "normalized", "mismatch", "undetermined")


def compare_codex_queries(
    tasks_path: str | Path,
    codex_reference_path: str | Path,
    *,
    documents_root: str | Path | None = None,
    split_name: str | None = None,
    pdftotext_executable: str = "pdftotext",
    renderer_executable: str = "pdftoppm",
    tesseract_executable: str = "tesseract",
    fallback_dpi: int = 200,
    timeout_seconds: float = 30.0,
    workers: int = 1,
) -> dict[str, Any]:
    """Recover scenario queries without labels and compare them with manifest queries."""
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between one and four")
    if fallback_dpi <= 0:
        raise ValueError("fallback_dpi must be positive")
    tasks_source = Path(tasks_path).resolve()
    reference_source = Path(codex_reference_path).resolve()
    root = Path(documents_root).resolve() if documents_root else tasks_source.parent
    tasks = _records_by_id(tasks_source, kind="tasks manifest")
    references = _records_by_id(reference_source, kind="Codex reference output")
    missing = sorted(set(tasks) - set(references))
    extra = sorted(set(references) - set(tasks))
    if missing or extra:
        raise ValueError(
            f"Codex query comparison coverage mismatch: missing={missing}, extra={extra}"
        )

    tasks_sha256 = _sha256_file(tasks_source)
    reference_sha256 = _sha256_file(reference_source)
    resolved_split = split_name or _inferred_split(tasks_source)

    def compare_task(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        instance_id, task = item
        query = _required_string(task, "user_query", "tasks manifest")
        pdf_value = Path(_required_string(task, "document_pdf", "tasks manifest"))
        pdf_path = (pdf_value if pdf_value.is_absolute() else root / pdf_value).resolve()
        try:
            pdf_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"document_pdf escapes documents_root for {instance_id}") from exc
        pdf_sha256 = _sha256_file(pdf_path)
        record = _compare_one(
            instance_id,
            query,
            task.get("split") or resolved_split,
            references[instance_id],
            pdf_path=pdf_path,
            pdf_sha256=pdf_sha256,
            pdftotext_executable=pdftotext_executable,
            renderer_executable=renderer_executable,
            tesseract_executable=tesseract_executable,
            fallback_dpi=fallback_dpi,
            timeout_seconds=timeout_seconds,
        )
        record["source"] = {
            "tasks_manifest_sha256": tasks_sha256,
            "codex_reference_sha256": reference_sha256,
            "document_pdf_path": str(pdf_path),
            "document_pdf_sha256": pdf_sha256,
        }
        return record

    task_items = list(tasks.items())
    if workers == 1:
        records = [compare_task(item) for item in task_items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(compare_task, task_items))

    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_kind": COMPARISON_KIND,
        "sources": {
            "tasks_manifest": {"path": str(tasks_source), "sha256": tasks_sha256},
            "codex_reference_output": {
                "path": str(reference_source),
                "sha256": reference_sha256,
            },
            "documents_root": str(root),
            "split": resolved_split,
            "pdf_text_extraction": {
                "primary_executable": pdftotext_executable,
                "fallback_renderer_executable": renderer_executable,
                "fallback_ocr_executable": tesseract_executable,
                "fallback_dpi": fallback_dpi,
                "fallback_language": "eng",
                "fallback_page_segmentation_mode": 6,
            },
        },
        "summary": _summarize(records),
        "records": records,
    }


def write_codex_query_comparison(
    comparison: Mapping[str, Any],
    jsonl_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    """Write per-item JSONL and a label-free Markdown mismatch report."""
    jsonl_destination = Path(jsonl_path).resolve()
    markdown_destination = Path(markdown_path).resolve()
    if jsonl_destination == markdown_destination:
        raise ValueError("JSONL and Markdown output paths must differ")
    records = comparison.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("comparison records must be a sequence")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("comparison records must contain mappings")
    source_paths = _comparison_source_paths(comparison)
    collisions = sorted(
        str(path)
        for path in (jsonl_destination, markdown_destination)
        if path in source_paths
    )
    if collisions:
        raise ValueError(f"comparison outputs collide with source paths: {collisions}")
    forbidden = [
        sorted({"answer", "evidence"}.intersection(record))
        for record in records
        if {"answer", "evidence"}.intersection(record)
    ]
    if forbidden:
        raise ValueError(f"comparison records contain forbidden fields: {forbidden}")
    jsonl_payload = "".join(
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode()
    jsonl_sha256 = hashlib.sha256(jsonl_payload).hexdigest()
    markdown = _markdown_report(comparison, jsonl_destination, jsonl_sha256)
    _publish_completed_pair(
        jsonl_destination,
        jsonl_payload,
        markdown_destination,
        markdown.encode(),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_kind": COMPARISON_KIND,
        "sources": comparison.get("sources"),
        "outputs": {
            "jsonl": {"path": str(jsonl_destination), "sha256": jsonl_sha256},
            "markdown": {
                "path": str(markdown_destination),
                "sha256": _sha256_file(markdown_destination),
            },
        },
        "summary": comparison.get("summary"),
    }


def _compare_one(
    instance_id: str,
    query: str,
    split: object,
    reference: Mapping[str, Any],
    *,
    pdf_path: Path,
    pdf_sha256: str,
    pdftotext_executable: str,
    renderer_executable: str,
    tesseract_executable: str,
    fallback_dpi: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "comparison_kind": COMPARISON_KIND,
        "instance_id": instance_id,
        "split": split if isinstance(split, str) and split else None,
        "user_query": query,
        "recovered_query": None,
        "pdf_recovered_query": None,
        "evidence_block_id": None,
        "evidence_pages": [],
        "comparison_status": "undetermined",
        "category": "undetermined",
        "diff": [],
    }
    if reference.get("status") != "ok":
        base["undetermined_reason"] = "reference_status_not_ok"
        return base
    provenance = reference.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"missing reference provenance for {instance_id}")
    if provenance.get("input_pdf_sha256") != pdf_sha256:
        raise ValueError(f"reference PDF SHA-256 mismatch for {instance_id}")
    blocks = reference.get("blocks")
    if not isinstance(blocks, list):
        base["undetermined_reason"] = "invalid_reference_blocks"
        return base
    transcript_matches = _scenario_matches(blocks)
    if len(transcript_matches) != 1:
        base["undetermined_reason"] = f"reference_lead_in_matches_{len(transcript_matches)}"
        return base
    transcript_match = transcript_matches[0]
    base["recovered_query"] = transcript_match["query"]

    try:
        page_texts = _extract_pdf_pages(
            pdf_path,
            executable=pdftotext_executable,
            renderer_executable=renderer_executable,
            tesseract_executable=tesseract_executable,
            fallback_dpi=fallback_dpi,
            timeout_seconds=timeout_seconds,
        )
        pdf_blocks = _pdf_blocks(page_texts)
        pdf_matches = _scenario_matches(pdf_blocks)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError) as exc:
        base["undetermined_reason"] = f"pdf_extraction_failed:{type(exc).__name__}"
        return base
    if len(pdf_matches) != 1:
        base["undetermined_reason"] = f"pdf_lead_in_matches_{len(pdf_matches)}"
        return base
    pdf_match = pdf_matches[0]
    if pdf_match["block_id"].lower() != transcript_match["block_id"].lower():
        base["undetermined_reason"] = "reference_pdf_block_id_mismatch"
        return base

    recovered = str(transcript_match["query"])
    pdf_recovered = str(pdf_match["query"])
    base.update(
        pdf_recovered_query=pdf_recovered,
        evidence_block_id=pdf_match["block_id"].lower(),
        evidence_pages=list(pdf_match.get("page_numbers", [])),
    )
    category = _category(query, recovered, pdf_recovered)
    if category == "exact":
        status = "exact"
    elif category == "line_break_or_whitespace":
        status = "normalized"
    else:
        status = "mismatch"
        base["diff"] = _character_diff(query, recovered)
    base["comparison_status"] = status
    base["category"] = category
    base["pdf_sha256"] = pdf_sha256
    return base


def _scenario_matches(blocks: Sequence[object]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        block_id = block.get("block_id")
        text = block.get("text")
        if not isinstance(block_id, str) or not isinstance(text, str):
            continue
        found = [
            match for pattern in _LEAD_IN_PATTERNS for match in pattern.finditer(text)
        ]
        matches.extend(
            {
                "block_id": block_id,
                "query": text[match.end() :].strip(),
                "page_numbers": block.get("page_numbers", []),
            }
            for match in found
            if text[match.end() :].strip()
        )
    return matches


def _extract_pdf_pages(
    pdf_path: Path,
    *,
    executable: str,
    renderer_executable: str = "pdftoppm",
    tesseract_executable: str = "tesseract",
    fallback_dpi: int = 200,
    timeout_seconds: float,
) -> tuple[str, ...]:
    pages: list[str] = []
    for page_number in range(1, 10_001):
        completed = subprocess.run(
            [
                executable,
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-layout",
                str(pdf_path),
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            if pages and _is_page_range_end(completed.stderr):
                break
            detail = completed.stderr.strip() or f"exit status {completed.returncode}"
            raise ValueError(f"pdftotext failed for page {page_number}: {detail}")
        pages.append(completed.stdout)
    if not pages:
        raise ValueError("pdftotext produced no pages")
    if not any(page.strip() for page in pages):
        return _ocr_pdf_pages(
            pdf_path,
            renderer_executable=renderer_executable,
            tesseract_executable=tesseract_executable,
            dpi=fallback_dpi,
            timeout_seconds=timeout_seconds,
        )
    return tuple(pages)


def _ocr_pdf_pages(
    pdf_path: Path,
    *,
    renderer_executable: str,
    tesseract_executable: str,
    dpi: int,
    timeout_seconds: float,
) -> tuple[str, ...]:
    with tempfile.TemporaryDirectory(prefix="docinsights-query-ocr-") as temporary:
        images = render_pdf(
            pdf_path,
            temporary,
            dpi=dpi,
            executable=renderer_executable,
            timeout_seconds=timeout_seconds,
        )
        if len(images) != 2:
            raise ValueError(f"query comparison requires exactly two PDF pages, got {len(images)}")
        page_texts: list[str] = []
        for image in images:
            completed = subprocess.run(
                [
                    tesseract_executable,
                    str(image.resolve()),
                    "stdout",
                    "-l",
                    "eng",
                    "--psm",
                    "6",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            page_texts.append(completed.stdout)
    return tuple(page_texts)


def _is_page_range_end(stderr: str) -> bool:
    message = " ".join(stderr.lower().split())
    return "page range" in message or ("first page" in message and "last page" in message)


def _pdf_blocks(page_texts: Sequence[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for page_number, page_text in enumerate(page_texts, 1):
        for line in page_text.splitlines():
            marker = _BLOCK_MARKER.match(line)
            if marker:
                if current is not None:
                    current["text"] = "\n".join(current.pop("lines")).strip()
                    blocks.append(current)
                current = {
                    "block_id": _canonical_block_marker(marker.group(1)),
                    "lines": [marker.group(2)],
                    "page_numbers": [page_number],
                }
            elif current is not None:
                current["lines"].append(line)
                if line.strip() and page_number not in current["page_numbers"]:
                    current["page_numbers"].append(page_number)
    if current is not None:
        current["text"] = "\n".join(current.pop("lines")).strip()
        blocks.append(current)
    return blocks


def _canonical_block_marker(marker: str) -> str:
    suffix = marker.lower()[1:]
    if len(suffix) == 3 and suffix.startswith("o"):
        suffix = suffix[1:]
    return f"b{suffix.replace('o', '0')}"


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _without_punctuation(value: str) -> str:
    return "".join(
        character
        for character in _normalized(value)
        if not unicodedata.category(character).startswith("P")
    )


def _category(query: str, recovered: str, pdf_recovered: str) -> str:
    if query == recovered:
        return "exact"
    if _normalized(query) == _normalized(recovered):
        return "line_break_or_whitespace"
    if _normalized(_without_punctuation(query)) == _normalized(_without_punctuation(recovered)):
        return "punctuation"
    if _normalized(recovered) != _normalized(pdf_recovered):
        return "ocr"
    return "actual_content_difference"


def _character_diff(query: str, recovered: str) -> list[dict[str, Any]]:
    matcher = difflib.SequenceMatcher(None, query, recovered, autojunk=False)
    return [
        {
            "operation": operation,
            "user_query": {"start": i1, "end": i2, "text": query[i1:i2]},
            "recovered_query": {"start": j1, "end": j2, "text": recovered[j1:j2]},
        }
        for operation, i1, i2, j1, j2 in matcher.get_opcodes()
        if operation != "equal"
    ]


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = _summary_bucket(records)
    split_names = sorted({_record_split(record) for record in records})
    return {
        "total_count": len(records),
        "counts": overall["counts"],
        "instance_ids": overall["instance_ids"],
        "splits": {
            split: _summary_bucket(
                [
                    record
                    for record in records
                    if _record_split(record) == split
                ]
            )
            for split in split_names
        },
    }


def _record_split(record: Mapping[str, Any]) -> str:
    split = record.get("split")
    return split if isinstance(split, str) else "unspecified"


def _summary_bucket(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = {
        outcome: sorted(
            str(record["instance_id"])
            for record in records
            if record.get("comparison_status") == outcome
        )
        for outcome in _OUTCOMES
    }
    return {
        "total_count": len(records),
        "counts": {outcome: len(ids[outcome]) for outcome in _OUTCOMES},
        "category_counts": {
            category: sum(record.get("category") == category for record in records)
            for category in (
                "exact",
                "line_break_or_whitespace",
                "punctuation",
                "ocr",
                "actual_content_difference",
                "undetermined",
            )
        },
        "instance_ids": ids,
    }


def _inferred_split(tasks_path: Path) -> str:
    parent = tasks_path.parent.name.lower()
    if parent == "val":
        return "validation"
    if parent == "train":
        return "train"
    return "unspecified"


def _markdown_report(
    comparison: Mapping[str, Any], jsonl_path: Path, jsonl_sha256: str
) -> str:
    sources = comparison.get("sources")
    summary = comparison.get("summary")
    records = comparison.get("records")
    if not isinstance(sources, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("comparison must contain sources and summary mappings")
    if not isinstance(records, Sequence):
        raise ValueError("comparison must contain records")
    tasks = sources.get("tasks_manifest", {})
    reference = sources.get("codex_reference_output", {})
    lines = [
        "# Codex Query Comparison",
        "",
        "## Sources and outputs",
        "",
        f"- Tasks manifest: `{tasks.get('path')}` (`{tasks.get('sha256')}`)",
        f"- Codex reference output: `{reference.get('path')}` (`{reference.get('sha256')}`)",
        f"- Comparison JSONL: `{jsonl_path}` (`{jsonl_sha256}`)",
        "",
        "## Split summary",
        "",
        "| Split | Exact | Normalized | Mismatch | Undetermined |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    splits = summary.get("splits", {})
    if isinstance(splits, Mapping):
        for split, bucket in splits.items():
            counts = bucket.get("counts", {}) if isinstance(bucket, Mapping) else {}
            lines.append(
                f"| {split} | {counts.get('exact', 0)} | {counts.get('normalized', 0)} | "
                f"{counts.get('mismatch', 0)} | {counts.get('undetermined', 0)} |"
            )
        lines.extend(["", "### Instance IDs", ""])
        for split, bucket in splits.items():
            ids = bucket.get("instance_ids", {}) if isinstance(bucket, Mapping) else {}
            for outcome in _OUTCOMES:
                outcome_ids = ids.get(outcome, []) if isinstance(ids, Mapping) else []
                lines.append(
                    f"- `{split}` {outcome} instance IDs: "
                    f"{', '.join(f'`{instance_id}`' for instance_id in outcome_ids) or 'None'}"
                )
    mismatches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("comparison_status") == "mismatch"
    ]
    lines.extend(["", "## Mismatches", ""])
    if not mismatches:
        lines.append("None.")
    for record in mismatches:
        lines.extend(
            [
                f"### {record.get('instance_id')}",
                "",
                f"- Category: `{record.get('category')}`",
                f"- Evidence: `{record.get('evidence_block_id')}`, pages "
                f"`{record.get('evidence_pages')}`",
                "",
                "Query:",
                "",
                "```text",
                str(record.get("user_query", "")),
                "```",
                "",
                "Recovered query:",
                "",
                "```text",
                str(record.get("recovered_query", "")),
                "```",
                "",
                "PDF recovered query:",
                "",
                "```text",
                str(record.get("pdf_recovered_query", "")),
                "```",
                "",
                "Character diff:",
                "",
                "```json",
                json.dumps(record.get("diff", []), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    undetermined = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("comparison_status") == "undetermined"
    ]
    lines.extend(["", "## Undetermined", ""])
    if not undetermined:
        lines.append("None.")
    for record in undetermined:
        lines.extend(
            [
                f"### {record.get('instance_id')}",
                "",
                f"- Reason: `{record.get('undetermined_reason')}`",
                f"- Evidence: `{record.get('evidence_block_id')}`, pages "
                f"`{record.get('evidence_pages')}`",
                "",
                "Query:",
                "",
                "```text",
                str(record.get("user_query", "")),
                "```",
                "",
                "Recovered query:",
                "",
                "```text",
                str(record.get("recovered_query", "")),
                "```",
                "",
                "PDF recovered query:",
                "",
                "```text",
                str(record.get("pdf_recovered_query", "")),
                "```",
                "",
                "Character diff:",
                "",
                "```json",
                json.dumps(record.get("diff", []), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _records_by_id(path: Path, *, kind: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        instance_id = _required_string(record, "instance_id", kind)
        if _TASK_ID.fullmatch(instance_id) is None:
            raise ValueError(f"invalid instance_id in {kind}: {instance_id}")
        if instance_id in records:
            raise ValueError(f"duplicate instance_id in {kind}: {instance_id}")
        records[instance_id] = record
    return records


def _required_string(record: Mapping[str, Any], field: str, kind: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing or invalid {field} in {kind}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison_source_paths(comparison: Mapping[str, Any]) -> set[Path]:
    sources = comparison.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("comparison must contain sources mapping")
    paths: set[Path] = set()
    for key in ("tasks_manifest", "codex_reference_output"):
        source = sources.get(key)
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            raise ValueError(f"comparison source {key} must contain a path")
        paths.add(Path(source["path"]).resolve())
    return paths


def _publish_completed_pair(
    first_path: Path,
    first_payload: bytes,
    second_path: Path,
    second_payload: bytes,
) -> None:
    temporary_paths: list[Path] = []
    try:
        for destination, payload in (
            (first_path, first_payload),
            (second_path, second_payload),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        temporary_paths[0].replace(first_path)
        temporary_paths.pop(0)
        temporary_paths[0].replace(second_path)
        temporary_paths.pop(0)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
