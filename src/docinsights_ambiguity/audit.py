"""Deterministic, label-separated screening of DocSem training questions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
AUDIT_KIND = "docsem-train-ambiguity-screen"
ADJUDICATION_STATUS = "automated_screened_agent_text_audited_human_adjudication_pending"
DISCLAIMER = "Automated screening and agent text audit; not completed human adjudication."
SURFACE_VALUES = {"intact", "awkward_but_parseable", "corrupted"}
SEMANTIC_VALUES = {
    "unique_explicit",
    "unique_with_convention",
    "multiple_plausible",
    "underdetermined",
}
ALIGNMENT_VALUES = {
    "aligned",
    "normalized_equivalent",
    "semantic_conflict",
    "label_unverifiable",
}
SEVERITIES = {"S0", "S1", "S2", "S3", "S4"}
ISSUE_TAGS = {
    "clean",
    "comparison_polarity_or_sign",
    "role_or_subject_attribution",
    "condition_insertion_or_deletion",
    "cardinality_operand_gap",
    "quantifier_scope_last_remaining",
    "ordinal_or_count_mismatch",
    "rate_unit_relation_corruption",
    "target_or_subquestion_drift",
    "distractor_operand_interference",
    "label_template_carryover_suspected",
    "answer_normalization_abs_sign",
    "implicit_default_assumption",
    "extraction_or_ocr_uncertain",
}
_TAG_ALIASES = {
    "comparison_sign_direction": "comparison_polarity_or_sign",
    "role_attribution": "role_or_subject_attribution",
    "condition_insertion_deletion": "condition_insertion_or_deletion",
    "quantifier_last_vs_remaining": "quantifier_scope_last_remaining",
    "ordinal_mismatch": "ordinal_or_count_mismatch",
    "rate_unit_corruption": "rate_unit_relation_corruption",
    "target_subquestion_drift": "target_or_subquestion_drift",
    "distractor": "distractor_operand_interference",
    "label_carryover_suspected": "label_template_carryover_suspected",
    "abs_sign_normalization": "answer_normalization_abs_sign",
    "implicit_assumption": "implicit_default_assumption",
    "ocr_uncertainty": "extraction_or_ocr_uncertain",
}
_NUMBER = r"-?\d+(?:\.\d+)?"
_ORDINAL = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b",
    re.I,
)
_ROLE = re.compile(r"\b(?:respectively|former|latter|he|she|his|her|their)\b", re.I)
_CONDITION = re.compile(r"\b(?:unless|except|after|before|provided that|only if)\b", re.I)
_QUANTIFIER = re.compile(r"\b(?:last|remaining|remainder|rest|left over|left)\b", re.I)
_COMPARISON = re.compile(
    r"\b(?:more than|less than|fewer than|greater than|difference between|increase|decrease)\b",
    re.I,
)
_ABS_SIGN = re.compile(r"\b(?:difference|absolute|negative|positive|increase|decrease)\b", re.I)
_RATE = re.compile(
    r"\b(?:rate|per (?:hour|minute|day|week)|every \d+(?:\.\d+)? (?:minutes?|hours?|days?))\b",
    re.I,
)
_ASSUMPTION = re.compile(
    r"\b(?:uniformly|constant(?: rate| speed)?|assum(?:e|ing)|same rate)\b", re.I
)
_DISTRACTOR = re.compile(r"\b(?:unrelated|irrelevant|despite|although|however)\b", re.I)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_FIELDS = {
    "benchmark_block_ids",
    "recovered_block_id",
    "recovered_pages",
    "recovered_query_sha256",
    "recovered_block_text_sha256",
    "document_pdf_sha256",
    "codex_reference_file_sha256",
    "tasks_manifest_file_sha256",
    "benchmark_blocks_present_in_reference",
    "benchmark_matches_recovered_block",
}


@dataclass(frozen=True)
class SemanticAnswer:
    value: str | None
    status: str
    method: str | None


def validate_schema_contract(schema: object) -> list[str]:
    """Validate the checked-in JSON Schema without requiring a third-party validator."""
    if not isinstance(schema, Mapping):
        return ["schema root must be an object"]
    errors: list[str] = []
    expected_properties = {
        "schema_version",
        "audit_kind",
        "screening_statement",
        "adjudication_status",
        "screening_classification",
        "instance_id",
        "split",
        "blind_question_screen",
        "axes",
        "semantic_answer",
        "benchmark_answer",
        "issue_tags",
        "severity",
        "benchmark_risk",
        "auto_signals",
        "evidence",
        "agent_reviews",
        "review_required",
    }
    properties = schema.get("properties")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append("schema draft identifier is invalid")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        errors.append("schema root must be a closed object")
    if not isinstance(properties, Mapping) or set(properties) != expected_properties:
        errors.append("schema properties do not match the record contract")
    if set(schema.get("required", [])) != expected_properties:
        errors.append("schema required fields do not match the record contract")
    definitions = schema.get("$defs")
    if not isinstance(definitions, Mapping):
        errors.append("schema definitions are missing")
        return errors
    issue_tags = definitions.get("issue_tags")
    try:
        tag_enum = set(issue_tags["items"]["enum"])
    except (KeyError, TypeError):
        errors.append("schema issue-tag enumeration is missing")
    else:
        if tag_enum != ISSUE_TAGS:
            errors.append("schema issue-tag enumeration does not match the taxonomy")
    if isinstance(properties, Mapping):
        agent_reviews = properties.get("agent_reviews", {})
        if agent_reviews.get("minItems") != 1 or agent_reviews.get("maxItems") != 1:
            errors.append("schema must require exactly one agent review")
        evidence = properties.get("evidence", {})
        if evidence.get("additionalProperties") is not False:
            errors.append("schema evidence object must be closed")
        if set(evidence.get("required", [])) != _EVIDENCE_FIELDS:
            errors.append("schema evidence required fields do not match the provenance contract")
        evidence_properties = evidence.get("properties")
        if (
            not isinstance(evidence_properties, Mapping)
            or set(evidence_properties) != _EVIDENCE_FIELDS
        ):
            errors.append("schema evidence properties do not match the provenance contract")
    return errors


def build_blind_screen(tasks_path: str | Path, comparison_path: str | Path) -> dict[str, Any]:
    """Complete the label-blind pass; this function cannot receive a label path."""
    tasks = _read_unique(Path(tasks_path).resolve(), "tasks")
    comparisons = _read_unique(Path(comparison_path).resolve(), "query comparison")
    _require_coverage("query comparison", set(tasks), set(comparisons))
    return {
        "records": [screen_question(key, comparisons[key]) for key in tasks],
        "sources": {
            "tasks": _source(Path(tasks_path).resolve()),
            "query_comparison": _source(Path(comparison_path).resolve()),
        },
    }


def write_blind_screen(blind: Mapping[str, Any], output_path: str | Path) -> dict[str, Any]:
    records = _record_sequence(blind, "blind")
    destination = Path(output_path).resolve()
    _write_jsonl(destination, records)
    return {"path": str(destination), "sha256": _sha256(destination), "record_count": len(records)}


def build_audit(
    tasks_path: str | Path,
    labels_path: str | Path,
    reference_path: str | Path,
    comparison_path: str | Path,
    blind_path: str | Path,
    *,
    review_shards: Sequence[str | Path] = (),
    expected_count: int = 908,
) -> dict[str, Any]:
    """Consume an already-persisted blind layer, then compare labels and agent reviews."""
    paths = {
        "tasks": Path(tasks_path).resolve(),
        "labels": Path(labels_path).resolve(),
        "reference": Path(reference_path).resolve(),
        "query_comparison": Path(comparison_path).resolve(),
        "blind_screen": Path(blind_path).resolve(),
    }
    tasks = _read_unique(paths["tasks"], "tasks")
    labels = _read_unique(paths["labels"], "labels")
    references = _read_unique(paths["reference"], "reference")
    comparisons = _read_unique(paths["query_comparison"], "query comparison")
    blind = _read_unique(paths["blind_screen"], "blind screen")
    if len(tasks) != expected_count:
        raise ValueError(f"task record count {len(tasks)} != contract count {expected_count}")
    tasks_file_sha256 = _sha256(paths["tasks"])
    reference_file_sha256 = _sha256(paths["reference"])
    expected = set(tasks)
    for name, records in (
        ("labels", labels),
        ("reference", references),
        ("query comparison", comparisons),
        ("blind screen", blind),
    ):
        _require_coverage(name, expected, set(records))
    for key in tasks:
        recovered = comparisons[key].get("recovered_query")
        normalized = " ".join(recovered.split()) if isinstance(recovered, str) else None
        digest = hashlib.sha256(normalized.encode()).hexdigest() if normalized else None
        if blind[key].get("question_sha256") != digest:
            raise ValueError(f"blind screen question hash mismatch for {key}")
        source = comparisons[key].get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"query comparison source provenance missing for {key}")
        if source.get("tasks_manifest_sha256") != tasks_file_sha256:
            raise ValueError(f"tasks manifest source hash mismatch for {key}")
        if source.get("codex_reference_sha256") != reference_file_sha256:
            raise ValueError(f"Codex reference source hash mismatch for {key}")
    reviews, review_sources = _read_reviews(review_shards, expected)
    records = [
        _compare_benchmark(
            tasks[key], labels[key], references[key], comparisons[key], blind[key], reviews.get(key)
        )
        for key in tasks
    ]
    _mark_label_carryover(records)
    return {
        "records": records,
        "sources": {name: _source(path) for name, path in paths.items()},
        "review_sources": review_sources,
        "expected_count": expected_count,
        "expected_instance_ids": list(tasks),
    }


def screen_question(instance_id: str, comparison: Mapping[str, Any]) -> dict[str, Any]:
    query = comparison.get("recovered_query")
    if not isinstance(query, str) or not query.strip():
        return {
            "instance_id": instance_id,
            "question": None,
            "question_sha256": None,
            "surface_integrity": "corrupted",
            "semantic_determinacy": "underdetermined",
            "semantic_answer": {"value": None, "status": "unresolved", "method": None},
            "issue_tags": ["extraction_or_ocr_uncertain"],
            "auto_signals": [{"code": "missing_recovered_question", "detail": instance_id}],
        }
    query = " ".join(query.split())
    signals: list[dict[str, str]] = []
    for pattern, code in (
        (_COMPARISON, "comparison_language"),
        (_ROLE, "role_language"),
        (_CONDITION, "condition_language"),
        (_QUANTIFIER, "quantifier_language"),
        (_ORDINAL, "ordinal_language"),
        (_ABS_SIGN, "sign_sensitive_language"),
        (_ASSUMPTION, "assumption_language"),
        (_DISTRACTOR, "distractor_language"),
    ):
        match = pattern.search(query)
        if match:
            signals.append({"code": code, "detail": match.group(0)})
    if comparison.get("category") == "ocr":
        signals.append({"code": "ocr_normalization_used", "detail": "comparison category=ocr"})
    numbers = re.findall(_NUMBER, query)
    if re.search(r"\b(?:each|per)\b", query, re.I) and len(numbers) >= 3:
        signals.append({"code": "cardinality_cue", "detail": f"numeric_tokens={len(numbers)}"})
    if len(_RATE.findall(query)) >= 2:
        signals.append({"code": "multiple_rate_cues", "detail": "two or more rate phrases"})
    if query.count("?") > 1:
        signals.append({"code": "multiple_question_marks", "detail": f"count={query.count('?')}"})

    tags: set[str] = set()
    surface = "intact"
    determinacy = "unique_explicit"
    if _has_rate_conflict(query):
        tags.add("rate_unit_relation_corruption")
        surface = "awkward_but_parseable"
        determinacy = "multiple_plausible"
    if re.search(
        r"\bpercentage more likely\b|\bmore likely.*expressed as a percentage\b", query, re.I
    ):
        tags.update({"comparison_polarity_or_sign", "answer_normalization_abs_sign"})
        determinacy = "multiple_plausible"
    if _has_target_replacement(query):
        tags.add("target_or_subquestion_drift")
        determinacy = "multiple_plausible"
    if _has_absolute_amount_ratio_conflict(query):
        tags.update({"condition_insertion_or_deletion", "implicit_default_assumption"})
        surface = "awkward_but_parseable"
        determinacy = "unique_with_convention"
    if re.search(r"\b(?:last|remaining)\b.*\b(?:last|remaining)\b", query, re.I):
        tags.add("quantifier_scope_last_remaining")
    if re.search(r"\b(?:unrelated|irrelevant)\b", query, re.I):
        tags.add("distractor_operand_interference")
    if re.search(r"\b\d+(?:\.\d+)?\s*=\s*\d+\b|\b(?:it|to|the)[a-z]{8,}\b", query):
        tags.add("extraction_or_ocr_uncertain")
        surface = "awkward_but_parseable"
    semantic = (
        _semantic_answer(query)
        if determinacy == "unique_explicit"
        else SemanticAnswer(None, "not_computed", None)
    )
    return {
        "instance_id": instance_id,
        "question": query,
        "question_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "surface_integrity": surface,
        "semantic_determinacy": determinacy,
        "semantic_answer": {
            "value": semantic.value,
            "status": semantic.status,
            "method": semantic.method,
        },
        "issue_tags": sorted(tags),
        "auto_signals": signals,
    }


def _compare_benchmark(
    task: Mapping[str, Any],
    label: Mapping[str, Any],
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    blind: Mapping[str, Any],
    review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    instance_id = _required_id(task, "task")
    benchmark = label.get("answer")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError(f"invalid benchmark answer for {instance_id}")
    semantic = dict(blind["semantic_answer"])
    alignment = _alignment_for(semantic["value"], benchmark)
    tags = set(blind["issue_tags"])
    severity = _auto_severity(tags, alignment)
    surface = str(blind["surface_integrity"])
    determinacy = str(blind["semantic_determinacy"])
    review_layers: list[dict[str, Any]] = []
    if review is not None:
        review_layers.append(dict(review))
        severity = str(review.get("severity", severity))
        reviewed_tags = {
            _TAG_ALIASES.get(str(tag), str(tag)) for tag in review.get("issue_tags", tags)
        }
        tags = reviewed_tags if severity == "S0" else tags | reviewed_tags
        if "clean" in tags and len(tags) > 1:
            tags.remove("clean")
        surface = review.get("surface_integrity")
        determinacy = review.get("semantic_determinacy")
        reviewed_alignment = review.get("benchmark_alignment", review.get("label_alignment"))
        reviewed_semantic = review.get("semantic_answer")
        if isinstance(reviewed_semantic, Mapping):
            value = reviewed_semantic.get("value")
            method = reviewed_semantic.get("method")
            if not isinstance(value, str) or not value or not isinstance(method, str) or not method:
                raise ValueError(f"invalid review semantic answer for {instance_id}")
            semantic = {"value": value, "status": "computed", "method": method}
            derived_alignment = _alignment_for(value, benchmark)
            if (
                reviewed_alignment in ALIGNMENT_VALUES - {"label_unverifiable"}
                and reviewed_alignment != derived_alignment
            ):
                raise ValueError(
                    f"review alignment does not match semantic answer for {instance_id}: "
                    f"reported={reviewed_alignment}, derived={derived_alignment}"
                )
        surface = str(surface) if surface in SURFACE_VALUES else str(blind["surface_integrity"])
        determinacy = (
            str(determinacy)
            if determinacy in SEMANTIC_VALUES
            else str(blind["semantic_determinacy"])
        )
        if reviewed_alignment in ALIGNMENT_VALUES:
            alignment = reviewed_alignment
    if severity == "S0" and not tags:
        tags.add("clean")
    if severity != "S0":
        tags.discard("clean")
    evidence_ids = label.get("evidence")
    if not isinstance(evidence_ids, list) or not all(
        isinstance(value, str) for value in evidence_ids
    ):
        evidence_ids = []
    reference_blocks = {
        block.get("block_id"): block
        for block in reference.get("blocks", [])
        if isinstance(block, Mapping) and isinstance(block.get("block_id"), str)
    }
    reference_block_ids = set(reference_blocks)
    recovered_block = comparison.get("evidence_block_id")
    if not isinstance(recovered_block, str) or recovered_block not in reference_blocks:
        raise ValueError(f"recovered evidence block missing from reference for {instance_id}")
    recovered_block_text = reference_blocks[recovered_block].get("text")
    if not isinstance(recovered_block_text, str):
        raise ValueError(f"recovered evidence block text missing for {instance_id}")
    comparison_source = comparison.get("source")
    reference_provenance = reference.get("provenance")
    if not isinstance(comparison_source, Mapping) or not isinstance(reference_provenance, Mapping):
        raise ValueError(f"evidence provenance missing for {instance_id}")
    document_hashes = {
        _require_sha256(comparison.get("pdf_sha256"), f"comparison PDF for {instance_id}"),
        _require_sha256(
            comparison_source.get("document_pdf_sha256"), f"source PDF for {instance_id}"
        ),
        _require_sha256(
            reference_provenance.get("input_pdf_sha256"), f"reference PDF for {instance_id}"
        ),
    }
    if len(document_hashes) != 1:
        raise ValueError(f"document PDF provenance mismatch for {instance_id}")
    recovered_query_sha256 = _require_sha256(
        blind.get("question_sha256"), f"recovered query for {instance_id}"
    )
    evidence = {
        "benchmark_block_ids": evidence_ids,
        "recovered_block_id": recovered_block,
        "recovered_pages": comparison.get("evidence_pages", []),
        "recovered_query_sha256": recovered_query_sha256,
        "recovered_block_text_sha256": hashlib.sha256(recovered_block_text.encode()).hexdigest(),
        "document_pdf_sha256": document_hashes.pop(),
        "codex_reference_file_sha256": _require_sha256(
            comparison_source.get("codex_reference_sha256"),
            f"Codex reference file for {instance_id}",
        ),
        "tasks_manifest_file_sha256": _require_sha256(
            comparison_source.get("tasks_manifest_sha256"),
            f"tasks manifest file for {instance_id}",
        ),
        "benchmark_blocks_present_in_reference": all(
            value in reference_block_ids for value in evidence_ids
        ),
        "benchmark_matches_recovered_block": recovered_block in evidence_ids,
    }
    classification = "clean_candidate" if severity == "S0" else "flagged"
    review_required = severity in {"S2", "S3", "S4"} or alignment == "semantic_conflict"
    if review is not None and isinstance(review.get("review_required"), bool):
        review_required = review["review_required"]
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": AUDIT_KIND,
        "screening_statement": DISCLAIMER,
        "adjudication_status": ADJUDICATION_STATUS,
        "screening_classification": classification,
        "instance_id": instance_id,
        "split": "train",
        "blind_question_screen": dict(blind),
        "axes": {
            "surface_integrity": surface,
            "semantic_determinacy": determinacy,
            "benchmark_alignment": alignment,
        },
        "semantic_answer": semantic,
        "benchmark_answer": benchmark,
        "issue_tags": sorted(tags),
        "severity": severity,
        "benchmark_risk": _risk(severity),
        "auto_signals": list(blind["auto_signals"]),
        "evidence": evidence,
        "agent_reviews": review_layers,
        "review_required": review_required,
    }


def _semantic_answer(query: str) -> SemanticAnswer:
    percent = re.search(
        rf"(?:a|the) ({_NUMBER})[- ]foot\b.*?\b({_NUMBER})\b[^.?!]*?"
        rf"each (?:measuring )?({_NUMBER}) inches?[^?]*?\bpercentage",
        query,
        re.I,
    )
    if percent:
        feet, count, inches = map(Decimal, percent.groups())
        if feet:
            return _computed(
                count * inches * Decimal(100) / (feet * Decimal(12)), "percent_of_length"
            )
    ratio = re.search(
        rf"ratio of ({_NUMBER}) parts?\b.*?to ({_NUMBER}) parts?\b.*?"
        rf"({_NUMBER})\s+(?:teaspoons?|liters?) in total",
        query,
        re.I,
    )
    if ratio:
        first, second, total = map(Decimal, ratio.groups())
        if first + second:
            return _computed(total * first / (first + second), "ratio_share")
    coverage = re.search(
        rf"(?:takes? )?({_NUMBER}) minutes? to cover every ({_NUMBER}) "
        rf"(?:miles?|kilometers?|km|inches?).*?(?:is|spans?) ({_NUMBER}) "
        rf"(?:miles?|kilometers?|km|inches?)",
        query,
        re.I,
    )
    if coverage:
        minutes, amount, total = map(Decimal, coverage.groups())
        if amount:
            return _computed(total / amount * minutes, "constant_rate_coverage")
    return SemanticAnswer(None, "not_computed", None)


def _has_rate_conflict(query: str) -> bool:
    per_hour = re.search(rf"(?:rate of )?({_NUMBER}) [a-z ]+ per hour", query, re.I)
    interval = re.search(rf"each ({_NUMBER})-minute interval covers ({_NUMBER})", query, re.I)
    if not per_hour or not interval:
        return False
    hourly = Decimal(interval.group(2)) * Decimal(60) / Decimal(interval.group(1))
    return Decimal(per_hour.group(1)) != hourly


def _has_target_replacement(query: str) -> bool:
    sided = re.findall(r"\b(?:an? )?(\w+)-sided (?:die|spinner)\b", query, re.I)
    return len(set(value.lower() for value in sided)) > 1 and bool(
        re.search(r"\b(?:instead|for comparison|but since)\b", query, re.I)
    )


def _has_absolute_amount_ratio_conflict(query: str) -> bool:
    match = re.search(
        rf"used ({_NUMBER}) [^.]+ and ({_NUMBER}) [^.]+\..*?total of ({_NUMBER})",
        query,
        re.I,
    )
    return bool(
        match and Decimal(match.group(1)) + Decimal(match.group(2)) != Decimal(match.group(3))
    )


def _computed(value: Decimal, method: str) -> SemanticAnswer:
    normalized = format(value.normalize(), "f")
    return SemanticAnswer(
        normalized.rstrip("0").rstrip(".") if "." in normalized else normalized, "computed", method
    )


def _mark_label_carryover(records: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for record in records:
        if (
            previous
            and record["axes"]["benchmark_alignment"] == "semantic_conflict"
            and record["benchmark_answer"] == previous["benchmark_answer"]
            and _template(record["blind_question_screen"]["question"] or "")
            == _template(previous["blind_question_screen"]["question"] or "")
        ):
            record["issue_tags"] = sorted(
                set(record["issue_tags"]) | {"label_template_carryover_suspected"}
            )
            record["severity"] = "S4"
            record["benchmark_risk"] = "high"
            record["screening_classification"] = "flagged"
            record["review_required"] = True
        previous = record


def write_audit(
    result: Mapping[str, Any], output_path: str | Path, summary_path: str | Path
) -> dict[str, Any]:
    records = _record_sequence(result, "audit")
    expected_count = result.get("expected_count")
    expected_ids = result.get("expected_instance_ids")
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("audit result lacks a positive contract count")
    if not isinstance(expected_ids, list) or not all(
        isinstance(value, str) for value in expected_ids
    ):
        raise ValueError("audit result lacks the contract instance IDs")
    output = Path(output_path).resolve()
    summary_destination = Path(summary_path).resolve()
    _write_jsonl(output, records)
    summary = _summary(records, result.get("sources", {}), result.get("review_sources", []), output)
    summary["validation"] = validate_artifacts_from_records(
        records, expected_count=expected_count, expected_ids=set(expected_ids)
    )
    _atomic_write(
        summary_destination,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def validate_artifacts(
    tasks_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    *,
    expected_count: int = 908,
) -> dict[str, Any]:
    tasks = _read_unique(Path(tasks_path).resolve(), "tasks")
    tasks_source = Path(tasks_path).resolve()
    output_source = Path(output_path).resolve()
    records = list(_read_jsonl(output_source))
    validation = validate_artifacts_from_records(
        records, expected_count=expected_count, expected_ids=set(tasks)
    )
    record_validation = {
        key: list(value) if key == "errors" else value for key, value in validation.items()
    }
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        validation["errors"].append(f"invalid summary: {exc}")
    else:
        if summary.get("record_count") != len(records):
            validation["errors"].append("summary record_count mismatch")
        if summary.get("screening_statement") != DISCLAIMER:
            validation["errors"].append("summary screening disclaimer mismatch")
        expected = _summary(
            records,
            summary.get("sources", {}),
            summary.get("review_sources", []),
            output_source,
        )
        if summary.get("counts") != expected["counts"]:
            validation["errors"].append("summary counts mismatch")
        if summary.get("scope") != expected["scope"]:
            validation["errors"].append("summary scope mismatch")
        output_report = summary.get("output")
        if not isinstance(output_report, Mapping):
            validation["errors"].append("summary output provenance missing")
        else:
            if output_report.get("sha256") != _sha256(output_source):
                validation["errors"].append("summary output SHA-256 mismatch")
            reported_path = output_report.get("path")
            if not isinstance(reported_path, str) or _resolve_reported_path(
                reported_path
            ) != output_source:
                validation["errors"].append("summary output path mismatch")
        if summary.get("validation") != record_validation:
            validation["errors"].append("summary embedded validation mismatch")
        sources = summary.get("sources")
        if not isinstance(sources, Mapping):
            validation["errors"].append("summary sources missing")
        else:
            for name, source in sources.items():
                validation["errors"].extend(_validate_source_report(str(name), source))
            task_source = sources.get("tasks")
            if not isinstance(task_source, Mapping) or task_source.get("sha256") != _sha256(
                tasks_source
            ):
                validation["errors"].append("summary tasks source SHA-256 mismatch")
        review_sources = summary.get("review_sources")
        if not isinstance(review_sources, list):
            validation["errors"].append("summary review sources missing")
        else:
            for index, source in enumerate(review_sources):
                validation["errors"].extend(
                    _validate_source_report(f"review[{index}]", source)
                )
    validation["passed"] = not validation["errors"]
    validation["status"] = "passed" if validation["passed"] else "failed"
    validation["output_sha256"] = _sha256(Path(output_path))
    validation["summary_sha256"] = _sha256(Path(summary_path))
    return validation


def validate_artifacts_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_ids: set[str] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    ids: list[str] = []
    if len(records) != expected_count:
        errors.append(f"record count {len(records)} != {expected_count}")
    for line, record in enumerate(records, 1):
        errors.extend(f"line {line}: {error}" for error in _validate_record(record))
        if isinstance(record.get("instance_id"), str):
            ids.append(record["instance_id"])
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate instance_ids: {duplicates}")
    if expected_ids is not None and set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        errors.append(f"coverage mismatch: missing={missing}, extra={extra}")
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "record_count": len(records),
        "unique_instance_count": len(set(ids)),
        "schema_error_count": sum(error.startswith("line ") for error in errors),
        "errors": errors,
    }


def _validate_record(record: Mapping[str, Any]) -> list[str]:
    required = {
        "schema_version",
        "audit_kind",
        "screening_statement",
        "adjudication_status",
        "screening_classification",
        "instance_id",
        "split",
        "blind_question_screen",
        "axes",
        "semantic_answer",
        "benchmark_answer",
        "issue_tags",
        "severity",
        "benchmark_risk",
        "auto_signals",
        "evidence",
        "agent_reviews",
        "review_required",
    }
    missing = sorted(required - set(record))
    if missing:
        return [f"missing fields {missing}"]
    errors: list[str] = []
    axes = record.get("axes")
    if not isinstance(axes, Mapping) or set(axes) != {
        "surface_integrity",
        "semantic_determinacy",
        "benchmark_alignment",
    }:
        errors.append("axes must contain exactly the three exclusive axes")
    elif (
        axes["surface_integrity"] not in SURFACE_VALUES
        or axes["semantic_determinacy"] not in SEMANTIC_VALUES
        or axes["benchmark_alignment"] not in ALIGNMENT_VALUES
    ):
        errors.append("invalid axis value")
    tags = record.get("issue_tags")
    if not isinstance(tags, list) or any(tag not in ISSUE_TAGS for tag in tags):
        errors.append("invalid issue_tags")
    severity = record.get("severity")
    if severity not in SEVERITIES:
        errors.append("invalid severity")
    elif severity == "S0" and tags != ["clean"]:
        errors.append("S0 record must have exactly the clean tag")
    elif severity != "S0" and (not tags or "clean" in tags):
        errors.append("S1-S4 record must have at least one non-clean tag")
    if record.get("screening_statement") != DISCLAIMER:
        errors.append("invalid automated-screening disclaimer")
    semantic = record.get("semantic_answer")
    benchmark = record.get("benchmark_answer")
    if isinstance(axes, Mapping) and axes.get("benchmark_alignment") in {
        "aligned",
        "normalized_equivalent",
        "semantic_conflict",
    }:
        if not isinstance(semantic, Mapping) or not isinstance(semantic.get("value"), str):
            errors.append("proof-bearing alignment requires a semantic answer")
        elif not isinstance(semantic.get("method"), str) or not semantic["method"]:
            errors.append("proof-bearing alignment requires a semantic method")
        elif isinstance(benchmark, str):
            derived = _alignment_for(semantic["value"], benchmark)
            if axes["benchmark_alignment"] != derived:
                errors.append("benchmark alignment disagrees with the semantic answer")
    blind = record.get("blind_question_screen")
    if not isinstance(blind, Mapping) or "benchmark_answer" in blind:
        errors.append("blind layer contains benchmark data")
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_FIELDS:
        errors.append("evidence must contain exactly the provenance contract fields")
    else:
        for field in (
            "recovered_query_sha256",
            "recovered_block_text_sha256",
            "document_pdf_sha256",
            "codex_reference_file_sha256",
            "tasks_manifest_file_sha256",
        ):
            if not isinstance(evidence[field], str) or not _SHA256.fullmatch(evidence[field]):
                errors.append(f"evidence has invalid {field}")
        if isinstance(blind, Mapping) and evidence["recovered_query_sha256"] != blind.get(
            "question_sha256"
        ):
            errors.append("evidence query hash disagrees with the blind layer")
        if not isinstance(evidence["benchmark_block_ids"], list) or not all(
            isinstance(value, str) and value for value in evidence["benchmark_block_ids"]
        ):
            errors.append("evidence benchmark block IDs are invalid")
        if not isinstance(evidence["recovered_block_id"], str) or not evidence[
            "recovered_block_id"
        ]:
            errors.append("evidence recovered block ID is invalid")
        if not isinstance(evidence["recovered_pages"], list) or not all(
            isinstance(value, int) and value >= 1 for value in evidence["recovered_pages"]
        ):
            errors.append("evidence recovered pages are invalid")
        for field in (
            "benchmark_blocks_present_in_reference",
            "benchmark_matches_recovered_block",
        ):
            if not isinstance(evidence[field], bool):
                errors.append(f"evidence has invalid {field}")
    reviews = record.get("agent_reviews")
    if not isinstance(reviews, list) or len(reviews) > 1:
        errors.append("record may contain at most one agent text review")
    elif reviews:
        review = reviews[0]
        if not isinstance(review.get("rationale"), str) or not review["rationale"]:
            errors.append("agent review lacks rationale")
        confidence = review.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append("agent review lacks valid confidence")
        for field, allowed in (
            ("surface_integrity", SURFACE_VALUES),
            ("semantic_determinacy", SEMANTIC_VALUES),
            ("benchmark_alignment", ALIGNMENT_VALUES),
        ):
            if review.get(field) not in allowed:
                errors.append(f"agent review lacks valid {field}")
        if review.get("benchmark_alignment") in {
            "aligned",
            "normalized_equivalent",
            "semantic_conflict",
        }:
            review_semantic = review.get("semantic_answer")
            if not (
                isinstance(review_semantic, Mapping)
                and isinstance(review_semantic.get("value"), str)
                and review_semantic["value"]
                and isinstance(review_semantic.get("method"), str)
                and review_semantic["method"]
            ):
                errors.append("proof-bearing agent review lacks a semantic answer")
        if not isinstance(review.get("review_required"), bool):
            errors.append("agent review lacks boolean review_required")
        elif record.get("review_required") is not review["review_required"]:
            errors.append("record and agent review disagree on review_required")
    review_required = record.get("review_required")
    if not isinstance(review_required, bool):
        errors.append("review_required must be boolean")
    elif severity in {"S0", "S1"} and review_required:
        errors.append("S0-S1 records cannot require adjudication")
    elif severity in {"S3", "S4"} and not review_required:
        errors.append("S3-S4 records must require adjudication")
    elif (
        isinstance(axes, Mapping)
        and axes.get("benchmark_alignment") == "semantic_conflict"
        and not review_required
    ):
        errors.append("semantic conflicts must require adjudication")
    return errors


def _summary(
    records: Sequence[Mapping[str, Any]], sources: object, review_sources: object, output: Path
) -> dict[str, Any]:
    def count(path: Sequence[str]) -> dict[str, int]:
        values: Counter[str] = Counter()
        for record in records:
            value: Any = record
            for key in path:
                value = value[key]
            values[str(value)] += 1
        return dict(sorted(values.items()))

    tags = Counter(tag for record in records for tag in record["issue_tags"])
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": AUDIT_KIND,
        "screening_statement": DISCLAIMER,
        "adjudication_status": ADJUDICATION_STATUS,
        "record_count": len(records),
        "scope": {
            "automated_screened": len(records),
            "agent_text_audited": sum(bool(record["agent_reviews"]) for record in records),
            "pdf_visual_audited": 0,
            "human_adjudicated": 0,
        },
        "output": {"path": _display_path(output), "sha256": _sha256(output)},
        "sources": sources,
        "review_sources": review_sources,
        "counts": {
            "surface_integrity": count(("axes", "surface_integrity")),
            "semantic_determinacy": count(("axes", "semantic_determinacy")),
            "benchmark_alignment": count(("axes", "benchmark_alignment")),
            "severity": count(("severity",)),
            "screening_classification": count(("screening_classification",)),
            "review_required": count(("review_required",)),
            "issue_tags": dict(sorted(tags.items())),
        },
    }


def _read_reviews(
    paths: Sequence[str | Path], expected_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    reviews: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).resolve()
        if path.suffix == ".json":
            shard = json.loads(path.read_text(encoding="utf-8"))
            shard_records = _expand_review_shard(shard)
        else:
            shard_records = list(_read_jsonl(path))
        source = _source(path)
        source["record_count"] = len(shard_records)
        source["override_count"] = sum(record.get("severity") != "S0" for record in shard_records)
        sources.append(source)
        for record in shard_records:
            instance_id = _required_id(record, f"review shard {path}")
            if instance_id not in expected_ids:
                raise ValueError(f"review shard contains unknown instance_id: {instance_id}")
            if instance_id in reviews:
                raise ValueError(f"duplicate review override: {instance_id}")
            severity = record.get("severity")
            if severity not in SEVERITIES:
                raise ValueError(f"invalid review severity for {instance_id}: {severity}")
            reviews[instance_id] = record
    return reviews, sources


def _expand_review_shard(shard: object) -> list[dict[str, Any]]:
    if not isinstance(shard, Mapping):
        raise ValueError("compact review shard must be an object")
    first = shard.get("first_instance_id")
    last = shard.get("last_instance_id")
    reviewer = shard.get("reviewer")
    if not isinstance(first, str) or not isinstance(last, str) or not isinstance(reviewer, str):
        raise ValueError("compact review shard is missing range or reviewer")
    start = int(first.removeprefix("task_"))
    stop = int(last.removeprefix("task_"))
    record_count = stop - start + 1
    if record_count <= 0 or shard.get("expected_record_count") != record_count:
        raise ValueError(
            "compact review shard range does not match expected_record_count: "
            f"range={record_count}, expected={shard.get('expected_record_count')}"
        )
    if shard.get("pdf_visual_review_performed") is not False:
        raise ValueError(
            "review shard must explicitly state that PDF visual review was not performed"
        )
    if "override_groups" in shard:
        return _expand_exact_review_shard(shard, start, stop, reviewer)
    overrides = shard.get("severity_overrides")
    if not isinstance(overrides, Mapping):
        raise ValueError("compact review shard is missing severity_overrides")
    severity_by_id: dict[str, str] = {}
    for severity, values in overrides.items():
        if severity not in SEVERITIES - {"S0"} or not isinstance(values, list):
            raise ValueError(f"invalid compact review severity bucket: {severity}")
        for instance_id in values:
            if not isinstance(instance_id, str) or instance_id in severity_by_id:
                raise ValueError(f"invalid or duplicate compact review id: {instance_id}")
            severity_by_id[instance_id] = severity
    expected_override_count = shard.get("expected_override_count")
    if expected_override_count != len(severity_by_id):
        raise ValueError(
            f"compact review override count {len(severity_by_id)} != {expected_override_count}"
        )
    decisions = shard.get("decision_defaults")
    if not isinstance(decisions, Mapping) or set(decisions) != SEVERITIES:
        raise ValueError("compact review shard must define S0-S4 decision_defaults")
    records: list[dict[str, Any]] = []
    for number in range(start, stop + 1):
        instance_id = f"task_{number:06d}"
        severity = severity_by_id.get(instance_id, "S0")
        decision = decisions[severity]
        if not isinstance(decision, Mapping):
            raise ValueError(f"invalid compact review decision for {severity}")
        record = dict(decision)
        record.update(
            {
                "instance_id": instance_id,
                "reviewer": reviewer,
                "review_kind": "agent_text_audit_no_pdf_visual_review",
                "severity": severity,
                "source_note": shard.get("source_note"),
            }
        )
        records.append(record)
    outside = sorted(set(severity_by_id) - {record["instance_id"] for record in records})
    if outside:
        raise ValueError(f"compact review overrides outside shard range: {outside}")
    return records


def _expand_exact_review_shard(
    shard: Mapping[str, Any], start: int, stop: int, reviewer: str
) -> list[dict[str, Any]]:
    default = shard.get("default_decision")
    groups = shard.get("override_groups")
    if not isinstance(default, Mapping) or not isinstance(groups, list):
        raise ValueError("exact review shard requires default_decision and override_groups")
    decision_by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("instance_ids"), list):
            raise ValueError("invalid exact review override group")
        decision = {key: value for key, value in group.items() if key != "instance_ids"}
        _validate_review_decision(decision, allow_clean=False)
        for instance_id in group["instance_ids"]:
            if not isinstance(instance_id, str) or instance_id in decision_by_id:
                raise ValueError(f"invalid or duplicate exact review id: {instance_id}")
            decision_by_id[instance_id] = dict(decision)
    expected_count = shard.get("expected_override_count")
    if len(decision_by_id) != expected_count:
        raise ValueError(f"exact review override count {len(decision_by_id)} != {expected_count}")
    _validate_review_decision(default, allow_clean=True)
    records: list[dict[str, Any]] = []
    for number in range(start, stop + 1):
        instance_id = f"task_{number:06d}"
        record = dict(decision_by_id.get(instance_id, default))
        record["issue_tags"] = [
            _TAG_ALIASES.get(str(tag), str(tag)) for tag in record["issue_tags"]
        ]
        record.update(
            {
                "instance_id": instance_id,
                "reviewer": reviewer,
                "review_kind": "agent_text_audit_no_pdf_visual_review",
                "source_note": shard.get("source_note"),
            }
        )
        records.append(record)
    outside = sorted(set(decision_by_id) - {record["instance_id"] for record in records})
    if outside:
        raise ValueError(f"exact review overrides outside shard range: {outside}")
    return records


def _validate_review_decision(decision: Mapping[str, Any], *, allow_clean: bool) -> None:
    required = {
        "surface_integrity",
        "semantic_determinacy",
        "benchmark_alignment",
        "issue_tags",
        "severity",
        "review_required",
        "rationale",
        "confidence",
    }
    if not required <= set(decision):
        raise ValueError(f"review decision missing fields: {sorted(required - set(decision))}")
    severity = decision["severity"]
    tags = decision["issue_tags"]
    if severity not in SEVERITIES or not isinstance(tags, list):
        raise ValueError("review decision has invalid severity or tags")
    normalized_tags = {_TAG_ALIASES.get(str(tag), str(tag)) for tag in tags}
    if not normalized_tags <= ISSUE_TAGS:
        raise ValueError(
            f"review decision has unknown tags: {sorted(normalized_tags - ISSUE_TAGS)}"
        )
    if allow_clean and (severity != "S0" or normalized_tags != {"clean"}):
        raise ValueError("default review decision must be S0 clean")
    if not allow_clean and (severity == "S0" or not normalized_tags or "clean" in normalized_tags):
        raise ValueError("override review decision must be S1-S4 with non-clean tags")
    confidence = decision["confidence"]
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("review decision confidence must be between zero and one")
    if not isinstance(decision["rationale"], str) or not decision["rationale"]:
        raise ValueError("review decision rationale must be nonempty")
    alignment = decision["benchmark_alignment"]
    if alignment in {"aligned", "normalized_equivalent", "semantic_conflict"}:
        semantic = decision.get("semantic_answer")
        if not (
            isinstance(semantic, Mapping)
            and isinstance(semantic.get("value"), str)
            and semantic["value"]
            and isinstance(semantic.get("method"), str)
            and semantic["method"]
        ):
            raise ValueError(
                "proof-bearing review alignment requires semantic_answer value and method"
            )


def _record_sequence(value: Mapping[str, Any], kind: str) -> Sequence[Mapping[str, Any]]:
    records = value.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError(f"{kind} records must be a sequence")
    return records


def _auto_severity(tags: set[str], alignment: str) -> str:
    if alignment == "semantic_conflict":
        return "S4"
    if tags & {"rate_unit_relation_corruption", "target_or_subquestion_drift"}:
        return "S3"
    if tags & {"comparison_polarity_or_sign", "condition_insertion_or_deletion"}:
        return "S3"
    if tags:
        return "S2"
    return "S0"


def _risk(severity: str) -> str:
    return {"S0": "none", "S1": "low", "S2": "medium", "S3": "high", "S4": "high"}[severity]


def _template(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(_NUMBER, "#", text.lower())).strip()


def _answers_equal(left: str, right: str) -> bool:
    try:
        return Decimal(left.replace(",", "")) == Decimal(right.replace(",", ""))
    except InvalidOperation:
        return " ".join(left.lower().split()) == " ".join(right.lower().split())


def _alignment_for(semantic_value: object, benchmark: str) -> str:
    if not isinstance(semantic_value, str) or not semantic_value:
        return "label_unverifiable"
    if _answers_equal(semantic_value, benchmark):
        return "aligned" if semantic_value == benchmark else "normalized_equivalent"
    return "semantic_conflict"


def _source(path: Path) -> dict[str, Any]:
    return {"path": _display_path(path), "sha256": _sha256(path)}


def _display_path(path: Path) -> str:
    with suppress(ValueError):
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    return str(path.resolve())


def _resolve_reported_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _validate_source_report(name: str, source: object) -> list[str]:
    if not isinstance(source, Mapping):
        return [f"summary source {name} is invalid"]
    path_value = source.get("path")
    digest = source.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or not _SHA256.fullmatch(
        digest
    ):
        return [f"summary source {name} provenance is invalid"]
    path = _resolve_reported_path(path_value)
    if not path.is_file():
        return [f"summary source {name} does not exist: {path_value}"]
    if _sha256(path) != digest:
        return [f"summary source {name} SHA-256 mismatch"]
    return []


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"invalid SHA-256 for {context}")
    return value


def _read_unique(path: Path, kind: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line, record in enumerate(_read_jsonl(path), 1):
        instance_id = _required_id(record, f"{kind} line {line}")
        if instance_id in result:
            raise ValueError(f"duplicate {kind} instance_id: {instance_id}")
        result[instance_id] = record
    return result


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number} in {path}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} in {path} is not an object")
            yield record


def _required_id(record: Mapping[str, Any], context: str) -> str:
    value = record.get("instance_id")
    if not isinstance(value, str) or not re.fullmatch(r"task_\d{6}", value):
        raise ValueError(f"invalid instance_id in {context}")
    return value


def _require_coverage(name: str, expected: set[str], actual: set[str]) -> None:
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} coverage mismatch: missing={missing}, extra={extra}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    _atomic_write(path, content)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
