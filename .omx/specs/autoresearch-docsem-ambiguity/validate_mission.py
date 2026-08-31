"""Mission validator for the Issue 14 training ambiguity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from docinsights_ambiguity.audit import (
    _alignment_for,
    validate_artifacts,
    validate_schema_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _unique_map(
    rows: list[dict[str, Any]], kind: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    ids = [row.get("instance_id") for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if len(rows) != 908:
        errors.append(f"{kind} row count {len(rows)} != 908")
    if duplicates:
        errors.append(f"{kind} duplicate instance_ids: {duplicates}")
    return {
        instance_id: row
        for instance_id, row in zip(ids, rows, strict=True)
        if isinstance(instance_id, str)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query-comparison", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    validation = validate_artifacts(
        args.tasks, args.output, args.summary, expected_count=908
    )
    errors.extend(validation["errors"])
    blind = _rows(args.blind)
    merged = _rows(args.output)
    labels = _unique_map(_rows(args.labels), "labels", errors)
    references = _unique_map(_rows(args.reference), "reference", errors)
    comparisons = _unique_map(_rows(args.query_comparison), "query comparison", errors)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    errors.extend(validate_schema_contract(schema))
    state = json.loads(args.state.read_text(encoding="utf-8"))

    if state.get("status") != "complete":
        errors.append("autoresearch state is not complete")
    expected_state_paths = {
        "mission_path": ".omx/specs/autoresearch-docsem-ambiguity/mission.md",
        "sandbox_path": ".omx/specs/autoresearch-docsem-ambiguity/sandbox.md",
        "completion_artifact_path": ".omx/specs/autoresearch-docsem-ambiguity/result.json",
        "output_artifact_path": "artifacts/ambiguity/train-ambiguity-tags.jsonl",
    }
    for key, expected in expected_state_paths.items():
        if state.get(key) != expected:
            errors.append(f"autoresearch state path mismatch for {key}")

    if len(blind) != 908 or len({row["instance_id"] for row in blind}) != 908:
        errors.append("blind layer is not 908 unique rows")
    blind_ids = {row["instance_id"] for row in blind}
    if blind_ids != {row["instance_id"] for row in merged}:
        errors.append("blind and merged ID sets differ")
    if blind_ids != set(references):
        errors.append("reference and blind ID sets differ")
    if blind_ids != set(comparisons):
        errors.append("query comparison and blind ID sets differ")
    if blind_ids != set(labels):
        errors.append("labels and blind ID sets differ")
    if any("benchmark_answer" in row for row in blind):
        errors.append("blind layer contains benchmark answers")
    blind_by_id = {row["instance_id"]: row for row in blind}
    for row in merged:
        instance_id = row["instance_id"]
        if row.get("blind_question_screen") != blind_by_id.get(instance_id):
            errors.append(f"embedded blind decision differs from persisted blind row: {instance_id}")
    severity = Counter(row["severity"] for row in merged)
    if severity != Counter({"S0": 635, "S1": 67, "S2": 56, "S3": 56, "S4": 94}):
        errors.append(f"unexpected severity counts: {dict(severity)}")
    if sum(value for key, value in severity.items() if key != "S0") != 273:
        errors.append("review override count is not 273")
    flagged_rationales = {
        row["agent_reviews"][0].get("rationale")
        for row in merged
        if row["severity"] != "S0" and row["agent_reviews"]
    }
    if len(flagged_rationales) < 40:
        errors.append(
            "review overrides appear severity-templated; canonical per-case rationales are missing"
        )
    flagged_review_tags = {
        tag
        for row in merged
        if row["severity"] != "S0"
        for tag in row["agent_reviews"][0].get("issue_tags", [])
    }
    if len(flagged_review_tags) < 10:
        errors.append(
            "review overrides do not preserve the canonical per-case issue-tag taxonomy"
        )
    for row in merged:
        tags = row["issue_tags"]
        if row["severity"] == "S0" and tags != ["clean"]:
            errors.append(f"S0 clean-tag contract failed for {row['instance_id']}")
        if row["severity"] != "S0" and (not tags or "clean" in tags):
            errors.append(f"flagged tag contract failed for {row['instance_id']}")
        reviews = row["agent_reviews"]
        if len(reviews) != 1:
            errors.append(f"review provenance count failed for {row['instance_id']}")
            continue
        review = reviews[0]
        required = {
            "surface_integrity",
            "semantic_determinacy",
            "benchmark_alignment",
            "issue_tags",
            "severity",
            "review_required",
            "rationale",
            "confidence",
            "reviewer",
        }
        if not required <= set(review):
            errors.append(f"review fields missing for {row['instance_id']}")
        if not isinstance(review.get("rationale"), str) or not review["rationale"]:
            errors.append(f"review rationale missing for {row['instance_id']}")
        confidence = review.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"review confidence invalid for {row['instance_id']}")
        if not isinstance(review.get("review_required"), bool):
            errors.append(f"review decision missing for {row['instance_id']}")
        elif row["review_required"] is not review["review_required"]:
            errors.append(f"review decision mismatch for {row['instance_id']}")
        if row["severity"] in {"S0", "S1"} and row["review_required"]:
            errors.append(f"low-severity review policy failed for {row['instance_id']}")
        if row["severity"] in {"S3", "S4"} and not row["review_required"]:
            errors.append(f"high-severity review policy failed for {row['instance_id']}")
        if row["axes"]["benchmark_alignment"] == "semantic_conflict" and not row[
            "review_required"
        ]:
            errors.append(f"semantic conflict review policy failed for {row['instance_id']}")
        alignment = row["axes"]["benchmark_alignment"]
        if alignment in {"aligned", "normalized_equivalent", "semantic_conflict"}:
            semantic = row.get("semantic_answer", {})
            value = semantic.get("value")
            method = semantic.get("method")
            if not isinstance(value, str) or not value or not isinstance(method, str) or not method:
                errors.append(f"semantic proof missing for {row['instance_id']}")
            elif _alignment_for(value, row["benchmark_answer"]) != alignment:
                errors.append(f"semantic proof alignment mismatch for {row['instance_id']}")
        review_alignment = review.get("benchmark_alignment")
        if review_alignment in {"aligned", "normalized_equivalent", "semantic_conflict"}:
            review_semantic = review.get("semantic_answer", {})
            if not (
                isinstance(review_semantic.get("value"), str)
                and review_semantic["value"]
                and isinstance(review_semantic.get("method"), str)
                and review_semantic["method"]
            ):
                errors.append(f"review semantic proof missing for {row['instance_id']}")
        if "reported_benchmark_alignment" in review or "alignment_downgrade_reason" in review:
            errors.append(f"unresolved alignment downgrade remains for {row['instance_id']}")
    scope = summary.get("scope", {})
    expected_scope = {
        "automated_screened": 908,
        "agent_text_audited": 908,
        "pdf_visual_audited": 0,
        "human_adjudicated": 0,
    }
    if scope != expected_scope:
        errors.append(f"unexpected audit scope: {scope}")
    review_sources = summary.get("review_sources", [])
    if [source.get("record_count") for source in review_sources] != [303, 303, 302]:
        errors.append("review shard boundaries/counts are not 303/303/302")
    if [source.get("override_count") for source in review_sources] != [86, 85, 102]:
        errors.append("review shard override counts are not 86/85/102")
    by_id = {row["instance_id"]: row for row in merged}
    for instance_id, severity_value in {
        "task_000003": "S4",
        "task_000015": "S4",
        "task_000027": "S4",
        "task_000029": "S1",
        "task_000606": "S0",
        "task_000908": "S0",
    }.items():
        if by_id[instance_id]["severity"] != severity_value:
            errors.append(f"regression severity mismatch for {instance_id}")
    if any(row["adjudication_status"].endswith("pending") is False for row in merged):
        errors.append("a row does not declare pending human adjudication")
    if any(not row["evidence"]["benchmark_blocks_present_in_reference"] for row in merged):
        errors.append("benchmark evidence block missing from reference")
    if any(not row["evidence"]["benchmark_matches_recovered_block"] for row in merged):
        errors.append("recovered evidence block differs from benchmark evidence")
    tasks_sha256 = _sha256(args.tasks)
    labels_sha256 = _sha256(args.labels)
    reference_sha256 = _sha256(args.reference)
    comparison_sha256 = _sha256(args.query_comparison)
    supplied_source_hashes = {
        "tasks": tasks_sha256,
        "labels": labels_sha256,
        "reference": reference_sha256,
        "query_comparison": comparison_sha256,
        "blind_screen": _sha256(args.blind),
    }
    summary_sources = summary.get("sources", {})
    for name, digest in supplied_source_hashes.items():
        source = summary_sources.get(name, {})
        if source.get("sha256") != digest:
            errors.append(f"summary source hash differs from supplied {name}")
    for instance_id, row in by_id.items():
        evidence = row["evidence"]
        reference = references.get(instance_id, {})
        comparison = comparisons.get(instance_id, {})
        label = labels.get(instance_id, {})
        source = comparison.get("source", {})
        provenance = reference.get("provenance", {})
        blocks = {
            block.get("block_id"): block.get("text")
            for block in reference.get("blocks", [])
            if isinstance(block, dict)
        }
        recovered_block_id = evidence["recovered_block_id"]
        recovered_text = blocks.get(recovered_block_id)
        if not isinstance(recovered_text, str):
            errors.append(f"recovered evidence text missing for {instance_id}")
        elif evidence["recovered_block_text_sha256"] != hashlib.sha256(
            recovered_text.encode()
        ).hexdigest():
            errors.append(f"recovered evidence text hash mismatch for {instance_id}")
        if evidence["recovered_query_sha256"] != blind_by_id[instance_id]["question_sha256"]:
            errors.append(f"recovered query hash mismatch for {instance_id}")
        if row["benchmark_answer"] != label.get("answer"):
            errors.append(f"benchmark answer differs from label for {instance_id}")
        if evidence["benchmark_block_ids"] != label.get("evidence"):
            errors.append(f"benchmark evidence differs from label for {instance_id}")
        if recovered_block_id != comparison.get("evidence_block_id"):
            errors.append(f"recovered block differs from comparison for {instance_id}")
        if evidence["recovered_pages"] != comparison.get("evidence_pages"):
            errors.append(f"recovered pages differ from comparison for {instance_id}")
        if evidence["tasks_manifest_file_sha256"] != tasks_sha256:
            errors.append(f"tasks manifest hash mismatch for {instance_id}")
        if evidence["codex_reference_file_sha256"] != reference_sha256:
            errors.append(f"Codex reference hash mismatch for {instance_id}")
        document_hashes = {
            evidence["document_pdf_sha256"],
            comparison.get("pdf_sha256"),
            source.get("document_pdf_sha256"),
            provenance.get("input_pdf_sha256"),
        }
        if len(document_hashes) != 1:
            errors.append(f"document PDF hash mismatch for {instance_id}")

    passed = not errors
    result = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "summary": (
            "908/908 automated screens and agent text audits validated; "
            "273 overrides, 635 clean candidates; human adjudication pending"
        ),
        "errors": errors,
        "record_count": len(merged),
        "unique_instance_count": len({row["instance_id"] for row in merged}),
        "output_artifact_path": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "blind_artifact_path": str(args.blind.resolve()),
        "blind_sha256": _sha256(args.blind),
        "summary_sha256": _sha256(args.summary),
        "schema_path": str(args.schema.resolve()),
        "schema_sha256": _sha256(args.schema),
        "schema_version": schema.get("properties", {})
        .get("schema_version", {})
        .get("const"),
        "state_path": str(args.state.resolve()),
        "state_sha256": _sha256(args.state),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
