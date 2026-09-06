"""Paired OCR comparison against frozen, image-only consensus silver.

These are transcription metrics, never organizer answer or Joint Accuracy scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docinsights_analysis.solution_records import write_private_atomic

NUMBER = re.compile(
    r"(?<![\w.])[+−-]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][+−-]?\d+)?"
    r"(?:/[+−-]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][+−-]?\d+)?)?%?(?!\w)"
)
HEADING = re.compile(r"^\s*(\S+?):(?=\s|$)")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def token_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    ref, hyp = Counter(map(normalize, reference)), Counter(map(normalize, hypothesis))
    return {
        "tp": sum((ref & hyp).values()),
        "fp": sum((hyp - ref).values()),
        "fn": sum((ref - hyp).values()),
    }


def prf(counts: dict[str, int]) -> dict[str, float | int | None]:
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        **counts,
        "reference_tokens": tp + fn,
        "predicted_tokens": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else (0.0 if fn else None),
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def extract_literals(lines: list[str]) -> tuple[list[str], list[str]]:
    identifiers, numbers = [], []
    for line in lines:
        line = normalize(line)
        match = HEADING.match(line)
        if match:
            identifiers.append(match.group(1))
            line = line[match.end() :]
        numbers.extend(NUMBER.findall(line))
    return identifiers, numbers


def region_lines(lines: list[dict[str, Any]], bounds: dict[str, int]) -> tuple[list[str], int]:
    """Use geometry alone; never select the text giving the best reference match."""
    included, excluded = [], 0
    for index, line in enumerate(lines):
        x0, y0, x1, y1 = line["bbox"]
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)) or x1 < x0 or y1 < y0:
            raise ValueError("invalid OCR geometry")
        if y1 <= bounds["top"] or y0 >= bounds["bottom"]:
            continue
        if y0 < bounds["top"] or y1 > bounds["bottom"]:
            excluded += 1
            continue
        included.append((y0, x0, index, normalize(line["text"])))
    included.sort()
    return [item[3] for item in included], excluded


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_contract(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != (
        "issue24-ocr-benchmark-v1"
    ):
        raise ValueError("benchmark manifest schema is invalid")
    pages = manifest.get("pages")
    if manifest.get("page_count") != 60 or not isinstance(pages, list) or len(pages) != 60:
        raise ValueError("benchmark manifest must contain exactly 60 pages")
    page_ids = [page.get("benchmark_page_id") for page in pages if isinstance(page, dict)]
    if (
        len(page_ids) != len(pages)
        or any(not isinstance(page_id, str) or not page_id for page_id in page_ids)
        or len(set(page_ids)) != len(page_ids)
    ):
        raise ValueError("benchmark manifest contains a duplicate or invalid page ID")
    return pages


def validate_candidate_contract(
    engine: Any, manifest_hash: str, manifest_pages: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if not isinstance(engine, dict) or engine.get("schema_version") != (
        "issue24-ocr-candidate-v1"
    ):
        raise ValueError("candidate schema is invalid")
    manifest_binding = engine.get("benchmark_manifest")
    if (
        not isinstance(manifest_binding, dict)
        or manifest_binding.get("sha256") != manifest_hash
    ):
        raise ValueError("engine corpus mismatch")
    config = engine.get("config")
    if not isinstance(config, dict) or canonical_hash(config) != engine.get("config_sha256"):
        raise ValueError("candidate config hash mismatch")
    raw_pages = engine.get("pages")
    if not isinstance(raw_pages, list) or any(not isinstance(page, dict) for page in raw_pages):
        raise ValueError("candidate pages are invalid")
    pages = {page.get("benchmark_page_id"): page for page in raw_pages}
    if len(pages) != len(raw_pages) or any(not isinstance(page_id, str) for page_id in pages):
        raise ValueError("duplicate or invalid engine page")
    manifest_by_id = {page["benchmark_page_id"]: page for page in manifest_pages}
    if set(pages) - set(manifest_by_id):
        raise ValueError("engine includes a page outside the frozen corpus")
    for page_id, observed in pages.items():
        expected = manifest_by_id[page_id]
        expected_hash = expected["rendered_image_sha256"]
        if observed.get("input_image_sha256") != expected_hash:
            raise ValueError("engine input image hash mismatch")
        try:
            input_path = Path(observed["input_image"])
            input_hash = sha256(input_path)
        except (KeyError, OSError, TypeError) as error:
            raise ValueError("candidate input image cannot be verified") from error
        if input_hash != expected_hash:
            raise ValueError("candidate input image content mismatch")
        if observed.get("status") not in {"succeeded", "failed"}:
            raise ValueError("candidate page status is invalid")
    return pages


def sum_counts(rows: list[dict[str, Any]], metric: str) -> dict[str, int]:
    return {key: sum(row[metric][key] for row in rows) for key in ("tp", "fp", "fn")}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pages": len(rows),
        "failed_pages": sum(row["failed"] for row in rows),
        "body_eligible_regions": sum(row["body_regions"] for row in rows),
        "id_eligible_regions": sum(row["id_regions"] for row in rows),
        "numeric_eligible_regions": sum(row["numeric_regions"] for row in rows),
        "candidate_boundary_exclusions": sum(row["boundary_exclusions"] for row in rows),
    }
    for metric in ("ids", "numbers"):
        result[metric] = prf(sum_counts(rows, metric))
        for name in ("precision", "recall", "f1"):
            values = [prf(row[metric])[name] for row in rows]
            eligible = [value for value in values if value is not None]
            result[metric][f"page_macro_{name}"] = statistics.mean(eligible) if eligible else None
        result[metric]["pages_with_reference_tokens"] = sum(
            row[metric]["tp"] + row[metric]["fn"] > 0 for row in rows
        )
    for name, errors, total in (
        ("cer", "character_errors", "reference_characters"),
        ("wer", "word_errors", "reference_words"),
    ):
        denominator = sum(row[total] for row in rows)
        result[name] = sum(row[errors] for row in rows) / denominator if denominator else None
        result[errors] = sum(row[errors] for row in rows)
        result[total] = denominator
        page_rates = [row[errors] / row[total] for row in rows if row[total]]
        result[f"{name}_page_mean"] = statistics.mean(page_rates) if page_rates else None
        result[f"{name}_page_median"] = statistics.median(page_rates) if page_rates else None
    elapsed = sorted(
        row["elapsed_seconds"]
        for row in rows
        if row["elapsed_seconds"] is not None and not row["failed"]
    )
    result["median_success_seconds"] = statistics.median(elapsed) if elapsed else None
    result["p95_success_seconds"] = (
        elapsed[max(0, math.ceil(len(elapsed) * 0.95) - 1)] if elapsed else None
    )
    failed_elapsed = [
        row["elapsed_seconds"]
        for row in rows
        if row["failed"] and row["elapsed_seconds"] is not None
    ]
    result["failed_attempt_seconds"] = sum(failed_elapsed)
    result["failed_pages_with_timing"] = len(failed_elapsed)
    return result


def score_page(
    page: dict[str, Any], silver: dict[str, Any], observed: dict[str, Any] | None
) -> dict[str, Any]:
    failed = observed is None or observed.get("status") not in ("succeeded", "success", "ok")
    row: dict[str, Any] = {
        "benchmark_page_id": page["benchmark_page_id"],
        "split": page["split"],
        "instance_id": page["instance_id"],
        "failed": failed,
        "elapsed_seconds": observed.get("elapsed_seconds") if observed else None,
        "ids": {"tp": 0, "fp": 0, "fn": 0},
        "numbers": {"tp": 0, "fp": 0, "fn": 0},
        "body_regions": 0,
        "id_regions": 0,
        "numeric_regions": 0,
        "character_errors": 0,
        "reference_characters": 0,
        "word_errors": 0,
        "reference_words": 0,
        "boundary_exclusions": 0,
    }
    for region in silver["regions"]:
        lines, excluded = region_lines(
            observed["lines"] if not failed and observed else [], region["bounds"]
        )
        row["boundary_exclusions"] += excluded
        identifiers, numbers = extract_literals(lines)
        consensus = region["consensus"]
        for key, metric, prediction, count_key in (
            ("visible_ids", "ids", identifiers, "id_regions"),
            ("numeric_literals", "numbers", numbers, "numeric_regions"),
        ):
            if consensus[key]["eligible"]:
                row[count_key] += 1
                counts = token_counts(consensus[key]["value"], prediction)
                for name, value in counts.items():
                    row[metric][name] += value
        if consensus["body"]["eligible"]:
            reference, hypothesis = (
                normalize(consensus["body"]["value"]),
                normalize(" ".join(lines)),
            )
            row["body_regions"] += 1
            row["character_errors"] += edit_distance(reference, hypothesis)
            row["reference_characters"] += len(reference)
            row["word_errors"] += edit_distance(reference.split(), hypothesis.split())
            row["reference_words"] += len(reference.split())
    return row


def decision_values(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    summary = summarize(rows)
    return {
        "ids": summary["ids"]["f1"],
        "numbers": summary["numbers"]["f1"],
        "cer": summary["cer"],
        "wer": summary["wer"],
        "failure_rate": summary["failed_pages"] / len(rows) if rows else None,
        "median_success_seconds": summary["median_success_seconds"],
    }


def paired_intervals(
    left: list[dict[str, Any]], right: list[dict[str, Any]], repetitions: int = 1000
) -> dict[str, Any]:
    """Resample document clusters within split, preserving paired engine pages."""
    if [r["benchmark_page_id"] for r in left] != [r["benchmark_page_id"] for r in right]:
        raise ValueError("bootstrap requires identical paired pages")
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(left):
        groups[row["split"]][row["instance_id"]].append(index)
    rng = random.Random(240907)
    samples: dict[str, list[float]] = {key: [] for key in decision_values(left)}
    for _ in range(repetitions):
        indices = []
        for docs in groups.values():
            keys = list(docs)
            for chosen in rng.choices(keys, k=len(keys)):
                indices.extend(docs[chosen])
        left_values = decision_values([left[i] for i in indices])
        right_values = decision_values([right[i] for i in indices])
        for metric in samples:
            a, b = left_values[metric], right_values[metric]
            if a is not None and b is not None:
                samples[metric].append(a - b)
    output = {}
    for metric, values in samples.items():
        values.sort()
        output[metric] = {
            "valid_resamples": len(values),
            "difference_95pct_interval": [
                values[int((len(values) - 1) * q)] for q in (0.025, 0.975)
            ]
            if values
            else None,
        }
    return output


def provisional_order(all_rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Point-estimate ordering; intervals still govern claims of superiority."""

    def key(name: str) -> tuple[float, ...]:
        values = decision_values(all_rows[name])
        return tuple(
            float("inf") if value is None else (-value if metric in ("ids", "numbers") else value)
            for metric, value in values.items()
        )

    return sorted(all_rows, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--engine", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_pages = validate_manifest_contract(manifest)
    manifest_hash = sha256(args.manifest)
    references = {}
    reference_hashes = {}
    for page in manifest_pages:
        path = args.references / page["benchmark_page_id"] / "consensus.json"
        reference = json.loads(path.read_text())
        if reference.get("schema_version") != "consensus-silver-v3":
            raise ValueError("scoring requires corrected consensus-silver-v3 references")
        if (
            reference["manifest_sha256"] != manifest_hash
            or reference["benchmark_page_id"] != page["benchmark_page_id"]
        ):
            raise ValueError("reference does not bind the frozen corpus")
        if reference["source"]["page_image_sha256"] != page["rendered_image_sha256"]:
            raise ValueError("reference image hash mismatch")
        if sha256(Path(page["rendered_image"])) != page["rendered_image_sha256"]:
            raise ValueError("frozen source image changed")
        expected_bounds = [
            {
                "left": 0,
                "top": page["height"] * k // 3,
                "right": page["width"],
                "bottom": page["height"] * (k + 1) // 3,
            }
            for k in range(3)
        ]
        if [region["bounds"] for region in reference["regions"]] != expected_bounds:
            raise ValueError("reference does not use fixed scoring cores")
        references[page["benchmark_page_id"]] = reference
        reference_hashes[page["benchmark_page_id"]] = sha256(path)
    report: dict[str, Any] = {
        "schema_version": "ocr-consensus-silver-comparison-v1",
        "reference_type": "two independent image-only model transcriptions; not human gold",
        "manifest_sha256": manifest_hash,
        "scorer_sha256": sha256(Path(__file__)),
        "normalization": "NFC and whitespace collapse only",
        "literal_parser": {
            "heading": HEADING.pattern,
            "number": NUMBER.pattern,
            "identifier_scope": (
                "lexical leading-heading tokens, an Evidence-ID transcription proxy; "
                "colon labels can also qualify, so this is not semantic gold block identification"
            ),
        },
        "reference_sha256": reference_hashes,
        "reference_regions": sum(len(value["regions"]) for value in references.values()),
        "reference_failed_pages": sum(
            value["status"] != "complete" for value in references.values()
        ),
        "reference_uncertain_regions": sum(
            any(region["disagreements"]["uncertain"])
            for value in references.values()
            for region in value["regions"]
        ),
        "selection_order": [
            "heading-token F1",
            "numeric F1",
            "CER",
            "WER",
            "failure rate",
            "runtime",
        ],
        "selection_scope": "heldout; train is a separately reported transfer check",
        "engines": {},
        "paired_document_bootstrap": {},
        "heldout_paired_document_bootstrap": {},
    }
    all_rows = {}
    for path in args.engine:
        engine = json.loads(path.read_text())
        pages = validate_candidate_contract(engine, manifest_hash, manifest_pages)
        name = engine["engine"]["name"]
        if name in all_rows:
            raise ValueError("duplicate engine name")
        rows = [
            score_page(
                page, references[page["benchmark_page_id"]], pages.get(page["benchmark_page_id"])
            )
            for page in manifest_pages
        ]
        all_rows[name] = rows
        report["engines"][name] = {
            "result_sha256": sha256(path),
            "config_sha256": engine["config_sha256"],
            "initialization_seconds": engine["initialization_seconds"],
            "summary": summarize(rows),
            "by_split": {
                split: summarize([row for row in rows if row["split"] == split])
                for split in ("heldout", "train")
            },
            "pages": rows,
        }
    names = list(all_rows)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            report["paired_document_bootstrap"][f"{left} minus {right}"] = paired_intervals(
                all_rows[left], all_rows[right]
            )
            report["heldout_paired_document_bootstrap"][f"{left} minus {right}"] = paired_intervals(
                [row for row in all_rows[left] if row["split"] == "heldout"],
                [row for row in all_rows[right] if row["split"] == "heldout"],
            )
    report["selection"] = {
        "point_estimate_order": provisional_order(
            {
                name: [row for row in rows if row["split"] == "heldout"]
                for name, rows in all_rows.items()
            }
        ),
        "status": "provisional; inspect paired intervals and eligible support before selecting",
        "uncertainty_rule": (
            "An interval spanning zero does not prove equality or superiority; "
            "expand evidence or report an unresolved comparison."
        ),
    }
    write_private_atomic(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({name: value["summary"] for name, value in report["engines"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
