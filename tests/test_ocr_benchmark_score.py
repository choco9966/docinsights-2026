"""Small deterministic checks for source-transcription benchmark metrics."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "score_ocr_benchmark", Path(__file__).parents[1] / "scripts/score_ocr_benchmark.py"
)
assert SPEC and SPEC.loader
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def test_normalization_does_not_erase_critical_characters():
    assert score.normalize("  A:9\n −1.20  1/2 ") == "A:9 −1.20 1/2"
    assert score.normalize("e\u0301") == "é"
    assert score.normalize("O") != score.normalize("0")


def test_multisets_preserve_repeated_occurrences():
    counts = score.token_counts(["7", "7", "8"], ["7", "8", "8", "9"])
    assert counts == {"tp": 2, "fp": 2, "fn": 1}
    assert score.prf(counts)["f1"] == pytest.approx(4 / 7)


def test_missing_engine_page_is_penalized():
    assert score.token_counts(["7", "b12"], []) == {"tp": 0, "fp": 0, "fn": 2}
    assert score.edit_distance("seven", "") == 5


def test_levenshtein_counts_insertion_deletion_and_substitution():
    assert score.edit_distance("kitten", "sitting") == 3
    assert score.edit_distance(["one", "two"], ["one", "three", "four"]) == 2


def test_geometry_assignment_excludes_boundary_crossings():
    lines = [
        {"text": "inside", "bbox": [0, 10, 50, 20]},
        {"text": "crossing", "bbox": [0, 90, 50, 110]},
        {"text": "later", "bbox": [0, 130, 50, 140]},
    ]
    result, excluded = score.region_lines(lines, {"left": 0, "top": 0, "right": 100, "bottom": 100})
    assert result == ["inside"]
    assert excluded == 1


def test_literals_exclude_heading_identifier_numbers():
    ids, numbers = score.extract_literals(["A:9: There are 12.50 and −3, plus 1/2 and 25%."])
    assert ids == ["A:9"]
    assert numbers == ["12.50", "−3", "1/2", "25%"]


def sample_page(body="b1: 5", observed=True):
    page = {"benchmark_page_id": "p1", "split": "heldout", "instance_id": "doc1"}
    silver = {
        "regions": [
            {
                "bounds": {"left": 0, "top": 0, "right": 100, "bottom": 100},
                "consensus": {
                    "body": {"eligible": True, "value": body},
                    "visible_ids": {"eligible": True, "value": ["b1"]},
                    "numeric_literals": {"eligible": True, "value": ["5"]},
                },
            }
        ]
    }
    result = {
        "status": "succeeded",
        "elapsed_seconds": 2.0,
        "lines": [{"text": "b1: 5", "bbox": [0, 10, 50, 20]}],
    }
    return score.score_page(page, silver, result if observed else None)


def test_empty_reference_still_counts_inserted_text():
    row = sample_page(body="")
    assert row["body_regions"] == 1
    assert row["character_errors"] == len("b1: 5")
    assert row["reference_characters"] == 0


def test_bootstrap_includes_text_failures_and_runtime_uncertainty():
    intervals = score.paired_intervals([sample_page()], [sample_page(observed=False)], 10)
    assert intervals["ids"]["difference_95pct_interval"] == [1.0, 1.0]
    assert intervals["cer"]["difference_95pct_interval"] == [-1.0, -1.0]
    assert intervals["failure_rate"]["difference_95pct_interval"] == [-1.0, -1.0]
    assert intervals["median_success_seconds"]["valid_resamples"] == 0


def test_failure_latency_is_reported_separately():
    failed = sample_page(observed=False)
    failed["elapsed_seconds"] = 300.0
    summary = score.summarize([sample_page(), failed])
    assert summary["median_success_seconds"] == 2.0
    assert summary["failed_attempt_seconds"] == 300.0


def test_numeric_literals_preserve_leading_decimals_signs_and_exponents():
    _, numbers = score.extract_literals(["Values .5, -.5, −.25, 1e3, +2.5E-4, 1/.5, and 50%."])
    assert numbers == [".5", "-.5", "−.25", "1e3", "+2.5E-4", "1/.5", "50%"]
    assert score.token_counts([".5", "-.5", "1e3"], ["5", "5", "1"])["tp"] == 0


def test_manifest_contract_requires_v1_sixty_unique_pages():
    pages = [{"benchmark_page_id": f"page-{index}"} for index in range(60)]
    manifest = {
        "schema_version": "issue24-ocr-benchmark-v1",
        "page_count": 60,
        "pages": pages,
    }
    assert score.validate_manifest_contract(manifest) == pages

    for changed, message in (
        ({**manifest, "schema_version": "future-v2"}, "schema"),
        ({**manifest, "page_count": 59}, "60 pages"),
        ({**manifest, "pages": [*pages[:-1], pages[0]]}, "duplicate"),
    ):
        with pytest.raises(ValueError, match=message):
            score.validate_manifest_contract(changed)


def test_candidate_contract_binds_schema_config_and_input_bytes(tmp_path: Path):
    image = tmp_path / "page.png"
    image.write_bytes(b"frozen pixels")
    image_sha = score.sha256(image)
    manifest_page = {
        "benchmark_page_id": "page-1",
        "rendered_image": str(image),
        "rendered_image_sha256": image_sha,
    }
    config = {"mode": "accurate", "threads": 1}
    assert score.canonical_hash(config) == (
        "1488f66898a995c74d80c478ecc0e4858cc6522bc735d4779214789ef107ef0e"
    )
    candidate = {
        "schema_version": "issue24-ocr-candidate-v1",
        "benchmark_manifest": {"sha256": "a" * 64},
        "config": config,
        "config_sha256": score.canonical_hash(config),
        "pages": [
            {
                "benchmark_page_id": "page-1",
                "status": "succeeded",
                "input_image": str(image),
                "input_image_sha256": image_sha,
            }
        ],
    }
    assert score.validate_candidate_contract(candidate, "a" * 64, [manifest_page]) == {
        "page-1": candidate["pages"][0]
    }

    wrong_schema = {**candidate, "schema_version": "future-v2"}
    with pytest.raises(ValueError, match="candidate schema"):
        score.validate_candidate_contract(wrong_schema, "a" * 64, [manifest_page])

    changed_config = json.loads(json.dumps(candidate))
    changed_config["config"]["threads"] = 2
    with pytest.raises(ValueError, match="config hash"):
        score.validate_candidate_contract(changed_config, "a" * 64, [manifest_page])

    changed_image = tmp_path / "changed.png"
    changed_image.write_bytes(b"different pixels")
    mismatched_input = json.loads(json.dumps(candidate))
    mismatched_input["pages"][0]["input_image"] = str(changed_image)
    with pytest.raises(ValueError, match="input image content"):
        score.validate_candidate_contract(mismatched_input, "a" * 64, [manifest_page])
