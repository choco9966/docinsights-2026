from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

VALIDATOR = runpy.run_path(str(Path(__file__).parents[1] / "scripts/validate_issue15_research.py"))
EXPECTED_STATE = VALIDATOR["EXPECTED_STATE"]
PRIORITIES_PATH = VALIDATOR["PRIORITIES_PATH"]
REPORT_PATH = VALIDATOR["REPORT_PATH"]
STATE_PATH = VALIDATOR["STATE_PATH"]
Validation = VALIDATOR["Validation"]
build_result = VALIDATOR["build_result"]
validate_priorities = VALIDATOR["validate_priorities"]
validate_report_text = VALIDATOR["validate_report_text"]
validate_state = VALIDATOR["validate_state"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_duplicate_or_empty_metrics_fail_their_independent_check() -> None:
    priorities = load_json(PRIORITIES_PATH)
    duplicate = copy.deepcopy(priorities)
    duplicate["candidates"][0]["metrics"].append(duplicate["candidates"][0]["metrics"][0])
    empty = copy.deepcopy(priorities)
    empty["candidates"][0]["metrics"][0] = "  "

    duplicate_validation = Validation()
    validate_priorities(duplicate, duplicate_validation)
    empty_validation = Validation()
    validate_priorities(empty, empty_validation)

    assert not duplicate_validation.checks["metrics_unique_nonempty_strings"]
    assert not empty_validation.checks["metrics_unique_nonempty_strings"]
    assert duplicate_validation.checks["candidate_contract_complete"]


def test_state_identity_and_referenced_path_mutations_fail() -> None:
    state = load_json(STATE_PATH)
    wrong_identity = copy.deepcopy(state)
    wrong_identity["workflow"] = "other"
    missing_artifact = copy.deepcopy(state)
    missing_artifact["output_artifact_path"] = "docs/research/does-not-exist.md"

    identity_validation = Validation()
    validate_state(wrong_identity, identity_validation)
    path_validation = Validation()
    validate_state(missing_artifact, path_validation)

    assert not identity_validation.checks["autoresearch_state_complete"]
    assert identity_validation.checks["referenced_artifacts_exist"]
    assert not path_validation.checks["autoresearch_state_complete"]
    assert not path_validation.checks["referenced_artifacts_exist"]
    assert state == EXPECTED_STATE


def test_design_only_claim_contract_rejects_observed_claims() -> None:
    priorities = load_json(PRIORITIES_PATH)
    priorities["claim_contract"]["observed_claims"] = ["accuracy improved"]

    validation = Validation()
    validate_priorities(priorities, validation)

    assert not validation.checks["design_only_claim_contract"]


def test_report_empirical_claim_mutation_is_rejected() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8") + "\n99% improved on the benchmark.\n"

    validation = Validation()
    validate_report_text(report, validation)

    assert not validation.checks["no_observed_empirical_claims"]
    assert validation.checks["report_contract"]


def test_result_preserves_independent_check_outcomes() -> None:
    validation = Validation()
    validation.fail("metrics_unique_nonempty_strings", "mutation")

    result = build_result(validation, {})

    assert result["passed"] is False
    assert result["checks"]["metrics_unique_nonempty_strings"] is False
    assert result["checks"]["primary_sources_present"] is True
    assert result["checks"]["autoresearch_state_complete"] is True
