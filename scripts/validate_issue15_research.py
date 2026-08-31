#!/usr/bin/env python3
"""Validate Issue #15 research artifacts and write the autoresearch verdict."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/research/issue-15-docsem-followup-research.md"
PRIORITIES_PATH = ROOT / "artifacts/research/docsem-followup-priorities.json"
MISSION_PATH = ROOT / ".omx/specs/autoresearch-docsem-followup/mission.md"
SANDBOX_PATH = ROOT / ".omx/specs/autoresearch-docsem-followup/sandbox.md"
STATE_PATH = ROOT / ".omx/state/autoresearch-docsem-followup/autoresearch-state.json"
RESULT_PATH = ROOT / ".omx/specs/autoresearch-docsem-followup/result.json"

CHECK_NAMES = (
    "artifact_files_exist",
    "priorities_schema",
    "design_only_claim_contract",
    "candidate_count_at_least_8",
    "candidate_contract_complete",
    "metrics_unique_nonempty_strings",
    "truth_and_holdout_contract",
    "template_family_contract",
    "primary_sources_present",
    "verifier_precedes_rl",
    "first_actions_total_60_minutes",
    "three_day_mvp_complete",
    "report_contract",
    "no_observed_empirical_claims",
    "autoresearch_state_complete",
    "referenced_artifacts_exist",
)
REQUIRED_CANDIDATES = {
    "ambiguity_taxonomy_audit",
    "metamorphic_contrastive_consistency",
    "ocr_fact_equation_error_decomposition",
    "structure_consensus_selective_verification",
    "small_qwen_verifier",
    "family_disjoint_leakage_audit",
    "rl_trajectory_pilot",
    "image_text_hybrid",
}
REQUIRED_CANDIDATE_FIELDS = {
    "rank",
    "id",
    "name",
    "track",
    "hypothesis",
    "novelty",
    "action_1h",
    "extension_1d",
    "extension_3d",
    "metrics",
    "stop_condition",
    "failure_condition",
    "compute",
    "leakage_risk",
    "dependencies",
    "sources",
}
REQUIRED_SOURCE_URLS = {
    "https://arxiv.org/abs/2605.07053",
    "https://github.com/oracle-samples/gsm-sem",
    "https://arxiv.org/abs/2410.05229",
    "https://openreview.net/forum?id=x2W2dKdNI8",
    "https://arxiv.org/abs/2406.14024",
    "https://arxiv.org/abs/2304.09102",
    "https://arxiv.org/abs/2503.16219",
}
REQUIRED_REPORT_TERMS = {
    "semantic_truth",
    "benchmark_label",
    "silver_agreement_not_human_gold_accuracy",
    "family-disjoint",
    "template_family_contract",
    "숨겨진 validation/test",
    "구조 합의와 선택적 검수",
    "소형 Qwen verifier",
    "OCR→사실→방정식 오류 분해",
    "모호성 taxonomy 감사",
    "Metamorphic·contrastive consistency",
    "Family-disjoint 누출 감사",
    "이미지–텍스트 hybrid",
    "RL trajectory pilot",
    "## 3일 MVP",
}
EXPECTED_STATE = {
    "workflow": "autoresearch",
    "slug": "docsem-followup",
    "validation_mode": "mission-validator-script",
    "status": "completed",
    "mission_path": ".omx/specs/autoresearch-docsem-followup/mission.md",
    "sandbox_path": ".omx/specs/autoresearch-docsem-followup/sandbox.md",
    "completion_artifact_path": ".omx/specs/autoresearch-docsem-followup/result.json",
    "output_artifact_path": "docs/research/issue-15-docsem-followup-research.md",
    "machine_readable_artifact_path": "artifacts/research/docsem-followup-priorities.json",
    "mission_validator_command": "uv run python scripts/validate_issue15_research.py",
}
REFERENCED_ARTIFACT_FIELDS = (
    "mission_path",
    "sandbox_path",
    "completion_artifact_path",
    "output_artifact_path",
    "machine_readable_artifact_path",
)
EMPIRICAL_CLAIM_PATTERNS = (
    re.compile(
        r"(?i)\b\d+(?:\.\d+)?%\s*(?:improved|improvement|better|higher|lower|gain(?:ed)?|"
        r"increase(?:d)?|decrease(?:d)?|accuracy|precision|recall|향상|개선|증가|감소|달성)\b"
    ),
    re.compile(
        r"(?i)\b(?:accuracy|precision|recall|f1|auroc|auprc|score|performance)\s*"
        r"(?:=|:|was|is|rose|reached|increased|decreased|improved)\s*\d+(?:\.\d+)?%?"
    ),
    re.compile(
        r"(?:정확도|정밀도|재현율|점수|성능|F1|AUROC|AUPRC)\s*(?:가|는|은|:|=)?\s*"
        r"\d+(?:\.\d+)?%?\s*(?:였다|이다|달성했다|향상됐다|향상되었다|개선됐다|"
        r"개선되었다|증가했다|감소했다)"
    ),
    re.compile(r"(?:실측값|실측 결과|측정 결과)\s*(?:은|는|:|=)\s*[^\n]*\d+(?:\.\d+)?"),
)


class Validation:
    """Collect independent check outcomes and their evidence."""

    def __init__(self) -> None:
        self.checks = dict.fromkeys(CHECK_NAMES, True)
        self.errors: list[str] = []

    def fail(self, check: str, message: str) -> None:
        if check not in self.checks:
            raise KeyError(f"unknown validation check: {check}")
        self.checks[check] = False
        self.errors.append(f"[{check}] {message}")


def load_json(path: Path, validation: Validation, check: str) -> dict[str, Any]:
    if not path.is_file():
        validation.fail(check, f"missing file: {path.relative_to(ROOT)}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        validation.fail(check, f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        return {}
    if not isinstance(value, dict):
        validation.fail(check, f"JSON root must be an object: {path.relative_to(ROOT)}")
        return {}
    return value


def require_nonempty_text(value: Any, label: str, validation: Validation, check: str) -> None:
    if not isinstance(value, str) or not value.strip():
        validation.fail(check, f"{label} must be non-empty text")


def validate_claim_contract(data: dict[str, Any], validation: Validation) -> None:
    claim = data.get("claim_contract")
    if not isinstance(claim, dict):
        validation.fail("design_only_claim_contract", "claim_contract must be an object")
        return
    expected = {
        "mode": "design_only",
        "empirical_experiments_run": False,
        "observed_claims": [],
        "observed_metrics": [],
    }
    for field, value in expected.items():
        if claim.get(field) != value:
            validation.fail(
                "design_only_claim_contract", f"claim_contract.{field} must equal {value!r}"
            )
    allowed = claim.get("allowed_claim_types")
    if (
        not isinstance(allowed, list)
        or not allowed
        or not all(isinstance(item, str) and item.strip() for item in allowed)
    ):
        validation.fail(
            "design_only_claim_contract",
            "claim_contract.allowed_claim_types must be non-empty strings",
        )


def validate_template_family_contract(data: dict[str, Any], validation: Validation) -> None:
    contract = data.get("template_family_contract")
    if not isinstance(contract, dict):
        validation.fail("template_family_contract", "template_family_contract must be an object")
        return

    expected_scalars = {
        "producer": "issue_15_preprocessing",
        "join_key": "instance_id",
        "issue_14_role": "ambiguity axes/tags and separated semantic/benchmark fields only",
        "unknown_rate_abort_threshold": 0.05,
        "split_invariant": "one template_family_id belongs to exactly one split",
    }
    for field, expected in expected_scalars.items():
        if contract.get(field) != expected:
            validation.fail(
                "template_family_contract",
                f"template_family_contract.{field} must equal {expected!r}",
            )

    expected_lists = {
        "source_priority": [
            "canonical_gsm_sem_provenance",
            "versioned_label_blind_structural_fingerprint",
        ],
        "forbidden_inputs": ["benchmark_label", "hidden_holdout", "submission_score"],
        "required_manifest_fields": [
            "instance_id",
            "template_family_id",
            "derivation",
            "algorithm_version",
            "source_sha256",
        ],
    }
    for field, expected in expected_lists.items():
        if contract.get(field) != expected:
            validation.fail(
                "template_family_contract",
                f"template_family_contract.{field} must equal {expected!r}",
            )


def validate_priorities(data: dict[str, Any], validation: Validation) -> None:
    if data.get("schema_version") != "issue-15-docsem-followup-priorities-v2":
        validation.fail("priorities_schema", "unexpected schema_version")
    if data.get("research_mode") != "design_only_no_empirical_claims":
        validation.fail("priorities_schema", "research_mode must prohibit empirical claims")
    if "verifier" not in str(data.get("primary_strategy", "")):
        validation.fail("priorities_schema", "primary_strategy must prioritize verification")
    validate_claim_contract(data, validation)
    validate_template_family_contract(data, validation)

    truth = data.get("truth_contract")
    if not isinstance(truth, dict):
        validation.fail("truth_and_holdout_contract", "truth_contract must be an object")
    else:
        for field in ("semantic_truth", "benchmark_label", "ocr_reference"):
            require_nonempty_text(
                truth.get(field),
                f"truth_contract.{field}",
                validation,
                "truth_and_holdout_contract",
            )
        holdout = truth.get("hidden_holdout_policy")
        require_nonempty_text(
            holdout,
            "truth_contract.hidden_holdout_policy",
            validation,
            "truth_and_holdout_contract",
        )
        if isinstance(holdout, str):
            for term in ("사용하지 않는다", "family-disjoint", "한 번만"):
                if term not in holdout:
                    validation.fail(
                        "truth_and_holdout_contract",
                        f"hidden_holdout_policy missing term: {term}",
                    )
        benchmark = truth.get("benchmark_label", "")
        if "별도 필드" not in str(benchmark) or "덮어쓰지 않는다" not in str(benchmark):
            validation.fail(
                "truth_and_holdout_contract",
                "benchmark_label must remain separate from semantic_truth",
            )

    sources = data.get("sources")
    if not isinstance(sources, list):
        validation.fail("primary_sources_present", "sources must be a list")
        sources = []
    source_ids: set[str] = set()
    source_urls: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            validation.fail("primary_sources_present", f"sources[{index}] must be an object")
            continue
        source_id = source.get("id")
        url = source.get("url")
        if not isinstance(source_id, str) or not source_id:
            validation.fail("primary_sources_present", f"sources[{index}].id must be non-empty")
        else:
            source_ids.add(source_id)
        if not isinstance(url, str) or not url.startswith("https://"):
            validation.fail(
                "primary_sources_present", f"sources[{index}].url must be a direct HTTPS URL"
            )
        else:
            source_urls.add(url)
    missing_urls = REQUIRED_SOURCE_URLS - source_urls
    if missing_urls:
        validation.fail(
            "primary_sources_present",
            f"missing required primary source URLs: {sorted(missing_urls)}",
        )

    timeline = data.get("first_60_minutes")
    if not isinstance(timeline, list) or not timeline:
        validation.fail(
            "first_actions_total_60_minutes", "first_60_minutes must be a non-empty list"
        )
    else:
        ranks = [item.get("rank") for item in timeline if isinstance(item, dict)]
        minutes = [item.get("minutes") for item in timeline if isinstance(item, dict)]
        if ranks != list(range(1, len(timeline) + 1)):
            validation.fail(
                "first_actions_total_60_minutes", "timeline ranks must be consecutive and ordered"
            )
        if any(not isinstance(value, int) or value <= 0 for value in minutes):
            validation.fail(
                "first_actions_total_60_minutes", "timeline minutes must be positive integers"
            )
        elif sum(minutes) != 60:
            validation.fail(
                "first_actions_total_60_minutes", f"timeline must total 60, got {sum(minutes)}"
            )
        for index, item in enumerate(timeline):
            if not isinstance(item, dict):
                validation.fail(
                    "first_actions_total_60_minutes", f"timeline[{index}] must be an object"
                )
                continue
            require_nonempty_text(
                item.get("action"),
                f"first_60_minutes[{index}].action",
                validation,
                "first_actions_total_60_minutes",
            )
            require_nonempty_text(
                item.get("deliverable"),
                f"first_60_minutes[{index}].deliverable",
                validation,
                "first_actions_total_60_minutes",
            )

    mvp = data.get("three_day_mvp")
    if not isinstance(mvp, dict):
        validation.fail("three_day_mvp_complete", "three_day_mvp must be an object")
    else:
        for field in ("primary_track", "day_1", "day_2", "day_3", "rl_role"):
            require_nonempty_text(
                mvp.get(field), f"three_day_mvp.{field}", validation, "three_day_mvp_complete"
            )
        if "verifier" not in str(mvp.get("primary_track", "")):
            validation.fail("three_day_mvp_complete", "primary track must include a verifier")
        if "MVP 밖" not in str(mvp.get("rl_role", "")):
            validation.fail("three_day_mvp_complete", "RL must remain outside the primary MVP")
        gate = mvp.get("success_gate")
        if not isinstance(gate, list) or len(gate) < 5:
            validation.fail(
                "three_day_mvp_complete", "success_gate must contain at least five checks"
            )

    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        validation.fail("candidate_count_at_least_8", "candidates must be a list")
        validation.fail("candidate_contract_complete", "candidate contracts cannot be checked")
        validation.fail("metrics_unique_nonempty_strings", "candidate metrics cannot be checked")
        return
    if len(candidates) < 8:
        validation.fail(
            "candidate_count_at_least_8", f"at least 8 candidates required, got {len(candidates)}"
        )

    candidate_ids: set[str] = set()
    ranks: list[Any] = []
    rank_by_id: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            validation.fail("candidate_contract_complete", f"candidates[{index}] must be an object")
            continue
        missing_fields = REQUIRED_CANDIDATE_FIELDS - candidate.keys()
        if missing_fields:
            validation.fail(
                "candidate_contract_complete",
                f"candidates[{index}] missing fields: {sorted(missing_fields)}",
            )
        candidate_id = candidate.get("id")
        if isinstance(candidate_id, str):
            candidate_ids.add(candidate_id)
            if isinstance(candidate.get("rank"), int):
                rank_by_id[candidate_id] = candidate["rank"]
        ranks.append(candidate.get("rank"))
        for field in (
            "name",
            "hypothesis",
            "novelty",
            "action_1h",
            "extension_1d",
            "extension_3d",
            "stop_condition",
            "failure_condition",
            "compute",
            "leakage_risk",
        ):
            require_nonempty_text(
                candidate.get(field),
                f"candidates[{index}].{field}",
                validation,
                "candidate_contract_complete",
            )
        metrics = candidate.get("metrics")
        valid_metrics = (
            isinstance(metrics, list)
            and len(metrics) >= 3
            and all(isinstance(metric, str) and metric.strip() for metric in metrics)
        )
        if not valid_metrics:
            validation.fail(
                "metrics_unique_nonempty_strings",
                f"candidates[{index}].metrics must have at least three non-empty strings",
            )
        elif len(metrics) != len(set(metrics)):
            validation.fail(
                "metrics_unique_nonempty_strings",
                f"candidates[{index}].metrics must be unique",
            )
        dependencies = candidate.get("dependencies")
        if not isinstance(dependencies, dict):
            validation.fail(
                "candidate_contract_complete",
                f"candidates[{index}].dependencies must be an object",
            )
        else:
            for issue in ("issue_14", "issue_8", "issue_11"):
                require_nonempty_text(
                    dependencies.get(issue),
                    f"candidates[{index}].dependencies.{issue}",
                    validation,
                    "candidate_contract_complete",
                )
            issue_14_dependency = dependencies.get("issue_14", "")
            if any(
                forbidden in str(issue_14_dependency)
                for forbidden in ("template family", "family ID 또는")
            ):
                validation.fail(
                    "template_family_contract",
                    f"candidates[{index}] must not assign family production to Issue #14",
                )
        candidate_sources = candidate.get("sources")
        if not isinstance(candidate_sources, list) or not candidate_sources:
            validation.fail(
                "candidate_contract_complete", f"candidates[{index}].sources must be non-empty"
            )
        elif unknown := set(candidate_sources) - source_ids:
            validation.fail(
                "candidate_contract_complete",
                f"candidates[{index}] cites unknown sources: {sorted(unknown)}",
            )

    missing_candidates = REQUIRED_CANDIDATES - candidate_ids
    if missing_candidates:
        validation.fail(
            "candidate_contract_complete",
            f"missing required candidates: {sorted(missing_candidates)}",
        )
    if ranks != list(range(1, len(candidates) + 1)):
        validation.fail(
            "candidate_contract_complete", "candidate ranks must be consecutive and ordered"
        )
    verifier_ranks = [
        rank_by_id.get("structure_consensus_selective_verification", 999),
        rank_by_id.get("small_qwen_verifier", 999),
    ]
    rl_rank = rank_by_id.get("rl_trajectory_pilot", -1)
    if rl_rank < max(verifier_ranks):
        validation.fail(
            "verifier_precedes_rl", "RL must rank after structure/selective and Qwen verification"
        )


def empirical_claim_matches(report: str) -> list[str]:
    """Return suspicious observed-result fragments from a design-only report."""

    matches: list[str] = []
    for pattern in EMPIRICAL_CLAIM_PATTERNS:
        for match in pattern.finditer(report):
            fragment = match.group(0).strip()
            if fragment not in matches:
                matches.append(fragment)
    return matches


def validate_report_text(report: str, validation: Validation) -> None:
    for term in REQUIRED_REPORT_TERMS:
        if term not in report:
            validation.fail("report_contract", f"report missing required term: {term}")
    for url in REQUIRED_SOURCE_URLS:
        if url not in report:
            validation.fail("report_contract", f"report missing primary source URL: {url}")
    if "모델 학습이나 정확도 측정은 실행하지 않았으므로" not in report:
        validation.fail(
            "report_contract", "report must disclose that empirical experiments were not run"
        )
    if "uv run python scripts/validate_issue15_research.py" not in report:
        validation.fail("report_contract", "report must include the validator command")
    suspicious = empirical_claim_matches(report)
    if suspicious:
        validation.fail(
            "no_observed_empirical_claims",
            f"design-only report contains observed-result-like claim(s): {suspicious}",
        )


def validate_report(validation: Validation) -> str:
    if not REPORT_PATH.is_file():
        validation.fail("artifact_files_exist", f"missing file: {REPORT_PATH.relative_to(ROOT)}")
        validation.fail("report_contract", "report cannot be checked")
        validation.fail("no_observed_empirical_claims", "report cannot be checked")
        return ""
    report = REPORT_PATH.read_text(encoding="utf-8")
    validate_report_text(report, validation)
    return report


def validate_state(state: dict[str, Any], validation: Validation, root: Path = ROOT) -> None:
    for field, value in EXPECTED_STATE.items():
        if state.get(field) != value:
            validation.fail("autoresearch_state_complete", f"state.{field} must equal {value!r}")
    for field in REFERENCED_ARTIFACT_FIELDS:
        relative = state.get(field)
        if not isinstance(relative, str) or not relative:
            validation.fail("referenced_artifacts_exist", f"state.{field} must be a non-empty path")
            continue
        path = root / relative
        if not path.is_file():
            validation.fail(
                "referenced_artifacts_exist", f"state.{field} does not exist: {relative}"
            )


def validate_workflow_files(validation: Validation) -> None:
    for path in (MISSION_PATH, SANDBOX_PATH, STATE_PATH, PRIORITIES_PATH, RESULT_PATH):
        if not path.is_file():
            validation.fail("artifact_files_exist", f"missing file: {path.relative_to(ROOT)}")
    for path in (MISSION_PATH, SANDBOX_PATH):
        if path.is_file() and not path.read_text(encoding="utf-8").strip():
            validation.fail(
                "artifact_files_exist", f"empty workflow file: {path.relative_to(ROOT)}"
            )
    state = load_json(STATE_PATH, validation, "autoresearch_state_complete")
    validate_state(state, validation)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_result(validation: Validation, hashes: dict[str, str]) -> dict[str, Any]:
    passed = not validation.errors and all(validation.checks.values())
    error_count = len(validation.errors)
    summary = "Issue #15 research mission validator passed"
    if not passed:
        summary = f"Issue #15 research mission validator failed with {error_count} error(s)"
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "summary": summary,
        "validation_mode": "mission-validator-script",
        "validator_command": "uv run python scripts/validate_issue15_research.py",
        "checks": validation.checks,
        "errors": validation.errors,
        "artifact_sha256": hashes,
    }


def write_result(validation: Validation) -> None:
    tracked = [REPORT_PATH, PRIORITIES_PATH, MISSION_PATH, SANDBOX_PATH, STATE_PATH]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in tracked if path.is_file()}
    result = build_result(validation, hashes)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    validation = Validation()
    priorities = load_json(PRIORITIES_PATH, validation, "artifact_files_exist")
    validate_priorities(priorities, validation)
    validate_report(validation)
    validate_workflow_files(validation)
    write_result(validation)

    if validation.errors:
        for error in validation.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"FAILED: {len(validation.errors)} validation error(s)", file=sys.stderr)
        return 1
    print("PASSED: Issue #15 research artifacts satisfy the mission contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
