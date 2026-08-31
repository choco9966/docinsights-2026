# Autoresearch mission · DocSem follow-up

## 목표

Issue #15의 공개·1차 출처 기반 1시간 후속 연구를 문서와 기계 판독 가능한 우선순위 목록으로 고정한다. 최소 8개 후보 각각에 가설, novelty, 1시간 실행, 1일·3일 확장, 지표, 중단·실패 조건, compute, 누출 위험, Issue #14/#8/#11 의존성을 기록한다.

## 필수 판정

- `semantic_truth`와 `benchmark_label`을 명시적으로 분리하고 숨겨진 holdout을 선택·조정에 사용하지 않는다.
- ambiguity audit, metamorphic/contrastive consistency, OCR→fact→equation decomposition, structure consensus/selective verification, small Qwen verifier, family-disjoint audit, RL pilot, image–text hybrid를 모두 포함한다.
- 첫 60분 실행을 합계 60분으로 순위화하고 3일 MVP를 제시한다.
- verifier와 selective review를 primary로 두고 RL은 gate 뒤의 진단으로 둔다.
- 지정된 7개 primary source URL을 문서와 JSON에 기록한다.
- JSON의 `claim_contract`에 design-only 상태와 빈 observed claim/metric을 기록하고 문서에 미실행 실측 주장을 두지 않는다.
- `uv run python scripts/validate_issue15_research.py`가 통과하고 completion artifact에 `status=passed`, `passed=true`를 기록한다.

## 완료 산출물

- `docs/research/issue-15-docsem-followup-research.md`
- `artifacts/research/docsem-followup-priorities.json`
- `.omx/specs/autoresearch-docsem-followup/result.json`

실험을 실행하지 않은 상태에서 성능 향상을 실측처럼 주장하면 미션 실패다.
