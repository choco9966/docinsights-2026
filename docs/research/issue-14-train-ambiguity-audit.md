# Issue #14 · DocSem Train 908개 모호성 전수 감사

## 결론

DocSem Train 908개를 텍스트 기반으로 전수 처리해 908개 고유 `instance_id`를 모두 보존했고, 635개(69.93%)를 `S0` clean candidate, 273개(30.07%)를 `S1`–`S4` 주의 문항으로 분류했다. 주의 문항은 `S1` 67개, `S2` 56개, `S3` 56개, `S4` 94개이며 실제 재검수 대상으로 표시된 행은 170개다.

이 결과는 **자동 규칙 검사와 세 에이전트의 텍스트 전수 감사 결과이며, 사람의 최종 판정이나 PDF 시각 검수가 아니다.** `S0`도 정답 라벨이 검증됐다는 뜻이 아니라 현재 추출 텍스트에서 명시적 손상·모호성 신호를 찾지 못했다는 뜻이다.

## 핵심 분포

| 축 | 분포 |
| --- | --- |
| `surface_integrity` | `intact` 784, `awkward_but_parseable` 80, `corrupted` 44 |
| `semantic_determinacy` | `unique_explicit` 660, `unique_with_convention` 103, `multiple_plausible` 95, `underdetermined` 50 |
| `benchmark_alignment` | `aligned` 41, `semantic_conflict` 12, `label_unverifiable` 855 |
| `review_required` | `true` 170, `false` 738 |

`label_unverifiable` 855개를 라벨 오류 855개로 해석하면 안 된다. 독립적인 의미 답과 계산 근거가 함께 기록되지 않은 경우 보수적으로 검증 불가를 부여했으며, 구조화된 semantic answer로 실제 충돌을 입증한 것은 12개다.

가장 많이 붙은 비-clean 태그는 `implicit_default_assumption` 104개, `label_template_carryover_suspected` 77개, `condition_insertion_or_deletion` 55개, `rate_unit_relation_corruption` 55개, `cardinality_operand_gap` 50개, `quantifier_scope_last_remaining` 30개, `target_or_subquestion_drift` 28개, `role_or_subject_attribution` 27개 순이다. 태그는 중복될 수 있으며 `label_template_carryover_suspected`는 오류 확정이 아니라 인접 템플릿과 라벨의 재사용 가능성을 알리는 검수 신호다.

## 판정 계약

라벨을 받지 않는 blind pass를 먼저 `artifacts/ambiguity/train-ambiguity-blind.jsonl`에 기록하고 질문 SHA-256을 고정한 뒤 별도 비교 pass에서 공개 Train label과 agent review를 결합한다. 최종 `train-ambiguity-tags.jsonl`은 `semantic_answer`와 `benchmark_answer`를 분리하고, 근거 block ID·복원 block ID·page와 복원 질문·근거 block text·원 PDF·Codex reference file·task manifest의 SHA-256을 함께 보존한다. 최종 행 안의 `blind_question_screen`은 persisted blind 행과 field-for-field 동일하게 유지하며 reviewer 판정은 오직 최상위 `axes`에 반영한다.

세 축은 각각 `surface_integrity = intact | awkward_but_parseable | corrupted`, `semantic_determinacy = unique_explicit | unique_with_convention | multiple_plausible | underdetermined`, `benchmark_alignment = aligned | normalized_equivalent | semantic_conflict | label_unverifiable`로 고정했다. 약한 단어 신호는 `auto_signals`에만 기록하며 단어 하나만으로 모호성 태그를 확정하지 않는다.

`review_required`는 심각도와 별도 판정이다. `S0`·`S1`은 검수 보류를 만들지 않고 `S3`·`S4` 및 semantic conflict는 반드시 검수 대상으로 보내며, `S2`는 명시적 복구 가능성과 confidence에 따라 에이전트 판단을 보존한다.

## 대표 오류 유형

- `task_000003`: 42 blocks/hour와 10분당 3 blocks가 서로 다른 시간을 만들며 `rate_unit_relation_corruption`으로 분류했다.
- `task_000008` 계열: “percentage more likely”가 percentage-point 차이와 relative increase를 모두 허용해 `comparison_polarity_or_sign`으로 분류했다.
- `task_000015`: 7·13 teaspoons라는 절대량과 120-teaspoon total이 충돌하지만 라벨은 7:13 비율처럼 계산해 `S4`로 분류했다.
- `task_000027`: eight-sided target을 중간에 twelve-sided target으로 교체해 `target_or_subquestion_drift`로 분류했다.
- `task_000366`: 보이는 피연산자는 1.0을 만들지만 라벨 68이 인접 템플릿과 닮아 `semantic_conflict`와 carryover 의심으로 분류했다.
- `task_000647`: group rate와 individual rate가 구분되지 않고 장비 설명도 충돌해 `cardinality_operand_gap`·`role_or_subject_attribution`으로 분류했다.
- `task_000029`: 물음표가 여러 개라는 자동 신호는 남겼지만 최종 target은 해석 가능하므로 고위험 target drift로 승격하지 않았다.

## 분할 경계와 전수 범위

| reviewer | 정확한 범위 | 전체 | override | clean candidate |
| --- | --- | ---: | ---: | ---: |
| agent-a | `task_000001`–`task_000303` | 303 | 86 | 217 |
| agent-b | `task_000304`–`task_000606` | 303 | 85 | 218 |
| agent-c | `task_000607`–`task_000908` | 302 | 102 | 200 |
| 합계 | Train 전체 | 908 | 273 | 635 |

초기 병렬 집계에서 `task_000303`이 두 shard에 들어가 274개로 잘못 합산됐지만 exact ID partition 검증으로 중복을 제거했다. `task_000606`은 agent-b의 기본 clean 판단이고 `task_000908`은 agent-c의 기본 clean 판단이다.

각 review shard는 severity별 공통 문구를 재사용하지 않고 `default_decision + override_groups`로 저장한다. 273개 override는 full decision tuple인 axes, canonical tags, severity, `review_required`, 문항별 rationale, confidence를 보존하며 validator는 범위·중복·태그 계약·rationale 다양성을 모두 검사한다.

## 재현

```bash
uv run python -m docinsights_ambiguity audit \
  --tasks artifacts/ambiguity/inputs/train-tasks.jsonl \
  --labels artifacts/ambiguity/inputs/train-labels.jsonl \
  --reference artifacts/ambiguity/inputs/codex-train-reference.jsonl \
  --query-comparison artifacts/ambiguity/inputs/codex-train-query-comparison.jsonl \
  --blind-output artifacts/ambiguity/train-ambiguity-blind.jsonl \
  --review-shard artifacts/ambiguity/reviews/shard-a-review.json \
  --review-shard artifacts/ambiguity/reviews/shard-b-review.json \
  --review-shard artifacts/ambiguity/reviews/shard-c-review.json \
  --output artifacts/ambiguity/train-ambiguity-tags.jsonl \
  --summary artifacts/ambiguity/train-ambiguity-summary.json

uv run python -m docinsights_ambiguity validate \
  --tasks artifacts/ambiguity/inputs/train-tasks.jsonl \
  --output artifacts/ambiguity/train-ambiguity-tags.jsonl \
  --summary artifacts/ambiguity/train-ambiguity-summary.json \
  --expected-count 908
```

네 입력 JSONL은 `artifacts/ambiguity/inputs/`에 정확한 908행 스냅샷으로 추적하며 파일별 SHA-256은 같은 디렉터리의 `README.md`에 고정했다. 따라서 위 명령은 별도 임시 worktree나 ignored `data/`·`artifacts/ocr/` 디렉터리를 요구하지 않는다.

Validator는 output SHA-256·집계·embedded validation·입력 및 review source hash를 실제 파일과 대조한다. 회귀 테스트는 output만 바꾸고 이전 summary를 재사용하는 경우와 query-comparison에 중복 행을 추가하는 경우가 모두 exit code 1로 실패함을 고정한다.

## 한계와 다음 단계

현재 deterministic semantic-answer engine은 회귀 테스트로 고정한 일부 산술 형태만 계산하며, agent review도 추출 텍스트만 사용했다. 따라서 우선순위는 `review_required=true` 170개 중 `semantic_conflict`, `corrupted`, `underdetermined`, `extraction_or_ocr_uncertain` 순으로 PDF 시각 검수와 독립 2인 판정을 수행하는 것이다. 숨겨진 validation/test 제출 점수는 taxonomy, threshold, prompt 선택에 사용하지 않고 공개 Train의 family-disjoint calibration에서 정책을 동결해야 한다.
