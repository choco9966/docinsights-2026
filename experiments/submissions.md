# DocSem 제출 실험 기록

공식 Submission Portal의 집계 결과와 제출 산출물 식별 정보만 기록합니다. 대회 진행 중인 Validation의 개별 `instance_id`, answer, evidence, 계산 과정과 포털 차분 판정은 Git에서 제외된 `artifacts/submissions/`와 `artifacts/docsem_validation/`의 비공개 감사 산출물에만 보관합니다.

## 제출 결과

| 버전 | 제출일 (KST) | 제출 파일 | SHA-256 | Examples | Answer accuracy | Evidence exact match | Evidence F1 | 연구 구분 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `v1` | 2026-08-29 | `artifacts/submissions/v1.jsonl` | `ae5e57f45edf6c094e82cb02b749f62fc811af8a631c05790d5dec09a5d3a996` | 217 | 0.97235 | 1.0 | 1.0 | PDF·`user_query` 기반 최초 블라인드 풀이 |
| `v2` | 2026-08-29 18:29:53 | `artifacts/submissions/v2.jsonl` | `2be417abe62c528c10bb35e06dfd10e585e7e606804756abf343693b313bef35` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v3` | 2026-08-29 18:36:16 | `artifacts/submissions/v3.jsonl` | `2a424d2ac66050718678b346a600c5c236f4a0ef90ae2ea8cecafd6fec0a9a52` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v4` | 2026-08-29 19:35:21 | `artifacts/submissions/v4.jsonl` | `c7784972ac231df59e4102b112e3354ab5e58292da7e6ac3daef2905d5b0cd99` | 217 | 0.981567 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v5` | 2026-08-29 20:40:53 | `artifacts/submissions/v5.jsonl` | `c950f2511e8486a722da268ad6d39a96fab34f0afc3e2ae8943235fd22ffe7ed` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v6` | 2026-08-29 20:42:37 | `artifacts/submissions/v6.jsonl` | `889263ea9a6ace1562a8ea7eeb2715765d83d95e8c3b825607516ea159802a44` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v7` | 2026-08-29 20:45:40 | `artifacts/submissions/v7.jsonl` | `db953923ca2ec0b9c6c0ad5e8009e64484ea81e560f03e8ecd92a6dba31d19fe` | 217 | 0.995392 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v8` | 2026-08-29 20:48:24 | `artifacts/submissions/v8.jsonl` | `a194f3d81969dbb7686fe0f676b5571d320c2af521dc20e72d87a79c810e64b1` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v9` | 2026-08-29 20:52:18 | `artifacts/submissions/v9.jsonl` | `a909b9e603eaa64a02e6ff1fc1cc98773da5f1c0327040ddab12c2dec155b46b` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v10` | 2026-08-29 20:53:55 | `artifacts/submissions/v10.jsonl` | `34f7f14808d80071c1238721b74ca60268b24376b2c96d51139c4b4b16d84ed3` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v11` | 2026-08-29 | `artifacts/submissions/v11.jsonl` | `3b31480cce4dd29303d43447211ff8a96589c9edf035e487fb8f7ac5e6addab3` | 217 | 0.990783 | 1.0 | 1.0 | 포털 점수 차분 진단 |
| `v12` | 2026-08-30 15:14:43 | `artifacts/submissions/v12.jsonl` | `38a386805a3c9b87f50f52b009006cf67bb7f8b71b2ceb1bbd2f4502e99f79fd` | 217 | 1.0 | 1.0 | 1.0 | Post-hoc source-label recovery |

## 연구 해석

- `v1`은 제공된 PDF와 `user_query`만 사용한 최초 풀이이며, 이후 제출은 숨겨진 Validation 포털 점수를 관찰하면서 후보를 바꾼 post-hoc 진단입니다. 따라서 `v2`~`v12` 성능을 독립 holdout 일반화 성능이나 재현 가능한 모델 정확도로 해석하지 않습니다.
- `v12`는 공개 원천 데이터의 label을 사후 대조한 source-label recovery 결과입니다. 포털의 세 집계 지표가 모두 `1.0`인 사실은 기록하되, Issue #5의 PDF+query-only 규약을 준수한 완료 결과나 논문의 주 성능으로 사용하지 않습니다.
- 포털 피드백으로 선택한 규칙, 후보, 정답 또는 evidence를 Training 감독 신호, prompt 선택, reward 설계, checkpoint 선택에 사용하지 않습니다. 후속 연구는 Training-only 개발과 family-disjoint 내부 평가를 사용하고 숨겨진 holdout은 마지막 one-shot 평가로 봉인합니다.

## 비공개 감사 계약

- 각 제출 JSONL과 개별 변경 기록은 Git에서 제외된 경로에 보관하고 공개 문서에는 파일명, 전체 SHA-256과 집계 점수만 남깁니다.
- 포털에서 확인한 행을 도구에 전달할 때는 `--portal-confirmations`와 `--portal-confirmations-sha256`을 함께 지정합니다. 도구는 비공개 산출물 자체의 SHA-256과 그 안에 선언된 기준 제출물 SHA-256을 모두 검증하며, 누락되거나 다르면 비교를 중단합니다.
- 현재 tip에서 민감한 상세를 제거해도 이미 공개된 Git 이력은 자동으로 삭제되지 않습니다. 이력 재작성은 복제본과 열린 작업에 영향을 주는 별도의 파괴적 조치이므로 범위와 주최 측 통지 필요성을 확인한 뒤 별도로 결정합니다.

비공개 포털 확정 산출물은 다음 계약을 따릅니다. 예시의 placeholder를 실제 값으로 바꾼 뒤 파일 자체의 SHA-256을 별도 채널에 기록합니다.

```json
{"schema_version":"1.0","baseline_sha256":"<64-hex>","rows":[{"instance_id":"task_XXXXXX","answer":"<final-answer>","evidence":["bNN"]}]}
```

```bash
uv run docinsights compare-blind-review artifacts/docsem_validation/review.jsonl --baseline artifacts/submissions/<baseline>.jsonl --portal-confirmations artifacts/docsem_validation/portal-confirmations.json --portal-confirmations-sha256 <64-hex>
```

## 새 제출 기록 방법

새 제출은 `artifacts/submissions/<version>.jsonl`에 저장하고 제출 전에 형식과 SHA-256을 확인합니다.

```bash
uv run docinsights validate-submission artifacts/submissions/<version>.jsonl --tasks data/raw/docsem/val/tasks.jsonl
shasum -a 256 artifacts/submissions/<version>.jsonl
```

같은 SHA-256은 동일한 예측으로 취급합니다. 개별 변경 사유와 검수 계산은 비공개 감사 기록에만 추가하고 이 문서에는 집계 결과만 한 행으로 기록합니다.
