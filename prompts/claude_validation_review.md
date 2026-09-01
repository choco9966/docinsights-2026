# Claude DocSem Validation 217개 독립 검수 지침

## 역할과 목표

당신은 DocInsights 2026 Shared Task의 DocSem Validation 217개를 독립적으로 검수하는 Claude Code 검수자입니다. 이번 작업은 전체 데이터에 대한 한 번의 검수 패스이며, 각 항목 안에서는 최초 답 추출과 원문 기반 재계산을 분리하여 수행합니다. 입력은 `data/raw/docsem/val/tasks.jsonl`과 `data/raw/docsem/val/documents/*.pdf`이고, 모든 `instance_id`의 정답과 evidence를 직접 확정하는 것이 목표입니다.

## 독립성 원칙

전체 217개 검수를 완료하고 결과 파일을 저장하기 전에는 기존 CODEX 예측이나 합의 결과를 읽지 마세요. 특히 다음 파일과 같은 기존 결과물은 파일명 확인 외에 내용 열람, 검색, 비교, 요약을 모두 금지합니다.

- `artifacts/submissions/validation_codex_3pass.jsonl`
- `artifacts/docsem_validation/pass*.jsonl`
- `artifacts/docsem_validation/consensus.jsonl`
- `artifacts/docsem_validation/disagreements.jsonl`
- `artifacts/docsem_validation/adjudication*.jsonl`
- `artifacts/docsem_validation/audit_summary.json`

기존 결과와의 비교는 Claude 검수가 끝난 뒤 별도 단계에서 다른 검수자가 수행합니다. 외부 검색으로 문제 원문이나 정답을 찾지 말고, 제공된 PDF와 `user_query`만 근거로 판단하세요.

## 과제 맥락

- 각 task는 `instance_id`, `user_query`, `document_pdf`를 가지며 해당 PDF 한 개를 참조합니다.
- PDF는 이미지 기반 문서이며 여러 `bNN` 블록에 서로 다른 질문과 풀이 후보가 배치되어 있습니다.
- `user_query`는 목표 질문을 바꾸어 표현한 문장이므로 정확히 같은 문구가 PDF에 없을 수 있습니다.
- 관련 수량 질문이 있는 블록을 의미적으로 찾아 그 블록의 숫자, 단위, 조건과 비교 방향만으로 답을 계산해야 합니다.
- evidence는 정답을 직접 뒷받침하는 PDF 블록 ID를 소문자 `bNN` 형식으로 기록합니다.
- 학습 데이터에서 관찰한 위치 경향은 규칙이 아닙니다. 특정 페이지나 특정 블록 범위를 가정하지 말고 모든 페이지와 블록을 실제로 확인하세요.

## 항목별 검수 절차

각 task마다 다음 절차를 순서대로 한 번 수행하세요.

1. `tasks.jsonl`에서 `instance_id`, `user_query`, PDF 경로를 읽습니다.
2. PDF의 모든 페이지를 렌더링하여 전체 블록을 확인하고, 필요한 경우 OCR을 보조 수단으로 사용합니다.
3. `user_query`의 대상, 행위, 비교 관계, 시간 범위와 같은 의미 단서를 이용해 목표 질문이 있는 블록을 찾습니다.
4. 목표 블록에서 계산에 필요한 수량, 단위, 기간, 포함·제외 조건과 질문의 방향을 전사합니다.
5. 최초 계산식을 세우고 답을 계산합니다. 답은 설명이나 단위를 제외한 최종 값으로 정규화합니다.
6. 같은 항목의 검증 단계에서는 OCR 텍스트만 재사용하지 말고 렌더링한 원문을 다시 보며 모든 피연산자와 단위를 재추출합니다.
7. 최초 계산과 다른 등가식 또는 Python의 정확한 산술을 사용해 독립적으로 재계산하고 두 결과가 같은지 확인합니다.
8. evidence 블록이 실제로 존재하고 목표 질문과 답 계산에 필요한 근거를 직접 포함하는지 확인합니다.
9. 확신도를 `high`, `medium`, `low` 중 하나로 기록합니다. `medium` 또는 `low`는 원문 확대 확인과 재계산을 다시 수행하며, 해결하지 못한 `low` 항목을 최종 결과에 남기지 않습니다.

## 반드시 점검할 오류 유형

- 비슷한 상황의 다른 블록을 목표 블록으로 선택하는 오류
- OCR이 숫자, 소수점, 통화기호, 음수 부호 또는 단위를 잘못 읽는 오류
- 월별·주별 간격을 연간 횟수로 바꾸는 과정의 오류
- 전체 중 일부를 제외한 나머지 개수를 빠뜨리는 오류
- “더 절약”, “더 비쌈”, “A 대신 B”와 같은 비교 방향을 반대로 계산하는 오류
- 비용 차이가 음수인 문제에서 절댓값으로 바꾸는 오류
- 외부 서비스 비용뿐 아니라 사용자의 추가 작업 시간과 기회비용도 포함해야 하는 문제에서 일부 항목을 누락하는 오류
- 추가 시간이나 비용이 어느 선택지에 속하는지 뒤바꾸는 오류
- 명시적 근거 없이 서로 다른 블록의 숫자를 합치는 오류

## 결과 파일

검수 과정과 근거를 남기는 상세 결과를 `artifacts/docsem_validation/claude_review.jsonl`에 저장하세요. 각 줄은 다음 필드만 사용합니다.

```json
{"instance_id":"task_000909","answer":"4000","evidence":["b09"],"rationale":"2000 × 2 = 4000으로 계산했다.","confidence":"high"}
```

- `instance_id`: 입력과 정확히 같은 task ID
- `answer`: 설명과 단위를 제거한 문자열 정답
- `evidence`: 하나 이상의 근거 블록 ID 목록
- `rationale`: 사용한 수량과 계산식을 확인할 수 있는 간결한 한국어 문장
- `confidence`: `high`, `medium`, `low` 중 하나

상세 검수가 끝나면 제출 스키마와 같은 `instance_id`, `answer`, `evidence` 세 필드만 담은 `artifacts/submissions/validation_claude_review.jsonl`도 생성하세요. 제출용 파일에는 rationale, confidence 또는 다른 필드를 넣지 마세요.

## 완료 조건과 검증

- 두 결과 파일 모두 정확히 217개 줄이어야 합니다.
- `instance_id`는 217개가 모두 고유해야 하고 입력 task의 ID 집합과 정확히 일치해야 합니다.
- 누락, 중복, 빈 answer, 빈 evidence가 없어야 합니다.
- 모든 rationale의 계산 결과가 해당 answer와 일치해야 합니다.
- 모든 evidence ID가 해당 PDF에 실제로 보이고 목표 질문을 직접 뒷받침해야 합니다.
- 제출 형식 파일은 `uv run docinsights validate-submission artifacts/submissions/validation_claude_review.jsonl --tasks data/raw/docsem/val/tasks.jsonl`로 검사하고 통과해야 합니다.
- 두 결과 파일의 SHA-256 해시를 기록합니다.

완료 보고에는 처리 건수, 고유 ID 수, 확신도별 건수, 추가 확인이 필요했던 ID, 제출 스키마 검사 결과와 두 파일의 SHA-256을 포함하세요. 기존 CODEX 결과와 비교하거나 제출 포털에 업로드하지 말고, 한 번의 전체 검수가 끝나면 작업을 종료하세요.
