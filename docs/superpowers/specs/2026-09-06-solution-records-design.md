# 전체 split 풀이 기록 설계

Issue: #24. 사용자 요청에 따라 자율 실행한다. 기존 #23 작업은 원 checkout에 보존하고 feature/24 worktree에서 독립 실행한다.

## 목표와 순서

먼저 validation 24차 이력의 일반적 실패 원인을 감사하고 validation 1건을 공개 입력만으로 풀어 고정한 뒤 v24-sep03-check-05와 비교한다. 이어 Held-out 1730건 → Train 908건 → Validation 217건의 순서로 전체 기록을 생성한다. 각 split의 정확한 수와 SHA-256은 실제 manifest에서 검증한다. 미완료 split을 완료로 표시하거나 순서를 바꾸지 않는다.

## 대안과 선택

기존 제출에 설명을 사후 부착하면 정답에 맞춘 풀이가 될 수 있다. 기존 bNN 한 개 전용 solver를 확장하면 #23 실험과 결합된다. 현재 공개 문서·질문만 받는 독립 생성 경로와 생성 후 평가 경로를 만들고 기존 OCR/데이터 도구만 재사용한다.

## 출력

로컬 gitignored artifacts/solution-records/issue24 아래에 split별 solutions.jsonl, solutions.md, submission.jsonl, evaluation.json, manifest.json을 둔다. 각 기록에는 instance_id, split, question, solution(간결한 근거 설명, 재검산 가능한 계산식), answer, evidence(ID 배열), evidence_details(ID, 페이지, 원문 인용), provenance(PDF SHA-256, 입력·설정 해시, 모델/실행 출처), uncertainties를 기록한다. 풀이란 사용자에게 검증 가능한 설명이며 내부 추론 로그를 뜻하지 않는다.

## 불변식

1. 입력은 현재 문항의 공개 query/PDF 또는 검증된 OCR뿐이다. 정답·이전 풀이·피드백은 solver 입력에 넣지 않는다.
2. 비교 정답은 생성 파일의 해시를 고정한 뒤 별도 평가에서만 읽는다. Validation v24는 과거 제출 reference이며 organizer gold라고 부르지 않는다.
3. Evidence ID는 문서에 보이는 값을 그대로 보존하며 bNN 또는 한 개로 제한하지 않는다. 인용은 실제 페이지 텍스트에 있어야 한다.
4. 계산식은 허용된 산술 표현만 결정적으로 계산한다. 의미상 모호함·OCR 불확실성·부족한 수치는 숨기지 않는다.
5. ID 중복·누락·타 split 혼입·변경된 입력을 거부한다. 중단과 재개는 입력 및 설정 해시가 일치할 때만 허용한다.
6. 공개 Git에는 Validation/Held-out 문항별 답·근거·계산을 넣지 않는다. 이슈·PR은 집계와 절차만 포함한다.
7. 외부 포털 제출·PR 병합·기존 작업 변경은 하지 않는다. 실행 실패는 실패로 기록하고 출력이 없는 작업을 완료로 표시하지 않는다.

## 검증

합성 문서로 다중 Evidence·임의 ID, 인용과 페이지 검증, 산술 검산, ID coverage, 변경 입력 재개 거부, 생성/평가 분리를 검사한다. 실제 validation 1건에서 문서 검토와 v24 비교를 완료하고 순차 실행한다. 최종 cleaner 후 재검증, 독립 code-reviewer와 architect 검토 및 불변식 감사를 통과해야 aggregate goal을 완료한다.
