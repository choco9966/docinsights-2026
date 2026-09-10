# Held-out 1,730건 12시간 전수 재검수 계획

- 추적 이슈: #26
- 기준선: 0910 제출 후보와 문항별 풀이·Evidence·판정 이력
- 대상: held-out 1,730개 고유 instance_id
- 경계: Train/Validation 재생성, 새 OCR 비교, 공식 포털 제출은 제외

## 목표

기존 답변을 보존하면서 번복이 자주 발생한 지점을 원문·계산·Evidence·버전 단위로 재검증한다. 자동점검, 의미검토, 독립 수락, 제출 준비를 별도 상태로 집계하고 12시간 뒤 미완료·미확정 범위를 숨기지 않는다.

## 반복 번복 유형

1. 답만 바꾸고 풀이·가정·Evidence에 이전 값이 남는 부분 튜플 병합
2. 6/8, 0/O, I/l, &/8, 부호·소수점·통화·긴 숫자의 OCR 오독
3. 위치·주제가 비슷한 비관련 문단 선택 및 질문 대상·행위·수량 관계 누락
4. 날짜 양끝 포함, 주/월 환산, 누적량/당일량, 분할 불가 수량, 종료 조건, 반올림·단위 오류
5. crop/PDF 해시는 맞지만 다른 문단의 block ID를 Evidence로 등록
6. 생성·구조 통과·원문 검토·독립 수락·제출 준비 상태 혼동
7. stale 버전의 재사용, 다른 문항 결과 결합, 근거 없는 중복 재검토
8. answer 설명/단위 혼입, evidence에 문장 입력 등 직렬화 오류

## 시간표

### H0–H0.5 기준 고정

0910 파일, ID manifest, PDF/source hash, baseline tuple hash, 상태 ledger를 read-only로 고정하고 중복·누락 검사 기준을 만든다.

### H0.5–H1.5 5건 계측 게이트

동일 실행 경로 5건의 wall time, 입력/캐시/출력/reasoning 토큰, 실패·재시도, 저장량을 기록한다. 형식 5/5와 원문 Evidence 검증을 통과하기 전에는 대량 배정하지 않는다.

### H1.5–H5.5 위험 우선 병렬 검수

서로 겹치지 않는 shard로 분리한다.

- Source-first: 기존 답을 보지 않고 PDF full-page/crop에서 질문 대상·숫자·단위·부호·block ID 전사
- Calculation: 원문 입력만으로 식·조건·단위·반올림 재계산 및 경쟁 해석 기록
- Evidence: 질문과 모든 계산 입력을 직접 지지하는 최소 block 집합 확인
- Version/contradiction: before/after 전체 튜플, stale 값, 상태 전이, cross-task 결합 검사

### H5.5–H8.5 독립 수락

변경 후보만 blind source packet과 현재 튜플을 다른 검토자가 대조한다. answer 또는 Evidence가 바뀌면 전체 튜플을 다시 검증하고, 불일치 시 쟁점만 한 번 재개방한다. 100건 체크포인트마다 처리량·변경·상태·unresolved·토큰·시간을 기록한다.

### H8.5–H10.5 자동점검·projection

1,730 ID exactly-once, JSONL/CSV 대응, answer 문자열, Evidence block ID, 빈 Evidence·진단문구 누출, 중복·누락을 검사한다. 내부 풀이에서 제출용 instance_id/answer/evidence를 결정적으로 생성하고 전후 hash를 비교한다.

### H10.5–H12 통합·보존

독립 수락된 변경만 release candidate에 반영한다. 최종 JSONL, 팀 CSV, 문항별 풀이·Evidence, risk/unresolved ledger, usage/verification report를 묶고 미완료·공식 미확정 범위를 별도 기록한다.

## 운영 규칙

- 동일 문항 동시 수정 금지; 제안 파일과 통합 파일을 분리한다.
- 원문·해시·버전 ledger를 에이전트 결론보다 우선한다.
- 새 근거 없는 재검토, 무한 재시도, 완화된 schema 수락을 금지한다.
- timeout/저장/전송 오류는 실패 단계만 복구하고 성공 풀이를 통째로 재생성하지 않는다.
- 조건부·미확정은 내용 상태이며 점수용 예측과 별도 보존한다.
- 공식 포털 점수·정답률·순위는 제출 및 채점 전에는 주장하지 않는다.

## 완료 기준

- [ ] 1,730/1,730 ID 자동점검 및 baseline/after hash 검증
- [ ] 5건 계측의 처리량·토큰·실패율 보고
- [ ] 모든 변경 문항의 source packet·계산·Evidence·독립 수락
- [ ] deterministic 제출 projection JSONL·팀 CSV
- [ ] unresolved/conditional 목록과 미완료 범위
- [ ] 실행 로그·검증 결과·재현 명령 보존
