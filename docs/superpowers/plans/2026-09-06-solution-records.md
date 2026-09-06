# 전체 split 풀이 기록 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for assigned independent tasks. Respect file ownership and existing user changes.

**Goal:** 질문/풀이/답/Evidence를 보존하는 검증 가능한 전체 split 기록 생성.

**Architecture:** 공개 입력 → 독립 풀이 → 산술·Evidence·coverage 검증 → 출력 동결 → 별도 reference 비교. 전체 실행 순서는 heldout, train, validation이다.

**Tech Stack:** Python 3.11+, 기존 pytest/ruff, 표준 라이브러리, 설치된 Codex/PDF/OCR 도구.

**Spec:** docs/superpowers/specs/2026-09-06-solution-records-design.md

## Global Constraints

- 새 의존성을 추가하지 않는다.
- Validation/Held-out 출력은 gitignored 로컬 파일로만 보존한다.
- 임의 Evidence ID와 다중 블록을 지원한다.
- 기존 #23 소스와 ledger를 수정하지 않는다.
- 정답을 먼저 읽고 풀이를 만들지 않는다.

## Task 1 — 기록 계약과 exporter

Files: src/docinsights_analysis/solution_records.py, tests/test_solution_records.py.

- [ ] 합성 다중 Evidence 문항으로 contract, 잘못된 인용·산술식·중복 ID·누락 거부를 먼저 테스트하고 실패 확인.
- [ ] validate_solution(record, task, pages), export_records(records, tasks, output_dir), evaluate_records(records, references, reference_kind)를 구현한다. pages는 page_number/text 객체 배열이며 record에 source_pages로 보존한다. solution은 summary 문자열 및 expression/result 객체 배열 calculations다.
- [ ] 같은 tests를 재실행하고 ruff 검사한다.

## Task 2 — 사전 감사와 실제 1건 검증

Files: docs/research/issue-24-solution-records.md, artifacts/solution-records/issue24/preflight/ (private).

- [ ] v1~v24의 확인 가능한 집계와 원인, 미복구 구간, source-label recovery의 한계를 기록한다.
- [ ] manifest와 PDF 해시를 고정하고 validation 1건의 질문/PDF를 읽는다.
- [ ] 출력 기록을 생성·동결한 뒤 v24에서 해당 행만 읽어 별도 비교한다. 차이가 있으면 원문 기반 원인을 기록한다.

## Task 3 — 순차 실행 및 검토

Files: scripts/run_solution_records.py, tests/test_solution_runner.py, .omx/ultragoal/ (private).

- [ ] 순서 차단·해시 재개·실패 기록 테스트를 먼저 실행한다.
- [ ] 동일 공개 입력 계약으로 실제 solver 호출을 실행하고 raw final response 및 provenance를 보존한다.
- [ ] Held-out 전체 coverage 통과 후 Train, Train 전체 coverage 통과 후 Validation을 실행한다.
- [ ] 각 단계에서 fresh get_goal snapshot과 실제 출력/검증으로 checkpoint한다.
- [ ] cleaner 이후 재검증, 독립 code-reviewer/architect 및 불변식 증명을 완료하고 코드·문서만 PR로 게시한다.
