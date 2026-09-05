# Issue #21 · Validation 복원과 최신 Train 기반 held-out 전략

기준일은 2026-09-06이다. 이 문서는 실행 결과와 다음 실험 계획을 구분한다. 연결 이슈는 [#21](https://github.com/choco9966/docinsights-2026/issues/21)이며, 앞선 [Issue #15 연구](issue-15-docsem-followup-research.md)와 [Issue #7 실행 설계](issue-7-qwen-72h-plan.md)를 최신 release 기준으로 구체화한다.

## 1. 완료 결과와 해석 경계

Validation 최종 유지본은 `v24-sep03-check-05.jsonl`이다. 포털에서 217/217 answer accuracy 100.00%, evidence F1 100.00%를 확인했다. 포털 표시 시각은 `2026-09-05 16:34:19`이며 표시 시간대는 별도 확인하지 않았다.

| 제출 | Answer 정답 수 | Evidence F1 | 조치 |
| --- | ---: | ---: | --- |
| v19 재채점 기준 | 214/217 | 100.00% | 출발점 |
| v20 | 213/217 | 100.00% | 변경 폐기 |
| v21 | 215/217 | 100.00% | 유지 |
| v22 | 216/217 | 100.00% | 유지 |
| v23 | 215/217 | 100.00% | 변경 폐기 |
| v24 | 217/217 | 100.00% | 최종 동결 |

각 시도는 직전 제출이 아니라 **직전 유지본에서 answer 하나만 변경**했고 evidence는 유지했다. 모든 파일의 217개 ID·JSONL 형식을 제출 전에 검사했다. 최종 파일 SHA-256은 `265d89696e1216f887ea3722c13bf83b3ac5c03292c7460db9afc8732328ec13`이다.

이 100%는 포털 피드백으로 조정한 private reference의 적합도다. 이전 source-label recovery 패치도 상속하므로, 공식 공개 정답·누출 없는 모델 성능·held-out 예상 정확도로 표현하지 않는다. 이번 작업에서는 외부 원천 문제 검색을 추가 실행하지 않았다. Validation 항목별 답·계산·근거·변경 내역은 Git 추적 밖 `artifacts/submissions/`에만 보존한다. v24를 모델 학습, prompt example, family 생성, threshold 선택에 투입하지 않는다. Validation 추가 제출은 중단했다.

## 2. 최신 Train migration

현재 active 경로는 `data/raw/docsem/`, 구본 보존 경로는 `data/raw/docsem-before-sep06/`이다.

| 항목 | 고정 버전 / 확인 결과 |
| --- | --- |
| 기존 HF | `b171c5ad488f0c8c50df05951a5b288ea50e9501` |
| 현재 HF | `e6c9c75bea7575a64279072dcdf0f6050fef9e9f` |
| 현재 canonical DocSem | `971262d356c1e7fc2da534eeb9d2c828ade42157` |
| 공개 파일 | 1,132개; Train 908개, Validation 217개 |
| 변경 | Train PDF 7개와 README |
| 변경 없음 | Train tasks·labels, Validation tasks·전체 PDF |

최신본의 Train 수정은 **공개 labels.jsonl 값 변경이 아니라 PDF 내용 변경**이다. 변경된 Train ID는 `task_000419`, `task_000444`, `task_000447`, `task_000452`, `task_000454`, `task_000455`, `task_000857`이다. 이 ID 목록은 migration 진단용이며 모델 feature나 답 override가 아니다.

유지된 manifest 해시는 다음과 같다.

- Train tasks: `6d9cd9087d0c5e30bfc17c83aec30752403d4109fb93d8357f534da425969489`
- Train labels: `3e39dfb708cccc7999676d23aae8342fb0e71a94a8e1b5629339c6f0209dc33f`
- Validation tasks: `5b6f57a30f4dc8b27873162ca58434c0411fb89f4726b5f2988903344b43443a`

기존 OCR reference 908행 중 901행은 최신 PDF 해시와 일치하고 7행은 불일치한다. 변경 7건은 최신 PDF에서 Tesseract 200 DPI / PSM 6으로 재실행했으며 7/7 status `ok`였다. 그러나 독립 검증에서 한 행의 `b22`가 `622`로 읽혀 앞 블록에 합쳐진 것을 발견했다. 그 PDF만 300 DPI로 재실행하자 `b01`–`b23` 전체 ID가 복원되었다. 6개는 200 DPI, 1개는 300 DPI인 최신 선택을 별도 source manifest에 고정하고 원래 실패 출력도 보존했다. 이는 블록 구조와 처리 성공의 확인이지 OCR 텍스트의 의미 정확도 100%가 아니다.

구 Issue #8 silver reference와 신규 Tesseract 출력은 생성 방법이 다르므로 하나의 균일한 gold corpus로 합치지 않는다. 901행은 입력 PDF 해시가 맞는 engineering reference로만 재사용 가능하다. 신규 7행은 별도 provenance를 유지하고, 전체 최신 908행의 동일-engine benchmark와 Issue #14 의미 태그 재감사는 아직 수행하지 않았다. 과거 908건 결과에 새 revision 이름만 붙이는 일은 금지한다.

로컬 감사 기록:

- `data/release-audit/2026-09-06-migration.json`: 구·신 해시와 실제 변경 경로
- `data/release-audit/changed-train-tasks.jsonl`: 최신 7건 입력
- `artifacts/ocr/train-e6c9c75-changed-seven.jsonl`: 신규 OCR와 PDF SHA
- `artifacts/ocr/train-e6c9c75-task419-300dpi.jsonl`: 누락 block ID를 복원한 300 DPI 재추출
- `data/release-audit/latest-changed-train-ocr-sources.json`: 변경 7건의 명시적 최신 artifact 선택과 fingerprint
- `artifacts/submissions/sep03-restoration-audit.json`: 비공개 validation 제출 이력

다운로더 기본 SHA와 README 예제도 최신본으로 고정했고, 명시적 과거 revision 다운로드는 계속 지원한다. 활성 분석 Notebook은 공용 revision 상수를 사용하도록 바꾸고 manifest와 누락 PDF 다운로드 경로를 모두 regression test로 검증했다. 저장된 Notebook 이미지 사례는 변경 7건과 겹치지 않고 tasks·labels도 동일하므로 기존 실행 출력은 보존했다.

## 3. 변경 7건의 label-blind 진단

서로의 출력과 공개 라벨을 보지 않는 두 독립 에이전트가 최신 PDF 이미지와 user_query만으로 풀었다. A는 직접 풀이, C는 사실의 역할·단위·시간 범위와 방정식·역산을 명시하는 구조화 풀이였다. 두 응답이 모두 끝난 뒤 공개 Train label과 비교했다. 자세한 수치·관찰은 [진단 JSON](issue-21-train-diagnostic.json)에 보존한다.

C의 단일 수치 답은 공개 라벨과 6/7 일치했다. 두 방식 모두 단일 수치로 답한 공통 6건에 한정하면 A는 4/6, C는 5/6이었다. 나머지 한 문항에서 A는 여러 요청량을 모두 답했고 C는 마지막 요청량을 단일 답으로 선택했다. 이 차이를 숨겨 동일한 출력 계약의 성능 비교처럼 보고하지 않는다. 두 방식이 선택한 evidence block은 각각 7/7 공개 block과 일치했다.

이것은 수정된 입력을 의도적으로 고른 **소규모 진단**이며 random/family-disjoint 평가가 아니다. A/C의 출력 지시도 다르고, 동일 모델 계열의 상관 오류 가능성이 있다. Qwen을 실행한 결과도 아니며 두 trace의 구조 합의 시스템을 평가한 것도 아니다. 따라서 C를 최적 전략으로 확정할 통계적 근거는 없다.

### 실제로 드러난 실패 원인

1. 손실·잔액의 귀속: 직접 풀이가 지갑 분실 금액을 잘못 연결했다. 구조화 풀이의 entity·event binding과 역산이 이 차이를 드러냈다.
2. 전체 기간과 하루의 분모: 총 작업량을 먼저 일수로 나눈 뒤 근무시간과 비교해야 한다. 같은 숫자가 일수·시간·단위 환산에 반복될 때 역할을 각각 기록해야 한다.
3. 다중 요청과 단일 answer 계약: 계산이 맞아도 여러 값을 한 문자열에 담으면 제출 표현이 불안정하다. query가 지정한 목표량을 우선하고, generic query 아래 다중 목표가 남으면 검수 대상으로 보낸다.
4. 잔류자와 퇴장자: “are left”의 수식 대상을 근거 문장에 연결하고, 각 집단의 전후 합계로 검증해야 한다.
5. 문서와 공개 label의 불일치: 수정 Train 한 건은 두 풀이와 원본 PDF 재검수 모두 평균 40%를 지지하지만 공개 label은 50이다. 문서 계산은 `160 rooms × 15 min / (10 days × 10 h/day × 60 min/h) × 100 = 40`이다. 이는 미해결 benchmark conflict로 기록하며 공식 label을 덮어쓰지 않는다.

마지막 사례는 “같은 답을 여러 번 말하면 안전하다”의 한계를 보여준다. 모델의 의미상 정답과 benchmark exact match를 별도 측정해야 하며, label을 맞추기 위해 근거에 없는 50% 해설을 학습시키거나 해당 ID를 하드코딩해서는 안 된다. 사람 2인 독립 판정까지 완료한 human gold라고 주장하지 않는다.

## 4. 우선 검증할 robust 후보

권장 **첫 구현 후보**는 C: `근거 → 역할이 있는 사실 → 방정식 → 결정적 계산 → 역산 → 독립 구조 비교 → 선택적 이미지 검수`다. 검수는 내부 절차이고, 최종 제출에는 모든 문항의 answer와 evidence를 채운다. 보류 행을 누락시켜 selective accuracy만 높이는 방식은 사용하지 않는다.

| 비교군 | 고정 절차 | 검증 목적 |
| --- | --- | --- |
| A: direct | 동일 입력에서 단일 직접 답, 동일 JSON 스키마 | 비용·정확도 기준선 |
| B: answer repetition | 3회 독립 답, 사전 고정 tie-break | 답 다수결의 효과 |
| C: structure + review | 독립 구조 2회, 계산기 검증, 충돌 건만 이미지 재검수 | 잘못된 fact binding과 OCR 오류 감소 |
| D: small verifier | C의 trace에 소형 Qwen 판정 추가 | C보다 나아질 때만 후속 승격 |

모든 비교군에 동일한 제출 스키마·수치 정규화·입력 PDF·기본 OCR을 적용한다. 모델 이름과 checkpoint, prompt, decoder, seed, 최대 출력 토큰, 재시도·검수 예산을 기록한다. C의 추가 연산을 감추지 않고 accuracy–cost 곡선 및 p50/p95 latency를 보고한다.

C의 구체적 계약:

- Retrieval은 문서의 실제 visible block에서 목표 질문·계산 조건을 찾는다. 주제가 비슷한 distractor나 행정 숫자는 계산에 넣지 않는다.
- 각 수치는 `value, unit, entity, role, time_scope, source_block`으로 기록한다. 동일 숫자라도 역할이 다르면 별도 항목이다.
- 총량/일일량, paid/available/lost, remaining/departed, 순변화/절대량, percentage의 분모를 명시한다. 음수나 100% 초과라는 이유만으로 자동 보정하지 않는다.
- 계산은 제한된 산술 연산자로 실행한다. 자유 코드 eval, PDF 안의 지시문 실행, 외부 원천 문제 조회를 허용하지 않는다.
- 역산은 추출한 조건을 모두 복원하는지 검사한다. 올바른 역산도 잘못 추출한 사실을 정당화할 수 있으므로 원문 binding 검사와 분리한다.
- 두 trace는 서로의 답을 보지 않고 생성한다. 답이 같아도 evidence·역할·식이 다르면 충돌이다. 같은 모델의 두 응답을 완전 독립 표본으로 가정하지 않는다.
- 충돌, 모호한 수식, 단위/시간 불일치, OCR 숫자 의심, 다중 목표일 때만 원본 이미지 기반 추가 검수를 허용한다. 검수자에게 benchmark label을 보여 주지 않는다.
- 검수 후에도 후보가 여러 개면 Train calibration에서 미리 고정한 우선순위로 forced answer를 선택하고 불확실성을 내부 로그에 남긴다. hidden 점수로 우선순위를 바꾸지 않는다.

## 5. Train-only 실험 설계와 선택 gate

### 데이터 분리

공개 Train 908개만 사용한다. 최종 공개 label과 의미 판정은 별도 파일로 유지하고, solver 입력에는 PDF와 user_query만 전달한다. Validation 문항별 패치, 원천 문제 정답, portal feedback은 접근 금지 목록에 둔다.

먼저 최신 문서에서 label-blind extraction을 수행하고, 수치·고유명사를 제거한 문제 구조와 near-duplicate 검사를 사용해 versioned family graph를 만든다. PDF의 공통 행정 boilerplate 때문에 서로 다른 문제를 한 family로 묶지 않도록 목표 구절 중심으로 검사한다. 외부 원천 문제 검색 없이 공개 Train 입력으로 묶으며, 과거 upstream ID 매핑이 있다면 누출 감사용으로만 사용하고 추론에 넣지 않는다.

family 단위 development/calibration/sealed-test를 목표 비율 60/20/20, seed `20260906`으로 고정한다. 실제 개수는 family 크기에 따라 달라질 수 있으며 manifest 생성 전 숫자를 지어내지 않는다. 이번에 사람이 이미 검토한 7건과 연결 sibling은 development로만 보낸다. 기존 Issue #14/#8 전수 노출 이력이 있으므로 이는 새로운 독립 데이터가 아니라 **Train 내부 out-of-family 추정치**다. 모델 호출별 입력 격리와 평가 데이터 접근 로그를 남기고, 진짜 독립 검증은 공식 held-out으로만 판단한다.

다음이면 평가를 중단한다: family overlap > 0, unknown-family > 5%, PDF/manifest SHA 불일치, blind output에 label 흔적, 동일 문항 중복/누락. family graph와 split SHA를 고정한 뒤에만 label을 join한다.

### 비교·선택 규칙

1. Development 24건에서 A/B/C의 동일 스키마 smoke와 예외 처리를 확인한다. 이는 성능 확정용이 아니다.
2. Development 내부에서 prompt·정규화·검수 규칙을 수정하고 버전을 남긴다. 개선 전후 회귀를 기간 분모·손실 귀속·잔류/퇴장·다중 목표·부호·OCR 혼동·distractor 유형별로 검사한다.
3. Calibration 전체에서 A/B/C를 같은 입력과 고정 예산으로 비교한다. Primary는 **모든 문항에 forced answer한 benchmark accuracy**다. Evidence exact/F1, 수동 판정된 subset의 semantic accuracy, schema 실패율, family별 worst-group score, 검수율과 시간·비용을 함께 본다.
4. C 승격 조건은 A보다 높은 calibration forced-answer accuracy, evidence exact 비감소, schema/누락 오류 0, 사전 정한 총 비용 예산 이내다. 초기 예산 상한은 A 추론 비용의 3배이며 타당성은 development에서만 조정한다. 차이와 불확실성은 family-cluster bootstrap으로 보고한다. 효과가 불명확하거나 같으면 더 단순하고 저렴한 A를 유지한다.
5. 선택된 구성과 최대 2개 대안의 checkpoint/prompt/threshold/예산/출력 규칙을 동결한 뒤 sealed-test를 한 번 평가한다. sealed 결과로 재튜닝하거나 여러 후보 중 새로 골라 성능을 과장하지 않는다. 누출 또는 심각한 회귀가 발견되면 “검증 실패”로 기록한다.
6. D는 C보다 오류 검출 AUPRC·forced-answer 결과가 개선되고 비용 상한을 통과할 때만 후보가 된다. RL/SFT를 먼저 시작하지 않는다. 특히 conflict label에서 정답 해설을 역으로 생성해 학습하는 경로는 제외한다.

risk–coverage와 confidence calibration은 내부 검수 자원 배분용 보조 지표다. 검수 후 전체 정답률과 함께 보고하며, 낮은 coverage의 높은 accuracy를 대회 최종 점수로 바꾸어 표현하지 않는다.

## 6. 실행 순서와 남은 구현

| 순서 | 계획 시간 | 산출물 / 완료 gate |
| --- | --- | --- |
| 0. 입력 복구 | 완료 | 최신 PDF·manifest, 이전본 보존, SHA diff, 변경 7건 OCR |
| 1. provenance와 family | 실행 시작 후 0–6시간 목표 | 최신 908행 소비 manifest, engine 구분, label-blind family graph, split overlap 0 |
| 2. 공통 runner | 6–24시간 목표 | A/B/C 동일 JSON 계약, 24건 smoke, label 접근 차단 테스트, 산술·수치 정규화 회귀 |
| 3. Train 비교 | 24–48시간 목표 | calibration 전체 결과, family별 오류와 비용, review policy 동결 |
| 4. 봉인 평가·패키징 | 48–72시간 목표 | sealed-test 1회, 모델·prompt·입력·출력 SHA, 재현 명령, 최종 제출 후보 |
| 5. official held-out | 공개 후 실제 잔여시간 기준 | 공식 release 무결성 확인, 전수 추론, 사전 등록 후보 순차 제출 |

이는 작업량 추정치이며 예약된 자동 실행이나 완료 약속이 아니다. 현재 checked-in Qwen QA/trajectory runner와 family split manifest는 없고, 기존 small-model notebook은 OCR smoke이지 solver/SFT runner가 아니다. 이번 7건 진단도 native-agent 진단이다. 모델·runtime 확보와 비용 확인 없이 Qwen 학습 완료를 주장하지 않는다. 새 유료 GPU 구매나 계정 변경은 이 계획에 포함하지 않는다.

Held-out 공개가 늦어지면 위의 실험 규모부터 축소한다. 공식 최종 마감의 시간대와 연장 여부를 확인한 뒤, 남은 시간의 최소 25%는 전수 추론·검수·업로드 오류 대응에 남긴다. 시간이 부족하면 D/RL을 생략하고 검증된 A 또는 C를 동결한다.

## 7. 공식 held-out 제출 전략

[공식 포털](https://amitbcp-docsem-docinsights.hf.space/)과 [워크숍 공지](https://docinsights-workshop.github.io/docinsights-2026/shared-task/)에 따르면 기준일 현재 held-out은 아직 공개되지 않았고 test submissions도 닫혀 있다. 예전 “마감 5일 전 공개” 안내를 실제 공개 완료로 해석하지 않는다. 현재 공지된 최종 제출 날짜는 2026-09-10이며 정확한 제출 마감 시간대는 제출 전에 재확인한다.

[워크숍에 공개된 제출 정책](https://docinsights-workshop.github.io/docinsights-2026/shared-task/)은 signed-in Hugging Face account당 accepted test submission 최대 3회다. 첫 accepted attempt만 accuracy/evidence F1을 즉시 반환하고 2·3회는 최종 집계까지 보류한다. 최종 순위는 best eligible test attempt를 사용한다. 이는 발표된 정책이며 현재 test 창이 열려 있다는 뜻은 아니다. Validation의 latest-submission 표시와 구분하고, 제출 직전에 공식 안내와 열린 UI를 다시 확인한다.

1. 공식 release가 나오면 task ID·문서 수·schema·PDF hash·라이선스/규칙을 새로 고정한다. ID나 metadata로 답을 추론하지 않는다.
2. 첫 점수를 보기 전에 최대 3개 후보와 순서를 manifest에 사전 등록한다. 예: Train에서 선택된 primary, 같은 정책의 독립 decoding 대안, Train에서 검증된 modality 대안. 동일 약점을 가진 후보를 억지로 3개 만들 필요는 없다.
3. 1회차는 Train 근거가 가장 강한 동결 primary다. 가능하면 2·3회 후보도 첫 점수 확인 전에 생성·해시 고정한다. 즉시 점수를 얻을 수 있다는 이유로 실험용 저품질 파일을 먼저 내지 않는다.
4. 2·3회는 사전 등록된 후보만 제출한다. 첫 hidden 점수를 보고 prompt·규칙·후보·threshold를 새로 조정하지 않는다. 대안의 평가가 없다면 primary 1회로 끝낼 수 있다.
5. 업로드 전 전수 ID·누락/중복·answer 형식·visible evidence 존재·파일 SHA를 검사한다. 제출 뒤 accepted receipt·attempt 수·제출명·SHA를 대조한다.
6. 이번 validation API는 결과 직렬화 오류를 반환해도 제출이 저장되는 경우가 있었다. test에서 timeout/error가 나면 UI/receipt를 먼저 대조하고 재전송 여부를 판단한다. 동일 시도를 무작정 반복해 기회를 소모하지 않는다.
7. Test는 공식 Hugging Face OAuth 경로를 사용한다. Validation의 비인증 API 흐름을 test에 적용하거나 다른 계정으로 횟수 제한을 우회하지 않는다.

## 8. 완료 체크리스트

- [x] Validation 217/217, evidence F1 100.00% 공식 UI 확인
- [x] v24 private reference 동결과 개별 validation 정보 공개 차단
- [x] 최신 Train 실제 다운로드, 구본 보존, 변경 PDF 7개 식별
- [x] 다운로더 기본 revision 수정과 regression test
- [x] 수정 PDF 7개 OCR 재생성·블록 누락 1건 재추출, label-blind 직접/구조 풀이 진단
- [x] 문서 계산과 공개 Train label의 미해결 불일치 기록
- [ ] 최신 908행 provenance 소비 manifest 및 family-disjoint split 생성
- [ ] 동일 출력 계약의 A/B/C runner와 Train 전체 calibration 비교
- [ ] Train 내부 sealed 평가와 후보 동결
- [ ] 공식 held-out release 확인 후 전수 추론·순차 제출

결론: Validation 100% 복원과 최신 Train 입력 갱신은 완료했다. Held-out의 최적 전략은 아직 실측으로 확정하지 않았으며, 현재 우선 후보는 구조화 풀이와 선택적 검수다. 위 Train-only 비교를 통과한 구성만 최종 후보로 승격한다.

## 공식 근거와 provenance

- [HF 고정 release](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data/tree/e6c9c75bea7575a64279072dcdf0f6050fef9e9f): 최신 공개 입력과 데이터 카드
- [고정 participant instructions](https://github.com/oracle-samples/gsm-sem/blob/971262d356c1e7fc2da534eeb9d2c828ade42157/docsem/PARTICIPANT_INSTRUCTIONS.md): 입력·출력·근거 및 추론 경계
- [고정 changelog](https://github.com/oracle-samples/gsm-sem/blob/971262d356c1e7fc2da534eeb9d2c828ade42157/docsem/CHANGELOG.md): canonical 정정 이력
- [포털](https://amitbcp-docsem-docinsights.hf.space/): 현재 제출 창 상태, validation leaderboard와 제출 receipt 확인
- [워크숍](https://docinsights-workshop.github.io/docinsights-2026/shared-task/): held-out 공지, 제출 횟수·피드백·최종 순위 정책과 대회 일정
