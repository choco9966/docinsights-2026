# 전체 split 풀이 기록 설계

Issue: #24. 사용자 요청에 따라 자율 실행한다. 기존 #23 작업은 원 checkout에 보존하고 feature/24 worktree에서 독립 실행한다.

## 목표와 순서

먼저 validation 24차 이력의 일반적 실패 원인을 감사하고 validation 1건을 공개 입력만으로 풀어 고정한 뒤 v24-sep03-check-05와 비교한다. 이어 Held-out 1730건 → Train 908건 → Validation 217건의 순서로 전체 기록을 생성한다. 각 split의 정확한 수와 SHA-256은 실제 manifest에서 검증한다. 미완료 split을 완료로 표시하거나 순서를 바꾸지 않는다.

## 대안과 선택

기존 제출에 설명을 사후 부착하면 정답에 맞춘 풀이가 될 수 있다. 기존 bNN 한 개 전용 solver를 확장하면 #23 실험과 결합된다. 현재 공개 문서·질문만 받는 독립 생성 경로와 생성 후 평가 경로를 만들고 기존 OCR/데이터 도구만 재사용한다.

## 출력

로컬 gitignored artifacts/solution-records/issue24 아래에 split별 solutions.jsonl, solutions.md, submission.jsonl, evaluation.json, manifest.json을 둔다. 모델의 raw 응답은 `answer`, `solution`(간결한 근거 설명, 재검산 가능한 계산식), `uncertainties`, `evidence_regions`(page, ocr_anchor)를 생성한다. Runner가 instance_id, split, question, source_pages 및 provenance를 결합하여 primary 기록을 동결한다. Enriched final에는 instance_id, split, question과 동결된 primary의 answer/solution/uncertainties, source_status, evidence(확정 ID 배열), evidence_details(각 region의 판정, ID 또는 명시적 null, 페이지, 원문 전사), source_uncertainties 및 provenance(PDF·입력·설정·관측·실행 해시)를 기록한다. 확정되지 않은 관측 후보는 evidence의 확정 ID 배열에 넣지 않는다. 풀이란 사용자에게 검증 가능한 설명이며 내부 추론 로그를 뜻하지 않는다. 문서에 답이 직접 적힌 경우 calculations는 빈 배열을 허용하며 요약과 인용이 이를 설명해야 한다. 계산이 필요한 경우 마지막 계산 결과는 숫자 답과 일치해야 한다.

## 불변식

1. 입력은 현재 문항의 공개 query/PDF 또는 검증된 OCR뿐이다. 정답·이전 풀이·피드백은 solver 입력에 넣지 않는다.
2. 비교 정답은 생성 파일의 해시를 고정한 뒤 별도 평가에서만 읽는다. Validation v24는 과거 제출 reference이며 organizer gold라고 부르지 않는다.
3. Evidence ID는 문서에 보이는 값을 그대로 보존하며 bNN 또는 한 개로 제한하지 않는다. 인용은 실제 페이지 텍스트에 있어야 한다.
4. 계산식은 허용된 산술 표현만 결정적으로 계산한다. 필요한 반올림은 최상위 `round(expression, places)`로 명시하며 places는 0~12의 정수다. 정확한 유리수 산술 후 HALF_UP을 적용하고 선언한 결과와 수치가 정확히 일치하는지 확인한다. 반올림 요청 또는 사용한 근사 정밀도는 요약에 남기며 암묵적인 허용 오차는 없다. 의미상 모호함·OCR 불확실성·부족한 수치는 숨기지 않는다.
5. 문항 instance_id의 중복·누락·타 split 혼입·변경된 입력을 거부한다. 근거 미확정을 명시한 evidence_details의 null ID는 문항 누락과 구분하며 유효하다. 중단과 재개는 입력 및 설정 해시가 일치할 때만 허용한다.
6. 공개 Git에는 Validation/Held-out 문항별 답·근거·계산을 넣지 않는다. 이슈·PR은 집계와 절차만 포함한다.
7. 외부 포털 제출·PR 병합·기존 작업 변경은 하지 않는다. 실행 실패는 실패로 기록하고 출력이 없는 작업을 완료로 표시하지 않는다.

8. 초기 공개 OCR 선호 이후 사용자가 10시간 비교와 최고 성능 모델 선정을 지시했다. 최신 지시에 따라 Apple Vision도 공개 RapidOCR/PP-OCRv5와 같은 조건의 비교 후보로 허용한다. 최종 OCR은 정확도 중심 비교 결과로 선정하고 코드/가중치 공개 여부·라이선스·버전·실행 출처를 명시한다. 상세 비교 계획은 2026-09-07-ocr-comparison.md를 따른다.

## 공개 OCR과 독립적인 원본 근거 추출

2026-09-07 세 번째 pilot에서도 두 문항이 검증에 실패했다. OCR이 제목을 누락하면 본문이 앞 블록에 잘못 연결됐고, 모델이 반환한 전체 페이지 좌표는 내부 이미지 축소 때문에 실제 픽셀 좌표와 달랐다. 또한 OCR이 훼손한 인용문을 원본 픽셀 전사와 동시에 정확히 일치시키는 요구는 성립하지 않는다. 이 진단에 따라 앞선 heading alias 설계는 폐기하고 원본 위치 선택과 최종 인용을 분리한다. 이전 실패 산출물은 그대로 보존한다.

선택한 OCR의 모든 페이지 원문과 bbox를 먼저 동결한다. Primary solver는 질문·OCR·페이지 이미지에서 답, 풀이, 불확실성, `evidence_regions`를 생성한다. 답 입력은 명확히 읽히는 PDF 이미지 값을 우선하고 OCR와 다르면 그 차이를 불확실성에 기록한다. 원래 OCR와 anchor 문자열은 변경하지 않는다. 이미지가 판독 불가하면 수치를 만들어내지 않는다. 각 region은 `page`와 해당 페이지 OCR에서 정확히 한 번 나타나는 `ocr_anchor`를 갖는다. Anchor의 정확한 문자열 위치를 포함하는 유일한 순서 있는 OCR line-index 범위를 구하고, 동결된 geometry에서 union bbox를 재계산하여 증명에 묶는다. Solver가 좌표를 제공하지 않는다. Anchor는 원본 위치를 찾기 위한 문자열이며 정답 근거나 최종 인용으로 주장하지 않는다. 수치 답은 단위 없이 숫자 문자열로 반환하고 단위는 풀이 요약에 남긴다.

Primary 응답을 해시 고정한 다음 공개 PDF의 해당 페이지를 같은 renderer와 설정으로 무손실 재렌더하고 기존 `ocr_image_sha256`과 일치함을 확인한다. Anchor의 OCR line bbox를 기준으로 결정적인 전체 폭 context crop과 anchor-only crop을 만든다. 별도 fresh checker에는 무손실 전체 페이지, context crop, anchor crop만 제공한다. 질문·답·풀이·anchor 문자열·예상 인용·후보 ID·OCR ID·오류는 전달하지 않는다.

Checker는 anchor가 포함된 보이는 블록의 heading line과 본문을 전사하고 ID·본문의 판독 가능 여부를 명시한다. 모델의 픽셀 좌표는 요구하지 않는다. 코드가 heading line의 본문 구분 colon을 분리하며, 내부 구두점과 ID의 대소문자는 보존한다. 명확한 단 하나의 블록만 확정 근거로 인정한다. 최종 `evidence`와 `evidence_details`는 이 독립 전사에서 채우므로 OCR/primary 추측과 달라도 된다. 여러 region의 확정 ID가 중복되면 id/page/quote가 바이트 단위로 동일한 경우에만 결정적으로 합치며, 같은 ID의 다른 페이지 또는 인용은 거부한다. Primary의 답·풀이·계산식·원래 불확실성은 변경하지 않는다. 원시 primary, OCR, checker 관측값과 차이를 provenance에 보존한다.

PDF/renderer/무손실 페이지/context/anchor crop, selector 설정, primary 응답, checker prompt/schema/config/model/executable/raw response를 해시로 묶는다. 최종 근거를 채운 기록도 reference 접근 전에 다시 동결한다. 공개 OCR은 위치 선택과 풀이 입력이고, PDF 픽셀이 최종 출처다. 같은 계열 모델의 독립 호출은 절차적 독립성을 제공하며 사람의 정답 검증이나 완전한 오류 독립성을 뜻하지 않는다.

## 판독 불가와 완료 집계

원래 brief의 불확실성 기록 원칙에 따라 확정할 수 없는 ID를 추측으로 채우지 않는다. `fully_grounded`와 `evidence_unresolved`를 구분하고, 후자는 원본 위치·해시·판독 가능한 본문·관측 후보·`source_uncertainties`를 남기며 확정 ID는 null로 둔다. 도구 실행 자체의 실패는 별도 `runtime_failed`로 기록한다. 사용자의 출력 선호 질문은 진행 중이며, 현재 기본안은 검토 필요를 명시한 전체 기록을 만드는 것이다.

`coverage_complete`는 공식 모든 문항에 실제 답·풀이와 근거 판정 기록이 있음을 뜻한다. 근거 확정 건수, 검토 필요 건수, runtime 실패 건수를 별도로 보고하며 전부 근거가 확정되지 않으면 전체를 정답 검증 완료라고 부르지 않는다. 다음 split의 시작에는 이전 split의 완전한 답·풀이 기록·동결·분리 평가가 필요하다. 유효한 private 답·풀이가 있는 문항 수를 answer_coverage로 별도 집계한다. Primary 형식·산술·anchor 검증 실패는 생성 실패이며 재시도 대상이다. Checker의 timeout·잘못된 응답·출처 파일 누락은 답이 이미 있더라도 runtime_failed이며 terminal 근거 판정을 대신하지 않는다. 동일 설정으로 resume할 때 원래 primary 해시와 모든 입력을 다시 검증하고, 실패한 source check만 제한된 횟수의 fresh 호출로 복구할 수 있다. Primary 답·풀이는 재생성하지 않고 기존 실패 산출물과 새 호출을 모두 보존한다. 실제 판독이 완료된 ambiguous/unreadable 결과에는 이 runtime 재시도를 적용하지 않는다. 이러한 실패가 한 건이라도 있으면 coverage_complete를 충족하지 못하고 다음 split을 열지 못한다. 모든 region이 clear인 기록만 fully_grounded이고, 하나라도 실제 판독 결과가 미확정이면 evidence_unresolved이다. 2026-09-07 확인한 최신 held-out 정책은 `answer: null`, `evidence: []` abstention을 허용하지만 이는 Answer·Joint에서 오답이다. 사용자의 후속 지적에 따라 미확정 근거를 자동으로 null 제출로 바꾸지 않는다. Private 답·풀이와 원본 관측을 보존하고 재검토 대상으로 남긴다. `coverage_complete`와 `submission_ready`를 구분하며 기본 full submission은 모든 문항의 근거가 확정되어야 생성한다. Grounded-only partial artifact와 명시적으로 선택한 abstention export는 별도 모드이며 완전한 풀이 성공으로 부르지 않는다. Test 예외를 Train/Validation 형식에 자동 적용하지 않는다. 제출용 점수와 private 답 비교는 별도로 집계한다. 답이 생성된 검토 필요 행은 답 비교에 포함할 수 있지만 Evidence 비교에서는 미확정으로 표시한다. Runtime 실패에 답을 만들어 넣지 않는다.

## 검증

합성 문서로 다중 Evidence·임의 ID, 인용과 페이지 검증, 산술 검산, ID coverage, 변경 입력 재개 거부, 생성/평가 분리를 검사한다. 실제 validation 1건에서 문서 검토와 v24 비교를 완료하고 순차 실행한다. 최종 cleaner 후 재검증, 독립 code-reviewer와 architect 검토 및 불변식 감사를 통과해야 aggregate goal을 완료한다.
