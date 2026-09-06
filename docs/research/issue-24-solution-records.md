# Issue #24: 전체 split 풀이 기록과 사전 감사

기준일: 2026-09-07. 실행 브랜치 `feature/24`, 격리 작업 경로 `.worktrees/issue24-solutions`. 기존 #23의 작업 트리와 ultragoal은 보존한다.

## 최신 실행 상태

사용자의 후속 지시로 Apple Vision을 다시 비교 대상에 포함했다. 현재 동일한 60페이지에서 Apple Vision, RapidOCR, Native Paddle OCR을 비교 중이며, Held-out의 Evidence ID와 숫자 인식 정확도를 우선하여 선택한다. 아래 공개 OCR 전환 설명은 결정 이력이며 현재 최종 선택을 뜻하지 않는다. 10시간 작업 창은 2026-09-06 15:46:20~2026-09-07 01:46:20 UTC다. v4 실행기의 독립 코드·구조 검토와 관련 86개 테스트는 통과했으나 선택 OCR 통합, 동일 validation 사전 문항 재검증과 전체 2855건 생성은 아직 완료하지 않았다. 비교 프로토콜은 `docs/superpowers/plans/2026-09-07-ocr-comparison.md`에 고정했다.

## 프로젝트와 데이터

DocSem은 PDF의 시나리오를 골라 정량적 질문을 풀고 직접 뒷받침하는 Evidence ID를 반환하는 과제다. Train 908건, Validation 217건, Held-out 1730건으로 전체 2855건이다. 공개 라벨은 Train만 존재한다. 현재 저장소에는 데이터 다운로드/분석, OCR, blind review, 제출 검증 도구가 있고 #23은 별도 Train 전략 실험이다.

공식 held-out은 HF revision `d9e1a394b46d2ac0a4dd87e12dd4a917a69f46e2`, release `docsem-test-a4205880-r1`로 공개되어 있다. Test manifest SHA-256은 `5fe8fbb8169b0c2b396fe155d263db36f4fa34b02a0cedd9075423b0bd3fc40d`다. 기존 README의 미공개 상태는 오래된 정보다. 제출 상태와 시도 정책은 [공식 workshop 공지](https://docinsights-workshop.github.io/docinsights-2026/shared-task/)를 기준으로 한다. 이 작업은 포털에 제출하지 않는다.

공식 자료: [고정 release metadata](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data/resolve/d9e1a394b46d2ac0a4dd87e12dd4a917a69f46e2/test/release.json), [고정 instructions](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data/resolve/d9e1a394b46d2ac0a4dd87e12dd4a917a69f46e2/INSTRUCTIONS.md), [Issue #5](https://github.com/choco9966/docinsights-2026/issues/5), [Issue #19](https://github.com/choco9966/docinsights-2026/issues/19), [Issue #21](https://github.com/choco9966/docinsights-2026/issues/21), [Issue #23](https://github.com/choco9966/docinsights-2026/issues/23).

## Validation 24차 이력의 해석

| 버전 | 기록된 Answer accuracy | 접근/실패 요약 |
| --- | --- | --- |
| v1 | 0.97235 | 최초 PDF+query blind 풀이 |
| v2 | 0.990783 | 포털 차분 진단 |
| v3 | 0.990783 | 포털 차분 진단, 이전 후보 수정 |
| v4 | 0.981567 | 포털 차분 진단 |
| v5 | 0.990783 | 포털 차분 진단 |
| v6 | 0.990783 | 포털 차분 진단 |
| v7 | 0.995392 | 포털 차분 진단 |
| v8 | 0.990783 | 포털 차분 진단 |
| v9 | 0.990783 | 포털 차분 진단 |
| v10 | 0.990783 | 포털 차분 진단 |
| v11 | 0.990783 | 포털 차분 진단 |
| v12 | 1.0 | 사후 source-label recovery; clean benchmark 아님 |
| v13 | 미복구 | 점수/receipt/산출물 확인 불가 |
| v14 | 미복구 | 로컬 probe 산출물만 보존 |
| v15 | 미복구 | 로컬 probe 산출물만 보존 |
| v16 | 미복구 | 로컬 probe 산출물만 보존 |
| v17 | 미복구 | 점수/receipt/산출물 확인 불가 |
| v18 | 미복구 | 로컬 probe 산출물만 보존 |
| v19 | 214/217 | 9월 3일 재채점 이후 복원 기준 |
| v20 | 213/217 | 단일 행 answer 변경, 폐기 |
| v21 | 215/217 | 단일 행 answer 변경, 유지 |
| v22 | 216/217 | 단일 행 answer 변경, 유지 |
| v23 | 215/217 | 단일 행 answer 변경, 폐기 |
| v24 | 217/217 | 단일 행 answer 변경, 최종 동결 |

v1~v12 집계의 출처는 `experiments/submissions.md`, v19~v24는 `docs/research/issue-21-validation-heldout-strategy.md`와 비공개 복원 audit다. v1~v12와 v19~v24는 서로 다른 정답 revision의 기록일 수 있으므로 하나의 동일 조건 성능 곡선으로 해석하지 않는다. v1~v24라는 24개 버전명이 모두 accepted submission 24회를 뜻하는 것은 아니다. v13~v18의 score/receipt가 미복구여서 실제 accepted 총횟수는 확정하지 않는다.

v24 파일 `v24-sep03-check-05.jsonl`의 SHA-256은 `265d89696e1216f887ea3722c13bf83b3ac5c03292c7460db9afc8732328ec13`이다. 9월 3일 organizer-only 라벨 정정 이후 재채점과 v19~v24의 단일 행 변경/수용 여부는 gitignored `artifacts/submissions/sep03-restoration-audit.json`에 있다. v24의 accepted receipt와 100% 점수 기록은 확인했지만 이번 작업에서 재제출하지 않았다. 원래 공개 기록과 비공개 audit를 함께 보존한다.

## 반복하지 않을 오류

| 실패 유형 | 새 실행의 주의사항 |
| --- | --- |
| 손실·잔액의 주체를 바꿈 | 각 수치의 entity와 requested target을 확인 |
| 하루와 전체 기간의 분모 혼동 | 시간 범위와 단위를 식에 명시 |
| remaining/departed 방향 혼동 | 대상 집합과 빼기의 방향을 확인 |
| 질문의 다중 요청·불완전 조건 | 하나의 answer 계약과 의미적 불확실성을 분리 기록 |
| 숫자·block marker OCR 병합 | 원본 PDF 시각 확인, 임의 보정 금지 |
| 유사 query의 다른 조건·수치 | 현재 문서만 사용, 다른 문항 정답을 복사하지 않음 |
| validation feedback/source-label recovery 누수 | 과거 reference는 생성 동결 후 별도 evaluator에서만 사용 |
| 기존 bNN 한 개 제한 | Test의 token 전체와 구두점을 그대로 보존, 다중 Evidence 허용 |
| 라벨에 맞춘 사후 풀이 | 문서 근거와 식을 먼저 저장·해시 고정한 후 비교 |

## 사전 1건 검증

Validation에서 고정한 1건의 공개 query와 PDF/OCR를 읽고, PDF의 관련 페이지를 렌더해 직접 확인했다. 계산식과 결과를 먼저 저장하고 SHA-256을 고정한 다음 v24의 해당 행을 읽었다. 최종 답과 Evidence 집합이 모두 일치했고 산술 재검산도 통과했다. 이전 이력을 본 세션이므로 이 검사를 blind benchmark라고 부르지 않는다.

문항별 정보는 로컬 `artifacts/solution-records/issue24/preflight/`에만 보존한다. `generation-freeze.json`은 비교 전 출력 해시, `comparison.json`은 비교 시각과 reference 해시 및 일치 여부를 담는다. 이 단계의 성공은 포맷과 한 문항 검증이며 전체 split 완료를 뜻하지 않는다.

## 실행과 완료 조건

별도 worktree의 `.omx/ultragoal/goals.json`은 정확히 세 목표를 가진다. G001은 사전 검증과 Held-out 전체, G002는 Train 전체 및 생성 후 공개 라벨 비교, G003은 Validation 전체 및 생성 후 v24 비교와 최종 검토다. 각 split에 JSONL, Markdown, 제출 projection, manifest와 평가 기록을 만든다. 실제 coverage가 완전하고 검증된 뒤에만 다음 목표로 넘어간다.

## 재현 실행 명령

아래 명령은 `feature/24` worktree에서 실행한다. 실행기는 공개 tasks/PDF만 읽으며 reference 파일 인수를 받지 않는다. 본 실행의 요청 모델은 `gpt-5.6-sol`, reasoning effort는 high로 고정한다. 아래는 실행 순서이며 공개 OCR 환경 설치와 pilot 통과가 선행되어야 한다. Tesseract, Apple Vision, 공개 RapidOCR를 사용한 앞선 세 pilot은 각각 0/2 통과했다. `runs`, `runs-v2`, `runs-v3`의 생성·실패 산출물을 보존한다. v4 출처 계약 구현과 동일 validation 사전 문항 검증을 마친 뒤 새 `runs-v4`에서 pilot을 진행한다. Pilot의 실제 기록·출처 확인이 통과하면 `--limit`을 제거해 재개한다. 아래 명령은 준비 중인 v4 실행 순서를 기술하며 실행 완료를 뜻하지 않는다.

새 환경에는 Python 3.11.6, Poppler의 `pdftoppm`, 인증된 Codex CLI가 필요하다. 현재 Native baseline의 59개 패키지 pin과 공식 모델 준비 명령은 [Native 재현 구성](issue-24-native-ocr-reproduction.md)을 따른다. 실행기는 pinned `ppocr-env` Python을 요구한다. 기존 Rapid 환경과 실패 산출물은 역사적 진단으로 보존한다. 가속 후보가 최종 채택되면 설정을 고정하고 이 명령도 함께 갱신한 뒤 실행한다.

```bash
PYTHONPATH=src data/issue24/ppocr-env/bin/python scripts/run_solution_records.py --split heldout --tasks artifacts/solution-records/issue24/inputs/heldout/tasks.jsonl --pdf-root data/issue24 --output-root artifacts/solution-records/issue24/runs-v4 --limit 2 --workers 2
PYTHONPATH=src data/issue24/ppocr-env/bin/python scripts/run_solution_records.py --split heldout --tasks artifacts/solution-records/issue24/inputs/heldout/tasks.jsonl --pdf-root data/issue24 --output-root artifacts/solution-records/issue24/runs-v4 --workers 4 --resume
PYTHONPATH=src data/issue24/ppocr-env/bin/python scripts/evaluate_solution_records.py --split-dir artifacts/solution-records/issue24/runs-v4/heldout --no-reference
PYTHONPATH=src data/issue24/ppocr-env/bin/python scripts/run_solution_records.py --split train --tasks artifacts/solution-records/issue24/inputs/train/tasks.jsonl --pdf-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem --output-root artifacts/solution-records/issue24/runs-v4 --workers 4
PYTHONPATH=src data/issue24/ppocr-env/bin/python scripts/evaluate_solution_records.py --split-dir artifacts/solution-records/issue24/runs-v4/train --reference /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/train/labels.jsonl --reference-kind train-public-labels
PYTHONPATH=src data/issue24/ppocr-env/bin/python scripts/run_solution_records.py --split validation --tasks artifacts/solution-records/issue24/inputs/validation/tasks.jsonl --pdf-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem --output-root artifacts/solution-records/issue24/runs-v4 --workers 4
PYTHONPATH=src data/issue24/ppocr-env/bin/python scripts/evaluate_solution_records.py --split-dir artifacts/solution-records/issue24/runs-v4/validation --reference /Users/choco/Documents/project/docinsights-2026/artifacts/submissions/v24-sep03-check-05.jsonl --reference-kind validation-v24-reference
```

`--limit 2`는 전체 manifest 중 앞의 2건만 실행하며 전체 완료 표식은 만들지 않는다. 이전 split의 완전한 입력/출력 검증이 없으면 다음 split 실행은 거부된다. 생성 후 비교는 별도 evaluator에서 동결된 export 해시와 reference 해시를 묶어 기록한다. Train 비교가 확인되기 전에는 Validation을 시작하지 않는다. 구형 `docinsights validate-submission`의 bNN 검증은 Train/Validation에만 사용하고, 임의 ID를 가진 Held-out은 이번 source-grounded export 계약으로 검증한다.

독립 검토에서 정규화 비교, 임의 토큰 내부의 콜론, 소수의 정확한 계산, 입력 원문과 출력의 독립 검증 및 split 혼입 차단을 점검한다. 원문 추출은 OCR silver이며 시각적으로 확인하지 않은 값을 human-gold라고 부르지 않는다. 콜론을 포함한 본문 레이블이 블록 경계처럼 보이면 검증이 보수적으로 거부할 수 있으므로 해당 실패는 원본 이미지 재검토 대상으로 보존한다.

## OCR·저장 공간 검증

최초 2건 pilot은 Tesseract 200dpi/PSM6의 숫자·Evidence ID 손상 및 응답 형식 문제로 실패했다. 실패한 두 번의 응답과 원래 입력을 `runs`에 보존한다. 후속 Apple Vision 실험은 기존 어댑터를 사용했지만 두 문항 모두 bbox 검증에서 실패했다. 사용자 결정에 따라 Apple Vision을 본 실행에서 제외했으며 이 실험은 실패 이력으로만 남긴다.

175dpi JPEG quality65의 16페이지 표본은 4,419,336 bytes였다. 전체 16,383페이지를 같은 평균으로 추정하면 약 4.52GB지만 실제 문서별 차이는 있으므로 여유 공간을 실행 중 확인한다. 모든 페이지의 JPEG와 OCR를 보존하고 해시로 묶으며, 공식 원본 PDF를 최종 출처로 유지한다. v4에서는 인용한 페이지의 무손실 PNG와 결정적 crop을 출처 증명용으로 추가 보존하므로 실제 저장량은 위 JPEG 추정치보다 크다. 이 설정도 synthetic ID를 완벽하게 읽는다는 보장은 없으므로 시각 확인과 실패 기록을 유지한다.

Train reference는 생성 전에 라벨 내용을 읽지 않고 [고정 HF revision의 파일 메타데이터](https://huggingface.co/api/datasets/amitbcp/docinsights-2026-shared-task-data/tree/d9e1a394b46d2ac0a4dd87e12dd4a917a69f46e2/train?recursive=false)로 고정했다: `train/labels.jsonl`, 62,030 bytes, Git blob SHA-1 `686429d03b2d4ba5fe4fe6b07398feef7d0cd884`. 생성 동결 후 evaluator가 실제 바이트의 이 메타데이터와 SHA-256을 검증한다. Held-out 평가 파일은 라벨 부재를 `reference_unavailable`로 명시하고 정확도를 만들지 않는다.

## 사용자 후속 결정: 공개 OCR

논문·과제의 재현성을 위해 사용자가 공개 OCR 사용을 명시했다. 이에 따라 Apple Vision을 본 실행에서 제외한다. 이전 Vision 검토는 진단 이력으로 남기며 최종 방법으로 제시하지 않는다. `runs-v2`의 두 문항은 모델 호출 전에 기존 Apple Vision 어댑터가 page 8 watermark의 이미지 바깥 bbox를 거부해 실패했다. 새로 생성한 풀이가 없고 원래 실패 파일은 보존했다.

기존 저장소의 공개 PP-OCRv5 mobile 어댑터와 고정 detector/English recognizer 가중치를 먼저 검증한다. 모델 코드·가중치 라이선스, 버전, 해시와 전처리를 기록하고 실제 source-grounding pilot을 통과한 설정을 `runs-v3`에 고정한다. 공개 모델 설치 또는 smoke 성공만으로 전수 OCR 정확도를 주장하지 않는다.

공개 OCR 후보의 공식 라이선스와 가중치 접근성을 확인했다. [PP-OCRv5 mobile detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det/tree/0d63e78e2b680928f6b1747d76a08db6e645efb7)와 [English mobile recognizer](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec/tree/267c36e24c331595590fe7bd72bde2436fd286f2)의 고정 모델 카드에는 Apache-2.0이 명시되어 있고 공개 추론 가중치를 제공한다. 코드 인용은 upstream이 안내하는 [PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595)를 사용한다. 실제 환경의 패키지 버전과 모델 디렉터리 해시를 함께 기록해야 하며, 모델 카드의 설치 예제를 모든 버전 조합의 호환성 보장으로 해석하지 않는다.

추론 LLM의 기본 모델명은 별도 합성 입력 1회에서 CLI 헤더의 `gpt-5.6-sol`로 확인했다. 이후 본 실행에서는 이 요청 모델명을 명시하고 high effort를 사용한다. 과거 `unresolved-default` 기록을 사후 변경하지 않는다. 공개 OCR 선택은 전체 시스템의 LLM 가중치까지 공개된다는 의미가 아니며, 방법 설명에서 OCR과 답 생성 모델을 각각 명시한다.

### 공개 ONNX 실행 후보

Native Paddle CPU 표본은 warm 상태에서도 두 페이지 평균 약 30.56초가 걸렸다. 이 두 페이지를 기준으로 한 단순 직렬 외삽은 전체 OCR에 약 139시간이며 실제 전체 측정값은 아니다. 이에 따라 `rapidocr==3.9.2`, `onnxruntime==1.23.2`의 CPU 실행을 별도 공개 PDF 표본으로 확인한다. [ONNX Runtime 공식 문서](https://onnxruntime.ai/docs/get-started/with-python.html)는 macOS와 Arm CPU에 CPU 패키지를 안내한다.

RapidOCR의 [고정 모델 manifest](https://raw.githubusercontent.com/RapidAI/RapidOCR/095232a4c94f7f0e6600ba5bba1177010ad696d4/python/rapidocr/default_models.yaml)에 있는 공개 파일을 사용한다. Detector `ch_PP-OCRv5_det_mobile.onnx`의 SHA-256은 `4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae`, recognizer `en_PP-OCRv5_rec_mobile.onnx`는 `c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8`이다. [RapidOCR 코드](https://github.com/RapidAI/RapidOCR/blob/v3.9.2/LICENSE)와 [모델 저장소](https://www.modelscope.cn/api/v1/models/RapidAI/RapidOCR?Revision=v3.9.2)는 Apache-2.0을 명시한다.

이 파일들은 RapidAI가 배포하는 PP-OCRv5 계열 ONNX 변환본이다. 위 HF revision의 Paddle 가중치와 정확히 같은 변환 계보라는 증거는 없으므로 동일 가중치라고 주장하지 않는다. 실제 사용한 배포 URL·체크섬·런타임·전처리를 기록하고 PaddleOCR와 RapidOCR를 함께 인용한다. 설치 성공만으로 속도나 품질을 주장하지 않는다.

2026-09-07 로컬 probe에서 초기화 0.713초, 공개 Held-out 표본 두 페이지는 각각 4.648초·7.396초, 기존 validation 표본 페이지는 4.867초였다. 동일 warm 재실행의 OCR 텍스트·geometry가 일치했다. 세 페이지의 측정 결과이며 전체 정확도나 전체 처리시간을 뜻하지 않는다. 압축 JPEG와 무손실 PNG의 인식 결과는 같지 않았으므로 OCR에는 175dpi 임시 PNG를 사용하고, solver 및 사람이 확인할 JPEG quality65는 같은 페이지 크기로 보존한다. 원본 PDF·전처리·PNG 해시·OCR 원문과 bbox·JPEG 해시를 함께 고정한다. 임시 PNG는 OCR 후 제거하므로 재현에는 고정 PDF와 렌더 설정을 사용한다.

### 폐기한 v3의 Evidence ID 대조

OCR 원문에 없는 Evidence ID를 solver가 반환한 경우, 해당 인용이 선언한 페이지의 정확히 한 OCR block에 존재할 때만 별도 시각 대조를 허용한다. 먼저 solver 응답을 동결하고, 새 checker에는 전체 페이지와 OCR heading bbox로 결정한 crop만 제공한다. 질문·정답·인용문·후보 ID·OCR ID는 checker prompt에 넣지 않는다. 유일하고 명확한 heading이 동결된 ID와 문자 단위로 일치해야 인정한다. OCR 본문이나 숫자는 수정하지 않으며 불명확하거나 다른 ID이면 실패로 보존한다.

페이지·crop·설정·원시 응답 해시와 판정 근거를 함께 남긴다. 동일 계열 모델의 별도 호출에 의한 절차적 분리이며 독립된 정답 라벨은 아니다. 따라서 OCR은 공개 모델이고 답 생성 및 필요한 시각 대조는 별도 LLM이라는 시스템 구성을 명시한다.


## v3 실패에서 수정한 출처 계약

공개 RapidOCR를 적용한 `runs-v3`도 2건 모두 시각 검증에서 멈췄다. 첫 문항은 모델이 반환한 전체 페이지 bbox가 실제 좌표와 달랐고 본문 구분 colon을 ID에 포함했다. 두 번째는 OCR이 제목을 누락해 대상 인용이 앞의 다른 블록에 붙었다. 두 primary 응답에는 계산식이 있으면서 답에 단위를 포함한 형식 오류도 있었으며, 이후 prompt는 수치만 답에 쓰고 단위는 요약에 쓰도록 수정했다. 오류 재현에 사용한 실제 정답 수치는 미래 prompt 예제로 넣지 않는다.

독립 진단에서 첫 문항의 동일 영역을 무손실로 렌더했을 때 PNG 해시가 원래 OCR 입력과 일치했고, fresh checker의 판독은 `clear`였다. JPEG에서의 `ambiguous` 판정만으로 원본 자체가 판독 불가라고 결론 내리지 않는다. 이 사례에서 primary의 ID 추측은 별도 checker의 전사와 달랐다. 어느 관측도 organizer gold로 간주하지 않는다.

v4에서는 공개 OCR를 **위치 탐색용 관측값**으로 유지하고, primary의 `evidence_regions`는 페이지와 유일한 OCR anchor만 제공한다. 답·풀이·계산식을 먼저 동결한 뒤 checker가 무손실 전체 페이지와 위치 crop만 보고 제목·본문을 독립적으로 전사한다. 최종 ID와 인용은 그 전사에서 채우며 primary 답·풀이를 변경하지 않는다. 예상 ID·anchor 문자열·답·질문은 checker에 주지 않는다. 원래 OCR를 수정하지 않고 모든 관측·차이·해시를 보존한다. 상세 계약은 `docs/superpowers/specs/2026-09-06-solution-records-design.md`를 따른다.

완전한 답·풀이 coverage와 근거 자동검증 통과 건수를 분리한다. 판독 불가 근거는 null ID와 검토 필요 사유로 남기며 확정된 근거처럼 제출하지 않는다. Runtime 실패로 답이 없는 문항은 coverage 완료에 포함하지 않고 다음 split을 열지 못한다. 이 문서를 갱신한 시점에는 v4 구현·검증이 진행 중이며 전체 split 생성은 완료되지 않았다.


## 실행 환경과 최신 Held-out 정책 확인 (2026-09-07)

[공식 참가 안내](https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md)는 Mac 또는 특정 운영체제를 제한하지 않으며 오픈소스 OCR만 사용하라는 조항도 없다. 이는 특정 도구의 조직위 개별 승인 확인과는 구분한다. Mac에서 공개 RapidOCR/ONNX Runtime을 실행하는 구성은 Apple Vision 사용과 별개다. 공개 OCR 선호 이후 사용자가 가장 높은 성능을 기준으로 비교·선택하도록 지시했으므로 Apple Vision도 다시 평가한다. 선택 결과와 OS·하드웨어·패키지·모델 체크섬·전처리·답 생성 모델·프롬프트를 각각 문서화한다. 별도 과제의 도구 제한은 이 대회 규정만으로 판단할 수 없다.

[공식 Held-out 정책](https://docinsights-workshop.github.io/docinsights-2026/shared-task/)은 1,730건 기준 partial 제출과 `answer: null`, `evidence: []` abstention을 허용하며 누락은 오답으로 처리한다. Evidence 미확정 행의 실제 답·풀이·관측값은 private corpus에 보존한다. 사용자가 null의 오답 처리를 지적하여 자동 abstention 기본안은 철회했다. 근거 재확인 대상으로 남기고 기본 full submission은 모든 근거가 확정된 경우에만 준비 완료로 표시한다. 명시적으로 선택한 partial/abstention export는 별도 형식이다. Runtime 실패로 답이 없는 문항을 전체 풀이 coverage 완료로 간주하지 않는다. Train/Validation에는 test 예외를 자동 적용하지 않는다.

현재 안내의 제출 제한은 계정당 accepted attempt 최대 3회, distinct attempt 사이 6시간이며 첫 제출만 점수를 공개한다. 마감은 2026-09-11 12:00 UTC이다. 포털 제출은 이 기록 생성 작업과 별도의 외부 동작이며 아직 실행하지 않았다.


## 최신 v4 실행 검증과 저장 계획 (2026-09-07 KST)

OCR 선택과 분리하여 기존 RapidOCR 구성으로 같은 validation `task_000920` 한 건을 새 v4 실행기에 통과시켰다. 질문·풀이 요약·검증 가능한 계산식·답·Evidence와 원본 이미지 증명을 저장했고, `generation-freeze.json`의 2026-09-06 16:46:02.794135 UTC 동결 뒤 16:46:17.013131 UTC에 이미 허용된 v24 한 행과 비교했다. Answer와 Evidence가 모두 일치하며 `fully_grounded=1`, unresolved=0이다. 이는 알려진 한 문항의 실행 계약 확인이며 독립 validation 정확도나 최종 OCR 선택 결과가 아니다. 최종 선택 구성이 달라지면 같은 한 문항으로 다시 확인한다. 산출물은 비공개 `artifacts/solution-records/issue24/diagnostics/v4-contract-smoke-rapid/smoke/validation/`에 보존한다.

60페이지의 동일 렌더에서 JPEG 크기를 측정하고 실제 split 페이지 수로 가중한 결과, quality65 전체 JPEG는 약 4.58GB, quality85는 약 6.63GB로 추정된다. 문항당 무손실 근거 페이지 한 장만 추가해도 약 4.55GB가 더 필요하며, 여러 근거·crop·실행 기록은 별도다. 현재 quality65는 저장량을 고려한 설정이고 quality85보다 답 정확도가 낮거나 높다는 실측 근거는 아직 없다. OCR 자체와 독립 Evidence checker는 모두 무손실 PNG를 사용한다.

재생성 가능한 uv 패키지 캐시를 정리한 뒤 사용 가능한 공간은 13,555,679,232 bytes였다. 설치된 환경, PDF와 실험 산출물은 유지했다. 최소 1GiB 여유를 확인하는 실행 guard를 유지하고 실제 증가량으로 잔여 용량을 갱신한다. 전체 결과를 보존할 공간이 부족하면 기존 결과를 삭제하거나 완주한 것으로 처리하지 않는다.


## OCR 비교 후 최신 선택

60페이지/독립 이미지 전사120회 비교를 완료했고 [실측 성능표와 선택 근거](issue-24-ocr-comparison.md), [상세 CSV](issue-24-ocr-comparison.csv)를 작성했다. Held-out 합의 표본에서 Native Paddle PP-OCRv5의 ID F1 77.21%, CER 3.91%가 가장 좋아 본 실행 후보로 선택했다. 숫자 F1은 Native91.77%와 Apple91.97%의 차이가 불확실하다. 전체 정답률·Evidence set 정확도·대회 순위로 해석하지 않는다. 속도가 느려 동일 품질 가속 가능성을 별도로 검증하며, 검증 전에는 가속 후보에 Native 점수를 적용하지 않는다. 위 공개 OCR/Apple 제외·재허용 문단은 결정 변화의 과거 이력으로 유지한다.
