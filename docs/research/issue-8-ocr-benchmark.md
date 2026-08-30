# Issue #8 · Qwen용 OCR 전용 파이프라인 검증

## 연구 질문

DocSem의 이미지형 PDF에서 수학 추론을 수행하지 않는 작은 OCR 엔진이 visible block ID와 정량 구절을 얼마나 충실하게 복원하며, 그 결과를 작은 Qwen의 추론·RL 입력으로 사용할 수 있는지를 검증합니다. 이 이슈는 Qwen의 답안 생성·학습과 분리된 입력 계층이며 기존 모델 답변, 제출 파일과 포털 점수를 OCR 모델 선택이나 평가에 사용하지 않습니다.

## 데이터 계약

- 입력은 고정 revision의 Validation `tasks.jsonl`과 217개 PDF뿐입니다.
- Validation PDF는 217개 모두 2페이지이므로 전체 입력은 434페이지입니다.
- 샘플과 전수 정적 검사에서 usable native text layer가 발견되지 않았으므로 페이지 렌더링과 OCR이 필요합니다.
- OCR 엔진에는 `user_query`, 정답, evidence label을 전달하지 않습니다.
- `user_query`는 OCR 이후 Qwen 입력 레코드를 조립할 때만 결합합니다.
- 출력에는 `answer`와 `evidence` 필드를 두지 않고 ordered `bNN` blocks, 원문, 위치, confidence, 입력·설정 SHA만 기록합니다.

## 후보

| 후보 | 역할 | 실행 환경 | 선택 이유 |
| --- | --- | --- | --- |
| Tesseract 5 `eng` | 재현 가능한 고전 baseline | CPU | TSV 좌표·confidence를 제공하고 설치가 단순함 |
| Apple Vision `VNRecognizeTextRequest` | macOS neural baseline | Apple Silicon | OS 내장 neural OCR이며 line bbox와 confidence를 제공함 |
| PP-OCRv5 mobile detector + English mobile recognizer | portable neural challenger | macOS·Kaggle·Colab CPU | 약 13MB의 고정 모델로 line bbox·confidence를 제공하고 Mac 스모크에서 숫자 자릿수를 Tesseract보다 안정적으로 보존함 |
| LightOnOCR-2-1B | hard-page 생성형 challenger | Apple MPS | 공식 Transformers MPS 경로가 있으나 약 2.01GB weight와 FP32 실행 자원이 필요함 |
| GLM-OCR 8-bit | hard-page 생성형 challenger | Apple MLX | 약 1.58GB의 MLX 변환 모델로 Mac Studio에 적합하지만 생성형 누락·환각을 별도로 검증해야 함 |
| Falcon-OCR | Linux GPU 전용 후보 | Linux NVIDIA | 공식 구현이 Triton JIT와 FlexAttention에 의존해 macOS arm64에서는 import 단계부터 실행되지 않음 |

Tesseract와 Apple Vision을 전수 비교한 뒤 PP-OCRv5 mobile을 동일 계약의 CPU challenger로 추가했습니다. LightOnOCR와 GLM-OCR은 Mac Studio 연결 후 고정 6문서 smoke set을 먼저 통과해야 하며, 설치 성공만으로 canonical extractor를 변경하지 않습니다.

## 실행 환경과 고정 산출물

이번 Validation benchmark는 제공 예정인 Mac Studio가 아니라 `Mac15,13`, Apple Silicon `arm64`, unified memory 16GiB, 물리·논리 CPU 8코어 환경에서 macOS 14.6으로 실행했습니다. Qwen 학습용 Mac Studio 자원 gate는 Issue #7에서 별도로 측정하며 이 OCR 시간·메모리 결과를 Mac Studio 수치로 해석하지 않습니다.

| 항목 | 값 |
| --- | --- |
| Dataset revision | Hugging Face cache revision `b171c5ad488f0c8c50df05951a5b288ea50e9501` |
| OCR 코드 commit | `6b787146663ccb4f907e7fccb6119730178bd768` |
| Python / uv | Python 3.12.12 / uv 0.9.7 |
| Tesseract / Leptonica | 5.5.3 / 1.87.0 |
| Poppler `pdftoppm` | 26.05.0 |
| Swift | 5.10, target `arm64-apple-darwin23.6.0` |
| 공통 설정 | 200 DPI, 영어, 문서별 120초 timeout, 엔진별 순차 실행 |

| 산출물 | SHA-256 |
| --- | --- |
| `uv.lock` | `6327d68cd763b315e8feb74029f65bff9da87655bb377687410b59be1449d994` |
| Tesseract config | `0c056b626066005dcf3de8a21524666d9e669f34db172692f78a216e53134be6` |
| Apple Vision config | `a780c04cb22be1261be4959a659173fb6dcd616b5f1203ad2950b5b6ab92afce` |
| JSON schema | `f34d4d05ec1229ada9ad6c6c521a5a4df17eaf1d189047986e5f4cf117b5ee5e` |
| Validation manifest | `87afbb83968c597398ca94c1ffc5940f6322c4381319f2b8f154cbf99789abe9` |
| Tesseract executable | `3f4357a4b0d7fa6ae7c3ee72db6845f6bef179ca7398f4ac13af0d44953f7125` |
| Poppler renderer executable | `de772e88ab9977ccde25def9b403bf42675d75f5dd82b19fbd7d8123ad183159` |
| Apple Vision executable | `e9976ba75650c6d9ec136d4f5df7f0bb99c29592649feb9e9aeedfba211f612b` |

## 평가 층위

1. Operational coverage: 217/217 문서와 434/434 페이지 성공, 빈 출력·crash·timeout 수, deterministic rerun SHA를 측정합니다.
2. Block structure: ordered block-ID sequence exact, block-ID precision·recall·F1, missing·hallucinated·reordered block 수를 측정합니다.
3. OCR fidelity: 사람이 전사한 sealed 표본에서 strict CER/WER를 측정하며 대소문자, 문장부호, 통화기호와 소수점을 유지합니다.
4. Critical tokens: 숫자, 부호, 소수점, 통화, 단위와 quantity tuple exact를 측정합니다.
5. Cross-engine disagreement: 두 엔진의 block text 또는 critical token이 다르면 자동 정답을 선택하지 않고 visual review queue로 보냅니다.
6. Qwen input coverage: reasoning 없이 Qwen이 소비할 ordered block JSONL이 전 문서에서 생성되는지를 측정합니다.

## Gold와 누수 방지

Validation에는 공개 OCR 전사가 없으므로 엔진 합의를 gold로 간주하지 않습니다. 모델 출력과 무관한 seeded manifest로 OCR-dev 30개와 OCR-eval 187개를 고정했지만, 설정 보정에 사용한 `task_000909`가 manifest상 OCR-eval에 포함된 사실을 사후 확인했습니다. 따라서 이번 217개 전수 실행은 sealed 또는 unbiased 평가가 아니라 exploratory coverage·cross-engine agreement 실험으로만 보고합니다. 향후 정확도 수치는 접촉 문서 ledger를 먼저 동결하고, 설정 선택에 사용하지 않은 PDF 이미지만 보고 사람이 만든 exact transcription에 대해서만 산출합니다.

Gold annotator에게는 candidate OCR 출력, 기존 모델의 답·풀이·검수 결과, 제출 파일과 포털 점수를 제공하지 않습니다. 독립된 사람이 없으면 산출물에 `model-assisted-review`라고 표시하고 unbiased human gold라고 주장하지 않습니다.

## 접촉 문서 ledger

| 범위 | 접촉 내용 | 이번 보고서의 처리 |
| --- | --- | --- |
| `task_000909` | Tesseract PSM 선택, Apple 행 군집화와 watermark 제거 preflight에 사용 | OCR-eval에서 제외해 다시 계산하더라도 이미 설정 선택에 영향을 줬으므로 holdout으로 주장하지 않음 |
| Validation의 기존 검수 사례 | OCR 이슈 이전의 답안 검수 과정에서 일부 PDF·query·후보 답이 대화와 작업 공간에 노출됨 | 217개 전체를 exploratory coverage·agreement로만 보고하고 accuracy·sealed 성능을 주장하지 않음 |
| 향후 주최 측 held-out test | 현재 미공개이며 OCR 설정 선택에 사용하지 않음 | Issue #8 extractor와 설정을 먼저 동결한 뒤 label-free coverage 검사에만 한 번 사용 |

이번 실험에는 unbiased human transcription이 없고 기존 Validation 접촉 범위도 완전하게 복원할 수 없으므로, 임의로 untouched subset을 만들지 않습니다. 새 정확도 평가는 별도의 독립 annotator와 사전 동결한 접촉 ledger가 확보됐을 때만 수행합니다.

## 예비 측정

2026-08-30에 `task_000909` 첫 페이지를 200 DPI로 렌더링해 실행 경로를 확인했습니다.

| 엔진·설정 | 경과 시간 | visible block ID | 핵심 `b09` 숫자 |
| --- | ---: | --- | --- |
| Tesseract 5.5.3, PSM 3 | 3.411초 | `b01`~`b13` 전부 복원 | `10`, `100`, `3` 복원 |
| Tesseract 5.5.3, PSM 6 | 3.252초 | `b01`~`b13` 전부 복원 | `10`, `100`, `3` 복원 |
| Tesseract 5.5.3, PSM 11 | 3.158초 | `b01`~`b13` 전부 복원 | `10`, `100`, `3` 복원 |
| Apple Vision accurate | 0.951초 | `b01`~`b13` 전부 복원 | `10`, `100`, `3` 복원 |

이 결과는 단일 페이지 preflight일 뿐 정확도 우위를 의미하지 않습니다. 또한 해당 문서가 사후에 OCR-eval 소속으로 확인됐으므로 이번 전수 비교에서 untouched holdout이나 설정 선택 근거로 취급하지 않습니다.

같은 페이지의 `b04` 표에서 더 중요한 구조 오류가 확인됐습니다. Tesseract PSM 3은 실제 셀 값 `13`을 `1 3`으로 분리했고, Apple Vision의 원시 관측값은 같은 행 셀의 top 좌표가 3~4px 다른 탓에 열 순서를 섞었습니다. Tesseract를 PSM 6으로 고정하고 Apple 관측값을 bbox 높이의 50% 허용범위로 행 군집화한 뒤 두 엔진 모두 `Team On-call staff Next check-in Roads 13 09:30 Parks 5 10:15 Communications 5 11:00`을 같은 순서로 복원했습니다.

이 보정 뒤 `task_000909` 2페이지 전체에서 두 엔진의 block-ID F1과 critical-token F1은 1.0, 상호 CER은 0.00744, WER은 0.01601이었습니다. 이는 사람이 전사한 gold에 대한 정확도가 아니라 한 문서의 cross-engine agreement이므로 모델 선택 근거로 단독 사용하지 않습니다.

## Exploratory operational 결과

이번 전수 실행은 정확도 평가가 아니라 217개 Validation 문서에서 OCR 실행 경로, 출력 계약과 운영 안정성을 확인하는 exploratory benchmark입니다. Apple Vision과 수정된 Tesseract 수치는 코드 commit `6b787146663ccb4f907e7fccb6119730178bd768` 기준 전수 실행에서 산출했습니다.

| 엔진·설정 | 문서 | 페이지 | blocks | 실패 | wall time | 문서당 평균 | 중앙값 | p95 | 최대 | process max RSS | OCR aggregate hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Apple Vision accurate, 200 DPI | 217 | 434 | 4,991 | 0 | 842.06초 | 3.878133초 | 3.842681초 | 4.098554초 | 5.241440초 | 224,903,168B | `63767f0736c441a5f59269082231588248325596a203787faa34787827373cb7` |
| Tesseract 5.5.3 `eng`, PSM 6, 200 DPI | 217 | 434 | 4,991 | 0 | 1,310.04초 | 6.033493초 | 5.986718초 | 6.342317초 | 7.046745초 | 112,754,688B | `15ceb5f26d42e9c7556c7c5e25b2267a4ce3344bae4282c3aec37b819beee0be` |

표의 p95와 최대 시간은 문서 단위 `total_seconds`입니다. 모든 문서가 2페이지이지만 이를 2로 나눈 값을 실제 페이지별 p95로 주장하지 않습니다. `process max RSS`는 `/usr/bin/time -l`의 maximum resident set size이며 Mac 전체 unified memory 사용량이 아닙니다. 엔진 confidence는 Tesseract의 평균 word confidence와 Apple Vision observation confidence처럼 의미와 보정 상태가 다르므로 엔진 간 품질 비교에 직접 사용하지 않습니다.

## Cross-engine agreement 해석

| 비교 방향 | 문서 | mean CER | mean WER | mean block F1 | ordered block exact | exact-token F1 | ordered-quantity F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 수정된 Tesseract를 reference로, Apple Vision을 prediction으로 비교 | 217 | 0.00665542 | 0.01431542 | 1.0 | 217/217 | 0.98020314 | 0.97680491 |

이 비교는 accuracy가 아니라 agreement 진단입니다. 구현은 첫 번째 파일의 텍스트를 reference denominator로 사용해 CER·WER를 계산하므로 비교 방향을 바꾸면 값이 달라질 수 있고, aggregate는 문서별 지표의 단순 평균입니다. 어느 엔진도 human gold가 아니므로 높은 agreement는 두 엔진이 같은 오류를 내지 않았다는 증거가 아니며, 낮은 agreement도 어느 쪽이 맞는지를 결정하지 않습니다. 최종 비교에는 reference와 prediction의 파일명·aggregate hash를 함께 기록하고, critical-token 또는 block 순서가 다른 문서는 `ocr_uncertain` review queue로 보냅니다.

## Model-assisted visual findings

독립 human transcription을 확보하지 못해 아래 내용은 `model-assisted-review`입니다. accuracy 수치나 unbiased gold 판정으로 사용하지 않고, 실패 모드와 보수적 후처리 규칙을 정하는 진단 근거로만 사용합니다.

- Tesseract pre-fix 출력에서는 32개 문서의 33개 block 경계에서 숫자 `0`을 문자 `O`로 읽은 `bO5:`·`bO6.` 계열 marker 오류가 확인됐습니다. 수정기는 줄 시작, 단일 후속 숫자, marker 종결 구두점이 모두 있는 경우에만 `O`를 `0`으로 바꾸며 본문 안의 `bO5`, 다자리 `bO10:`과 소수형 `bO5.2`는 변경하지 않습니다. 이 deterministic repair는 원문 line을 수정하지 않고 block 경계 인식에만 적용합니다.
- Apple Vision은 표의 같은 행에 있는 숫자 cell을 bbox top 값 차이 때문에 잘못 배열하거나, `¥`를 `*`, `#`, 점 문자 또는 빈 문자열로 읽고, 페이지 footer를 본문 continuation으로 포함하는 사례가 확인됐습니다. 행 군집화와 명시적·반복 margin artifact 제거로 구조 오류를 줄였지만, 통화기호 복원은 추측으로 자동 보정하지 않습니다.
- 통화 glyph가 선행 숫자로 바뀌면 단순 철자 오류를 넘어 금액의 자릿수가 커질 수 있습니다. 예를 들어 Tesseract는 `task_000984`에서 `5.75`, `18.5`, `21`을 `115.75`, `118.5`, `1121`로, `task_001104`에서 `11`, `6`, `7`을 `1111`, `116`, `17`로 읽었습니다. 해당 계열은 magnitude-changing 오류로 분류하고 근거 이미지 없이 통화기호를 복원하거나 값을 자동 교체하지 않습니다.
- PP-OCRv5 mobile은 같은 `task_000984`와 `task_001104` 이미지에서 `5.75`, `18.5`, `21`, `11`, `6`, `7`을 자릿수 증폭 없이 보존했고 `bNN` marker와 표 순서도 복원했습니다. 다만 PDF 자체가 tofu box로 렌더링한 통화 glyph를 출력에서 삭제했고 diagonal `TRAINING COPY` watermark를 높은 confidence로 포함했으므로 canonical gold가 아니라 disagreement challenger로 유지합니다.
- PP-OCRv5 mobile의 Mac M3 16GB CPU 스모크는 첫 페이지 cold inference 269.716초, 같은 프로세스의 warm page 17.041초와 16.508초, process max RSS 약 2.60GB였습니다. 초기화·첫 그래프 준비 비용이 크므로 문서마다 프로세스를 새로 띄우지 않고 cloud shard마다 pipeline 하나를 유지합니다.

## Qwen 전달 계약

OCR 엔진이 받는 의미 입력은 렌더링된 문서 이미지만입니다. 엔진은 읽기 순서가 보존된 `bNN` block, 원문 line, page, bbox와 engine-specific confidence를 생성하고, `user_query`는 OCR 호출이 모두 끝난 뒤 Qwen-facing JSONL을 조립할 때만 결합합니다. `answer`, `evidence` label, Claude·Codex 답안, v12 제출 파일과 포털 점수는 OCR 입력·설정 선택·후처리에 사용하지 않으며 출력 writer도 `answer`와 `evidence` 필드를 거부합니다. 이미지와 block ID만 받은 Codex-assisted 전사는 disagreement 진단과 사람 검수 queue 생성에 한해 silver reference로 사용할 수 있지만 human gold, accuracy denominator 또는 단독 모델 선택 근거로 사용하지 않습니다.

이번 이슈가 검증하는 범위는 [`docsem-ocr-v1`](../../schemas/docsem-ocr-v1.schema.json)에 맞는 Qwen 입력 계약 생성까지입니다. 실제 Qwen loader가 이 JSONL을 읽고 expected block order와 query를 받는 smoke test를 통과하기 전에는 "Qwen이 정상 소비한다"거나 Qwen 정확도가 개선됐다고 주장하지 않습니다. Qwen 성능 영향은 동일한 Qwen·prompt·decoder를 고정하고 OCR 입력만 바꾸는 후속 ablation에서 평가합니다.

## Canonical extractor 정책

수정된 Tesseract PSM 6을 portable primary로 사용하고 Apple Vision accurate와 PP-OCRv5 mobile을 diagnostic challenger로 유지합니다. primary가 실패·timeout·빈 block을 반환하거나 challenger와 block ID, 순서, critical token 또는 ordered quantity가 다르면 해당 문서를 `ocr_uncertain`으로 표시해 visual review로 보내며 challenger 출력을 primary 결과 위에 자동으로 덮어쓰지 않습니다. 실패 레코드는 partial block을 남기지 않는 fail-closed 형식을 유지하고, 재시도나 수동 판정은 별도 provenance로 기록합니다.

이 정책은 Apple보다 Tesseract가 더 정확하다는 주장이 아닙니다. 사람 전사 gold가 없는 현재에는 portability, deterministic marker repair, 실행 가능 범위와 오류 추적성을 우선한 운영 선택이며, canonical 변경은 독립 gold에서 critical error 감소가 확인되거나 동등한 품질에서 wall time 또는 memory가 2배 이상 개선될 때만 검토합니다.

## 한계와 validity threats

- Validation 문서와 기존 답안 검수 사례가 설정 선택 전에 완전히 봉인되지 않았으므로 이번 결과는 sealed evaluation이 아닙니다.
- human transcription이 없어 CER·WER, block F1과 quantity F1은 cross-engine agreement일 뿐 OCR accuracy가 아닙니다.
- model-assisted visual review는 독립 annotator gold를 대체하지 못하고 review 모델의 편향을 포함할 수 있습니다.
- Tesseract와 Apple Vision이 같은 renderer와 문서 패턴에서 동일한 오류를 낼 수 있으므로 agreement가 높아도 correctness를 보장하지 않습니다.
- Apple Vision의 구현과 confidence는 macOS·Vision framework 버전에 종속되며, Swift 실행 파일 hash가 같아도 OS가 바뀌면 결과가 달라질 수 있습니다.
- Tesseract의 marker repair는 좁은 문법만 허용해 false positive를 줄였지만 다른 `O`/`0` 변형은 놓칠 수 있습니다.
- process RSS는 전체 unified memory가 아니고, 문서 단위 timing은 실제 페이지별 latency 분포를 제공하지 않습니다.
- PSM 3 산출물이 final code 이전에 생성됐다면 최종 PSM 6과 동일 조건 ablation으로 사용하지 않고 historical preflight로만 남깁니다.
- Qwen loader와 end-to-end inference를 아직 검증하지 않았으므로 downstream usability와 task accuracy는 미측정입니다.

## Issue #8 잔여 완료 조건

현재 구현은 전수 coverage, baseline·neural OCR 실행 경로, Qwen-facing JSONL 계약과 cloud 분산 실행 준비까지 완료한 상태입니다. Issue #8은 독립 annotator가 만든 층화 human OCR gold의 CER·WER·critical-token·evidence-block 검수와 실제 Qwen loader smoke를 완료하기 전까지 닫지 않습니다. 그 전의 Codex-assisted 비교와 cross-engine agreement는 exploratory 진단으로만 보고합니다.

## 재현 산출물과 해시

| 산출물 | 역할 | 상태·해시 |
| --- | --- | --- |
| `artifacts/ocr/validation-manifest.jsonl` | 입력 순서, PDF SHA와 exploratory split 고정 | SHA-256 `87afbb83968c597398ca94c1ffc5940f6322c4381319f2b8f154cbf99789abe9` |
| `artifacts/ocr/apple-vision-200dpi.jsonl` | Apple Vision 217개 전수 출력 | OCR aggregate hash `63767f0736c441a5f59269082231588248325596a203787faa34787827373cb7` |
| `artifacts/ocr/apple-vision-200dpi.hash.json` | Apple record별·aggregate semantic hash | 생성 완료 |
| `artifacts/ocr/tesseract-200dpi-psm6-final.jsonl` | 수정된 Tesseract 최종 전수 출력 | OCR aggregate hash `15ceb5f26d42e9c7556c7c5e25b2267a4ce3344bae4282c3aec37b819beee0be` |
| `artifacts/ocr/tesseract-200dpi-psm6-final.hash.json` | Tesseract record별·aggregate semantic hash | 생성 완료 |
| `artifacts/ocr/tesseract-vs-apple-200dpi-final.json` | 방향이 명시된 cross-engine agreement | 217개 비교, 누락·예상 밖 ID 0 |
| `artifacts/ocr/tesseract-rerun20-final.jsonl` | 고정 20개 deterministic rerun | OCR aggregate hash `111c78780e34300291fe3897e3077d01c42cadba5b5558caf56ebc5562a5bd5b` |
| `artifacts/ocr/apple-rerun20-final.jsonl` | 동일 20개 Apple deterministic rerun | OCR aggregate hash `84d008b8b8bf2f0e4bfdd6ad14583f8ae2bd7d36242d1604cf9224d351095655` |
| `artifacts/ocr/docsem-validation-ocr-input.tar.gz` | Colab·Kaggle label-free 입력 archive | SHA-256 `9fb35e81feead385fedd3a5bd66ca780ca2aaee5b92b2247f75114cfae642967` |
| `artifacts/ocr/cloud-8/shard-plan.json` | 8개 balanced shard 배정 | assignment hash `860337a43768bd3d9d915125b92bf6acfde7dfe907818a3b151ea497ebfeb20d` |

OCR aggregate hash는 timing, `user_query`와 machine-local absolute path를 제외한 schema version, instance ID, status, engine, pages, ordered blocks와 안정적인 provenance를 hash합니다. 이 hash만으로 입력 PDF 집합을 증명하지는 않으므로 manifest SHA와 record의 `input_pdf_sha256`, 실행 파일 identity와 `run_fingerprint`를 함께 보존합니다. 최종 보고서에는 각 파일의 line count, status count와 schema validation 결과를 함께 기록합니다.

## 향후 sealed 평가의 Go/No-Go 기준

- Operational: 217/217 문서, 434/434 페이지 성공, 빈 출력 0, crash·timeout 0
- Structure: gold evidence block recall 100%, ordered block sequence 손실 0
- Critical: currency·부호·소수점 answer-changing 오류 0, quantity tuple exact 99.5% 이상
- Fidelity: evidence-region CER 0.5% 이하, WER 1.0% 이하
- Efficiency: process max RSS 8GB 이하, p95 20초/document 이하, 434페이지 90분 이하

아래 기준은 독립 human gold와 접촉 이력이 없는 holdout을 새로 동결한 뒤에만 적용합니다. 이번 exploratory 전수 비교에서는 operational coverage와 효율성만 직접 판정하고, 엔진 간 agreement를 정확도로 해석하지 않습니다.

품질이 동률이면 더 단순하고 portable한 엔진을 유지합니다. neural OCR로 교체하려면 critical error를 줄이거나 동일 품질에서 wall time 또는 memory를 2배 이상 개선해야 합니다.

## 공식·업스트림 자료

- [Apple Vision text recognition](https://developer.apple.com/documentation/vision/recognizing-text-in-images)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PP-OCRv5 mobile detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det/tree/0d63e78e2b680928f6b1747d76a08db6e645efb7)
- [PP-OCRv5 English mobile recognizer](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec/tree/267c36e24c331595590fe7bd72bde2436fd286f2)
- [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)
- [GLM-OCR](https://github.com/zai-org/GLM-OCR)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)
