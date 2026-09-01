# 소형 OCR 모델 탐색 및 DocSem 고정 사례 비교

## 결론

Issue #11의 현재 비교는 checked-in v2 runner commit `5498d5a`와 실행 소스 SHA-256
`4e5be04c3afb6d487b547765a813e9737047cafa18df1882705a06b57ca728e3`로 생성된
Kaggle Version #3 fresh-child 증거를 사용한다. run ID는
`task_000909-1788113789-4e5be04c3afb`, ZIP SHA-256은
`08995ebc6283c082fd9add596412a870910d2a875f63948cf1ae824939d2ec17`이다.
한 모델마다 새 child를 순차 실행했고 결과 bundle의
텍스트 파일은 `raw/v2/`에 byte-exact하게 보존했다. 이전 reconstructed 자료는
`raw/v1-historical/`에 분리했으며 현재 비교에는 사용하지 않는다.

Pinned metadata gate를 통과해 선정된 후보는 PaddleOCR-VL 1.6, Surya OCR 2, Granite
Docling 258M, SmolDocling 256M preview 네 개다. GLM-OCR은 실제 추론 및 OCR 형식 검사는
통과했지만 1,325,258,240 parameters로 10억 gate를 초과하므로 diagnostic 행만 유지하고
선정하지 않는다. Paddle은 metadata gate를 통과했지만 Transformers 5.12.1의
`PaddleOCRVLConfig.text_config` 오류로 load에 실패했다. Granite는 추론 프로세스가
끝났지만 두 페이지가 반복 `!` 문자여서 invalid OCR이다. 추론 성공과 OCR 유효성을 같은
의미로 사용하지 않는다.

Issue #8은 main merge commit `f57df2b6ab01b1a3024e97f09ab14ed66db8e1a2`로
병합·종료됐다. `src/docinsights_ocr/silver_evaluation.py`와
`docinsights-ocr codex-silver-evaluate`를 재사용해 Apple Vision, Tesseract, PP-OCRv5
mobile을 Codex Validation silver 217건 전수로 채점했다. 절대 로컬 경로는 재현 가능한
논리 라벨로 바꿨다. 해석 계약은
`silver_agreement_not_human_gold_accuracy`이며 human-gold accuracy가 아니다.

HF document-model 품질은 여전히 `task_000909` 한 건만 실제 추론한 결과다. 세 OCR 엔진의
217건 full-silver 결과와 HF의 1건 smoke 결과는 cohort와 device/runtime가 다르다. 따라서
아래 수치를 한 표에 보존하더라도 교차 cohort 품질 순위나 승자를 만들 수 없다.

| 모델 | inference | valid OCR | silver CER | silver WER | block F1 | 누락 block |
| --- | --- | --- | ---: | ---: | ---: | --- |
| PaddleOCR-VL 1.6 | 실패 | 실패 | N/A | N/A | N/A | 출력 없음 |
| GLM-OCR (diagnostic) | 성공 | 성공 | 0.154754 | 0.203540 | 0.954545 | b12, b13 |
| Surya OCR 2 | 성공 | 성공 | 0.812291 | 0.808260 | 0.357143 | 18개 |
| Granite Docling 258M | 성공 | 실패 | N/A | N/A | N/A | 반복문자 invalid |
| SmolDocling 256M preview | 성공 | 성공 | 0.452238 | 0.452802 | 0.756757 | b08-b13, b21-b23 |

217건 full-silver cohort 행은 다음과 같다.

| 엔진 | samples | inference | strict valid | silver score | micro CER | micro WER | block exact | s/doc | docs/min | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple Vision accurate 200 DPI | 217 | 217/217 | 217/217 | 99.377738 | 0.006223 | 0.008946 | 217/217 | 3.878133 | 15.471362 | 224,903,168 B |
| Tesseract 5.5.3 eng PSM 6 | 217 | 217/217 | 217/217 | 99.941495 | 0.000585 | 0.006029 | 217/217 | 6.033493 | 9.944489 | 112,754,688 B |
| PP-OCRv5 mobile, Kaggle CPU | 217 | 217/217 | 216/217 | 99.617639 | 0.003824 | 0.014463 | 216/217 | 58.915511 | 1.018408 | N/A |

Tesseract의 silver agreement가 가장 높고 Apple Vision의 처리량이 가장 높으며 PP-OCRv5는
그 사이의 text agreement지만 가장 느리다. 이 비교는 동일 217건 cohort 안에서만 유효하고,
동일 renderer 또는 reference의 공통 오류 가능성이 있으므로 사람 정답 정확도로 해석하지
않는다.

PP-OCRv5 raw output은 217건과 runtime SHA-256 계약을 모두 충족했지만 `task_001108`에서
`b09` marker를 `b0`로 인식했다. 본문은 Codex reference와 일치해도 canonical block-order
계약에는 어긋나므로 strict `cloud-merge`는 실패했고, 표는 raw output을 수정하지 않은 채
동일 silver scorer로 평가한 diagnostic 결과다. 이 한 건을 자동 보정하거나 217/217 strict
valid로 보고하지 않는다.

SmolDocling의 DocTags 원문에는 `b01` 같은 marker가 남아 있어 현재 parser가 block-ID
alignment를 수행했다. 구조 marker가 없는 출력을 성공으로 간주하는 예외는 두지 않는다.

공식 model card 기준 언어·문서 특성·실행 route는 `candidates.json`에 구조화했다. Paddle은
multilingual parsing/layout/table/formula, GLM은 8개 명시 언어와 복합 문서 추출, Surya는
91개 언어와 layout/reading-order/table/HTML, Granite는 영어와 실험적 ja/ar/zh 및
full-page structure, SmolDocling은 영어와 bbox/table/formula/code/chart를 다룬다. 장치별
`feasibility`에서 T4만 이 v2의 실제 측정이며, 공식 CPU·Apple route는 모두 로컬 미측정으로
구분한다.

## 실행 시간과 자원

비교의 peak 값은 fresh child를 감시한 parent-sampled RSS/VRAM을 우선한다. child 내부
RSS와 allocator VRAM도 별도 열로 함께 남긴다. 둘은
측정 정의가 다르므로 parent 값을 child 값으로 대체하지 않는다.
현재 evaluator는 parent RSS를 양의 exact integer, VRAM을 0 이상의 exact integer로
요구한다. Version #3의 다섯 행은 `peak_process_rss_sampling_error`와
`peak_vram_sampling_error`를 모두 명시적으로 `null`로 기록했다. sampling error가 하나라도
기록되면 일부 peak가 남아 있어도 해당 행 전체를 거부하며, runner도 sampling error를
inference 실패로 전환하고 raw 출력을 제거한다.

| 모델 | load s | s/doc | parent RSS B | child RSS B | parent VRAM B | child allocated VRAM B | output B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaddleOCR-VL 1.6 | N/A | N/A | 1,112,231,936 | 1,115,394,048 | 106,954,752 | 0 | 0 |
| GLM-OCR | 3.0923 | 55.4766 | 4,199,813,120 | 4,360,134,656 | 7,442,792,448 | 6,586,189,824 | 4,271 |
| Surya OCR 2 | 1.8647 | 47.0260 | 2,859,540,480 | 3,020,025,856 | 3,298,820,096 | 2,900,749,312 | 1,333 |
| Granite Docling 258M | 1.7479 | 354.4969 | 2,262,228,992 | 2,369,581,056 | 1,124,073,472 | 890,579,456 | 1,024 |
| SmolDocling 256M preview | 1.3555 | 38.1619 | 2,316,926,976 | 2,369,949,696 | 1,157,627,904 | 887,485,952 | 3,729 |

비용은 Kaggle 무료 quota다. 할당된 T4 두 장 중 child에는 `cuda:0`만 노출했다.
PP-OCRv5 217건 실행은 별도 Kaggle CPU Version #3에서 217/217을 완료했으며 평균
58.915511초/문서, p95 62.197254초/문서였다. 해당 runtime은 peak RSS를 기록하지 않아
추정값을 채우지 않았다.

## 증거, 환경, 입력 격리

v2 bundle의 34개 artifact manifest를 먼저 검증한 뒤 `results.jsonl`, CSV, report,
environment, input, audit, run manifest, artifact hashes, child results, logs, raw text와 실제
실행된 runner를 보존했다. 렌더링 PNG와 원본 zip은
저장소에 중복 체크인하지 않았지만 `input.json`과 artifact manifest의 bytes/SHA-256은
유지한다. Poppler는 `/usr/bin/pdftoppm` 22.02.0, binary SHA-256
`9854d5c30e9e56b972bc89f88dec75679296e86768715e25fb1cef45d3c7a03e`, 200 dpi다.

`raw/v2/pip-freeze.txt`와 `requirements-kaggle-v2.txt`는 v2 `environment.json`의
`pip_freeze_all_verbatim` 문자열과 byte-identical한 실행 후 환경 snapshot이다. runner가
Kaggle base 위에 다섯 pinned package를 `--no-deps`로 설치한 결과이며 local file URL도
포함하므로 reconstructible lock이라고 주장하지 않는다. `pyproject.toml`은 dependency-free
evaluator 패키지 정의일 뿐 ML 실행 환경을 재구성하지 않는다.

모델에는 두 PNG와 고정 prompt만 전달했다. query, labels, answer, evidence 파일은 읽거나
child에 넘기지 않았다. 생성기는 엄격한 v2 결과 검증이 끝난 뒤에만 독립 source task에서
`joined-queries.jsonl`을 만들며, 각 행을 raw `results.jsonl` SHA-256에 결합한다. 기존 join이
그 결합과 byte-exact하게 일치하지 않으면 stale/prebuilt artifact로 거부한다. 현재
`user_query`의 raw exact, normalized exact, SHA-256 exact는 모두 1/1이다.
두 성공 페이지 각각의 page identity 1/2, path, bytes, SHA-256 및 전체 bytes/digest projection을
실제 raw 파일과 모두 대조한 뒤에만 join 파일을 생성하거나 읽는다. raw 검증이 실패하면 기존
join은 byte-exact하게 유지되고, join이 없었다면 새로 만들지 않는다.

모든 `status=ok` reference 행은 `codex-assisted-silver` /
`codex-assisted-visual-transcription` identity와 PDF·rendered image·renderer·Codex executable
SHA-256 provenance를 갖춰야 한다. 현재 reference artifact SHA-256은
`d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a`이며 생성 결과에도
기록한다. 다른 engine이 섞인 reference는 비교 전에 fail closed한다.

## Full-silver baseline 근거와 한계

`baselines.json`은 더 이상 full-silver 품질 값을 수동 복제하지 않는다. 생성기는 Apple
Vision, Tesseract, PP-OCRv5의 `raw/silver/*-evaluation.json`을 읽고,
evaluation artifact SHA-256, reference SHA-256, prediction SHA-256, 217건 exactly-once
coverage, instance ID exact set, reference/prediction status 합계와
`silver_agreement_not_human_gold_accuracy` 계약을 검증한 뒤 표를 만든다.

| artifact | SHA-256 |
| --- | --- |
| Codex Validation reference 217 | `d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a` |
| Apple prediction 217 | `8d55f10f9f628cdc6744f451d1c04de5158495a6452ae123d0ff9670d1908c01` |
| Apple scorer JSON | `5e7a85338f58ad766cdcc0353e5bd9e45e3a32a4394d41f73d0c20751fb32645` |
| Tesseract prediction 217 | `8b5db676267a0a1ab51c345798994eb5f38f4b5148728e54adbb40cf94acadaf` |
| Tesseract scorer JSON | `3db904ee7e4278b101915fbb701ecf4b38025e105c5d51033290f57e52446e49` |
| PP-OCRv5 Kaggle raw prediction 217 | `60e1844155e70fc5f4cea218e86be4ac2e6ca9fa35d4699fc820c568231c0fd1` |
| PP-OCRv5 scorer JSON | `359ea3dd74f7995e2c710da80165134fad3147917587e0658d9efffa2808fb47` |
| PP-OCRv5 runtime JSON | `913b5b5e80a3e8a23f2542a23978f255ee1a4e2b93f8847965984e3bdc6d0a48` |

통합 비교표에는 각 행의 params, primary weight bytes, Hugging Face downloads snapshot,
설치 크기, 성공률, CER/WER, 속도와 peak RAM/VRAM을 포함한다. HF snapshot의 설치 footprint는
checkpoint cache와 패키지를 분리 측정하지 않았으므로 `NA`다. Apple Vision weight/install은
OS 통합 모델이라 분리할 수 없다. Tesseract `eng.traineddata`는 4,113,088 B이고 현재
Homebrew Cellar `du -sk`는 34,848,768 B다. 실행하지 않은 수치는 추정하지 않았다.

HF의 표본 1건과 512-token ceiling은 full-page coverage나 결정성을 확정하지 못한다.
선정은 metadata 및 실행 가능성 판단이고, 단일 silver 사례의 성공을 품질이라고 부르지
않는다. 같은 scorer로 HF를 217건 평가하려면 먼저 모델별 217건 prediction JSONL을 실제로
생성해야 하며, 이번 갱신에서는 존재하지 않는 값을 만들지 않았다.

## 재현 및 Issue #11 완료 기준

```bash
export ISSUE8_REFERENCE=/absolute/path/to/codex-validation-reference.jsonl
export ISSUE8_REFERENCE_SHA256=d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a
export APPLE_PREDICTION=/absolute/path/to/apple-vision-200dpi.jsonl
export TESSERACT_PREDICTION=/absolute/path/to/tesseract-200dpi-psm6-final.jsonl
export PPOCRV5_PREDICTION=/absolute/path/to/result-shard-00-of-01.jsonl
test "$(shasum -a 256 "$ISSUE8_REFERENCE" | cut -d ' ' -f 1)" = \
  "$ISSUE8_REFERENCE_SHA256"

uv run docinsights-ocr codex-silver-evaluate \
  "$ISSUE8_REFERENCE" "$APPLE_PREDICTION" \
  research/ocr-small-models/raw/silver/apple-vision-evaluation.json \
  --markdown research/ocr-small-models/raw/silver/apple-vision-evaluation.md \
  --engine-label 'Apple Vision accurate 200 DPI' \
  --reference-label issue8/codex-validation-reference.jsonl \
  --prediction-label issue8/apple-vision-200dpi.jsonl
uv run docinsights-ocr codex-silver-evaluate \
  "$ISSUE8_REFERENCE" "$TESSERACT_PREDICTION" \
  research/ocr-small-models/raw/silver/tesseract-evaluation.json \
  --markdown research/ocr-small-models/raw/silver/tesseract-evaluation.md \
  --engine-label 'Tesseract eng PSM 6 200 DPI' \
  --reference-label issue8/codex-validation-reference.jsonl \
  --prediction-label issue8/tesseract-200dpi-psm6-final.jsonl
uv run docinsights-ocr codex-silver-evaluate \
  "$ISSUE8_REFERENCE" "$PPOCRV5_PREDICTION" \
  research/ocr-small-models/raw/silver/ppocrv5-kaggle-evaluation.json \
  --markdown research/ocr-small-models/raw/silver/ppocrv5-kaggle-evaluation.md \
  --engine-label 'PP-OCRv5 mobile (Kaggle CPU)' \
  --reference-label issue8/codex-validation-reference.jsonl \
  --prediction-label kaggle/version-3/result-shard-00-of-01.jsonl

uvx check-jsonschema \
  --schemafile schemas/codex-silver-evaluation-v1.schema.json \
  research/ocr-small-models/raw/silver/apple-vision-evaluation.json \
  research/ocr-small-models/raw/silver/tesseract-evaluation.json \
  research/ocr-small-models/raw/silver/ppocrv5-kaggle-evaluation.json

PYTHONPATH=src python3 -m docinsights_hf_ocr generate \
  --raw-results research/ocr-small-models/raw/v2/results.jsonl \
  --raw-dir research/ocr-small-models/raw/v2/raw \
  --candidates research/ocr-small-models/candidates.json \
  --reference "$ISSUE8_REFERENCE" \
  --reference-sha256 "$ISSUE8_REFERENCE_SHA256" \
  --tasks research/ocr-small-models/manifests/source-queries.jsonl \
  --joined-tasks research/ocr-small-models/generated/joined-queries.jsonl \
  --environment research/ocr-small-models/manifests/environment.json \
  --baselines research/ocr-small-models/baselines.json \
  --out-dir research/ocr-small-models/generated
ISSUE8_REFERENCE="$ISSUE8_REFERENCE" \
ISSUE8_REFERENCE_SHA256="$ISSUE8_REFERENCE_SHA256" python3 -m pytest -q
ruff check .
ruff format --check src/docinsights_hf_ocr \
  tests/test_evaluation.py tests/test_v2_evidence.py
python3 -m compileall -q src tests notebooks
git diff --check -- . \
  ':(exclude)research/ocr-small-models/raw/v2/results.csv'
```

Issue #11의 완료 주장은 v2 evidence ingestion, pinned revision/file OID 검증, 네 selected
후보와 GLM diagnostic rejection, inference/validity 분리, HF 고정 1건 silver alignment,
세 OCR 엔진의 217건 scorer 실행, PP-OCRv5 strict schema 실패의 보존, schema 및 byte/hash
재현 검사가 통과하는 범위로 제한한다. 217건 HF document-model 추론 또는 human-gold
정확도는 완료 주장에 포함하지 않는다.
`raw/v2/results.csv`는 원본 Kaggle ZIP의 CRLF를 byte-exact 보존하므로 whitespace diff
검사에서만 제외하며, 내용은 artifact hash와 `cmp` 회귀 테스트로 검증한다.
