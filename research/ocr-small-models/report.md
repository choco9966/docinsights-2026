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

이 평가는 `task_000909` 한 건이다. Codex-assisted silver reference 217건을 사용할 수
있지만 품질 비교의 실제 표본은 1건이며 human gold가 아니다. 따라서 아래 agreement는
모델 accuracy 또는 품질 승자 근거가 아니다.

| 모델 | inference | valid OCR | silver CER | silver WER | block F1 | 누락 block |
| --- | --- | --- | ---: | ---: | ---: | --- |
| PaddleOCR-VL 1.6 | 실패 | 실패 | N/A | N/A | N/A | 출력 없음 |
| GLM-OCR (diagnostic) | 성공 | 성공 | 0.154754 | 0.203540 | 0.954545 | b12, b13 |
| Surya OCR 2 | 성공 | 성공 | 0.812291 | 0.808260 | 0.357143 | 18개 |
| Granite Docling 258M | 성공 | 실패 | N/A | N/A | N/A | 반복문자 invalid |
| SmolDocling 256M preview | 성공 | 성공 | 0.452238 | 0.452802 | 0.756757 | b08-b13, b21-b23 |

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

## 운영 baseline과 한계

생성기는 `baselines.json`에서 Apple Vision과 Tesseract 행을 매번 동적으로 만든다. 현재
운영 대조는 217문서·434쪽·4,991 blocks·실패 0건이다. Apple Vision 3.878 s/doc,
224,903,168 B peak RSS와 Tesseract PSM 6 6.033 s/doc, 112,754,688 B peak RSS는 다른
217문서 workload다. 두 엔진 간 CER/WER는 agreement이지 gold accuracy가 아니다.

현재 표본 1건과 512-token ceiling은 full-page coverage나 결정성을 확정하지 못한다.
선정은 metadata 및 실행 가능성 판단이고, 단일 silver 사례의 성공을 품질이라고 부르지
않는다. 다음 단계는 라이선스/trust 승인을 거쳐 S1=6, S2=30, S3=217, R20=20으로
확장하는 것이다.

## 재현 및 Issue #11 완료 기준

```bash
test -n "$ISSUE8_REFERENCE"
test -n "$ISSUE8_REFERENCE_SHA256"
test "$(shasum -a 256 "$ISSUE8_REFERENCE" | cut -d ' ' -f 1)" = \
  "$ISSUE8_REFERENCE_SHA256"
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
python3 -m pytest -q
ruff check src tests notebooks
ruff format --check src tests notebooks
python3 -m compileall -q src tests notebooks
```

Issue #11의 완료 주장은 v2 evidence ingestion, pinned revision/file OID 검증, 네 selected
후보와 GLM diagnostic rejection, inference/validity 분리, silver alignment 재생성, 동적
baseline 및 byte/hash 재현 검사가 통과하는 범위로 제한한다.
