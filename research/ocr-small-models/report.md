# 소형 OCR 모델 탐색 및 DocSem 고정 사례 비교

## 결론

Issue #11의 현재 비교는 checked-in v2 runner commit `95af62c`와 실행 소스 SHA-256
`64706348c218f729e94430ab0fa4b33e9ec6467e41f05e665731a3a7c78644cf`로 생성된
Kaggle fresh-child 증거를 사용한다. 한 모델마다 새 child를 순차 실행했고 결과 bundle의
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

## 실행 시간과 자원

비교의 peak 값은 fresh child를 감시한 parent-sampled RSS/VRAM을 우선한다. child 내부
RSS는 모든 행에서 4,690,857,984 B로 기록되었고 allocator VRAM도 함께 남긴다. 둘은
측정 정의가 다르므로 parent 값을 child 값으로 대체하지 않는다.

| 모델 | load s | s/doc | parent RSS B | child RSS B | parent VRAM B | child allocated VRAM B | output B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaddleOCR-VL 1.6 | N/A | N/A | 1,209,257,984 | 4,690,857,984 | 106,954,752 | 0 | 0 |
| GLM-OCR | 3.3231 | 55.9483 | 4,142,870,528 | 4,690,857,984 | 7,442,792,448 | 6,586,189,824 | 4,271 |
| Surya OCR 2 | 1.9743 | 50.6099 | 2,892,279,808 | 4,690,857,984 | 3,298,820,096 | 2,900,749,312 | 1,333 |
| Granite Docling 258M | 1.6992 | 352.6226 | 2,299,080,704 | 4,690,857,984 | 1,124,073,472 | 890,579,456 | 1,024 |
| SmolDocling 256M preview | 1.1509 | 38.9622 | 2,263,355,392 | 4,690,857,984 | 1,157,627,904 | 887,485,952 | 3,729 |

비용은 Kaggle 무료 quota다. 할당된 T4 두 장 중 child에는 `cuda:0`만 노출했다.

## 증거, 환경, 입력 격리

v2 bundle의 `results.jsonl`, CSV, report, environment, input, audit, run manifest, artifact
hashes, child results, logs, raw text와 실제 실행된 runner를 보존했다. 렌더링 PNG와 원본 zip은
저장소에 중복 체크인하지 않았지만 `input.json`과 artifact manifest의 bytes/SHA-256은
유지한다. Poppler는 `/usr/bin/pdftoppm` 22.02.0, binary SHA-256
`9854d5c30e9e56b972bc89f88dec75679296e86768715e25fb1cef45d3c7a03e`, 200 dpi다.

`raw/v2/pip-freeze.txt`와 `requirements-kaggle-v2.txt`는 v2 `environment.json`의
`pip_freeze_all_verbatim` 문자열과 byte-identical한 evidence/ML runtime lock이다.
`pyproject.toml`은 dependency-free evaluator 패키지
정의이므로 ML 실행 lock을 대신하지 않는다.

모델에는 두 PNG와 고정 prompt만 전달했다. query, labels, answer, evidence 파일은 읽거나
child에 넘기지 않았다. `user_query`는 inference 뒤에 instance ID로 join하며 raw exact,
normalized exact, SHA-256 exact가 모두 1/1이다.

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
ISSUE8=/Users/choco/.codex/worktrees/bed4/docinsights-2026
PYTHONPATH=src python3 -m docinsights_hf_ocr generate \
  --raw-results research/ocr-small-models/raw/v2/results.jsonl \
  --raw-dir research/ocr-small-models/raw/v2/raw \
  --candidates research/ocr-small-models/candidates.json \
  --reference "$ISSUE8/artifacts/ocr/codex-validation-reference.jsonl" \
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
