# Issue #8 Codex Silver 텍스트 평가

## 판정 범위

Validation 217개 PDF를 이미지로만 읽어 검증한 Codex 전사를 `codex-assisted-silver` reference로 사용해 Apple Vision, Tesseract, PP-OCRv5 mobile OCR 텍스트를 전수 채점했다. 이 평가는 Golden Label과 같은 계산 방향을 제공하지만 reference 자체가 사람이 만든 gold가 아니므로 결과 이름과 JSON 계약에 `silver_agreement_not_human_gold_accuracy`를 강제한다.

정답, evidence, `labels.jsonl`, 기존 답안은 reference 생성이나 OCR 입력에 사용하지 않았다.

## 점수 정의

대표 점수는 임의의 복합 가중치를 두지 않고 문자 오류율에서 직접 계산한다.

```text
micro CER = 전체 Levenshtein 문자 편집거리 합 / 전체 silver reference 문자 수
silver_text_score = 100 × max(0, 1 − micro CER)
symmetric_similarity = 1 − 편집거리 / max(reference 길이, prediction 길이)
```

strict 문자열은 Unicode NFC, CRLF/CR→LF, 수평 공백 통합만 적용한다. compatible 문자열은 Unicode NFKC와 모든 공백·줄바꿈 통합만 적용하며 대소문자와 문장부호를 보존한다. 의미 교정, 의역, 숫자·단위 교정은 하지 않는다.

대표 점수와 함께 다음 지표를 독립적으로 기록한다.

- strict/compatible document exact count
- macro/micro CER와 WER
- 대칭 문자·단어 Levenshtein 유사도
- ordered block exact와 block F1
- 숫자·통화·단위 exact-token F1
- 순서와 modifier를 보존한 ordered-quantity F1
- 평균·중앙값·p95 sec/doc와 docs/min

## 코드와 데이터 계약

- `src/docinsights_ocr/metrics.py`
  - `edit_distance`: exact Levenshtein distance
  - `edit_similarity`: 길이 비대칭을 제거한 0~1 대칭 유사도
  - `nfkc_whitespace_normalize_text`: 의미를 바꾸지 않는 compatible 정규화
- `src/docinsights_ocr/silver_evaluation.py`
  - `evaluate_codex_silver`: exactly-once ID와 silver reference kind를 확인하고 전수 평가
  - `write_silver_evaluation`: JSON/Markdown을 원자적으로 게시하고 SHA-256 반환
- `schemas/codex-silver-evaluation-v1.schema.json`
  - `reference_kind=codex-assisted-silver`
  - `interpretation=silver_agreement_not_human_gold_accuracy`
  - source 절대경로·SHA·record 수, normalization 정책, 대표 점수 정의, aggregate와 instance별 지표
- CLI: `docinsights-ocr codex-silver-evaluate`

coverage가 다르거나 instance/block ID가 중복되거나 reference kind/status가 잘못되면 점수를 만들지 않고 실패한다. JSON/Markdown 출력 경로가 reference 또는 prediction 입력과 충돌해도 원본을 덮어쓰지 않고 실패한다. 출력에는 reference/prediction 원문, query, answer, evidence를 복사하지 않는다.

## Validation 217개 실측

| 엔진 | 모델/weight 크기 | inference | strict valid | Silver score | micro CER | micro WER | block exact | exact-token F1 | quantity F1 | 평균 sec/doc | docs/min | p95 sec/doc | process max RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tesseract 5.5.3 `eng`, PSM 6, 200 DPI | `eng.traineddata` 3.9MiB; Homebrew 설치 36MiB | 217/217 | 217/217 | **99.9415** | 0.000585 | 0.006029 | 217/217 | 0.992057 | 0.991993 | 6.0335 | 9.9445 | 6.3423 | 107.531MiB |
| Apple Vision accurate, 200 DPI | OS 내장 모델이라 weight 분리 측정 불가; runner 162KiB | 217/217 | 217/217 | **99.3777** | 0.006223 | 0.008946 | 217/217 | 0.985062 | 0.981604 | 3.8781 | 15.4714 | 4.0986 | 214.484MiB |
| PP-OCRv5 mobile, 200 DPI | detector+English recognizer 결합 크기 미측정 | 217/217 | 216/217 | **99.6176** | 0.003824 | 0.014463 | 216/217 | 0.996611 | 0.996448 | 58.9155 | 1.0184 | 62.1973 | 미측정 |

Codex silver 기준에서는 Tesseract가 문자 agreement가 가장 높고 Apple Vision이 가장 빠르며 PP-OCRv5는 text agreement가 두 엔진 사이지만 가장 느리다. 이 결과는 사람 gold 정확도 순위가 아니다. 세 엔진 모두 동일 문서 패턴이나 renderer에 의해 같은 오류를 낼 수 있고 Codex도 전사 오류를 포함할 수 있다.

PP-OCRv5는 `task_001108`의 `b09` marker를 `b0`로 읽어 strict canonical merge를 통과하지 못했다. 본문은 reference와 일치하지만 구조 계약 위반이므로 raw output을 고치지 않고 diagnostic으로만 점수화했으며 strict valid는 216/217로 기록했다.

## 입력과 산출물 해시

| 종류 | 절대경로 | SHA-256 |
| --- | --- | --- |
| Codex silver reference | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-reference.jsonl` | `d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a` |
| Apple Vision prediction | `/Users/choco/Documents/project/docinsights-2026-issue-8/artifacts/ocr/apple-vision-200dpi.jsonl` | `8d55f10f9f628cdc6744f451d1c04de5158495a6452ae123d0ff9670d1908c01` |
| Tesseract prediction | `/Users/choco/Documents/project/docinsights-2026-issue-8/artifacts/ocr/tesseract-200dpi-psm6-final.jsonl` | `8b5db676267a0a1ab51c345798994eb5f38f4b5148728e54adbb40cf94acadaf` |
| PP-OCRv5 Kaggle raw prediction | `kaggle/version-3/result-shard-00-of-01.jsonl` | `60e1844155e70fc5f4cea218e86be4ac2e6ca9fa35d4699fc820c568231c0fd1` |
| Apple evaluation JSON | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-silver-apple-vision-evaluation.json` | `0dbe819e5a2a0f7ec6103d53f7a566d8f3df4ee53104c115d4dbe44bc530ab06` |
| Apple evaluation Markdown | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-silver-apple-vision-evaluation.md` | `76645a807306bad4070ba7ab43b2f9d7e0a5c1f4c963a71f88fce111c1cff47c` |
| Tesseract evaluation JSON | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-silver-tesseract-evaluation.json` | `7ec24f24e907358091aa393ba7d65dc8d5a2890fede3d2ad0f9655fde36ea35c` |
| Tesseract evaluation Markdown | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-silver-tesseract-evaluation.md` | `d37432a7a9623e36891278b7ac0951b3be60c7fc2314f3966307b15c9c350f21` |
| PP-OCRv5 evaluation JSON | `research/ocr-small-models/raw/silver/ppocrv5-kaggle-evaluation.json` | `359ea3dd74f7995e2c710da80165134fad3147917587e0658d9efffa2808fb47` |
| PP-OCRv5 runtime JSON | `research/ocr-small-models/raw/silver/ppocrv5-kaggle-runtime.json` | `913b5b5e80a3e8a23f2542a23978f255ee1a4e2b93f8847965984e3bdc6d0a48` |

## 재현 명령

```bash
uv run docinsights-ocr codex-silver-evaluate \
  artifacts/ocr/codex-validation-reference.jsonl \
  /Users/choco/Documents/project/docinsights-2026-issue-8/artifacts/ocr/apple-vision-200dpi.jsonl \
  artifacts/ocr/codex-silver-apple-vision-evaluation.json \
  --markdown artifacts/ocr/codex-silver-apple-vision-evaluation.md \
  --engine-label 'Apple Vision accurate 200 DPI'

uv run docinsights-ocr codex-silver-evaluate \
  artifacts/ocr/codex-validation-reference.jsonl \
  /Users/choco/Documents/project/docinsights-2026-issue-8/artifacts/ocr/tesseract-200dpi-psm6-final.jsonl \
  artifacts/ocr/codex-silver-tesseract-evaluation.json \
  --markdown artifacts/ocr/codex-silver-tesseract-evaluation.md \
  --engine-label 'Tesseract eng PSM 6 200 DPI'

uv run docinsights-ocr codex-silver-evaluate \
  artifacts/ocr/codex-validation-reference.jsonl \
  /absolute/path/to/result-shard-00-of-01.jsonl \
  research/ocr-small-models/raw/silver/ppocrv5-kaggle-evaluation.json \
  --markdown research/ocr-small-models/raw/silver/ppocrv5-kaggle-evaluation.md \
  --engine-label 'PP-OCRv5 mobile (Kaggle CPU)' \
  --reference-label issue8/codex-validation-reference.jsonl \
  --prediction-label kaggle/version-3/result-shard-00-of-01.jsonl

uv run pytest -q
uv run ruff check .
git diff --check
```

## 최종 운영 판정

portable primary는 Tesseract를 유지한다. Apple Vision은 처리량 우위가 있어 macOS challenger/fallback으로 유지하고, PP-OCRv5는 구조 marker 불일치를 탐지하는 diagnostic challenger로 유지한다. canonical 엔진 변경이나 대회 정확도 주장은 독립 human gold 또는 Qwen end-to-end ablation으로 다시 확인해야 한다.

## 독립 검수

별도 읽기 전용 verifier가 Apple Vision과 Tesseract의 reference/prediction source 해시, 217 unique instance, score/CER/WER/F1/latency/block aggregate를 instance 행에서 독립 재계산했다. PP-OCRv5는 runtime/result SHA, 217 unique instance, silver schema와 aggregate를 다시 확인했고 canonical merge 실패를 별도 audit에 보존했다. `edit_distance`는 무작위 문자열 20,000쌍을 독립 dynamic programming 구현과 대조해 불일치 0건이었다.

source overwrite 방지 회귀 테스트와 PP-OCRv5 provenance 검사를 포함한 현재 검증은 `276 passed`, Ruff 통과, 변경 파일 Ruff format과 `git diff --check` 통과다. 세 evaluation JSON은 `check-jsonschema`로 같은 schema를 통과했다.
