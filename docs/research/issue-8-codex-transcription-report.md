# Issue #8 Codex 전수 전사 및 Query 비교 보고서

## 결론

Validation 217개를 먼저 전사하고 완전성 게이트를 통과한 뒤 Train 908개를 전사했다. 두 split 모두 expected ID가 정확히 한 번 존재하고 `status=ok`, 입력 이미지 2페이지, ordered `b01`~`b23`, raw 응답, stderr sidecar, provenance, run fingerprint, PDF/PNG SHA 계약을 충족한다.

저장소의 기존 Query 정본은 각 split `tasks.jsonl`의 `user_query`다. 이 값은 PDF의 정량 구절을 그대로 옮긴 문자열이 아니라 데이터셋이 제공하는 패러프레이즈 질의다. PDF와 Codex 전사에서 복원한 시나리오 Query를 정본과 전수 비교한 결과 raw exact와 제한적 normalized exact 일치는 모두 0건이었다. 모든 항목은 불일치로 확정됐으며 미결정은 없다.

## 입력 및 기존 Query 원천

| 구분 | 절대경로 | 레코드 수 | SHA-256 | Query 스키마 |
| --- | --- | ---: | --- | --- |
| Validation manifest | `/Users/choco/Documents/project/docinsights-2026/data/raw/docsem/val/tasks.jsonl` | 217 | `5b6f57a30f4dc8b27873162ca58434c0411fb89f4726b5f2988903344b43443a` | `instance_id`, `user_query`, `document_pdf`, ... |
| Train manifest | `/Users/choco/Documents/project/docinsights-2026/data/raw/docsem/train/tasks.jsonl` | 908 | `6d9cd9087d0c5e30bfc17c83aec30752403d4109fb93d8357f534da425969489` | `instance_id`, `user_query`, `document_pdf`, ... |

`/Users/choco/Documents/project/docinsights-2026/data/raw/docsem/README.md`는 참가자에게 PDF와 패러프레이즈 `user_query`가 제공된다고 명시한다. 따라서 비교의 “기존 Query”는 manifest의 `user_query`로 고정했다.

이미 존재하던 Validation Query 파생 산출물도 확인했다.

| 산출물 | 절대경로 | 수 | SHA-256 | 관련 필드 |
| --- | --- | ---: | --- | --- |
| Claude blind questions | `/Users/choco/Documents/project/docinsights-2026/artifacts/docsem_validation/claude_blind/questions.jsonl` | 217 | `2f21124f656c479096ba313a55e505440370f3867f2fc2929f86f9c9c03c7af5` | `instance_id`, `user_query`, `document_pages_ocr`, `pdf_sha256`, ... |
| Codex blind review | `/Users/choco/Documents/project/docinsights-2026/artifacts/docsem_validation/codex_blind/review.jsonl` | 217 | `5e899155c4aff2a43eabd9f6b6f31ce35235f15a6e68fdf6c24421cfd21d45bb` | `instance_id`, `question_text`, `answer`, `evidence_block_ids`, ... |

이 파생 산출물은 원천 확인에만 사용했다. Codex 전사 호출에는 렌더링된 PDF 페이지만 전달했고 `user_query`, `labels.jsonl`, answer, evidence, 기존 답안 또는 기존 OCR 텍스트를 프롬프트·입력·raw 응답에 넣지 않았다.

## 전사 완전성 결과

| 항목 | Validation | Train |
| --- | ---: | ---: |
| expected / record / unique / ok | 217 / 217 / 217 / 217 | 908 / 908 / 908 / 908 |
| 실패 / 누락 / 중복 / 초과 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 정확히 2페이지 위반 | 0 | 0 |
| ordered `b01`~`b23` 위반 | 0 | 0 |
| raw 응답 / stderr sidecar | 217 / 217 | 908 / 908 |
| verifier 판정 | `valid=true` | `valid=true` |

### Validation 산출물

| 종류 | 절대경로 | SHA-256 |
| --- | --- | --- |
| 전사 JSONL | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-reference.jsonl` | `d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a` |
| raw 디렉터리 | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-reference-raw` | raw 집합 `fb97829c8772cbc8f61abf154045cb92728f101e664e397d5623ffe03d101c7b`, stderr 집합 `041a154e5f4f30df03a67f7f386674b39cfff4d89639f09fb5ac60e675042b53` |
| 검증 보고서 | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-verification.json` | `7caf3df6c198a67d0a59924bc2b616c5605a5b2b6575bc374b337dad83f84578` |

Validation은 기존 checkpoint의 8/217에서 `--resume`으로 이어서 완료했다. 217/217 게이트 통과 증거를 생성한 다음에만 Train을 시작했다.

### Train 산출물

| 종류 | 절대경로 | SHA-256 |
| --- | --- | --- |
| 전사 JSONL | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-train-reference.jsonl` | `2e06fb6bac61601776049c03e3c20f3dcf905feee77510342f67cf19bc1f0558` |
| raw 디렉터리 | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-train-reference-raw` | raw 집합 `21f025236418992ae828c829549177693e917a092e8634d5ccd4e436b7a85205`, stderr 집합 `e450d2288acdb03711bed716f76197df587faf3f7827c87efc19f2a395c8c4e6` |
| 검증 보고서 | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-train-verification.json` | `9122c67ccb455216c58223cad8a88a96ce0d394c930700c58ba419fbf921eb77` |

Train 최초 실행은 908건 중 906건 성공, 2건 실패였다. `task_000382`는 300초 timeout, `task_000427`은 Codex exit 1로 기록한 뒤 `--resume --retry-failed --workers 2`로 두 항목만 제한 재시도해 모두 성공했다.

## Query 전수 비교 결과

비교 순서는 다음과 같다.

1. raw exact match
2. Unicode NFKC 후 공백과 줄바꿈만 단일 공백으로 정규화한 exact match
3. 불일치에만 문자 단위 diff, PDF 근거 block/page, 원인 범주 기록

의미 교정, 의역, 숫자·단위 정규화는 하지 않았다. PDF 텍스트가 이미지형이면 `pdftotext`의 빈 결과를 확인한 뒤 Poppler 200 DPI 렌더링과 Tesseract `eng --psm 6`을 근거 추출에 사용했다.

| split | raw exact | normalized exact | mismatch | undetermined | 실제 내용 차이 | OCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0 | 0 | 217 | 0 | 166 | 51 |
| Train | 0 | 0 | 908 | 0 | 732 | 176 |

`actual_content_difference`는 기존 `user_query`가 메타 수준의 패러프레이즈이고 PDF에는 실제 정량 시나리오 질문이 적혀 있어 양쪽 의미·문자열이 다른 경우다. `ocr`은 Codex 전사에서 복원한 Query와 독립 Tesseract PDF 근거 문자열 사이에도 차이가 관찰된 경우다. `ocr` 분류는 사람 정답 판정이 아니다.

| 종류 | 절대경로 | SHA-256 |
| --- | --- | --- |
| Validation JSONL | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-query-comparison.jsonl` | `c66b57bbc55fa34e663beced094234c6031b05ef49895893d3f069d1b5c9e795` |
| Validation Markdown | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-query-comparison.md` | `a6fb226cee4de3014663123582c16ce941914895942e1d17b46896122775a23a` |
| Train JSONL | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-train-query-comparison.jsonl` | `99d642ace2e5080222c75ca6b5b1d6ea78707ea9261ab267a7d67c655e2f4628` |
| Train Markdown | `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-train-query-comparison.md` | `941c253b8a35c58cd41c66e356f4e4c0ddeaae5cf69153d2156999332999c3d7` |

Train 최초 비교에서 `task_000502`와 `task_000632`가 각각 고정 안내문 `afternoon`→`aftemoon`, marker `b09`→`bO09` Tesseract 변형 때문에 미결정이었다. Query 본문을 교정하지 않고 알려진 고정 lead-in과 block marker 판독에만 국소 허용 규칙을 적용하고 회귀 테스트를 추가해 최종 미결정을 0으로 만들었다.

## 재현 명령

아래 명령은 저장소 루트 `/Users/choco/.codex/worktrees/bed4/docinsights-2026`에서 실행한다. 전사 실행은 기존 checkpoint와 raw 응답을 보존하는 `--resume`을 기본으로 한다.

```bash
uv run docinsights-ocr codex-reference \
  /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/val/tasks.jsonl \
  artifacts/ocr/codex-validation-reference.jsonl \
  --documents-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem \
  --raw-dir artifacts/ocr/codex-validation-reference-raw \
  --model gpt-5.6-sol --model-config 'model_reasoning_effort="high"' \
  --dpi 200 --workers 2 --timeout-seconds 300 --resume

uv run docinsights-ocr codex-verify \
  /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/val/tasks.jsonl \
  artifacts/ocr/codex-validation-reference.jsonl \
  artifacts/ocr/codex-validation-reference-raw \
  --documents-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem \
  --report artifacts/ocr/codex-validation-verification.json --timeout-seconds 300

uv run docinsights-ocr codex-reference \
  /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/train/tasks.jsonl \
  artifacts/ocr/codex-train-reference.jsonl \
  --documents-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem \
  --raw-dir artifacts/ocr/codex-train-reference-raw \
  --model gpt-5.6-sol --model-config 'model_reasoning_effort="high"' \
  --dpi 200 --workers 2 --timeout-seconds 300 --resume

# 실패 레코드가 있을 때만 제한 재시도
uv run docinsights-ocr codex-reference \
  /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/train/tasks.jsonl \
  artifacts/ocr/codex-train-reference.jsonl \
  --documents-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem \
  --raw-dir artifacts/ocr/codex-train-reference-raw \
  --model gpt-5.6-sol --model-config 'model_reasoning_effort="high"' \
  --dpi 200 --workers 2 --timeout-seconds 300 --resume --retry-failed

uv run docinsights-ocr codex-verify \
  /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/train/tasks.jsonl \
  artifacts/ocr/codex-train-reference.jsonl \
  artifacts/ocr/codex-train-reference-raw \
  --documents-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem \
  --report artifacts/ocr/codex-train-verification.json --timeout-seconds 300

uv run docinsights-ocr codex-query-compare \
  /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/val/tasks.jsonl \
  artifacts/ocr/codex-validation-reference.jsonl \
  artifacts/ocr/codex-validation-query-comparison.jsonl \
  artifacts/ocr/codex-validation-query-comparison.md \
  --documents-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem \
  --split-name validation --workers 4 --timeout-seconds 60

uv run docinsights-ocr codex-query-compare \
  /Users/choco/Documents/project/docinsights-2026/data/raw/docsem/train/tasks.jsonl \
  artifacts/ocr/codex-train-reference.jsonl \
  artifacts/ocr/codex-train-query-comparison.jsonl \
  artifacts/ocr/codex-train-query-comparison.md \
  --documents-root /Users/choco/Documents/project/docinsights-2026/data/raw/docsem \
  --split-name train --workers 4 --timeout-seconds 60

uv run pytest -q
uv run ruff check .
git diff --check
```

## 구현 중 발견한 결함과 회귀 방지

- 이미지형 PDF에서 `pdftotext`가 form feed만 반환할 때 Tesseract fallback이 동작하도록 했다.
- Tesseract의 marker 오인식 `bO6`, `bO09`를 marker 위치에서만 `b06`, `b09`로 복원했다.
- raw 디렉터리 파일 집합, raw/record canonical JSON, provenance, fingerprint, PDF/PNG SHA를 verifier가 엄격히 검사한다.
- query 비교가 참조 전사와 현재 PDF의 SHA binding을 확인하고, 부분 파일 대신 임시 파일 완성 후 원자적으로 게시한다.
- Codex subprocess 실패 시 제한된 stdout/stderr 진단을 실패 레코드에 보존한다.
- 각 결함에 회귀 테스트를 추가했다. 최종 로컬 검증은 `176 passed`, Ruff 통과, `git diff --check` 통과다.

## 검수 한계

- Codex 전사는 human gold가 아닌 `codex-assisted-silver` reference다. 전수 계약 검증은 완전성과 재현 가능한 provenance를 증명하지만 문자 정확도 자체를 사람 정답 수준으로 보증하지 않는다.
- PDF 근거 Query 추출의 독립 비교자는 이미지형 페이지에 Tesseract를 사용한다. `ocr` 범주는 Codex와 Tesseract 사이의 관찰된 차이이며 어느 한쪽이 정답이라는 판정이 아니다.
- normalized exact는 Unicode NFKC와 공백·줄바꿈 통합만 허용한다. 구두점 교정, 의미 교정, 의역, 숫자·단위 변환은 허용하지 않았다.
- 기존 `user_query`는 설계상 PDF 시나리오 Query의 패러프레이즈/메타 질의이므로 exact 0건은 전사 실패율이나 데이터 오류율을 의미하지 않는다.
- 절대경로와 실행 파일 identity가 provenance에 포함된다. 다른 호스트에서 재실행하면 내용이 같아도 run fingerprint 또는 보고서 해시가 달라질 수 있다.

## 독립 verifier

별도 읽기 전용 verifier가 최종 코드와 산출물을 `APPROVED`했다.

- Validation manifest/reference/comparison은 각각 217 unique, Train은 각각 908 unique이며 두 split 모두 missing/duplicate/extra가 0이었다.
- 모든 전사 레코드의 status/schema, page image 순서 1·2, ordered `b01`~`b23`, PDF SHA, raw SHA, run fingerprint 불일치가 0이었다. raw JSON/stderr 수도 Validation 217/217, Train 908/908이며 raw 파일 missing/extra가 0이었다.
- verifier 보고서의 `valid=true`, raw/stderr 집합 SHA를 재계산해 일치를 확인했다.
- Query 비교의 manifest `user_query`, source/PDF SHA, diff/evidence, category를 재계산했다. Validation `0/0/217/0`과 category `actual_content_difference=166`, `ocr=51`; Train `0/0/908/0`과 category `actual_content_difference=732`, `ocr=176`이 모두 일치했다.
- 이 보고서에 나열된 전사·검증·Query 산출물 12개와 기존 Validation 파생 산출물 2개의 SHA-256이 모두 일치했다.
- 독립 실행한 `uv run pytest -q`는 `176 passed`, `uv run ruff check .`와 `git diff --check`는 통과했다. Pyright 실행 파일은 설치되어 있지 않아 새로 설치하지 않았으며, 앞선 독립 코드 리뷰에서는 Pyright `0 errors, 0 warnings`가 확인됐다.

Validation raw 최종 시각은 Train raw 최초 시각보다 앞서 전사 순서가 확인된다. 현재 Validation 검증 보고서는 최종 코드로 재검증하며 덮어쓴 파일이므로 “Train 시작 직전 실행된 최초 게이트”의 파일 시각은 보존되지 않는다. 게이트 순서는 실행 로그와 본 작업의 checkpoint 진행 기록으로 확인했으며, 최종 보고서는 동일 Validation 217개에 대해 더 엄격한 verifier를 재실행한 결과다.
