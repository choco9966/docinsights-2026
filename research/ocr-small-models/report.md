# 소형 OCR 모델 탐색 및 DocSem 고정 사례 비교

## 결론

2026-08-31에 Hugging Face Hub의 변동 지표와 공식 모델 카드를 조사하고, 10억
파라미터·2.5 GiB 하드 게이트를 만족하는 네 후보를 immutable revision으로 고정했다.
Kaggle T4의 동일한 200 dpi PNG 두 장에서 실제 선행 smoke를 실행한 결과, GLM-OCR과
Surya는 호출 및 OCR 형식 검사를 통과했고 PaddleOCR-VL은 pinned Transformers 조합의
config 오류로 load에 실패했다. Granite Docling은 호출이 끝났으나 512개의 `!`만 반복해
OCR 유효성 검사를 통과하지 못했다.

이 결과는 `task_000909` 한 건뿐이다. GLM-OCR의 silver agreement CER 0.145579,
WER 0.194690, block F1 0.954545와 Surya의 CER 0.803457, WER 0.799410, block F1
0.357143은 모델 정확도가 아니라 **Codex-assisted silver 전사와의 일치도**다. Codex
전사는 human gold가 아니므로 이 수치로 품질 승자를 선언하지 않는다. GLM은 최대 512
토큰에서 b12~b13을 누락했고, Surya는 b04~b13 및 b16~b23을 누락했다.

## 실행 결과와 자원

- GLM-OCR: load 12.176 s, 55.580 s/doc, 1.080 docs/min, peak RSS 4,463,763,456 B,
  peak VRAM 6,586,189,824 B, output 4,271 B.
- Surya OCR 2: load 16.819 s, 47.771 s/doc, 1.256 docs/min, peak RSS
  4,691,189,760 B, peak VRAM 2,901,400,064 B, output 1,333 B.
- PaddleOCR-VL: `PaddleOCRVLConfig.text_config` 부재로 load 실패. 실패 시점 peak RSS는
  2,093,256,704 B이고 OCR 출력은 0 B다.
- Granite Docling: 생성 호출은 성공했지만 두 페이지 모두 반복문자라 invalid다. 내려받은
  `results.jsonl`에 이 실행의 시간·메모리 행이 없으므로 해당 값은 이유가 있는 N/A로 둔다.
- 비용은 별도 Kaggle 무료 GPU quota다. 기존 PP-OCR shard 런타임은 재사용하거나 중단하지
  않았다. T4 두 장이 할당됐으나 이 실행은 `cuda:0`만 사용했다.

로컬 Apple M3 16 GB 환경은 CPU/MPS 가능성을 구분해 후보표에 기록했지만, 당시 디스크
여유와 swap 상태 때문에 checkpoint를 새로 내려받지 않고 미측정으로 남겼다. CPU와 Apple
Silicon 수치를 Kaggle GPU 수치로 추정하지 않는다.

## Query 및 Issue #8 의존성

OCR inference에는 페이지 픽셀과 고정 OCR 지시만 들어갔다. `user_query`는 inference가
끝난 뒤 tasks manifest에서 instance ID로 join한다. 원문·정규화·SHA-256 passthrough가
각각 217/217이었다. 이는 데이터 계약 보존 검사이며, 문서 속 quantitative scenario와
paraphrased `user_query`가 의미적으로 동일하다는 주장이 아니다.

Issue #8의 현재 validated Codex reference는 217행이며, 고정 사례의 Codex 시간은
41.952초다. 생성기는 매번 `status=ok` 행 수와 존재하는 timing 합계를 다시 집계하므로
부분 집합 상태에서도 sample 수를 과장하지 않는다. 기존 Apple Vision/Tesseract 비교는
217건·434쪽·4,991블록·실패 0건의 운영 대조이며, 두 엔진 간 CER/WER도 gold accuracy가
아니다. 원수치 스냅샷은 `baselines.json`에 있다.

## 재현

```bash
ISSUE8=/path/to/issue-8-worktree
DOCSEM=/path/to/data/raw/docsem
PYTHONPATH=src python -m docinsights_hf_ocr generate \
  --raw-results research/ocr-small-models/raw/results.jsonl \
  --raw-dir research/ocr-small-models/raw \
  --candidates research/ocr-small-models/candidates.json \
  --reference "$ISSUE8/artifacts/ocr/codex-validation-reference.jsonl" \
  --tasks "$DOCSEM/val/tasks.jsonl" \
  --out-dir research/ocr-small-models/generated
python -m pytest -q
ruff check src tests notebooks
ruff format --check src tests notebooks
PYTHONPATH=src python -m docinsights_hf_ocr hash \
  research/ocr-small-models/raw/results.jsonl \
  research/ocr-small-models/generated/comparison.json \
  research/ocr-small-models/generated/comparison.csv \
  research/ocr-small-models/generated/comparison.md
```

Kaggle 실행 소스는 `notebooks/docsem_hf_small_ocr_smoke.py`다. 고정 PDF의 SHA-256이
다르면 즉시 중단하며 다른 split 파일이나 정답 계열 파일을 탐색하지 않는다. 실제 저장된
notebook은 [version 1](https://www.kaggle.com/code/chocozzz/notebooke3d56c2c4c/edit),
`DocSem fixed-sample smoke evidence`다.

## 한계와 다음 단계

현재 실제 표본은 1건이고 max output은 512 tokens다. 다음 단계는 license와 remote-code
위험을 승인한 후보만 S1=6으로 확장하고, 출력 길이 및 모델별 공식 prompt를 사전 고정한 뒤
S2=30, S3=217, R20 결정성 검사를 순서대로 통과시키는 것이다. Human-gold 표본이 생기기
전에는 운영 Pareto(성공·속도·메모리·구조 보존)만 비교한다.
