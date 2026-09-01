# DocInsights 2026 · DocSem

이 저장소는 [DocInsights 2026 Shared Task](https://docinsights-workshop.github.io/docinsights-2026/shared-task/)의 **DocSem** 과제를 수행하고 실험 결과를 관리하기 위한 작업 공간입니다.

> 기준일: 2026-08-29. 일정과 제출 규정은 바뀔 수 있으므로 제출 전에는 공식 워크숍 페이지와 제출 포털을 다시 확인하세요. 제출과 최종 규정의 기준은 공식 포털입니다.

## 연구 기록

- [소형 OCR 모델 탐색 및 DocSem 고정 사례 비교](research/ocr-small-models/report.md)

## 과제 개요

DocSem은 **근거 귀속(evidence attribution)을 포함한 문서 기반 정량 추론** 과제입니다. 각 인스턴스에는 PDF 문서와 그 문서에 근거한 패러프레이즈 질의 `user_query`가 주어집니다. 시스템은 다음을 수행해야 합니다.

1. PDF에서 질의와 관련된 정량적 구절을 찾습니다.
2. 해당 구절만을 근거로 요청된 수치 답을 계산합니다.
3. 최종 답과 이를 직접 뒷받침하는 PDF 블록 ID를 반환합니다.

파일명, 문서 메타데이터 또는 외부 원문 질의 검색으로 답을 추론해서는 안 됩니다. 목표 시나리오가 명시적으로 요구하지 않는 한 서로 다른 구절의 사실도 임의로 결합하지 않습니다. PDF의 각 콘텐츠 블록 앞에는 `b01`과 같은 식별자가 표시되며, 제출 시 이 식별자를 근거로 사용합니다.

## 공식 자료

| 자료 | 용도 |
| --- | --- |
| [워크숍 Shared Task 페이지](https://docinsights-workshop.github.io/docinsights-2026/shared-task/) | 과제 개요, 전체 일정, 최신 공지 |
| [Dataset and guide](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data) | 공개 데이터 미러, 파일 구성, 로딩 예시 |
| [Submission portal](https://amitbcp-docsem-docinsights.hf.space/) | validation/test 예측 제출 및 리더보드 |
| [Canonical source](https://github.com/oracle-samples/gsm-sem/tree/main/docsem) | 원본 participant release |
| [Participant instructions](https://github.com/oracle-samples/gsm-sem/blob/main/docsem/PARTICIPANT_INSTRUCTIONS.md) | 입출력 형식과 평가 규칙의 원문 |

Hugging Face 데이터셋은 canonical source의 participant release를 미러링합니다. 원본 규칙을 판단할 때는 canonical source와 participant instructions를 우선하고, 편리한 다운로드와 Python 로딩에는 Hugging Face 미러를 사용할 수 있습니다.

재현 가능한 실험의 기준 버전은 다음과 같습니다.

- Canonical release: [`oracle-samples/gsm-sem@332158b`](https://github.com/oracle-samples/gsm-sem/tree/332158b2549e7e8a1186e2ae3a922669e9018808/docsem)
- Hugging Face mirror: [`amitbcp/docinsights-2026-shared-task-data@b171c5a`](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data/tree/b171c5ad488f0c8c50df05951a5b288ea50e9501)

데이터 카드에 따르면 미러의 1,125개 PDF는 canonical release와 byte-identical입니다. 실험 기록에는 사용한 두 revision SHA를 함께 남깁니다.

## 데이터 구성

| Split | 인스턴스 수 | 공개 라벨 | 용도 |
| --- | ---: | --- | --- |
| `train` | 908 | 있음 | 로컬 개발과 평가 |
| `validation` / `val` | 217 | 없음 | 공식 validation 리더보드 제출 |

공개 패키지에는 총 1,125개 PDF가 포함됩니다. Hugging Face의 `tasks` config는 train/validation 입력을, `labels` config는 train 라벨을 제공합니다. Validation 라벨은 주최 측이 비공개로 보관합니다.

Canonical source의 주요 파일은 다음과 같습니다.

```text
docsem/
├── PARTICIPANT_INSTRUCTIONS.md
├── README.md
├── train/
│   ├── tasks.jsonl
│   ├── labels.jsonl
│   └── documents/*.pdf
└── val/
    ├── tasks.jsonl
    └── documents/*.pdf
```

Hugging Face 미러에서는 `document_pdf`가 저장소 루트에서 바로 해석되도록 `train/` 또는 `val/` 접두사가 붙습니다.

## 입출력 형식

입력 `tasks.jsonl`의 각 줄은 하나의 JSON 객체입니다.

```json
{
  "instance_id": "task_000001",
  "user_query": "Use the relevant quantitative passage in this document to determine the requested result.",
  "document_pdf": "documents/task_000001.pdf"
}
```

Train 라벨과 제출 예측은 다음 형태를 사용합니다.

```json
{
  "instance_id": "task_000001",
  "answer": "140",
  "evidence": ["b14"]
}
```

제출 파일은 인스턴스당 JSON 객체 하나를 갖는 JSONL이어야 합니다.

- `instance_id`는 입력의 값과 정확히 일치해야 합니다.
- `answer`에는 설명을 붙이지 말고 최종 답만 넣습니다. 답 자체에 단위가 필요한 경우가 아니라면 단위도 제외합니다.
- `evidence`는 비어 있지 않은 블록 ID 목록이어야 합니다.
- 목표 질문과 계산 입력을 직접 제시하는 데 필요한 블록을 모두 포함합니다.
- 제출 대상 split의 모든 인스턴스를 정확히 한 번씩 포함합니다.

시스템 논문과 실험 기록에는 사용한 모델, 외부 학습 데이터, 검색 리소스, 도구, 프롬프트 전략을 문서화합니다.

## 평가

주 평가지표는 정규화된 `answer` exact-match accuracy입니다. 정규화 과정은 앞뒤 공백과 대소문자를 무시하고, 선행 final-answer 표식을 제거하며, 적용 가능한 경우 수치적으로 같은 소수 표현을 동일하게 취급합니다.

근거는 별도로 평가합니다.

- **Evidence exact block-set match:** 예측한 블록 집합이 정답 집합과 정확히 같은지 평가
- **Evidence F1:** 근거 품질을 확인하기 위한 진단 지표

## 데이터 로드 예시

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

repo_id = "amitbcp/docinsights-2026-shared-task-data"

tasks = load_dataset(repo_id, "tasks")
train_tasks = tasks["train"]
validation_tasks = tasks["validation"]
train_labels = load_dataset(repo_id, "labels")["train"]

first_pdf = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=train_tasks[0]["document_pdf"],
)
```

## 저장소 데이터 분석 도구

이 저장소는 고정된 Hugging Face revision의 DocSem 데이터를 내려받는 CLI와 실제 Query·정답·evidence·PDF 페이지 이미지를 탐색하는 Jupyter Notebook을 제공합니다. Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
uv sync --extra notebook
```

먼저 용량이 작은 train/validation manifest와 공개 train 정답을 내려받습니다.

```bash
uv run docinsights download --manifests-only
```

Notebook을 실행합니다.

```bash
uv run --extra notebook jupyter lab notebooks/01_docsem_data_analysis.ipynb
```

저장소의 Notebook에는 실행한 표, 그래프, PDF 페이지 이미지 결과가 포함되어 있어 GitHub에서도 바로 확인할 수 있습니다. 값을 변경해 다시 실행하면 새로운 분석 결과로 갱신할 수 있습니다.

Notebook은 다음 분석을 포함합니다.

- train task와 공개 정답을 `instance_id`로 결합해 실제 `user_query`, `answer`, `evidence` 확인
- Query 길이, 정답 빈도와 수치 분포, 인스턴스당 evidence block 수 시각화
- 선택한 train 인스턴스의 PDF를 필요한 시점에만 다운로드하고 모든 페이지를 이미지로 렌더링
- PDF 텍스트에서 정답 evidence block ID가 포함된 구절 탐색
- 여러 인스턴스의 첫 페이지 이미지, Query, 정답, evidence를 한 화면에서 비교

전체 공개 데이터와 PDF를 미리 내려받으려면 다음 명령을 사용합니다. 전체 데이터 크기는 약 1.3GB입니다.

```bash
uv run docinsights download
```

다운로드 경로는 `--output /path/to/data`로 변경할 수 있습니다. Notebook의 `DATA_DIR`도 같은 경로로 맞춰야 합니다. 테스트는 실제 네트워크나 전체 데이터 없이 실행됩니다.

```bash
uv run pytest
```

## Qwen용 OCR 전처리

[Issue #8](https://github.com/choco9966/docinsights-2026/issues/8)은 이미지형 PDF에서 답을 추론하지 않고 ordered `bNN` block, 원문, 위치와 confidence만 복원해 Qwen 추론·학습에 전달하는 독립 OCR 계층입니다. OCR 엔진에는 `user_query`, 정답, evidence를 전달하지 않으며 `user_query`는 OCR이 끝난 뒤 Qwen 입력 레코드를 조립할 때만 결합합니다.

Python 환경과 Apple Vision 실행 파일을 준비합니다. Apple Vision은 macOS에서만 사용할 수 있으며 Tesseract 실행에는 `tesseract`와 Poppler의 `pdftoppm`이 필요합니다.

```bash
uv sync --all-groups
mkdir -p artifacts/ocr/bin
swiftc -O -warnings-as-errors tools/apple_vision_ocr.swift -o artifacts/ocr/bin/apple_vision_ocr
```

Validation 217개를 모델 출력과 무관한 SHA-256 순서로 OCR-dev 30개와 OCR-eval 187개로 고정하고, 두 OCR 엔진을 같은 200 DPI 입력에서 실행합니다. 현재 실험에서는 설정 보정에 사용한 `task_000909`가 OCR-eval에 포함됐으므로 187개를 sealed holdout으로 주장하지 않고 217개 전체를 exploratory coverage·agreement 분석에만 사용합니다.

```bash
uv run docinsights-ocr prepare \
  data/raw/docsem/val/tasks.jsonl \
  artifacts/ocr/validation-manifest.jsonl \
  --documents-root data/raw/docsem

uv run docinsights-ocr run \
  artifacts/ocr/validation-manifest.jsonl \
  artifacts/ocr/tesseract-200dpi-psm6-final.jsonl \
  --engine tesseract \
  --dpi 200 \
  --page-segmentation-mode 6 \
  --documents-root data/raw/docsem \
  --timeout-seconds 120

uv run docinsights-ocr run \
  artifacts/ocr/validation-manifest.jsonl \
  artifacts/ocr/apple-vision-200dpi.jsonl \
  --engine apple-vision \
  --dpi 200 \
  --apple-vision-executable artifacts/ocr/bin/apple_vision_ocr \
  --documents-root data/raw/docsem \
  --timeout-seconds 120

uv run docinsights-ocr hash \
  artifacts/ocr/apple-vision-200dpi.jsonl \
  --output artifacts/ocr/apple-vision-200dpi.hash.json

uv run docinsights-ocr compare \
  artifacts/ocr/tesseract-200dpi-psm6-final.jsonl \
  artifacts/ocr/apple-vision-200dpi.jsonl \
  --output artifacts/ocr/tesseract-vs-apple-200dpi-final.json

uv run docinsights-ocr codex-silver-evaluate \
  artifacts/ocr/codex-validation-reference.jsonl \
  artifacts/ocr/tesseract-200dpi-psm6-final.jsonl \
  artifacts/ocr/codex-silver-tesseract-evaluation.json \
  --markdown artifacts/ocr/codex-silver-tesseract-evaluation.md
```

`codex-silver-evaluate`는 완전성 검증을 통과한 Codex 전사를 engineering silver reference로 사용해 CER·WER·문자 유사도·block exact·critical-token F1과 0~100 `silver_text_score`를 계산합니다. 이 점수는 human-gold accuracy가 아니며 출력 schema도 그 해석을 강제합니다. 계산식, 함수, 데이터 계약과 Validation 217개 실측은 [`docs/research/issue-8-silver-text-evaluation.md`](docs/research/issue-8-silver-text-evaluation.md), JSON 계약은 [`schemas/codex-silver-evaluation-v1.schema.json`](schemas/codex-silver-evaluation-v1.schema.json)에 있습니다.

Qwen 입력 JSONL은 [`schemas/docsem-ocr-v1.schema.json`](schemas/docsem-ocr-v1.schema.json)을 따르며 `answer`와 `evidence`를 포함하지 않습니다. 전체 실험 설계와 누수 경계는 [`docs/research/issue-8-ocr-benchmark.md`](docs/research/issue-8-ocr-benchmark.md)에 기록합니다.

### Colab·Kaggle CPU와 Mac Studio

PP-OCRv5 mobile은 [`notebooks/ocr/cloud_cpu_ppocrv5.ipynb`](notebooks/ocr/cloud_cpu_ppocrv5.ipynb)에서 217개 Validation 문서를 결정적 shard로 나눠 Kaggle 또는 Colab CPU에서 실행할 수 있습니다. Kaggle Version #3의 1-shard 전수 실행은 217/217을 완료했지만 `task_001108`의 block marker 오류로 strict contract 기준 216/217만 유효해 canonical merge가 실패하므로 raw 결과를 diagnostic으로만 비교합니다. 상세 점수와 해시는 [소형 OCR 비교 보고서](research/ocr-small-models/report.md)에 기록합니다.

```bash
uv run docinsights-ocr cloud-pack artifacts/ocr/validation-manifest.jsonl /absolute/path/to/docsem artifacts/ocr/docsem-validation-ocr-input.tar.gz
uv run docinsights-ocr cloud-shard artifacts/ocr/validation-manifest.jsonl artifacts/ocr/cloud-8 --shard-count 8
uv run docinsights-ocr cloud-merge artifacts/ocr/validation-manifest.jsonl artifacts/ocr/paddleocr-kaggle-200dpi.jsonl artifacts/ocr/kaggle/result-shard-*-of-08.jsonl --runtimes artifacts/ocr/kaggle/runtime-shard-*-of-08.json --report artifacts/ocr/paddleocr-kaggle-200dpi.merge.json
```

공식 자원 제약, checkpoint 복구, 고정 model revision과 Mac Studio 인수 절차는 [`docs/research/issue-8-cloud-compute-runbook.md`](docs/research/issue-8-cloud-compute-runbook.md)에 기록합니다.

## 제출과 주요 일정

- 개발 데이터는 **2026-08-05 릴리스**로 동결되었습니다. 그 전에 내려받았다면 최신 버전으로 갱신해야 합니다.
- 현재는 217개 validation 인스턴스 전체를 포함한 JSONL을 [공식 제출 포털](https://amitbcp-docsem-docinsights.hf.space/)에 제출합니다.
- 포털이 다른 JSON 형식을 일부 처리하더라도 canonical participant instructions가 요구하는 표준 형식은 JSONL이므로, 이 저장소에서는 JSONL만 제출 형식으로 사용합니다.
- 최종 순위는 별도의 held-out test set으로 결정됩니다. 주최 측은 **2026-09-10 최종 제출 마감 5일 전**에 test set을 공개하고 별도 제출을 안내할 예정입니다.
- DocSem 최종 제출 마감은 **2026-09-10**입니다.
- DocSem 또는 Dr.DocBench 시스템 논문 제출 마감은 **2026-09-15 23:59 UTC**이며, archival/non-archival 제출을 모두 받습니다.

제출 전에 JSONL의 스키마, ID 중복·누락, 대상 split과의 일치 여부를 검사합니다.

공식 포털에 제출한 파일의 버전, SHA-256, 평가 점수와 검수 방법은 [DocSem 제출 실험 기록](experiments/submissions.md)에 누적합니다.

```bash
uv run docinsights validate-submission artifacts/submissions/validation.jsonl
```

다른 split이나 경로를 검증할 때는 기준 `tasks.jsonl`을 직접 지정합니다.

```bash
uv run docinsights validate-submission artifacts/submissions/test.jsonl --tasks data/raw/docsem/test/tasks.jsonl
```

세 개 이상의 독립 검수 파일은 각 행에 파일별로 고유한 `run_id`와 `instance_id`, `answer`, `evidence`, `rationale`, `confidence`를 기록하고 다음 명령으로 비교합니다. 도구는 중복된 실제 경로와 `run_id`, 입력과 겹치는 출력 경로를 거부하며 답과 Evidence가 전원 일치한 항목만 `consensus.jsonl`에 기록합니다. 하나라도 다른 항목은 `disagreements.jsonl`로 분리됩니다. `run_id`와 경로 검사는 실수로 같은 결과를 재사용하는 일을 막기 위한 장치이며, 독립 실행 자체의 증명은 아니므로 모델·프롬프트·실행 시각과 원시 응답의 해시는 비공개 실행 기록에 별도로 남깁니다.

```bash
uv run docinsights compare-reviews \
  artifacts/docsem_validation/pass1.jsonl \
  artifacts/docsem_validation/pass2.jsonl \
  artifacts/docsem_validation/pass3.jsonl
```

공식 페이지에는 2026-09-10의 구체적인 마감 시각과 시간대가 기재되어 있지 않습니다. 최종 제출 직전에 [공식 포털](https://amitbcp-docsem-docinsights.hf.space/)의 최신 공지를 다시 확인합니다.

## Qwen 연구 계획

Training 908개 라벨 QA, family-disjoint 내부 평가, Qwen3.5-4B 중심 SFT·GRPO/RLOO와 선택적 9B 실험을 다루는 독립 연구 범위와 72시간 gate는 [Issue #7 Qwen 연구 계획](docs/research/issue-7-qwen-72h-plan.md)에 정리했습니다. SFT/RL은 train ID만 허용하고 `sealed_internal_test`는 H+64 one-shot 전까지 어떤 Qwen lane도 실행하지 않습니다. CUDA가 없으면 4B OCR-text inference·trajectory validator·LoRA SFT smoke와 GRPO feasibility까지만 보장합니다. Qwen3-0.6B는 교차세대 feasibility 하한선이고, Qwen3.8-27B는 128GB 이상 Mac에서 multimodal smoke를 통과할 때도 diagnostic-only upper bound입니다. Codex·Claude 산출물과 Validation·기존 제출·포털 feedback은 supervision이나 모델 선택에 사용하지 않습니다.

## 라이선스와 인용

공개 participant release와 Hugging Face 미러는 [Universal Permissive License v1.0](https://github.com/oracle-samples/gsm-sem/blob/main/LICENSE.txt)에 따라 제공됩니다. 데이터 또는 과제를 사용한 결과물에는 canonical source가 안내하는 GSM-SEM 논문을 인용하세요.

```bibtex
@article{singh2026gsmsem,
  title={GSM-SEM: Benchmark and Framework for Generating Semantically Variant Augmentations},
  author={Jyotika Singh and Fang Tu and Aziza Mirsaidova and Amit Agarwal and Hitesh Laxmichand Patel and Sandip Ghoshal and Miguel Ballesteros and Karan Dua and Yassine Benajiba and Weiyi Sun and Tao Sheng and Graham Horwood and Sujith Ravi and Dan Roth},
  year={2026},
  eprint={2605.07053},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2605.07053}
}
```
