# Issue #8 · Colab·Kaggle CPU와 Mac Studio 실행 계획

## 결론

Validation OCR은 Kaggle CPU를 PP-OCRv5의 주 분할 실행 환경으로, Colab CPU를 독립 재현 또는 overflow 환경으로, Mac Studio를 입력 관리·대형 모델 실행·결과 병합 환경으로 사용합니다. 아직 사양을 확인하지 않은 Mac Studio를 고정 기준선이라고 부르지 않으며, Kaggle과 Colab 결과도 runtime sidecar가 동일 cohort임을 통과하지 않으면 한 결과로 섞지 않습니다.

## 공식 자원 제약

| 환경 | 공식적으로 확인된 조건 | 이번 실험의 역할 |
| --- | --- | --- |
| Kaggle Notebook CPU | 4 CPU cores, 30GB RAM, 최대 12시간, `/kaggle/working` 자동 저장 20GB | PP-OCRv5 mobile 8-shard 주 실행과 clean `Save & Run All` 재현 |
| Google Colab CPU | 자원·idle timeout·최대 VM 수명은 동적이며 보장되지 않음, 무료 런타임은 조건에 따라 최대 12시간 | Kaggle과 분리한 독립 재현 또는 중단된 shard의 별도 cohort 실행 |
| Mac Studio | 사용자가 연결할 Apple Silicon 장비이며 사양·OS·디스크는 아직 미확인 | 입력 묶음 생성, shard 병합·감사, Apple Vision 전수 재현, LightOnOCR·GLM-OCR 실행 |

Colab 무료 또는 compute unit 잔액이 없는 managed runtime에서는 distributed computing worker가 제한될 수 있고 다계정으로 제한을 우회해서는 안 됩니다. 따라서 중앙 worker pool을 만들지 않고, 사용자가 각 노트북에서 명시적인 `SHARD_INDEX` 하나를 선택해 독립적으로 실행합니다. Kaggle도 공식 동시 CPU 세션 수를 보장하지 않으므로 처음에는 2개 shard만 병렬로 확인하고 계정에 허용된 범위 안에서 최대 4개까지 늘립니다.

공식 근거는 [Google Colab FAQ](https://research.google.com/colaboratory/faq.html), [Colab runtime version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html), [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks), [Kaggle package management](https://www.kaggle.com/docs/packages)를 따릅니다.

## 고정 입력

로컬에서 생성한 label-free cloud input은 다음과 같습니다.

| 항목 | 값 |
| --- | --- |
| 입력 archive | `artifacts/ocr/docsem-validation-ocr-input.tar.gz` |
| archive 크기 | 87,640,387 bytes |
| archive SHA-256 | `9fb35e81feead385fedd3a5bd66ca780ca2aaee5b92b2247f75114cfae642967` |
| archive 내부 canonical manifest SHA-256 | `08bb8ef1948bdbb69ceddfc669d31adf7002707cdd149937b04615dae0eb2d3b` |
| 문서 수 | 217 |
| shard 수·크기 | 8개, `[28, 27, 27, 27, 27, 27, 27, 27]` |
| assignment hash | `860337a43768bd3d9d915125b92bf6acfde7dfe907818a3b151ea497ebfeb20d` |

Archive에는 Validation PDF와 allowlist 여섯 필드(`instance_id`, `user_query`, `document_pdf`, `input_pdf_sha256`, `split`, `split_seed`)로 다시 직렬화한 canonical manifest만 들어 있습니다. 알 수 없는 top-level 또는 중첩 metadata를 허용하지 않으므로 `answer`, `evidence`, train labels, 기존 OCR 출력, Codex·Claude 전사, 제출 파일과 포털 점수가 함께 복사되지 않습니다. OCR 엔진 호출에는 렌더링된 페이지만 전달되고 `user_query`는 출력 레코드를 조립할 때만 다시 붙습니다.

입력과 shard manifest는 다음 명령으로 다시 생성합니다.

```bash
uv run docinsights-ocr cloud-pack artifacts/ocr/validation-manifest.jsonl /absolute/path/to/docsem artifacts/ocr/docsem-validation-ocr-input.tar.gz
uv run docinsights-ocr cloud-shard artifacts/ocr/validation-manifest.jsonl artifacts/ocr/cloud-8 --shard-count 8
```

Shard 배정은 `seed + instance_id + input_pdf_sha256`의 SHA-256 정렬 순위를 8로 나눈 나머지입니다. 이 방식은 manifest 순서, 질의 문구, label과 모델 출력에 독립적이고 shard별 문서 수 차이를 최대 1개로 제한합니다.

## Kaggle·Colab 실행

공용 노트북은 [`notebooks/ocr/cloud_cpu_ppocrv5.ipynb`](../../notebooks/ocr/cloud_cpu_ppocrv5.ipynb)입니다. Kaggle에서는 archive를 private Dataset으로 추가하고 Internet 옵션을 켭니다. Colab에서는 archive를 `/content`로 업로드하거나 Drive에서 로컬 VM으로 한 번 복사한 뒤 `DOCSEM_BUNDLE_PATH`를 지정합니다. Drive의 PDF 217개를 직접 반복해서 읽지 않습니다.

각 실행에서 바꾸는 값은 `SHARD_INDEX` 하나뿐입니다. 노트북은 archive·manifest hash와 immutable repository commit을 실행 전에 확인하고 한 shard에서 PP-OCR pipeline 하나를 유지합니다. `--retry-failed` checkpoint는 매 문서 뒤 원자적으로 갱신되며 미처리 문서를 먼저 완료한 다음 실패 문서를 최대 2회 다시 시도합니다. 처리 수가 늘지 않거나 최종 실패가 남으면 성공으로 끝내지 않습니다.

Colab Drive를 쓰려면 노트북 실행 전에 아래처럼 mount한 뒤 새 실험 디렉터리를 지정합니다. PDF archive는 Drive에서 `/content`로 한 번 복사하는 편이 좋고, checkpoint만 Drive에 직접 기록합니다.

```python
from google.colab import drive
drive.mount("/content/drive")
%env DOCSEM_PERSIST_DIR=/content/drive/MyDrive/docsem-ocr/kaggle-independent-cohort/shard-00
```

Checkpoint 옆에는 현재 VM의 boot/session fingerprint가 저장됩니다. 같은 VM의 kernel 재시작은 재개할 수 있지만 다른 VM 또는 다른 package·commit에서 기존 checkpoint를 이어 쓰면 노트북이 중단됩니다. 새 VM에서 복구해야 하면 기존 결과에 덧붙이지 않고 새 experiment directory에서 해당 shard 전체를 다시 실행합니다.

고정 PP-OCRv5 구성은 다음과 같습니다.

| 구성 요소 | revision |
| --- | --- |
| `PaddlePaddle/PP-OCRv5_mobile_det` | `0d63e78e2b680928f6b1747d76a08db6e645efb7` |
| `PaddlePaddle/en_PP-OCRv5_mobile_rec` | `267c36e24c331595590fe7bd72bde2436fd286f2` |
| PaddlePaddle / PaddleOCR / PaddleX | `3.2.0` / `3.3.2` / `3.3.13` |
| 입력 | 200 DPI PNG, CPU, MKL-DNN 비활성, orientation·unwarping 비활성 |

각 shard에서 회수할 파일은 `result-shard-XX-of-08.jsonl`, `runtime-shard-XX-of-08.json`, `pip-freeze-shard-XX-of-08.txt`입니다. Runtime sidecar에는 platform role, OS·architecture·Python, clean repository SHA, package·model revision, bundle·full/shard manifest·result hash, session fingerprint와 시작·종료 시각이 들어갑니다. Paddle `predict()`는 현재 Python 프로세스 안에서 실행되므로 CLI의 timeout은 renderer에만 적용되고 inference hang을 강제 종료하지 못합니다. 이 한계는 cloud 실행 기록에 남기고 전체 shard session timeout으로 감시합니다.

이 노트북과 archive는 로컬에서 JSON·코드 compile, 고정 hash, 샤딩·병합 회귀 테스트를 통과했지만 Kaggle/Colab의 실제 clean one-document와 full-shard 실행 결과는 아직 없습니다. 각 플랫폼에서 첫 shard를 완료한 뒤 notebook version과 runtime sidecar를 실험 ledger에 추가해야 cloud 성능 수치로 사용합니다.

## Mac Studio 연결과 역할

Mac Studio가 연결되면 비밀번호나 개인 키를 대화에 붙이지 않고 로컬 SSH config의 host alias와 공개키 인증을 사용합니다. Codex가 해당 장비에서 직접 실행되는 경우에는 SSH 없이 그 작업 공간을 사용합니다. 최초 점검은 아래 항목을 읽기 전용으로 확인합니다.

```bash
system_profiler SPHardwareDataType
sysctl -n hw.memsize
sw_vers
df -h /
python3 --version
xcrun swift --version
```

LightOnOCR-2-1B의 weight 파일은 약 2.01GB이고 GLM-OCR 8-bit MLX weight는 약 1.58GB이므로, 모델 캐시·렌더링 이미지·출력을 고려해 최소 20GB의 여유 공간을 확보합니다. 16GB 장비에서도 단일 스모크는 시도할 수 있지만 전수 실행과 병렬 비교에는 32GB 이상을 권장합니다. 장비 사양을 확인하기 전에는 batch size, 동시 worker 수 또는 전체 실행 시간을 확정하지 않습니다.

Mac Studio는 다음 순서로 사용합니다.

1. cloud archive와 manifest SHA-256을 다시 확인합니다.
2. Kaggle shard 8개와 runtime sidecar 8개를 같은 디렉터리에 모읍니다.
3. `cloud-merge`로 runtime cohort, clean repository SHA, bundle·manifest·result hash, 누락·중복·오배정·PDF SHA·질의·split·OCR 구성과 strict JSONL schema를 fail-closed로 검사합니다.
4. 병합한 PP-OCRv5 결과를 Tesseract, Apple Vision, Codex-assisted silver와 비교하되 agreement를 accuracy로 부르지 않습니다.
5. 고정 6문서 smoke set에서 LightOnOCR와 GLM-OCR을 먼저 실행하고 자원·출력 계약을 통과한 모델만 217개 전수 후보로 올립니다.

## 병합과 판정

Kaggle cohort 8개가 모두 도착하면 Mac Studio에서 다음과 같이 병합합니다.

```bash
uv run docinsights-ocr cloud-merge artifacts/ocr/validation-manifest.jsonl artifacts/ocr/paddleocr-kaggle-200dpi.jsonl artifacts/ocr/kaggle/result-shard-*-of-08.jsonl --runtimes artifacts/ocr/kaggle/runtime-shard-*-of-08.json --report artifacts/ocr/paddleocr-kaggle-200dpi.merge.json
uv run docinsights-ocr hash artifacts/ocr/paddleocr-kaggle-200dpi.jsonl --output artifacts/ocr/paddleocr-kaggle-200dpi.hash.json
uv run docinsights-ocr compare artifacts/ocr/codex-assisted-reference.jsonl artifacts/ocr/paddleocr-kaggle-200dpi.jsonl --output artifacts/ocr/codex-vs-paddleocr-kaggle.json
```

`cloud-merge`는 모든 217개 ID가 정확히 한 번 존재하고 지정 shard에 있으며 manifest의 query·PDF SHA·split, runtime cohort와 OCR 구성 fingerprint가 동일하고 strict schema를 통과할 때만 최종 파일을 원자적으로 게시합니다. 기본 canonical merge는 실패 레코드가 하나라도 있으면 기존 출력도 건드리지 않고 거부합니다. 실패 전달 기록을 보존해야 하는 진단 작업에서만 명시적으로 `--allow-failed`를 사용하며 이를 canonical 후보로 취급하지 않습니다.

Codex-assisted 전사는 human gold가 아니라 silver reference입니다. 모델 순위는 전체 agreement 하나로 정하지 않고 `bNN` block 순서, 숫자·부호·소수점·통화 glyph, evidence 후보 영역의 visual review를 함께 보고합니다.

## 중단과 복구

- 같은 VM에서 kernel만 재시작한 경우에는 session state와 checkpoint를 유지한 채 실행 셀을 다시 시작할 수 있습니다.
- VM이 교체되면 기존 checkpoint를 이어 쓰지 않고 새 experiment directory에서 해당 shard 전체를 다시 실행합니다. 서로 다른 VM이 만든 레코드를 하나의 runtime sidecar로 포장하지 않습니다.
- `--resume --retry-failed`는 입력·engine·model·renderer뿐 아니라 immutable pipeline revision을 포함한 run fingerprint를 확인한 뒤 이전 성공 레코드를 건너뜁니다.
- shard count, seed, 모델 revision, DPI 또는 renderer를 바꾸면 기존 checkpoint를 재사용하지 않고 새 experiment directory를 만듭니다.
- Kaggle과 Colab 결과를 섞어야 할 상황이 생기면 먼저 각각 별도 cohort로 완성하고 비교합니다. 일부 문서를 플랫폼 A, 나머지를 플랫폼 B에서 실행한 혼합 결과는 플랫폼과 문서 난이도가 교란되므로 논문 결과로 사용하지 않습니다.
