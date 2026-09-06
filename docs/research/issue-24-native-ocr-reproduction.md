# Issue #24 Native Paddle 재현 구성

이 구성은 OCR 비교에서 사용한 macOS arm64 / Python 3.11.6의 Native Paddle baseline이다. 최종 처리량 최적화 후보와 구분한다. 모델 변경이나 ONNX 변환 결과를 baseline 점수로 대신 표시하지 않는다.

## 환경과 모델 준비

새 환경에서 다음 순서로 실행한다. 기존 실험 환경이 있으면 이를 덮어쓰지 않는다.

```sh
uv venv --python 3.11.6 data/issue24/ppocr-env
uv pip install --python data/issue24/ppocr-env/bin/python paddlepaddle==3.2.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
uv pip install --python data/issue24/ppocr-env/bin/python -r requirements/solution-paddle-ocr.txt
uv run python scripts/setup_solution_paddle_ocr.py --model-dir data/issue24/ppocr-models
uv run python scripts/setup_solution_paddle_ocr.py --model-dir data/issue24/ppocr-models --verify-only
```

`requirements/solution-paddle-ocr.txt`는 측정 환경에서 관측한 59개 배포 패키지의 정확한 버전을 고정한다. 다른 OS나 Python 버전의 호환성을 검증한 범용 lock은 아니다. 공개 Paddle CPU index는 Paddle 설치에만 사용한다. 시스템의 `pdftoppm`도 필요하며 실행기는 실제 경로, 버전과 실행 파일 해시를 기록한다.

모델 준비 스크립트는 기존 checksum-first downloader를 재사용한다. Detector와 English recognizer를 각각 고정된 Hugging Face revision에서 받고, `.gitattributes`를 포함한 총 12개 파일의 SHA-256을 검증한다. 기존 파일의 해시가 다르면 덮어쓰지 않고 실패한다. 모델 준비는 문항이나 라벨을 다운로드하지 않는다.

| 역할 | 저장소 | Revision |
|---|---|---|
| Detector | `PaddlePaddle/PP-OCRv5_mobile_det` | `0d63e78e2b680928f6b1747d76a08db6e645efb7` |
| Recognizer | `PaddlePaddle/en_PP-OCRv5_mobile_rec` | `267c36e24c331595590fe7bd72bde2436fd286f2` |

두 모델은 Apache-2.0이다. 출처와 비교 지표는 [OCR 실측 보고서](issue-24-ocr-comparison.md)에 있다. 가중치 파일별 기대 해시는 [모델 준비 스크립트](../../scripts/setup_solution_paddle_ocr.py)에 고정되어 있다.

## 실제 확인한 범위

2026-09-07 KST에 기존 모델의 12/12 파일 검증과 별도 임시 디렉터리로의 공개 다운로드·12/12 checksum 검증을 통과했다. 임시 다운로드 사본은 검증 뒤 정리했고 같은 모델 바이트는 기존 실험 경로에 보존했다. private 진단 기록은 `artifacts/solution-records/issue24/diagnostics/native-setup-download-smoke.json`이다. 전체 59개 패키지 환경을 별도 새 환경에 재설치한 시험은 하지 않았다.

준비 스크립트는 Ruff와 Pyright를 통과했다. 모델 준비의 성공은 전체 문항 풀이 성공이나 제출 준비 완료를 의미하지 않는다.

## 선정한 동일 가중치 ONNX CPU 구성

Baseline 모델의 Paddle graph/parameters를 변환하고 PaddleX 전·후처리는 유지한다. 실제 실행 Python은 `ppocr-env`이며, ONNX Runtime 1.23.2만 별도 `rapidocr-env`의 site-packages에서 가져온다. Runner는 실제 import 경로·버전·RECORD와 설치 파일 해시를 검사하고 Native 환경 패키지를 우선한다. RapidOCR 배포 모델은 이 구성에 사용하지 않는다.

새 환경의 추가 준비 명령은 다음과 같다. 변환용 환경은 production 실행에 필요하지 않으며 기존 환경을 덮어쓰지 않는다.

```sh
uv venv --python 3.11.6 data/issue24/rapidocr-env
uv pip install --python data/issue24/rapidocr-env/bin/python -r requirements/solution-ocr.txt
uv venv --python 3.11.6 data/issue24/paddle-onnx-export-env
uv pip install --python data/issue24/paddle-onnx-export-env/bin/python paddlepaddle==3.2.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
uv pip install --python data/issue24/paddle-onnx-export-env/bin/python -r requirements/solution-paddle-onnx-export.txt
data/issue24/paddle-onnx-export-env/bin/python scripts/setup_solution_paddle_onnx.py --converter data/issue24/paddle-onnx-export-env/bin/paddle2onnx
python3 scripts/setup_solution_paddle_onnx.py --verify-only
```

변환은 Paddle2ONNX 2.1.0, opset 11, ONNX checker 활성화, optimizer 미적용이다. 입력 graph/parameters와 출력 두 ONNX의 알려진 SHA를 검증하며 기존 파일이 다르면 덮어쓰지 않는다. 새 변환은 임시 경로에서 완료하고 gated 출력 SHA와 같을 때만 저장한다. 준비 스크립트를 추가한 시점에는 기존 production 경로의 두 출력에 대해 `--verify-only`와 대상 테스트 4개를 통과했고, 새 환경 설치·고비용 재변환은 반복하지 않았다. 따라서 이 문서는 독립적인 새 설치 재현 성공을 주장하지 않는다. 출력 report의 `fresh_byte_identical_reexport_verified_this_run`이 이를 구분한다.

60페이지 출력 동등성과 작은 CPU capacity probe는 [실측 보고서](issue-24-ocr-comparison.md)의 별도 표를 따른다. Production은 CPU EP, intra-op 4, inter-op 1, sequential execution, enable-all graph optimization, OCR worker 1개로 동결한다.
