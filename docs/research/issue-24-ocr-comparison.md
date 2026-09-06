# Issue #24 OCR 실측 비교 (2026-09-07)

정확도 우선 실행 후보로 **Native Paddle PP-OCRv5**를 선택했다. 사전에 정한 Held-out 기준에서 ID 전사 F1과 본문 CER/WER가 가장 좋았다. Apple Vision은 훨씬 빠르고 숫자 F1도 비슷했지만, 이 표본의 ID·본문 인식에서는 Paddle보다 낮았다. 이는 아래 세 구성과 합의 가능한 영역의 비교이며 대회 정답률·Joint Accuracy·예상 순위가 아니다.

## Held-out: 선택 기준

| OCR | ID F1 ↑ | 숫자 F1 ↑ | CER ↓ | WER ↓ | 페이지 중앙값 | p95 | 실패 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native Paddle PP-OCRv5 | 77.21% | 91.77% | 3.91% | 7.66% | 42.32초 | 46.82초 | 0/40 |
| Apple Vision accurate | 63.22% | 91.97% | 8.46% | 11.63% | 1.01초 | 1.38초 | 0/40 |
| RapidOCR PP-OCRv5 ONNX | 31.05% | 69.16% | 22.41% | 33.22% | 9.21초 | 14.88초 | 0/40 |

Held-out은 20개 문서의 40페이지다. 120개 고정 영역 중 본문 28개(명확한 빈 영역 1개 포함), ID 49개, 숫자 80개가 두 독립 전사에서 합의됐다. ID 191회, 숫자 1,028회, 본문 47,692자/7,403단어가 각 지표의 기준이다. ID는 줄 앞의 콜론 제목 토큰을 세므로 Evidence ID의 대리 지표이며 일반 콜론 레이블도 포함될 수 있다.

문서 단위 paired bootstrap 1,000회, seed 240907의 Held-out 95% 구간은 다음과 같다. 차이는 Paddle − Apple이며 percentage point 단위다.

| 지표 차이 | 점 추정 | 95% 구간 |
|---|---:|---:|
| ID F1 | +14.00 pp | +5.24 ~ +21.85 pp |
| 숫자 F1 | -0.20 pp | -2.52 ~ +2.06 pp |
| CER | -4.55 pp | -7.42 ~ -2.08 pp |
| WER | -3.97 pp | -6.95 ~ -1.30 pp |

숫자 F1 차이의 구간은 0을 포함하므로 숫자 인식의 우열이나 동등성을 확정하지 않는다. ID와 본문 지표는 이 합의 표본에서 Paddle 우세를 지지한다. 더 어려운 불일치 영역까지 같은 격차가 유지된다고 가정하지 않는다.

## Train: 별도 전이 확인

| OCR | ID F1 ↑ | 숫자 F1 ↑ | CER ↓ | WER ↓ | 페이지 중앙값 | 실패 |
|---|---:|---:|---:|---:|---:|---:|
| Native Paddle PP-OCRv5 | 98.89% | 100.00% | 2.39% | 4.22% | 23.20초 | 0/20 |
| Apple Vision accurate | 100.00% | 100.00% | 2.98% | 4.97% | 0.69초 | 0/20 |
| RapidOCR PP-OCRv5 ONNX | 93.51% | 100.00% | 2.16% | 3.84% | 7.58초 | 0/20 |

Train 10문서/20페이지에서 본문 35/60, ID 60/60, 숫자 55/60 영역이 합의됐다. 기준은 ID 227회, 숫자 333회, 본문 25,264자/3,884단어다. Train의 숫자 F1 100%는 OCR 표본 수치이며 908개 train 문제의 정답률이 아니다. Train 수치로 Held-out의 선택을 바꾸지 않았다.

정밀도·재현율·page-macro·분모·전체 60페이지 합산 지표는 [CSV](issue-24-ocr-comparison.csv)에 있다.

## 방법과 한계

- 공식 PDF만 사용했다. 문서·페이지는 결과를 보기 전 SHA 순서로 고정했고, 이전 Held-out 파일럿 두 문서는 제외했다. 질문, train 라벨, v24 답 또는 포털 점수는 OCR 비교에 넣지 않았다.
- 모든 후보는 동일한 175dpi 무손실 PNG를 받았다. 두 reference reader는 같은 페이지와 세 고정 crop만 보고 독립 전사했다. 모델은 gpt-5.6-sol/high이며 후보 출력과 서로의 답을 볼 수 없었다. 모델 합의에 따른 silver이며 사람의 gold가 아니다.
- NFC와 공백만 정규화했다. 숫자와 ID는 양쪽에 같은 lexical parser를 적용했고 반복 출현을 유지했다. 영역 경계를 가로지르는 줄은 geometry로 제외했으며 후보별 제외 incidence도 CSV에 남겼다.
- 읽을 수 없거나 불일치하는 영역은 지표별로 제외했다. 15 reader-region 관측이 uncertain이며, 이는 13개의 서로 다른 영역에 해당한다. 따라서 특히 본문 지표는 전체 PDF를 대표하는 오류율이 아니다.
- 같은 16GiB/8-core Mac, macOS14.6 arm64에서 후보별 직렬 추론 시간을 기록했다. Apple은 플랫폼 가속을 사용할 수 있고 native Paddle/ONNX는 서로 다른 실행 경로다. 동일 CPU 조건이나 완전히 격리한 호스트의 속도 비교가 아니다. 중앙값/p95는 초기화 후 성공한 모든 페이지(첫 페이지 포함) 기준이다.
- 후보별 엔진 내부 `initialization_seconds`는 Apple 0.021초, Rapid 0.883초, Paddle 2.203초다. 별도로 기록한 첫 페이지 cold-start 시간은 각각 1.643초, 15.761초, 58.209초다. 다른 프로그램과 cloud-reader orchestration이 일부 겹쳤으므로 처리량은 본 실행에서 다시 측정한다.

## 구성과 출처

- Native: paddlepaddle 3.2.0, paddleocr 3.3.2, paddlex 3.3.13, CPU threads 2, recognition batch 6. [PP-OCRv5 mobile detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det/tree/0d63e78e2b680928f6b1747d76a08db6e645efb7), [English recognizer](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec/tree/267c36e24c331595590fe7bd72bde2436fd286f2), Apache-2.0.
- Rapid: rapidocr 3.9.2, onnxruntime 1.23.2, 공개 RapidAI v3.9.2 PP-OCRv5 mobile detector/English recognizer ONNX. [고정 upstream manifest](https://raw.githubusercontent.com/RapidAI/RapidOCR/095232a4c94f7f0e6600ba5bba1177010ad696d4/python/rapidocr/default_models.yaml). Native HF 가중치와 정확히 같은 변환 계보라는 증거는 없으므로 동일 모델 바이트라고 주장하지 않는다.
- Apple: Vision accurate, en-US, language correction enabled, request revision 3, Swift5.10/macOS14.6. 독점 플랫폼 OCR이며 Mac에서 공개 OCR를 실행하는 것과 별개다.
- 모델 계열 인용: [PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595). 상세 설정과 모델·런타임 해시는 private candidate JSON에 보존한다.

## 다음 실행

선택한 native 구성을 출처 검증 실행기에 통합하고 동일 validation 한 문항을 다시 동결·비교한 뒤 Held-out 파일럿을 실행한다. Native 속도로 전체 16,383페이지를 처리하는 데에는 상당한 시간이 필요하므로, 같은 가중치·전처리와 출력 품질을 보존할 수 있는 가속 경로도 별도 확인한다. 가속 설정이 실제로 일치하는지 확인하기 전에는 측정된 Native 성능을 그 설정의 성능으로 옮겨 적지 않는다. 전수 결과 생성과 공식 정답률 평가는 아직 완료되지 않았다.

## 후속 구성 실험

아래는 baseline 결과를 본 뒤 수행한 posthoc 실험이다. 같은 60페이지와 동결된 silver reference를 사용했으며 baseline 세 파일과 위 표는 보존했다.

| Apple 설정 | Held-out ID F1 ↑ | 숫자 F1 ↑ | CER ↓ | WER ↓ |
|---|---:|---:|---:|---:|
| 언어 교정 켬 (baseline) | 63.22% | 91.97% | 8.46% | 11.63% |
| 언어 교정 끔 (후속 후보) | 69.57% | 92.04% | 8.71% | 13.04% |

언어 교정을 끄면 ID 점 추정치는 개선됐지만 본문 오류는 늘었다. Native − 교정 끔의 ID F1 차이 95% 구간은 −1.51~+15.86 pp로 우열이 확정되지 않았다. CER/WER 차이 구간은 각각 −7.79~−2.38 pp, −8.25~−2.74 pp로 Native의 본문 우세를 지지한다. 교정 끔 실행은 다른 CPU 실험과 겹쳤으므로 이번 속도를 baseline과 직접 비교하지 않는다. 이 결과로 Native 기준 구성을 교체하지 않았다.

Native의 실제 합성된 detector 설정은 `limit_side_len=64`, `limit_type=min`, `max_side_limit=4000`이며 표본 페이지를 축소하지 않는다. 모델의 `inference.yml`만 읽어 `960/max`라고 해석하면 실제 OCR pipeline과 달라진다. Rapid와의 차이에는 정규화, DB 후처리, 런타임 및 가중치 변환 계보가 함께 있으므로 ONNX 형식 자체를 품질 차이의 원인으로 단정하지 않는다.

Native baseline 환경과 고정 모델 준비 방법은 [재현 구성](issue-24-native-ocr-reproduction.md)에 정리했다. 교정 끔 candidate SHA-256은 `5e11c32aee2078d6177e63ab4e773ae0cce98eeef8e8c1cca48cb80ebb1cd077`, Native와의 paired comparison SHA-256은 `8d1d99f302601cca9e3af444183a2e1103c692330b7e7552d3a99ce869219f10`이다.

## 감사 해시

- Corpus manifest: `3627ef1d3589e320ea12b353fda907a1ccd283a46b3be8a76e29864e894fafd9`
- Scorer: `b878a3556390d346ba34d442e391e4683b40df019593b47523e8153c68699b4c`
- Comparison JSON: `671704b508996489785366f2d962655f135066a660a4d2e4b6bad2ea3ce9e267`
- Native Paddle PP-OCRv5 candidate: `da9d4104c5fa468f4db9a9bd9d6286dc6d75562e3410675e06d540b9079189de`
- Apple Vision accurate candidate: `44d0d108641934c3653c1e5539a47350ebe8391788ab972f83722fb0ac5b782f`
- RapidOCR PP-OCRv5 ONNX candidate: `7da5fe07cc0e05a63b35d9e2cecc72c143599700bb76d2b640515e031d140b41`
