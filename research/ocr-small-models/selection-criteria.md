# 소형 OCR 후보 선정 및 단계적 표본 기준

후보는 immutable Hugging Face commit과 `model.safetensors` LFS SHA-256/bytes로 고정한다.
metadata hard gate는 parameters 10억 이하, weight 2,684,354,560 B 이하, 명시적 license와
trust 검토다. 이 gate를 통과한 PaddleOCR-VL 1.6, Surya OCR 2, Granite Docling 258M,
SmolDocling 256M preview 네 개를 selected로 기록한다. 1,325,258,240-parameter GLM-OCR은
실행 수치를 보존하는 diagnostic reject다.

실행 결과는 선정 gate와 별도로 판정한다. Paddle은 현재 pinned Transformers에서 load
실패, Granite는 inference 성공/valid OCR 실패, Surya와 SmolDocling은 inference 및 marker
검사를 통과했다. GLM도 diagnostic에서 둘 다 통과했다. `success=true`만으로 OCR 품질이나
유효성을 주장하지 않는다.

언어·문서 특성·장치 경로는 각 pinned 공식 model card에서 별도 구조화했다. Paddle은
multilingual 문서 parsing/layout/table/formula와 공식 CPU 설치 경로, GLM은
zh/en/fr/es/ru/de/ja/ko 및 복합 layout/formula/table/information extraction, Surya는
91개 언어와 OCR/layout/reading-order/table/HTML, Granite는 영어와 실험적
일본어·아랍어·중국어 및 full-page layout/table/formula/code, SmolDocling은 영어와
full-page structure/bbox/table/formula/code/chart를 명시한다. `feasibility`의 T4 상태만
v2 로컬 측정이다. CPU·Apple Silicon은 공식 route가 있어도 모두 로컬 미측정으로 표시하며,
현재 runner 지원이나 성능을 추정하지 않는다.

100점 rubric은 freshness 15, parameter 15, weight 10, clean install 10, smoke 15,
217건 24시간 feasibility 10, throughput 10, memory 5, determinism 5, output contract 5다.
사전 등록 threshold가 없는 freshness/throughput과 실행하지 않은 determinism은 모든 후보
0/NA다. 나머지는 pinned metadata와 v2 실행으로 입증된 경우만 득점하며 상세 component,
evidence, total은 `candidates.json`에 있다. 점수는 hard gate 실패를 상쇄하지 않는다.

표본 키는 `sha256("docsem-ocr-screen-v1\0" + instance_id + "\0" + pdf_sha256)`이며,
PDF 크기 삼분위와 200 dpi grayscale entropy bin을 교차해 S1=6, S2=30, S3=217,
결정성 R20=20으로 확장한다. 현재 quality sample은 1이고 silver reference 217건이
available하다. Human gold 없이 단일 사례 agreement를 accuracy나 품질 순위로 해석하지 않는다.
