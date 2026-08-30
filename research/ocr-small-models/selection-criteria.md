# 소형 OCR 후보 선정 및 단계적 표본 기준

조사 시각은 2026-08-31 KST다. 인기 신호는 변동값이므로 다운로드·좋아요·trending은
조회 시점 스냅샷으로만 사용하고, 실행 대상은 반드시 commit revision으로 고정한다.

하드 게이트는 10억 파라미터 이하, weight 2.5 GiB 이하, clean smoke 성공, Validation
217건 예상 24시간 이하, peak RAM/VRAM 80% 이하이다. 100점 평가는 최신성 15,
파라미터 15, weight 10, clean install 10, smoke 15, 전체 적용 가능성 10, 처리량 10,
메모리 5, 결정성 5, 출력 계약 5점이다. 70점부터 실행 가능 후보, 85점부터 전체 평가
우선순위로 본다. 라이선스 제한과 `trust_remote_code`는 점수와 별도로 차단 사유가 된다.

표본 키는 `sha256("docsem-ocr-screen-v1\0" + instance_id + "\0" + pdf_sha256)`이며,
PDF 크기 삼분위와 200 dpi grayscale entropy bin을 교차해 S1=6, S2=30, S3=217,
결정성 재실행 R20=20으로 확장한다. 현재 실제 결과는 고정 선행 사례 1건뿐이다.
따라서 품질 승자나 전체 표본 정확도를 주장하지 않는다.
