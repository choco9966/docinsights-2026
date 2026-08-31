# Autoresearch sandbox · DocSem follow-up

## 허용 범위

- 공개 Train과 저장소의 Issue #8/#11 산출물, 완료 후 제공될 Issue #14 태그를 사용한 설계
- arXiv, OpenReview, Oracle 공식 저장소의 1차 문헌 검토
- 문서·JSON·validator·autoresearch 상태와 completion artifact 수정

## 금지 범위

- 숨겨진 validation/test label, 제출 점수, 리더보드 순위로 prompt·규칙·threshold·checkpoint·weight를 선택하거나 반복 조정
- benchmark label을 semantic truth로 덮어쓰기
- Issue #8 silver score를 human-gold accuracy로 해석
- 원천 GSM-SEM sibling/template 식별자를 hidden 추론 feature로 사용
- verifier와 reward audit gate 전에 RL을 primary로 권고
- 실행하지 않은 실험을 실측 결과로 서술

## 중단 조건

Issue #14 필수 산출물이 없으면 schema smoke 외 경험적 비교를 중단한다. family overlap이 0이 아니거나 truth/label 필드가 분리되지 않으면 모든 성능 집계를 중단한다.
