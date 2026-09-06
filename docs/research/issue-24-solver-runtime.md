# Issue #24 풀이 모델과 실행 환경

정확도 우선 요청에 따라 GPT-6 Astra/high를 풀이 실행 후보로 확인했다. [공식 모델 문서](https://developers.openai.com/api/docs/models/gpt-6-astra)는 Astra를 복잡한 작업을 위한 최상위 모델로 소개하며 이미지 입력과 structured output을 지원한다. 이는 DocSem에서 다른 모델보다 정확하다는 실측 결과가 아니다. 기존 OCR silver reference는 gpt-5.6-sol/high의 동결된 결과를 그대로 사용한다.

## 고정된 별도 CLI

기존 Codex CLI 0.144.1은 Astra 요청에 최신 CLI가 필요하다는 HTTP 400을 반환했다. 전역 설치를 유지한 채 공식 npm의 `@openai/codex@0.153.4`를 별도 경로에 설치했다. [공식 CLI 안내](https://learn.chatgpt.com/docs/codex/cli)와 실제 `exec --help`로 실행 옵션을 확인했다.

새 환경의 재현 명령은 다음과 같다. 기존 환경에 덮어쓰는 명령이 아니다.

```sh
mkdir -p data/issue24/codex-0.153.4
cp requirements/solution-codex/package.json requirements/solution-codex/package-lock.json data/issue24/codex-0.153.4/
npm ci --prefix data/issue24/codex-0.153.4 --ignore-scripts --no-audit --no-fund
data/issue24/codex-0.153.4/node_modules/.bin/codex --version
```

공개 Git의 package manifest와 lock은 실제 설치에 사용한 파일의 사본이며, 공식 npm tarball URL과 SHA-512 SRI를 고정한다. Node.js/npm 및 Codex 인증이 필요하다. 모델 호출은 기존 계정 인증을 사용하며 별도 결제나 사용량 초기화를 수행하지 않았다.

macOS arm64에서 관측한 wrapper SHA-256은 `61b0194f3bb6534439c8d26a3ed57d0805f84b884588b761795323eeb92fcf70`, native executable SHA-256은 `b973d440acac501fd2594a43e7ca9ce41e0a65b9dfb28d0d7a7837c99e1261e3`이다. 다른 OS의 실행 파일 해시로 사용하지 않는다. Runner는 실제 실행 경로·버전·바이트를 새 실행 설정에 동결한다.

## 확인한 범위

- 도구 비활성화, read-only, ephemeral, 사용자 설정·규칙 제외, JSON schema 및 이미지 입력 옵션을 실제 CLI에서 확인했다. 합성 산술 structured-output 호출은 5.98초에 완료됐다.
- 기존 공개 validation 입력과 이미지로 생성한 Astra primary는 11.75초에 완료됐고 산술·형식을 통과했다. 이 결과는 OCR를 다시 실행한 최종 Native 구성의 시험이 아니다.
- 최초 blind source checker는 inline ID를 별도 제목으로 분류하지 않아 근거 미확정으로 종료했다. 이전 결과를 보존하고 일반적인 inline-heading 필드 정의를 명확히 한 별도 1회 진단에서는 최종 답과 Evidence가 생성 동결 후 기준 행과 일치했다.
- 동일 문항의 stochastic before/after 한 건으로 prompt의 통계적 효과, 모델 간 우위, 전수 정답률 또는 순위 향상을 주장하지 않는다. 선정 OCR를 결합한 실제 pilot은 별도 gate다.
- 기존 공개 RapidOCR 입력만 재사용한 held-out 2건에서도 fresh Astra 풀이와 blind source check가 형식·정확 산술·Evidence 출처 검사를 통과했다. 이전 답이나 실패 피드백은 입력에 넣지 않았다. 비공개 정답과의 일치 또는 전수 정확도 검증은 아니다.
- JSON event stream은 backend model 식별자를 제공하지 않았다. 관측 가능한 사실은 `gpt-6-astra`를 요청했고 서버가 unknown-model/fallback 경고 없이 수락했다는 것이다.

문항별 입력·답·인용과 원시 실행 기록은 private `artifacts/solution-records/issue24/diagnostics/` 아래에 보존한다.
