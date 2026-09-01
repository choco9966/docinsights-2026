# Issue #7 · Qwen 중심 72시간 DocSem 연구 계획

## 목표와 종료 조건

이 계획은 72시간 안에 공개 Training 908개의 라벨 품질을 감사하고, 작은 Qwen이 `select_evidence → extract_quantities → build_equation → check_completeness → classify_malformed/repair → submit` trajectory를 학습·재생할 수 있는지를 검증한다. 학습·평가 대상은 Qwen뿐이며 Codex 또는 Claude가 만든 답, 근거, 풀이, 검수 결과를 supervision, reward, 모델 선택 또는 threshold 조정에 사용하지 않는다.

72시간의 성공 조건은 다음과 같다.

1. Training 908개가 누락과 중복 없이 고정된 family-disjoint manifest에 배정되고, 각 인스턴스에 query-free OCR 입력과 원본 PDF provenance가 연결된다.
2. H+64 전에는 train과 `internal_dev`에 대해서만 Qwen correlated triage ensemble 세 lane의 schema-valid trajectory 또는 명시적 실패 상태를 만든다. `sealed_internal_test`는 H+64 one-shot bundle에서 처음 추론하며, 그 뒤에야 908개 전체 공개 라벨 QA를 완결한다.
3. 주 학생 `Qwen/Qwen3.5-4B`의 OCR-text inference, trajectory validation과 LoRA SFT smoke를 보장한다. 실측 ETA와 CUDA gate가 통과할 때만 full SFT와 순차 GRPO/RLOO 비교를 추가하고, 9B와 Qwen3-0.6B는 각각 선택 학생과 교차세대 feasibility 하한선으로 분리한다.
4. `semantic_reward`와 `benchmark_reward`가 별도 필드, 별도 집계, 별도 모델 카드로 보고되며 어느 한쪽도 다른 쪽의 별칭으로 사용되지 않는다.
5. 모든 선택을 동결한 뒤 Training 내부 `sealed_internal_test`를 한 번만 열어 평가하고, 결과를 본 뒤 재학습·재선택·repair 규칙 변경을 하지 않는다. 주최 측 `official_held_out_test`는 공개 뒤 별도의 외부 one-shot 제출 절차로 실행한다.
6. 입력, 코드, 모델, tokenizer, prompt, split, config, checkpoint와 출력의 revision 또는 SHA, 실행 환경, 비용과 실패 원인을 재현 manifest에 기록한다.

Gate가 실패하면 해당 단계 이후의 성능 주장을 중단하고 실제 coverage, 실패 원인과 재개 조건을 보고한다. 72시간 안에 paper-grade RL 우월성이나 family-unseen 일반화를 확정하는 것은 종료 조건이 아니다.

## 연구 경계

| 자원 | 허용 용도 | 금지 용도 |
| --- | --- | --- |
| Training 908 tasks, PDF, 공개 labels | family-disjoint 학습·내부 평가, blind artifact 동결 뒤 라벨 QA | sibling 답 복원, split 밖 family 정보 이월 |
| Issue #8 OCR 계약과 동결된 extractor 설정 | query-free block JSONL 생성, OCR provenance와 실패 상태 전달 | Issue #8의 Validation OCR 결과를 학습 데이터로 사용, Qwen 결과로 OCR 설정 재선택 |
| Validation 217 | 이 연구에서는 입력도 열지 않음 | 학습, 평가, 모델 선택, reward, threshold 조정 |
| v1~v12 답안·리뷰·점수와 제출 포털 feedback | 최종 보고서의 별도 역사적 배경에도 사용하지 않음 | supervision, 오류 taxonomy 작성, prompt·repair·reward·checkpoint 선택 |
| Codex·Claude 산출물 | 사용하지 않음 | 답, 근거, trajectory, adjudication, pseudo-label, preference pair |
| Training `internal_dev` | evaluation-only metric, checkpoint·prompt·threshold 선택 | SFT target, RL prompt·rollout·reward buffer, optimizer 입력 |
| Training `sealed_internal_test` | 모든 모델·prompt·reward·repair operator 동결 후 H+64 one-shot bundle에서 최초 추론 | H+64 이전의 OCR 이후 Qwen inference, lane audit, 결과 확인 뒤 수정 또는 재실행 |
| 주최 측 `official_held_out_test` | 72시간 연구 package와 최종 선택을 동결한 뒤 대회 test 공개 시 한 번 추론·제출 | 결과나 포털 feedback을 본 뒤 수정·재선택·재제출 |

사람이 데이터 QA를 수행할 때도 원본 PDF, query, 공개 Training label과 Qwen trajectory만 본다. Codex·Claude 검수물이나 기존 제출 산출물은 검수 작업 공간에 넣지 않는다. 외부 원문 검색과 원본 GSM 문제·정답 복원도 금지한다.

## 입력 계층: Issue #8과의 독립 계약

Issue #8은 별도 브랜치와 별도 평가를 유지한다. Issue #7이 Issue #8 경로를 소비하려면 H+4까지 extractor 구현, query-free block JSONL schema와 config가 하나의 immutable merged commit에 존재하고, 그 commit SHA와 각 artifact SHA가 manifest에 동결되어야 한다. Issue #7은 그 commit의 Validation 산출물을 가져오지 않고 extractor와 schema만 사용해 Training PDF를 별도로 처리한다.

H+4까지 이 조건을 충족하지 못하면 Issue #8 branch의 미병합 코드를 가져오지 않는다. 대신 train PDF 전용 Tesseract fallback을 실제 실행 경로로 선택한다. 실행 시 해석된 `tesseract` binary의 symlink 해소 absolute path, binary SHA-256, version, language `eng`, page render DPI, `--psm 6`과 전체 config SHA를 동결하며, 같은 경로와 config로 pilot과 나머지 train PDF를 처리한다. Fallback은 OCR 비교 실험이 아니라 72시간 Qwen 입력을 만드는 고정 baseline이다.

각 입력 레코드는 `instance_id`, 원본 PDF SHA-256, page image SHA-256, extractor 이름·버전·설정 SHA, ordered block ID, block text, page, bounding box, confidence와 OCR 상태를 포함한다. `user_query`, `answer`, `evidence`와 어떤 풀이도 OCR 엔진 입력에 포함하지 않는다. Qwen runner가 OCR 레코드를 받은 뒤에만 `user_query`를 결합한다.

OCR 상태와 추론 상태를 분리한다. block ID 누락, 숫자·부호·소수점·통화·단위 불일치 또는 engine disagreement가 있으면 `ocr_uncertain`이나 `ocr_failed`로 기록한다. Qwen이 답을 내더라도 OCR 실패를 추론 실패로 덮어쓰지 않으며, 원본 PDF 또는 page image와 OCR span의 provenance가 일치하지 않는 수량은 reward 대상에서 제외한다.

### Train-only staging 구현 계약

구현할 CLI는 `fetch-train`, `cache-train`, `stage-train`, `assert-split`, `scan-denylist`, `hash-manifest`의 여섯 fail-closed subcommand를 제공한다. 이는 아직 구현되지 않은 계약이므로 현재 실행 가능한 명령으로 제시하지 않으며, 각 subcommand의 fixture와 smoke가 통과한 뒤에만 runbook에 실제 argv를 기록한다.

- `fetch-train`은 고정 revision의 train task, train label과 train PDF만 허용하고 Validation 또는 official test config 요청을 거부한다.
- `cache-train`은 train allowlist의 relative path와 expected SHA에 일치하는 blob만 cache에 승격하고, unknown split·path·hash를 발견하면 cache transaction 전체를 중단한다.
- `stage-train`은 family split manifest의 allowlist를 기준으로 파일을 hard-link 또는 copy하며 symlink, allowlist 밖 ID, 절대 외부 경로와 중복 ID를 거부한다.
- `assert-split`은 SFT data manifest와 RL prompt·rollout manifest의 모든 ID가 정확히 `train`인지 검사한다. `internal_dev`, `sealed_internal_test`, `official_held_out_test` ID가 하나라도 있으면 전체 run을 시작하지 않고 fail closed한다.
- `scan-denylist`은 `/val/`, `/validation/`, 제출·review·portal artifact 경로 패턴과 별도 보안 manifest의 Validation content SHA를 검사한다. Staging의 경로나 파일 hash가 하나라도 일치하면 해당 파일만 건너뛰지 않고 staging 전체를 폐기 대상으로 표시하고 run을 중단한다.
- `hash-manifest`는 정렬된 relative path, size, content SHA-256, split과 source revision을 기록하고 downstream manifest가 그 SHA를 부모로 참조하게 한다.

## Family-disjoint 분할과 봉인

라벨과 기존 제출 정보를 읽기 전에 answer-blind family manifest를 만든다. family는 정규화한 query 구조, 요구 연산 단계, 문서의 정량 관계 구조와 주체 역할을 사용해 묶고, 라벨 값·answer·evidence ID는 family feature에 넣지 않는다. 자동 군집의 경계만 원본 query와 PDF를 보고 사람이 확인하며, 확인자는 정답과 기존 모델 출력을 보지 않는다. family 정의 규칙, seed `20260830`, 입력 순서와 구현 SHA를 함께 동결한다.

family 단위로 약 70%를 `train`, 15%를 `internal_dev`, 15%를 `sealed_internal_test`에 배정한다. family 크기 때문에 정확한 인스턴스 비율이 달라질 수 있으므로 목표 비율과 실제 family·인스턴스 수를 모두 보고한다. 동일 family가 둘 이상의 split에 나타나면 즉시 중단한다. 이 Training 내부 15%는 주최 측이 나중에 공개하는 `official_held_out_test`와 이름, manifest, 접근 권한과 보고 표를 모두 분리한다.

`sealed_internal_test`의 task ID와 입력은 H+64 전에는 Qwen runner와 blind lane이 읽을 수 없는 sealed manifest로 분리한다. PDF hash와 family 배정 무결성만 label-free coordinator가 확인하며 OCR-text inference, trajectory 생성, audit lane 실행과 public-label join을 모두 금지한다. H+64 one-shot bundle이 이 split에 대한 최초 model inference다. 따라서 H+64 전 coverage는 `train + internal_dev`만 보고하고, 908개 전체 라벨 QA는 one-shot bundle과 label join이 끝난 뒤 완결한다.

## Qwen blind label audit

공개 라벨은 정답의 절대적 진실로 가정하지 않고 research data QA 대상으로 다룬다. 먼저 pretrained Qwen checkpoint로 세 lane을 실행하고 raw output, normalized trajectory와 ordered instance manifest의 SHA를 동결한다. H+64 전에는 train과 `internal_dev`만 실행하며, `sealed_internal_test` lane은 one-shot bundle에 사전 등록한 그대로 H+64에 처음 실행한다. 각 split의 blind artifact가 동결된 뒤에만 별도 join 작업이 공개 `answer`와 `evidence`를 결합한다.

| Lane | Qwen checkpoint·seed | 고정 관점 | 독립성 해석 |
| --- | --- | --- | --- |
| A | `Qwen/Qwen3.5-4B`, seed 11 | evidence와 quantity binding 우선 | 동일 학생의 role/seed replicate |
| B | `Qwen/Qwen3.5-4B`, seed 29 | typed equation과 unit replay 우선 | 동일 학생의 role/seed replicate |
| C | `Qwen/Qwen3.5-4B`, seed 47 | completeness와 malformed 반례 우선 | 동일 학생의 role/seed replicate |

세 lane은 서로의 출력, 공개 label과 사람 판정을 보지 않는 `correlated triage ensemble`이다. 독립 연구자 세 명이나 독립 model family가 아니라 동일 학생 checkpoint의 상관된 role/seed replicate이므로 합의를 gold나 ground truth라고 부르지 않는다. lane별 model revision, tokenizer revision, chat template, quantization, dtype, decoding config, seed와 runner SHA를 기록한다. 선택적 9B audit은 capacity diagnostic으로 별도 집계하며 세 lane 합의에 표를 추가하거나 표를 대체하지 않는다.

Blind artifact 동결 뒤 라벨 QA는 `consistent`, `answer_mismatch`, `evidence_mismatch`, `replay_conflict`, `likely_malformed`, `ocr_blocked`, `unresolved`로 분류한다. 세 lane 일치, 다수결, 전원 불일치, schema-invalid와 미결 비율을 별도로 보고한다. `likely_malformed`는 Qwen 합의만으로 라벨을 수정하지 않고 사람이 원본 PDF와 공개 label을 검수한 뒤 `accepted`, `repairable`, `underdetermined`, `contradictory`, `unresolved` 중 하나로 adjudicate한다. 방향 반전, 절댓값, 단계 누락, sibling-template 이월, 불완전 변수와 질문·본문 주체 불일치의 여섯 오류 유형은 연구 brief에서 사전 선언한 taxonomy로 동결하고, 빈도와 사례는 Training 908에서만 채운다. v1~v12 사례나 점수로 taxonomy 정의, 심각도, reward weight 또는 repair 규칙을 바꾸지 않는다.

SFT target은 train split에서 공개 label과 일치하고 deterministic replay를 통과한 Qwen trajectory만 후보가 된다. `repairable` 사례는 train split에서 사람이 변경 이유와 source block을 기록한 데이터 QA patch가 있어야 별도 repair 학습군에 들어가며, 원래 benchmark label은 보존한다. Internal_dev의 같은 상태는 evaluation-only이고 target 후보가 아니다. `underdetermined`, `contradictory`, `ocr_blocked`, `unresolved`는 억지 수치 답 supervision에서 제외하고 malformed classification 또는 abstention 평가군으로만 남긴다. Codex·Claude에서 유래한 답이나 설명은 어떤 경로로도 target에 합류하지 않는다.

## 제한된 action 환경

모든 action은 닫힌 JSON schema를 따르고 이전 action의 ID만 참조한다. 자연어 장문 chain-of-thought는 저장하거나 supervision으로 사용하지 않으며, 검증 가능한 구조 상태만 남긴다.

1. `select_evidence`: query의 대상 주체와 요청량을 직접 뒷받침하는 ordered block ID와 정확한 span을 선택한다.
2. `extract_quantities`: 각 span에서 값, 단위, 부호, 주체, 역할, evidence reference를 추출하고 암묵적으로 보충한 값은 금지한다.
3. `build_equation`: quantity ID만 피연산자로 쓰는 typed AST, target variable, 연산과 허용된 단위 변환을 구성한다.
4. `check_completeness`: missing variable, 방향 반전, 절댓값 오용, 단계 누락, sibling-template 이월, 불완전 변수, 질문·본문 주체 불일치와 unit mismatch를 검사한다.
5. `classify_malformed/repair`: `well_formed`, `repairable`, `underdetermined`, `contradictory`, `ocr_corrupted`를 분류한다. Repair는 같은 문서의 인용 가능한 block으로만 operand, target 또는 relation을 보완하며 외부 사실이나 sibling 사례를 사용하지 않는다. Repair 뒤에는 `check_completeness`를 다시 통과해야 한다.
6. `submit`: replay된 최종 answer, 최소 evidence block set, document status, repair 여부와 calibrated uncertainty를 반환한다. 답을 결정할 수 없으면 null answer와 원인 코드를 제출한다.

Validator는 schema, block/span 존재, quantity-source 결합, AST reference, unit type, deterministic replay, completeness 상태 전이와 제출값 일치를 검사한다. Schema-invalid나 replay mismatch는 reward 0이 아니라 명시적 hard failure로 처리해 잘못된 trajectory가 학습 자료에 섞이지 않게 한다.

## 모델 역할과 27B 제외 조건

주 학생과 주 실험 checkpoint는 `Qwen/Qwen3.5-4B`다. 같은 Qwen3.5 세대에서 추가 자원 대비 이득을 확인할 필요가 있고 H+28 gate에 예정 시간이 남을 때만 `Qwen/Qwen3.5-9B`를 선택 학생으로 추가한다. `Qwen/Qwen3-0.6B`는 다른 세대·구조의 작은 모델이므로 4B·9B와 같은-family scaling curve로 묶지 않고, direct inference와 짧은 SFT가 action schema를 수용하는지만 확인하는 `cross_generation_feasibility_baseline`으로 표시한다.

정확한 `Qwen/Qwen3.8-27B`는 vision encoder를 포함하는 native image-text 모델이며 공식 BF16 저장소 크기는 55.6GB다. 공식 FP8 배포도 30.9GB이지만, 이는 weight 저장 크기일 뿐 학습 activation, optimizer, KV cache와 vision 입력 여유를 포함하지 않는다. 2026-08-30 기준 공식 Qwen 배포에는 이 checkpoint의 MLX 또는 4-bit artifact가 없으므로 community conversion을 핵심 재현 경로로 채택하지 않는다.

Mac Studio unified memory가 64GB 이하이면 27B를 다운로드·실행 후보에서 제외한다. 128GB 이상일 때만 공식 Transformers에서 image와 text를 함께 넣는 load/generate smoke가 통과한 경우 `diagnostic_only_multimodal_upper_bound`라는 단일 역할로 추가할 수 있다. 27B는 SFT, pseudo-label, reward, model selection, GRPO/RLOO와 72시간 critical path에서 제외한다. 27B 출력은 train trajectory 후보나 학생 target으로 보존하지 않고 별도 diagnostic report에만 기록한다. 64GB 초과 128GB 미만 장비도 이번 72시간 계획의 27B 허용 조건을 충족하지 않는다.

## SFT와 GRPO/RLOO 실험

먼저 `Qwen/Qwen3.5-4B`에 direct-submit baseline과 trajectory SFT를 실행한다. 선택적 `Qwen/Qwen3.5-9B`를 추가할 때는 같은 SFT data manifest, action schema, effective token budget와 seed set을 적용하고 parameter 수 차이 외의 prompt·decoding·평가 코드를 동일하게 유지한다. 0.6B cross-generation baseline은 별도 표에 두며 4B·9B와 같은-family 효율성 기울기를 계산하지 않는다. SFT는 action별 loss와 end-to-end schema-valid rate, replay rate, answer/evidence 지표, malformed detection, latency, peak memory와 tokens/sec를 기록한다.

SFT data manifest와 RL prompt·rollout manifest에는 `train` ID만 들어간다. Manifest 생성 직후와 trainer 시작 직전에 split assertion을 두 번 실행하며 `internal_dev`, `sealed_internal_test`, `official_held_out_test` ID가 하나라도 있으면 optimizer나 rollout worker를 시작하지 않는다. `internal_dev` label join과 normalized trajectory는 evaluation store에만 두고 training dataloader, replay buffer, preference/group construction과 reward fitting path에서 물리적으로 분리한다.

Internal development split에서 semantic gate를 통과한 4B와 선택적 9B SFT checkpoint 중 품질-비용 Pareto frontier의 한 학생만 72시간 RL 후보로 승격한다. 시간이 남아도 `sealed_internal_test`를 보지 않고 두 번째 모델을 추가한다. CUDA가 없으면 승격 학생에 대해 reward replay와 사전 선택한 GRPO feasibility만 실행한다. CUDA, backend parity와 순차 ETA gate가 모두 통과할 때만 동일 rollout 수, 최대 생성 토큰, seed와 wall-clock cap으로 GRPO/RLOO의 semantic/benchmark 네 arm을 정해진 순서로 확장하고 SFT-only를 공통 기준선으로 둔다. 0.6B와 27B는 RL 승격 대상이 아니다.

### Reward 분리

`semantic_reward`는 label answer를 직접 보상하지 않는다. V1 scalar는 schema·상태 전이 0.10, evidence grounding 0.20, quantity binding과 invented-quantity 부재 0.15, typed AST replay와 unit consistency 0.25, target·주체·completeness 0.15, malformed/repair 판정 0.15의 합으로 고정한다. 근거 없는 quantity를 만들거나 malformed·underdetermined 문항에 숫자를 강제 제출하면 1.0을 감점하고 최종 값을 `[-1, 1]`로 제한한다. 구성 요소와 penalty는 모두 개별 로그로 남긴다.

`benchmark_reward`는 데이터 QA에서 `accepted`로 판정된 train-fold 공개 label에 대해서만 `0.75 × normalized answer exact match + 0.25 × evidence set F1`로 계산한다. `repairable`, `underdetermined`, `contradictory`, `ocr_blocked`, `unresolved`에는 benchmark reward를 부여하지 않는다. 이는 RL ablation용 scalar이며 공식 종합 점수가 아니다. Answer EM, evidence exact와 evidence F1 원지표도 따로 보존한다.

두 reward는 하나의 합산 점수로 모델을 고르지 않는다. Internal development 결과에서 먼저 schema-valid, replay, completeness, malformed precision과 forced-answer rate의 semantic gate를 통과해야 하며, 그 안에서 answer/evidence와 비용의 Pareto frontier를 보고한다. Benchmark-only arm이 answer EM을 높여도 malformed precision이나 forced-answer rate를 악화시키면 승격하지 않는다.

## Mac Studio 역할과 RL portability

72시간 계획에서 유일하게 확정된 장비는 앞으로 제공될 Mac Studio이며 chip, unified memory, free disk와 실제 accelerator 상태는 아직 확정되지 않았다. H+4까지 실기기에서 chip model, unified memory bytes, free disk bytes, macOS, Metal/MLX/Transformers 버전과 CUDA 원격 자원 유무를 측정해 hardware manifest를 만든다. Mac Studio는 PDF 렌더링·OCR 실행, 4B와 선택적 9B Qwen inference, action validator·replay, LoRA SFT의 메모리·속도 feasibility 확인에만 사용한다. 0.6B는 cross-generation feasibility baseline이고, 27B는 위 128GB·Transformers multimodal smoke 조건을 통과한 경우의 diagnostic-only inference만 허용한다.

Mac에서 짧은 rollout 또는 optimizer smoke가 실행되더라도 이를 GRPO/RLOO 본 실험 지원이나 성능 근거로 해석하지 않는다. Stateful RL, multi-sample generation, trainer callback과 checkpoint resume는 별도 portable Linux/CUDA 환경에서 동일 trajectory schema와 reward replay test를 통과해야 한다. MLX와 CUDA 간 tokenizer, chat template, generated action, reward component와 AST replay의 golden fixture가 일치하지 않으면 RL branch를 중단한다. Mac 결과는 OCR·inference·SFT feasibility로만 보고하고 CUDA 결과와 직접 속도 우위를 주장하지 않는다.

## 72시간 자원과 ETA 계약

30개 pilot은 각 장비·model·lane에 대해 `seconds_per_example`, `tokens_per_trajectory`, input/output token p50·p95, peak memory와 실패·retry율을 기록한다. SFT smoke는 `training_tokens_per_second`, `seconds_per_step`, peak memory와 checkpoint bytes를, RL smoke는 rollout당 생성 초, reward replay 초, update 초와 peak memory를 기록한다. 시간은 p95, 처리량은 보수적인 p05, 메모리는 관측 peak, 전체 ETA는 retry를 포함한 안전계수를 사용한다. 아래 식에서 `N_open = N_train + N_internal_dev`이고 `measured_parallelism`은 pilot에서 오류 없이 유지된 실제 동시 worker 수다.

| 단계 | 공식 계산식 | 진행 조건 | 보장/선택 |
| --- | --- | --- | --- |
| Correlated triage audit | `ETA_audit = N_open × 3 lanes × p95_seconds_per_example ÷ measured_parallelism × 1.20` | H+22 이전 완료 ETA, peak memory가 unified memory의 80% 이하, 예상 artifact bytes가 free disk의 70% 이하 | Train+dev audit은 보장, sealed audit은 H+64 one-shot에만 실행 |
| 4B SFT | `SFT_tokens = epochs × Σ train_trajectory_tokens`, `ETA_sft = SFT_tokens ÷ p05_training_tokens_per_second × 1.20` | H+42 이전 완료 ETA, save/reload 여유를 포함한 peak memory 80% 이하, checkpoint 포함 disk 70% 이하 | CUDA와 무관하게 Mac 20-step LoRA save/reload smoke까지 보장, full SFT는 ETA gate |
| RL | `ETA_rl = updates × (group_size × p95_rollout_seconds + group_size × p95_reward_seconds + p95_update_seconds) × 1.25` | H+56 이전 완료 ETA, CUDA backend parity·resume 통과, H+56~72의 평가·봉인 16시간을 침범하지 않음 | CUDA 미확보 시 reward replay와 train prompt 2개·group size 2·update 1회의 사전 선택 GRPO smoke만 보장 |
| Optional RL arms | 각 arm의 pilot `ETA_rl`을 직전 arm 종료 뒤 다시 계산 | 남은 시간이 해당 arm ETA와 16시간 reserve의 합 이상일 때만 다음 arm 시작 | `GRPO-semantic → RLOO-semantic → GRPO-benchmark → RLOO-benchmark` 순차 실행, 병렬 시작 금지 |
| 9B·27B | 9B는 audit/SFT 계산식을 별도 적용, 27B는 load와 1개 image+text generation wall time만 기록 | 9B는 4B critical path 완료 ETA를 침범하지 않아야 함, 27B는 unified memory 128GB 이상 | 9B 선택, 27B diagnostic-only이며 ETA와 무관하게 critical path 제외 |

CUDA 자원이 H+4에 확인되지 않으면 72시간 guaranteed deliverable은 train PDF OCR text, Qwen3.5-4B OCR-text inference, six-action trajectory validator·typed replay, 4B LoRA SFT smoke와 재현 manifest다. RL은 golden reward replay와 H+4에 결과를 보기 전에 선택한 GRPO의 train prompt 2개·group size 2·update 1회 smoke만 시도하고, 실행 여부와 오류를 feasibility 결과로 남긴다. 네 RL arm은 CUDA, backend parity와 각 순차 ETA gate가 모두 충족될 때만 optional로 확장하며, 실행하지 못한 arm은 실패가 아니라 `not_run_resource_gate`로 보고한다.

## H+0~72 실행 일정과 gate

| 시간 | 실행 | 필수 산출물 | Stop/Go gate |
| --- | --- | --- | --- |
| H+0~4 | Mac Studio의 chip, unified memory, free disk와 software stack을 실측하고 CUDA 원격 자원 유무를 기록한다. Train-only fetch/cache/denylist/hash와 split assertion 계약을 동결한다. Issue #8 immutable merged commit·schema·extractor SHA를 확인하고, 없으면 train PDF용 Tesseract PSM 6 executable realpath·binary/config SHA를 fallback으로 동결한다. | hardware manifest, `boundary-policy`, OCR source decision, trajectory schema v1, reward spec v1, train-only CLI contract, denylist/hash 검사 결과 | Validation path/hash, v1~v12, portal, Codex·Claude artifact가 staging에 하나라도 있거나 OCR provenance가 없으면 중단한다. Issue #8은 세 artifact가 같은 merged commit에 없으면 사용하지 않는다. |
| H+4~10 | answer-blind family를 만들고 train/internal_dev/sealed_internal_test manifest를 봉인한다. Sealed ID·입력은 runner denylist에 추가한다. 30개 train-family pilot에서 OCR join, six-action trajectory, validator와 replay를 4B role/seed 세 lane으로 실행하고 ETA 공식을 채운다. | family manifest, leakage report, 30×3 pilot records, seconds/example·tokens/trajectory·memory·ETA 표, model eligibility report | family overlap 0, sealed inference 0, schema-valid 95% 이상, replay 가능한 determinate record 95% 이상이어야 전수 audit으로 간다. 27B는 critical path와 무관하게 128GB·smoke gate를 통과하지 못하면 diagnostic에서 제외한다. |
| H+10~22 | 동일 4B 학생의 correlated triage ensemble을 train과 internal_dev에만 실행한다. Sealed runner access는 계속 0으로 유지하고 재시도는 같은 config로 실패 유형당 1회만 허용한다. | `3 × (N_train + N_internal_dev)` terminal records, raw-output manifest, train+dev coverage·failure report, sealed access log 0건 | 각 open split ID에 lane별 terminal record 하나, 누락·중복 0을 목표로 한다. H+22 audit ETA를 넘기거나 한 lane coverage가 95% 미만이면 SFT 준비를 멈추고 H+28까지 audit 복구에 집중한다. 908/908 audit 완료를 주장하지 않는다. |
| H+22~28 | 공개 label join을 split별로 분리한다. Train join은 SFT data QA와 오류 taxonomy·repair/abstention 후보 생성에 사용하고, internal_dev join은 evaluation-only store와 metric 계산에만 사용한다. SFT와 RL manifest를 train ID만으로 생성하고 이중 split assertion을 실행한다. | train SFT-QA manifest, internal_dev evaluation-only manifest, QA status table, 오답노트, SFT data SHA, split assertion report | Internal_dev/sealed/official ID가 SFT/RL manifest에 하나라도 있거나 target provenance가 불명확하면 fail closed한다. Replay-valid SFT 후보가 train split의 60% 미만이면 full SFT와 RL로 진행하지 않는다. |
| H+28~42 | 주 학생 Qwen3.5-4B direct baseline, OCR-text inference와 20-step LoRA save/reload smoke를 먼저 완료한다. ETA gate가 통과하면 full 4B SFT를 실행하고, 4B critical path를 침범하지 않을 때만 9B 또는 0.6B cross-generation diagnostic을 순차 추가한다. | Guaranteed 4B inference·validator·SFT-smoke artifact, 선택적 full 4B/9B checkpoint manifest, 별도 0.6B feasibility 표, ETA·learning curve·비용 표 | Trainer 시작 직전 train-only assertion, save/reload와 internal_dev evaluation을 통과해야 한다. CUDA 부재나 ETA 실패 시 guaranteed deliverable에서 멈추고 full SFT/RL 성능을 주장하지 않는다. |
| H+42~56 | Reward golden replay를 먼저 완료한다. CUDA가 없으면 4B SFT-smoke checkpoint에서 train prompt 2개·group size 2·update 1회의 사전 선택 GRPO feasibility만 시도한다. CUDA와 parity·ETA gate가 있으면 `GRPO-semantic → RLOO-semantic → GRPO-benchmark → RLOO-benchmark`를 순차 실행하며 각 arm 종료 뒤 남은 시간을 다시 계산한다. | 항상 reward replay와 GRPO feasibility report, 선택적으로 최대 4개 RL arm의 config·rollout·reward-component manifest, `not_run_resource_gate` 기록 | Golden reward·AST replay 100% 일치, train-only assertion, NaN 0이 필수다. 다음 arm ETA와 16시간 reserve를 확보하지 못하면 그 arm부터 시작하지 않는다. |
| H+56~64 | Family-disjoint internal_dev에서 SFT-only와 RL arm을 평가한다. Semantic gate, answer/evidence, malformed detection·repair precision, forced-answer rate, latency·memory와 비용을 비교하고 최종 모델·prompt·reward·threshold를 동결한다. | internal evaluation report, Pareto table, frozen selection manifest, sealed-internal-open authorization SHA | Validation·portal feedback 접근 0과 sealed_internal_test 접근 0을 확인해야 한다. Schema-valid 98% 이상, replay 99% 이상, completeness 95% 이상, malformed precision 80% 이상, non-determinate forced-answer rate 5% 이하를 모두 통과한 모델이 없으면 sealed_internal_test를 열지 않고 no-go로 종료한다. |
| H+64~68 | 사전 등록한 one-shot bundle을 Training `sealed_internal_test`에 처음이자 마지막으로 실행한다. Bundle에는 동결된 최종 checkpoint 평가 1회와 4B correlated triage ensemble 세 lane이 포함된다. 이후에만 sealed public label을 join해 908/908 라벨 QA를 완결한다. | internal one-shot access log, final predictions SHA, sealed lane artifacts, internal sealed metrics, 전체 908 QA status와 unresolved 목록 | H+64 이전 sealed inference 0건과 bundle config SHA 일치를 먼저 확인한다. 실행 뒤에는 어떤 수정·재실행도 금지하며 실패도 최종 결과로 보존한다. 이 결과를 official held-out 결과라고 부르지 않는다. |
| H+68~72 | 결과, 한계, 비용과 재현 정보를 묶고 artifact integrity를 검사한다. Mac feasibility와 portable RL 결과를 분리하고, 대회 test 공개 뒤 실행할 external one-shot protocol을 동결한다. | 최종 연구 보고서, model/data cards, artifact index, SHA-256 manifest, 환경 lock, 재현 runbook, official external one-shot protocol, go/no-go 결정 | 필수 SHA·config·split·access log가 빠졌거나 명시한 record 수와 manifest 수가 다르면 완료로 표시하지 않는다. Official protocol은 Validation 또는 portal feedback을 전제로 해서는 안 된다. |

## Internal과 official one-shot 분리

`sealed_internal_test`는 공개 Training 908에서 family 단위로 떼어 둔 약 15%다. 라벨이 저장소에 존재하더라도 H+64 전에는 Qwen runner와 모든 blind lane 밖에서 봉인하며 model inference를 한 번도 실행하지 않는다. H+64에는 동결된 최종 checkpoint 1회와 사전 등록한 correlated triage ensemble 세 lane을 하나의 one-shot bundle로 실행·채점한다. 이 결과는 72시간 연구의 내부 일반화 진단과 908 라벨 QA 완결에만 쓰며 공식 대회 test 결과가 아니다. 실패, timeout과 OCR 누락도 재실행하지 않고 terminal result로 센다.

`official_held_out_test`는 주최 측이 최종 대회를 위해 나중에 공개하는 별도 test다. H+72에 선택된 model revision, checkpoint SHA, OCR extractor, prompt, action schema, reward와 decoding config를 external submission package로 동결한다. 공식 test 공개 뒤에는 입력 manifest와 SHA를 만들고, 동결된 query-free OCR adapter와 Qwen runner를 한 번 실행하고, instance coverage·JSONL schema·중복만 label-free validator로 확인한 뒤 하나의 prediction SHA를 한 번 제출한다. 공식 answer, leaderboard 또는 포털 feedback을 확인한 뒤 모델, prompt, OCR, repair, threshold나 제출 답을 수정하지 않으며 추가 제출도 하지 않는다. 입력 계약이 동결 package와 호환되지 않으면 결과를 보정하지 않고 protocol failure로 기록한다.

## 산출물 구조

아래 경로는 구현 단계에서 생성할 계약이며, 대용량 raw output과 checkpoint는 Git에 직접 넣지 않는다.

| 경로 | 추적 | 내용 |
| --- | --- | --- |
| `docs/research/issue-7-qwen-72h-plan.md` | Git | 본 계획과 경계 |
| `schemas/qwen-trajectory-v1.schema.json` | Git | six-action JSON 계약 |
| `configs/qwen/` | Git | 모델별 SFT, GRPO, RLOO, reward와 decoding 설정 |
| `manifests/issue-7/` | Git | input, family split, label join, selection, sealed internal access, official external access와 SHA index |
| `reports/issue-7-qwen/` | Git | coverage, label QA, internal evaluation, sealed internal one-shot, official external one-shot, 비용·한계 보고 |
| `tests/fixtures/issue-7-qwen/` | Git | schema, typed AST, unit, malformed, reward parity golden fixture |
| `artifacts/issue-7-qwen/<run_id>/` | 로컬/외부 artifact store | raw generations, normalized trajectories, rollouts, logs와 checkpoints |

각 normalized record는 `instance_id`, split, family ID, input SHA, OCR status, model·prompt·config SHA, action sequence, validator version, semantic reward components, benchmark reward components, latency, token count와 terminal status를 가진다. Artifact index는 logical URI, byte size, record count, content SHA-256, 생성 시각, 생성 코드 revision과 상위 artifact SHA를 기록해 `PDF → OCR → trajectory → SFT data → checkpoint → evaluation` lineage를 역추적할 수 있게 한다.

## 재현성과 보고 규칙

- README에 고정한 canonical release와 Hugging Face mirror revision을 사용하고 실제 checkout SHA를 manifest에 다시 기록한다.
- Qwen model과 tokenizer는 이름만 적지 않고 immutable revision SHA, chat template SHA, quantization, dtype와 license snapshot을 기록한다.
- Python과 trainer 환경은 lockfile 또는 container digest, OS·accelerator·driver, seed, deterministic 설정과 known nondeterminism을 기록한다.
- 모든 shuffle, family 배정, sample과 rollout seed를 기록하고 입력은 `instance_id` 정렬본의 SHA로 고정한다.
- Validator, typed AST evaluator와 reward 함수는 golden fixture를 같은 process와 portable RL backend에서 재생해 component별 완전 일치를 검사한다.
- 실행 명령은 구현된 CLI의 smoke test가 통과한 뒤 runbook에 기록한다. 이 계획서에는 아직 존재하지 않는 명령을 실행 가능한 것처럼 제시하지 않는다.
- 중단된 run도 삭제하지 않고 terminal status, retry count, stdout/stderr SHA와 실패 원인을 남긴다.
- 최종 표는 데이터 coverage, schema validity, replay, answer EM, evidence exact/F1, malformed precision·recall, repair precision, forced-answer rate, latency, peak memory, token·wall-clock 비용을 split과 model arm별로 분리한다.

## 공식 모델 근거

- [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)와 [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)는 같은 Qwen3.5 세대의 공식 post-trained Transformers checkpoint이므로 각각 주 학생과 선택 학생으로 고정한다.
- [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)는 Qwen3 세대 모델이므로 Qwen3.5 학생과 같은-family 크기 비교에 넣지 않는다.
- [`Qwen/Qwen3.8-27B` 공식 파일 트리](https://huggingface.co/Qwen/Qwen3.8-27B/tree/main)는 55.6GB와 Transformers `image-text-to-text` 사용법을, [`Qwen/Qwen3.8-27B-FP8` 공식 파일 트리](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/main)는 30.9GB FP8 배포와 같은 multimodal 입력 경로를 제시한다. 본 계획은 이 모델을 diagnostic-only multimodal upper bound 이외의 역할에 사용하지 않는다.
- [Qwen3.8 공식 저장소](https://github.com/QwenLM/Qwen3.8)는 Apple Silicon에서 MLX 이름으로 배포된 모델을 찾도록 안내하지만, 기준일의 exact official `Qwen/Qwen3.8-27B` 배포에는 MLX 또는 4-bit artifact가 없다. 따라서 community MLX·4-bit 변환은 이 계획의 핵심 경로와 27B 허용 판단에서 제외한다.

## 최종 Go/No-Go 해석

`GO-SFT`는 label provenance와 family 격리가 검증되고 replay-valid SFT 후보가 충분할 때만 선언한다. `GO-RL`은 4B 또는 선택적 9B SFT checkpoint가 internal semantic gate와 backend parity를 통과할 때만 선언한다. `GO-SEALED-INTERNAL`은 모든 선택 SHA가 동결되고 Validation·portal·sealed split 접근 감사가 깨끗할 때만 선언한다. `GO-OFFICIAL-EXTERNAL`은 72시간 package가 변경 없이 보존되고 공식 test 입력 외의 신규 feedback을 사용하지 않을 때만 선언한다. 어느 gate에서든 실패하면 낮은 기준으로 통과시키지 않고 `NO-GO`와 재개에 필요한 구체 조건을 남긴다.

72시간 결과는 Qwen3.5-4B 중심 trajectory 학습과 reward 설계의 feasibility evidence다. Mac Studio smoke를 RL portability의 증명으로, internal development나 sealed internal 개선을 공식 Validation 또는 official held-out 성능으로, Qwen lane 합의를 정답으로, 공개 label 일치를 semantic correctness로 확대 해석하지 않는다.
