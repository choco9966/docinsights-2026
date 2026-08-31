# Issue #15 · DocSem 후속 연구 1시간 스프린트

## 결론

첫 구현은 `구조 합의 + 선택적 검수`로 고정한다. 답 문자열의 반복 일치만 보지 않고 `근거 block → 사실 → 변수/방정식 → 계산 → 답`을 독립 생성·대조한 뒤, 불일치하거나 모호한 사례만 보류 또는 사람 검수로 보낸다. 다음 순위는 이 구조 신호를 입력으로 받는 소형 Qwen verifier다. RL은 verifier와 보상 감사가 먼저 통과해야 하는 후속 진단이며 3일 MVP의 주 경로가 아니다.

이번 스프린트는 공개 1차 문헌과 저장소 계약을 검토해 실험 가설·지표·중단 조건을 설계한 문헌 연구다. 모델 학습이나 정확도 측정은 실행하지 않았으므로 아래 수치는 목표·gate이지 실측 결과가 아니다. Issue #14의 에이전트 텍스트 감사 산출물은 병합됐지만 PDF·2인 최종 판정은 Issue #18에서 계속하므로, 사람 판정이 필요한 분석은 두 상태를 구분해 소비한다.

## 해석과 누출 계약

- `semantic_truth`: 공개 라벨을 보지 않은 상태에서 문서와 query가 허용하는 의미상 답이다. 단일값, 복수 후보, 결정 불가, 문항 손상을 허용한다.
- `benchmark_label`: 공개 Train의 주최 측 exact-match 기준값이다. 평가 대상이지만 자동으로 semantic truth가 되지 않는다. 두 값과 불일치 사유를 별도 필드로 보존하고 하나로 덮어쓰지 않는다.
- `codex-assisted-silver`: Issue #8의 OCR engineering reference다. `silver_agreement_not_human_gold_accuracy`이며 의미 정답이나 benchmark label이 아니다.
- 공개 Train도 `template_family_id` 단위로 train/calibration/test를 분리한다. 동일 family의 sibling, 원천 GSM-SEM 식별자, 숫자 치환본이 서로 다른 split에 들어가면 해당 결과를 무효화한다.
- `template_family_id`는 Issue #14 산출 필드가 아니다. `template_family_contract`에 따라 Issue #15 전처리가 canonical GSM-SEM provenance를 우선 사용하고, 매핑이 없으면 versioned label-blind structural fingerprint로 생성한다. 생성에는 benchmark label, hidden holdout, 제출 점수를 사용하지 않는다. 결과 manifest는 `instance_id`, `template_family_id`, `derivation`, `algorithm_version`, `source_sha256`를 보존하며 unknown-family rate가 5%를 넘으면 평가를 중단한다.
- 숨겨진 validation/test의 라벨, 제출 점수, 순위는 prompt, 규칙, threshold, checkpoint, fusion weight, abstention cutoff 선택에 사용하지 않는다. 모든 선택을 공개 Train의 family-disjoint calibration에서 동결한 뒤 hidden holdout에는 한 번만 적용한다. 포털 피드백을 보고 재조정한 결과는 공식 비교에서 제외한다.

## 1차 출처에서 얻은 설계 근거

- [GSM-SEM](https://arxiv.org/abs/2605.07053)은 entity·attribute·relation을 바꾸면서 계산과 답을 보존하는 semantic variant 생성과 strictness를 제시한다. 따라서 표면 paraphrase가 아니라 의미 구조 보존·변경을 구분한 대조군이 필요하다.
- [Oracle GSM-SEM 저장소](https://github.com/oracle-samples/gsm-sem)에는 `original_id`, `SEM_id`, `symbolic_id`, strictness와 원문·변형 문항이 함께 있다. [DocSem participant release](https://github.com/oracle-samples/gsm-sem/tree/main/docsem)는 Train 908개 공개 label과 Validation 217개 비공개 label, opaque ID, PDF 근거 block 계약을 명시한다. family/provenance 정보는 누출 감사에만 쓰고 hidden 추론 입력에는 넣지 않는다.
- [GSM-Symbolic](https://arxiv.org/abs/2410.05229)은 같은 symbolic template의 수치 인스턴스만 달라져도 성능 변동이 생기며 무관 clause 추가에도 성능이 악화될 수 있음을 보인다. random split 하나만으로 일반화를 주장할 수 없다.
- [Selectively Answering Ambiguous Questions](https://openreview.net/forum?id=x2W2dKdNI8)은 모호한 질문에서 sampled output의 반복도를 이용한 confidence가 likelihood나 self-verification보다 신뢰도 높은 선택 답변에 유리했다고 보고한다. DocSem에서는 반복 답뿐 아니라 독립 구조의 반복도까지 비교한다.
- [Math-Minos](https://arxiv.org/abs/2406.14024)는 이진 판정만 쓰는 verifier보다 step-wise 자연어 오류 피드백을 추가하는 접근을 제시한다. 작은 verifier의 target을 최종 정오 하나가 아니라 최초 오류 단계와 근거 설명으로 둔다.
- [Declarative Math Word Problems](https://arxiv.org/abs/2304.09102)은 word problem을 변수와 방정식으로 점진 형식화하고 외부 solver로 실행한다. OCR·사실 추출·방정식 구성·계산 오류를 분리하는 근거다.
- [Small-LLM RL 연구](https://arxiv.org/abs/2503.16219)는 1.5B 모델을 4×A40 48GB, 24시간 조건에서 GRPO로 실험했고 장기 학습의 최적화 불안정성과 길이 제약도 보고한다. 현재 로컬 1시간/3일 범위에서는 RL보다 verifier·보상 감사가 먼저다.

## 첫 60분 실행 순위

| 순위 | 시간 | 실행 | 완료 증거 |
| ---: | ---: | --- | --- |
| 1 | 0–10분 | `semantic_truth`, `benchmark_label`, stage status와 별도 family manifest 계약을 고정하고 Issue #14 입력 존재 여부를 fail-closed 검사 | 필수 필드·분리·누출 gate 체크리스트 |
| 2 | 10–25분 | 공개 Train의 family-disjoint 고정 표본 24건에서 2개 독립 구조 추출을 실행 | evidence/fact/equation JSON 48건, family overlap 0 |
| 3 | 25–40분 | 구조 합의 규칙과 `answer / abstain / review` 정책을 적용 | 불일치 유형, coverage, selective risk 초안 |
| 4 | 40–50분 | Issue #14 S3/S4·semantic-label conflict와 review queue의 교집합을 감사 | tag별 review recall 표 |
| 5 | 50–60분 | direct answer, answer repetition, structure consensus를 같은 24건에 비교하고 다음 1일 gate 결정 | 방법별 오류 검출·coverage 표와 go/no-go 기록 |

Issue #14의 908/908 태그가 없으면 1순위 입력 gate에서 중단하고 synthetic 결과를 만들지 않는다. 그 경우 공개된 수동 고정 표본으로 schema smoke만 수행하며 성능 수치로 보고하지 않는다.

## 후보별 실험 설계

### 1. 구조 합의와 선택적 검수

가설은 독립 추출한 evidence·fact·equation의 합의가 최종 답 반복보다 오답을 잘 검출하고, 낮은 coverage를 허용할수록 선택 정확도가 높아진다는 것이다. novelty는 DocSem visible block 근거와 Issue #14 모호성 축을 동일 review policy에 결합하는 데 있다. 1시간에는 family-disjoint 24건을 두 번 구조화해 합의·보류 규칙을 비교한다. 1일에는 120건 calibration과 사람 오류 판정, 3일에는 전 공개 Train out-of-family 평가와 고정 threshold를 만든다. risk–coverage AUC, coverage 80%의 selective accuracy, 오답 recall, evidence exact/F1, review rate를 측정한다. direct answer보다 오답 recall이 높지 않거나 coverage 70%에서 risk가 줄지 않으면 중단한다. CPU/API 추론 2회와 소량 수동 검수면 충분하다. family 혼합과 label을 구조 생성에 보여 주는 것이 핵심 누출 위험이다. Issue #14 태그가 필수이고, Issue #8 block OCR 계약을 소비하며, Issue #11 OCR 후보는 provenance 검증 후에만 교체 실험에 쓴다.

### 2. 소형 Qwen verifier

가설은 답을 새로 생성시키는 것보다 evidence·fact·equation·answer의 정합성과 최초 오류 단계를 판정시키는 작은 Qwen이 적은 compute로 위험을 낮춘다는 것이다. novelty는 Math-Minos식 단계 피드백을 DocSem의 OCR/fact/equation 단계와 abstention 결정에 맞추는 것이다. 1시간에는 frozen Qwen checkpoint로 24건 zero/few-shot verifier schema smoke를 한다. 1일에는 Issue #14 기반 hard negative를 만들고 LoRA 또는 prompt calibration을 비교하며, 3일에는 family-disjoint 3-fold와 선택 검수 결합을 완료한다. error AUROC/AUPRC, first-error macro-F1, ECE, Brier, risk–coverage AUC, latency를 측정한다. direct structural rule보다 AUPRC가 개선되지 않거나 ECE가 악화되고 latency가 2배를 넘으면 학습을 중단한다. 0.5–3B Qwen, 단일 Apple Silicon 또는 16–24GB GPU 범위로 제한한다. sibling hard negative와 label-derived rationale의 split 간 이동이 누출 위험이다. Issue #14 오류 태그와 #8 OCR record가 필수이고 #11의 소형 OCR 선택은 입력 ablation에 사용한다.

### 3. OCR→사실→방정식 오류 분해

가설은 최종 answer error를 OCR, evidence retrieval, fact binding, equation construction, execution 단계로 분리하면 단일 재계산보다 우선 수정할 병목을 안정적으로 찾을 수 있다는 것이다. novelty는 Issue #8 silver agreement를 의미 정답과 분리한 채 단계별 counterfactual replacement로 인과적 상한을 재는 것이다. 1시간에는 24건에 stage schema와 최초 오류 라벨을 적용한다. 1일에는 각 단계를 silver/manual artifact로 하나씩 교체하고, 3일에는 태그·OCR engine별 error transition matrix를 만든다. stage exact, first-error agreement, oracle-replacement uplift, downstream conditional error, annotator agreement를 측정한다. 최초 오류 합의 κ가 0.6 미만이거나 교체 실험으로 단계 귀속을 재현하지 못하면 taxonomy를 축소한다. CPU와 수동 검수 중심이다. silver를 human gold로 취급하거나 label을 보고 fact를 고치는 누출을 금지한다. #8이 핵심 입력, #14는 의미/손상 층화, #11은 OCR engine ablation에 필요하다.

### 4. 모호성 taxonomy 감사

가설은 Issue #14의 surface integrity, semantic determinacy, benchmark alignment와 세부 태그가 OCR score보다 answer error를 더 잘 설명한다는 것이다. novelty는 `semantic_truth`와 `benchmark_label` 충돌을 오류가 아닌 별도 결과변수로 보존하는 분석이다. 1시간에는 24건 stratified re-audit와 blind/label-pass 분리 검사를 한다. 1일에는 S3/S4 전수 일치도와 tag별 error odds ratio, 3일에는 family cluster bootstrap과 calibration drift를 산출한다. coverage, 누락/중복, Cohen κ, tag별 error rate/odds ratio와 confidence interval, semantic-label conflict rate를 측정한다. blind pass에 label 흔적이 있거나 S3/S4 κ<0.7이면 모델 비교를 중단하고 재판정한다. CPU 집계와 사람 검수 2인이 필요하다. label-first 판정과 sibling 복제가 가장 큰 누출 위험이다. #14가 필수이며 #8 텍스트는 입력 품질층, #11은 OCR 민감도 대조군이다.

### 5. Metamorphic·contrastive consistency

가설은 answer-preserving 변형에는 구조와 답이 유지되고 answer-changing 최소대조에는 해당 fact/equation/answer만 바뀌어야 하며, 두 방향을 함께 검사해야 무조건 같은 답을 내는 false consistency를 잡을 수 있다는 것이다. novelty는 GSM-SEM strictness와 DocSem block evidence를 결합한 쌍 단위 invariant다. 1시간에는 12 family에서 보존/변경 쌍 각 1개를 수동 작성해 dry run한다. 1일에는 100쌍 생성·검수, 3일에는 strictness별 consistency curve와 오류 유형 회귀를 만든다. pair accuracy, preservation consistency, contrastive sensitivity, false-consistency rate, evidence stability를 측정한다. 변형 의미 보존/변경의 사람 합의가 95% 미만이거나 원천 family가 split을 가로지르면 생성물을 폐기한다. API/CPU 추론과 사람 검수가 필요하다. 원천 답·sibling을 hidden inference에 노출하거나 자동 변형 오류를 정답처럼 쓰는 것이 누출 위험이다. #14 family/tag, #8 block text, #11 OCR 변형 민감도에 의존한다.

### 6. Family-disjoint 누출 감사

가설은 random split이 template sibling 암기를 허용해 verifier와 solver 성능을 과대평가하며 family-disjoint split에서 유의한 하락이 나타난다는 것이다. novelty는 `original_id`·symbolic provenance와 lexical/structure fingerprint를 결합한 fail-closed family graph다. 1시간에는 908건의 family key coverage와 split overlap을 검사한다. 1일에는 random/family-disjoint 쌍을 고정하고, 3일에는 모든 후보의 gap과 cluster bootstrap interval을 보고한다. family overlap count, unknown-family rate, random-minus-disjoint gap, duplicate/near-duplicate rate, confidence interval을 측정한다. overlap이 0이 아니거나 unknown family가 5%를 넘으면 성능 평가를 중단한다. CPU 해시/클러스터링으로 충분하다. ID를 모델 feature로 사용하거나 sibling이 calibration과 test에 동시에 존재하는 것이 직접 누출이다. Issue #14는 ambiguity tag와 truth 분리 상태만 제공하고 family ID는 별도 manifest에서 생성하며, #8/#11 산출물은 동일 instance mapping과 provenance를 유지해야 한다.

### 7. 이미지–텍스트 hybrid

가설은 OCR 텍스트와 원본 이미지에서 독립 추출한 구조가 합의할 때만 답하면 OCR의 숫자·부호 오류를 줄일 수 있다는 것이다. novelty는 단순 late fusion이 아니라 modality disagreement를 review trigger와 stage attribution으로 사용하는 것이다. 1시간에는 OCR disagreement 상위 12건과 대조 12건을 paired inference한다. 1일에는 합의/충돌 rule과 latency를 조정하고, 3일에는 OCR engine×modality×ambiguity ablation을 수행한다. answer/evidence accuracy, OCR-error rescue rate, new-error rate, disagreement precision, p95 latency, peak memory를 측정한다. new-error가 rescue보다 많거나 p95가 text-only의 3배를 넘으면 primary 경로에서 제외한다. vision model 1회 추가 추론이 필요하다. label 기반 disagreement 선별과 이미지 metadata/source lookup을 금지한다. #8 OCR·PDF mapping이 필수, #11 엔진 선택이 권장, #14 태그는 층화에 쓴다.

### 8. RL trajectory pilot

가설은 fact grounding, equation execution, verification, abstention을 분리 보상하고 각 보상을 감사하면 작은 Qwen의 trajectory가 개선될 수 있다는 것이다. novelty는 final-answer reward 하나가 아니라 단계별 verifier 신뢰도와 semantic-label conflict abstention을 분리하는 데 있다. 1시간에는 학습 없이 20 trajectory에 reward unit test와 adversarial reward-hacking 사례를 만든다. 1일에는 100–300 trajectory offline scoring/rejection sampling만 수행하고, 3일에는 앞선 verifier gate를 통과한 경우에만 0.5–1.5B Qwen의 짧은 LoRA/GRPO pilot을 실행한다. component reward accuracy, reward-hacking rate, family-disjoint answer/evidence 성능, abstention utility, KL, length, seed variance를 측정한다. reward 정확도 95% 미만, hacking 2% 초과, 두 seed 모두 verifier baseline 이하, 또는 출력 길이 폭증 시 즉시 중단한다. 학습 시 단일 24–48GB GPU 또는 동급 accelerator가 필요하며 CPU-only는 reward audit까지만 한다. hidden score 보상, label leakage, sibling trajectory 혼합이 치명적이다. #14/#8의 단계·truth 계약과 #11 입력 선택이 모두 필요하고 후보 1·2가 통과하기 전에는 시작하지 않는다.

## 3일 MVP

1일차에는 Issue #14 태그와 truth 분리 필드를 fail-closed로 수입하고 Issue #15 전처리가 별도 family manifest와 family graph를 만든다. 공개 Train에서 family-disjoint development/calibration/test를 고정한 뒤 120건에 두 개의 독립 structure trace와 최초 오류 라벨을 작성한다.

2일차에는 규칙 기반 structure consensus와 frozen small-Qwen verifier를 같은 trace에 적용한다. direct answer, answer repetition, structure consensus, Qwen verifier 네 방법을 동일 split·decoder·OCR 입력으로 비교하고 threshold는 calibration에서만 선택한다.

3일차에는 risk–coverage, first-error F1, calibration, tag별·OCR별 결과와 사람 review cost를 산출한다. 권장 정책을 동결하고 hidden holdout에는 선택·조정 없이 한 번만 적용할 수 있는 manifest를 만든다. RL은 MVP 밖이다. 다만 verifier component reward accuracy 95% 이상, family overlap 0, reward-hacking dry run 2% 이하일 때만 별도 pilot 후보로 승격한다.

MVP 성공 기준은 (1) family overlap 0, (2) semantic truth와 benchmark label 덮어쓰기 0, (3) structure consensus 또는 Qwen verifier가 direct answer 대비 public family-disjoint test의 오답 AUPRC와 risk–coverage AUC를 모두 개선, (4) coverage 70% 이상에서 selective risk 감소, (5) hidden holdout 기반 선택 0이다. 하나라도 깨지면 배포나 RL로 확장하지 않고 데이터 계약 또는 verifier로 돌아간다.

## 재현과 검증

```bash
uv run python scripts/validate_issue15_research.py
```

이 명령은 우선순위 JSON의 8개 후보, 필수 필드·의존성·출처, truth/holdout 계약, verifier 우선순위, 60분 합계, autoresearch 상태와 문서 필수 구절을 검사하고 `.omx/specs/autoresearch-docsem-followup/result.json`에 판정을 기록한다.
