# Issue #14 입력 스냅샷

이 디렉터리는 DocSem Train 908개 모호성 감사에 실제로 사용한 입력을 clean checkout에서도 재현할 수 있도록 byte-for-byte 보존한다. `train-tasks.jsonl`과 `train-labels.jsonl`은 Hugging Face 데이터셋의 공개 Train split에서 내려받았고, `codex-train-reference.jsonl`과 `codex-train-query-comparison.jsonl`은 Issue #8의 Codex 시각 전사 및 query 비교 산출물이다. Codex 전사는 사람 검수 gold가 아니라 silver reference다.

| 파일 | 행 | SHA-256 |
| --- | ---: | --- |
| `train-tasks.jsonl` | 908 | `6d9cd9087d0c5e30bfc17c83aec30752403d4109fb93d8357f534da425969489` |
| `train-labels.jsonl` | 908 | `3e39dfb708cccc7999676d23aae8342fb0e71a94a8e1b5629339c6f0209dc33f` |
| `codex-train-reference.jsonl` | 908 | `2e06fb6bac61601776049c03e3c20f3dcf905feee77510342f67cf19bc1f0558` |
| `codex-train-query-comparison.jsonl` | 908 | `99d642ace2e5080222c75ca6b5b1d6ea78707ea9261ab267a7d67c655e2f4628` |

원 데이터 revision은 [`amitbcp/docinsights-2026-shared-task-data@b171c5a`](https://huggingface.co/datasets/amitbcp/docinsights-2026-shared-task-data/tree/b171c5ad488f0c8c50df05951a5b288ea50e9501), 문서 생성의 canonical revision은 [`oracle-samples/gsm-sem@332158b`](https://github.com/oracle-samples/gsm-sem/tree/332158b2549e7e8a1186e2ae3a922669e9018808/docsem)이다. 이 스냅샷은 공개 Train에만 해당하며 validation이나 hidden holdout을 포함하지 않는다.
