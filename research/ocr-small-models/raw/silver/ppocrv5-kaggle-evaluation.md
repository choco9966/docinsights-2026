# Codex Silver OCR Evaluation

> 이 점수는 Codex-assisted silver agreement이며 human-gold accuracy가 아니다.

- Engine: `PP-OCRv5 mobile (Kaggle CPU)`
- Instances: 217
- Silver text score: 99.6176 / 100
- Micro CER / WER: 0.003824 / 0.014463
- Symmetric character similarity: 0.996184
- NFKC+whitespace similarity: 0.996184
- Ordered block exact: 216 / 217
- Exact-token F1 / ordered-quantity F1: 0.996611 / 0.996448
- Mean latency: 58.9155 sec/doc

## Sources

- Reference: `issue8/codex-validation-reference.jsonl` (`d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a`)
- Prediction: `kaggle/version-3/result-shard-00-of-01.jsonl` (`60e1844155e70fc5f4cea218e86be4ac2e6ca9fa35d4699fc820c568231c0fd1`)
