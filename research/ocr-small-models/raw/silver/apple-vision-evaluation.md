# Codex Silver OCR Evaluation

> 이 점수는 Codex-assisted silver agreement이며 human-gold accuracy가 아니다.

- Engine: `Apple Vision accurate 200 DPI`
- Instances: 217
- Silver text score: 99.3777 / 100
- Micro CER / WER: 0.006223 / 0.008946
- Symmetric character similarity: 0.993800
- NFKC+whitespace similarity: 0.993799
- Ordered block exact: 217 / 217
- Exact-token F1 / ordered-quantity F1: 0.985062 / 0.981604
- Mean latency: 3.8781 sec/doc

## Sources

- Reference: `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-reference.jsonl` (`d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a`)
- Prediction: `/Users/choco/Documents/project/docinsights-2026-issue-8/artifacts/ocr/apple-vision-200dpi.jsonl` (`8d55f10f9f628cdc6744f451d1c04de5158495a6452ae123d0ff9670d1908c01`)
