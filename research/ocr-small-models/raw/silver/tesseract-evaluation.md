# Codex Silver OCR Evaluation

> 이 점수는 Codex-assisted silver agreement이며 human-gold accuracy가 아니다.

- Engine: `Tesseract eng PSM 6 200 DPI`
- Instances: 217
- Silver text score: 99.9415 / 100
- Micro CER / WER: 0.000585 / 0.006029
- Symmetric character similarity: 0.999415
- NFKC+whitespace similarity: 0.999415
- Ordered block exact: 217 / 217
- Exact-token F1 / ordered-quantity F1: 0.992057 / 0.991993
- Mean latency: 6.0335 sec/doc

## Sources

- Reference: `/Users/choco/.codex/worktrees/bed4/docinsights-2026/artifacts/ocr/codex-validation-reference.jsonl` (`d8cefce5507a74e6424bd6555fb9f67a14881f2b53891b3d08e39013ca10bc4a`)
- Prediction: `/Users/choco/Documents/project/docinsights-2026-issue-8/artifacts/ocr/tesseract-200dpi-psm6-final.jsonl` (`8b5db676267a0a1ab51c345798994eb5f38f4b5148728e54adbb40cf94acadaf`)
