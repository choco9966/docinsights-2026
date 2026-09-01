# Fixed DocSem OCR smoke v2

- run_id: `task_000909-1788113789-4e5be04c3afb`
- input: `task_000909` PDF only; pages 1-2 rendered by Poppler at 200 dpi
- preflight generation ceiling: `512` tokens per page
- truncation warning: 512 generated tokens can truncate a page; this run does not claim full coverage
- device policy: `CUDA_VISIBLE_DEVICES=0`, model device `cuda:0`, sequential fresh children
- blind-input policy: no task, query, label, answer, or evidence source is read

| model | gate | status | load s | doc s | parent RSS B | child RSS B | parent VRAM B | child allocated VRAM B | raw B |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PaddleOCR-VL-1.6 | candidate | failed | None | None | 1112231936 | 1115394048 | 106954752 | 0 | 0 |

`PaddleOCR-VL-1.6` error: `AttributeError: 'PaddleOCRVLConfig' object has no attribute 'text_config'`
| GLM-OCR | diagnostic_gate_fail | succeeded | 3.0922998229998484 | 55.47661837199985 | 4199813120 | 4360134656 | 7442792448 | 6586189824 | 4271 |
| surya-ocr-2 | candidate | succeeded | 1.8646921310000835 | 47.02600554599985 | 2859540480 | 3020025856 | 3298820096 | 2900749312 | 1333 |
| granite-docling-258M | candidate | succeeded | 1.7478552319998926 | 354.49694065899985 | 2262228992 | 2369581056 | 1124073472 | 890579456 | 1024 |
| SmolDocling-256M-preview | replacement_candidate | succeeded | 1.3554703590002646 | 38.16190364899967 | 2316926976 | 2369949696 | 1157627904 | 887485952 | 3729 |

The JSONL is authoritative. CSV is a lossless field-for-field projection; nested values are JSON strings.
Child stdout/stderr and raw model text are stored verbatim and content-addressed in the artifact manifest.
