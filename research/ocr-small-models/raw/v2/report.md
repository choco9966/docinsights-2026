# Fixed DocSem OCR smoke v2

- run_id: `task_000909-1788108847-64706348c218`
- input: `task_000909` PDF only; pages 1-2 rendered by Poppler at 200 dpi
- preflight generation ceiling: `512` tokens per page
- truncation warning: 512 generated tokens can truncate a page; this run does not claim full coverage
- device policy: `CUDA_VISIBLE_DEVICES=0`, model device `cuda:0`, sequential fresh children
- blind-input policy: no task, query, label, answer, or evidence source is read

| model | gate | status | load s | doc s | peak RSS B | peak VRAM B | raw B |
|---|---|---:|---:|---:|---:|---:|---:|
| PaddleOCR-VL-1.6 | candidate | failed | None | None | 4690857984 | 106954752 | 0 |

`PaddleOCR-VL-1.6` error: `AttributeError: 'PaddleOCRVLConfig' object has no attribute 'text_config'`
| GLM-OCR | diagnostic_gate_fail | succeeded | 3.323099165000258 | 55.94829136299995 | 4690857984 | 7442792448 | 4271 |
| surya-ocr-2 | candidate | succeeded | 1.9743208589998176 | 50.6098722910001 | 4690857984 | 3298820096 | 1333 |
| granite-docling-258M | candidate | succeeded | 1.6991891850002503 | 352.622557785 | 4690857984 | 1124073472 | 1024 |
| SmolDocling-256M-preview | replacement_candidate | succeeded | 1.150883315000101 | 38.96223640900007 | 4690857984 | 1157627904 | 3729 |

The JSONL is authoritative. CSV is a lossless field-for-field projection; nested values are JSON strings.
Child stdout/stderr and raw model text are stored verbatim and content-addressed in the artifact manifest.
