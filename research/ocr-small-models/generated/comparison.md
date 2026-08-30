# DocSem 소형 OCR 비교표

Codex 전사는 human gold가 아닌 silver reference다.

| model | revision | params | weight_gib | license | device_runtime | samples | quality_samples | inference_success_rate | valid_ocr_rate | silver_agreement_cer | silver_agreement_wer | query_raw_exact | query_normalized_exact | query_sha256_exact | block_fidelity | load_sec | sec_per_doc | docs_per_min | peak_ram_bytes | peak_vram_bytes | output_bytes | cost | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PaddlePaddle/PaddleOCR-VL-1.6 | c5630abae1d940eafe0697512a0325494b02ab42 | 900000000 | 1.79 | Apache-2.0 | Kaggle Python 3.12.13 Linux; NVIDIA T4 x2 allocated; cuda:0 only used; transformers 5.12.1 | 1 | 0 | 0.0 | 0.0 | NA(no_valid_output) | NA(no_valid_output) | 1/1 | 1/1 | 1/1 | NA(no_valid_output) | NA(not_measured) | NA(not_measured) | NA(not_measured) | 2093256704 | 0 | 0 | free Kaggle quota | inference_failed |
| zai-org/GLM-OCR | ca5d8b3e287e52589e37c28385d9655ee4372f9d | 900000000 | 2.47 | MIT | Kaggle Python 3.12.13 Linux; NVIDIA T4 x2 allocated; cuda:0 only used; transformers 5.12.1 | 1 | 1 | 1.0 | 1.0 | 0.15475395234914274 | 0.20353982300884957 | 1/1 | 1/1 | 1/1 | F1=0.954545; ordered=False | 12.176340469000024 | 55.58000593600002 | 1.079524893701695 | 4463763456 | 6586189824 | 4273 | free Kaggle quota | valid OCR; compared with Codex silver, not human gold |
| datalab-to/surya-ocr-2 | 3b3d4cdf88d6928b0acdc75181b13206ea67c4a3 | 650000000 | 1.28 | modified AI Pubs Open RAIL-M (commercial restriction above USD 5M revenue) | Kaggle Python 3.12.13 Linux; NVIDIA T4 x2 allocated; cuda:0 only used; transformers 5.12.1 | 1 | 1 | 1.0 | 1.0 | 0.8122912491649966 | 0.8082595870206489 | 1/1 | 1/1 | 1/1 | F1=0.357143; ordered=False | 16.818865213000038 | 47.770574561000046 | 1.2560033148310523 | 4691189760 | 2901400064 | 1335 | free Kaggle quota | valid OCR; compared with Codex silver, not human gold |
| ibm-granite/granite-docling-258M | 982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe | 258000000 | 0.48 | Apache-2.0 | Kaggle Python 3.12.13 Linux; NVIDIA T4 x2 allocated; cuda:0 only used; transformers 5.12.1 | 1 | 0 | 1.0 | 0.0 | NA(no_valid_output) | NA(no_valid_output) | 1/1 | 1/1 | 1/1 | NA(no_valid_output) | NA(not_measured) | NA(not_measured) | NA(not_measured) | NA(not_captured_in_downloaded_results_jsonl) | NA(not_captured_in_downloaded_results_jsonl) | 1026 | free Kaggle quota | degenerate_repeated_character |

## 기존 OCR 운영 baseline (엔진 간 agreement이며 accuracy가 아님)

| model | documents | pages | blocks | failures | total_seconds | sec_per_doc | peak_ram_bytes | engine_agreement_cer | engine_agreement_wer | engine_agreement_block_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Apple Vision | 217 | 434 | 4991 | 0 | 842.06 | 3.878 | 224903168 | 0.00665542 | 0.01431542 | 1.0 |
| Tesseract PSM 6 | 217 | 434 | 4991 | 0 | 1310.04 | 6.033 | 112754688 | 0.00665542 | 0.01431542 | 1.0 |
