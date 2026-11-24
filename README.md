# OPRB: lightweight OCR and VLM evaluation

Evaluate five **pretrained** models on the OPRB test split. No training, test-time updates, clean-image input, or GT-guided crops. This is experimental benchmark code, not a claim that the models have already been evaluated.

| CLI key | Model | Execution |
|---|---|---|
| `paddle` | PP-OCRv5 Mobile detector + recognizer | PaddleOCR; default FP32 |
| `granite` | ibm-granite/granite-docling-258M | FP16, SDPA, DocTags to plain text |
| `got` | stepfun-ai/GOT-OCR-2.0-hf | FP16, SDPA, unformatted OCR |
| `internvl` | OpenGVLab/InternVL2_5-1B | FP16, dynamic tiles + thumbnail, no FlashAttention |
| `qwen` | Qwen/Qwen3-VL-2B-Instruct | FP16, SDPA, fixed OCR prompt |

## Primary notebooks

These five notebooks use the same package in `src/oprb_lab`. There is no automatic substitution of models after a load failure.

| Model | Standalone notebook |
|---|---|
| PP-OCRv5 Mobile | `OPRB_01_PP_OCRv5_Mobile_Evaluation.ipynb` |
| Granite-Docling-258M | `OPRB_02_Granite_Docling_258M_Evaluation.ipynb` |
| GOT-OCR2.0 | `OPRB_03_GOT_OCR2_Evaluation.ipynb` |
| InternVL2_5-1B | `OPRB_04_InternVL2_5_1B_Evaluation.ipynb` |
| Qwen3-VL-2B-Instruct | `OPRB_05_Qwen3_VL_2B_Evaluation.ipynb` |

Upload one notebook to Colab or Kaggle with Internet enabled. A primary notebook defaults to `GPUS = "0"`; change this to `"0,1"` for Kaggle T4 x2. Use a fresh session per model and download the results ZIP after each run. Set `LIMIT = None` only after a GPU smoke test. The default selects five random pages per category, not the first ten pages. Equal limits, seed and unchanged dataset yield equal samples; different limits are not guaranteed nested subsets.

`OPRB_06_Comparative_Analysis_and_Report.ipynb` embeds the comparison package. Restore five result folders, then set `RUNS` to their fingerprinted directory paths. It checks identical sample manifests, data hashes and evaluation code, reports coverage, and computes metrics on pages successfully processed by all five models. Outputs include `coverage.csv`, `comparison.csv`, `comparison.tex`, grouped metrics, paired transcripts and a latency/WER plot with Pareto membership recorded in the table. It does not infer hallucination from WER or generate an entire written report. Use equivalent hardware, software stacks where possible, and the same GPU concurrency for latency comparisons; metadata records these differences for review.

Historical EasyOCR, PaddleOCR-VL, Qwen2.5 and superseded standalone notebooks are in `notebooks/legacy/`. They are excluded from the primary benchmark and retain known issues for historical reference.

## Kaggle quick start

Open `notebooks/OPRB_Kaggle_2GPU.ipynb`, enable Internet and **GPU T4 x2**. Set the model key, run the install cell, then run all remaining cells. Use a fresh session when changing to/from InternVL because its remote code uses an older Transformers version. Initial default: **5 images per category**, not a full evaluation. Set `LIMIT = None` after the technical smoke test for all 9,000 test pages.

The launcher starts **one independent process and model replica per GPU**, each with its own `CUDA_VISIBLE_DEVICES`. It splits a common manifest by index, not model layers. Workers write separate JSONL files and logs; results are merged by category-qualified sample IDs. No DDP or inter-GPU communication is necessary. This improves dataset throughput; per-image latency is measured while both GPUs are active, so CPU contention may affect it.

The notebook embeds the package, requirements, and core tests: upload only the notebook to Kaggle. After changing package sources or tests, regenerate all primary notebooks with `python generate_notebooks.py`. GitHub Actions checks CPU unit tests on Python 3.10 and 3.12. Model inference still requires a separate GPU run.

## Local environment / command line

```bash
python -m venv .venv
# Linux/Kaggle
source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
python -m pip install -r requirements/vlm.txt
python -m pytest -q
python -m oprb_lab download --data data/oprb
python -m oprb_lab run --model qwen --data data/oprb --gpus 0,1 --limit 5
# Full benchmark: omit --limit
python -m oprb_lab run --model qwen --data data/oprb --gpus 0,1
```

Install a matching CUDA torch/torchvision pair first on a local GPU machine. Kaggle normally provides these. For InternVL use `requirements/internvl.txt` instead of `requirements/vlm.txt`. For Paddle use `requirements/paddle.txt` plus a **CUDA-enabled PaddlePaddle wheel matching the machine** from https://www.paddlepaddle.org.cn/install/quick . The notebook includes a CUDA 12.6 wheel-index example; change it if required, and verify Paddle's CUDA support before running. Do not install the CPU `paddlepaddle` wheel for a GPU benchmark.

## Results and resume

Each configuration creates `results/<model>-<fingerprint>/` with a resolved HF checkpoint commit, source-code hash, sample manifest, config, per-worker metadata/logs, raw output, plain text prediction, and per-page scores. Repeating the same command skips successful rows and retries errors/missing pages. Keep the same GPU count to resume. Configuration changes create a new directory. Restore a saved results directory under `--output` after a Kaggle session reset; `/kaggle/working` does not automatically persist into a fresh session.

The resolved HF revision is pinned for the run. For Paddle, package version is recorded but upstream asset URLs are not immutable: archive the downloaded Paddle model cache with your final experiment for reproducibility. Paddle uses FP32 defaults; do not label the whole comparison FP16. Weight parameter counts are recorded for VLMs, not inferred from model names. A model load/warm-up failure stops that worker and is visible in its log. Incomplete evaluations exit with failure; do not present them as full test results.

Dataset downloads resolve a commit before fetching all four archives and record it in completion markers. Selected image and annotation SHA-256 hashes enter the manifest; checking these adds disk-read overhead outside inference timing. Installed inference package versions enter the run fingerprint. Workers seed Python, NumPy and their GPU framework; PyTorch requests deterministic algorithms with warnings for unsupported operations. This improves repeatability but does not promise bitwise reproducibility across machines, kernels or dependency versions.

Outputs: `per_image_results.csv`, `summary.json`, grouped CSVs by region/type/density, and `error_rates.png`. Summaries are **macro means over successful pages**, with expected/success/error/missing counts. Errors are not silently assigned zero error. Also report failures alongside scores. The `session.json` wall time includes loading and warm-up and is not per-image inference latency.

## Evaluation policy and limitations

- Inputs are only `test/text/img` and `test/nontext/img`; `gt` is read by scoring code only. Archives are fetched separately, not through `load_dataset`.
- GT tokens are concatenated in annotation-file order. Inspect representative multi-column pages before interpreting sequence CER/WER; reading-order differences can contribute substantial error. The implementation does not use GT geometry to reorder model output or crop images.
- CER/WER use minimum Levenshtein edit distance divided by reference character/word count. Scores can exceed 1. Normalize NFKC, whitespace, and case identically. Token precision/recall/F1 use **multiset overlap**, preserve duplicate counts, and ignore order.
- Timing includes image loading, preprocessing, model inference, decoding, and structural conversion; excludes GT scoring and model loading. CUDA is synchronized and one page is used for untimed warm-up. Batch size is one.
- Generative outputs have a 4,096-token default budget; track limit hits. InternVL's chat API does not expose raw output IDs here, so truncation is **unknown**, not false. Inspect raw predictions for incomplete pages before final evaluation.
- Granite's structural tokens are converted through Docling. Formula serialization can differ from DocBank tokens. Paddle joins recognized lines in its pipeline order; this is not a separate layout-aware reading-order model.
- Resolution/tile settings follow each model's processor, not an identical visual-token budget. Record these differences. Do not tune prompts, resolutions, model selection, or output cleaning to maximize scores on test GT.
- No automatic precision/resolution fallback after OOM. Pick and document a new configuration, then rerun consistently.
- A T4 GPU test is still required; unit tests validate data/metrics/resume/report behavior, not pretrained model quality or CUDA compatibility.

## Sources

- Assignment: `LabOCR.pdf`; original one-model example: `notebooks/legacy/OPRB_Qwen2_5_VL_3B_Test_Evaluation.ipynb` (unchanged, Colab/NF4).
- Dataset: https://huggingface.co/datasets/kpurkayastha/OPRB
- Paddle: https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html
- Granite: https://huggingface.co/ibm-granite/granite-docling-258M
- GOT: https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf
- InternVL (dynamic preprocessing adapted from the model card): https://huggingface.co/OpenGVLab/InternVL2_5-1B
- Qwen: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

Models/data retain their upstream licenses; consult their cards before redistribution. This repository does not bundle weights or the OPRB dataset.
