# OPRB: four lightweight OCR pipelines + one small VLM

Evaluate **pretrained** models on the OPRB test split. No training, clean images, GT-guided crops or tuning on test annotations. GT is used only for scoring. Full evaluation means 9,000 pages per model (45,000 page/model pairs). No benchmark accuracy or GPU success is claimed without saved outputs.

| Key | Pretrained pipeline | Execution | Primary notebook |
|---|---|---|---|
| `paddle` | PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec | Paddle CUDA, default FP32 | `OPRB_01_PP_OCRv5_Mobile_Evaluation.ipynb` |
| `rapid` | PP-OCRv4 mobile detector + recognizer bundled in rapidocr_onnxruntime 1.4.4 | ONNX CUDA, FP32 | `OPRB_02_PP_OCRv4_RapidOCR_Evaluation.ipynb` |
| `easyocr` | CRAFT + English CRNN (`english_g2`) | PyTorch CUDA, FP32 | `OPRB_03_EasyOCR_Evaluation.ipynb` |
| `tesseract` | Tesseract LSTM, `eng` from `tessdata_fast` | CPU; OEM 1, PSM 3 | `OPRB_04_Tesseract_Fast_Evaluation.ipynb` |
| `granite` | ibm-granite/granite-docling-258M | PyTorch CUDA, FP16, SDPA | `OPRB_05_Granite_Docling_258M_Evaluation.ipynb` |

RapidOCR is an execution wrapper, not an additional model name: its explicitly selected PP-OCRv4 weights differ from PP-OCRv5. The bundled v4 dictionary is Chinese/English, while EasyOCR and Tesseract use English. Granite is the only generative VLM in the primary set. Historical GOT/InternVL/Qwen and superseded notebooks are preserved in `notebooks/legacy`, excluded from comparisons.

## Kaggle T4 x2

Enable Internet and T4 x2. Use `notebooks/OPRB_Kaggle_2GPU.ipynb` to run one selected `MODEL`, or `notebooks/OPRB_Kaggle_All5_Smoke.ipynb` to install and smoke-test all five sequentially. Each notebook embeds the current package and tests; no repository upload is needed.

Defaults: `LIMIT=5` selects five pages per category (10 total), `GPUS="0,1"`, recognition batch 16, page batch 1. After every engine passes inference and produces valid results, use `LIMIT=None` for the complete 6,000 text-region + 3,000 nontext-region pages. The smoke test alone is not evidence of model ranking. Each selected file is hashed; identical seed, limit and data produce the same manifest across engines. Different limits are not guaranteed nested subsets.

Paddle, RapidOCR, EasyOCR and Granite use one independent replica per GPU and disjoint page shards. There is no DDP overhead. Tesseract uses `CPU_WORKERS=2` CPU processes and does not use GPUs. Increasing its process count should respect available CPU cores and one OpenMP thread per process.

Each engine has an isolated Python virtualenv under `/tmp/oprb-envs-v2/<model>`. It does not inherit host Python packages, but preserves driver library paths. Paddle uses CPU-only Torch for indirect ModelScope imports while OCR runs with CUDA Paddle. EasyOCR/Granite use a matching pinned Torch/torchvision CUDA 12.6 pair. RapidOCR replaces its CPU ONNX dependency with ONNX Runtime GPU 1.23.2 plus CUDA/cuDNN dependencies, preloads those libraries, and checks actual detector/recognizer session providers; it refuses silent CPU fallback. Tesseract installs its system binary if missing and downloads `eng.traineddata` at a resolved official repository commit. Package versions, weight hashes when accessible, device settings and model metadata are saved. Do not describe all five as GPU or FP16 runs.

The notebook checks the NVIDIA runtime, imports model dependencies and preloads weights once before launching two workers to avoid concurrent downloads of the same cache. Download result ZIPs before ending the session; a new Kaggle session does not automatically restore `/kaggle/working` or `/tmp`.

## Batching and timing

`REC_BATCH_SIZE=16` batches recognized **line crops** in Paddle, RapidOCR and EasyOCR. Detection still processes one page at a time. `PAGE_BATCH_SIZE` applies only to Granite; start at 1 and test 2 or 4 separately if VRAM permits. There is no automatic precision/resolution/batch fallback after OOM.

Single-page timing includes image load, preprocessing, inference, text decoding and structural conversion, with framework synchronization where applicable. It excludes GT scoring, model load and an untimed one-page warm-up. Recognition-crop batching is included. With Granite page batch >1, `latency_sec` is the completed batch wall time divided by its actual page count: an amortized throughput measure, **not single-request latency**. Rows also record batch wall time and actual batch size, including the final partial batch. Record the chosen batch sizes. The comparison table exposes these settings and omits Pareto labels when latency and throughput timing modes are mixed. CPU and GPU timing comparisons must be described as measurements of the configured deployments.

## Results, resume and comparison

Each run creates `results/<model>-<fingerprint>/` containing config, manifest, worker metadata/logs and per-worker JSONL. Config changes, code changes and changed selected file hashes create a different fingerprint. Resume skips only successful UIDs (`category/sample_id`), retries errors and recovers an interrupted final JSONL line. Keep the same worker count when resuming. Restore the exact result directory after a session reset. Model load and warm-up failures stop workers; their log tails are printed and incomplete evaluations return a failure exit status.

Outputs: `per_image_results.csv`, `summary.json`, metrics by category/type/density and `error_rates.png`. Report expected/success/error/missing counts alongside macro means over successful pages. Errors are not silently assigned zero error.

Use `OPRB_06_Comparative_Analysis_and_Report.ipynb`, set `RUNS` to one fingerprinted folder per model. It requires the same selected content hashes and evaluation protocol, reports coverage, and computes a second comparison on the intersection of successful pages across all five. It exports paired transcripts, quantitative CSV/LaTeX tables and plots. Review examples manually; a WER difference alone does not prove hallucination or recovery. For the assignment, write a PDF of **less than four A4 pages** with Introduction, Dataset Statistics, Model Descriptions, Experimental Results and Conclusion. Include region/type comparisons, qualitative examples and accuracy/size/time trade-offs.

## Local development

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e '.[test]'
.venv\Scripts\python generate_notebooks.py
.venv\Scripts\python -m pytest -q
```

The notebook handles engine-specific installs. For manual execution, install the appropriate `requirements/<key>.txt` and GPU runtime in a separate environment. RapidOCR's legacy wrapper declares CPU ONNX Runtime as a dependency: uninstall that distribution before installing the GPU build as shown in the notebook. A global `pip check` will still report that distribution-name requirement; execution is validated through actual ORT imports/providers instead. All other dependencies should remain consistent.

```bash
python -m oprb_lab download --data data/oprb
python -m oprb_lab run --model paddle --data data/oprb --gpus 0,1 --limit 5 --rec-batch-size 16
python -m oprb_lab run --model granite --data data/oprb --gpus 0,1 --limit 5 --page-batch-size 2
python -m oprb_lab run --model tesseract --data data/oprb --cpu-workers 2 --tessdata path/to/tessdata_fast --limit 5
# Full test: omit --limit
python -m oprb_lab.comparison RUN_PADDLE RUN_RAPID RUN_EASY RUN_TESS RUN_GRANITE --output comparison
```

Regenerate every notebook with `python generate_notebooks.py` after changing source, requirements or tests. CPU tests cover adapters with mocked runtimes, provider-fallback rejection, batching/remainders, error/resume behavior, worker isolation, metrics, reporting and embedded notebook freshness. They do not substitute for actual pretrained GPU inference. Model and dataset revisions are recorded where supported; archive downloaded model caches for final reproducibility. Seeded inference improves repeatability but is not a bitwise guarantee across hardware/software changes.

## Scoring limitations and sources

NFKC, whitespace normalization and casefold are applied equally to GT and predictions. CER/WER use minimum Levenshtein distance over reference character/word counts and may exceed 1. Token P/R/F1 use multiset overlap and preserve duplicates. GT token order follows annotation-file order; multi-column reading-order differences can increase sequence error. Do not use GT geometry to reorder predictions. OCR engines keep their own predicted reading order. Granite DocTags are converted through Docling rather than deleted with regex. A generative token budget can truncate pages; limit hits are recorded.

- Assignment: `LabOCR.pdf`.
- Dataset: https://huggingface.co/datasets/kpurkayastha/OPRB
- Paddle: https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html
- RapidOCR pinned API: https://rapidai.github.io/RapidOCRDocs/v1.4.4/install_usage/api/RapidOCR/
- ONNX CUDA dependencies: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Tesseract fast pretrained weights: https://github.com/tesseract-ocr/tessdata_fast
- Granite: https://huggingface.co/ibm-granite/granite-docling-258M

Models/data retain upstream licenses; consult them before redistribution.
