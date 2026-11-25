import argparse
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from .data import collect, download
from .models import MODELS, PROMPT
from .data import file_sha
from .report import summarize
from .storage import fingerprint


def show_worker_logs(root, workers, tail=80):
    """Expose startup/load failures hidden in redirected worker output."""
    for rank, worker in enumerate(workers):
        path = Path(root) / f"rank{rank}.log"
        print(f"Worker {rank}: exit code {worker.returncode}; log: {path}", flush=True)
        if path.exists():
            from collections import deque
            with path.open(encoding="utf-8", errors="replace") as stream:
                print("".join(deque(stream, maxlen=tail)), flush=True)


def launch(args):
    from huggingface_hub import model_info
    samples = collect(args.data, args.limit, args.seed)
    gpus = args.gpus.split(",") if args.model != "tesseract" else [""] * getattr(args, "cpu_workers", 2)
    if args.model != "tesseract" and (any(not g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus)):
        raise ValueError("gpus must be unique numeric IDs, e.g. 0,1")
    page_batch_size = getattr(args, "page_batch_size", 1)
    if args.model != "granite" and page_batch_size != 1:
        raise ValueError("Page batching is supported only for Granite; use recognition batching for OCR")
    revision = model_info(MODELS[args.model], revision=args.revision).sha if args.model == "granite" else "package-pretrained-assets"
    tessdata = getattr(args, "tessdata", None)
    packages = {}
    for name in ("torch", "torchvision", "transformers", "paddleocr", "paddlepaddle-gpu", "docling-core", "Pillow", "rapidfuzz", "easyocr", "rapidocr-onnxruntime", "onnxruntime-gpu"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    config = dict(schema=3, model=args.model, model_id=MODELS[args.model], revision=revision, packages=packages,
                  precision="float16" if args.model == "granite" else "pretrained engine default",
                  max_tokens=args.max_tokens, rec_batch_size=getattr(args, "rec_batch_size", 16),
                  page_batch_size=page_batch_size, tessdata=str(Path(tessdata).resolve()) if tessdata else None,
                  tessdata_sha=file_sha(Path(tessdata) / "eng.traineddata") if args.model == "tesseract" and tessdata else None,
                  execution="CPU processes" if args.model == "tesseract" else "one process per GPU",
                  timing_mode="single page latency" if page_batch_size == 1 else "batch wall time / actual batch pages (throughput)",
                  prompt=PROMPT if args.model == "granite" else None,
                  limit=args.limit, seed=args.seed, world_size=len(gpus),
                  manifest_sha=fingerprint([{k: v for k, v in s.items() if k not in ("image_path", "gt_path")} for s in samples]),
                  normalization="NFKC whitespace casefold", gt_order="annotation order",
                  code_sha=fingerprint({p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")}))
    root = Path(args.output) / f"{args.model}-{fingerprint(config)}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "manifest.json").write_text(json.dumps(samples, indent=2))
    workers, logs = [], []
    start = time.perf_counter()
    try:
        for rank, gpu in enumerate(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
                       PYTHONHASHSEED=str(args.seed), CUBLAS_WORKSPACE_CONFIG=":4096:8")
            log = (root / f"rank{rank}.log").open("a", encoding="utf-8")
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable, "-u", "-m", "oprb_lab.worker", str(root), str(rank), str(len(gpus))],
                                             env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"Running {len(workers)} workers. Logs/results: {root}", flush=True)
        while any(w.poll() is None for w in workers):
            counts = {f"rank{rank}": sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
                      for rank in range(len(workers)) for p in [root / f"rank{rank}.jsonl"]}
            print(f"Saved rows: {counts}", flush=True)
            time.sleep(10)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            worker.wait()
        for log in logs:
            log.close()
    (root / "session.json").write_text(json.dumps(dict(wall_sec=time.perf_counter()-start,
                                                        exit_codes=[w.returncode for w in workers]), indent=2))
    summary = summarize(root)
    if any(w.returncode for w in workers) or not summary["complete"]:
        show_worker_logs(root, workers)
        raise SystemExit("Incomplete evaluation: inspect logs; rerun the same command to retry errors/missing pages.")


def main():
    parser = argparse.ArgumentParser(description="OPRB pretrained OCR benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--data", default="data/oprb")
    fetch.add_argument("--revision", default="main", help="Dataset revision; resolved to a commit before downloading")
    report = sub.add_parser("summarize")
    report.add_argument("run_dir")
    run = sub.add_parser("run")
    run.add_argument("--model", choices=MODELS, required=True)
    run.add_argument("--data", default="data/oprb")
    run.add_argument("--output", default="results")
    run.add_argument("--gpus", default="0,1")
    run.add_argument("--limit", type=int, default=None, help="Pages per category; omit for full test")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--revision", default="main")
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--max-pixels", type=int, default=1280*28*28)
    run.add_argument("--max-tiles", type=int, default=12)
    run.add_argument("--rec-batch-size", type=int, default=16, help="OCR line-crop recognition batch size")
    run.add_argument("--page-batch-size", type=int, default=1, help="Granite pages per batch; >1 measures amortized throughput")
    run.add_argument("--cpu-workers", type=int, default=2, help="Tesseract CPU process count")
    run.add_argument("--tessdata", help="Directory containing tessdata_fast/eng.traineddata")
    args = parser.parse_args()
    if args.command == "download":
        download(args.data, args.revision)
    elif args.command == "summarize":
        summarize(args.run_dir)
    else:
        if min(args.max_tokens, args.max_pixels, args.max_tiles, args.rec_batch_size, args.page_batch_size, args.cpu_workers) < 1:
            parser.error("Token, pixel, and tile limits must be positive")
        if args.model == "tesseract" and not args.tessdata:
            parser.error("Tesseract requires --tessdata pointing to tessdata_fast/eng.traineddata")
        launch(args)


if __name__ == "__main__":
    main()
