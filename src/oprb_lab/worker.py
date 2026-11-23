"""Independent GPU worker; launch via the CLI, never fork a CUDA context."""
import argparse
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from .data import read_gt, shard
from .metrics import score
from .models import Adapter
from .storage import read_rows, rewrite_rows


def run(run_dir, rank, world_size):
    # CUDA_VISIBLE_DEVICES is set by the parent before importing any GPU framework.
    import torch
    root = Path(run_dir)
    config = json.loads((root / "config.json").read_text())
    samples = json.loads((root / "manifest.json").read_text())
    selected = shard(samples, rank, world_size)
    path = root / f"rank{rank}.jsonl"
    old = read_rows(path)
    completed = {r["uid"]: r for r in old if r["status"] == "ok"}
    rewrite_rows(path, list(completed.values()))
    todo = [s for s in selected if s["uid"] not in completed]
    if not todo:
        return
    model = Adapter(config["model"], config["revision"], config["max_tokens"],
                    config["max_pixels"], config["max_tiles"])
    def synchronize():
        if config["model"] == "paddle":
            import paddle
            paddle.device.cuda.synchronize()
        else:
            torch.cuda.synchronize()
    # Warm-up is intentionally outside measured latency. Failure stops the worker.
    model.predict(todo[0]["image_path"])
    synchronize()
    metadata = dict(rank=rank, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    gpu=torch.cuda.get_device_name(0), parameter_count=model.parameter_count,
                    torch=version("torch"))
    if config["model"] != "paddle":
        metadata["transformers"] = version("transformers")
    else:
        metadata["paddleocr"] = version("paddleocr")
    (root / f"rank{rank}_metadata.json").write_text(json.dumps(metadata, indent=2))
    for i, sample in enumerate(todo):
        row = dict(sample)
        try:
            synchronize()
            start = time.perf_counter()
            output = model.predict(sample["image_path"])
            synchronize()
            elapsed = time.perf_counter() - start
            gt = read_gt(sample["gt_path"])
            row.update(status="ok", latency_sec=elapsed, ground_truth=gt,
                       **output, **score(gt, output["prediction"]))
        except Exception as error:
            row.update(status="error", error=repr(error))
            print(f"rank={rank} ERROR {sample['uid']}: {error}", flush=True)
            # Do not change resolution or precision silently after OOM.
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"rank={rank} {i+1}/{len(todo)} {row['status']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("rank", type=int)
    parser.add_argument("world_size", type=int)
    args = parser.parse_args()
    run(args.run_dir, args.rank, args.world_size)
