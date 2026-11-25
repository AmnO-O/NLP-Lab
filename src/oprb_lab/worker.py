"""Shard pages over independent GPU processes, or CPU processes for Tesseract."""
import argparse
import json
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

from .data import read_gt, shard
from .metrics import score
from .models import Adapter, TORCH_MODELS
from .storage import resume_rows


def run(run_dir, rank, world_size):
    root = Path(run_dir)
    config = json.loads((root / 'config.json').read_text())
    name = config['model']
    random.seed(config['seed'])
    import numpy as np
    np.random.seed(config['seed'])
    torch = None
    if name == 'paddle':
        import paddle
        paddle.seed(config['seed'])
    elif name in TORCH_MODELS:
        import torch
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    samples = json.loads((root / 'manifest.json').read_text())
    selected = shard(samples, rank, world_size)
    path = root / f'rank{rank}.jsonl'
    todo = resume_rows(path, selected)
    if not todo:
        return
    model = Adapter(name, config['revision'], config['max_tokens'],
                    rec_batch_size=config.get('rec_batch_size', 16), tessdata=config.get('tessdata'))
    def synchronize():
        if name == 'paddle':
            paddle.device.cuda.synchronize()
        elif torch is not None:
            torch.cuda.synchronize()
        # ONNX run() and the Tesseract subprocess return completed CPU text.
    model.predict(todo[0]['image_path'])
    synchronize()
    metadata = dict(rank=rank, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    parameter_count=model.parameter_count, python=platform.python_version(),
                    platform=platform.platform(), adapter=getattr(model, 'metadata', {}))
    if torch is not None:
        metadata.update(gpu=torch.cuda.get_device_name(0), torch=version('torch'), cuda=torch.version.cuda)
    elif name == 'paddle':
        metadata.update(gpu=paddle.device.cuda.get_device_name(0), paddle=paddle.__version__,
                        cuda=paddle.version.cuda(), paddleocr=version('paddleocr'))
    elif name == 'rapid':
        metadata.update(onnxruntime=version('onnxruntime-gpu'), gpu='CUDAExecutionProvider; worker GPU mask')
    else:
        metadata.update(gpu=None, device='CPU')
    (root / f'rank{rank}_metadata.json').write_text(json.dumps(metadata, indent=2))
    batch_size = config.get('page_batch_size', 1)
    for offset in range(0, len(todo), batch_size):
        batch = todo[offset:offset+batch_size]
        try:
            synchronize()
            start = time.perf_counter()
            outputs = model.predict_batch([sample['image_path'] for sample in batch]) if batch_size > 1 else [model.predict(batch[0]['image_path'])]
            synchronize()
            elapsed = time.perf_counter() - start
            if len(outputs) != len(batch):
                raise ValueError('Adapter output count differs from batch size')
            batch_error = None
        except Exception as error:
            outputs, elapsed, batch_error = [None] * len(batch), None, repr(error)
            if torch is not None and isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        for index, (sample, output) in enumerate(zip(batch, outputs), start=offset+1):
            row = dict(sample)
            try:
                if batch_error:
                    raise RuntimeError(batch_error)
                gt = read_gt(sample['gt_path'])
                row.update(status='ok', latency_sec=elapsed/len(batch), batch_wall_sec=elapsed,
                           actual_page_batch=len(batch), timing_mode=config.get('timing_mode', 'single page latency'),
                           ground_truth=gt, **output, **score(gt, output['prediction']))
            except Exception as error:
                row.update(status='error', error=repr(error))
                print(f"rank={rank} ERROR {sample['uid']}: {error}", flush=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            print(f"rank={rank} {index}/{len(todo)} {row['status']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('rank', type=int)
    parser.add_argument('world_size', type=int)
    args = parser.parse_args()
    run(args.run_dir, args.rank, args.world_size)
