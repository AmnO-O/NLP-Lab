"""Paddle must run even when the installed PyTorch CUDA build cannot import."""
import builtins
import json
import sys
from types import SimpleNamespace

import pytest

from oprb_lab import worker
from oprb_lab.storage import read_rows


@pytest.mark.parametrize("fail_page", [False, True])
def test_paddle_worker_never_imports_torch(tmp_path, monkeypatch, fail_page):
    calls = []
    class OCR:
        def __init__(self, **kwargs):
            assert kwargs["text_detection_model_name"] == "PP-OCRv5_mobile_det"
        def predict(self, path):
            calls.append(path)
            if fail_page and len(calls) > 1:
                raise RuntimeError("page failed")
            return [{"rec_texts": ["cat"]}]
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=OCR))
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(
        seed=lambda seed: None, __version__="test", version=SimpleNamespace(cuda=lambda: "12.6"),
        device=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda n: "fake GPU"))))
    monkeypatch.setattr(worker, "version", lambda name: "test")
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            raise AssertionError("Paddle tried to import broken torch")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    (tmp_path / "config.json").write_text(json.dumps(dict(model="paddle", seed=42, revision="test", max_tokens=10, max_pixels=100, max_tiles=1)))
    gt = tmp_path / "gt.txt"
    gt.write_text("cat\t" + "\t".join(["0"] * 10))
    (tmp_path / "manifest.json").write_text(json.dumps([dict(uid="text/a", image_path="fake.jpg", gt_path=str(gt))]))
    worker.run(tmp_path, 0, 1)
    rows = read_rows(tmp_path / "rank0.jsonl")
    assert rows[0]["status"] == ("error" if fail_page else "ok")
    if not fail_page:
        assert rows[0]["cer"] == 0
    assert json.loads((tmp_path / "rank0_metadata.json").read_text())["paddle"] == "test"
