"""Verify page/token alignment and EOS accounting without downloading a VLM."""
import contextlib
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from oprb_lab.models import Adapter


def test_granite_batch_preserves_page_mapping_and_counts_eos(tmp_path, monkeypatch):
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (10+i, 10)).save(path)
        paths.append(path)
    class InputTensor:
        shape = (2, 3)
        def is_floating_point(self): return False
    class Inputs(dict):
        def to(self, device): return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs): return "prompt"
        def __call__(self, **kwargs):
            assert len(kwargs["images"]) == 2 and kwargs["padding"]
            return Inputs(input_ids=InputTensor())
        def decode(self, ids, **kwargs): return ",".join(map(str, ids.tolist()))
    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return np.array([[1, 2, 3, 4, 5, 9, 0], [1, 2, 3, 6, 9, 0, 0]])
    class Doc:
        @staticmethod
        def load_from_doctags(value, **kwargs):
            return SimpleNamespace(export_to_text=lambda: value)
    class Tags:
        @staticmethod
        def from_doctags_and_image_pairs(raw, images):
            return f"{images[0].width}:{raw[0]}"
    monkeypatch.setitem(sys.modules, "docling_core.types.doc", SimpleNamespace(DoclingDocument=Doc))
    monkeypatch.setitem(sys.modules, "docling_core.types.doc.document", SimpleNamespace(DocTagsDocument=Tags))
    a = Adapter.__new__(Adapter)
    a.name, a.max_tokens, a.device, a.dtype = "granite", 4, "cuda:0", "fp16"
    a.processor, a.model = Processor(), Model()
    a.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    result = a.predict_batch(paths)
    assert [r["prediction"] for r in result] == ["10:4,5,9", "11:6,9"]
    assert [r["generated_tokens"] for r in result] == [3, 2]
    assert not any(r["possibly_truncated"] for r in result)
