"""Offline adapter regression tests; real GPU tests run separately on Kaggle."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from oprb_lab.models import Adapter, MODELS, ort_session


def test_registry_is_exact_new_five():
    assert set(MODELS) == {'paddle', 'rapid', 'easyocr', 'tesseract', 'granite'}
    with pytest.raises(ValueError):
        Adapter('qwen')


def test_paddle_passes_recognition_batch(monkeypatch):
    params = {}
    class OCR:
        def __init__(self, **kw): params.update(kw)
        def predict(self, path): return [{'rec_texts':['one','two']}]
    monkeypatch.setitem(sys.modules,'paddleocr',SimpleNamespace(PaddleOCR=OCR))
    adapter=Adapter('paddle',rec_batch_size=32)
    assert params['text_recognition_batch_size']==32
    assert params['text_detection_model_name']=='PP-OCRv5_mobile_det'
    assert adapter.predict('x')['prediction']=='one\ntwo'


def test_rapid_verifies_real_sessions_and_pinned_weights(tmp_path,monkeypatch):
    models=tmp_path/'models'
    models.mkdir()
    for name in ('ch_PP-OCRv4_det_infer.onnx','ch_PP-OCRv4_rec_infer.onnx'):
        (models/name).write_bytes(b'weights')
    providers=['CUDAExecutionProvider','CPUExecutionProvider']
    session=SimpleNamespace(get_providers=lambda:providers)
    params={}
    class Rapid:
        def __init__(self,**kw):
            params.update(kw)
            self.text_det=SimpleNamespace(infer=SimpleNamespace(session=session))
            self.text_rec=SimpleNamespace(session=SimpleNamespace(session=session))
        def __call__(self,path,**kw):
            assert kw['use_cls'] is False
            return [[[], 'cat', .9]],None
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(preload_dlls=lambda **kw:None))
    monkeypatch.setitem(sys.modules,'rapidocr_onnxruntime',SimpleNamespace(__file__=str(tmp_path/'__init__.py'),RapidOCR=Rapid))
    a=Adapter('rapid',rec_batch_size=16)
    assert a.predict('x')['prediction']=='cat'
    assert params['det_use_cuda'] and params['rec_use_cuda']
    providers[:]=['CPUExecutionProvider']
    with pytest.raises(RuntimeError,match='fallback'):
        Adapter('rapid')


def test_ort_unknown_wrapper_fails():
    with pytest.raises(ValueError): ort_session(object())


def test_tesseract_uses_explicit_fast_weights_and_timeout(tmp_path,monkeypatch):
    from oprb_lab import models
    (tmp_path/'eng.traineddata').write_bytes(b'fast')
    monkeypatch.setattr(models.shutil,'which',lambda name:'/usr/bin/tesseract')
    commands=[]
    def execute(cmd,**kw):
        commands.append((cmd,kw))
        return 'tesseract 5.5' if '--version' in cmd else 'some text\n'
    monkeypatch.setattr(models.subprocess,'check_output',execute)
    a=Adapter('tesseract',tessdata=tmp_path)
    assert a.predict('page.jpg')['prediction']=='some text'
    command,options=commands[-1]
    assert command[command.index('--tessdata-dir')+1]==str(tmp_path)
    assert command[command.index('--psm')+1]=='3'
    assert options['timeout']==180
    assert a.metadata['weights_sha']
