import json
from types import SimpleNamespace

import pytest

from oprb_lab import worker
from oprb_lab.models import output_text
from oprb_lab.storage import read_rows


@pytest.mark.parametrize('fail',[False,True])
def test_batch_worker_remainder_errors_and_resume(tmp_path,monkeypatch,fail):
    monkeypatch.setattr(worker,'TORCH_MODELS',set())
    calls=[]
    class Fake:
        parameter_count=None
        metadata={}
        def __init__(self,*args,**kw): pass
        def predict(self,path): return output_text('cat')
        def predict_batch(self,paths):
            calls.append(paths)
            if fail: raise RuntimeError('batch failed')
            return [output_text('cat') for p in paths]
    monkeypatch.setattr(worker,'Adapter',Fake)
    gt=tmp_path/'gt.txt'
    gt.write_text('cat\t'+'\t'.join(['0']*10))
    (tmp_path/'config.json').write_text(json.dumps(dict(model='granite',seed=42,revision='test',max_tokens=10,page_batch_size=2)))
    (tmp_path/'manifest.json').write_text(json.dumps([dict(uid=f'text/{i}',image_path=str(i),gt_path=str(gt)) for i in range(3)]))
    worker.run(tmp_path,0,1)
    rows=read_rows(tmp_path/'rank0.jsonl')
    assert len(rows)==3
    assert calls==[['0','1'],['2']]
    assert {r['status'] for r in rows}==({'error'} if fail else {'ok'})
    if not fail:
        assert [r['actual_page_batch'] for r in rows]==[2,2,1]
        assert rows[0]['latency_sec']*2==pytest.approx(rows[0]['batch_wall_sec'])
        worker.run(tmp_path,0,1)
        assert len(calls)==2


def test_tesseract_worker_does_not_import_gpu_frameworks(tmp_path,monkeypatch):
    import builtins
    original=builtins.__import__
    def guard(name,*args,**kw):
        if name in ('torch','paddle','onnxruntime'): raise AssertionError('CPU worker imported GPU framework')
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',guard)
    monkeypatch.setattr(worker,'Adapter',lambda *args,**kw:SimpleNamespace(parameter_count=None,metadata={},predict=lambda p:output_text('cat')))
    gt=tmp_path/'gt.txt'
    gt.write_text('cat\t'+'\t'.join(['0']*10))
    (tmp_path/'config.json').write_text(json.dumps(dict(model='tesseract',seed=42,revision='x',max_tokens=1)))
    (tmp_path/'manifest.json').write_text(json.dumps([dict(uid='text/a',image_path='a',gt_path=str(gt))]))
    worker.run(tmp_path,0,1)
    assert read_rows(tmp_path/'rank0.jsonl')[0]['cer']==0
