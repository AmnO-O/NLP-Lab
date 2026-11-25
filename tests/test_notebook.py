"""Validate generated notebooks and environment routing without downloading models."""
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def notebook(name='notebooks/OPRB_Kaggle_2GPU.ipynb'):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def context(model, calls):
    return dict(MODEL=model,GPUS='0,1',DATA='unused',TESSDATA=Path('assets'),REC_BATCH_SIZE=16,
                sys=sys,prepare_model_python=lambda:'/isolated/python',run=lambda *args:calls.append(args))


def test_five_notebooks_and_suite_embed_current_sources():
    from generate_notebooks import NAMES
    assert set(NAMES)=={'paddle','rapid','easyocr','tesseract','internvl'}
    for name in [*NAMES.values(),'notebooks/OPRB_Kaggle_2GPU.ipynb','notebooks/OPRB_Kaggle_All5_Smoke.ipynb','OPRB_06_Comparative_Analysis_and_Report.ipynb']:
        nb=notebook(name)
        for cell in nb['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
        tree=ast.parse(''.join(nb['cells'][1]['source']))
        files=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets))
        for folder in ('src','tests','requirements'):
            for path in (ROOT/folder).rglob('*.py' if folder!='requirements' else '*.txt'):
                assert files[path.relative_to(ROOT).as_posix()]==path.read_text(encoding='utf-8-sig')


def test_model_isolation_keeps_driver_paths(tmp_path):
    tree=ast.parse(''.join(notebook()['cells'][1]['source']))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    calls=[]
    env={'LD_LIBRARY_PATH':'/driver','PYTHONPATH':'/host'}
    ns=dict(BASE=tmp_path,ROOT=tmp_path,MODEL='rapid',os=os,sys=sys,ENV=env,
            subprocess=SimpleNamespace(run=lambda *args,**kw:calls.append((args,kw))))
    exec(compile(ast.Module(body=functions,type_ignores=[]),'setup','exec'),ns)
    ns['PYTHON']=ns['prepare_model_python']()
    ns['run']('-c','pass')
    assert 'rapid' in calls[-1][0][0][0]
    assert env['LD_LIBRARY_PATH']=='/driver'
    assert 'PYTHONPATH' not in env
    assert '--system-site-packages' not in calls[1][0][0]


@pytest.mark.parametrize('model',['paddle','rapid','easyocr','internvl'])
def test_install_selects_matching_engine_and_complete_dependencies(model):
    calls=[]
    exec(''.join(notebook()['cells'][2]['source']),context(model,calls))
    installs=[c for c in calls if c[:3]==('-m','pip','install')]
    editable=next(c for c in installs if '-e' in c)
    assert '.[test]' in editable and '--no-deps' not in editable
    assert any(f'requirements/{model}.txt' in c for c in installs)
    if model=='paddle':
        assert any('https://download.pytorch.org/whl/cpu' in c for c in installs)
        assert any('from paddleocr import PaddleOCR' in str(c) for c in calls)
    if model=='rapid':
        assert ('-m','pip','uninstall','-y','onnxruntime') in calls
        assert any('onnxruntime-gpu[cuda,cudnn]==1.23.2' in c for c in installs)


def test_no_gpu_is_detected_before_querying_device_name(monkeypatch):
    calls=[]
    exec(''.join(notebook()['cells'][3]['source']),context('internvl',calls))
    fake=SimpleNamespace(__version__='test',cuda=SimpleNamespace(device_count=lambda:0,is_available=lambda:False))
    monkeypatch.setitem(sys.modules,'torch',fake)
    with pytest.raises(AssertionError,match='CUDA unavailable'):
        exec(calls[0][1])
