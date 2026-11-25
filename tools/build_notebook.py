"""Build standalone notebooks; all engines share the tested src package."""
import json
from pathlib import Path
from tools.notebook_cells import SETUP, INSTALL, GPU_DATA, RUN, EXPORT


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build(root, model='paddle', portable=False, suite=False):
    root = Path(root)
    paths = [root / 'pyproject.toml', *sorted((root/'src').rglob('*.py')),
             *sorted((root/'tests').glob('test_*.py')), *sorted((root/'requirements').glob('*.txt'))]
    payload = {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8-sig') for p in paths}
    setup = 'from pathlib import Path\nimport json, os, subprocess, sys\n' + f'FILES = {payload!r}\n'
    setup += ('BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/content") if Path("/content").exists() else Path.cwd()\n'
              if portable else 'BASE = Path("/kaggle/working")\n')
    setup += f'MODEL = {model!r}\n' + ('GPUS = "0"\n' if portable else 'GPUS = "0,1"\n') + SETUP
    cells = [cell('markdown', '# OPRB: four lightweight OCR pipelines + Granite-Docling\n'
                  'Kaggle: Internet ON, T4 x2. LIMIT=5 is a smoke test; None selects all 9,000 pages. '
                  'Recognition batching is not page batching. Tesseract uses CPU processes. '
                  'Each model runs in a separate virtualenv. Download results before ending the session.\n'), cell('code', setup)]
    if suite:
        source = f'INSTALL_CELL = {INSTALL!r}\nGPU_DATA_CELL = {GPU_DATA!r}\nRUN_CELL = {RUN!r}\n'
        source += 'MODELS_TO_RUN = ["paddle", "rapid", "easyocr", "tesseract", "granite"]\nfailures = {}\n'
        source += 'for MODEL in MODELS_TO_RUN:\n    try:\n        PYTHON = sys.executable\n        exec(INSTALL_CELL)\n        exec(GPU_DATA_CELL)\n        exec(RUN_CELL)\n    except Exception as error:\n        import traceback\n        traceback.print_exc()\n        failures[MODEL] = repr(error)\n'
        source += '(BASE / "suite_status.json").write_text(json.dumps(failures, indent=2))\n' + EXPORT
        source += '\nassert not failures, f"Incomplete model smoke tests: {failures}"\n'
        cells.append(cell('code', source))
    else:
        cells += [cell('code', INSTALL), cell('code', GPU_DATA), cell('code', RUN), cell('code', EXPORT)]
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')), nbformat=4, nbformat_minor=5)


if __name__ == '__main__':
    # Use python -m tools.build_notebook from the project root.
    root = Path(__file__).resolve().parents[1]
    (root/'notebooks/OPRB_Kaggle_2GPU.ipynb').write_text(json.dumps(build(root), indent=1), encoding='utf-8')
