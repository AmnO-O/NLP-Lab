"""Generate the five primary notebooks from the shared, tested package."""
import json
from pathlib import Path
from tools.build_notebook import build

NAMES = {
    'paddle': 'OPRB_01_PP_OCRv5_Mobile_Evaluation.ipynb',
    'granite': 'OPRB_02_Granite_Docling_258M_Evaluation.ipynb',
    'got': 'OPRB_03_GOT_OCR2_Evaluation.ipynb',
    'internvl': 'OPRB_04_InternVL2_5_1B_Evaluation.ipynb',
    'qwen': 'OPRB_05_Qwen3_VL_2B_Evaluation.ipynb',
}
LEGACY = ['OPRB_01_EasyOCR_Evaluation.ipynb', 'OPRB_03_PaddleOCR_Mobile_Evaluation.ipynb',
          'OPRB_04_PaddleOCR_VL_Evaluation.ipynb', 'OPRB_Qwen2_5_VL_3B_Test_Evaluation.ipynb',
          'OPRB_05_Comparative_Analysis_and_Report.ipynb']


def generate(root):
    root = Path(root).resolve()
    archive = root / 'notebooks' / 'legacy'
    archive.mkdir(parents=True, exist_ok=True)
    # Move only these explicit files, within this project, without replacing an archive.
    for name in LEGACY:
        source, target = root / name, archive / name
        if source.exists():
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
    (archive / 'README.md').write_text(
        '# Historical notebooks\n\nExcluded from the primary benchmark. These files retain known issues '
        '(resume, timing, model selection) and are preserved only for reference. '
        'Use the five notebooks at the project root.\n', encoding='utf-8')
    for key, name in NAMES.items():
        nb = build(root, model=key, portable=True)
        nb['cells'][0]['source'] = [f'# OPRB primary benchmark: {key}\n',
            'Shared tested pipeline. Colab: one GPU; Kaggle: set GPUS = "0,1" for T4 x2.\n',
            'Use a fresh session per model. LIMIT=5 is a smoke test; None selects all test pages.\n',
            'Download results before ending the session. Reproducibility is documented, not guaranteed bitwise.\n']
        (root / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    nb = build(root)
    (root / 'notebooks' / 'OPRB_Kaggle_2GPU.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    # Reuse the embedded package setup so the analysis notebook also works independently.
    analysis = build(root, portable=True)
    analysis['cells'] = [analysis['cells'][0], analysis['cells'][1]]
    analysis['cells'][0]['source'] = ['# Five-model matched-page comparison\n',
        'Restore all five result folders, set RUNS below, then run. Coverage is reported separately.\n',
        'Keep hardware and GPU concurrency identical when comparing latency.\n']
    for source in ['run("-m", "pip", "install", "-e", ".")\n',
                   'RUNS = []  # Five fingerprinted folders containing config.json, manifest.json and rank*.jsonl\n'
                   'assert len(RUNS) == 5, "Set one result directory for each of the five models"\n'
                   'run("-m", "oprb_lab.comparison", *RUNS, "--output", BASE / "oprb-comparison")\n']:
        analysis['cells'].append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source.splitlines(keepends=True)))
    (root / 'OPRB_06_Comparative_Analysis_and_Report.ipynb').write_text(json.dumps(analysis, indent=1), encoding='utf-8')


if __name__ == '__main__':
    generate(Path(__file__).resolve().parent)
