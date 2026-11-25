"""Generate the current primary notebooks and preserve superseded model notebooks."""
import json
from pathlib import Path
from tools.build_notebook import build, cell

NAMES = {
 'paddle': 'OPRB_01_PP_OCRv5_Mobile_Evaluation.ipynb',
 'rapid': 'OPRB_02_PP_OCRv4_RapidOCR_Evaluation.ipynb',
 'easyocr': 'OPRB_03_EasyOCR_Evaluation.ipynb',
 'tesseract': 'OPRB_04_Tesseract_Fast_Evaluation.ipynb',
 'internvl': 'OPRB_05_InternVL2_5_1B_Evaluation.ipynb',
}
LEGACY = ['OPRB_01_EasyOCR_Evaluation.ipynb', 'OPRB_03_PaddleOCR_Mobile_Evaluation.ipynb',
 'OPRB_04_PaddleOCR_VL_Evaluation.ipynb', 'OPRB_Qwen2_5_VL_3B_Test_Evaluation.ipynb',
 'OPRB_05_Comparative_Analysis_and_Report.ipynb', 'OPRB_02_Granite_Docling_258M_Evaluation.ipynb',
 'OPRB_03_GOT_OCR2_Evaluation.ipynb', 'OPRB_04_InternVL2_5_1B_Evaluation.ipynb', 'OPRB_05_Qwen3_VL_2B_Evaluation.ipynb',
 'OPRB_05_Granite_Docling_258M_Evaluation.ipynb']


def generate(root):
 root = Path(root).resolve()
 archive = root/'notebooks/legacy'
 archive.mkdir(parents=True, exist_ok=True)
 for name in LEGACY:
  source, target = root/name, archive/name
  if source.exists():
   if target.exists():
    raise FileExistsError(target)
   source.rename(target)
 (archive/'README.md').write_text('# Historical notebooks\n\nExcluded from the current four-OCR + one-VLM benchmark; preserved for reference only.\n', encoding='utf-8')
 for model, name in NAMES.items():
  (root/name).write_text(json.dumps(build(root, model=model, portable=True), ensure_ascii=False, indent=1), encoding='utf-8')
 (root/'notebooks/OPRB_Kaggle_2GPU.ipynb').write_text(json.dumps(build(root), indent=1), encoding='utf-8')
 (root/'notebooks/OPRB_Kaggle_All5_Smoke.ipynb').write_text(json.dumps(build(root, suite=True), indent=1), encoding='utf-8')
 analysis = build(root, portable=True)
 analysis['cells'] = [cell('markdown', '# Matched-page five-model comparison\nSet RUNS to five fingerprinted run folders; compare timing only on matching devices/concurrency and timing modes.'), analysis['cells'][1]]
 analysis['cells'] += [cell('code','run("-m", "pip", "install", "-e", ".")\n'), cell('code','RUNS = []\nassert len(RUNS) == 5\nrun("-m", "oprb_lab.comparison", *RUNS, "--output", BASE / "oprb-comparison")\n')]
 (root/'OPRB_06_Comparative_Analysis_and_Report.ipynb').write_text(json.dumps(analysis, indent=1), encoding='utf-8')


if __name__ == '__main__':
 generate(Path(__file__).resolve().parent)
