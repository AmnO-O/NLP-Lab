"""Build a standalone Kaggle notebook from the current benchmark sources."""
import json
from pathlib import Path


def build(root, model="qwen", portable=False):
    root = Path(root)
    files = [root / "pyproject.toml", *sorted((root / "src").rglob("*.py")),
             *sorted((root / "tests").glob("test_*.py")),
             *sorted((root / "requirements").glob("*.txt"))]
    payload = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in files}
    cells = []

    def cell(kind, source):
        entry = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    cell("markdown", "# OPRB pretrained OCR benchmark — Kaggle T4 ×2\n"
         "Enable **Internet** and **GPU T4 ×2** in notebook settings. This notebook includes the source code. "
         "Start with 5 pages per category; use `LIMIT = None` for the full test. "
         "Use a fresh session when switching to/from InternVL. Download results before ending the session.")
    cell("code", 'from pathlib import Path\nimport json, subprocess, sys\n'
         f'FILES = {payload!r}\n' +
         ('BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/content") if Path("/content").exists() else Path.cwd()\n' if portable else 'BASE = Path("/kaggle/working")\n') +
         'ROOT = BASE / "oprb-lab"\n'
         'ROOT.mkdir(parents=True, exist_ok=True)\n'
         'for name, content in FILES.items():\n'
         '    path = ROOT / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n'
         '    path.write_text(content, encoding="utf-8")\n'
         f'MODEL = {model!r}  # paddle, granite, got, internvl, qwen\n'
         'LIMIT = 5\nDATA = BASE / "oprb-data"\n'
         'OUTPUT = BASE / "oprb-results"\n' +
         ('GPUS = "0"  # Change to "0,1" on Kaggle T4 x2\n' if portable else 'GPUS = "0,1"\n') +
         'def run(*args):\n'
         '    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)\n')
    cell("code", 'assert MODEL in ("paddle", "granite", "got", "internvl", "qwen")\n'
         'run("-m", "pip", "install", "-e", ".[test]")\n'
         'requirement = "paddle" if MODEL == "paddle" else "internvl" if MODEL == "internvl" else "vlm"\n'
         'run("-m", "pip", "install", "-r", f"requirements/{requirement}.txt")\n'
         'if MODEL == "paddle":\n'
         '    # Change this index if the Kaggle CUDA environment requires another wheel.\n'
         '    run("-m", "pip", "install", "paddlepaddle-gpu==3.2.0", "-i",\n'
         '        "https://www.paddlepaddle.org.cn/packages/stable/cu126/")\n'
         'run("-m", "pytest", "-q", "tests/test_core.py", "tests/test_comparison.py")\n')
    cell("code", 'run("-c", "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); "\n'
         '    f"assert torch.cuda.device_count() >= {len(GPUS.split(chr(44)))}, \'Enable the selected GPUs\'")\n'
         'if MODEL == "paddle":\n'
         '    run("-c", "import paddle; assert paddle.is_compiled_with_cuda(); "\n'
         '        f"assert paddle.device.cuda.device_count() >= {len(GPUS.split(chr(44)))}")\n'
         'run("-m", "oprb_lab", "download", "--data", DATA)\n')
    cell("code", 'args = ["-m", "oprb_lab", "run", "--model", MODEL, "--data", DATA,\n'
         '        "--output", OUTPUT, "--gpus", GPUS]\n'
         'if LIMIT is not None:\n    args += ["--limit", str(LIMIT)]\n'
         'run(*args)  # Repeat this cell to resume the same configuration.\n')
    cell("code", 'import shutil\n'
         'for path in sorted(OUTPUT.glob("*/summary.json")):\n'
         '    print(path.parent.name)\n    print(path.read_text())\n'
         'archive = shutil.make_archive(str(BASE / "oprb-results"), "zip", OUTPUT)\n'
         'from IPython.display import FileLink, display\n'
         'display(FileLink(archive))\n')
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                language_info=dict(name="python", version="3.11")), nbformat=4, nbformat_minor=5)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    destination = root / "notebooks" / "OPRB_Kaggle_2GPU.ipynb"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(build(root), ensure_ascii=False, indent=1), encoding="utf-8")
