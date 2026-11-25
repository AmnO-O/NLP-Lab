"""Readable code cells embedded by the notebook generator."""
from textwrap import dedent

SETUP = dedent('''
ROOT = BASE / "oprb-lab"
ROOT.mkdir(parents=True, exist_ok=True)
for name, content in FILES.items():
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
LIMIT = 5  # None = complete 9,000-page test set, after all smoke tests pass.
DATA = Path("/tmp/oprb-data") if os.name != "nt" else BASE / "oprb-data"
OUTPUT = BASE / "oprb-results"
OUTPUT.mkdir(parents=True, exist_ok=True)
REC_BATCH_SIZE = 16  # Line crops, not pages (Paddle / RapidOCR / EasyOCR).
PAGE_BATCH_SIZE = 1  # Granite only; >1 measures amortized batch throughput.
CPU_WORKERS = 2  # Tesseract is CPU-only.
TESSDATA = OUTPUT / "_assets" / "tessdata_fast"
ENV = dict(os.environ)
ENV["MPLBACKEND"] = "Agg"
PYTHON = sys.executable

def run(*args):
    subprocess.run([PYTHON, *map(str, args)], cwd=ROOT, env=ENV, check=True)

def prepare_model_python():
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        ENV.pop(key, None)
    ENV["PYTHONNOUSERSITE"] = "1"
    # Keep large virtualenvs out of Kaggle output uploads.
    target = (Path("/tmp/oprb-envs-v2") if os.name != "nt" else BASE / "oprb-envs-v2") / MODEL
    subprocess.run([sys.executable, "-m", "pip", "install", "virtualenv>=20.26"], env=ENV, check=True)
    subprocess.run([sys.executable, "-m", "virtualenv", str(target)], env=ENV, check=True)
    return str(target / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
''')

INSTALL = dedent('''
assert MODEL in ("paddle", "rapid", "easyocr", "tesseract", "granite")
gpu_ids = GPUS.split(",")
assert all(g.isdigit() for g in gpu_ids) and len(set(gpu_ids)) == len(gpu_ids), "GPUS must be unique numeric IDs"
print("Selected model:", MODEL, "GPU IDs:", GPUS, "host Python:", sys.version)
if MODEL != "tesseract":
    run("-c", "import shutil, subprocess; smi=shutil.which('nvidia-smi'); assert smi, 'Enable GPU accelerator'; subprocess.run([smi, '-L'], check=True)")
PYTHON = prepare_model_python()
if MODEL == "paddle":
    run("-m", "pip", "install", "torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cpu")
elif MODEL in ("easyocr", "granite"):
    run("-m", "pip", "install", "torch==2.9.1", "torchvision==0.24.1", "--index-url", "https://download.pytorch.org/whl/cu126")
run("-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "-e", ".[test]")
run("-m", "pip", "install", "-r", f"requirements/{MODEL}.txt")
if MODEL == "paddle":
    run("-m", "pip", "install", "paddlepaddle-gpu==3.2.0", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    run("-c", "import torch; assert torch.version.cuda is None; from paddleocr import PaddleOCR; print('PaddleOCR import passed')")
elif MODEL == "rapid":
    # The legacy wrapper depends on the CPU ORT distribution; replace that distribution
    # with the pinned GPU build, rather than installing both in the same namespace.
    run("-m", "pip", "uninstall", "-y", "onnxruntime")
    run("-m", "pip", "install", "onnxruntime-gpu[cuda,cudnn]==1.23.2")
    run("-c", "import onnxruntime as ort; ort.preload_dlls(directory=''); assert 'CUDAExecutionProvider' in ort.get_available_providers()")
elif MODEL == "tesseract":
    import shutil, urllib.request
    if not shutil.which("tesseract"):
        prefix = [] if os.geteuid() == 0 else ["sudo"]
        subprocess.run([*prefix, "apt-get", "update", "-qq"], check=True)
        subprocess.run([*prefix, "apt-get", "install", "-y", "tesseract-ocr"], check=True)
    TESSDATA.mkdir(parents=True, exist_ok=True)
    weights = TESSDATA / "eng.traineddata"
    if not weights.exists():
        # Resolve a commit before downloading pretrained tessdata_fast weights.
        request = urllib.request.Request("https://api.github.com/repos/tesseract-ocr/tessdata_fast/commits/main", headers={"User-Agent": "oprb-lab"})
        with urllib.request.urlopen(request) as response:
            revision = json.load(response)["sha"]
        temporary = weights.with_suffix(".part")
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{revision}/eng.traineddata", temporary)
        temporary.replace(weights)
        (TESSDATA / "revision.txt").write_text(revision)
run("-c", "import rapidfuzz, PIL, huggingface_hub, pandas, matplotlib, tqdm, pytest; from oprb_lab.metrics import score; assert score('cat','cat')['cer']==0; print('Scoring dependencies verified')")
run("-m", "pytest", "-q", "tests/test_core.py", "tests/test_comparison.py", "tests/test_adapters.py", "tests/test_launcher.py", "tests/test_worker.py")
''')

GPU_DATA = dedent('''
if MODEL == "paddle":
    run("-c", "import paddle; assert paddle.is_compiled_with_cuda(); "
        f"ids = {list(map(int, GPUS.split(',')))!r}; "
        "n = paddle.device.cuda.device_count(); print('Paddle:',paddle.__version__,'GPUs:',n); "
        "assert ids and min(ids)>=0 and max(ids)<n, 'Selected GPU IDs unavailable'; "
        "[print(i,paddle.device.cuda.get_device_name(i)) for i in ids]; "
        "paddle.set_device('gpu:'+str(ids[0])); paddle.zeros([1]); paddle.device.cuda.synchronize()")
elif MODEL in ("easyocr", "granite"):
    run("-c", "import torch; "
        f"ids = {list(map(int, GPUS.split(',')))!r}; "
        "n=torch.cuda.device_count(); print('Torch:',torch.__version__,'GPUs:',n); "
        "assert torch.cuda.is_available() and ids and min(ids)>=0 and max(ids)<n, 'CUDA unavailable or selected GPU IDs missing'; "
        "[print(i,torch.cuda.get_device_name(i)) for i in ids]; torch.zeros(1,device='cuda:'+str(ids[0])); torch.cuda.synchronize(ids[0])")
elif MODEL == "rapid":
    run("-c", "import onnxruntime as ort; ort.preload_dlls(directory=''); print(ort.get_available_providers())")
else:
    print("Tesseract uses CPU processes, not GPUs.")
run("-m", "oprb_lab", "download", "--data", DATA)
# Preload weights once to avoid two GPU workers racing to download the same files.
preflight = ("from huggingface_hub import model_info; from oprb_lab.models import Adapter, MODELS; "
             f"name={MODEL!r}; rev=model_info(MODELS[name]).sha if name=='granite' else 'assets'; "
             f"a=Adapter(name,rev,rec_batch_size={REC_BATCH_SIZE},tessdata={str(TESSDATA)!r}); print(a.metadata)")
run("-c", preflight)
''')

RUN = dedent('''
args = ["-m", "oprb_lab", "run", "--model", MODEL, "--data", DATA, "--output", OUTPUT,
        "--gpus", GPUS, "--rec-batch-size", str(REC_BATCH_SIZE), "--cpu-workers", str(CPU_WORKERS),
        "--page-batch-size", str(PAGE_BATCH_SIZE if MODEL == "granite" else 1)]
if MODEL == "tesseract":
    args += ["--tessdata", str(TESSDATA)]
if LIMIT is not None:
    args += ["--limit", str(LIMIT)]
run(*args)
''')

EXPORT = dedent('''
import shutil
for path in sorted(OUTPUT.glob("*/summary.json")):
    print(path.parent.name, path.read_text())
archive = shutil.make_archive(str(BASE / "oprb-results"), "zip", OUTPUT)
from IPython.display import FileLink, display
display(FileLink(archive))
''')
