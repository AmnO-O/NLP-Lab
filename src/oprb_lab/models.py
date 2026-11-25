"""Five image-only pretrained OCR adapters. GT is never passed here."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image

MODELS = {
    "paddle": "PP-OCRv5_mobile_det+PP-OCRv5_mobile_rec",
    "rapid": "PP-OCRv4-Mobile-ONNX (rapidocr_onnxruntime-1.4.4)",
    "easyocr": "EasyOCR-English-CRAFT+CRNN",
    "tesseract": "Tesseract-eng-tessdata_fast",
    "granite": "ibm-granite/granite-docling-258M",
}
PROMPT = "Convert this page to docling."
TORCH_MODELS = {"easyocr", "granite"}


def output_text(text, raw=None, tokens=None, truncated=False):
    return dict(prediction=text, raw_prediction=text if raw is None else raw,
                generated_tokens=tokens, possibly_truncated=truncated)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ort_session(component):
    """Find the actual ORT session in the pinned RapidOCR wrapper."""
    if hasattr(component, "get_providers"):
        return component
    for attr in ("session", "infer", "predictor"):
        child = getattr(component, attr, None)
        if child is not None and child is not component:
            try:
                return ort_session(child)
            except ValueError:
                pass
    raise ValueError("Cannot verify RapidOCR CUDA session; refusing an unverified GPU run")


class Adapter:
    def __init__(self, name, revision=None, max_tokens=4096, max_pixels=None, max_tiles=None,
                 rec_batch_size=16, tessdata=None):
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")
        self.name, self.max_tokens, self.rec_batch_size = name, max_tokens, rec_batch_size
        self.parameter_count = None
        self.metadata = {}
        if name == "paddle":
            from paddleocr import PaddleOCR
            self.model = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                text_recognition_batch_size=rec_batch_size, device="gpu:0")
            cache = Path.home() / ".paddlex" / "official_models"
            self.metadata["weight_hashes"] = {
                str(path.relative_to(cache)): sha(path)
                for model in ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec")
                for path in (cache / model).glob("*.pdiparams")}
            return
        if name == "rapid":
            import onnxruntime as ort
            ort.preload_dlls(directory="")
            import rapidocr_onnxruntime as package
            from rapidocr_onnxruntime import RapidOCR
            model_dir = Path(package.__file__).parent / "models"
            det, rec = model_dir / "ch_PP-OCRv4_det_infer.onnx", model_dir / "ch_PP-OCRv4_rec_infer.onnx"
            if not det.exists() or not rec.exists():
                raise FileNotFoundError("Pinned PP-OCRv4 weights missing from RapidOCR package")
            self.model = RapidOCR(det_model_path=str(det), rec_model_path=str(rec),
                                  det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=False,
                                  rec_batch_num=rec_batch_size, intra_op_num_threads=1, inter_op_num_threads=1)
            providers = {key: ort_session(getattr(self.model, key)).get_providers() for key in ("text_det", "text_rec")}
            if any("CUDAExecutionProvider" not in values for values in providers.values()):
                raise RuntimeError(f"RapidOCR GPU provider failed; refusing silent CPU fallback: {providers}")
            self.metadata = dict(providers=providers, det_sha=sha(det), rec_sha=sha(rec),
                                 language="bundled Chinese/English PP-OCRv4 dictionary", classifier=False)
            return
        if name == "tesseract":
            self.binary = shutil.which("tesseract")
            if not self.binary:
                raise RuntimeError("Install the tesseract-ocr system binary first")
            self.tessdata = Path(tessdata or os.environ.get("OPRB_TESSDATA", ""))
            weights = self.tessdata / "eng.traineddata"
            if not weights.exists():
                raise FileNotFoundError("Provide --tessdata containing tessdata_fast/eng.traineddata")
            self.metadata = dict(tesseract=subprocess.check_output([self.binary, "--version"], text=True).splitlines()[0],
                                 weights_sha=sha(weights), language="eng", oem=1, psm=3)
            return
        import torch
        self.torch, self.device, self.dtype = torch, "cuda:0", torch.float16
        if name == "easyocr":
            import easyocr
            self.model = easyocr.Reader(["en"], gpu=self.device, recog_network="standard", verbose=False)
            self.parameter_count = sum(p.numel() for module in (self.model.detector, self.model.recognizer) for p in module.parameters())
            self.metadata = dict(language="en", detector="CRAFT", recognizer="english_g2")
            cache = getattr(self.model, "model_storage_directory", None)
            if cache:
                self.metadata["weight_hashes"] = {path.name: sha(path) for path in Path(cache).glob("*.pth")}
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(MODELS[name], revision=revision)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(MODELS[name], revision=revision,
                    torch_dtype=self.dtype, low_cpu_mem_usage=True, attn_implementation="sdpa").to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def predict(self, path):
        if self.name == "paddle":
            text = "\n".join(t for result in self.model.predict(str(path)) for t in result["rec_texts"])
            return output_text(text)
        if self.name == "rapid":
            rows, _ = self.model(str(path), use_cls=False)
            return output_text("\n".join(row[1] for row in (rows or [])))
        if self.name == "easyocr":
            rows = self.model.readtext(str(path), batch_size=self.rec_batch_size, detail=1, paragraph=False, workers=0)
            return output_text("\n".join(row[1] for row in rows))
        if self.name == "tesseract":
            text = subprocess.check_output([self.binary, str(path), "stdout", "--tessdata-dir", str(self.tessdata),
                                           "-l", "eng", "--oem", "1", "--psm", "3"], text=True, encoding="utf-8", timeout=180)
            return output_text(text.strip())
        return self.predict_batch([path])[0]

    def predict_batch(self, paths):
        if self.name != "granite":
            return [self.predict(path) for path in paths]
        from docling_core.types.doc import DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument
        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt] * len(images), images=images, padding=True, return_tensors="pt").to(self.device)
        for key, tensor in inputs.items():
            if tensor.is_floating_point():
                inputs[key] = tensor.to(self.dtype)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        answers = []
        for ids, image in zip(generated[:, inputs["input_ids"].shape[-1]:], images):
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            tokens = ids.tolist()
            stop = next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens))
            raw = self.processor.decode(ids[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            doc = DoclingDocument.load_from_doctags(DocTagsDocument.from_doctags_and_image_pairs([raw], [image]), document_name="page")
            answers.append(output_text(doc.export_to_text(), raw, stop, stop >= self.max_tokens))
        return answers
