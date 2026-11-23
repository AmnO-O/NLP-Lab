"""Image-only adapters. Ground truth is never provided to a model."""
from PIL import Image

MODELS = {
    "paddle": "PP-OCRv5_mobile_det+PP-OCRv5_mobile_rec",
    "granite": "ibm-granite/granite-docling-258M",
    "got": "stepfun-ai/GOT-OCR-2.0-hf",
    "internvl": "OpenGVLab/InternVL2_5-1B",
    "qwen": "Qwen/Qwen3-VL-2B-Instruct",
}
PROMPT = ("Transcribe all text visible on this document page in natural reading order. "
          "Return only the transcription. Do not explain, describe, or invent unreadable text.")


def intern_tiles(image, max_tiles=12, size=448):
    """Dynamic grid + thumbnail, adapted from the official InternVL model card."""
    w, h = image.size
    ratios = sorted({(i, j) for i in range(1, max_tiles + 1)
                     for j in range(1, max_tiles + 1) if i * j <= max_tiles},
                    key=lambda p: (p[0] * p[1], p[0], p[1]))
    best, distance = (1, 1), float("inf")
    for ratio in ratios:
        diff = abs(w / h - ratio[0] / ratio[1])
        if diff < distance or (diff == distance and w * h > .5 * size * size * ratio[0] * ratio[1]):
            best, distance = ratio, diff
    cols, rows = best
    resized = image.resize((cols * size, rows * size))
    tiles = [resized.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
             for y in range(rows) for x in range(cols)]
    if len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    return tiles


class Adapter:
    def __init__(self, name, revision, max_tokens, max_pixels, max_tiles):
        import torch
        self.name, self.max_tokens, self.max_tiles = name, max_tokens, max_tiles
        self.torch = torch
        self.device = "cuda:0"  # Worker sees only its assigned physical GPU.
        self.dtype = torch.float16
        self.parameter_count = None
        if name == "paddle":
            from paddleocr import PaddleOCR
            self.model = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, device="gpu:0")
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model_id = MODELS[name]
        options = dict(revision=revision, torch_dtype=self.dtype, low_cpu_mem_usage=True)
        if name == "internvl":
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                                  use_flash_attn=False, **options).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                           trust_remote_code=True, use_fast=False)
        else:
            processor_options = {"max_pixels": max_pixels} if name == "qwen" else {}
            self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, **processor_options)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, attn_implementation="sdpa", **options).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def predict(self, path):
        if self.name == "paddle":
            results = list(self.model.predict(str(path)))
            texts = []
            for result in results:
                # PaddleOCR Results expose the raw prediction as a dictionary.
                texts.extend(result["rec_texts"])
            return dict(prediction="\n".join(texts), raw_prediction="\n".join(texts),
                        generated_tokens=None, possibly_truncated=False)
        with Image.open(path) as source:
            image = source.convert("RGB")
        with self.torch.inference_mode():
            if self.name == "internvl":
                from torchvision.transforms import Compose, ToTensor, Normalize
                transform = Compose([ToTensor(), Normalize((.485, .456, .406), (.229, .224, .225))])
                pixels = self.torch.stack([transform(tile) for tile in intern_tiles(image, self.max_tiles)])
                pixels = pixels.to(device=self.device, dtype=self.dtype)
                raw = self.model.chat(self.tokenizer, pixels, "<image>\n" + PROMPT,
                                      dict(max_new_tokens=self.max_tokens, do_sample=False), history=None)
                # chat returns decoded text, not raw generated token IDs.
                return dict(prediction=raw, raw_prediction=raw, generated_tokens=None,
                            possibly_truncated=None)
            if self.name == "got":
                inputs = self.processor(image, return_tensors="pt", format=False)
            else:
                text = "Convert this page to docling." if self.name == "granite" else PROMPT
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)
            for key, tensor in inputs.items():
                if tensor.is_floating_point():
                    inputs[key] = tensor.to(self.dtype)
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
            tokens = output[0, inputs["input_ids"].shape[-1]:]
            raw = self.processor.decode(tokens, skip_special_tokens=self.name != "granite",
                                        clean_up_tokenization_spaces=False)
            if self.name == "granite":
                from docling_core.types.doc import DoclingDocument
                from docling_core.types.doc.document import DocTagsDocument
                doc = DoclingDocument.load_from_doctags(
                    DocTagsDocument.from_doctags_and_image_pairs([raw], [image]), document_name="page")
                prediction = doc.export_to_text()
            else:
                prediction = raw.strip()
            return dict(prediction=prediction, raw_prediction=raw, generated_tokens=len(tokens),
                        possibly_truncated=len(tokens) >= self.max_tokens)
