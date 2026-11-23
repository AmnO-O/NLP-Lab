import re
import unicodedata
from collections import Counter


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def score(reference, prediction):
    from rapidfuzz.distance import Levenshtein
    ref, hyp = normalize(reference), normalize(prediction)
    if not ref:
        raise ValueError("Empty reference")
    a, b = Counter(ref.split()), Counter(hyp.split())
    matched = sum((a & b).values())
    precision = matched / sum(b.values()) if b else 0.0
    recall = matched / sum(a.values())
    return dict(cer=Levenshtein.distance(ref, hyp) / len(ref),
                wer=Levenshtein.distance(ref.split(), hyp.split()) / len(ref.split()),
                token_precision=precision, token_recall=recall,
                token_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
