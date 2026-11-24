"""Download only occluded test images and scoring annotations."""
import random
import hashlib
import shutil
import tarfile
from pathlib import Path

EXPECTED = {"text": 6000, "nontext": 3000}


def extract_flat(archive, destination, suffix):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for item in tar:
            if not item.isfile() or not item.name.lower().endswith(suffix):
                continue
            name = item.name.replace("\\", "/").split("/")[-1]
            if name in seen:
                raise ValueError(f"Duplicate archive basename: {name}")
            seen.add(name)
            with tar.extractfile(item) as source, (destination / name).open("wb") as target:
                shutil.copyfileobj(source, target)


def download(root, revision="main"):
    from huggingface_hub import hf_hub_download, dataset_info
    revision = dataset_info("kpurkayastha/OPRB", revision=revision).sha
    for category in EXPECTED:
        for kind, suffix in (("img", ".jpg"), ("gt", ".txt")):
            destination = Path(root) / category / kind
            marker = destination / ".complete"
            if marker.exists() and marker.read_text() == revision and len(list(destination.glob(f"*{suffix}"))) == EXPECTED[category]:
                continue
            archive = hf_hub_download("kpurkayastha/OPRB", f"test/{category}/{kind}.tar.gz", repo_type="dataset", revision=revision)
            extract_flat(archive, destination, suffix)
            count = len(list(destination.glob(f"*{suffix}")))
            if count != EXPECTED[category]:
                raise ValueError(f"{category}/{kind}: expected {EXPECTED[category]}, got {count}")
            marker.write_text(revision)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(root, limit=None, seed=42):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    samples = []
    for category in EXPECTED:
        images = {p.stem: p for p in (Path(root) / category / "img").glob("*.jpg")}
        labels = {p.stem: p for p in (Path(root) / category / "gt").glob("*.txt")}
        if not images or images.keys() != labels.keys():
            raise ValueError(f"Missing or unmatched images/GT in {category}")
        if limit is None and len(images) != EXPECTED[category]:
            raise ValueError(f"Full test requires {EXPECTED[category]} pairs in {category}; got {len(images)}")
        keys = sorted(images)
        if limit is not None:
            keys = sorted(random.Random(seed).sample(keys, min(limit, len(keys))))
        for key in keys:
            parts = key.split("__", 2)
            samples.append(dict(uid=f"{category}/{key}", sample_id=key, category=category,
                                image_path=str(images[key].resolve()), gt_path=str(labels[key].resolve()),
                                image_sha=file_sha(images[key]), gt_sha=file_sha(labels[key]),
                                occlusion_type=parts[0], density=parts[1] if len(parts) > 1 else "unknown"))
    return samples


def read_gt(path):
    # Keep annotation order: this policy must be disclosed for multi-column pages.
    tokens = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) >= 11 and fields[0].strip():
            tokens.append(fields[0].strip())
    if not tokens:
        raise ValueError(f"No GT tokens: {path}")
    return " ".join(tokens)


def shard(samples, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid rank/world_size")
    return samples[rank::world_size]
