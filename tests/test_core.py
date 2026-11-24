import io
import json
import tarfile

import pytest
from PIL import Image

from oprb_lab.data import collect, extract_flat, read_gt, shard
from oprb_lab.metrics import normalize, score
from oprb_lab.models import intern_tiles
from oprb_lab.storage import fingerprint, read_rows, rewrite_rows


def test_metrics_duplicates_and_insertions():
    result = score("cat cat dog", "cat dog dog dog")
    assert result["token_precision"] == .5
    assert result["token_recall"] == pytest.approx(2/3)
    assert score("a", "a b c")["wer"] == 2
    assert score("abc", "")["cer"] == 1
    assert score("abc", "abc")["cer"] == 0
    assert score("The cat sat", "The cat")["wer"] == pytest.approx(1/3)
    with pytest.raises(ValueError):
        score("", "abc")


def test_normalization():
    assert normalize("  Ａ\nB\t c ") == "a b c"


def test_shards_are_disjoint_complete():
    samples = list(range(13))
    shards = [shard(samples, i, 2) for i in range(2)]
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == samples
    with pytest.raises(ValueError):
        shard(samples, 2, 2)


def test_dataset_pairs_gt_and_seed(tmp_path):
    for category in ("text", "nontext"):
        for kind in ("img", "gt"):
            (tmp_path / category / kind).mkdir(parents=True)
        for i in range(5):
            stem = f"Black_Ink__2p0pct__{i}"
            Image.new("RGB", (10, 10)).save(tmp_path / category / "img" / f"{stem}.jpg")
            (tmp_path / category / "gt" / f"{stem}.txt").write_text("word\t" + "\t".join(["0"] * 10))
    samples = collect(tmp_path, 2, 42)
    assert len(samples) == 4
    assert samples == collect(tmp_path, 2, 42)
    assert len({s["uid"] for s in samples}) == 4
    assert samples[0]["density"] == "2p0pct"
    assert read_gt(samples[0]["gt_path"]) == "word"
    original = samples[0]
    from pathlib import Path
    Path(original["gt_path"]).write_text("changed\t" + "\t".join(["0"] * 10))
    updated = next(s for s in collect(tmp_path, 2, 42) if s["uid"] == original["uid"])
    assert updated["gt_sha"] != original["gt_sha"]
    assert updated["image_sha"] == original["image_sha"]
    with pytest.raises(ValueError, match="Full test requires"):
        collect(tmp_path)
    (tmp_path / "text" / "gt" / "extra.txt").write_text("")
    with pytest.raises(ValueError):
        collect(tmp_path)


def test_archive_paths_flattened_and_filtered(tmp_path):
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("../../escape.txt", "folder/keep.txt", "ignore.jpg"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
    dest = tmp_path / "out"
    extract_flat(archive, dest, ".txt")
    assert sorted(p.name for p in dest.iterdir()) == ["escape.txt", "keep.txt"]
    assert not (tmp_path / "escape.txt").exists()


def test_archive_duplicate_rejected(tmp_path):
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("a/x.txt", "b/x.txt"):
            info = tarfile.TarInfo(name)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"a"))
    with pytest.raises(ValueError):
        extract_flat(archive, tmp_path / "out", ".txt")


def test_resume_recovers_partial_last_line(tmp_path):
    path = tmp_path / "rank0.jsonl"
    path.write_text('{"uid":"a","status":"ok"}\n{"uid":')
    rows = read_rows(path)
    assert len(rows) == 1
    rewrite_rows(path, rows)
    assert json.loads(path.read_text())["uid"] == "a"
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_tiles_respect_budget_and_keep_thumbnail():
    image = Image.new("RGB", (1200, 1800))
    tiles = intern_tiles(image, max_tiles=6)
    assert 1 <= len(tiles) <= 7
    assert all(t.size == (448, 448) for t in tiles)


def test_report_marks_incomplete(tmp_path):
    from oprb_lab.report import summarize
    (tmp_path / "manifest.json").write_text(json.dumps([{"uid": "a"}, {"uid": "b"}]))
    row = dict(uid="a", status="ok", category="text", occlusion_type="Ink", density="2",
               possibly_truncated=None, latency_sec=1, **score("cat", "cat"))
    rewrite_rows(tmp_path / "rank0.jsonl", [row])
    result = summarize(tmp_path)
    assert result["missing"] == 1
    assert not result["complete"]
    assert result["truncation_unknown"] == 1
    assert (tmp_path / "per_image_results.csv").exists()


def test_report_when_all_workers_fail_loading(tmp_path):
    from oprb_lab.report import summarize
    (tmp_path / "manifest.json").write_text(json.dumps([{"uid": "a"}]))
    result = summarize(tmp_path)
    assert result["missing"] == 1
    assert result["success"] == 0
    assert not result["complete"]
    assert (tmp_path / "summary.json").exists()


def test_report_counts_failed_pages(tmp_path):
    from oprb_lab.report import summarize
    (tmp_path / "manifest.json").write_text(json.dumps([{"uid": "a"}]))
    rewrite_rows(tmp_path / "rank0.jsonl", [dict(uid="a", status="error", error="OOM")])
    result = summarize(tmp_path)
    assert result["errors"] == 1
    assert result["missing"] == 0
    assert not result["complete"]
