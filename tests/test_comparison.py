import json

import pytest

from oprb_lab.comparison import compare
from oprb_lab.metrics import score
from oprb_lab.models import MODELS
from oprb_lab.storage import rewrite_rows, resume_rows, read_rows


def make_runs(tmp_path):
    runs = []
    for key in MODELS:
        path = tmp_path / key
        path.mkdir()
        (path / "config.json").write_text(json.dumps(dict(model=key, manifest_sha="same", normalization="nfkc", seed=42, code_sha="same", gt_order="annotation")))
        (path / "manifest.json").write_text(json.dumps([dict(uid="text/a"), dict(uid="nontext/a")]))
        rows = [dict(uid=f"{cat}/a", category=cat, status="ok", ground_truth="cat", prediction="cat",
                     occlusion_type="Ink", density="2", latency_sec=1., **score("cat", "cat")) for cat in ("text", "nontext")]
        rewrite_rows(path / "rank0.jsonl", rows)
        runs.append(path)
    return runs


def test_comparison_uses_common_success_and_reports_failure(tmp_path):
    runs = make_runs(tmp_path)
    rows = read_rows(runs[0] / "rank0.jsonl")
    rows[1] = dict(uid="nontext/a", status="error", error="OOM")
    rewrite_rows(runs[0] / "rank0.jsonl", rows)
    table = compare(runs, tmp_path / "output")
    assert set(table.index) == set(MODELS)
    assert (table.n_common == 1).all()
    import pandas as pd
    coverage = pd.read_csv(tmp_path / "output/coverage.csv").set_index("model")
    assert coverage.loc["paddle", "errors"] == 1
    assert coverage.loc["paddle", "expected"] == 2


@pytest.mark.parametrize("field", ["manifest_sha", "normalization", "code_sha", "seed"])
def test_comparison_rejects_changed_protocol(tmp_path, field):
    runs = make_runs(tmp_path)
    path = runs[0] / "config.json"
    config = json.loads(path.read_text())
    config[field] = "changed"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="protocol"):
        compare(runs, tmp_path / "output")


def test_comparison_requires_all_five(tmp_path):
    with pytest.raises(ValueError, match="All five"):
        compare(make_runs(tmp_path)[:4], tmp_path / "output")


def test_comparison_rejects_inconsistent_annotations(tmp_path):
    runs = make_runs(tmp_path)
    path = runs[0] / "rank0.jsonl"
    rows = read_rows(path)
    rows[0]["ground_truth"] = "different"
    rewrite_rows(path, rows)
    with pytest.raises(ValueError, match="Ground truth"):
        compare(runs, tmp_path / "output")


def test_comparison_no_common_success_writes_coverage(tmp_path):
    runs = make_runs(tmp_path)
    rewrite_rows(runs[0] / "rank0.jsonl", [])
    with pytest.raises(ValueError, match="No common"):
        compare(runs, tmp_path / "output")
    assert (tmp_path / "output/coverage.csv").exists()


def test_resume_retries_errors_and_qualifies_category(tmp_path):
    path = tmp_path / "rank0.jsonl"
    selected = [dict(uid="text/a"), dict(uid="nontext/a")]
    rewrite_rows(path, [dict(uid="text/a", status="ok"), dict(uid="nontext/a", status="error"), dict(uid="other", status="ok")])
    assert resume_rows(path, selected) == [dict(uid="nontext/a")]
    assert read_rows(path) == [dict(uid="text/a", status="ok")]
