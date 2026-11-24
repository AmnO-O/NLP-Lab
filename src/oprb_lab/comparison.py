"""Compare the five registered models on matched category-qualified pages."""
import argparse
import json
from pathlib import Path

from .models import MODELS
from .report import METRICS
from .storage import read_rows


def compare(run_dirs, output):
    import pandas as pd
    runs, manifests, configs, frames = {}, [], [], []
    for directory in map(Path, run_dirs):
        config = json.loads((directory / "config.json").read_text())
        key = config["model"]
        if key not in MODELS or key in runs:
            raise ValueError("Supply exactly one run per registered model")
        runs[key] = directory
        configs.append(config)
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = {s["uid"] for s in manifest}
        if len(expected) != len(manifest):
            raise ValueError("Duplicate manifest UIDs")
        manifests.append(expected)
        rows = {}
        for path in sorted(directory.glob("rank[0-9]*.jsonl")):
            for row in read_rows(path):
                if row["uid"] in expected:
                    rows[row["uid"]] = dict(row, model=key)
        frames.append(pd.DataFrame(rows.values(), columns=None if rows else ["uid", "status", "model"]))
    if set(runs) != set(MODELS):
        raise ValueError(f"All five models are required: {list(MODELS)}")
    if any(ids != manifests[0] for ids in manifests):
        raise ValueError("Runs use different sample manifests")
    for field in ("manifest_sha", "normalization", "gt_order", "seed", "code_sha"):
        if any(c.get(field) != configs[0].get(field) for c in configs):
            raise ValueError(f"Incompatible evaluation protocol: {field}")
    data = pd.concat(frames, ignore_index=True)
    coverage = []
    common = set(manifests[0])
    for key in MODELS:
        subset = data[data.model == key]
        successful = set(subset.loc[subset.status == "ok", "uid"])
        common &= successful
        coverage.append(dict(model=key, expected=len(manifests[0]), success=len(successful),
                             errors=int((subset.status == "error").sum()),
                             missing=len(manifests[0] - set(subset.uid))))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    if not common:
        raise ValueError("No common successful pages; inspect coverage.csv")
    paired = data[(data.status == "ok") & data.uid.isin(common)].copy()
    # Comparing GT also catches inconsistent annotations despite equal sample names.
    if paired.groupby("uid").ground_truth.nunique(dropna=False).gt(1).any():
        raise ValueError("Ground truth differs across runs")
    table = paired.groupby("model")[METRICS].mean()
    table["n_common"] = len(common)
    table["pareto"] = [not any((other.latency_sec <= row.latency_sec and other.wer <= row.wer)
                       and (other.latency_sec < row.latency_sec or other.wer < row.wer)
                       for _, other in table.iterrows()) for _, row in table.iterrows()]
    table.to_csv(output / "comparison.csv")
    (output / "comparison.tex").write_text(table.to_latex(float_format="%.4f"), encoding="utf-8")
    paired.to_csv(output / "paired_pages.csv", index=False)
    for group in ("category", "occlusion_type", "density"):
        paired.groupby(["model", group])[METRICS].mean().to_csv(output / f"by_{group}.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for key, row in table.iterrows():
        ax.scatter(row.latency_sec, row.wer)
        ax.annotate(key, (row.latency_sec, row.wer))
    ax.set(xlabel="Mean end-to-end latency (s/page)", ylabel="Macro WER", title="Matched-page comparison")
    fig.tight_layout()
    fig.savefig(output / "latency_wer.png", dpi=180)
    plt.close(fig)
    (output / "protocol.json").write_text(json.dumps(dict(
        runs={k: str(v) for k, v in runs.items()}, configs=configs,
        common_uids=sorted(common), note="Report coverage alongside paired scores. Compare latency only on equivalent hardware and GPU concurrency."), indent=2))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Exactly five fingerprinted run directories")
    parser.add_argument("--output", default="comparison")
    args = parser.parse_args()
    print(compare(args.runs, args.output))
