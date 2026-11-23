import json
from pathlib import Path

from .storage import read_rows

METRICS = ["cer", "wer", "token_precision", "token_recall", "token_f1", "latency_sec"]


def summarize(run_dir):
    import pandas as pd
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {s["uid"] for s in manifest}
    rows = {}
    for path in sorted(root.glob("rank[0-9]*.jsonl")):
        for row in read_rows(path):
            if row["uid"] in expected:
                rows[row["uid"]] = row
    df = pd.DataFrame(rows.values()) if rows else pd.DataFrame(columns=["uid", "status"])
    df.to_csv(root / "per_image_results.csv", index=False)
    ok = df.loc[df.status == "ok"]
    summary = dict(expected=len(expected), observed=len(df), success=len(ok),
                   errors=int((df.status == "error").sum()), missing=len(expected - rows.keys()),
                   complete=len(ok) == len(expected), aggregation="macro mean over successful pages")
    if len(ok):
        summary.update({m: float(ok[m].mean()) for m in METRICS})
        summary["possibly_truncated"] = int(ok.possibly_truncated.fillna(False).sum())
        summary["truncation_unknown"] = int(ok.possibly_truncated.isna().sum())
        for groups, name in [(["category"], "category"), (["category", "occlusion_type"], "type"),
                             (["category", "density"], "density")]:
            ok.groupby(groups, dropna=False).agg(n=("uid", "count"), **{m: (m, "mean") for m in METRICS}).reset_index().to_csv(
                root / f"summary_by_{name}.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = ok.groupby("category")[["cer", "wer"]].mean().plot.bar(rot=0)
        ax.set_ylabel("Macro error rate (lower is better)")
        plt.tight_layout()
        plt.savefig(root / "error_rates.png", dpi=160)
        plt.close()
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
