import hashlib
import json
from pathlib import Path


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # Interrupted append: caller rewrites valid records before resuming.
    return rows


def rewrite_rows(path, rows):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def resume_rows(path, selected):
    """Keep only successful rows for this shard; errors remain eligible for retry."""
    ids = {sample["uid"] for sample in selected}
    completed = {row["uid"]: row for row in read_rows(path)
                 if row.get("status") == "ok" and row["uid"] in ids}
    rewrite_rows(path, list(completed.values()))
    return [sample for sample in selected if sample["uid"] not in completed]
