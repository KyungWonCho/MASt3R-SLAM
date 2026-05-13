"""Aggregate cached eval_geometry.py results into a paste-friendly summary.

Walks logs/<dataset_root>/<variant>/<split>/<scene>/eval.json and prints
one markdown table per metric (rows = scene, cols = variant), plus a
per-variant overall row.

Usage:
  python scripts/summary_table.py --root logs/7-scenes
  python scripts/summary_table.py --root logs/7-scenes --split calib
  python scripts/summary_table.py --root logs/7-scenes --out logs/results/summary.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    ("ate", "ATE RMSE (m)"),
    ("accuracy_rmse", "Accuracy RMSE (m)"),
    ("completion_rmse", "Completion RMSE (m)"),
    ("chamfer_rmse", "Chamfer RMSE (m)"),
    ("fps", "FPS"),
]


def collect(root, split, cache_name):
    """Return dict[variant][scene] -> metrics dict."""
    by = defaultdict(dict)
    for eval_path in Path(root).rglob(cache_name):
        # path = logs/7-scenes/<variant>/<split>/<scene>/eval.json
        try:
            rel = eval_path.relative_to(root)
            parts = rel.parts
            if len(parts) < 4:
                continue
            variant, this_split, scene, _ = parts[0], parts[1], parts[2], parts[3]
            if split is not None and this_split != split:
                continue
            with open(eval_path) as f:
                d = json.load(f)
            entry = dict(d.get("metrics", {}))
            for k in ("fps", "frames", "keyframes", "total_time_s"):
                if k in d:
                    entry[k] = d[k]
            by[variant][scene] = entry
        except Exception as e:
            print(f"[skip] {eval_path}: {e}")
    return by


def fmt_cell(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return format(v, fmt)


def best_in_row(values, lower_is_better=True):
    """Indices of the best value(s) for highlighting."""
    nums = [(i, v) for i, v in enumerate(values)
            if isinstance(v, (int, float)) and np.isfinite(v)]
    if not nums:
        return set()
    if lower_is_better:
        best = min(v for _, v in nums)
    else:
        best = max(v for _, v in nums)
    return {i for i, v in nums if abs(v - best) < 1e-9}


def render(by, metric, label, lower_is_better, fmt=".4f"):
    variants = sorted(by.keys())
    scenes = sorted({s for v in variants for s in by[v]})
    lines = [f"\n## {label}{' (lower is better)' if lower_is_better else ''}\n"]
    header = "| scene | " + " | ".join(variants) + " |"
    sep = "|" + "---|" + ":---:|" * len(variants)
    lines.append(header)
    lines.append(sep)
    # Per-scene rows
    for scene in scenes:
        row_vals = [by[v].get(scene, {}).get(metric) for v in variants]
        best = best_in_row(row_vals, lower_is_better)
        cells = []
        for i, v in enumerate(row_vals):
            c = fmt_cell(v, fmt)
            if i in best and v is not None:
                c = f"**{c}**"
            cells.append(c)
        lines.append(f"| {scene} | " + " | ".join(cells) + " |")
    # Overall mean row
    mean_vals = []
    for v in variants:
        xs = [by[v].get(s, {}).get(metric) for s in scenes]
        xs = [x for x in xs
              if isinstance(x, (int, float)) and np.isfinite(x)]
        mean_vals.append(float(np.mean(xs)) if xs else None)
    best = best_in_row(mean_vals, lower_is_better)
    cells = []
    for i, v in enumerate(mean_vals):
        c = fmt_cell(v, fmt)
        if i in best and v is not None:
            c = f"**{c}**"
        cells.append(c)
    lines.append(f"| **mean** | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="logs/7-scenes")
    p.add_argument("--split", default="calib")
    p.add_argument("--cache-name", default="eval.json")
    p.add_argument("--out", default="")
    args = p.parse_args()

    by = collect(args.root, args.split, args.cache_name)
    if not by:
        print(f"no cached eval.json found under {args.root}/<variant>/{args.split}/")
        return

    out = [f"# Eval summary — {args.root} (split={args.split or 'all'})\n"]
    out.append(f"Variants: {sorted(by.keys())}\n")
    out.append(f"Scenes: {sorted({s for v in by.values() for s in v})}\n")
    for key, label in METRICS:
        lower = (key != "fps")
        fmt = ".2f" if key == "fps" else ".4f"
        out.append(render(by, key, label, lower_is_better=lower, fmt=fmt))

    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
