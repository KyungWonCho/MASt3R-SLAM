"""Analyze diag NPZs collected by --diag-dir runs.

Reads tracking.npz and (if present) loop.npz from a directory tree of shape

  <diag_root>/<scene>/<mode>/{tracking,loop}.npz

Outputs (in <out_dir>):
  summary.md          one human-readable file with every table — paste-friendly
  summary.csv         per (scene, mode, role) stats + calibration fits (legacy)
  per_pair.csv        per-pair table across all scenes (one row per recorded pair)
  *.png               per-scene + cross-scene plots
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------- #
# NPZ loading + per-pair extraction
# ---------------------------------------------------------------------- #

def load_npz(path):
    if not path.exists():
        return None
    d = dict(np.load(path, allow_pickle=True))
    n = int(d["valid_match_len"])
    d["valid_match"] = np.unpackbits(d["valid_match_packed"])[:n].astype(bool)
    return d


def per_pair_rows(d, scene, mode, role):
    """One row per recorded pair: pair-level metadata + per-pair err/conf stats."""
    err_all = d["err"]
    conf_all = d["conf"]
    offs = d["pair_offset"]
    n_pair = len(d["n_pixels"])
    rows = []
    for i in range(n_pair):
        a, b = int(offs[i]), int(offs[i + 1])
        e = err_all[a:b]
        c = conf_all[a:b]
        finite = np.isfinite(e)
        n_f = int(finite.sum())
        if n_f == 0:
            err_med = err_p90 = err_mean = conf_med = float("nan")
        else:
            ef = e[finite].astype(np.float32)
            cf = c[finite].astype(np.float32)
            err_med = float(np.median(ef))
            err_p90 = float(np.percentile(ef, 90))
            err_mean = float(ef.mean())
            conf_med = float(np.median(cf))
        rows.append({
            "scene": scene,
            "mode": mode,
            "role": role,
            "pair_idx": i,
            "kind": str(d["kind"][i]) if d["kind"].size else role,
            "baseline": float(d["baseline"][i]),
            "view_angle_deg": float(d["view_angle_deg"][i]),
            "frame_diff": int(d["frame_diff"][i]),
            "is_consecutive": int(d["is_consecutive"][i]),
            "n_pixels_finite": n_f,
            "err_median": err_med,
            "err_p90": err_p90,
            "err_mean": err_mean,
            "conf_median": conf_med,
        })
    return rows


# ---------------------------------------------------------------------- #
# σ(c) calibration fit (online histogram)
# ---------------------------------------------------------------------- #

def calibration_from_hist(hist, edges):
    """hist (NBIN, 3) [count, sum_err, sum_err2] -> (c_mid, count, mean, rmse, std)."""
    count = hist[:, 0]
    s = hist[:, 1]
    s2 = hist[:, 2]
    valid = count > 5
    c_mid = np.sqrt(edges[:-1] * edges[1:])
    mean = np.where(valid, s / np.maximum(count, 1), np.nan)
    rmse = np.where(valid, np.sqrt(s2 / np.maximum(count, 1)), np.nan)
    var = np.where(valid, np.maximum(s2 / np.maximum(count, 1) - mean ** 2, 0.0), np.nan)
    std = np.sqrt(var)
    return c_mid, count, mean, rmse, std


def fit_powerlaw_sigma_c(c_mid, sigma):
    m = np.isfinite(sigma) & (sigma > 0) & np.isfinite(c_mid) & (c_mid > 0)
    if m.sum() < 5:
        return float("nan"), float("nan")
    x = np.log(c_mid[m])
    y = np.log(sigma[m])
    A = np.vstack([x, np.ones_like(x)]).T
    p, log_a = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.exp(log_a)), float(p)


def overall_rmse_from_hist(hist):
    total = hist.sum(axis=0)
    if total[0] <= 0:
        return float("nan")
    return float(np.sqrt(total[2] / total[0]))


# ---------------------------------------------------------------------- #
# Binned stats over per-pair table
# ---------------------------------------------------------------------- #

def binned_table(rows, x_key, x_bins, role=None, scenes=None):
    """For each [bin_lo, bin_hi): n_pairs, err_median(of pair medians),
       err_p90(of pair p90s, then median), conf_median, view/baseline/frame_diff mean."""
    sel = rows
    if role is not None:
        sel = [r for r in sel if r["role"] == role]
    if scenes is not None:
        sel = [r for r in sel if r["scene"] in scenes]
    sel = [r for r in sel
           if np.isfinite(r["err_median"]) and np.isfinite(r[x_key])]

    out = []
    for lo, hi in zip(x_bins[:-1], x_bins[1:]):
        in_bin = [r for r in sel if lo <= r[x_key] < hi]
        if not in_bin:
            out.append({"lo": lo, "hi": hi, "n": 0,
                        "err_med": float("nan"), "err_p90": float("nan"),
                        "conf_med": float("nan"), "x_mean": float("nan")})
            continue
        em = np.array([r["err_median"] for r in in_bin])
        e9 = np.array([r["err_p90"] for r in in_bin])
        cm = np.array([r["conf_median"] for r in in_bin])
        xv = np.array([r[x_key] for r in in_bin])
        out.append({
            "lo": lo, "hi": hi, "n": len(in_bin),
            "err_med": float(np.median(em)),
            "err_p90": float(np.median(e9)),
            "conf_med": float(np.median(cm)),
            "x_mean": float(xv.mean()),
        })
    return out


def fmt_bin(lo, hi):
    return f"[{lo:.3g}, {hi:.3g})"


# ---------------------------------------------------------------------- #
# Plots
# ---------------------------------------------------------------------- #

def plot_sigma_vs_conf(c_mid, rmse, std, count, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.loglog(c_mid, rmse, "o-", label=r"RMSE  $\sqrt{E[r^2|c]}$", color="tab:blue")
    ax.loglog(c_mid, std, "s--", label=r"std  $\sqrt{Var[r|c]}$", color="tab:orange")
    ax.set_xlabel("confidence c (log)")
    ax.set_ylabel("error (m, log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax2 = ax.twinx()
    ax2.semilogx(c_mid, count, "k:", alpha=0.4)
    ax2.set_ylabel("bin count (dotted)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_pair_scatter(rows, x_key, role, out_path, title, xscale="linear"):
    sel = [r for r in rows
           if r["role"] == role
           and np.isfinite(r["err_median"]) and np.isfinite(r[x_key])]
    if not sel:
        return
    scenes = sorted({r["scene"] for r in sel})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for i, s in enumerate(scenes):
        ss = [r for r in sel if r["scene"] == s]
        x = np.array([r[x_key] for r in ss])
        y = np.array([r["err_median"] for r in ss])
        ax.scatter(x, y, s=12, alpha=0.5, color=cmap(i % 10), label=f"{s} ({len(ss)})")
    if xscale == "log":
        ax.set_xscale("symlog", linthresh=max(1e-3, np.min([r[x_key] for r in sel if r[x_key] > 0] or [1e-3])))
    ax.set_xlabel(x_key)
    ax.set_ylabel("per-pair median err (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_view_angle_hist(rows, role, out_path, title):
    sel = [r["view_angle_deg"] for r in rows
           if r["role"] == role and np.isfinite(r["view_angle_deg"])]
    if not sel:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(sel, bins=40)
    ax.set_xlabel("view_angle (deg)")
    ax.set_ylabel("n_pairs")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Markdown helpers
# ---------------------------------------------------------------------- #

def md_table(headers, rows, aligns=None):
    """Format a markdown table; rows is list of tuples/lists of strings."""
    if aligns is None:
        aligns = ["---"] * len(headers)
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(aligns) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag-root", default="logs/diag")
    p.add_argument("--out-dir", default="logs/diag/plots")
    args = p.parse_args()

    root = Path(args.diag_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load all NPZs, build per-scene-mode-role summary and per-pair table.
    per_scene_rows = []
    per_pair = []
    hist_agg = {"tracking": [], "loop": []}  # for cross-scene σ(c)
    hist_edges_ref = None

    scenes = sorted([p.name for p in root.iterdir()
                     if p.is_dir() and p.name != "plots"])
    print(f"Found scenes: {scenes}")

    for scene in scenes:
        for mode in ("calib", "no_calib"):
            scene_dir = root / scene / mode
            if not scene_dir.exists():
                continue
            for role in ("tracking", "loop"):
                npz_path = scene_dir / f"{role}.npz"
                d = load_npz(npz_path)
                if d is None:
                    continue

                # Per-scene summary stats (legacy summary.csv)
                err = d["err"]
                conf = d["conf"]
                finite = np.isfinite(err) & np.isfinite(conf)
                err_f = err[finite].astype(np.float32)
                conf_f = conf[finite].astype(np.float32)
                stats = {
                    "scene": scene, "mode": mode, "role": role,
                    "n_pairs": int(len(d["n_pixels"])),
                    "n_pixels_total": int(err.size),
                    "n_pixels_finite": int(finite.sum()),
                }
                if err_f.size:
                    stats.update({
                        "err_mean": float(err_f.mean()),
                        "err_median": float(np.median(err_f)),
                        "err_p90": float(np.percentile(err_f, 90)),
                        "err_rmse": float(np.sqrt((err_f ** 2).mean())),
                        "conf_mean": float(conf_f.mean()),
                        "conf_median": float(np.median(conf_f)),
                    })
                else:
                    for k in ("err_mean", "err_median", "err_p90",
                              "err_rmse", "conf_mean", "conf_median"):
                        stats[k] = float("nan")
                for k in ("baseline", "view_angle_deg", "frame_diff"):
                    if k in d and d[k].size:
                        stats[f"{k}_mean"] = float(np.nanmean(d[k]))
                        stats[f"{k}_max"] = float(np.nanmax(d[k]))

                # σ(c) fit
                hist = d.get("hist")
                edges = d.get("hist_edges")
                if hist is not None and edges is not None and hist.size > 0:
                    total_hist = hist.sum(axis=0)
                    c_mid, count, mean, rmse, std = calibration_from_hist(total_hist, edges)
                    a, pe = fit_powerlaw_sigma_c(c_mid, rmse)
                    stats["sigma_rmse_a"] = a
                    stats["sigma_rmse_p"] = pe
                    plot_sigma_vs_conf(
                        c_mid, rmse, std, count,
                        f"{scene}/{mode}/{role}   σ≈{a:.3g}·c^{pe:.2f}",
                        out_dir / f"{scene}_{mode}_{role}_sigma_vs_conf.png",
                    )
                    # Aggregate (only calib to avoid duplicating ~identical no_calib)
                    if mode == "calib":
                        hist_agg[role].append(hist)
                        if hist_edges_ref is None:
                            hist_edges_ref = edges
                else:
                    stats["sigma_rmse_a"] = stats["sigma_rmse_p"] = float("nan")
                per_scene_rows.append(stats)

                # Per-pair table (only calib — no_calib is duplicate-ish)
                if mode == "calib":
                    per_pair.extend(per_pair_rows(d, scene, mode, role))

                print(f"[{scene}/{mode}/{role}]  pairs={stats['n_pairs']}  "
                      f"err_rmse={stats['err_rmse']:.4f}  "
                      f"σ≈{stats['sigma_rmse_a']:.3g}·c^{stats['sigma_rmse_p']:.2f}")

    # --- 2. Cross-scene aggregate σ(c) for tracking + loop (calib only)
    agg_sigma = {}
    for role in ("tracking", "loop"):
        if not hist_agg[role] or hist_edges_ref is None:
            continue
        H = np.concatenate(hist_agg[role], axis=0)  # (n_pair_total, NBIN, 3)
        total = H.sum(axis=0)
        c_mid, count, mean, rmse, std = calibration_from_hist(total, hist_edges_ref)
        a, pe = fit_powerlaw_sigma_c(c_mid, rmse)
        agg_sigma[role] = {
            "n_pair_total": int(H.shape[0]),
            "n_obs": int(total[:, 0].sum()),
            "a": a, "p": pe,
            "err_rmse_overall": overall_rmse_from_hist(total),
        }
        plot_sigma_vs_conf(
            c_mid, rmse, std, count,
            f"AGGREGATE {role} (calib, all scenes)   σ≈{a:.3g}·c^{pe:.2f}",
            out_dir / f"aggregate_{role}_sigma_vs_conf.png",
        )

    # --- 3. Per-pair scatter plots, cross-scene
    plot_pair_scatter(per_pair, "view_angle_deg", "tracking",
                      out_dir / "tracking_err_vs_view_angle.png",
                      "Tracking: per-pair median err vs view angle")
    plot_pair_scatter(per_pair, "baseline", "tracking",
                      out_dir / "tracking_err_vs_baseline.png",
                      "Tracking: per-pair median err vs baseline")
    plot_pair_scatter(per_pair, "frame_diff", "tracking",
                      out_dir / "tracking_err_vs_frame_diff.png",
                      "Tracking: per-pair median err vs frame diff",
                      xscale="log")
    plot_pair_scatter(per_pair, "view_angle_deg", "loop",
                      out_dir / "loop_err_vs_view_angle.png",
                      "Loop: per-pair median err vs view angle")
    plot_pair_scatter(per_pair, "baseline", "loop",
                      out_dir / "loop_err_vs_baseline.png",
                      "Loop: per-pair median err vs baseline")

    # --- 4. View angle histograms (sanity for user's assumption)
    plot_view_angle_hist(per_pair, "tracking",
                        out_dir / "tracking_view_angle_hist.png",
                        "Tracking pairs — view_angle distribution")
    plot_view_angle_hist(per_pair, "loop",
                        out_dir / "loop_view_angle_hist.png",
                        "Loop pairs — view_angle distribution")

    # --- 5. Write per_pair.csv
    pp_csv = out_dir / "per_pair.csv"
    if per_pair:
        keys = list(per_pair[0].keys())
        with open(pp_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_pair:
                w.writerow(r)

    # --- 6. Legacy summary.csv
    if per_scene_rows:
        csv_path = out_dir / "summary.csv"
        keys = sorted({k for r in per_scene_rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_scene_rows:
                w.writerow(r)

    # --- 7. Build summary.md with everything
    md = ["# Diag analysis summary\n"]
    md.append("All tables here are paste-friendly markdown. Plots are in the same directory.\n")

    # 7a. Per-scene basic table
    md.append("\n## 1. Per-scene calibration\n")
    md.append("σ ≈ a · c^p  fit on RMSE per conf bin (log-spaced). "
              "|p|=1 ⇒ σ ∝ 1/c, |p|=0 ⇒ confidence carries no info.\n")
    headers = ["scene", "mode", "role", "pairs", "pixels",
               "err_rmse", "err_median", "conf_mean", "a", "p"]
    rows = []
    for r in per_scene_rows:
        rows.append([
            r["scene"], r["mode"], r["role"], r["n_pairs"],
            f"{r['n_pixels_finite']:,}",
            f"{r['err_rmse']:.4f}", f"{r['err_median']:.4f}",
            f"{r['conf_mean']:.2f}",
            f"{r.get('sigma_rmse_a', float('nan')):.3g}",
            f"{r.get('sigma_rmse_p', float('nan')):.3f}",
        ])
    md.append(md_table(headers, rows))

    # 7b. Cross-scene aggregate σ(c)
    md.append("\n## 2. Cross-scene aggregate σ(c) (calib only)\n")
    md.append("All scenes' pairs combined, then σ fit.\n")
    rows = []
    for role, s in agg_sigma.items():
        rows.append([
            role, s["n_pair_total"], f"{s['n_obs']:,}",
            f"{s['err_rmse_overall']:.4f}",
            f"{s['a']:.3g}", f"{s['p']:.3f}",
        ])
    md.append(md_table(
        ["role", "n_pairs", "n_pixel_obs", "err_rmse", "a", "p"], rows))

    # 7c. Tracking — binned by view_angle, baseline, frame_diff
    view_bins = [0, 2, 5, 10, 20, 30, 45, 60, 90]
    base_bins = [0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    fdiff_bins = [0, 2, 5, 10, 30, 100, 300]

    md.append("\n## 3. Tracking — err binned by view angle (all scenes, calib)\n")
    bt = binned_table(per_pair, "view_angle_deg", view_bins, role="tracking")
    rows = [[fmt_bin(b["lo"], b["hi"]), b["n"],
             f"{b['err_med']:.4f}", f"{b['err_p90']:.4f}",
             f"{b['conf_med']:.2f}"] for b in bt]
    md.append(md_table(
        ["view_angle [deg]", "n_pairs", "err_median", "err_p90", "conf_median"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:"]))

    md.append("\n## 4. Tracking — err binned by baseline\n")
    bt = binned_table(per_pair, "baseline", base_bins, role="tracking")
    rows = [[fmt_bin(b["lo"], b["hi"]), b["n"],
             f"{b['err_med']:.4f}", f"{b['err_p90']:.4f}",
             f"{b['conf_med']:.2f}"] for b in bt]
    md.append(md_table(
        ["baseline [m]", "n_pairs", "err_median", "err_p90", "conf_median"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:"]))

    md.append("\n## 5. Tracking — err binned by frame diff\n")
    bt = binned_table(per_pair, "frame_diff", fdiff_bins, role="tracking")
    rows = [[fmt_bin(b["lo"], b["hi"]), b["n"],
             f"{b['err_med']:.4f}", f"{b['err_p90']:.4f}",
             f"{b['conf_med']:.2f}"] for b in bt]
    md.append(md_table(
        ["frame_diff", "n_pairs", "err_median", "err_p90", "conf_median"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:"]))

    # 7d. Loop — same binnings
    md.append("\n## 6. Loop — err binned by view angle\n")
    bt = binned_table(per_pair, "view_angle_deg", view_bins, role="loop")
    rows = [[fmt_bin(b["lo"], b["hi"]), b["n"],
             f"{b['err_med']:.4f}", f"{b['err_p90']:.4f}",
             f"{b['conf_med']:.2f}"] for b in bt]
    md.append(md_table(
        ["view_angle [deg]", "n_pairs", "err_median", "err_p90", "conf_median"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:"]))

    md.append("\n## 7. Loop — err binned by baseline\n")
    bt = binned_table(per_pair, "baseline", base_bins, role="loop")
    rows = [[fmt_bin(b["lo"], b["hi"]), b["n"],
             f"{b['err_med']:.4f}", f"{b['err_p90']:.4f}",
             f"{b['conf_med']:.2f}"] for b in bt]
    md.append(md_table(
        ["baseline [m]", "n_pairs", "err_median", "err_p90", "conf_median"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:"]))

    # 7e. Loop vs tracking ratio per scene
    md.append("\n## 8. Loop vs tracking per scene (calib)\n")
    by_sr = {}
    for r in per_scene_rows:
        if r["mode"] != "calib":
            continue
        by_sr.setdefault(r["scene"], {})[r["role"]] = r
    rows = []
    for scene in sorted(by_sr.keys()):
        t = by_sr[scene].get("tracking")
        l = by_sr[scene].get("loop")
        if t is None or l is None:
            continue
        ratio = l["err_rmse"] / t["err_rmse"] if t["err_rmse"] > 0 else float("nan")
        rows.append([
            scene,
            f"{t['err_rmse']:.4f}", f"{l['err_rmse']:.4f}", f"{ratio:.2f}",
            f"{t['conf_mean']:.2f}", f"{l['conf_mean']:.2f}",
            f"{l.get('baseline_max', float('nan')):.3f}",
            f"{l.get('view_angle_deg_max', float('nan')):.2f}",
        ])
    md.append(md_table(
        ["scene", "track_err_rmse", "loop_err_rmse", "loop/track",
         "track_conf", "loop_conf", "loop_base_max", "loop_view_max[°]"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))

    # 7f. View-angle distribution check
    md.append("\n## 9. View angle distribution (sanity check — \"tracking view angles small?\")\n")
    for role in ("tracking", "loop"):
        vs = [r["view_angle_deg"] for r in per_pair
              if r["role"] == role and np.isfinite(r["view_angle_deg"])]
        if not vs:
            continue
        vs = np.array(vs)
        md.append(f"\n**{role}** (n={vs.size}):\n")
        md.append(f"- mean={vs.mean():.2f}°  median={np.median(vs):.2f}°  "
                  f"p90={np.percentile(vs, 90):.2f}°  p99={np.percentile(vs, 99):.2f}°  "
                  f"max={vs.max():.2f}°\n")
        bins = [0, 2, 5, 10, 20, 30, 45, 60, 90]
        counts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            counts.append(int(((vs >= lo) & (vs < hi)).sum()))
        rows = [[fmt_bin(lo, hi), c, f"{c/vs.size*100:.1f}%"]
                for (lo, hi), c in zip(zip(bins[:-1], bins[1:]), counts)]
        md.append(md_table(["bin [deg]", "n", "fraction"], rows,
                           aligns=["---", "---:", "---:"]))

    # Write summary.md
    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"\nwrote {md_path}")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'per_pair.csv'}")
    print(f"plots in {out_dir}/")


if __name__ == "__main__":
    main()
