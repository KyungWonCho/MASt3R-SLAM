"""Analyze diag NPZs collected by --diag-dir runs.

Reads tracking.npz and (if present) loop.npz from a directory tree of shape

  <diag_root>/<scene>/<mode>/{tracking,loop}.npz

Produces:
  <out_dir>/summary.csv          per (scene, mode, role) stats + calibration fits
  <out_dir>/<scene>_<mode>_sigma_vs_conf.png   σ(c) curves
  <out_dir>/<scene>_<mode>_err_vs_baseline.png loop edges: err vs baseline
  <out_dir>/aggregate_sigma_vs_conf.png        all scenes/modes overlaid
  <out_dir>/summary.md                         human-readable summary
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_npz(path):
    if not path.exists():
        return None
    d = dict(np.load(path, allow_pickle=True))
    # Unpack valid_match
    n = int(d["valid_match_len"])
    d["valid_match"] = np.unpackbits(d["valid_match_packed"])[:n].astype(bool)
    return d


def calibration_from_hist(hist, edges):
    """Returns per-bin mid-conf c, count, mean_err, rmse_err, std_err."""
    count = hist[:, 0]
    s = hist[:, 1]
    s2 = hist[:, 2]
    valid = count > 5  # need a few samples
    c_mid = np.sqrt(edges[:-1] * edges[1:])  # geometric midpoint (log-spaced)
    mean = np.where(valid, s / np.maximum(count, 1), np.nan)
    rmse = np.where(valid, np.sqrt(s2 / np.maximum(count, 1)), np.nan)
    # std: sqrt(E[x^2] - E[x]^2)
    var = np.where(valid, np.maximum(s2 / np.maximum(count, 1) - mean ** 2, 0.0), np.nan)
    std = np.sqrt(var)
    return c_mid, count, mean, rmse, std


def fit_powerlaw_sigma_c(c_mid, sigma):
    """Fit sigma ≈ a * c^p. Returns (a, p) via log-log linear regression
    on bins with finite sigma > 0. Returns (nan, nan) if too few points.
    """
    m = np.isfinite(sigma) & (sigma > 0) & np.isfinite(c_mid) & (c_mid > 0)
    if m.sum() < 5:
        return float("nan"), float("nan")
    x = np.log(c_mid[m])
    y = np.log(sigma[m])
    A = np.vstack([x, np.ones_like(x)]).T
    p, log_a = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.exp(log_a)), float(p)


def summarize_pairs(d):
    """Return a dict of high-level stats from a loaded NPZ dict."""
    err = d["err"]
    conf = d["conf"]
    finite = np.isfinite(err) & np.isfinite(conf)
    err_f = err[finite].astype(np.float32)
    conf_f = conf[finite].astype(np.float32)
    out = {
        "n_pairs": int(len(d["n_pixels"])),
        "n_pixels_total": int(err.size),
        "n_pixels_finite": int(finite.sum()),
    }
    if err_f.size:
        out.update({
            "err_mean": float(err_f.mean()),
            "err_median": float(np.median(err_f)),
            "err_p90": float(np.percentile(err_f, 90)),
            "err_rmse": float(np.sqrt((err_f ** 2).mean())),
            "conf_mean": float(conf_f.mean()),
            "conf_median": float(np.median(conf_f)),
        })
    else:
        out.update({"err_mean": float("nan"), "err_median": float("nan"),
                    "err_p90": float("nan"), "err_rmse": float("nan"),
                    "conf_mean": float("nan"), "conf_median": float("nan")})
    if "baseline" in d:
        out["baseline_mean"] = float(np.nanmean(d["baseline"]))
        out["baseline_max"] = float(np.nanmax(d["baseline"]))
    if "view_angle_deg" in d:
        out["view_angle_mean"] = float(np.nanmean(d["view_angle_deg"]))
        out["view_angle_max"] = float(np.nanmax(d["view_angle_deg"]))
    if "frame_diff" in d:
        out["frame_diff_max"] = int(np.nanmax(d["frame_diff"]))
    return out


def plot_sigma_vs_conf(c_mid, rmse, std, count, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.loglog(c_mid, rmse, "o-", label=r"RMSE err  $\sqrt{E[r^2 | c]}$", color="tab:blue")
    ax.loglog(c_mid, std, "s--", label=r"std err  $\sqrt{\mathrm{Var}[r | c]}$", color="tab:orange")
    ax.set_xlabel("confidence c (log)")
    ax.set_ylabel("error  (m, log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # secondary axis: bin count
    ax2 = ax.twinx()
    ax2.semilogx(c_mid, count, "k:", alpha=0.4)
    ax2.set_ylabel("bin count (dotted)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_err_vs_baseline(d, title, out_path):
    """Loop NPZ: per-pair mean err vs baseline."""
    if "baseline" not in d or len(d["baseline"]) == 0:
        return
    err = d["err"]
    n_pix = d["n_pixels"]
    offs = d["pair_offset"]
    per_pair_mean = []
    per_pair_med = []
    for i in range(len(n_pix)):
        a, b = int(offs[i]), int(offs[i + 1])
        chunk = err[a:b]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == 0:
            per_pair_mean.append(np.nan)
            per_pair_med.append(np.nan)
        else:
            per_pair_mean.append(float(chunk.mean()))
            per_pair_med.append(float(np.median(chunk)))
    per_pair_mean = np.array(per_pair_mean)
    per_pair_med = np.array(per_pair_med)
    baseline = np.asarray(d["baseline"])
    consec = np.asarray(d.get("is_consecutive", np.zeros_like(baseline, np.int8)))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    m_loop = consec == 0
    m_consec = consec == 1
    ax.scatter(baseline[m_consec], per_pair_med[m_consec], s=10, alpha=0.5,
               label=f"consecutive ({m_consec.sum()})", color="tab:blue")
    ax.scatter(baseline[m_loop], per_pair_med[m_loop], s=18, alpha=0.7,
               label=f"loop closure ({m_loop.sum()})", color="tab:red")
    ax.set_xlabel("baseline  ||t_i - t_j||  (m)")
    ax.set_ylabel("per-pair median err  (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag-root", default="logs/diag")
    p.add_argument("--out-dir", default="logs/diag/plots")
    args = p.parse_args()

    root = Path(args.diag_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # for CSV
    # for aggregate plot
    agg = []  # list of (label, c_mid, rmse, count)

    scenes = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name != "plots"])
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
                stats = summarize_pairs(d)
                # online histogram fit
                hist = d.get("hist")
                edges = d.get("hist_edges")
                if hist is not None and edges is not None and hist.size > 0:
                    total_hist = hist.sum(axis=0)
                    c_mid, count, mean, rmse, std = calibration_from_hist(total_hist, edges)
                    a, p_exp = fit_powerlaw_sigma_c(c_mid, rmse)
                    stats["sigma_rmse_a"] = a
                    stats["sigma_rmse_p"] = p_exp
                    a2, p2 = fit_powerlaw_sigma_c(c_mid, std)
                    stats["sigma_std_a"] = a2
                    stats["sigma_std_p"] = p2

                    title = f"{scene}/{mode}/{role}   σ ≈ {a:.3g} · c^{p_exp:.2f}"
                    out_png = out_dir / f"{scene}_{mode}_{role}_sigma_vs_conf.png"
                    plot_sigma_vs_conf(c_mid, rmse, std, count, title, out_png)
                    agg.append((f"{scene}/{mode}/{role}", c_mid, rmse, count))
                else:
                    stats["sigma_rmse_a"] = stats["sigma_rmse_p"] = float("nan")
                    stats["sigma_std_a"] = stats["sigma_std_p"] = float("nan")

                # loop edges: per-pair err vs baseline
                if role == "loop":
                    out_png2 = out_dir / f"{scene}_{mode}_loop_err_vs_baseline.png"
                    plot_err_vs_baseline(d, f"{scene}/{mode} loop edges", out_png2)

                rows.append({
                    "scene": scene,
                    "mode": mode,
                    "role": role,
                    **stats,
                })
                print(f"[{scene}/{mode}/{role}]  pairs={stats['n_pairs']}  "
                      f"pix={stats['n_pixels_finite']:,}  "
                      f"err_rmse={stats['err_rmse']:.4f}  "
                      f"σ≈{stats['sigma_rmse_a']:.3g}·c^{stats['sigma_rmse_p']:.2f}")

    # CSV
    if rows:
        csv_path = out_dir / "summary.csv"
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {csv_path}")

    # Aggregate plot: all sigma_vs_conf overlaid (tracking only — loop is sparse)
    if agg:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for (label, c_mid, rmse, count) in agg:
            if "tracking" not in label:
                continue
            ax.loglog(c_mid, rmse, "o-", alpha=0.6, label=label, markersize=4)
        ax.set_xlabel("confidence c (log)")
        ax.set_ylabel("RMSE err  (m, log)")
        ax.set_title("σ(c) — all scenes/modes (tracking)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "aggregate_sigma_vs_conf.png", dpi=110)
        plt.close(fig)

    # Human-readable summary
    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("# Diag analysis summary\n\n")
        f.write("Confidence vs error calibration: σ ≈ a · c^p  fit on RMSE per conf bin.\n\n")
        f.write("| scene | mode | role | pairs | pixels | err_rmse | a | p |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['scene']} | {r['mode']} | {r['role']} | "
                f"{r['n_pairs']} | {r['n_pixels_finite']:,} | "
                f"{r['err_rmse']:.4f} | {r['sigma_rmse_a']:.3g} | {r['sigma_rmse_p']:.3f} |\n"
            )
        f.write("\nLower |p| ⇒ confidence less sensitive to error; |p|≈1 ⇒ σ ∝ 1/c calibration\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
