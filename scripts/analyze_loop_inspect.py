"""Analyse loop-closure acceptance dump from `--loop-inspect-dir`.

Joins the per-edge summary (i, j, match_frac, Q_mean, pointmap residual,
estimated poses) against an optional GT trajectory (TUM format) and answers:
  - For accepted loops, do the estimated relative poses match GT?
  - Is pointmap residual a good signal for "this loop is geometrically OK"?
  - What threshold on residual would have rejected the bad ones?

Usage:
  python scripts/analyze_loop_inspect.py \
      --inspect-dir logs/loop_inspect/tum_fr1_xyz \
      --gt datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
      --traj logs/<save-as>/<seq>.txt    # optional, the SLAM-output trajectory
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_tum_gt(path):
    """TUM groundtruth.txt: t tx ty tz qx qy qz qw. Returns dict {row_idx: 4x4}.

    Index keying is by row (line-order), since the SLAM side stores keyframe
    integer ids that the inspector echoes back. Caller can join differently if
    needed.
    """
    data = np.loadtxt(path, comments="#")
    out = []
    for row in data:
        t = row[1:4]
        q = row[4:8]  # qx qy qz qw
        R = quat_to_R(q)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        out.append(T)
    return np.stack(out), data[:, 0]  # poses, timestamps


def load_slam_traj(path):
    """TUM-format SLAM trajectory output. Returns Nx4x4 + timestamps."""
    return load_tum_gt(path)


def quat_to_R(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    s = 2.0 / n
    R = np.array([
        [1 - s*(y*y + z*z),  s*(x*y - z*w),     s*(x*z + y*w)],
        [s*(x*y + z*w),      1 - s*(x*x + z*z), s*(y*z - x*w)],
        [s*(x*z - y*w),      s*(y*z + x*w),     1 - s*(x*x + y*y)],
    ])
    return R


def view_angle_deg(R_i, R_j):
    R_rel = R_i.T @ R_j
    cos_a = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect-dir", required=True)
    p.add_argument("--gt", default="", help="TUM groundtruth.txt (optional).")
    p.add_argument("--traj", default="", help="SLAM trajectory output (optional).")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    inspect_dir = Path(args.inspect_dir)
    npz_path = inspect_dir / "loop_inspect.npz"
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path}")
    d = dict(np.load(npz_path, allow_pickle=False))
    n = len(d["i"])
    print(f"loaded {n} loop edges from {npz_path}")

    out_dir = Path(args.out_dir) if args.out_dir else inspect_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-edge SLAM-estimated baseline / view angle from t_i / t_j
    bl_est = np.linalg.norm(d["t_i"] - d["t_j"], axis=1)
    va_est = np.array([view_angle_deg(d["R_i"][k], d["R_j"][k]) for k in range(n)])

    # Join GT if available. For now we assume the inspector's `i`, `j` indices
    # are *keyframe* indices, not raw frame indices. Without a keyframe→raw
    # mapping we can still inspect relative consistency: if SLAM's estimated
    # baseline is small but pointmap residual is large, MASt3R disagrees with
    # the SLAM-internal estimate (regardless of GT correctness).
    have_gt = bool(args.gt)
    if have_gt:
        gt_poses, gt_ts = load_tum_gt(args.gt)
        # We don't have a kf→raw map here; report the GT-only statistic
        # against (i, j) bounded into the gt length, just as a coarse check.
        # Real join would require recording dataset.timestamps[frame_id] in
        # the inspector. For now, leave the per-edge GT join unfilled but
        # report aggregate stats below.
        print(f"GT loaded: {gt_poses.shape}, but kf→raw map not embedded — "
              f"skipping per-edge GT join. Aggregate stats below use SLAM "
              f"poses as a proxy.")

    # Tag edges
    consec = d["is_consecutive"].astype(bool)
    real_loop = ~consec

    # Plot 1: residual vs SLAM baseline, color by consec/loop
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(bl_est[consec], d["med_residual"][consec], s=12, alpha=0.5,
               label=f"consecutive (n={consec.sum()})", color="tab:blue")
    ax.scatter(bl_est[real_loop], d["med_residual"][real_loop], s=14,
               alpha=0.7, label=f"real loop (n={real_loop.sum()})",
               color="tab:red")
    ax.set_xlabel("SLAM-estimated baseline ||t_i − t_j|| (m)")
    ax.set_ylabel("|X_canon_i − X_ji| median (m) at matched px")
    ax.set_title("Loop edge: pointmap residual vs estimated baseline")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loop_residual_vs_baseline.png", dpi=110)
    plt.close(fig)

    # Plot 2: residual vs match_frac (acceptance criterion)
    mf_min = np.minimum(d["match_frac_i"], d["match_frac_j"])
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(mf_min[consec], d["med_residual"][consec], s=12, alpha=0.5,
               label="consecutive", color="tab:blue")
    ax.scatter(mf_min[real_loop], d["med_residual"][real_loop], s=14, alpha=0.7,
               label="real loop", color="tab:red")
    ax.set_xlabel("min(match_frac_i, match_frac_j)")
    ax.set_ylabel("median residual (m)")
    ax.set_title("Loop edge: pointmap residual vs acceptance match_frac")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loop_residual_vs_match_frac.png", dpi=110)
    plt.close(fig)

    # Plot 3: residual histogram, threshold candidates
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    bins = np.linspace(0, max(d["med_residual"].max(), 1e-3), 40)
    ax.hist(d["med_residual"][consec], bins=bins, alpha=0.55,
            label=f"consecutive", color="tab:blue")
    ax.hist(d["med_residual"][real_loop], bins=bins, alpha=0.55,
            label=f"real loop", color="tab:red")
    for t in (0.1, 0.3, 1.0):
        ax.axvline(t, ls="--", alpha=0.4)
        ax.text(t, ax.get_ylim()[1] * 0.9, f" {t:g}m", fontsize=8)
    ax.set_xlabel("median residual (m)")
    ax.set_ylabel("n edges")
    ax.set_title("Distribution of pointmap residual at matched pixels")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loop_residual_hist.png", dpi=110)
    plt.close(fig)

    # Text summary
    md = ["# Loop closure inspection summary\n"]
    md.append(f"Total accepted loop edges: **{n}**  "
              f"({consec.sum()} consecutive + {real_loop.sum()} retrieval-discovered)\n")
    md.append("\n## Per-category stats\n")
    md.append("| category | n | med residual | p90 residual | mean match_frac | mean Q |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for tag, mask in [("consecutive", consec), ("retrieval", real_loop), ("all", np.ones(n, bool))]:
        if mask.sum() == 0:
            continue
        md.append(
            f"| {tag} | {int(mask.sum())} | "
            f"{np.median(d['med_residual'][mask]):.3f} | "
            f"{np.percentile(d['med_residual'][mask], 90):.3f} | "
            f"{mf_min[mask].mean():.3f} | "
            f"{d['Q_mean'][mask].mean():.3f} |\n"
        )

    md.append("\n## Candidate residual thresholds (would-be reject rate)\n")
    md.append("| residual cap | retrieval rejected | consecutive rejected |\n")
    md.append("|---:|---:|---:|\n")
    for t in (0.1, 0.2, 0.3, 0.5, 1.0):
        rej_real = int((d["med_residual"][real_loop] > t).sum())
        rej_consec = int((d["med_residual"][consec] > t).sum())
        md.append(f"| {t:.2f} | {rej_real} / {int(real_loop.sum())} | "
                  f"{rej_consec} / {int(consec.sum())} |\n")

    md_path = out_dir / "loop_inspect_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {md_path}")
    print(f"plots in {out_dir}/")


if __name__ == "__main__":
    main()
