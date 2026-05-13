"""Geometry + ATE evaluation following paper Section 11.2.

For 7-Scenes (calib / no-calib):
  reference pcd  = depth backproject + GT pose for every frame in seq-01
  est pcd        = saved .ply from main.py (--save-as)
  alignment      = (1) Sim(3) trajectory align (Umeyama)  +  (2) Sim(3) ICP refine
  metric         = Accuracy / Completion / Chamfer (RMSE & mean) within max_dist=0.5m
  ATE            = RMSE of translation residuals after Sim(3) trajectory alignment

Usage:
  python scripts/eval_geometry.py \
      --est-ply  logs/7-scenes/calib/chess/chess.ply \
      --est-traj logs/7-scenes/calib/chess/chess.txt \
      --scene-dir datasets/7-scenes/chess
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# ---------- I/O ----------

def load_ply_xyz(ply_path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return np.asarray(pcd.points)


def load_est_traj(traj_path):
    """TUM format: t tx ty tz qx qy qz qw."""
    data = np.loadtxt(traj_path)
    return data[:, 0], data[:, 1:4]


def load_gt_traj_7scenes(seq_dir):
    """7-Scenes: one pose.txt per frame (4x4), index from filename."""
    pose_files = sorted(Path(seq_dir).glob("frame-*.pose.txt"))
    ts, pos = [], []
    for pf in pose_files:
        T = np.loadtxt(pf)
        if not np.all(np.isfinite(T)):
            continue
        idx = int(pf.stem.split("-")[1].split(".")[0])
        ts.append(float(idx))
        pos.append(T[:3, 3])
    return np.array(ts), np.array(pos)


# ---------- Alignment ----------

def umeyama_sim3(src, dst):
    """Sim(3) that best fits src -> dst.  Returns (s, R, t) with s*R @ src + t ≈ dst."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / n
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / var_s
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def align_traj(est_ts, est_pos, gt_ts, gt_pos):
    """Match each est timestamp to nearest gt; return (s, R, t) and the matched pairs."""
    j = np.array([np.argmin(np.abs(gt_ts - t)) for t in est_ts])
    src, dst = est_pos, gt_pos[j]
    s, R, t = umeyama_sim3(src, dst)
    return s, R, t, src, dst


def apply_sim3(pts, s, R, t):
    return s * (pts @ R.T) + t


def to_o3d(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pcd


def icp_refine_sim3(est_pcd, gt_pcd, voxel, max_iter=200):
    """Sim(3) ICP refinement.  Returns 4x4 transform (with scale baked in)."""
    threshold = voxel * 3.0
    reg = o3d.pipelines.registration.registration_icp(
        est_pcd, gt_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return reg.transformation, reg.inlier_rmse, reg.fitness


# ---------- GT pcd ----------

def backproject_depth(depth_mm, K, pose_w_c):
    fx, fy, cx, cy = K
    valid = (depth_mm > 0) & (depth_mm < 65535)
    if not valid.any():
        return None
    ys, xs = np.where(valid)
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    return pts_cam @ pose_w_c[:3, :3].T + pose_w_c[:3, 3]


def build_7scenes_gt_pcd(seq_dir, K=(585.0, 585.0, 320.0, 240.0), subsample=1, voxel=0.01,
                         kf_timestamps=None):
    """Build GT pointcloud from 7-Scenes depth+pose.

    If kf_timestamps is provided, only depths at those frame indices are used
    (matches the SLAM keyframe coverage — paper's 'unobservable removed' interpretation).
    """
    seq_dir = Path(seq_dir)
    depth_files = sorted(seq_dir.glob("frame-*.depth.png"))

    if kf_timestamps is not None:
        kf_indices = set(int(round(ts)) for ts in kf_timestamps)
        depth_files = [
            f for f in depth_files
            if int(f.stem.split("-")[1].split(".")[0]) in kf_indices
        ]

    all_pts = []
    for i, dpath in enumerate(depth_files):
        if kf_timestamps is None and i % subsample != 0:
            continue
        depth = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        ppath = dpath.parent / dpath.name.replace(".depth.png", ".pose.txt")
        pose = np.loadtxt(ppath)
        if not np.all(np.isfinite(pose)):
            continue
        pts = backproject_depth(depth, K, pose)
        if pts is not None:
            all_pts.append(pts)
    pts = np.concatenate(all_pts, axis=0)
    pcd = to_o3d(pts)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


# ---------- Metrics ----------

def chamfer_metrics(est_pcd, gt_pcd, max_dist=0.5):
    """Accuracy (est->gt), Completion (gt->est), Chamfer.  RMSE & mean within max_dist."""
    d_e2g = np.asarray(est_pcd.compute_point_cloud_distance(gt_pcd))
    d_g2e = np.asarray(gt_pcd.compute_point_cloud_distance(est_pcd))

    def stat(d):
        v = d[d < max_dist]
        if v.size == 0:
            return float("nan"), float("nan")
        return float(np.sqrt((v ** 2).mean())), float(v.mean())

    acc_r, acc_m = stat(d_e2g)
    com_r, com_m = stat(d_g2e)
    return {
        "accuracy_rmse": acc_r, "completion_rmse": com_r,
        "chamfer_rmse": (acc_r + com_r) / 2,
        "accuracy_mean": acc_m, "completion_mean": com_m,
        "chamfer_mean": (acc_m + com_m) / 2,
    }


def ate_rmse(src, dst):
    """ATE: RMSE of residuals after Sim(3) Umeyama (already aligned src in dst frame)."""
    diff = src - dst
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--est-ply", required=True)
    p.add_argument("--est-traj", required=True)
    p.add_argument("--scene-dir", required=True, help="datasets/7-scenes/<scene>")
    p.add_argument("--seq", default="seq-01")
    p.add_argument("--max-dist", type=float, default=0.5)
    p.add_argument("--voxel", type=float, default=0.01)
    p.add_argument("--subsample", type=int, default=1, help="GT frame subsample (only if --all-frames-gt)")
    p.add_argument("--no-icp", action="store_true", help="skip ICP refinement")
    p.add_argument("--all-frames-gt", action="store_true",
                   help="use every frame for GT (default: only SLAM keyframe timestamps, matches paper)")
    p.add_argument("--force", action="store_true",
                   help="recompute even if eval.json cache exists.")
    p.add_argument("--cache-name", default="eval.json",
                   help="filename for cached eval result next to est-traj.")
    args = p.parse_args()

    seq_dir = Path(args.scene_dir) / args.seq

    # Cache: if a previous eval has the same key args, just print it back.
    cache_path = Path(args.est_traj).parent / args.cache_name
    cache_key = {
        "max_dist": args.max_dist,
        "voxel": args.voxel,
        "subsample": args.subsample,
        "no_icp": bool(args.no_icp),
        "all_frames_gt": bool(args.all_frames_gt),
    }
    if cache_path.exists() and not args.force:
        import json
        with open(cache_path) as f:
            cached = json.load(f)
        # If args match, print cached numbers and skip ICP.
        if cached.get("_args") == cache_key:
            m = cached["metrics"]
            print(f"[cached] {cache_path}")
            print(f"      ATE              : {m['ate']:.4f} m")
            print(f"      Accuracy   RMSE  : {m['accuracy_rmse']:.4f} m   "
                  f"(mean = {m['accuracy_mean']:.4f})")
            print(f"      Completion RMSE  : {m['completion_rmse']:.4f} m   "
                  f"(mean = {m['completion_mean']:.4f})")
            print(f"      Chamfer    RMSE  : {m['chamfer_rmse']:.4f} m   "
                  f"(mean = {m['chamfer_mean']:.4f})")
            if "fps" in cached:
                print(f"      FPS              : {cached['fps']:.2f}  "
                      f"(frames={cached.get('frames','?')}, "
                      f"kf={cached.get('keyframes','?')}, "
                      f"time={cached.get('total_time_s', 0):.1f}s)")
            return
        else:
            print(f"[cache mismatch — recomputing]  cached args: "
                  f"{cached.get('_args')}, requested: {cache_key}")

    print(f"[1/5] Loading est pcd       : {args.est_ply}")
    est_pts = load_ply_xyz(args.est_ply)
    print(f"      {len(est_pts):,} points")

    print(f"[2/5] Trajectory alignment  : {args.est_traj}  <-  {seq_dir}")
    est_ts, est_pos = load_est_traj(args.est_traj)
    gt_ts, gt_pos = load_gt_traj_7scenes(seq_dir)
    s, R, t, src_aligned_src, src_aligned_dst = align_traj(est_ts, est_pos, gt_ts, gt_pos)
    est_pos_aligned = apply_sim3(src_aligned_src, s, R, t)
    ate = ate_rmse(est_pos_aligned, src_aligned_dst)
    print(f"      scale={s:.4f}, |t|={np.linalg.norm(t):.3f}, ATE RMSE = {ate:.4f} m")

    # Apply traj alignment to estimated points
    est_pts_aligned = apply_sim3(est_pts, s, R, t)
    est_pcd = to_o3d(est_pts_aligned).voxel_down_sample(args.voxel)
    print(f"      est: {len(est_pcd.points):,} points (after voxel={args.voxel}m)")

    if args.all_frames_gt:
        print(f"[3/5] Building GT pcd       : depth + pose backproject (all frames, subsample={args.subsample})")
        gt_pcd = build_7scenes_gt_pcd(seq_dir, subsample=args.subsample, voxel=args.voxel)
    else:
        print(f"[3/5] Building GT pcd       : keyframe-only ({len(est_ts)} kf timestamps)")
        gt_pcd = build_7scenes_gt_pcd(seq_dir, voxel=args.voxel, kf_timestamps=est_ts)
    print(f"      gt:  {len(gt_pcd.points):,} points")

    if not args.no_icp:
        print(f"[4/5] ICP refinement        : Sim(3) point-to-point")
        T_icp, inlier_rmse, fitness = icp_refine_sim3(est_pcd, gt_pcd, args.voxel)
        est_pcd = est_pcd.transform(T_icp)
        print(f"      inlier RMSE={inlier_rmse:.4f}, fitness={fitness:.3f}")
    else:
        print(f"[4/5] ICP refinement        : skipped")

    print(f"[5/5] Metrics (max_dist={args.max_dist}m):")
    m = chamfer_metrics(est_pcd, gt_pcd, args.max_dist)
    print(f"      ATE              : {ate:.4f} m")
    print(f"      Accuracy   RMSE  : {m['accuracy_rmse']:.4f} m   (mean = {m['accuracy_mean']:.4f})")
    print(f"      Completion RMSE  : {m['completion_rmse']:.4f} m   (mean = {m['completion_mean']:.4f})")
    print(f"      Chamfer    RMSE  : {m['chamfer_rmse']:.4f} m   (mean = {m['chamfer_mean']:.4f})")

    # SLAM runtime stats (if saved by main.py)
    fps_stats = None
    stats_path = Path(args.est_traj).parent / (Path(args.est_traj).stem + "_stats.json")
    if stats_path.exists():
        import json as _json
        with open(stats_path) as f:
            fps_stats = _json.load(f)
        print(f"      FPS              : {fps_stats['fps']:.2f}  "
              f"(frames={fps_stats['frames']}, kf={fps_stats['keyframes']}, "
              f"time={fps_stats['total_time_s']:.1f}s)")

    # Cache the result for fast re-display next time.
    import json as _json
    cached_out = {
        "_args": cache_key,
        "metrics": {
            "ate": float(ate),
            "accuracy_rmse": float(m["accuracy_rmse"]),
            "accuracy_mean": float(m["accuracy_mean"]),
            "completion_rmse": float(m["completion_rmse"]),
            "completion_mean": float(m["completion_mean"]),
            "chamfer_rmse": float(m["chamfer_rmse"]),
            "chamfer_mean": float(m["chamfer_mean"]),
        },
        "sim3": {"scale": float(s), "t_norm": float(np.linalg.norm(t))},
        "n_points_est": int(len(est_pcd.points)),
        "n_points_gt": int(len(gt_pcd.points)),
    }
    if not args.no_icp:
        cached_out["icp"] = {
            "inlier_rmse": float(inlier_rmse),
            "fitness": float(fitness),
        }
    if fps_stats is not None:
        cached_out.update({
            "fps": float(fps_stats["fps"]),
            "frames": int(fps_stats["frames"]),
            "keyframes": int(fps_stats["keyframes"]),
            "total_time_s": float(fps_stats["total_time_s"]),
        })
    with open(cache_path, "w") as f:
        _json.dump(cached_out, f, indent=2)
    print(f"      cached → {cache_path}")


if __name__ == "__main__":
    main()
