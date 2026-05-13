"""Web-based GT-vs-pred viewer using viser.

Walks logs/7-scenes/<variant>/calib/<scene>/{<scene>.ply, <scene>.txt}, plus
datasets/7-scenes/<scene>/seq-01/ for the GT pointcloud and trajectory.
Sim3-aligns each variant's prediction to GT (Umeyama on trajectory), then
shows everything in GT coords. Toggles let you compare any subset of variants
against GT for any scene.

Usage:
  python scripts/viz_compare.py
  python scripts/viz_compare.py --logs-root logs --datasets-root datasets/7-scenes
  python scripts/viz_compare.py --port 8081

Open the printed URL in a browser.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import viser

# Reuse alignment + GT build from the eval script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_geometry import (   # noqa: E402
    load_ply_xyz,
    load_est_traj,
    load_gt_traj_7scenes,
    align_traj,
    apply_sim3,
    build_7scenes_gt_pcd,
)


VARIANT_COLORS = {
    "vanilla":    (200, 200, 200),
    "calibonly":  (100, 220, 100),
    "caponly":    (220, 220, 100),
    "calibcap":   (100, 160, 255),
    "fusion":     (255, 120, 120),
    "loopfuse":   (220, 120, 255),
}
GT_COLOR = (255, 230, 0)


def discover_variants(logs_root):
    seven = logs_root / "7-scenes"
    if not seven.exists():
        return []
    return sorted([p.name for p in seven.iterdir() if p.is_dir()])


def discover_scenes(logs_root, variants):
    s = set()
    for v in variants:
        for d in (logs_root / "7-scenes" / v / "calib").glob("*"):
            if (d / f"{d.name}.txt").exists():
                s.add(d.name)
    return sorted(s)


def color_for(name):
    return VARIANT_COLORS.get(name, (180, 180, 180))


def load_pred(scene, variant, logs_root, scene_dir):
    """Returns dict with sim3-aligned pred trajectory + pointcloud, plus the
    matched (pred, GT) pairs used by ATE so we can color by per-frame error.
    None if any required file is missing.
    """
    base = logs_root / "7-scenes" / variant / "calib" / scene
    est_ply = base / f"{scene}.ply"
    est_traj = base / f"{scene}.txt"
    if not est_ply.exists() or not est_traj.exists():
        return None
    pred_pts = load_ply_xyz(est_ply)
    est_ts, est_pos = load_est_traj(est_traj)
    gt_ts, gt_pos = load_gt_traj_7scenes(scene_dir)
    s, R, t, src_aligned, dst_aligned = align_traj(est_ts, est_pos, gt_ts, gt_pos)
    pred_pts_aligned = apply_sim3(pred_pts, s, R, t)
    pred_pos_aligned = apply_sim3(est_pos, s, R, t)
    # The matched pairs used by ATE — same frames where pred (after Sim3) and
    # GT timestamps agree. Per-frame error = || pred_after_sim3 - gt ||.
    pred_matched = apply_sim3(src_aligned, s, R, t).astype(np.float32)
    gt_matched = dst_aligned.astype(np.float32)
    per_frame_err = np.linalg.norm(pred_matched - gt_matched, axis=1)
    ate_rmse = float(np.sqrt((per_frame_err ** 2).mean()))
    return {
        "pred_pts": pred_pts_aligned.astype(np.float32),
        "pred_traj": pred_pos_aligned.astype(np.float32),
        "pred_matched": pred_matched,
        "gt_matched": gt_matched,
        "per_frame_err": per_frame_err.astype(np.float32),
        "ate_rmse": ate_rmse,
        "scale": float(s),
    }


def load_gt(scene, datasets_root, subsample, voxel):
    """Returns dict with GT trajectory + sparse pointcloud."""
    scene_dir = datasets_root / scene / "seq-01"
    if not scene_dir.exists():
        # Some 7-Scenes use seq-02 etc as test; default to first available.
        candidates = sorted(scene_dir.parent.glob("seq-*"))
        if not candidates:
            return None
        scene_dir = candidates[0]
    gt_ts, gt_pos = load_gt_traj_7scenes(scene_dir)
    gt_pcd = build_7scenes_gt_pcd(scene_dir, subsample=subsample, voxel=voxel)
    return {
        "gt_pts": np.asarray(gt_pcd.points, dtype=np.float32),
        "gt_traj": gt_pos.astype(np.float32),
        "scene_dir": scene_dir,
    }


def voxel_down(pts, voxel):
    """Cheap voxel downsample by rounding-to-grid uniqueness."""
    if voxel <= 0 or pts.shape[0] == 0:
        return pts
    keys = np.round(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-root", default="logs")
    p.add_argument("--datasets-root", default="datasets/7-scenes")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--gt-subsample", type=int, default=20,
                   help="Every Nth GT frame when building GT pcd.")
    p.add_argument("--gt-voxel", type=float, default=0.02,
                   help="GT voxel-downsample size in metres.")
    p.add_argument("--pred-voxel", type=float, default=0.01,
                   help="Pred voxel-downsample size in metres.")
    args = p.parse_args()

    logs_root = Path(args.logs_root)
    datasets_root = Path(args.datasets_root)

    variants = discover_variants(logs_root)
    if not variants:
        raise SystemExit(f"no variants found under {logs_root}/7-scenes/")
    scenes = discover_scenes(logs_root, variants)
    if not scenes:
        raise SystemExit(f"no scenes with trajectories found")

    print(f"variants: {variants}")
    print(f"scenes: {scenes}")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")

    # Cache loaded data so toggling doesn't re-load
    gt_cache = {}
    pred_cache = {}

    # ------------------------------------------------------------------ #
    # GUI
    # ------------------------------------------------------------------ #
    with server.gui.add_folder("Scene"):
        scene_dd = server.gui.add_dropdown("scene", options=scenes, initial_value=scenes[0])

    with server.gui.add_folder("Show"):
        show_gt = server.gui.add_checkbox("GT pointcloud", initial_value=True)
        show_gt_traj = server.gui.add_checkbox("GT trajectory", initial_value=True)
        show_pred_pcd = server.gui.add_checkbox("Pred pointcloud", initial_value=True)
        show_pred_traj = server.gui.add_checkbox("Pred trajectory", initial_value=True)

    with server.gui.add_folder("Variants"):
        variant_cbs = {}
        for v in variants:
            default_on = v in ("vanilla", "loopfuse")
            variant_cbs[v] = server.gui.add_checkbox(v, initial_value=default_on)

    with server.gui.add_folder("Render"):
        point_size = server.gui.add_slider("point size", 0.001, 0.05, 0.001, 0.005)
        traj_width = server.gui.add_slider("traj width", 1.0, 12.0, 0.5, 4.0)
        error_view = server.gui.add_checkbox("color pred traj by error", initial_value=False)
        show_err_vecs = server.gui.add_checkbox("show error vectors (pred↔GT)", initial_value=False)
        err_clip = server.gui.add_slider("error colormap cap (m)", 0.005, 0.5, 0.005, 0.05)

    stats_md = server.gui.add_markdown("(no variant loaded yet)")

    # Track scene-graph handles so we can remove on update
    handles = []

    def clear_handles():
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def refresh():
        clear_handles()
        scene = scene_dd.value

        # GT load (cached)
        if scene not in gt_cache:
            print(f"loading GT for {scene} (subsample={args.gt_subsample}, "
                  f"voxel={args.gt_voxel}) ...")
            gt_cache[scene] = load_gt(
                scene, datasets_root,
                subsample=args.gt_subsample, voxel=args.gt_voxel,
            )
        gt = gt_cache[scene]
        if gt is None:
            print(f"no GT for {scene}")
            return

        if show_gt.value and gt["gt_pts"].size:
            gt_pts = gt["gt_pts"]
            handles.append(server.scene.add_point_cloud(
                f"/gt_{scene}",
                points=gt_pts,
                colors=np.tile(GT_COLOR, (gt_pts.shape[0], 1)).astype(np.uint8),
                point_size=point_size.value,
            ))

        if show_gt_traj.value and gt["gt_traj"].shape[0] >= 2:
            handles.append(server.scene.add_spline_catmull_rom(
                f"/gt_traj_{scene}",
                positions=gt["gt_traj"],
                color=GT_COLOR,
                line_width=traj_width.value,
            ))

        # Each enabled variant
        for v, cb in variant_cbs.items():
            if not cb.value:
                continue
            key = (scene, v)
            if key not in pred_cache:
                print(f"loading pred {v}/{scene} ...")
                pred_cache[key] = load_pred(
                    scene, v, logs_root, gt["scene_dir"]
                )
            data = pred_cache[key]
            if data is None:
                continue
            col = color_for(v)

            if show_pred_pcd.value and data["pred_pts"].size:
                pts = voxel_down(data["pred_pts"], args.pred_voxel)
                handles.append(server.scene.add_point_cloud(
                    f"/pred_{v}_{scene}",
                    points=pts,
                    colors=np.tile(col, (pts.shape[0], 1)).astype(np.uint8),
                    point_size=point_size.value,
                ))

            if show_pred_traj.value and data["pred_traj"].shape[0] >= 2:
                if error_view.value and data["pred_matched"].size:
                    # Render the matched pred points colored by per-frame error
                    # (red = ≥ cap, green = 0). This is what ATE actually
                    # measures, so it's where the metric's mass lives.
                    e = data["per_frame_err"]
                    cap = max(err_clip.value, 1e-6)
                    t01 = np.clip(e / cap, 0.0, 1.0)
                    colors = np.stack([
                        (255 * t01).astype(np.uint8),
                        (255 * (1.0 - t01)).astype(np.uint8),
                        np.zeros_like(t01, dtype=np.uint8),
                    ], axis=1)
                    handles.append(server.scene.add_point_cloud(
                        f"/pred_err_{v}_{scene}",
                        points=data["pred_matched"],
                        colors=colors,
                        point_size=max(point_size.value * 3, 0.01),
                    ))
                else:
                    handles.append(server.scene.add_spline_catmull_rom(
                        f"/pred_traj_{v}_{scene}",
                        positions=data["pred_traj"],
                        color=col,
                        line_width=traj_width.value,
                    ))

            # Error vectors — small line segments from each matched pred to its
            # GT counterpart. Reveals if ATE is dominated by a few outliers vs.
            # a uniform offset.
            if show_err_vecs.value and data["pred_matched"].size:
                pm = data["pred_matched"]
                gm = data["gt_matched"]
                n = pm.shape[0]
                # Use add_point_cloud-as-pairs trick: not all viser builds
                # have add_line_segments, so render midpoints colored by error
                # AND the endpoints as line by emitting many short splines.
                # Cheap fallback: draw one spline per pair.
                # Limit count for very long sequences.
                stride = max(1, n // 200)
                for k in range(0, n, stride):
                    handles.append(server.scene.add_spline_catmull_rom(
                        f"/err_vec_{v}_{scene}_{k}",
                        positions=np.stack([pm[k], gm[k]], axis=0),
                        color=col,
                        line_width=max(1.5, traj_width.value * 0.5),
                    ))

    # Add stats update at end of refresh — show per-variant ATE breakdown so
    # the user can see how "5 cm RMSE" decomposes (e.g. one big outlier vs.
    # uniform drift).
    _orig_refresh = refresh

    def refresh_with_stats():
        _orig_refresh()
        scene = scene_dd.value
        rows = ["| variant | n | mean | p50 | p90 | p99 | max | RMSE (= ATE) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for v, cb in variant_cbs.items():
            if not cb.value:
                continue
            data = pred_cache.get((scene, v))
            if data is None:
                continue
            e = data["per_frame_err"]
            if e.size == 0:
                continue
            rows.append(
                f"| {v} | {e.size} | {e.mean():.4f} | "
                f"{np.median(e):.4f} | {np.percentile(e, 90):.4f} | "
                f"{np.percentile(e, 99):.4f} | {e.max():.4f} | "
                f"**{data['ate_rmse']:.4f}** |"
            )
        if len(rows) > 2:
            stats_md.content = (
                f"### {scene} — per-frame translation error (m)\n\n"
                + "\n".join(rows)
            )
        else:
            stats_md.content = f"### {scene}\n\n(no variants enabled)"

    refresh = refresh_with_stats

    # ------------------------------------------------------------------ #
    # Wire up callbacks
    # ------------------------------------------------------------------ #
    for ctrl in (scene_dd, show_gt, show_gt_traj, show_pred_pcd,
                 show_pred_traj, point_size, traj_width,
                 error_view, show_err_vecs, err_clip,
                 *variant_cbs.values()):
        ctrl.on_update(lambda _: refresh())

    refresh()

    print(f"\nviser server running on http://localhost:{args.port}\n"
          f"(ctrl-c to stop)\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping")


if __name__ == "__main__":
    main()
