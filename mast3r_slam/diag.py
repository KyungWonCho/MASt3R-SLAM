"""Per-pixel error vs confidence diagnostic for pointmap fusion analysis.

Records MASt3R cross-view pointmap predictions against 7-Scenes GT
(backprojected depth + GT poses) at two call sites:

  - tracking: every frame, (frame, current keyframe) asymmetric inference.
              We record (X_kf, C_kf): keyframe's pointmap predicted in
              frame's coord. This is what gets transformed and fused into
              the keyframe's canonical pointmap.

  - loop:     in backend FactorGraph.add_factors, (keyframe i, keyframe j)
              symmetric inference. Records both cross-view predictions
              (X_ji in i's coord) and (X_ij in j's coord), per edge.

Per pair we save:
  pair metadata (small):  kind, frame_id_pred, frame_id_target, baseline,
                          view_angle_deg, frame_diff,
                          (loop only) is_consecutive, match_frac_pred, match_frac_target
  per-pixel arrays:       err_mag (fp16), conf (fp16), z_gt (fp16),
                          valid_match (bool, packed),
                          all subsampled by `stride` along (H, W)
  online histogram:       50 log-bins of conf over [1, 50]; per bin
                          (count, sum_err, sum_err2) at fp64. Captures
                          the full distribution without stride loss.

The collector is process-local: one instance for the main (tracking) process,
another for the backend (loop) process. Each writes its own NPZ on flush().
"""

import atexit
import re
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


# 7-Scenes constants (640x480 raw -> 512x384 after MASt3R resize_img(512))
_FX_RAW, _FY_RAW, _CX_RAW, _CY_RAW = 585.0, 585.0, 320.0, 240.0
_RAW_W, _RAW_H = 640, 480
_OUT_W, _OUT_H = 512, 384

# Online histogram: log-spaced conf bins.
_HIST_NBIN = 50
_HIST_LO, _HIST_HI = 1.0, 50.0  # MASt3R conf is typically in [1, ~30+]
_HIST_EDGES = np.logspace(np.log10(_HIST_LO), np.log10(_HIST_HI), _HIST_NBIN + 1)

_FRAME_RE = re.compile(r"frame-(\d+)\.")


# ---------------------------------------------------------------------- #
# Process-local singleton (null collector when disabled)
# ---------------------------------------------------------------------- #

class _NullCollector:
    enabled = False

    def record_tracking_pair(self, *a, **kw):
        pass

    def record_loop_pair(self, *a, **kw):
        pass

    def flush(self):
        pass


_DIAG = _NullCollector()


def get():
    return _DIAG


def init(out_dir, rgb_files, role, stride=2):
    """Initialize this process's diag collector.

    role: 'tracking' or 'loop'. Determines output NPZ filename.
    stride: per-axis pixel subsample for the raw arrays (online hist is full-res).
    """
    global _DIAG
    _DIAG = DiagCollector(out_dir, rgb_files, role, stride=stride)
    atexit.register(_DIAG.flush)


# ---------------------------------------------------------------------- #
# 7-Scenes GT loading helpers
# ---------------------------------------------------------------------- #

def _seven_scenes_frame_num(rgb_path):
    m = _FRAME_RE.search(Path(rgb_path).name)
    return int(m.group(1)) if m else None


def _K_at(h, w):
    """Scale 7-Scenes intrinsics from 640x480 to (h, w)."""
    sx = w / _RAW_W
    sy = h / _RAW_H
    return np.array(
        [[_FX_RAW * sx, 0.0, _CX_RAW * sx],
         [0.0, _FY_RAW * sy, _CY_RAW * sy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _load_pose(rgb_path):
    p = Path(rgb_path)
    pose_path = p.with_name(p.name.replace(".color.png", ".pose.txt"))
    if not pose_path.exists():
        return None
    T = np.loadtxt(pose_path)
    if not np.all(np.isfinite(T)):
        return None
    return T.astype(np.float64)


def _load_depth_and_pts(rgb_path, out_h, out_w, K):
    """Load 7-Scenes depth, resize NEAREST to (out_h, out_w), backproject.

    Returns (pts (out_h, out_w, 3) float32, valid (out_h, out_w) bool).
    Invalid pixels have pts=0 and valid=False.
    """
    p = Path(rgb_path)
    dp = p.with_name(p.name.replace(".color.png", ".depth.png"))
    if not dp.exists():
        return None, None
    d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None, None
    valid_raw = (d > 0) & (d < 65535)
    d_resized = cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    v_resized = cv2.resize(valid_raw.astype(np.uint8), (out_w, out_h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    z = d_resized.astype(np.float64) / 1000.0  # mm -> m
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(out_w, dtype=np.float64)[None, :].repeat(out_h, axis=0)
    v = np.arange(out_h, dtype=np.float64)[:, None].repeat(out_w, axis=1)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    pts[~v_resized] = 0.0
    return pts.astype(np.float32), v_resized


# ---------------------------------------------------------------------- #
# Collector
# ---------------------------------------------------------------------- #

class _LRU(OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.maxsize:
            self.popitem(last=False)


class DiagCollector:
    enabled = True

    def __init__(self, out_dir, rgb_files, role, stride=2):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.rgb_files = [str(p) for p in rgb_files]
        self.stride = max(1, int(stride))
        self._flushed = False

        # GT caches (LRU; depth+pts is heavy)
        self._pose_cache = {}  # dataset_idx -> 4x4
        self._pts_cache = _LRU(maxsize=32)  # dataset_idx -> (pts (H,W,3) fp32, valid (H,W) bool)
        self._K = None
        self._out_HW = None

        # Pair-level metadata (parallel arrays, appended per record)
        self.pair_kind = []
        self.pair_frame_pred = []
        self.pair_frame_target = []
        self.pair_baseline = []
        self.pair_view_angle = []
        self.pair_frame_diff = []
        self.pair_is_consecutive = []  # -1 = N/A (tracking)
        self.pair_match_frac_pred = []  # -1 = N/A
        self.pair_match_frac_target = []
        self.pair_n_pixels = []

        # Per-pixel concatenated chunks
        self._err_chunks = []
        self._conf_chunks = []
        self._zgt_chunks = []
        self._valid_chunks = []

        # Online per-pair histogram: shape (n_pair, NBIN, 3)  [count, sum_err, sum_err2]
        self._hist_chunks = []

    # ------------------------------------------------------------------ #

    def _K_for(self, h, w):
        if self._K is None or self._out_HW != (h, w):
            self._K = _K_at(h, w)
            self._out_HW = (h, w)
        return self._K

    def _pose(self, idx):
        if idx not in self._pose_cache:
            self._pose_cache[idx] = _load_pose(self.rgb_files[idx])
        return self._pose_cache[idx]

    def _gt_pts(self, idx, h, w):
        key = (idx, h, w)
        cached = self._pts_cache.get(key)
        if cached is not None:
            return cached
        K = self._K_for(h, w)
        pts, valid = _load_depth_and_pts(self.rgb_files[idx], h, w, K)
        self._pts_cache.put(key, (pts, valid))
        return pts, valid

    # ------------------------------------------------------------------ #

    def _record_one(
        self,
        X_pred,            # (N,3) torch
        C_pred,            # (N,1) torch
        frame_id_pred,
        frame_id_target,
        kind,
        valid_match=None,  # (N,) bool torch or None
        is_consecutive=-1,
        match_frac_pred=-1.0,
        match_frac_target=-1.0,
    ):
        if X_pred.dim() == 3:
            X_pred = X_pred[0]
        if C_pred.dim() == 3:
            C_pred = C_pred[0]
        if valid_match is not None and valid_match.dim() > 1:
            valid_match = valid_match.reshape(-1)

        N = X_pred.shape[0]
        # MASt3R output resolution. With img_downsample=1 (default), it's 512x384.
        if N == _OUT_H * _OUT_W:
            H, W = _OUT_H, _OUT_W
        else:
            # Fallback: assume square-ish aspect from MASt3R. If this fires, the
            # collector assumption (img_downsample=1) was violated.
            print(f"[diag] unexpected pointmap size N={N}, skipping")
            return

        T_pred = self._pose(frame_id_pred)
        T_target = self._pose(frame_id_target)
        if T_pred is None or T_target is None:
            return

        gt_pts, gt_valid = self._gt_pts(frame_id_target, H, W)
        if gt_pts is None:
            return

        # GT points expressed in pred's camera frame
        T_rel = np.linalg.inv(T_pred) @ T_target  # pred <- target
        R = T_rel[:3, :3]
        t = T_rel[:3, 3]
        gt_flat = gt_pts.reshape(-1, 3).astype(np.float64)
        gt_in_pred = gt_flat @ R.T + t  # (N, 3)
        gt_valid_flat = gt_valid.reshape(-1)

        X_np = X_pred.detach().cpu().numpy().astype(np.float64)
        C_np = C_pred.detach().cpu().numpy().reshape(-1).astype(np.float32)
        err = np.linalg.norm(X_np - gt_in_pred, axis=-1).astype(np.float32)
        z_gt = gt_in_pred[:, 2].astype(np.float32)
        err[~gt_valid_flat] = np.nan
        z_gt[~gt_valid_flat] = np.nan

        if valid_match is None:
            vm_np = np.zeros(N, dtype=bool)
        else:
            vm_np = valid_match.detach().cpu().numpy().reshape(-1).astype(bool)

        # Online histogram: use ALL valid pixels (not stride-subsampled)
        good = np.isfinite(err)
        if good.any():
            err_g = err[good].astype(np.float64)
            conf_g = C_np[good].astype(np.float64)
            bin_idx = np.clip(np.digitize(conf_g, _HIST_EDGES) - 1, 0, _HIST_NBIN - 1)
            hist = np.zeros((_HIST_NBIN, 3), dtype=np.float64)
            np.add.at(hist[:, 0], bin_idx, 1.0)
            np.add.at(hist[:, 1], bin_idx, err_g)
            np.add.at(hist[:, 2], bin_idx, err_g * err_g)
        else:
            hist = np.zeros((_HIST_NBIN, 3), dtype=np.float64)

        # Subsample raw per-pixel arrays
        if self.stride > 1:
            idx = np.arange(N).reshape(H, W)[::self.stride, ::self.stride].reshape(-1)
            err = err[idx]
            C_np = C_np[idx]
            z_gt = z_gt[idx]
            vm_np = vm_np[idx]

        # Baseline & view angle from GT poses
        baseline = float(np.linalg.norm(T_pred[:3, 3] - T_target[:3, 3]))
        R_rel = T_pred[:3, :3].T @ T_target[:3, :3]
        cos_a = (np.trace(R_rel) - 1.0) / 2.0
        cos_a = float(np.clip(cos_a, -1.0, 1.0))
        view_angle = float(np.degrees(np.arccos(cos_a)))

        # Frame diff via 7-Scenes filename frame numbers
        fn_pred = _seven_scenes_frame_num(self.rgb_files[frame_id_pred])
        fn_target = _seven_scenes_frame_num(self.rgb_files[frame_id_target])
        frame_diff = (abs(fn_pred - fn_target)
                      if fn_pred is not None and fn_target is not None else -1)

        self.pair_kind.append(kind)
        self.pair_frame_pred.append(int(frame_id_pred))
        self.pair_frame_target.append(int(frame_id_target))
        self.pair_baseline.append(baseline)
        self.pair_view_angle.append(view_angle)
        self.pair_frame_diff.append(int(frame_diff))
        self.pair_is_consecutive.append(int(is_consecutive))
        self.pair_match_frac_pred.append(float(match_frac_pred))
        self.pair_match_frac_target.append(float(match_frac_target))
        self.pair_n_pixels.append(int(err.shape[0]))

        self._err_chunks.append(err.astype(np.float16))
        self._conf_chunks.append(C_np.astype(np.float16))
        self._zgt_chunks.append(z_gt.astype(np.float16))
        self._valid_chunks.append(vm_np)
        self._hist_chunks.append(hist)

    # ------------------------------------------------------------------ #

    def record_tracking_pair(self, frame_id_f, frame_id_k, Xkf, Ckf, valid_match=None):
        """Asymmetric tracking call. Records keyframe pointmap predicted in frame coord."""
        self._record_one(
            Xkf, Ckf,
            frame_id_pred=frame_id_f,
            frame_id_target=frame_id_k,
            kind="tracking",
            valid_match=valid_match,
        )

    def record_loop_edge(
        self,
        frame_id_i, frame_id_j,
        Xji, Cji, Xij, Cij,
        valid_match_j=None, valid_match_i=None,
        is_consecutive=False,
        match_frac_i=-1.0, match_frac_j=-1.0,
    ):
        """Symmetric loop-closure call. Records both cross-view directions."""
        kind = "loop_consec" if is_consecutive else "loop_lc"
        # j's pointmap predicted in i's coord
        self._record_one(
            Xji, Cji, frame_id_pred=frame_id_i, frame_id_target=frame_id_j,
            kind=kind, valid_match=valid_match_j,
            is_consecutive=int(is_consecutive),
            match_frac_pred=match_frac_i, match_frac_target=match_frac_j,
        )
        # i's pointmap predicted in j's coord
        self._record_one(
            Xij, Cij, frame_id_pred=frame_id_j, frame_id_target=frame_id_i,
            kind=kind, valid_match=valid_match_i,
            is_consecutive=int(is_consecutive),
            match_frac_pred=match_frac_j, match_frac_target=match_frac_i,
        )

    # ------------------------------------------------------------------ #

    def flush(self):
        if self._flushed or not self.pair_n_pixels:
            return
        self._flushed = True
        err = np.concatenate(self._err_chunks) if self._err_chunks else np.zeros(0, np.float16)
        conf = np.concatenate(self._conf_chunks) if self._conf_chunks else np.zeros(0, np.float16)
        zgt = np.concatenate(self._zgt_chunks) if self._zgt_chunks else np.zeros(0, np.float16)
        valid = np.concatenate(self._valid_chunks) if self._valid_chunks else np.zeros(0, bool)
        valid_packed = np.packbits(valid)
        offsets = np.cumsum([0] + self.pair_n_pixels).astype(np.int64)
        hist = np.stack(self._hist_chunks, axis=0) if self._hist_chunks \
            else np.zeros((0, _HIST_NBIN, 3), np.float64)

        path = self.out_dir / f"{self.role}.npz"
        np.savez_compressed(
            path,
            # pair-level
            kind=np.array(self.pair_kind),
            frame_id_pred=np.array(self.pair_frame_pred, np.int32),
            frame_id_target=np.array(self.pair_frame_target, np.int32),
            baseline=np.array(self.pair_baseline, np.float32),
            view_angle_deg=np.array(self.pair_view_angle, np.float32),
            frame_diff=np.array(self.pair_frame_diff, np.int32),
            is_consecutive=np.array(self.pair_is_consecutive, np.int8),
            match_frac_pred=np.array(self.pair_match_frac_pred, np.float32),
            match_frac_target=np.array(self.pair_match_frac_target, np.float32),
            n_pixels=np.array(self.pair_n_pixels, np.int64),
            pair_offset=offsets,
            # per-pixel (concatenated)
            err=err,
            conf=conf,
            z_gt=zgt,
            valid_match_packed=valid_packed,
            valid_match_len=np.int64(valid.size),
            # online histogram
            hist=hist.astype(np.float32),  # downcast to fp32 to save space
            hist_edges=_HIST_EDGES.astype(np.float64),
            # meta
            stride=np.int32(self.stride),
            out_HW=np.array(self._out_HW if self._out_HW else (-1, -1), np.int32),
        )
        n_pair = len(self.pair_n_pixels)
        n_pix = int(err.size)
        size_mb = path.stat().st_size / 1e6
        print(f"[diag] wrote {path}  ({n_pair} pairs, {n_pix:,} pixels, {size_mb:.1f} MB)")
