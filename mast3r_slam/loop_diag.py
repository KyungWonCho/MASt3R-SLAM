"""Lightweight loop-closure inspector.

For every accepted loop edge in `global_opt.FactorGraph.add_factors`,
records a small per-edge summary:
    i, j, is_consecutive, match_frac_i, match_frac_j, Q_mean,
    median / p90 of |X_canon_i - X_ji|  at matched pixels,
    SLAM-estimated keyframe poses T_WC_i, T_WC_j.

Cost: one tensor norm at matched pixels per edge, one D2H of a handful
of scalars. Negligible compared to the symmetric MASt3R decode itself.

A companion analysis script joins these against GT (e.g. TUM
groundtruth.txt) post-hoc to evaluate whether the conf+match_frac
acceptance criterion is producing geometrically consistent loops.
"""

import atexit
from pathlib import Path

import numpy as np
import torch


class _Null:
    enabled = False

    def record(self, *a, **kw):
        pass

    def flush(self):
        pass


_INST = _Null()


def get():
    return _INST


def init(out_dir):
    global _INST
    _INST = LoopInspector(out_dir)
    atexit.register(_INST.flush)


class LoopInspector:
    enabled = True

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self._flushed = False

    @torch.inference_mode()
    def record(
        self,
        i, j,
        X_canon_i,        # (H*W, 3) — keyframe i's stored pointmap (in i coord)
        Xji,              # (H*W, 3) — j's pointmap in i coord (MASt3R sym decode)
        valid_match_j,    # (H*W,) bool — matched pixel mask in i side
        match_frac_i,
        match_frac_j,
        Q_mean,
        is_consecutive,
        T_WC_i,           # 4x4 numpy or torch — i's SLAM-estimated world pose
        T_WC_j,
    ):
        vm = valid_match_j.reshape(-1).bool()
        if vm.sum() == 0:
            med_res = float("nan")
            p90_res = float("nan")
            n_matched = 0
        else:
            kx = X_canon_i.reshape(-1, 3)
            xj = Xji.reshape(-1, 3)
            res = (kx[vm] - xj[vm]).norm(dim=-1).float()
            med_res = float(res.median().item())
            p90_res = float(res.quantile(0.9).item())
            n_matched = int(vm.sum().item())

        def _to_np(T):
            if isinstance(T, torch.Tensor):
                return T.detach().cpu().numpy().astype(np.float64)
            return np.asarray(T, dtype=np.float64)

        self.records.append({
            "i": int(i),
            "j": int(j),
            "is_consecutive": bool(is_consecutive),
            "match_frac_i": float(match_frac_i),
            "match_frac_j": float(match_frac_j),
            "Q_mean": float(Q_mean),
            "n_matched": n_matched,
            "med_residual": med_res,
            "p90_residual": p90_res,
            "t_i": _to_np(T_WC_i)[:3, 3],
            "t_j": _to_np(T_WC_j)[:3, 3],
            "R_i": _to_np(T_WC_i)[:3, :3],
            "R_j": _to_np(T_WC_j)[:3, :3],
        })

    def flush(self):
        if self._flushed or not self.records:
            return
        self._flushed = True
        path = self.out_dir / "loop_inspect.npz"
        # Scalar fields go in as 1D arrays; per-row 3-vector / 3×3 fields as 2D/3D
        out = {}
        for k in ("i", "j"):
            out[k] = np.array([r[k] for r in self.records], dtype=np.int32)
        for k in ("is_consecutive",):
            out[k] = np.array([r[k] for r in self.records], dtype=bool)
        for k in ("match_frac_i", "match_frac_j", "Q_mean",
                  "med_residual", "p90_residual"):
            out[k] = np.array([r[k] for r in self.records], dtype=np.float32)
        out["n_matched"] = np.array(
            [r["n_matched"] for r in self.records], dtype=np.int32
        )
        out["t_i"] = np.stack([r["t_i"] for r in self.records]).astype(np.float64)
        out["t_j"] = np.stack([r["t_j"] for r in self.records]).astype(np.float64)
        out["R_i"] = np.stack([r["R_i"] for r in self.records]).astype(np.float64)
        out["R_j"] = np.stack([r["R_j"] for r in self.records]).astype(np.float64)
        np.savez_compressed(path, **out)
        print(f"[loop_inspect] wrote {path}  ({len(self.records)} edges)")
