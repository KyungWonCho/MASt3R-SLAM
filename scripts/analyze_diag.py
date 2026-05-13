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
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Calibrated noise-model assumption (from cross-scene σ(c) fit, see summary 2)
# σ ≈ a · c^p_meas with p_meas ≈ -0.3.
# If we treat σ² as the precision-1, the optimal inverse-variance weight is
# w ∝ 1/σ² = c^(-2 p_meas) ≈ c^0.6.  We use this in the simulated reweighting.
P_MEAS = -0.3
W_EXP = -2.0 * P_MEAS   # ≈ 0.6  (this gets overwritten in main with the actual fit)
C_CAP_CANDIDATES = (100.0, 300.0, 1000.0)  # accumulated-C caps to simulate

# Defaults for the fusion simulation (overridden by the actual aggregate fit in main).
SIM_A = 0.708
SIM_P = -0.372
SIM_W_EXP = -2.0 * SIM_P     # ≈ 0.744; calibrated optimal weight exponent
SIM_CAP = 200.0              # accumulated-W cap; floor on sigma2 ≈ inverse


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
            "frame_id_pred": int(d["frame_id_pred"][i]),
            "frame_id_target": int(d["frame_id_target"][i]),
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
# Per-keyframe (fusion-target) analyses
# ---------------------------------------------------------------------- #

def per_keyframe_table(per_pair, role="tracking"):
    """Group tracking pairs by (scene, frame_id_target).
       Time-order them by frame_id_pred and compute fusion-relevant stats:
       - n_updates                  number of pairs that hit this keyframe
       - cum_C                      cumulative sum of conf_median (proxy for self.C)
       - k_freeze_100/300/1000      pair index where cum_C first crosses each cap
       - err_first, err_last        median-err of first / last pair
       - err_trend_slope            sign of (err_late - err_early) / n
                                    > 0 → late worse  (early wins)
                                    < 0 → late better (early-wrong; current code freezes late)
       - err_disagreement_std       std of per-pair err_median across this kf
       - effective_N_uncapped       N (geometric mean of relative weight = 1/N effectively)
       - effective_N_w_exp          same but with w = c^W_EXP instead of c
    """
    by_kf = defaultdict(list)
    for r in per_pair:
        if r["role"] != role:
            continue
        by_kf[(r["scene"], r["frame_id_target"])].append(r)

    out = []
    for (scene, kf), pairs in by_kf.items():
        pairs.sort(key=lambda r: r["frame_id_pred"])
        confs = np.array([p["conf_median"] for p in pairs], dtype=np.float64)
        errs = np.array([p["err_median"] for p in pairs], dtype=np.float64)
        bls = np.array([p["baseline"] for p in pairs], dtype=np.float64)
        vas = np.array([p["view_angle_deg"] for p in pairs], dtype=np.float64)

        n = len(pairs)
        cum_C = np.cumsum(confs)
        cum_W = np.cumsum(np.maximum(confs, 1e-6) ** W_EXP)

        k_freeze = {}
        for cap in C_CAP_CANDIDATES:
            idx = int(np.searchsorted(cum_C, cap))
            k_freeze[cap] = idx if idx < n else -1  # -1 = never reached

        # Trend: simple sign of (mean of last quartile) - (mean of first quartile)
        if n >= 4 and np.all(np.isfinite(errs)):
            q = max(1, n // 4)
            slope = float(errs[-q:].mean() - errs[:q].mean())
        else:
            slope = float("nan")

        out.append({
            "scene": scene,
            "frame_id_target": int(kf),
            "n_updates": n,
            "conf_mean_obs": float(np.nanmean(confs)),
            "conf_max_obs": float(np.nanmax(confs)),
            "cum_C_final": float(cum_C[-1]),
            "cum_W_final": float(cum_W[-1]),
            "k_freeze_100": k_freeze[100.0],
            "k_freeze_300": k_freeze[300.0],
            "k_freeze_1000": k_freeze[1000.0],
            "err_first": float(errs[0]) if n > 0 else float("nan"),
            "err_last": float(errs[-1]) if n > 0 else float("nan"),
            "err_min": float(np.nanmin(errs)) if n > 0 else float("nan"),
            "err_max": float(np.nanmax(errs)) if n > 0 else float("nan"),
            "err_mean": float(np.nanmean(errs)) if n > 0 else float("nan"),
            "err_std": float(np.nanstd(errs)) if n > 1 else 0.0,
            "err_trend_late_minus_early": slope,
            "baseline_max": float(np.nanmax(bls)) if n > 0 else float("nan"),
            "view_angle_max": float(np.nanmax(vas)) if n > 0 else float("nan"),
        })
    return out


def plot_kf_err_trajectories(per_pair, out_path, role="tracking", n_plot=12):
    """Per-keyframe err vs pair index. Picks the n_plot keyframes with most pairs."""
    by_kf = defaultdict(list)
    for r in per_pair:
        if r["role"] != role:
            continue
        by_kf[(r["scene"], r["frame_id_target"])].append(r)
    if not by_kf:
        return
    items = sorted(by_kf.items(), key=lambda kv: -len(kv[1]))[:n_plot]
    ncols = 3
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows),
                             squeeze=False)
    for ax, ((scene, kf), pairs) in zip(axes.flat, items):
        pairs.sort(key=lambda r: r["frame_id_pred"])
        errs = np.array([p["err_median"] for p in pairs])
        confs = np.array([p["conf_median"] for p in pairs])
        x = np.arange(len(pairs))
        ax.plot(x, errs, "o-", color="tab:blue", label="err_median (m)")
        ax2 = ax.twinx()
        ax2.plot(x, confs, "s--", color="tab:orange", alpha=0.6, label="conf_median")
        ax.set_title(f"{scene} kf={kf}  N={len(pairs)}", fontsize=9)
        ax.set_xlabel("pair index (time-ordered)")
        ax.set_ylabel("err", color="tab:blue")
        ax2.set_ylabel("conf", color="tab:orange")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.suptitle(f"Per-keyframe err & conf trajectories ({role})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Fusion simulation: compare schemes per-pixel per-keyframe.
#
# All schemes operate on a (n_pairs, n_pix) per-keyframe sequence of
# observations (err_obs_magnitude, conf). For each scheme we propagate
# E[|err_canon|^2] analytically per pixel assuming zero-mean independent
# observation errors (i.e., we're computing the variance of the running
# fused estimate, with the observed err magnitude as the per-pair std).
# This isn't a Monte Carlo of signed directions, but it gives the
# correct ranking under the usual zero-mean noise assumption.
#
# Innovation gating is NOT modelled here (requires directional MC). The
# four deterministic schemes still cleanly separate (i) miscalibrated
# linear monotone, (ii) recalibrated linear monotone, (iii) recalibrated
# + cap, (iv) per-pixel KF-lite with the calibrated noise model.
# ---------------------------------------------------------------------- #

def _fuse_linear_monotone(c, err_obs, w_of_c):
    """Linear weighted average, monotone-growing W. Returns E[|err_canon|^2] (n_pix,).
       c, err_obs: (n_pairs, n_pix) float arrays.  w_of_c: callable c->weight."""
    n_pairs = c.shape[0]
    w = w_of_c(c)
    mean_err2 = err_obs[0].astype(np.float64) ** 2
    W = w[0].astype(np.float64)
    for i in range(1, n_pairs):
        w_i = w[i].astype(np.float64)
        e2 = err_obs[i].astype(np.float64) ** 2
        denom = W + w_i
        a_old = W / denom
        a_new = w_i / denom
        mean_err2 = a_old ** 2 * mean_err2 + a_new ** 2 * e2
        W = W + w_i
    return mean_err2


def _fuse_linear_capped(c, err_obs, w_of_c, cap):
    """Linear weighted average with cap on accumulated W. (Phase 1 candidate.)"""
    n_pairs = c.shape[0]
    w = w_of_c(c)
    mean_err2 = err_obs[0].astype(np.float64) ** 2
    W = np.minimum(w[0].astype(np.float64), cap)
    for i in range(1, n_pairs):
        w_i = w[i].astype(np.float64)
        e2 = err_obs[i].astype(np.float64) ** 2
        denom = W + w_i
        a_old = W / denom
        a_new = w_i / denom
        mean_err2 = a_old ** 2 * mean_err2 + a_new ** 2 * e2
        W = np.minimum(W + w_i, cap)
    return mean_err2


def _fuse_kf_lite(c, err_obs, a_meas, p_meas, sigma2_floor=None):
    """Per-pixel Kalman with calibrated noise model σ²(c) = (a · c^p)².
       (Phase C candidate.) Each pixel keeps its own σ², updated proper KF style.
       Optional floor on σ² acts like the cap on W in the linear form."""
    n_pairs = c.shape[0]
    sigma2_obs = (a_meas * c.astype(np.float64) ** p_meas) ** 2
    sigma2_canon = sigma2_obs[0].copy()
    mean_err2 = err_obs[0].astype(np.float64) ** 2
    for i in range(1, n_pairs):
        s2_obs = sigma2_obs[i]
        K = sigma2_canon / (sigma2_canon + s2_obs)
        e2 = err_obs[i].astype(np.float64) ** 2
        mean_err2 = (1 - K) ** 2 * mean_err2 + K ** 2 * e2
        sigma2_canon = (1 - K) * sigma2_canon
        if sigma2_floor is not None:
            sigma2_canon = np.maximum(sigma2_canon, sigma2_floor)
    return mean_err2


def _fuse_kf_innov(c, err_obs, a_meas, p_meas,
                   mahala_thresh=4.0, inflation=4.0, sigma2_floor=None):
    """KF-lite + innovation-based variance inflation: when the per-pair
       innovation looks too large for the current (σ²_canon + σ²_obs),
       inflate σ²_canon → next obs immediately gets larger Kalman gain.
       This is the mechanism that enables fast correction of early-wrong.

       We use a proxy for the per-pixel innovation magnitude:
            innov² ≈ E[|err_canon|²] + E[|err_obs|²]
       (i.e. the expected squared distance under the zero-mean assumption,
       which is the right scale even when we don't know signed direction).
       The Mahalanobis-like test then asks whether the observation's
       expected residual variance dwarfs the current canonical's variance.
    """
    n_pairs = c.shape[0]
    sigma2_obs = (a_meas * c.astype(np.float64) ** p_meas) ** 2
    sigma2_canon = sigma2_obs[0].copy()
    mean_err2 = err_obs[0].astype(np.float64) ** 2
    for i in range(1, n_pairs):
        s2_obs = sigma2_obs[i]
        e2 = err_obs[i].astype(np.float64) ** 2
        # Proxy innovation magnitude² ≈ mean_err2 + e2 (independent zero-mean)
        innov2_proxy = mean_err2 + e2
        denom = sigma2_canon + s2_obs
        mahala2 = innov2_proxy / np.maximum(denom, 1e-12)
        # Inflate σ²_canon where mahala exceeds threshold
        inflate_mask = mahala2 > mahala_thresh
        sigma2_canon = np.where(inflate_mask, sigma2_canon * inflation, sigma2_canon)
        # Now run the standard KF step with (possibly inflated) sigma2_canon
        K = sigma2_canon / (sigma2_canon + s2_obs)
        mean_err2 = (1 - K) ** 2 * mean_err2 + K ** 2 * e2
        sigma2_canon = (1 - K) * sigma2_canon
        if sigma2_floor is not None:
            sigma2_canon = np.maximum(sigma2_canon, sigma2_floor)
    return mean_err2


def simulate_fusion_for_scene(scene, root, a_meas, p_meas, cap):
    """Load tracking NPZ for scene, run per-pixel fusion sim per keyframe.
       Returns list of dicts (one per (scene, kf, scheme))."""
    npz_path = root / scene / "calib" / "tracking.npz"
    d = load_npz(npz_path)
    if d is None:
        return []
    err_all = d["err"]
    conf_all = d["conf"]
    offs = d["pair_offset"]
    n_pix_per_pair = d["n_pixels"]
    targets = d["frame_id_target"]
    preds = d["frame_id_pred"]

    by_kf = defaultdict(list)
    for i in range(len(n_pix_per_pair)):
        by_kf[int(targets[i])].append(i)

    w_raw = lambda c: c.astype(np.float64)
    w_exp = -2.0 * p_meas
    w_calib = lambda c: np.maximum(c.astype(np.float64), 1e-6) ** w_exp
    # sigma2 floor matched to the cap: floor ≈ (a · c_typ^p)² · (c_typ^w_exp / cap)
    # — handwavy correspondence; we just pass cap-equivalent as sigma2_floor candidate
    sigma2_floor = None  # leave KF-lite without floor first; user can add later

    results = []
    for kf, pair_idx in by_kf.items():
        if len(pair_idx) < 3:
            continue
        pair_idx.sort(key=lambda i: int(preds[i]))
        n_pairs = len(pair_idx)
        n_pix_expected = int(n_pix_per_pair[pair_idx[0]])
        if any(int(n_pix_per_pair[i]) != n_pix_expected for i in pair_idx):
            continue  # rare: pair size mismatch

        err_stack = np.zeros((n_pairs, n_pix_expected), dtype=np.float32)
        conf_stack = np.zeros((n_pairs, n_pix_expected), dtype=np.float32)
        for k, i in enumerate(pair_idx):
            a, b = int(offs[i]), int(offs[i + 1])
            err_stack[k] = err_all[a:b]
            conf_stack[k] = conf_all[a:b]

        # Only consider pixels valid (finite) in every pair of this keyframe
        valid_all = np.all(
            np.isfinite(err_stack) & np.isfinite(conf_stack) & (conf_stack > 0),
            axis=0,
        )
        n_v = int(valid_all.sum())
        if n_v < 100:
            continue
        ev = err_stack[:, valid_all]
        cv = conf_stack[:, valid_all]

        sims = {}
        # (0) Trivial baselines
        sims["first_only"] = ev[0].astype(np.float64) ** 2
        sims["last_only"] = ev[-1].astype(np.float64) ** 2
        sims["oracle_best"] = np.min(ev.astype(np.float64) ** 2, axis=0)
        # (i) Current code: w = c, monotone
        sims["current_w=c"] = _fuse_linear_monotone(cv, ev, w_raw)
        # (ii) Calibration only (no cap): w = c^0.74
        sims["calib_w=c^0.74"] = _fuse_linear_monotone(cv, ev, w_calib)
        # (iii) Calibration + cap (Phase 1 proposal)
        sims[f"calib+cap_{int(cap)}"] = _fuse_linear_capped(cv, ev, w_calib, cap)
        # (iv) KF-lite with calibrated noise model (Phase C without floor)
        sims["kf_lite"] = _fuse_kf_lite(cv, ev, a_meas, p_meas, sigma2_floor=None)
        # (v) Phase 2 candidate: KF-lite + innovation inflation.
        #     This is the only scheme that can correct early-wrong "quickly".
        sims["kf_lite+innov_inflate"] = _fuse_kf_innov(
            cv, ev, a_meas, p_meas,
            mahala_thresh=4.0, inflation=4.0, sigma2_floor=None,
        )

        for name, m2 in sims.items():
            results.append({
                "scene": scene,
                "kf": kf,
                "n_pairs": n_pairs,
                "n_pix": n_v,
                "scheme": name,
                "fused_rmse": float(np.sqrt(m2.mean())),
                "fused_median": float(np.sqrt(np.median(m2))),
                "fused_p90": float(np.sqrt(np.percentile(m2, 90))),
            })
    return results


def plot_scheme_comparison(sim_rows, out_path):
    """Per-scene boxplot of fused_rmse, one box per scheme."""
    by_scheme = defaultdict(list)
    by_scene_scheme = defaultdict(lambda: defaultdict(list))
    for r in sim_rows:
        by_scheme[r["scheme"]].append(r["fused_rmse"])
        by_scene_scheme[r["scene"]][r["scheme"]].append(r["fused_rmse"])
    schemes = list(by_scheme.keys())
    scenes = sorted(by_scene_scheme.keys())

    fig, axes = plt.subplots(1, len(scenes), figsize=(3.0 * len(scenes), 4.5),
                             squeeze=False, sharey=True)
    for ax, scene in zip(axes[0], scenes):
        data = [by_scene_scheme[scene][s] for s in schemes]
        ax.boxplot(data, tick_labels=[s.replace("calib+", "c+\n") for s in schemes])
        ax.set_xticklabels(
            [s.replace("calib+", "c+\n").replace("kf_lite", "kf-\nlite")
             .replace("oracle_best", "oracle\nbest").replace("first_only", "first")
             .replace("last_only", "last").replace("current_w=c", "cur")
             .replace("calib_w=c^0.74", "calib") for s in schemes],
            rotation=0, fontsize=8,
        )
        ax.set_title(scene)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("per-kf fused err RMSE (m)")
    fig.suptitle("Fusion scheme comparison — per-keyframe fused err (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_sigma_with_refs(c_mid, rmse, count, a_meas, p_meas, out_path, title):
    """σ(c) on log-log with reference lines for c^-0.5 (current Kalman assumption),
       c^-1 (Gaussian std), and c^p_meas (the fit)."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    ax.loglog(c_mid, rmse, "o-", color="tab:blue",
              label=fr"measured  $\sigma$ (RMSE per bin)")
    # Reference lines anchored at the geometric-mean c
    finite = np.isfinite(rmse) & np.isfinite(c_mid) & (c_mid > 0)
    if finite.any():
        c_anchor = np.exp(np.mean(np.log(c_mid[finite])))
        sig_anchor = np.exp(np.mean(np.log(rmse[finite])))
        c_grid = np.geomspace(c_mid[finite].min(), c_mid[finite].max(), 50)
        for p_ref, label in [
            (-0.5, "current Kalman:  w∝c  ⇒  σ∝c^-0.5"),
            (-1.0, "Gaussian std:  conf=1/σ  ⇒  σ∝c^-1"),
            (p_meas, f"fit:  σ∝c^{p_meas:.2f}"),
        ]:
            ref = sig_anchor * (c_grid / c_anchor) ** p_ref
            ax.loglog(c_grid, ref, "--", alpha=0.6, label=label)
    ax.set_xlabel("confidence c (log)")
    ax.set_ylabel("σ  (m, log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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

    # ------------------------------------------------------------------ #
    # Fusion simulation across schemes (uses cross-scene a, p from §2)
    # ------------------------------------------------------------------ #
    sim_rows = []
    if "tracking" in agg_sigma and np.isfinite(agg_sigma["tracking"]["p"]):
        a_used = float(agg_sigma["tracking"]["a"])
        p_used = float(agg_sigma["tracking"]["p"])
        cap_used = SIM_CAP
        print(f"\nRunning fusion simulation with a={a_used:.3g} p={p_used:.3f} cap={cap_used}")
        for scene in scenes:
            scene_dir = root / scene / "calib"
            if not scene_dir.exists():
                continue
            sim_rows.extend(simulate_fusion_for_scene(
                scene, root, a_used, p_used, cap_used,
            ))
        if sim_rows:
            # Dump CSV
            sim_csv = out_dir / "fusion_sim.csv"
            keys = list(sim_rows[0].keys())
            with open(sim_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in sim_rows:
                    w.writerow(r)
            plot_scheme_comparison(sim_rows, out_dir / "fusion_scheme_boxplot.png")

    # ------------------------------------------------------------------ #
    # NEW: fusion-focused analyses (cap+calibration evidence)
    # ------------------------------------------------------------------ #

    # 7f-pre. σ(c) with reference lines (Kalman c^-0.5, Gaussian c^-1, fit c^p)
    if "tracking" in agg_sigma and hist_edges_ref is not None:
        H = np.concatenate(hist_agg["tracking"], axis=0)
        total = H.sum(axis=0)
        c_mid, count, mean, rmse, std = calibration_from_hist(total, hist_edges_ref)
        plot_sigma_with_refs(
            c_mid, rmse, count,
            agg_sigma["tracking"]["a"], agg_sigma["tracking"]["p"],
            out_dir / "aggregate_tracking_sigma_with_refs.png",
            "Aggregate tracking σ(c) — measured vs reference exponents",
        )

    # Per-keyframe table (tracking only; loop has very few pairs per kf)
    kf_rows = per_keyframe_table(per_pair, role="tracking")
    if kf_rows:
        # CSV
        kf_csv = out_dir / "per_keyframe.csv"
        keys = list(kf_rows[0].keys())
        with open(kf_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in kf_rows:
                w.writerow(r)
        plot_kf_err_trajectories(
            per_pair, out_dir / "kf_err_trajectories.png", role="tracking", n_plot=12,
        )

    # 7g. Per-keyframe summary: how many updates each keyframe gets, freeze, trend
    md.append("\n## 9. Per-keyframe fusion stats (tracking only)\n")
    md.append("Each row = one keyframe (frame_id_target). `k_freeze_X` = pair index at which "
              "accumulated `Σ c` first exceeds X (i.e., when gain from a new obs becomes < c/X). "
              "`err_trend` = `mean(err of last n/4 pairs) - mean(first n/4)`. "
              "**negative ⇒ late observations are BETTER than early ones** "
              "→ \"early-wrong\" candidate where current code's freeze hurts.\n")

    # Scene-level aggregated KF stats
    by_scene = defaultdict(list)
    for r in kf_rows:
        by_scene[r["scene"]].append(r)
    rows = []
    for scene in sorted(by_scene.keys()):
        rs = by_scene[scene]
        n_kf = len(rs)
        n_updates = np.array([r["n_updates"] for r in rs])
        cum_C = np.array([r["cum_C_final"] for r in rs])
        k100 = np.array([r["k_freeze_100"] for r in rs])
        k300 = np.array([r["k_freeze_300"] for r in rs])
        k1000 = np.array([r["k_freeze_1000"] for r in rs])
        slopes = np.array([r["err_trend_late_minus_early"] for r in rs])
        slopes = slopes[np.isfinite(slopes)]
        # "early-wrong" candidates: late_err < early_err - small_margin
        n_early_wrong = int((slopes < -0.05).sum())
        # never-frozen at 100 = pair count is fewer than steps needed to reach 100
        n_never100 = int((k100 == -1).sum())
        rows.append([
            scene,
            n_kf,
            f"{n_updates.mean():.1f}",
            f"{int(np.median(n_updates))}",
            int(n_updates.max()),
            f"{cum_C.mean():.0f}",
            f"{int(np.median(k100)) if (k100 != -1).any() else 'n/a'}",
            f"{int(np.median(k300)) if (k300 != -1).any() else 'n/a'}",
            f"{int(np.median(k1000)) if (k1000 != -1).any() else 'n/a'}",
            f"{n_early_wrong}/{len(slopes)}" if len(slopes) else "0/0",
            f"{n_never100}/{n_kf}",
        ])
    md.append(md_table(
        ["scene", "n_kf", "updates_mean", "updates_med", "updates_max",
         "cum_C_mean", "med_k_to_100", "med_k_to_300", "med_k_to_1000",
         "early_wrong (slope<-0.05)", "never_hit_C=100"],
        rows,
        aligns=["---"] + ["---:"] * 10,
    ))

    # 7h. Effective N under current (w=c) vs calibrated (w=c^W_EXP)
    md.append("\n## 10. Effective fusion weight: raw c  vs  c^{:.2f}  (calibrated)\n".format(W_EXP))
    md.append("For each keyframe, compare accumulated weight under the current code "
              "`Σ c` versus the calibrated weight `Σ c^{:.2f}`. The calibrated form "
              "compresses the dynamic range and makes high-conf observations less dominant "
              "(consistent with the measured σ∝c^{:.2f}).\n".format(W_EXP, P_MEAS))
    rows = []
    for scene in sorted(by_scene.keys()):
        rs = by_scene[scene]
        cum_C = np.array([r["cum_C_final"] for r in rs])
        cum_W = np.array([r["cum_W_final"] for r in rs])
        rows.append([
            scene,
            f"{cum_C.mean():.1f}",
            f"{cum_W.mean():.1f}",
            f"{(cum_C / np.maximum(cum_W, 1e-6)).mean():.2f}",
        ])
    md.append(md_table(
        ["scene", "Σc (mean)", "Σc^p (mean)", "ratio Σc/Σc^p"],
        rows,
        aligns=["---"] + ["---:"] * 3,
    ))

    # 7i. Cap candidates — what fraction of updates are "wasted" past each cap
    md.append("\n## 11. Cap candidates (tracking)\n")
    md.append("For each cap, fraction of updates where current accumulated Σ c already "
              "exceeds the cap (i.e., updates whose gain would have been < c/cap, "
              "and that the cap would now *preserve* the influence of).\n")
    # Build per-update Σc trajectory and count fraction above each cap
    sums_by_scene = defaultdict(list)
    for r in kf_rows:
        # need to recompute trajectory per kf — use n_updates and cum_C_final & k_freeze_*
        # simpler: count how many updates would exceed each cap globally
        pass
    # Instead: per-update test by re-walking per_pair grouped by kf
    by_kf = defaultdict(list)
    for r in per_pair:
        if r["role"] != "tracking":
            continue
        by_kf[(r["scene"], r["frame_id_target"])].append(r)
    cap_rows = []
    for cap in C_CAP_CANDIDATES:
        total_updates = 0
        above_cap = 0
        for key, pairs in by_kf.items():
            pairs_sorted = sorted(pairs, key=lambda r: r["frame_id_pred"])
            confs = np.array([p["conf_median"] for p in pairs_sorted])
            cum = np.cumsum(confs)
            # 'pre' Σc at each step (before adding the current obs)
            pre = np.concatenate([[0.0], cum[:-1]])
            total_updates += len(pre)
            above_cap += int((pre > cap).sum())
        frac = above_cap / max(total_updates, 1)
        cap_rows.append([f"{cap:.0f}", total_updates, above_cap, f"{frac*100:.1f}%"])
    md.append(md_table(
        ["cap", "total_updates", "updates_with_Σc>cap_already",
         "fraction_frozen"],
        cap_rows,
        aligns=["---:", "---:", "---:", "---:"],
    ))

    # 7j. Top "early-wrong" keyframes (negative err trend, decent N)
    md.append("\n## 12. Top early-wrong keyframes (late obs better than early)\n")
    md.append("Top 12 keyframes (across scenes) with the most negative err trend slope and "
              "n_updates ≥ 5. Indicates cases where the current monotone accumulator "
              "is locked on an early prediction that later observations would correct.\n")
    cand = [r for r in kf_rows
            if r["n_updates"] >= 5 and np.isfinite(r["err_trend_late_minus_early"])]
    cand.sort(key=lambda r: r["err_trend_late_minus_early"])
    rows = []
    for r in cand[:12]:
        rows.append([
            r["scene"], r["frame_id_target"], r["n_updates"],
            f"{r['err_first']:.3f}", f"{r['err_last']:.3f}",
            f"{r['err_trend_late_minus_early']:.3f}",
            f"{r['err_std']:.3f}",
            f"{r['conf_mean_obs']:.2f}",
        ])
    md.append(md_table(
        ["scene", "kf", "N", "err_first", "err_last", "slope",
         "err_std", "conf_mean"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    ))

    # 7k. Fusion simulation comparison
    if sim_rows:
        md.append("\n## 13. Fusion simulation: per-keyframe fused err under different schemes\n")
        md.append("For each tracking keyframe, we simulate per-pixel fusion under each scheme "
                  "and report the resulting fused err RMSE (averaged across pixels of that keyframe). "
                  "**lower is better**. `oracle_best` = pick the single best pair per pixel "
                  "(upper bound on what any combination scheme could achieve from these obs).\n\n"
                  f"Parameters used: a={agg_sigma['tracking']['a']:.3g}, "
                  f"p={agg_sigma['tracking']['p']:.3f}, "
                  f"calibrated weight = c^{-2*agg_sigma['tracking']['p']:.3f}, cap={SIM_CAP:.0f}.\n")

        # Table: per-scene aggregate per scheme
        by_scene_scheme = defaultdict(lambda: defaultdict(list))
        for r in sim_rows:
            by_scene_scheme[r["scene"]][r["scheme"]].append(r["fused_rmse"])
        schemes_order = ["oracle_best", "first_only", "last_only",
                         "current_w=c", "calib_w=c^0.74",
                         f"calib+cap_{int(SIM_CAP)}",
                         "kf_lite", "kf_lite+innov_inflate"]
        headers = ["scene"] + [s.replace("calib+cap_", "cap+\n") for s in schemes_order]
        rows = []
        for scene in sorted(by_scene_scheme.keys()):
            row = [scene]
            for s in schemes_order:
                vals = by_scene_scheme[scene].get(s, [])
                if not vals:
                    row.append("—")
                else:
                    row.append(f"{np.mean(vals):.3f}")
            rows.append(row)
        md.append(md_table(headers, rows,
                           aligns=["---"] + ["---:"] * len(schemes_order)))

        # Aggregate (all scenes pooled)
        md.append("\n**All scenes pooled (mean fused_rmse over all keyframes):**\n")
        agg_rows = []
        for s in schemes_order:
            vals = [r["fused_rmse"] for r in sim_rows if r["scheme"] == s]
            if not vals:
                continue
            agg_rows.append([
                s,
                len(vals),
                f"{np.mean(vals):.4f}",
                f"{np.median(vals):.4f}",
                f"{np.percentile(vals, 90):.4f}",
            ])
        md.append(md_table(
            ["scheme", "n_kf", "mean_fused_rmse", "median", "p90"],
            agg_rows,
            aligns=["---"] + ["---:"] * 4,
        ))

        # Targeted: only "early-wrong" keyframes
        ew_keys = {(r["scene"], r["frame_id_target"])
                   for r in kf_rows
                   if r["n_updates"] >= 5
                   and np.isfinite(r["err_trend_late_minus_early"])
                   and r["err_trend_late_minus_early"] < -0.05}
        md.append("\n**Restricted to early-wrong keyframes only "
                  f"(N={len(ew_keys)}, slope<-0.05, n_updates≥5):**\n")
        ew_rows = []
        for s in schemes_order:
            vals = [r["fused_rmse"] for r in sim_rows
                    if r["scheme"] == s and (r["scene"], r["kf"]) in ew_keys]
            if not vals:
                continue
            ew_rows.append([
                s,
                len(vals),
                f"{np.mean(vals):.4f}",
                f"{np.median(vals):.4f}",
                f"{np.percentile(vals, 90):.4f}",
            ])
        md.append(md_table(
            ["scheme", "n_kf", "mean_fused_rmse", "median", "p90"],
            ew_rows,
            aligns=["---"] + ["---:"] * 4,
        ))

    # Existing 7f (renumbered 14)
    md.append("\n## 14. View angle distribution (sanity check — \"tracking view angles small?\")\n")
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
