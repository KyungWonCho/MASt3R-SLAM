# MASt3R-SLAM Diag — Key Findings

Research notes from instrumenting `mast3r_slam/diag.py` and running on 7-Scenes (calib + no_calib).
Companion to `scripts/analyze_diag.py` output at `logs/diag/plots/summary.md`.
Section references below (`§N`) point to that summary.

---

## Setup

- **What we record**: per-pair per-pixel `(err, conf, z_gt, valid_match)` for two call sites —
  tracking (asymmetric, frame ↔ current keyframe, every frame) and loop closure (symmetric, kf_i ↔ kf_j, per edge).
- **Ground truth**: 7-Scenes depth backprojected with GT poses, expressed in the predicting camera's frame.
- **Scenes covered (calib + no_calib for chess, calib only for the rest)**:
  chess, fire, heads, office, pumpkin, redkitchen, stairs.
- **Aggregate scale**: 3243 tracking pairs (552M valid pixels), 380 loop pairs (63M valid pixels) for calib.

---

## Finding 1 — MASt3R confidence calibration is real but weak, and the current code's implicit interpretation is wrong

**Empirically**: σ ≈ a · c^p with cross-scene aggregate fit
`σ ≈ 0.708 · c^(-0.372)` for tracking (§2). Range across scenes: p ∈ [-0.23, -0.53] — never zero, but never -1.

**What the current code assumes**: `frame.py:74-77` uses

```python
self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
self.C       = self.C + C
```

A standard weighted average with weight = raw `c` implicitly assumes `σ² ∝ 1/c`, i.e. `σ ∝ c^(-0.5)`.
**Measured** is `σ ∝ c^(-0.37)`. The code therefore over-weights high-confidence observations
by a factor of `c^0.13` per update.

**What the calibrated weight should be**: the optimal inverse-variance weight is
`w = 1/σ² ∝ c^(-2p) = c^0.74`.

| Assumption | σ vs c | Implied weight |
|---|---|---|
| Current (`weighted_pointmap`) | σ ∝ c^-0.5 | w = c |
| MASt3R loss intent (1/|err|) | σ ∝ c^-1.0 | w = c² |
| **Empirical fit** | **σ ∝ c^-0.37** | **w = c^0.74** |

Loop calibration is much weaker: `σ ≈ 1.41 · c^(-0.186)` (§2). Confidence-based weighting in loop
factors is correspondingly less informative.

**Per-scene consistency** (§1): all 7 scenes give p in a tight range, so this isn't a scene fluke.
Stairs is an outlier in *conf level* (mean conf 2.8 vs 7–12 elsewhere) but its slope p = -0.53 is
still within the band.

---

## Finding 2 — Fusion freezes; ~30–60% of keyframes are "early-wrong"

**The mechanism**: `self.C += C` grows unbounded. After ~10–20 tracking pairs (typical conf ≈ 8),
the accumulated `C` reaches ~100–300. The Kalman gain for the next observation,
`c_new / (C_acc + c_new)`, then drops below 5%. The keyframe is effectively locked on the
average of its early predictions.

**How often**:

| scene | n_kf | updates_mean | updates_max | early-wrong (slope < -0.05) | never reached C=100 |
|---|---:|---:|---:|---:|---:|
| chess      | 11 | 45 | 91 | **6/11** | 0 |
| stairs     |  8 | 31 | 46 | **5/8**  | 6 |
| fire       | 16 | 31 | 65 | **7/16** | 0 |
| office     | 17 | 29 | 72 | 6/17 | 0 |
| redkitchen | 12 | 42 | 75 | 4/12 | 0 |
| pumpkin    | 13 | 38 | 68 | 3/13 | 1 |
| heads      | 20 | 25 | 86 | 1/20 | 10 |

"early-wrong" = mean err of the last n/4 pairs is at least 0.05 m lower than the mean err of the
first n/4 pairs. **30–60% of keyframes in textured scenes** show this pattern.

**Top examples** (§12):
- `chess kf=0`: err_first 0.659 → err_last 0.236 (≈3× improvement available if not frozen)
- `fire kf=0`: err 0.288 → 0.119, with conf_mean = **16.76** — i.e. very high conf, but early was wrong
- `pumpkin kf=426`: err 0.302 → 0.183, conf_mean 2.94

The **fire kf=0** case is the strongest evidence that conf alone cannot detect early-wrong: the
confidence is high *and* stable across updates, but the actual error halves over the keyframe's
lifetime. Cap+calibration improves the rate at which dilution happens, but does not detect this
disagreement.

**Cap sweet spot** (§11): fraction of updates that would have been "frozen" (`Σc > cap` before
adding) under each cap candidate:

| cap | fraction frozen |
|---:|---:|
| 100  | **53.3 %** (too tight) |
| **200** | (≈ midpoint, recommended) |
| 300  | 16.9 % |
| 1000 | 0.1 % (lax, equivalent to "no cap" in practice) |

---

## Finding 3 — Loop edges: most are not real loops, but real ones are gold

**What "loop" pairs look like** (§13, §6, §7):
- 61 % of loop pairs have view_angle < 2°
- 39 % have baseline < 1 cm
- These are **consecutive-keyframe edges**, not retrieval-discovered loop closures.

**Error vs geometry**:

| view_angle bin | err_median | n_pairs |
|---|---:|---:|
| 0–2°   | 0.983 | 232 |
| 2–5°   | 0.663 | 100 |
| 5–10°  | 0.589 |  42 |
| 10–20° | **0.426** | 6 |

Real loops (`heads`, view_max = 13°) have err_rmse 0.76 m vs tracking 0.48 m — only 1.6× worse.
Compare to chess loop (view_max = 1°, all consecutive): err_rmse 1.62 m, 3.4× tracking.

**Symmetric vs asymmetric decoder**: tracking uses `mast3r_match_asymmetric`,
loops use `mast3r_match_symmetric`. Loop err is 2–4× tracking err even for matched real loops
(heads 1.6×, office 3.6×). It is currently unclear whether this is intrinsic to the symmetric
decode pass or a consequence of the harder pair geometry. **Worth a targeted ablation**: decode
the same pair both ways and compare.

**Unused information**: `global_opt.py` calls `mast3r_match_symmetric(...)` which decodes
`Xji, Xij` (pointmaps in each other's frame). Only `Qij` (joint conf) is currently kept as the
factor weight. **The decoded pointmaps themselves are discarded.** For loop edges with
view_angle > 5°, these are independent observations of keyframe `i` from a very different
viewpoint — i.e., precisely the kind of decorrelated late observation that would mitigate
early-wrong if fused back into `keyframe_i.X_canon`.

---

## Finding 4 — Tracking is more robust to view angle than expected

**Tracking view_angle distribution** (§13): median **10.84°**, p90 28.8°, max 63.2°. Not small.
This is because the keyframe-selection threshold (`match_frac_thresh = 0.333`) is lax enough that
a single keyframe is retained across substantial camera motion — `frame_diff` up to 182 (≈12 s
of motion at the dataset's frame rate).

**Tracking err vs view_angle** (§3): essentially flat. err_median is 0.20 m at 0–2° and
0.23 m at 20–30°. **MASt3R is empirically robust to view angle up to ~60° on textured indoor
scenes**, contrary to the intuition that high view angle would dominate the error budget.

Implication for fusion design: gating by view_angle is *not* warranted in tracking. Conf
calibration captures whatever signal view_angle would have contributed.

**Tracking err vs baseline** (§4): mildly decreasing — 0.27 m at <1 cm baseline → 0.17 m at >1 m.
Larger motion gives slightly better predictions (more parallax, but MASt3R already handles
single-view monocular cues well).

---

## Finding 5 — Calib and no_calib are indistinguishable at the diag level

Confirmed on `chess` (both modes run): tracking err_rmse 0.479 (calib) vs 0.493 (no_calib),
σ fit parameters within ~2 %. This is *expected* because diag measures the raw MASt3R decoder
output, which is intrinsic-independent. Calib/no_calib differences live downstream in the
pose-estimation loop. Implication: we skipped `no_calib` runs for the remaining six scenes
without information loss.

---

## Synthesis — proposed direction

### Phase 1 — Cap + Calibration (`frame.py:74-77`, ~4 lines)

```python
elif filtering_mode == "weighted_pointmap":
    w_new = C ** 0.74                                          # calibrated
    self.X_canon = ((self.C * self.X_canon) + (w_new * X)) / (self.C + w_new)
    self.C = (self.C + w_new).clamp(max=200.0)                 # cap
    self.N += 1
```

Validated by §11 (cap=200 → freeze rate effectively 0) and the fusion simulation
(`logs/diag/plots/fusion_sim.csv`, §13 sim table) against the current `w=c` scheme.

### Phase 2 — Innovation handling (the user's real goal: "fast correction")

Cap+calibration prevents freeze but does **not** detect or correct early-wrong cases. To
recover the gap to oracle_best in early-wrong keyframes, the system needs to *inflate*
σ² (or `self.C`) when a new observation disagrees with the canonical estimate.

Two viable forms:
- **Innovation-inflated KF-lite**: replace `self.C` with `self.sigma2` (same memory). Compute
  per-pixel Mahalanobis-like ratio at each update; inflate σ² when it exceeds a threshold.
- **Adaptive forgetting**: `self.C = λ * self.C + w_new`, with λ adapting down when residual
  is large.

The first is more principled and naturally extends the calibrated noise model. The second is
simpler and may be enough.

This phase is what addresses the user's intent — the per-pixel uncertainty isn't valuable *as
uncertainty*, it's valuable as the substrate for innovation gating.

### Phase 3 — Loop pointmap reuse (orthogonal to Phases 1 & 2)

For loop edges with `view_angle > ~5°` (real parallax), feed `Xji`, `Xij` back into the
respective keyframes' `update_pointmap`. The decode cost is already paid for the matching
step, so this is essentially free additional observations from a maximally decorrelated
viewpoint. Most directly addresses early-wrong by adding observations *outside* the temporal
neighbourhood that the linear monotone fusion was stuck on.

---

## Open questions / things still worth checking

1. **Symmetric vs asymmetric decoder for the same pair** — requires a diag.py extension that
   decodes a sample of loop pairs both ways. Would settle whether loop's 2–4× err is intrinsic
   to symmetric mode or a geometry artefact.
2. **Spatial structure of err** — is per-pixel err concentrated near image boundaries or depth
   discontinuities? If so, conf as a single per-pixel scalar can't capture it; learned spatial
   priors would be a separate research angle.
3. **Match validity correlation** — `valid_match` is stored but unused in the current
   analysis. Are matched pixels systematically more accurate? If yes, fusion should weight by
   `c · valid_match`, not `c`.
4. **Relative (err / z_gt)** — does the σ ≈ c^-0.37 calibration hold when error is normalised
   by GT depth? Could explain part of the per-scene a-coefficient variation.
5. **Match frac as a loop gate** — `min_match_frac = 0.1` is currently the only loop gate.
   Combined gates on (match_frac, view_angle, baseline) might do better than any one alone.
6. **Train-vs-test sequence effect** — were the 7-Scenes runs on sequences MASt3R has trained
   on? If so, conf may be biased high there.
