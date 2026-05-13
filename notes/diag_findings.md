# MASt3R-SLAM Diag — 핵심 관찰 정리

`mast3r_slam/diag.py`를 새로 추가해서 7-Scenes (calib + no_calib) 데이터를 수집하고, `scripts/analyze_diag.py`로 분석한 결과 요약. 자세한 표는 `logs/diag/plots/summary.md`에 있고, 여기 본문의 `§N` 표시는 그쪽 섹션 번호.

---

## 셋업

- **무엇을 기록하는가**: 두 군데 호출 사이트에서 per-pair per-pixel `(err, conf, z_gt, valid_match)` 를 GT 대비로 저장.
  - **tracking** — asymmetric, frame ↔ 현재 keyframe, 매 frame
  - **loop closure** — symmetric, kf_i ↔ kf_j, edge별
- **Ground truth**: 7-Scenes의 depth를 GT pose로 backproject해서 predicting camera 좌표계에 표현
- **돌린 scene**: chess는 calib + no_calib, 나머지(fire, heads, office, pumpkin, redkitchen, stairs)는 calib만
- **전체 규모**: tracking 3243 pair (552M valid pixel), loop 380 pair (63M valid pixel)

---

## 관찰 1 — MASt3R confidence는 약하게 calibrated, 현재 코드의 암묵 해석은 어긋남

**측정값**: σ ≈ a · c^p, 모든 scene 합쳐서 fit하면
`σ ≈ 0.708 · c^(-0.372)` (tracking, §2). scene별로는 p ∈ [-0.23, -0.53] 범위. 절대 0은 아니지만, 절대 -1도 아님.

**현재 코드의 암묵 가정**: `frame.py:74-77`의 `weighted_pointmap`은

```python
self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
self.C       = self.C + C
```

weight를 raw `c`로 쓴다는 건 암묵적으로 **`σ² ∝ 1/c`, 즉 `σ ∝ c^(-0.5)` 가정**. 측정은 `σ ∝ c^(-0.37)`. 코드가 high-conf 관측을 매 업데이트마다 `c^0.13` 배 만큼 과신한다는 뜻.

**Calibrated optimal weight**: inverse-variance weight `w = 1/σ² ∝ c^(-2p) = c^0.74`.

| 가정 | σ vs c | 의미상 weight |
|---|---|---|
| 현재 코드 (`weighted_pointmap`) | σ ∝ c^-0.5 | w = c |
| MASt3R loss 의도 (1/\|err\|) | σ ∝ c^-1.0 | w = c² |
| **실측 fit** | **σ ∝ c^-0.37** | **w = c^0.74** |

Loop는 calibration이 훨씬 약함: `σ ≈ 1.41 · c^(-0.186)` (§2). loop factor에서 confidence 기반 weight는 그만큼 정보량이 적음.

**Scene별 일관성** (§1): 7 scene 모두 p 값이 좁은 범위에 들어와서 scene-fluke 아님. stairs는 *conf 절대값* 자체가 낮음 (mean 2.8, 다른 scene은 7~12) — 텍스처 빈약한 계단이라 MASt3R 자신감 자체가 떨어지는 case지만, slope p = -0.53은 여전히 비슷한 밴드.

---

## 관찰 2 — Fusion은 freeze됨, 30~60% 의 keyframe이 "early-wrong"

**메커니즘**: `self.C += C` 가 cap 없이 자라남. tracking pair ~10~20개 누적되면 (typical conf ≈ 8) `C_acc ≈ 100~300`에 도달. 다음 관측의 Kalman gain은 `c_new / (C_acc + c_new) < 5%`. 사실상 freeze.

**얼마나 자주 일어나나**:

| scene | n_kf | updates_mean | updates_max | early-wrong (slope < -0.05) | C=100 도달 못한 KF |
|---|---:|---:|---:|---:|---:|
| chess      | 11 | 45 | 91 | **6/11** | 0 |
| stairs     |  8 | 31 | 46 | **5/8**  | 6 |
| fire       | 16 | 31 | 65 | **7/16** | 0 |
| office     | 17 | 29 | 72 | 6/17 | 0 |
| redkitchen | 12 | 42 | 75 | 4/12 | 0 |
| pumpkin    | 13 | 38 | 68 | 3/13 | 1 |
| heads      | 20 | 25 | 86 | 1/20 | 10 |

"early-wrong"의 정의: 마지막 n/4 pair의 err 평균이 처음 n/4 pair의 err 평균보다 0.05 m 이상 *작은* 경우. **텍스처 풍부한 scene들에서 30~60% 의 keyframe**이 이 패턴.

**Top 예시** (§12):
- `chess kf=0`: err_first 0.659 → err_last 0.236 (freeze만 안 됐으면 ~3배 개선 가능)
- `fire kf=0`: 0.288 → 0.119, conf_mean **16.76** — conf가 매우 높은데도 초반이 틀림
- `pumpkin kf=426`: 0.302 → 0.183, conf_mean 2.94

**`fire kf=0` 사례가 가장 결정적인 증거**예요. confidence가 충분히 높고 시간적으로 안정적인데 실제 err는 절반으로 줄어듦. 즉 **conf만 봐서는 "초반 틀림"을 절대 감지 못함**. Cap+calibration은 dilute 속도만 빠르게 하지, 충돌을 감지하지는 못함.

**Cap sweet spot** (§11): 각 cap 후보에서 "이미 frozen 상태였을" update 비율:

| cap | frozen 비율 |
|---:|---:|
| 100  | **53.3 %** (너무 좁음) |
| **200** | (중간, 권장) |
| 300  | 16.9 % |
| 1000 | 0.1 % (사실상 cap 없음과 동치) |

---

## 관찰 3 — Loop edge: 대부분 진짜 loop가 아님, 진짜인 것은 정보 가치 큼

**Loop pair의 실제 구성** (§13, §6, §7):
- 61% 가 view_angle < 2°
- 39% 가 baseline < 1 cm
- 즉 **retrieval로 찾은 진짜 loop closure가 아니라 인접 keyframe edge** (consecutive)

**Geometry vs err**:

| view_angle bin | err_median | n_pairs |
|---|---:|---:|
| 0–2°   | 0.983 | 232 |
| 2–5°   | 0.663 | 100 |
| 5–10°  | 0.589 |  42 |
| 10–20° | **0.426** | 6 |

진짜 loop가 있는 `heads` (view_max = 13°): err_rmse 0.76 m vs tracking 0.48 m — **1.6배 차이**. 반면 chess loop (view_max = 1°, 전부 consecutive): err_rmse 1.62 m, tracking 대비 **3.4배**.

**Symmetric vs asymmetric decoder**: tracking은 `mast3r_match_asymmetric`, loop는 `mast3r_match_symmetric`. 실제 loop이 있어 베이스라인이 깔린 heads도 tracking 대비 1.6배 — 일반화하면 loop err가 tracking err보다 2~4배 큼. 이게 symmetric decode 자체의 노이즈인지, pair geometry가 어려운 것 때문인지 *지금 데이터로는 결론 안 남*. **같은 pair를 두 mode로 동시에 decode해서 비교하는 ablation이 필요**.

**버려지고 있는 정보**: `global_opt.py`의 `mast3r_match_symmetric` 호출은 `Xji, Xij` (서로의 좌표계에서의 pointmap)도 decode함. 하지만 코드는 `Qij` (joint conf)만 factor weight로 쓰고 **pointmap 자체는 폐기**. view_angle > 5° 의 실제 loop edge에 대해 이 폐기된 pointmap은 **keyframe i를 *전혀 다른 시점*에서 본 독립 관측**. 즉 temporal 이웃에 갇힌 linear monotone fusion의 약점을 메워줄 수 있는 가장 decorrelated한 정보.

---

## 관찰 4 — Tracking은 view angle에 의외로 robust

**Tracking view_angle 분포** (§13): median **10.84°**, p90 28.8°, max 63.2°. 작지 않음.
이유: keyframe 선정 임계치 `match_frac_thresh = 0.333` 이 느슨해서 카메라가 꽤 움직여도 같은 keyframe을 유지함 — `frame_diff_max = 182` (subsample=2 적용 후), 30fps라면 ~12초 분량의 운동.

**Tracking err vs view_angle** (§3): 거의 평평. 0~2°에서 err_median 0.20 m, 20~30°에서 0.23 m. **MASt3R는 60°까지도 텍스처 풍부한 실내 scene에서는 깨지지 않음**. "high view angle = 노이즈 dominant" 라는 직관이 *측정으로 기각됨*.

Fusion 설계에 대한 함의: tracking에서 view_angle gate는 **불필요**. conf calibration이 view_angle이 줄 만한 추가 signal을 이미 흡수.

**Tracking err vs baseline** (§4): 약하게 감소 — baseline < 1 cm 에서 0.27 m, > 1 m 에서 0.17 m. 큰 motion이 *조금* 더 좋은 예측 (parallax 효과), but monocular cue로 이미 잘 추정하니까 큰 폭은 아님.

---

## 관찰 5.5 — Loop closure 수용에 기하학적 검증이 없음

`global_opt.py:add_factors` 의 수용 기준을 코드 확인 결과:
1. `Qj = sqrt(Qii · Qji) > Q_conf` (per-pixel joint conf 임계치)
2. `match_frac > min_match_frac` (보통 0.1), 양방향 통과 필요
3. consecutive edge (`jj == ii+1`) 는 threshold 면제

**확인된 *없음*:**
- ❌ RANSAC / fundamental matrix 검증 (애초에 generic camera 모델이라 F-matrix 사용 불가)
- ❌ Relative pose 추정 후 reprojection error 체크
- ❌ Epipolar geometry 검증
- ❌ Depth scale consistency 검증

즉 **순수 neural verification** — MASt3R conf가 "기하학적으로 일관된 매칭"이라고 *말해주길* 가정. 반복 텍스처 (복도, 책장, 계단)에서 visually similar but spatially wrong matching이 통과 가능. retrieval로 찾아오는 멀리 떨어진 keyframe pair 에서 특히 위험.

**Diag로 확인된 증상**: chess loop 26개 전부 baseline 1cm — *진짜 loop가 거의 안 잡힘*. retrieval이 통과시키는 게 사실상 인접 KF뿐 (또는 적어도 우리가 본 7-Scenes test seq에서는).

**Generic camera 모델에서도 가능한 검증법** (Phase 4 후보):

1. **3D-3D Sim3 RANSAC (Procrustes/Umeyama)**: matched pixel 쌍 `(p_a in i, p_b in j)` 에서 `(X_ii(p_a), X_jj(p_b))` 가지고 RANSAC. camera 모델 무관.
2. **Per-pixel pointmap residual gate** (가장 단순): 매치된 픽셀에서 `|X_ii(p_a) - X_ji(p_b)|` (둘 다 i의 좌표계). 비용 거의 0, 이미 decode된 값들.
3. **Ray-point consistency**: 각 카메라의 unproject ray (generic 모델에 내장) 이용한 distance gate.

**제일 매력적인 건 2번** — 추가 연산 없고 camera 모델 무관.

---

## 관찰 5 — Diag 차원에서 Calib과 No_calib은 구분 불가

`chess` 양 mode로 검증: tracking err_rmse 0.479 (calib) vs 0.493 (no_calib), σ fit 파라미터 ~2% 차이.
이건 *원리적으로 예상*되는 결과 — diag는 raw MASt3R decoder 출력을 측정. 그 출력은 intrinsic 사용 여부와 무관. Calib/no_calib 차이는 *downstream의 pose-estimation loop* 에서 발생. 그래서 나머지 6 scene에서 no_calib을 skip한 게 정보 손실 없음.

---

## 구현 상태 (2026-05-13)

| Phase | 내용 | 상태 |
|---|---|---|
| 1 (calibration + cap) | `weighted_pointmap_linear` filtering mode: linear weighted avg with `w_exp` (calibrated weight 지수) 및 `cap` (누적 W 상한) 옵션 | ✅ 구현됨 |
| 2 (uncertainty / process noise) | `weighted_pointmap_kalman` filtering mode: per-pixel scalar Kalman with calibrated σ² + **process noise (forget_factor)** — 과거가 자연스럽게 잊혀짐 | ✅ 구현됨 |
| 3 | Loop edge에서 폐기되는 X_ji 재활용 | ⏳ 미구현 |
| 4 | Loop closure geometric verification (per-pixel pointmap residual gate 또는 3D-3D RANSAC) | ⏳ 미구현 |

**왜 Phase 1과 2를 별도 모드로 분리했나:**
정통 *static-state, no-Q* Kalman은 *linear weighted avg with cap* 과 **수학적으로 동치**. 그래서 KF로 짠 cap+calib는 의미적으로 새로운 게 아님. *process noise (decay)* 가 들어가야 Kalman이 진짜로 linear와 달라짐. 이 사실을 ablation에 반영하려면 calib/cap 변형은 linear 식으로 짜고, process noise만 Kalman 식 fusion 변형에 넣어야 함.

**Ablation 변형 (4개):**

| variant | mode | knobs | 의미 |
|---|---|---|---|
| vanilla | `weighted_pointmap` | (없음) | baseline, raw c, no cap |
| calibonly | `weighted_pointmap_linear` | `w_exp=0.74`, `cap=0` | calibration만 추가 |
| caponly | `weighted_pointmap_linear` | `w_exp=1.0`, `cap=200` | cap만 추가 (typical c=8 ⇒ eff N≈25) |
| calibcap | `weighted_pointmap_linear` | `w_exp=0.74`, `cap=120` | calibration + cap (cap 재scaling으로 caponly와 동일 effective N 매칭) |
| fusion | `weighted_pointmap_kalman` | `sigma_a=0.708`, `sigma_p=-0.372`, `forget_factor=0.95` | proper KF + process noise (effective window ≈ 20) |

**왜 calibcap의 cap이 120인지:** typical c=8에서 caponly의 cap=200 ⇒ effective fusion window ≈ 200/8 = 25. 동일 effective N으로 맞추려면 calibcap의 cap = 25 · 8^0.74 ≈ 120. 안 맞추면 calibcap이 "calibration 효과"인지 "더 큰 effective N 효과"인지 구분 안 됨.

**Process noise (`forget_factor`)의 의미:**
매 step 전에 `σ²_canon ← σ²_canon / λ` (λ < 1) 적용. prior의 certainty가 자라지 못함 → 오래된 obs는 자연스럽게 가중치 ↓. λ=0.95 ⇒ effective window ≈ 1/(1−λ) = 20. **이것이 user 의도한 "uncertainty가 update에 영향" 메커니즘** — disagree 감지 후 inflate하는 reset과 달리, *항상* 부드럽게 옛 obs를 잊는 EWMA-style 동작. 별도 threshold 없음.

**왜 reset (Mahalanobis gate) 안 넣었나:**
이전 commit (`002e306`) 에선 reset이 있었으나 — process noise가 "과거가 *자연스럽게* 잊혀지는" 메커니즘을 이미 제공하니까 reset은 over-engineering. 결과 보고 정말 필요하면 orthogonal knob로 추가 가능.

**평가 중 (서버, 4-GPU 병렬):**
- vanilla (`config/eval_calib.yaml`) vs fusion (`config/eval_calib_fusion.yaml`) on 7-Scenes 전 scene
- `scripts/eval_7_scenes.sh --variant {vanilla,fusion} --print` 로 ATE + pointmap geometry (Accuracy/Completion/Chamfer) 비교
- 기대: 
  - ATE 개선 (특히 chess/fire 같은 early-wrong 많은 scene)
  - Pointmap RMSE 개선 (cap 효과)
  - innovation gate 비율 vs scene 특성 상관관계 확인

---

## 종합 — 제안 방향

### Phase 1 — Cap + Calibration (`frame.py:74-77`, 변경 ~4줄)

```python
elif filtering_mode == "weighted_pointmap":
    w_new = C ** 0.74                                          # calibrated
    self.X_canon = ((self.C * self.X_canon) + (w_new * X)) / (self.C + w_new)
    self.C = (self.C + w_new).clamp(max=200.0)                 # cap
    self.N += 1
```

§11 (cap=200 이면 freeze 거의 0)과 §13의 fusion 시뮬레이션으로 검증.

### Phase 2 — Innovation 처리 (user의 진짜 목표: "빠른 correction")

Cap+calibration는 *freeze 방지*만 하지, early-wrong을 *감지하거나 빠르게 고치지* 않음. early-wrong 시 oracle_best까지의 갭을 메우려면 **disagreement 감지 시 σ² (또는 self.C) 을 inflate** 해서 다음 관측의 영향력을 즉시 회복시켜야 함.

두 가지 형태가 가능:
- **Innovation-inflated KF-lite**: `self.C` → `self.sigma2` 로 의미만 바꿈 (메모리 동일). 매 업데이트에서 per-pixel Mahalanobis-like ratio 계산, 임계치 초과 시 σ²을 곱해서 부풀림.
- **Adaptive forgetting**: `self.C = λ * self.C + w_new`, residual 크면 λ를 줄임.

전자가 더 원칙적이고 calibrated noise model의 자연스러운 확장. 후자가 더 단순. 둘 다 시도 가치.

**여기가 user 의도("초기가 틀리면 빨리 고치라")의 핵심.** Per-pixel uncertainty가 가치 있는 이유는 그 자체가 uncertainty라서가 아니라, innovation gating을 가능하게 하는 기반이라서.

### Phase 3 — Loop pointmap 재활용 (Phase 1, 2와 직교)

view_angle > ~5° 의 real loop edge에서 `Xji`, `Xij`를 각각 keyframe_j, keyframe_i 의 `update_pointmap` 에 재투입. Decode 비용은 이미 matching 단계에서 지불됨 → 사실상 *공짜로* 최대로 decorrelated된 시점의 관측 추가. early-wrong에 대한 *간접* 처방: linear monotone fusion이 갇혀 있던 temporal neighborhood *바깥의* 정보를 keyframe에 주입.

### Phase 4 — Loop closure geometric verification (관찰 5.5 참조)

현재 코드는 conf + match_frac threshold만으로 loop edge 수용 → false-positive 위험. Generic camera 모델 제약 하에서 가능한 검증법:

- **Per-pixel pointmap residual gate** (가장 가벼움): `|X_ii(p_a) - X_ji(p_b)|` 가 임계치 초과하는 픽셀 비율로 edge reject. 추가 연산 없음 — 이미 decode된 X_ii, X_ji 사용.
- **3D-3D Sim3 RANSAC**: matched 3D 점 대응에서 Sim3 fit → inlier ratio gate. 더 robust하지만 RANSAC 비용 추가.

지금은 chess loop가 다 baseline 1cm (= 사실상 consecutive)이라 false-positive 노출이 안 보임. **다른 dataset (TUM, ETH3D)이나 더 긴 시퀀스에서 효과 클 것**.

---

## 아직 미해결 / 추가로 확인할 가치 있는 것

1. **Symmetric vs asymmetric decoder를 같은 pair에 적용 비교** — diag.py 확장 필요. loop의 2~4× err가 symmetric mode 자체의 한계인지 geometry 탓인지 결론 가능. paper-level finding 후보.
2. **err의 spatial 구조** — per-pixel err가 image 경계나 depth 불연속에 몰리는지? 그렇다면 scalar conf 하나로는 못 잡고, learned spatial prior 등 별도 연구 방향.
3. **valid_match correlation** — `valid_match` 저장돼 있지만 아직 분석에 미사용. matched pixel이 unmatched보다 systematic하게 더 정확하다면, fusion weight는 `c · valid_match` 가 옳음.
4. **err를 z_gt로 정규화한 relative err** — σ ≈ c^-0.37 calibration이 깊이로 정규화해도 유지되나? scene별 a 계수 차이의 일부가 이걸로 설명될 수 있음.
5. **Loop gate criterion** — 현재 `min_match_frac = 0.1` 만 사용. (match_frac, view_angle, baseline) 결합 gate가 더 나을 수 있음.
6. **Train/test sequence 영향** — 7-Scenes의 어떤 sequence가 MASt3R 학습에 포함됐는지에 따라 conf가 편향될 수 있음.
