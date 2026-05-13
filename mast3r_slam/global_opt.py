import lietorch
import torch
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
from mast3r_slam import diag, loop_diag
import mast3r_slam_backends


class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.window_size = self.cfg["window_size"]

        self.K = K

        # Phase 3: deferred loop-pointmap fusion. Each entry holds the
        # symmetric-decode pointmaps that will be re-fused into the two
        # keyframes' canonical, *after* both have been touched by at least one
        # backend optimisation pass.
        self.pending_loop_fusions = []

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        diag_on = diag.get().enabled
        loop_diag_on = loop_diag.get().enabled
        # Phase 3: re-fuse the symmetric decoder's pointmaps (Xji, Xij) back
        # into the respective keyframes' canonical. The decode is already
        # paid for the match-frac filter; these are "free" extra observations
        # from a maximally decorrelated viewpoint.
        loop_fuse_cfg = config["tracking"].get("loop_fuse", {})
        loop_fuse_on = bool(loop_fuse_cfg.get("enabled", False))
        need_points = diag_on or loop_diag_on or loop_fuse_on
        sym_out = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j,
            return_points=need_points,
        )
        if need_points:
            (idx_i2j, idx_j2i, valid_match_j, valid_match_i,
             Qii, Qjj, Qji, Qij,
             Xji_full, Cji_full, Xij_full, Cij_full) = sym_out
        else:
            (idx_i2j, idx_j2i, valid_match_j, valid_match_i,
             Qii, Qjj, Qji, Qij) = sym_out

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        # Diagnostics on this loop-closure decode batch.
        if need_points:
            kept = valid_edges.nonzero(as_tuple=True)[0].cpu().tolist()
            Xji_kept = Xji_full[valid_edges]
            Cji_kept = Cji_full[valid_edges]
            Xij_kept = Xij_full[valid_edges]
            Cij_kept = Cij_full[valid_edges]
            consec_kept = consecutive_edges[valid_edges].cpu().tolist()
            mf_i_kept = match_frac_i[valid_edges].cpu().tolist()
            mf_j_kept = match_frac_j[valid_edges].cpu().tolist()
            vmj_kept = valid_match_j  # already filtered
            vmi_kept = valid_match_i
            ii_kept = ii_tensor.cpu().tolist()
            jj_kept = jj_tensor.cpu().tolist()

            # Per-pixel pred-vs-GT diag (heavy, optional)
            if diag_on:
                for b in range(len(kept)):
                    diag.get().record_loop_edge(
                        frame_id_i=ii_kept[b],
                        frame_id_j=jj_kept[b],
                        Xji=Xji_kept[b], Cji=Cji_kept[b],
                        Xij=Xij_kept[b], Cij=Cij_kept[b],
                        valid_match_j=vmj_kept[b],
                        valid_match_i=vmi_kept[b],
                        is_consecutive=bool(consec_kept[b]),
                        match_frac_i=float(mf_i_kept[b]),
                        match_frac_j=float(mf_j_kept[b]),
                    )

            # Phase 3: enqueue loop-pointmap fusion. The symmetric decode gives
            #   Xij : i-grid, j-coord  → into kf_i (needs T_{i←j} transform)
            #   Xji : j-grid, i-coord  → into kf_j (needs T_{j←i} transform)
            # The transforms use SLAM-estimated keyframe poses, which are
            # *drifted* on freshly added keyframes. We defer the fusion until
            # both keyframes have had at least one backend opt pass — see
            # FactorGraph.apply_pending_loop_fusions().
            if loop_fuse_on:
                exclude_consec = bool(loop_fuse_cfg.get("exclude_consecutive", False))
                for b in range(len(kept)):
                    if exclude_consec and consec_kept[b]:
                        continue
                    self.pending_loop_fusions.append({
                        "i": int(ii_kept[b]),
                        "j": int(jj_kept[b]),
                        # detach + clone so the tensors persist past this
                        # add_factors call's GPU memory churn.
                        "Xij": Xij_kept[b].detach().clone(),
                        "Cij": Cij_kept[b].detach().clone(),
                        "Xji": Xji_kept[b].detach().clone(),
                        "Cji": Cji_kept[b].detach().clone(),
                    })

            # Per-edge lightweight loop-acceptance summary (cheap).
            # We compare each kept Xji against keyframe i's stored canonical
            # pointmap at the matched pixels — non-zero residual ⇒ MASt3R's
            # symmetric decode disagrees with the keyframe's running estimate,
            # which is exactly the false-positive signal a geometric verifier
            # would use.
            if loop_diag_on:
                # Qj is the kept per-pixel joint conf; use a mean as edge summary.
                Qj_means = Qj.float().mean(dim=(1, 2)).cpu().tolist()
                for b in range(len(kept)):
                    i_b = ii_kept[b]
                    j_b = jj_kept[b]
                    kf_i = self.frames[i_b]
                    kf_j = self.frames[j_b]
                    loop_diag.get().record(
                        i=i_b, j=j_b,
                        X_canon_i=kf_i.X_canon,
                        Xji=Xji_kept[b],
                        valid_match_j=vmj_kept[b],
                        match_frac_i=mf_i_kept[b],
                        match_frac_j=mf_j_kept[b],
                        Q_mean=Qj_means[b],
                        is_consecutive=consec_kept[b],
                        T_WC_i=kf_i.T_WC.matrix()[0]
                            if hasattr(kf_i.T_WC, "matrix") else kf_i.T_WC,
                        T_WC_j=kf_j.T_WC.matrix()[0]
                            if hasattr(kf_j.T_WC, "matrix") else kf_j.T_WC,
                    )

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def apply_pending_loop_fusions(self):
        """Phase 3: drain the pending-fusion queue for edges where both
        keyframes have been refined by at least one backend opt pass.

        The symmetric-decode pointmaps come from the decoder in different
        coordinate frames than the destination keyframe's canonical:
            Xij is in j's coord  → transform to i's coord  → fuse into kf_i
            Xji is in i's coord  → transform to j's coord  → fuse into kf_j
        The Sim3 transform uses the keyframes' SLAM-estimated poses, so we
        only do this after both have been optimised at least once.
        """
        if not self.pending_loop_fusions:
            return
        remaining = []
        for entry in self.pending_loop_fusions:
            i, j = entry["i"], entry["j"]
            # Gate on shared storage so we see the real refined-pose counter,
            # not the temp Frame copy that __getitem__ would hand back.
            if (int(self.frames.n_opt_passes[i]) < 1
                    or int(self.frames.n_opt_passes[j]) < 1):
                remaining.append(entry)
                continue
            kf_i = self.frames[i]
            kf_j = self.frames[j]
            # T_{i ← j} = T_WCi^-1 · T_WCj  (matches tracker.py:T_CkCf idiom)
            T_i_from_j = kf_i.T_WC.inv() * kf_j.T_WC
            T_j_from_i = kf_j.T_WC.inv() * kf_i.T_WC
            # (H*W, 3) → (1, H*W, 3) for batched .act
            Xij_b = entry["Xij"].unsqueeze(0)
            Cij_b = entry["Cij"].unsqueeze(0)
            Xji_b = entry["Xji"].unsqueeze(0)
            Cji_b = entry["Cji"].unsqueeze(0)
            Xij_in_i = T_i_from_j.act(Xij_b)
            Xji_in_j = T_j_from_i.act(Xji_b)
            kf_i.update_pointmap(Xij_in_i, Cij_b)
            kf_j.update_pointmap(Xji_in_j, Cji_b)
            # Write back to shared storage — update_pointmap mutates the temp
            # Frame's X_canon/C; without write-back the fusion is silently a
            # no-op (this is exactly why our previous run came out bitwise
            # identical to calibonly).
            self.frames[i] = kf_i
            self.frames[j] = kf_j
        self.pending_loop_fusions = remaining

    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self):
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

        # Mark touched keyframes (write directly to shared storage — going
        # through self.frames[idx] returns a temp Frame copy, increments on
        # that copy would not persist).
        for idx in unique_kf_idx.cpu().tolist():
            self.frames.n_opt_passes[idx] += 1

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

        # Mark touched keyframes (write directly to shared storage — going
        # through self.frames[idx] returns a temp Frame copy, increments on
        # that copy would not persist).
        for idx in unique_kf_idx.cpu().tolist():
            self.frames.n_opt_passes[idx] += 1
