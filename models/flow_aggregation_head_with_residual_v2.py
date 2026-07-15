"""
FlowAggregationHeadWithResidualV2

Extends FlowAggregationHeadWithResidual with configurable parameters and two
optional improvements to flow guidance signal quality:

  topk              (int,   default=4)         – number of easiest samples
                                                 selected per batch (was hard-
                                                 coded to 2 in V1)

  boundary_threshold (float, default=pi/18)    – angle-change threshold for
                                                 detect_flow_changes_batch
                                                 (was hard-coded to pi/12 in
                                                 V1; smaller = more sensitive,
                                                 better for CMC small-motion)

  use_cycle_conf    (bool,  default=False)     – weight the mask-based spatial
                                                 averaging in flow aggregation
                                                 by per-pixel cycle-consistency
                                                 confidence:
                                                   conf = exp(-|fw+warp(bw,fw)| / σ)
                                                 Unreliable pixels (occlusion,
                                                 fast motion) contribute less to
                                                 each channel's flow estimate.
                                                 backward flows are already
                                                 pre-computed and loaded.

  cycle_conf_sigma  (float, default=1.0)       – temperature σ for cycle conf.
                                                 Smaller → sharper gating.

  cycle_conf_sigma_overrides (dict, default=None) – per-sample σ override via
                                                 {substring: sigma}, matched
                                                 against seq_names (same
                                                 pattern as
                                                 clamp_flow_t_overrides). v72
                                                 found σ=1.0 hurts adjacent-
                                                 frame pairs (cycle error
                                                 already ~0.3px median —
                                                 nothing to filter, and it
                                                 suppresses legitimate large-
                                                 motion instrument edges).
                                                 Large-gap bridge pairs
                                                 (~5-frame real gap) measured
                                                 much heavier cycle-error
                                                 tails (p90 ~12-20px vs ~2-3px
                                                 for adjacent pairs, ~17-23%
                                                 of pixels >5px vs ~5%) — a
                                                 sharper σ tuned for bridges
                                                 specifically, left large/off
                                                 for adjacent pairs, avoids
                                                 repeating v72's regression.

  detach_mask_patterns (list, default=None)    – substrings matched against
                                                 seq_names; any sample whose
                                                 seq_name matches gets its
                                                 mask1/mask2 detached before
                                                 aggregate_flow_with_residual,
                                                 so that sample's flow loss
                                                 still trains the residual
                                                 branch (decode_head3, a
                                                 separate pair-conditioned
                                                 head — not affected by this)
                                                 but contributes zero gradient
                                                 to the mask branch
                                                 (decode_head2). Motivated by
                                                 CMC bridge pairs (b105/b50):
                                                 each bridge frame is also one
                                                 endpoint of an adjacent gap1
                                                 pair (e.g. g5's pre-frame is
                                                 also b105's post-frame) — the
                                                 SAME image's per-image mask
                                                 prediction is pulled toward
                                                 fitting two motion-scale-
                                                 mismatched flow targets
                                                 (gap1's ~1-frame vs bridge's
                                                 ~5-frame real gap) in
                                                 different training samples.
                                                 Bridge flow itself was
                                                 verified reliable (composed-
                                                 vs-direct RAFT consistency,
                                                 cos_sim median 0.92-0.97 —
                                                 see README) so the fix routes
                                                 that signal only through the
                                                 conflict-free path instead of
                                                 discarding it.

  topk_scale_normalize (bool, default=False)   – rank samples for topk
                                                 selection by GT-flow-scale-
                                                 normalised loss instead of
                                                 raw squared error. Measured
                                                 on CMC bridge pairs (~5-frame
                                                 real gap, ~3-4x larger GT
                                                 flow magnitude than adjacent-
                                                 frame pairs): raw MSE loss is
                                                 ~7-9x larger purely from
                                                 scale, giving them only ~15%
                                                 topk=4 survival vs ~72% for
                                                 adjacent pairs in the same
                                                 batches (ResolutionGrouped-
                                                 BatchSampler mixes them
                                                 freely — see README). This
                                                 means large-motion sources
                                                 get starved of gradient
                                                 almost entirely under hard
                                                 topk, independent of fit
                                                 quality. Only the SORTING
                                                 criterion changes — the loss
                                                 actually backpropagated for
                                                 selected samples is still the
                                                 raw, unnormalised MSE, so
                                                 gradient magnitude semantics
                                                 for whichever samples get
                                                 selected are unchanged.

  use_bg_affine_removal (bool, default=False)  – subtract a robust background
                                                 flow estimate (spatial median
                                                 of fw_flow) before aggregation
                                                 so the per-channel flow averages
                                                 are computed on instrument-
                                                 residual motion rather than
                                                 mixed bg+instrument motion.
                                                 The bg estimate is added back
                                                 before loss computation so the
                                                 loss target is unchanged.

  use_per_channel_residual_scale (bool, default=False)
    Replace the single scalar residual_adjustment_scale with a learnable
    per-channel vector [mask_layer], initialised to residual_adjustment_scale
    so behaviour matches the scalar version at step 0. Background and
    instrument channels need very different residual freedom (background is
    ~static, instrument moves non-rigidly) — a single global scale forces a
    one-size-fits-all tradeoff (v77 showed tightening it globally hurts the
    instrument channel specifically). Each channel's scale is free to shrink
    or grow independently during training.

Usage in config YAML:
  decode_head:
    type: FlowAggregationHeadWithResidualV2
    topk: 4
    boundary_threshold: 0.1745     # pi/18 ≈ 0.1745
    use_cycle_conf: true           # enable cycle-consistency confidence gating
    cycle_conf_sigma: 1.0
    use_bg_affine_removal: true    # enable background motion subtraction
    ...  (all other kwargs same as V1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_aggregation_head_with_residual import FlowAggregationHeadWithResidual
from utils.warp_utils import flow_warp


class FlowAggregationHeadWithResidualV2(FlowAggregationHeadWithResidual):
    """
    V2: topk and boundary_threshold are configurable via __init__.
    Everything else is identical to FlowAggregationHeadWithResidual.
    """

    def __init__(self, *args, topk: int = 4,
                 topk_mode: str = 'hard',
                 topk_soft_temperature: float = 1.0,
                 boundary_threshold: float = math.pi / 18,
                 boundary_dilation: int = 7,
                 boundary_floor: float = 0.0,
                 use_cycle_conf: bool = False,
                 cycle_conf_sigma: float = 1.0,
                 cycle_conf_sigma_overrides: dict = None,
                 detach_mask_patterns: list = None,
                 topk_scale_normalize: bool = False,
                 use_bg_affine_removal: bool = False,
                 bg_removal_affine: bool = False,
                 use_per_channel_residual_scale: bool = False,
                 use_heteroscedastic_loss: bool = False,
                 use_flow_metric_loss: bool = False,
                 flow_metric_weight: float = 0.1,
                 use_mask_warp_consistency: bool = False,
                 mask_warp_consistency_weight: float = 0.1,
                 use_em_consistency_loss: bool = False,
                 em_consistency_weight: float = 0.1,
                 em_consistency_temperature: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert topk >= 1, "topk must be >= 1"
        assert topk_mode in ('hard', 'soft')
        assert 0.0 <= boundary_floor < 1.0
        self.topk = topk
        # 'hard': original behaviour — sort by per-sample loss, keep only the
        #   k easiest, zero weight on the rest (binary 0/1 inclusion).
        # 'soft': every sample in the batch contributes, weighted by
        #   softmax(-zscore(loss)/T) — easier samples still get more weight
        #   (same easy-example-mining intent) but nothing is thrown away
        #   outright. z-score (not raw loss) feeds the softmax so temperature
        #   has a stable meaning regardless of the loss's absolute scale,
        #   which shifts by orders of magnitude over training.
        self.topk_mode = topk_mode
        self.topk_soft_temperature = topk_soft_temperature
        self.boundary_threshold = boundary_threshold
        # supervision density controls: warp loss is masked to motion-angle
        # boundaries dilated by boundary_dilation px (was hard-coded 7 → only
        # ~20% of pixels supervised). boundary_floor>0 gives the remaining
        # pixels a small weight instead of zero.
        self.boundary_dilation = boundary_dilation
        self.boundary_floor = boundary_floor
        self.use_cycle_conf = use_cycle_conf
        self.cycle_conf_sigma = cycle_conf_sigma
        self.cycle_conf_sigma_overrides = cycle_conf_sigma_overrides
        self.detach_mask_patterns = detach_mask_patterns
        self.topk_scale_normalize = topk_scale_normalize
        self.use_bg_affine_removal = use_bg_affine_removal
        self.bg_removal_affine = bg_removal_affine

        self.use_per_channel_residual_scale = use_per_channel_residual_scale
        if use_per_channel_residual_scale:
            assert self.residual_adjustment_scale != -1., \
                "use_per_channel_residual_scale requires a finite residual_adjustment_scale as init value"
            init = torch.full((self.mask_layer,), float(self.residual_adjustment_scale))
            self.residual_scale_per_channel = nn.Parameter(init)

        # Heteroscedastic (learned per-pixel) uncertainty on the warp loss.
        # Replaces the fixed boundary_floor weighting with a LEARNED confidence:
        # loss = (error)^2 / (2*sigma^2) + log(sigma), sigma predicted per-pixel
        # from the residual-prediction feature (all_pred_residual_fw/bw, already
        # dense and at flow resolution). log(sigma) regularises against the
        # trivial "inflate sigma everywhere" collapse. Composes with (does not
        # replace) the existing boundary mask — multiplied together, so this
        # only adds a learned confidence layer on top of the validated
        # boundary_dilation/floor mechanism.
        self.use_heteroscedastic_loss = use_heteroscedastic_loss
        if use_heteroscedastic_loss:
            def _sigma_head():
                return nn.Sequential(
                    nn.Conv2d(2 * self.mask_layer, 16, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 1),
                )
            self.sigma_head_fw = _sigma_head()
            self.sigma_head_bw = _sigma_head()

        # Metric-learning loss on the flow_feat_before_agg embedding (already
        # exists as an intermediate representation en route to flow
        # reconstruction — see parent __init__). Reframes "regress small/
        # noisy raw flow values accurately" as "keep same-channel motion
        # embeddings close, different-channel centroids apart" — a
        # discriminative-clustering objective that tolerates low absolute
        # motion magnitude far better than direct regression, since it only
        # needs RELATIVE separability, not precise numeric values.
        self.use_flow_metric_loss = use_flow_metric_loss
        self.flow_metric_weight = flow_metric_weight

        # Cross-frame mask consistency: mask1/mask2 are currently predicted
        # and supervised completely independently (mask1 only ever sees
        # fw_flow's loss, mask2 only bw_flow's) — no term anywhere requires
        # that the SAME physical surface point gets the SAME channel
        # assignment in both frames. Warp mask2 back to frame1's coordinate
        # frame with fw_flow (and mask1 forward with bw_flow) and penalise
        # disagreement with the frame's own mask. Targets the channel-
        # instability / shaft-jaw-split failure mode documented across the
        # project (v61/v63/v81 etc.) directly, using a correspondence signal
        # the network has never been supervised on. Not self-referential
        # like the failed v8-v26 clustering losses: the target for mask1
        # comes from an INDEPENDENT prediction (mask2) plus an EXTERNAL
        # signal (RAFT flow), not from mask1's own statistics.
        self.use_mask_warp_consistency = use_mask_warp_consistency
        self.mask_warp_consistency_weight = mask_warp_consistency_weight

        # E-step-consistency auxiliary loss (see _em_consistency_loss):
        # pushes mask's soft assignment toward agreement with which
        # channel's own rigid motion model (flow_agg + affine, the "M-step"
        # this architecture already performs) best explains each pixel's
        # true flow — a signal no existing loss provides, since warp_seg
        # only ever sees the final mask-mixed reconstruction.
        self.use_em_consistency_loss = use_em_consistency_loss
        self.em_consistency_weight = em_consistency_weight
        self.em_consistency_temperature = em_consistency_temperature

        # coord_map is created by parent as a plain .cuda() tensor attribute
        # (not register_buffer), which causes CUDA illegal memory access when
        # PL resumes training and calls optimizers_to_device. Re-register it
        # as a proper buffer so PL manages device placement correctly.
        if hasattr(self, 'coord_map') and not isinstance(self.coord_map, torch.nn.Parameter):
            coord_map_data = self.coord_map.cpu()
            del self.coord_map
            self.register_buffer('coord_map', coord_map_data)

    # ------------------------------------------------------------------ #
    # Flow signal quality helpers                                          #
    # ------------------------------------------------------------------ #
    def _compute_cycle_conf(self, fw_flow, bw_flow, seq_names=None):
        """
        Per-pixel cycle-consistency confidence.
        conf = exp(-|fw_flow + warp(bw_flow, fw_flow)| / sigma)
        Shape: [B, 1, H, W], range (0, 1].

        sigma is per-sample when cycle_conf_sigma_overrides + seq_names are
        given (same {substring: value} override pattern as
        clamp_flow_t_overrides): lets sources with different cycle-error
        scales (e.g. large-gap bridge pairs vs. adjacent-frame pairs) get
        different gating sharpness without a global sigma compromising both.
        """
        bw_warped = flow_warp(bw_flow, fw_flow, pad='border')   # bw at warped pos
        cycle_err = (fw_flow + bw_warped).norm(dim=1, keepdim=True)  # [B,1,H,W]

        if self.cycle_conf_sigma_overrides and seq_names is not None:
            B = fw_flow.shape[0]
            sigmas = []
            for i in range(B):
                name = seq_names[i]
                s = self.cycle_conf_sigma
                for pattern, override_s in self.cycle_conf_sigma_overrides.items():
                    if pattern in name:
                        s = override_s
                        break
                sigmas.append(s)
            sigma_tensor = torch.tensor(sigmas, device=fw_flow.device, dtype=fw_flow.dtype).view(B, 1, 1, 1)
            return torch.exp(-cycle_err / sigma_tensor)

        return torch.exp(-cycle_err / self.cycle_conf_sigma)

    def _maybe_detach_mask(self, mask, seq_names):
        """
        Per-sample mask detach: samples whose seq_name matches any pattern in
        detach_mask_patterns get mask.detach() (zero gradient into
        decode_head2 for that sample's flow loss); other samples pass
        through unchanged. torch.where keeps this differentiable per-element
        — gradient flows only through the branch actually selected.
        """
        B = mask.shape[0]
        flags = [any(p in seq_names[i] for p in self.detach_mask_patterns) for i in range(B)]
        if not any(flags):
            return mask
        flag_tensor = torch.tensor(flags, device=mask.device).view(B, 1, 1, 1)
        return torch.where(flag_tensor, mask.detach(), mask)

    @staticmethod
    def _estimate_bg_flow(flow):
        """
        Robust background flow estimate: spatial median.
        Assumes instrument covers < 50% of frame so median is in bg region.
        Returns [B, 2, 1, 1] broadcast-ready.
        """
        B = flow.shape[0]
        bg, _ = flow.view(B, 2, -1).median(dim=-1)
        return bg.view(B, 2, 1, 1)

    def _estimate_bg_flow_affine(self, flow):
        """
        Global affine background fit: bg(x,y) = A·(x,y) + b (6 params).
        Captures camera rotation/zoom that the median (translation-only)
        estimate misses.
        get_demean_affine_flow with an all-ones mask gives the DE-MEANED
        affine component A·(u - mu) only — its translation part mu_F is
        dropped (in the reconstruction path flow_agg carries it). For a
        bg estimate we must add the translation back; we use the spatial
        median (robust to instrument outliers, matches the v73 estimator,
        and reduces exactly to it when A≈0).
        Not robust to instrument pixels biasing A, but the instrument
        covers a small frame fraction so the bias is modest.
        Returns [B, 2, H, W].
        """
        B, _, H, W = flow.shape
        ones = torch.ones(B, 1, H, W, device=flow.device, dtype=flow.dtype)
        affine_demean = self.get_demean_affine_flow(ones, flow)      # A·(u-mu): [B,2,H,W]
        translation = self._estimate_bg_flow(flow)                   # median:   [B,2,1,1]
        return affine_demean + translation

    def _bg_flow(self, flow):
        return (self._estimate_bg_flow_affine(flow) if self.bg_removal_affine
                else self._estimate_bg_flow(flow))

    def _flow_metric_loss(self, mask, flow):
        """
        Metric-learning loss on the flow_feat_before_agg embedding (the
        num_flow_feat_channels-dim per-pixel representation already computed
        en route to flow reconstruction, before mask-weighted pooling).
          intra: same-channel pixels' embeddings should align with their own
                 channel's centroid (tight clusters)
          inter: different channels' centroids should not be too similar
                 (clamped at 0 — only penalises actual overlap, no reward for
                 pushing centroids to be maximally opposite, which would be
                 an unnecessary/unstable extra constraint)
        Same pairwise-centroid pattern as _dino_merge_loss (v88), applied to
        the flow embedding space instead of DINO's appearance space.
        mask: [B, C, H, W] (softmax, same mask used for aggregation this step)
        flow: [B, 2, H, W] (same flow fed into aggregation this step, i.e.
              post-bg-removal if that's enabled, for a consistent input)
        Returns per-sample loss [B] (not pre-reduced) so it composes with topk
        easy-example selection the same way the reconstruction loss does.
        """
        B, C, H, W = mask.shape
        feat = self.flow_feat_before_agg(flow)              # [B, D, H, W]
        feat_n = F.normalize(feat, dim=1)

        centroids = []
        for c in range(C):
            w_c = mask[:, c:c + 1]                            # [B,1,H,W]
            W_c = w_c.sum(dim=(2, 3))                         # [B,1]
            centroid = (feat_n * w_c).sum(dim=(2, 3)) / (W_c + 1e-6)
            centroids.append(F.normalize(centroid, dim=1))    # [B,D]

        intra = torch.zeros(B, device=mask.device)
        for c in range(C):
            w_c = mask[:, c]                                  # [B,H,W]
            sim = (feat_n * centroids[c].view(B, -1, 1, 1)).sum(dim=1)
            W_c = w_c.sum(dim=(1, 2))
            intra = intra + (w_c * (1.0 - sim)).sum(dim=(1, 2)) / (W_c + 1e-6)
        intra = intra / C

        inter = torch.zeros(B, device=mask.device)
        n_pairs = 0
        for i in range(C):
            for j in range(i + 1, C):
                aff = (centroids[i] * centroids[j]).sum(dim=1)   # [B]
                inter = inter + aff.clamp(min=0)
                n_pairs += 1
        inter = inter / max(n_pairs, 1)

        return intra + inter

    def _mask_warp_consistency_loss(self, mask1, mask2, fw_flow, bw_flow,
                                     weight_fw, weight_bw):
        """
        Cross-frame channel-assignment consistency. Warp mask2 back into
        frame1's coordinate frame with fw_flow (flow_warp(x, flow12) ~=
        reconstructs frame1 from x=frame2-space input, same convention
        already used by _compute_cycle_conf) and compare against mask1 —
        the same physical surface point should get the same soft channel
        distribution in both frames. Symmetric: also warp mask1 forward
        with bw_flow and compare against mask2.
        weight_fw/weight_bw: the same boundary/motion-detection mask already
        computed for the flow-reconstruction loss (mask_fw_flow/mask_bw_flow)
        — reused here so pixels where correspondence is inherently unreliable
        (occlusion, fast motion, outside the boundary region) are naturally
        down-weighted, no new hyperparameter needed.
        mask1, mask2: [B, C, H, W] softmax. fw_flow, bw_flow: [B, 2, H, W],
        same resolution as mask1/mask2 (already resized to mask_size by the
        caller). Returns per-sample [B] loss so it composes with topk.
        """
        mask2_to_1 = flow_warp(mask2, fw_flow, pad='border')
        mask1_to_2 = flow_warp(mask1, bw_flow, pad='border')

        diff_1 = (mask1 - mask2_to_1).abs().sum(dim=1, keepdim=True)  # [B,1,H,W]
        diff_2 = (mask2 - mask1_to_2).abs().sum(dim=1, keepdim=True)

        loss_1 = (diff_1 * weight_fw).sum(dim=(1, 2, 3)) / (weight_fw.sum(dim=(1, 2, 3)) + 1e-6)
        loss_2 = (diff_2 * weight_bw).sum(dim=(1, 2, 3)) / (weight_bw.sum(dim=(1, 2, 3)) + 1e-6)
        return loss_1 + loss_2

    def _per_channel_rigid_flow(self, mask, flow):
        """
        Per-channel, per-pixel flow prediction from ONLY the rigid
        "M-step cluster models" (flow_agg's per-channel constant, plus
        affine if free_residual_with_affine is on) — i.e. what each
        channel's own fitted motion model predicts at every pixel, BEFORE
        the final mask-weighted collapse across channels that
        aggregate_flow_with_residual normally does. Deliberately excludes
        the residual: it's a free-form per-pixel correction with no
        channel-specific rigid-motion identity, so including it would let
        every channel "explain" every pixel equally well and defeat the
        purpose of asking which channel's OWN structural motion model
        actually fits. Returns [B, C, 2, H, W].
        """
        B, C, H, W = mask.shape
        mask_spatial_normalized = mask / mask.view(B, C, H * W, 1).sum(dim=2, keepdim=True)

        feat = self.flow_feat_before_agg(flow)                          # [B, D, H, W]
        pooled = feat[:, :, None, ...] * mask_spatial_normalized[:, None, ...]  # [B,D,C,H,W]
        pooled = pooled.flatten(3, 4).sum(-1)                            # [B, D, C]
        const_flow = self.flow_feat_after_agg(pooled)                    # [B, 2, C]
        per_channel = const_flow[..., None, None].expand(-1, -1, -1, H, W).clone()  # [B,2,C,H,W]

        if self.free_residual_with_affine:
            affine_per_channel = self._demean_affine_flow_per_channel(mask, flow)  # [B,C,H,W,2]
            per_channel = per_channel + affine_per_channel.permute(0, 4, 1, 2, 3)   # [B,2,C,H,W]

        return per_channel.permute(0, 2, 1, 3, 4)  # [B, C, 2, H, W]

    def _em_consistency_loss(self, mask, flow):
        """
        E-step-consistency auxiliary loss. mask is predicted purely from
        RGB appearance (backbone + seg head) — nothing currently checks
        whether the resulting soft assignment agrees with which channel's
        OWN rigid-motion model (the M-step: flow_agg + affine, computed by
        _per_channel_rigid_flow) best explains a given pixel's true flow.
        A classic EM/motion-clustering algorithm would compute exactly this
        as its E-step every iteration; here it's simply absent — mask only
        ever receives gradient through the FINAL mask-weighted-mixed
        reconstruction (warp_seg), which can fail to penalise a locally
        wrong assignment if other channels' flexibility (affine/residual)
        quietly compensates for it downstream.
        target = softmax(-per_channel_error / T), DETACHED: the
        pseudo-responsibility a real E-step would compute from the current
        rigid fits. Loss = cross-entropy(target, mask) — pushes assignment
        toward it WITHOUT touching the rigid models themselves (target's
        gradient is cut off; only mask receives gradient here).
        AUXILIARY, not a replacement for warp_seg: this loss only
        supervises RELATIVE assignment quality across channels and would
        collapse to a trivial pass on its own (all channels fit equally
        badly -> uniform target -> uniform mask "satisfies" it for free).
        warp_seg remains what keeps the rigid models themselves accurate in
        absolute terms.
        Returns per-sample [B] loss so it composes with topk.
        """
        per_channel_flow = self._per_channel_rigid_flow(mask, flow)   # [B, C, 2, H, W]
        error = ((per_channel_flow - flow.unsqueeze(1)) ** 2).sum(dim=2)  # [B, C, H, W]
        target = F.softmax(-error.detach() / self.em_consistency_temperature, dim=1)

        log_mask = torch.log(mask.clamp(min=1e-8))
        ce = -(target * log_mask).sum(dim=1)  # [B, H, W]
        return ce.mean(dim=(1, 2))  # [B]

    def _predict_sigma(self, all_pred_residual, head, target_hw):
        """
        Predict per-pixel flow-reconstruction uncertainty sigma > 0, shape
        [B, 1, H, W] matching target_hw. Input is the raw residual-prediction
        tensor [B, 2*mask_layer, H', W'] (already dense, decode_head3's
        output) — resized to target_hw if needed. softplus (not exp) for a
        smoother, more stable positivity mapping early in training; +1e-3
        floor avoids log(0) / division blowup.
        """
        x = all_pred_residual
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return F.softplus(head(x)) + 1e-3

    # ------------------------------------------------------------------ #
    # Override get_demean_affine_flow to avoid MAGMA batched LU          #
    # ------------------------------------------------------------------ #
    def _demean_affine_flow_per_channel(self, mask, flow):
        """
        Shared closed-form solve behind get_demean_affine_flow, stopping
        BEFORE the final mask-weighted collapse across channels — returns
        each channel's own affine-fitted prediction at every pixel,
        [B, C, H, W, 2]. This is the actual per-channel "M-step model"
        prediction; get_demean_affine_flow (below) is just this collapsed
        with mask. Split out so _per_channel_rigid_flow (used by
        _em_consistency_loss) can reuse the exact same fit without
        duplicating the closed-form math.

        torch.linalg.solve dispatches to MAGMA's batched LU on CUDA for
        small matrices.  With batch shape [B, C, 2, 2] = [8, 5, 2, 2],
        the per-matrix stride (20 floats = 80 bytes) is not a power-of-two
        multiple of 128 bytes, which triggers a CUDA misaligned-address
        error (cudaErrorMisalignedAddress 716) inside apply_lu_factor_
        batched_magma.  The closed-form 2×2 inverse is exact and avoids
        MAGMA entirely.
        """
        B, C, H, W = mask.shape
        mask_spatial_normalized = mask / mask.sum(dim=(2, 3), keepdim=True)
        img_preds_1d = torch.flatten(mask_spatial_normalized, 2, 3)  # [B, C, H*W]

        F_u = torch.flatten(flow, 2, 3).permute(0, 2, 1)            # [B, H*W, 2]
        mu_F = torch.bmm(img_preds_1d, F_u)                          # [B, C, 2]
        mu_omega = img_preds_1d @ self.coord_map                      # [B, C, 2]

        F_u_de_mean = F_u[:, None, ...] - mu_F[:, :, None, ...]      # [B, C, H*W, 2]
        u_de_mean = self.coord_map[None, None, ...] - mu_omega[:, :, None, ...]

        F_u_demean_u_demean_T = torch.einsum(
            'b i j k, b i j l -> b i j k l', F_u_de_mean, u_de_mean)
        sigma_F_omega = torch.einsum(
            'b i j, b i j k l -> b i k l', img_preds_1d, F_u_demean_u_demean_T)

        u_demean_u_demean_T = torch.einsum(
            'b i j k, b i j l -> b i j k l', u_de_mean, u_de_mean)
        sigma_omega_omega = torch.einsum(
            'b i j, b i j k l -> b i k l', img_preds_1d, u_demean_u_demean_T)

        # Analytical 2×2 solve: A_star = (sigma_omega_omega^{-1} @ sigma_F_omega^T)^T
        # For A = [[a,b],[c,d]]:  A^{-1} = [[d,-b],[-c,a]] / det(A)
        A = sigma_omega_omega.float()                                  # [B, C, 2, 2]
        a, b = A[..., 0, 0], A[..., 0, 1]
        c, d = A[..., 1, 0], A[..., 1, 1]
        det = (a * d - b * c).clamp(min=1e-6).unsqueeze(-1).unsqueeze(-1)
        A_inv = torch.stack(
            [torch.stack([d, -b], dim=-1),
             torch.stack([-c, a], dim=-1)], dim=-2
        ) / det                                                        # [B, C, 2, 2]
        # X = A_inv @ sigma_F_omega^T  →  A_star = X^T
        RHS = sigma_F_omega.permute(0, 1, 3, 2).float()               # [B, C, 2, 2]
        A_star = torch.matmul(A_inv, RHS).permute(0, 1, 3, 2)         # [B, C, 2, 2]

        F_pred_demean = torch.einsum('b i j k, b i l k -> b i l j', A_star, u_de_mean)
        return F_pred_demean.view(B, C, H, W, 2)   # per-channel, pre-collapse

    def get_demean_affine_flow(self, mask, flow):
        """Identical to the parent's contract: mask-weighted collapse of
        _demean_affine_flow_per_channel across channels -> [B, 2, H, W]."""
        F_pred2_2d = self._demean_affine_flow_per_channel(mask, flow)
        F_pred2_sum_2d = torch.einsum('b i j k, b i j k l -> b l j k', mask, F_pred2_2d)
        return F_pred2_sum_2d

    # ------------------------------------------------------------------ #
    # Override aggregate_flow_with_residual for per-channel residual scale #
    # ------------------------------------------------------------------ #
    def aggregate_flow_with_residual(self, mask, flow, all_pred_residual):
        """
        Identical to the parent implementation, except that when
        use_per_channel_residual_scale=True, residual_adjustment_scale (a
        python float) is replaced by self.residual_scale_per_channel, a
        learnable [mask_layer] vector, applied PER CHANNEL before the sum
        over the channel dimension (the parent multiplies by the scalar
        scale only after summing, so a channel-wise vector cannot simply be
        substituted post-hoc — the weighting must happen inside the sum).
        Falls back to the exact parent behaviour when the flag is False.
        """
        if not self.use_per_channel_residual_scale:
            return super().aggregate_flow_with_residual(mask, flow, all_pred_residual)

        B, C, H, W = mask.shape
        mask_spatial_normalized = mask / mask.view(B, C, H * W, 1).sum(dim=2, keepdim=True)

        flow_agg = self.flow_feat_before_agg(flow)
        flow_agg = flow_agg[:, :, None, ...] * mask_spatial_normalized[:, None, ...]
        flow_agg = flow_agg.flatten(3, 4).sum(dim=-1)
        flow_agg = self.flow_feat_after_agg(flow_agg)
        flow_agg = flow_agg[..., None, None]
        flow_agg = flow_agg * mask[:, None, ...]
        flow_agg = flow_agg.sum(dim=2)

        assert self.free_residual_with_affine or self.free_residual, \
            "use_per_channel_residual_scale requires free_residual or free_residual_with_affine"

        if self.allow_residual_resize and all_pred_residual.shape[-2:] != self.mask_size:
            all_pred_residual = F.interpolate(all_pred_residual, self.mask_size, mode='bilinear')
        all_pred_residual = all_pred_residual.unflatten(1, (2, self.mask_layer))

        # per-channel scale broadcast: [C] -> [1, 1, C, 1, 1], applied before
        # summing over the channel dim (dim=2) so each channel keeps its own
        # residual freedom instead of sharing one global scalar.
        scale = self.residual_scale_per_channel.view(1, 1, -1, 1, 1)
        residual_adjustment = (
            torch.tanh(all_pred_residual / self.pred_div_coeff) * mask[:, None, ...] * scale
        ).sum(dim=2)

        flow_affine = None
        if self.free_residual_with_affine:
            flow_affine = self.get_demean_affine_flow(mask, flow)
            flow_overall = flow_agg + flow_affine + residual_adjustment
        else:
            flow_overall = flow_agg + residual_adjustment

        return flow_overall, flow_agg, residual_adjustment, flow_affine

    # ------------------------------------------------------------------ #
    # Override detect_flow_changes_batch to use self.boundary_threshold   #
    # ------------------------------------------------------------------ #
    def detect_flow_changes_batch(self, flow_data,
                                  threshold=None,
                                  dilation_size=None):
        """Same as V1 but defaults to self.boundary_threshold / self.boundary_dilation,
        and applies boundary_floor so non-boundary pixels keep a small weight."""
        if threshold is None:
            threshold = self.boundary_threshold
        if dilation_size is None:
            dilation_size = self.boundary_dilation
        mask = super().detect_flow_changes_batch(
            flow_data, threshold=threshold, dilation_size=dilation_size)
        if self.boundary_floor > 0:
            mask = mask.clamp(min=self.boundary_floor)
        return mask

    # ------------------------------------------------------------------ #
    # Override forward to use self.topk instead of hard-coded 2           #
    # ------------------------------------------------------------------ #
    def forward(self, imgs, masks, gt_fw_flows, gt_bw_flows,
                all_pred_residual_fw, all_pred_residual_bw, seq_names=None):

        flow_loss = {'seg_fw': 0., 'seg_bw': 0.}
        flows = {'gt_flow': [], 'pred_flow': [], 'agg_flow': [],
                 'residual_adj': [], 'affine_flow': [],
                 'sigma_fw': [], 'sigma_bw': []}

        batch_size, im_num, _, im_h, im_w = imgs.shape
        assert im_num == 2, "Other im_num not implemented"

        individual_losses_fw = []
        individual_losses_bw = []
        individual_scale_fw = []
        individual_scale_bw = []

        from .flow_aggregation_head_with_residual import get_norm_flow

        for i in range(1, im_num):
            mask1 = masks[:, i - 1, :, :, :]
            mask2 = masks[:, i, :, :, :]

            if self.detach_mask_patterns and seq_names is not None:
                mask1 = self._maybe_detach_mask(mask1, seq_names)
                mask2 = self._maybe_detach_mask(mask2, seq_names)

            gt_fw_flow = gt_fw_flows[:, i - 1, ...]
            gt_bw_flow = gt_bw_flows[:, i - 1, ...]

            gt_fw_flow = self.norm_and_clamp_flow(gt_fw_flow, seq_names=seq_names)
            gt_bw_flow = self.norm_and_clamp_flow(gt_bw_flow, seq_names=seq_names)

            # ── cycle-consistency confidence (gates spatial averaging) ──
            if self.use_cycle_conf:
                conf_fw = self._compute_cycle_conf(gt_fw_flow, gt_bw_flow, seq_names=seq_names)
                conf_bw = self._compute_cycle_conf(gt_bw_flow, gt_fw_flow, seq_names=seq_names)
                mask1_agg = mask1 * conf_fw   # low-conf pixels contribute less
                mask2_agg = mask2 * conf_bw
            else:
                mask1_agg, mask2_agg = mask1, mask2

            # ── background flow removal (subtract dominant global motion) ─
            if self.use_bg_affine_removal:
                bg_fw = self._bg_flow(gt_fw_flow)   # median [B,2,1,1] or affine [B,2,H,W]
                bg_bw = self._bg_flow(gt_bw_flow)
                flow_fw_agg = gt_fw_flow - bg_fw   # residual = instrument motion
                flow_bw_agg = gt_bw_flow - bg_bw
            else:
                bg_fw = bg_bw = None
                flow_fw_agg, flow_bw_agg = gt_fw_flow, gt_bw_flow

            fw_flow_overall, fw_flow_agg, fw_residual_adjustment, fw_flow_affine = \
                self.aggregate_flow_with_residual(mask1_agg, flow_fw_agg, all_pred_residual_fw)
            bw_flow_overall, bw_flow_agg, bw_residual_adjustment, bw_flow_affine = \
                self.aggregate_flow_with_residual(mask2_agg, flow_bw_agg, all_pred_residual_bw)

            # add bg back so loss target (gt_fw_flow) remains the full flow
            if bg_fw is not None:
                fw_flow_overall = fw_flow_overall + bg_fw
                bw_flow_overall = bw_flow_overall + bg_bw

            mask_fw_flow = self.detect_flow_changes_batch(gt_fw_flow)
            mask_bw_flow = self.detect_flow_changes_batch(gt_bw_flow)

            if self.use_heteroscedastic_loss:
                sigma_fw = self._predict_sigma(all_pred_residual_fw, self.sigma_head_fw, gt_fw_flow.shape[-2:])
                sigma_bw = self._predict_sigma(all_pred_residual_bw, self.sigma_head_bw, gt_bw_flow.shape[-2:])
                nll_fw = ((gt_fw_flow - fw_flow_overall) ** 2 / (2 * sigma_fw ** 2) + torch.log(sigma_fw)) * mask_fw_flow
                nll_bw = ((gt_bw_flow - bw_flow_overall) ** 2 / (2 * sigma_bw ** 2) + torch.log(sigma_bw)) * mask_bw_flow
                losses_fw = nll_fw.sum(dim=(1, 2, 3)) / (mask_fw_flow.sum(dim=(1, 2, 3)) + 1e-6)
                losses_bw = nll_bw.sum(dim=(1, 2, 3)) / (mask_bw_flow.sum(dim=(1, 2, 3)) + 1e-6)
                # stash for visualization/debugging (e.g. checking sigma highlights
                # blood/specular/occlusion regions) — raw values, not flow-normalised
                flows['sigma_fw'].append(sigma_fw.detach())
                flows['sigma_bw'].append(sigma_bw.detach())
            elif not self.outlier_robust_loss:
                losses_fw = ((gt_fw_flow - fw_flow_overall) ** 2) * mask_fw_flow
                losses_fw = losses_fw.sum(dim=(1, 2, 3)) / (mask_fw_flow.sum(dim=(1, 2, 3)) + 1e-6)
                losses_bw = ((gt_bw_flow - bw_flow_overall) ** 2) * mask_bw_flow
                losses_bw = losses_bw.sum(dim=(1, 2, 3)) / (mask_bw_flow.sum(dim=(1, 2, 3)) + 1e-6)
            else:
                losses_fw = ((((gt_fw_flow - fw_flow_overall).abs()).view(batch_size, -1)
                              + self.eps) ** self.q).mean(dim=1)
                losses_bw = ((((gt_bw_flow - bw_flow_overall).abs()).view(batch_size, -1)
                              + self.eps) ** self.q).mean(dim=1)

            if self.use_flow_metric_loss:
                metric_fw = self._flow_metric_loss(mask1_agg, flow_fw_agg)
                metric_bw = self._flow_metric_loss(mask2_agg, flow_bw_agg)
                losses_fw = losses_fw + self.flow_metric_weight * metric_fw
                losses_bw = losses_bw + self.flow_metric_weight * metric_bw

            if self.use_mask_warp_consistency:
                mask_warp_loss = self._mask_warp_consistency_loss(
                    mask1, mask2, gt_fw_flow, gt_bw_flow, mask_fw_flow, mask_bw_flow)
                losses_fw = losses_fw + self.mask_warp_consistency_weight * mask_warp_loss

            if self.use_em_consistency_loss:
                # mask1_agg/flow_fw_agg (not raw mask1/gt_fw_flow): must match
                # exactly what aggregate_flow_with_residual actually fit its
                # rigid models on. NOTE: if use_cycle_conf is ever combined
                # with this loss, mask1_agg = mask1 * conf_fw no longer sums
                # to 1 across channels, and the cross-entropy's log(mask)
                # term stops being a valid log-probability — v83 (and this
                # loss's only tested config so far) has use_cycle_conf=False,
                # where mask1_agg == mask1 exactly, so this doesn't apply yet.
                em_fw = self._em_consistency_loss(mask1_agg, flow_fw_agg)
                em_bw = self._em_consistency_loss(mask2_agg, flow_bw_agg)
                losses_fw = losses_fw + self.em_consistency_weight * em_fw
                losses_bw = losses_bw + self.em_consistency_weight * em_bw

            individual_losses_fw.append(losses_fw)
            individual_losses_bw.append(losses_bw)

            if self.topk_scale_normalize:
                # Reference GT-flow-magnitude scale per sample, over the same
                # masked region the loss itself uses — so ranking reflects
                # RELATIVE fit quality instead of absolute squared error.
                # Large-motion sources (e.g. CMC bridge pairs) have several x
                # larger GT flow magnitude than adjacent-frame pairs, which
                # inflates raw MSE by the square of that ratio regardless of
                # fit quality — under hard topk this systematically starves
                # them of any gradient (measured: ~7-9x higher raw loss,
                # ~15% topk survival vs ~72% for adjacent pairs, see README).
                # Only used for SORTING here; the loss actually backpropagated
                # (selected_flow_loss below) still uses the raw, unnormalised
                # losses_fw/bw, so gradient magnitude for selected samples is
                # unaffected.
                scale_fw = ((gt_fw_flow ** 2).sum(dim=1, keepdim=True) * mask_fw_flow).sum(dim=(1, 2, 3)) \
                    / (mask_fw_flow.sum(dim=(1, 2, 3)) + 1e-6)
                scale_bw = ((gt_bw_flow ** 2).sum(dim=1, keepdim=True) * mask_bw_flow).sum(dim=(1, 2, 3)) \
                    / (mask_bw_flow.sum(dim=(1, 2, 3)) + 1e-6)
                individual_scale_fw.append(scale_fw)
                individual_scale_bw.append(scale_bw)

            _h, _w, flow, flow2 = get_norm_flow(lis1=gt_fw_flow, lis2=gt_bw_flow)
            flows['gt_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(lis1=fw_flow_overall, lis2=bw_flow_overall)
            flows['pred_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(lis1=fw_flow_agg, lis2=bw_flow_agg)
            flows['agg_flow'].append(torch.cat([flow, flow2], dim=1))

            _h, _w, flow, flow2 = get_norm_flow(lis1=fw_residual_adjustment, lis2=bw_residual_adjustment)
            flows['residual_adj'].append(torch.cat([flow, flow2], dim=1))

            if fw_flow_affine is not None:
                _h, _w, flow, flow2 = get_norm_flow(lis1=fw_flow_affine, lis2=bw_flow_affine)
                flows['affine_flow'].append(torch.cat([flow, flow2], dim=1))

        # ── topk selection (V2: configurable) ──────────────────────────
        total_losses_fw = torch.cat(individual_losses_fw)
        total_losses_bw = torch.cat(individual_losses_bw)
        total_losses = total_losses_fw + total_losses_bw

        if self.topk_scale_normalize:
            # ranking-only signal: relative error, not absolute squared error
            # (see comment at individual_scale_fw/bw). The loss actually
            # backpropagated (selected_flow_loss) always uses the raw
            # total_losses_fw/bw below, unaffected by this.
            total_scale_fw = torch.cat(individual_scale_fw)
            total_scale_bw = torch.cat(individual_scale_bw)
            ranking_losses = total_losses_fw / (total_scale_fw + 1e-6) + total_losses_bw / (total_scale_bw + 1e-6)
        else:
            ranking_losses = total_losses

        if self.topk_mode == 'soft':
            if len(ranking_losses) <= 1:
                weights = torch.ones_like(ranking_losses)
            else:
                std = ranking_losses.std()
                normalized = (ranking_losses - ranking_losses.mean()) / (std + 1e-6)
                weights = F.softmax(-normalized / self.topk_soft_temperature, dim=0)  # [B], sums to 1

            selected_flow_loss = {
                'seg_fw': (weights * total_losses_fw).sum(),
                'seg_bw': (weights * total_losses_bw).sum()
            }
            # for visualization only (loss above already uses all samples) —
            # order by weight so the logged frames are still the easiest first
            _, selected_indices = torch.sort(weights, descending=True)
        else:
            sorted_losses, sorted_indices = torch.sort(ranking_losses)
            # Clamp topk to batch size to avoid index out of range
            k = min(self.topk, len(ranking_losses))
            selected_indices = sorted_indices[:k]

            selected_flow_loss = {
                'seg_fw': total_losses_fw[selected_indices].mean(),
                'seg_bw': total_losses_bw[selected_indices].mean()
            }
        selected_flow_loss['seg'] = selected_flow_loss['seg_fw'] + selected_flow_loss['seg_bw']

        selected_flows = {}
        for key, value in flows.items():
            if len(value) >= len(selected_indices):
                selected_flows[key] = [value[i] for i in selected_indices]
            else:
                selected_flows[key] = value.copy()

        return selected_flows, selected_flow_loss
