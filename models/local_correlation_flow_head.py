"""
LocalCorrelationFlowHead / LocalCorrelationFlowHeadBidirectional: a self-
taught, coarse-to-fine dense flow estimator, added 260728 to remove this
project's dependence on RAFT-precomputed optical flow entirely.

Motivation (session discussion + script/diagnose_v102_bottleneck.py):
bottleneck diagnosis found mask miss-rate on true foreground pixels
correlates strongly with RAFT's OWN cycle-consistency confidence (3x+ gap
between low/high-confidence bins) -- i.e. RAFT's domain-transfer error
(trained on synthetic data, applied to endoscopic low-light/specular
footage) is a bigger error source here than anything wrong with the ResNet
appearance features. Rather than trying to clean up RAFT's output, this
module lets the model learn its own motion field directly from THIS
dataset via reconstruction, so it can never inherit RAFT's domain gap.

Reuses the exact local-window correlation mechanism from
LocalCorrelationCrossFrameAttention (models/local_correlation_cross_frame_attention.py,
built 260724) -- scaled dot-product similarity over a KxK local window via
F.unfold, softmax-normalised. That module CONSUMES the softmax weights to
aggregate VALUE features (for feeding the mask decoder); this module instead
takes the same softmax weights and computes a soft-argmax EXPECTED
DISPLACEMENT (weighted sum of the window's integer offset vectors) -- the
standard way to turn a correlation volume into an explicit flow vector
(FlowNet/PWC-Net's simplified correlation-to-flow heads).

Coarse-to-fine (PWC-Net style), NOT a single fixed-radius window: a single
small window (e.g. v138's radius=4, 9x9) cannot cover this dataset's full
motion range (README notes bridge/gap2/gap3 flow magnitudes up to 19-22px at
full resolution) without either an infeasibly large window (memory blows up
quadratically in window side) or missing large motions entirely -- which is
exactly the problem RAFT's global correlation + iterative refinement was
originally chosen to solve. Here the SAME two ResNet scales already computed
for the rest of the model stand in for a coarse-to-fine pyramid:
  - coarse stage: correlate at the coarser scale (e.g. feat2, 1024ch, 48x48)
    -- each grid cell covers more original-image pixels, so a small window
    in FEATURE space covers a much larger window in IMAGE space.
  - fine stage: upsample+rescale the coarse flow, WARP the fine-scale
    features (e.g. feat0, 256ch, 96x96) with it, then correlate only the
    residual disparity left after warping (small window suffices, since
    warping already did the "long-distance" work).

This is a plain, DETERMINISTIC flow estimator (soft-argmax, one forward
pass) -- no stochastic sampling, no ODE integration, no ConditionalFlowMatching
machinery, per the session's conclusion that a generative/multi-step
formulation is unwarranted when the frame0->frame1 correspondence is assumed
unique (not genuinely multimodal).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import flow_warp


def _make_offset_grid(radius: int) -> torch.Tensor:
    """[2, K*K] constant (dx, dy) offsets for every candidate in a
    (2*radius+1)^2 window, matching F.unfold's row-major patch ordering."""
    K = 2 * radius + 1
    ys, xs = torch.meshgrid(
        torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1),
        indexing="ij")
    offsets = torch.stack([xs, ys], dim=0).float()   # [2, K, K]  (dx, dy)
    return offsets.reshape(2, K * K)


class LocalCorrelationFlowHead(nn.Module):
    """One-directional (frame_i -> frame_j) coarse-to-fine flow estimator."""

    def __init__(self, coarse_channels: int, fine_channels: int,
                 proj_channels: int = 32, coarse_radius: int = 4, fine_radius: int = 3):
        super().__init__()
        self.proj_channels = proj_channels
        self.coarse_radius = coarse_radius
        self.fine_radius = fine_radius
        self.scale = proj_channels ** -0.5

        self.coarse_query_proj = nn.Conv2d(coarse_channels, proj_channels, kernel_size=1)
        self.coarse_key_proj = nn.Conv2d(coarse_channels, proj_channels, kernel_size=1)
        self.fine_query_proj = nn.Conv2d(fine_channels, proj_channels, kernel_size=1)
        self.fine_key_proj = nn.Conv2d(fine_channels, proj_channels, kernel_size=1)

        self.register_buffer("coarse_offsets", _make_offset_grid(coarse_radius))
        self.register_buffer("fine_offsets", _make_offset_grid(fine_radius))

    def _soft_argmax_flow(self, query_feat, key_feat, query_proj, key_proj,
                          radius, offsets):
        """Local-window correlation (identical mechanics to
        LocalCorrelationCrossFrameAttention._attend_one_direction) reduced
        to an expected-displacement vector instead of an aggregated feature.
        query_feat, key_feat: [B, C, H, W]. Returns flow [B, 2, H, W] in
        THIS tensor's own grid pixel units (caller rescales when changing
        resolution)."""
        B, C, H, W = query_feat.shape
        K = 2 * radius + 1
        P = self.proj_channels

        q = query_proj(query_feat)
        k = key_proj(key_feat)
        k_pad = F.pad(k, [radius, radius, radius, radius])
        k_unfold = F.unfold(k_pad, kernel_size=K).view(B, P, K * K, H, W)

        corr = (q.unsqueeze(2) * k_unfold).sum(dim=1) * self.scale   # [B, K*K, H, W]
        weights = F.softmax(corr, dim=1)

        off = offsets.to(q.device, q.dtype).view(1, 2, K * K, 1, 1)
        flow = (weights.unsqueeze(1) * off).sum(dim=2)               # [B, 2, H, W]
        return flow

    def forward(self, coarse_i, coarse_j, fine_i, fine_j):
        """coarse_i/j: [B, Cc, Hc, Wc]. fine_i/j: [B, Cf, Hf, Wf], Hf=k*Hc.
        Returns flow_i_to_j: [B, 2, Hf, Wf], in FINE grid pixel units."""
        flow_coarse = self._soft_argmax_flow(
            coarse_i, coarse_j, self.coarse_query_proj, self.coarse_key_proj,
            self.coarse_radius, self.coarse_offsets)                  # coarse-grid units

        Hf, Wf = fine_i.shape[-2:]
        Hc, Wc = flow_coarse.shape[-2:]
        flow_init = F.interpolate(flow_coarse, size=(Hf, Wf), mode="bilinear", align_corners=False)
        rescale = torch.tensor([Wf / Wc, Hf / Hc], device=flow_init.device, dtype=flow_init.dtype)
        flow_init = flow_init * rescale.view(1, 2, 1, 1)              # now in fine-grid units

        fine_j_warped = flow_warp(fine_j, flow_init, pad="border")
        flow_residual = self._soft_argmax_flow(
            fine_i, fine_j_warped, self.fine_query_proj, self.fine_key_proj,
            self.fine_radius, self.fine_offsets)

        return flow_init + flow_residual


class LocalCorrelationFlowHeadBidirectional(nn.Module):
    """Wraps LocalCorrelationFlowHead to produce both directions from a
    [B*2, C, H, W]-batched feature pair (same im_num=2 convention as
    LocalCorrelationCrossFrameAttention.forward), for the two ResNet scales
    used as coarse/fine."""

    def __init__(self, coarse_channels: int, fine_channels: int,
                 proj_channels: int = 32, coarse_radius: int = 4, fine_radius: int = 3):
        super().__init__()
        self.core = LocalCorrelationFlowHead(
            coarse_channels, fine_channels, proj_channels, coarse_radius, fine_radius)

    def forward(self, coarse_pair: torch.Tensor, fine_pair: torch.Tensor):
        """coarse_pair: [B*2, Cc, Hc, Wc], fine_pair: [B*2, Cf, Hf, Wf].
        Returns (flow_fw, flow_bw), each [B, 2, Hf, Wf]:
          flow_fw: frame0 -> frame1
          flow_bw: frame1 -> frame0
        """
        Bc2, Cc, Hc, Wc = coarse_pair.shape
        Bf2, Cf, Hf, Wf = fine_pair.shape
        assert Bc2 == Bf2 and Bc2 % 2 == 0
        B = Bc2 // 2

        coarse = coarse_pair.view(B, 2, Cc, Hc, Wc)
        fine = fine_pair.view(B, 2, Cf, Hf, Wf)

        flow_fw = self.core(coarse[:, 0], coarse[:, 1], fine[:, 0], fine[:, 1])
        flow_bw = self.core(coarse[:, 1], coarse[:, 0], fine[:, 1], fine[:, 0])
        return flow_fw, flow_bw

    def forward_asymmetric(self, coarse_query_pair, coarse_key_pair, fine_query_pair, fine_key_pair):
        """
        JEPA-style variant (added 260728, see models/rcf_jepa_flow_model.py):
        the query side (own-frame content initiating the search) and the
        key/value side (other-frame content being matched against) come
        from TWO DIFFERENT encoders -- typically an actively-trained
        "context" encoder for the query and a stop-gradient/EMA "target"
        encoder for the key/value, instead of forward()'s same-encoder
        convention. Reuses self.core UNCHANGED -- LocalCorrelationFlowHead's
        correlation math has always taken query/key/value as separate
        tensors, this only changes WHICH encoder's output gets routed into
        which argument.

        coarse_query_pair, fine_query_pair: [B*2, C, H, W] from the context
            encoder (both frames, own-encoder convention, e.g. all_feat[i]).
        coarse_key_pair, fine_key_pair: [B*2, C, H, W] from the target
            encoder (both frames, e.g. all_feat_ema[i]).
        Returns (flow_fw, flow_bw): flow_fw uses frame0(query)->frame1(key),
        flow_bw uses frame1(query)->frame0(key) -- same convention as
        forward().
        """
        Bc2, Cc, Hc, Wc = coarse_query_pair.shape
        Bf2, Cf, Hf, Wf = fine_query_pair.shape
        B = Bc2 // 2

        cq = coarse_query_pair.view(B, 2, Cc, Hc, Wc)
        ck = coarse_key_pair.view(B, 2, Cc, Hc, Wc)
        fq = fine_query_pair.view(B, 2, Cf, Hf, Wf)
        fk = fine_key_pair.view(B, 2, Cf, Hf, Wf)

        flow_fw = self.core(cq[:, 0], ck[:, 1], fq[:, 0], fk[:, 1])
        flow_bw = self.core(cq[:, 1], ck[:, 0], fq[:, 1], fk[:, 0])
        return flow_fw, flow_bw
