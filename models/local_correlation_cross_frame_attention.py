"""
LocalCorrelationCrossFrameAttention: replaces DeformableCrossFrameAttention's
"blind" offset regression (offset_weight_proj predicts where to look purely
from the query's OWN content, never checking what's actually at the target
location) with an explicit local-window content correlation, discussed
260724 -- the same mechanism classical/deep optical flow uses to find
correspondence (block matching / Lucas-Kanade's search window; FlowNet's
Correlation Layer; PWC-Net's cost volume), just consumed differently: instead
of regressing an explicit flow vector from the correlation surface, we
softmax it into attention weights and aggregate the VALUE content at each
candidate offset -- the query "actively checks" whether a candidate position
in the other frame actually looks similar before trusting it, rather than
guessing an offset from its own appearance alone and hoping gradient descent
eventually calibrates that guess correctly.

Same same-scale-only, no-external-flow-dependency constraints as
DeformableCrossFrameAttention (see that file's docstring) -- this is a
drop-in replacement at the model level, output shape/semantics
[B*2, out_channels, H, W] are identical, so MultiScaleSegHeadJoint4 needs
ZERO changes to consume this instead.

Design (memory-bounded, see session discussion for why full O(N^2) attention
at feat0's 96x96 resolution was already ruled out ~43GB): correlation is
NOT computed against the whole other frame, only a local window of radius R
around each query position (K = 2R+1 candidates per axis, K^2 total) --
same "sparse local search" spirit as Deformable attention's K sampling
points, except now the K candidates are a dense regular window (like a
correlation volume) instead of learned offsets, and scored by actual content
similarity instead of predicted blindly.

Implementation uses F.unfold to extract all K^2 candidate patches from the
(padded) key/value maps in one vectorized op (no python loop over
candidates) -- query/key/value are all projected down to a small
proj_channels dimension FIRST (before correlation), both to bound memory
(the unfold tensor is [B, proj_channels, K^2, H, W] -- scales with
proj_channels x K^2 x H x W, so keeping proj_channels modest matters far
more here than for the deformable version) and because raw high-dimensional
backbone features (256-2048ch) are overkill for a similarity comparison
that's ultimately reduced to a single scalar per candidate.

radius default 4 (window 9x9=81 candidates) is a deliberately conservative
starting point given gap1 (the dominant sampled gap, 70% of training draws)
has a measured mean/median flow magnitude of ~8.0/6.4px this session's
flow-quality analysis -- a wider radius would cover more of the motion
distribution but costs correlation memory quadratically in K, so this is
tunable per scale/config rather than hardcoded larger by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCorrelationCrossFrameAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int = 64,
                 proj_channels: int = 32, radius: int = 4):
        super().__init__()
        self.radius = radius
        self.proj_channels = proj_channels
        self.scale = proj_channels ** -0.5

        self.query_proj = nn.Conv2d(channels, proj_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(channels, proj_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(channels, proj_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(proj_channels, out_channels, kernel_size=1)

    def _attend_one_direction(self, query_feat: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        """query_feat, value_feat: [B, C, H, W] (single frame each, already split).
        Returns: [B, out_channels, H, W]."""
        B, C, H, W = query_feat.shape
        r = self.radius
        K = 2 * r + 1
        P = self.proj_channels

        q = self.query_proj(query_feat)   # [B, P, H, W]
        k = self.key_proj(value_feat)     # [B, P, H, W]
        v = self.value_proj(value_feat)   # [B, P, H, W]

        k_pad = F.pad(k, [r, r, r, r])
        v_pad = F.pad(v, [r, r, r, r])
        # F.unfold extracts every KxK patch in one vectorized op -- no python
        # loop over the K^2 candidate offsets (unlike DeformableCrossFrameAttention's
        # loop over K=4 learned points, which is cheap enough at K=4 but would
        # be wasteful here at K^2=81+ candidates).
        k_unfold = F.unfold(k_pad, kernel_size=K).view(B, P, K * K, H, W)
        v_unfold = F.unfold(v_pad, kernel_size=K).view(B, P, K * K, H, W)

        # scaled dot-product correlation between query and every candidate
        # position in the local window -- the actual "does this look similar"
        # check that DeformableCrossFrameAttention's blind regression never does.
        corr = (q.unsqueeze(2) * k_unfold).sum(dim=1) * self.scale   # [B, K*K, H, W]
        weights = F.softmax(corr, dim=1)                              # normalize over the window
        attended = (weights.unsqueeze(1) * v_unfold).sum(dim=2)       # [B, P, H, W]
        return self.out_proj(attended)                                # [B, out_channels, H, W]

    def forward(self, feat_pair: torch.Tensor) -> torch.Tensor:
        """feat_pair: [B*2, C, H, W] (both frames batched, matches all_feat[scale_idx]
        exactly). Returns: [B*2, out_channels, H, W], same im_num=2 batching
        convention as DeformableCrossFrameAttention."""
        total, C, H, W = feat_pair.shape
        assert total % 2 == 0, f"LocalCorrelationCrossFrameAttention requires im_num==2, got batch={total}"
        B = total // 2
        pair = feat_pair.view(B, 2, C, H, W)
        feat_i, feat_j = pair[:, 0], pair[:, 1]

        joint_i = self._attend_one_direction(feat_i, feat_j)    # frame_i queries frame_j's local window
        joint_j = self._attend_one_direction(feat_j, feat_i)    # frame_j queries frame_i's local window

        out = torch.stack([joint_i, joint_j], dim=1)            # [B, 2, out_channels, H, W]
        return out.reshape(total, *out.shape[2:])                # [B*2, out_channels, H, W]
