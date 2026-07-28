"""
DeformableCrossFrameAttention: Deformable-DETR-style attention (Zhu et al.,
"Deformable DETR", 2020) adapted to cross-FRAME (not cross-image-to-query)
attention at a single backbone scale, same-scale only (no cross-scale
mixing -- that's ASPP's job inside MultiScaleSegHead, see the design
discussion 260720 in README).

For each spatial position in frame_i's feature map, a small conv head
predicts K learned (possibly fractional) sampling offsets INTO frame_j's
feature map (no external flow, no fixed window -- offsets are learned
purely from the query feature itself), plus K softmax weights. The K
sampled values (bilinear, via grid_sample -- same normalization convention
as utils.warp_utils.flow_warp) are weighted-summed -> one attended feature
per query position, same spatial shape as the input. Multi-head, heads'
outputs concatenated then projected to out_channels.

Symmetric: one forward() call internally computes BOTH directions
(frame_i querying frame_j for joint_feat_i, and frame_j querying frame_i
for joint_feat_j) using the SAME shared weights, and returns them
re-interleaved back into the [B*2, ...] batched-pair layout every decode
head in this codebase already expects (matches
RCFJointMaskSoftTissueModel/RCFJointMaskV2SoftTissueModel's existing
[B*2, C, H, W] convention for all_feat).

Drop-in replacement for JointFrameFeatProjector (models/joint_frame_feat.py)
at the MODEL level -- output shape/semantics [B*2, out_channels, H, W] are
identical, so MultiScaleSegHeadJoint4 (models/multi_scale_seg_head_joint4.py)
needs ZERO changes to consume this instead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import mesh_grid, norm_grid


class DeformableCrossFrameAttention(nn.Module):
    def __init__(self, channels: int, out_channels: int = 64, heads: int = 8, num_points: int = 4):
        super().__init__()
        assert channels % heads == 0, f"channels={channels} must be divisible by heads={heads}"
        self.channels = channels
        self.heads = heads
        self.num_points = num_points
        self.head_dim = channels // heads

        self.value_proj = nn.Conv2d(channels, channels, kernel_size=1)
        # per head: K offsets (2 each) + K weight logits = heads * K * 3 output channels
        self.offset_weight_proj = nn.Conv2d(channels, heads * num_points * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, out_channels, kernel_size=1)

        # Deformable DETR init: start offsets near zero (query looks at its
        # own position first, learns to look elsewhere as training
        # progresses) -- sensible here too since gap1 (the dominant training
        # gap, see dataset/data.py's gap_options=[1,2,3] @ [.7,.2,.1]) has
        # small real displacement (mean ~8px, measured this session).
        nn.init.zeros_(self.offset_weight_proj.weight)
        nn.init.zeros_(self.offset_weight_proj.bias)

    def _attend_one_direction(self, query_feat: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        """query_feat, value_feat: [B, C, H, W] (single frame each, already split).
        Returns: [B, out_channels, H, W]."""
        B, C, H, W = query_feat.shape
        heads, K, head_dim = self.heads, self.num_points, self.head_dim

        v = self.value_proj(value_feat)                      # [B, C, H, W]
        v = v.view(B, heads, head_dim, H, W).reshape(B * heads, head_dim, H, W)

        raw = self.offset_weight_proj(query_feat)             # [B, heads*K*3, H, W]
        raw = raw.view(B, heads, K, 3, H, W)
        offsets = raw[:, :, :, :2]                             # [B, heads, K, 2, H, W]  (pixel-space Δx, Δy)
        weight_logits = raw[:, :, :, 2]                        # [B, heads, K, H, W]
        weights = F.softmax(weight_logits, dim=2)              # normalize over K

        base_grid = mesh_grid(B, H, W).type_as(query_feat)     # [B, 2, H, W], pixel-space (x, y)
        base_grid_bh = (base_grid.unsqueeze(1)                 # [B, 1, 2, H, W]
                        .expand(-1, heads, -1, -1, -1)
                        .reshape(B * heads, 2, H, W))

        attended = query_feat.new_zeros(B * heads, head_dim, H, W)
        for k in range(K):
            offset_k = offsets[:, :, k].reshape(B * heads, 2, H, W)
            sample_grid_px = base_grid_bh + offset_k            # [B*heads, 2, H, W]
            sample_grid_norm = norm_grid(sample_grid_px)        # [B*heads, H, W, 2], in [-1, 1]
            sampled = F.grid_sample(v, sample_grid_norm, mode='bilinear',
                                    padding_mode='border', align_corners=True)  # [B*heads, head_dim, H, W]
            weight_k = weights[:, :, k].reshape(B * heads, 1, H, W)
            attended = attended + weight_k * sampled

        attended = attended.view(B, heads * head_dim, H, W)     # [B, C, H, W]
        return self.out_proj(attended)                          # [B, out_channels, H, W]

    def forward(self, feat_pair: torch.Tensor) -> torch.Tensor:
        """feat_pair: [B*2, C, H, W] (both frames batched, matches all_feat[scale_idx]
        exactly). Returns: [B*2, out_channels, H, W] -- index 0::2-style layout
        (frame_i's joint feature at the frame_i position, frame_j's at frame_j's),
        same convention as everything else in this codebase's im_num=2 batching."""
        total, C, H, W = feat_pair.shape
        assert total % 2 == 0, f"DeformableCrossFrameAttention requires im_num==2, got batch={total}"
        B = total // 2
        pair = feat_pair.view(B, 2, C, H, W)
        feat_i, feat_j = pair[:, 0], pair[:, 1]

        joint_i = self._attend_one_direction(feat_i, feat_j)    # frame_i queries frame_j
        joint_j = self._attend_one_direction(feat_j, feat_i)    # frame_j queries frame_i

        out = torch.stack([joint_i, joint_j], dim=1)            # [B, 2, out_channels, H, W]
        return out.reshape(total, *out.shape[2:])                # [B*2, out_channels, H, W]
