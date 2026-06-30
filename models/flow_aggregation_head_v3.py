"""
FlowAggregationHeadV3

Two improvements over V2:

1. GroupNorm in flow_feat_before_agg
   V1/V2: Conv → LeakyReLU → Conv → LeakyReLU  (no normalisation, bias=True)
   V3:    Conv → GN(8) → ReLU → Conv → GN(8) → ReLU  (bias=False)
   GN stabilises the flow feature distribution across different motion magnitudes
   without the batch-size dependency of BN.

2. Flow-magnitude-weighted aggregation
   V1/V2: mask-weighted average (each pixel weighted by its mask probability)
   V3:    (mask × magnitude_weight)-weighted average

   magnitude_weight = clamp(|flow| / mean(|flow|), max=mag_clamp)

   Pixels with above-average motion (fast-moving instruments) contribute more;
   nearly-static pixels (tissue background) are down-weighted. This directly
   addresses the instrument signal being washed out by tissue in the aggregated
   flow prototype.

Config (decode_head):
  type: FlowAggregationHeadV3
  topk: 4
  boundary_threshold: 0.1745
  use_mag_weight: true    # default true
  mag_clamp: 3.0          # default 3.0; clamps per-pixel magnitude ratio
  ... (all other V2 kwargs unchanged)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2


class FlowAggregationHeadV3(FlowAggregationHeadWithResidualV2):

    def __init__(self, *args,
                 use_mag_weight: bool = True,
                 mag_clamp: float = 3.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mag_weight = use_mag_weight
        self.mag_clamp = mag_clamp

        # ── Replace flow_feat_before_agg with GN version ──────────────────────
        # Inherit kernel size from whatever was set via flow_feat_before_agg_kernel_size
        old = self.flow_feat_before_agg
        k = old[0].kernel_size[0]
        pad = k // 2
        m = self.num_flow_feat_channels
        num_groups = min(8, m)  # GN groups; m must be divisible by num_groups

        self.flow_feat_before_agg = nn.Sequential(
            nn.Conv2d(2, m, k, padding=pad, bias=False),
            nn.GroupNorm(num_groups, m),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, m, k, padding=pad, bias=False),
            nn.GroupNorm(num_groups, m),
            nn.ReLU(inplace=True),
        )

    # ── Override aggregate_flow_with_residual to inject magnitude weighting ───
    def aggregate_flow_with_residual(self, mask, flow, all_pred_residual):
        B, C, H, W = mask.shape
        mask_spatial_normalized = mask / mask.view(B, C, H * W, 1).sum(dim=2, keepdim=True)

        # Flow feature extraction (now with GN)
        flow_feat = self.flow_feat_before_agg(flow)  # [B, m, H, W]

        # ── Magnitude-weighted spatial pooling ─────────────────────────────────
        if self.use_mag_weight:
            # Per-pixel flow magnitude, normalised by per-image mean
            mag = flow.norm(dim=1, keepdim=True)                           # [B,1,H,W]
            mag_w = mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-6)     # relative weight
            mag_w = mag_w.clamp(max=self.mag_clamp)                        # [B,1,H,W]
            # Multiply mask probability by magnitude weight, then renormalise
            # so the per-channel weights still sum to 1 spatially
            pool_w = mask_spatial_normalized * mag_w                       # [B,C,H,W]
            pool_w = pool_w / (pool_w.sum(dim=(2, 3), keepdim=True) + 1e-6)
        else:
            pool_w = mask_spatial_normalized

        # Mask-weighted global pool: [B,m,1,H,W] × [B,1,C,H,W] → [B,m,C,H*W] → [B,m,C]
        flow_agg = flow_feat[:, :, None, ...] * pool_w[:, None, ...]
        flow_agg = flow_agg.flatten(3, 4).sum(dim=-1)                     # [B, m, C]

        # Per-channel flow prediction: [B, 2, C]
        flow_agg = self.flow_feat_after_agg(flow_agg)
        # Spread back to spatial: [B, 2, C, 1, 1] × [B, 1, C, H, W] → sum → [B, 2, H, W]
        flow_agg = (flow_agg[..., None, None] * mask[:, None, ...]).sum(dim=2)

        # ── Residual logic (unchanged from V1/V2) ─────────────────────────────
        flow_affine = None
        residual_adjustment = torch.zeros_like(flow_agg)

        if self.free_residual:
            if self.allow_residual_resize and all_pred_residual.shape[-2:] != self.mask_size:
                all_pred_residual = F.interpolate(all_pred_residual, self.mask_size, mode='bilinear')
            all_pred_residual = all_pred_residual.unflatten(1, (2, self.mask_layer))
            if self.residual_adjustment_scale != -1.:
                residual_adjustment = (
                    torch.tanh(all_pred_residual / self.pred_div_coeff) * mask[:, None, ...]
                ).sum(dim=2) * self.residual_adjustment_scale
            else:
                residual_adjustment = (all_pred_residual * mask[:, None, ...]).sum(dim=2)
            flow_overall = flow_agg + residual_adjustment

        elif self.free_residual_with_affine:
            flow_affine = self.get_demean_affine_flow(mask, flow)
            if self.allow_residual_resize and all_pred_residual.shape[-2:] != self.mask_size:
                all_pred_residual = F.interpolate(all_pred_residual, self.mask_size, mode='bilinear')
            all_pred_residual = all_pred_residual.unflatten(1, (2, self.mask_layer))
            residual_adjustment = (
                torch.tanh(all_pred_residual / self.pred_div_coeff) * mask[:, None, ...]
            ).sum(dim=2) * self.residual_adjustment_scale
            flow_overall = flow_agg + flow_affine + residual_adjustment

        else:
            flow_overall = flow_agg

        return flow_overall, flow_agg, residual_adjustment, flow_affine
