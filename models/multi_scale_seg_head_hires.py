"""
MultiScaleSegHeadHiRes: MultiScaleSegHead + a learned upsample-refine stage
that doubles output resolution (H/4 -> H/2, e.g. 96x96 -> 192x192 for a
384x384 crop), using the backbone2 stem's pre-maxpool activation (stride2,
64ch -- see models/rcf_model.py's stem-feature hook) as a genuine higher-
resolution detail source, discussed 260731.

Motivation: v102's decode_head2 output (and therefore everything downstream
-- mask_size, GT-flow-downsample target, decode_head's motion fit) has
always been bottlenecked at H/4 (96x96 for a 384 crop) -- feat0 itself, the
finest-resolution feature backbone2 exposes via out_indices, is already at
that resolution, and MultiScaleSegHead's own output never exceeds it. The
backbone stem (conv1+norm1+relu, BEFORE maxpool) computes a stride-2 (H/2,
e.g. 192x192), 64-channel activation that is discarded today (out_indices
only captures the 4 res_layer outputs) -- but it costs ZERO extra compute
(it's produced on every forward pass regardless, just not returned) and is
the only genuinely higher-resolution signal backbone2 has to offer without
changing its stride/dilation structure (which would touch the pretrained
weights' stride alignment and every other config).

Design -- safe, provable no-op at initialization:
  1. Take MultiScaleSegHead.forward_features()'s output x [B,mid,H/4,W/4]
     UNCHANGED (parent class, not touched by this file).
  2. x_up = bilinear-upsample(x) to H/2 -- a FIXED, non-trainable op.
  3. stem_feat [B,64,H/2,W/2] (captured via forward hook, see rcf_model.py)
     is projected + concatenated with x_up, refined by two conv layers, the
     LAST of which is zero-initialized -- so delta == 0 everywhere at init,
     regardless of what the other (randomly-initialized) layers before it
     produce.
  4. x_hires = x_up + delta == x_up EXACTLY at init.
  5. Classify with the SAME conv_seg inherited from the parent class (a 1x1
     conv -- resolution-agnostic, and because it's linear + a per-channel
     constant bias, it PROVABLY commutes with bilinear interpolation:
     conv_seg(bilinear_upsample(x)) == bilinear_upsample(conv_seg(x))
     exactly). So at initialization, this class's H/2 output is
     mathematically identical to bilinearly upsampling what v102's own H/4
     output would have been for the same input and the same conv_seg
     weights -- not just empirically close, but exact by construction. A
     checkpoint fine-tuned from v102 starts this class at v102's own
     (upsampled) predictions, with zero cold-start disruption, before any
     of the new hires_refine/stem_proj weights have learned anything.

This is the same "zero-init no-op" safety principle used throughout this
session's DINO-graph additions (v146/v149), applied here to a resolution
upgrade instead of a semantic-signal fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_scale_seg_head import MultiScaleSegHead


class MultiScaleSegHeadHiRes(MultiScaleSegHead):
    def __init__(self, *args, stem_channels: int = 64, hires_channels: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        # Consumed by rcf_model.py's hasattr-guard to know to capture/pass
        # the backbone stem feature -- see that file's use_stem_feat check.
        self.use_stem_feat = True

        mid_channels = self.decode_conv2[0].out_channels  # matches parent's mid_channels

        self.stem_proj = nn.Sequential(
            nn.Conv2d(stem_channels, hires_channels, 1, bias=False),
            nn.BatchNorm2d(hires_channels),
            nn.ReLU(inplace=True),
        )
        self.hires_refine1 = nn.Sequential(
            nn.Conv2d(mid_channels + hires_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.hires_refine2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        nn.init.zeros_(self.hires_refine2.weight)
        nn.init.zeros_(self.hires_refine2.bias)

    def forward(self, inputs, flow_feat=None, stem_feat=None):
        """
        Returns: [B, num_classes, H/2, W/2] (double MultiScaleSegHead's H/4
        resolution). stem_feat: [B, stem_channels, H/2', W/2'] or None
        (captured externally, see models/rcf_model.py) -- resized to match
        x_up if its resolution doesn't line up exactly (e.g. odd crop sizes).
        """
        x = self.forward_features(inputs, flow_feat=flow_feat)          # [B, mid, H/4, W/4]
        x_up = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        # [B, mid, H/2, W/2] -- fixed, non-trainable

        if stem_feat is not None:
            if stem_feat.shape[-2:] != x_up.shape[-2:]:
                stem_feat = F.interpolate(
                    stem_feat, size=x_up.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            s = self.stem_proj(stem_feat)
        else:
            s = x_up.new_zeros(x_up.shape[0], self.stem_proj[0].out_channels, *x_up.shape[-2:])

        delta = self.hires_refine1(torch.cat([x_up, s], dim=1))
        delta = self.hires_refine2(delta)          # zero-init -> delta == 0 at init
        x_hires = x_up + delta                     # == x_up exactly at init

        if self.dropout is not None:
            x_hires = self.dropout(x_hires)
        return self.conv_seg(x_hires)
        # [B, num_classes, H/2, W/2]
