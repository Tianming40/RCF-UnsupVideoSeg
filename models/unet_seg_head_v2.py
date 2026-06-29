"""
UNetSegHeadV2
=============
True multi-resolution U-Net / FPN decoder.

Requires backbone2 with standard strides (no dilation):
  dilations: [1, 1, 1, 1]
  strides:   [1, 2, 2, 2]

This gives four distinct spatial resolutions:
  feat0: [B, 256,  H/4,  W/4]   (96×96 for 384 input)
  feat1: [B, 512,  H/8,  W/8]   (48×48)
  feat2: [B, 1024, H/16, W/16]  (24×24)
  feat3: [B, 2048, H/32, W/32]  (12×12)

Top-down decoder (FPN-style lateral + add, concat only at final level):

  feat3 [H/32] → proj3 → stage1_conv → x1 → aux_seg1
                           ↓ upsample
  feat2 [H/16] → lateral2 ─→ add → stage2_conv → x2 → aux_seg2
                                      ↓ upsample
  feat1 [H/8]  → lateral1 ─→ add → stage3_conv → x3 → aux_seg3
                                      ↓ upsample
  feat0 [H/4]  ────────── concat → decode_conv1 → decode_conv2 → conv_seg

Lateral connections use add (FPN style) at H/32→H/16→H/8 for efficiency.
Final concat with feat0 at H/4 preserves maximum fine-grained detail.

Interface is identical to UNetSegHead / MultiScaleSegHead:
  forward(inputs, flow_feat=None) → [B, num_classes, H/4, W/4]
  self.last_aux_logits = [aux1(H/32), aux2(H/16), aux3(H/8)]

Optional add-ons (applied at H/4 before concat with feat0):
  use_edge_feat — Sobel on feat0
  use_flow_feat — gt flow features (training only, dropped with prob flow_drop_p)

Config example (must pair with matching backbone2 strides):
  backbone2:
    dilations: [1, 1, 1, 1]
    strides:   [1, 2, 2, 2]

  decode_head2:
    type: UNetSegHeadV2
    num_classes: 5
    mid_channels: 256
    feat_channels: [256, 512, 1024, 2048]
    dropout_ratio: 0.1
    align_corners: false
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetSegHeadV2(nn.Module):
    """
    Args:
        num_classes      (int)   : segmentation channels
        mid_channels     (int)   : internal feature dimension
        feat_channels    (list)  : [c0, c1, c2, c3] from backbone
        dropout_ratio    (float) : Dropout2d ratio; 0 disables
        align_corners    (bool)  : F.interpolate align_corners
        use_edge_feat    (bool)  : Sobel edge enhancement on feat0 (applied at H/4)
        use_flow_feat    (bool)  : optical flow guidance at H/4 (training only)
        flow_in_channels (int)   : channels of incoming flow feature
        flow_drop_p      (float) : probability of dropping flow during training
    """

    def __init__(
        self,
        num_classes: int = 5,
        mid_channels: int = 256,
        feat_channels=(256, 512, 1024, 2048),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_edge_feat: bool = False,
        use_flow_feat: bool = False,
        flow_in_channels: int = 64,
        flow_drop_p: float = 0.5,
        # compatibility shims for legacy config fields
        in_channels=None,
        in_index=None,
        input_transform=None,
        channels=None,
        norm_cfg=None,
        loss_decode=None,
        concat_input=None,
        dilation=None,
        num_convs=None,
        fuse_dilation=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.use_edge_feat = use_edge_feat
        self.use_flow_feat = use_flow_feat
        self.flow_drop_p = flow_drop_p

        c0, c1, c2, c3 = feat_channels
        m = mid_channels

        # ── Stage 1: feat3 [H/32] → coarsest prediction ───────────────────────
        self.proj3 = _proj(c3, m)
        self.stage1_conv = _conv_block(m, m)
        self.aux_seg1 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 2: upsample + feat2 lateral [H/16] ─────────────────────────
        self.lateral2 = _proj(c2, m)
        self.stage2_conv = _conv_block(m, m)
        self.aux_seg2 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 3: upsample + feat1 lateral [H/8] ──────────────────────────
        self.lateral1 = _proj(c1, m)
        self.stage3_conv = _conv_block(m, m)
        self.aux_seg3 = nn.Conv2d(m, num_classes, 1)

        # ── Stage 4: upsample + concat feat0 [H/4] → final ───────────────────
        self.decode_conv1 = _conv_block(m + c0, m)
        self.decode_conv2 = _conv_block(m, m)

        # ── Optional: edge enhancement (Sobel on feat0, H/4) ──────────────────
        if use_edge_feat:
            self.sobel_x: torch.Tensor
            self.sobel_y: torch.Tensor
            self.register_buffer(
                'sobel_x',
                torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]).view(1, 1, 3, 3),
            )
            self.register_buffer(
                'sobel_y',
                torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]).view(1, 1, 3, 3),
            )
            self.edge_proj = _proj(1, m)

        # ── Optional: flow guidance (H/4, training only) ──────────────────────
        if use_flow_feat:
            self.flow_proj = _proj(flow_in_channels, m)

        # ── Classifier ────────────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(m, num_classes, 1)

        self.last_aux_logits: List[torch.Tensor] = []

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for seg in [self.aux_seg1, self.aux_seg2, self.aux_seg3, self.conv_seg]:
            nn.init.normal_(seg.weight, mean=0, std=0.01)
            nn.init.zeros_(seg.bias)

    def _compute_edge(self, feat0: torch.Tensor) -> torch.Tensor:
        norm = feat0.norm(dim=1, keepdim=True)
        gx = F.conv2d(norm, self.sobel_x, padding=1)
        gy = F.conv2d(norm, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-6).sqrt()

    def _upsample_add(self, x: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=lateral.shape[-2:], mode='bilinear',
                             align_corners=self.align_corners)
        return x_up + lateral

    def forward(self, inputs, flow_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: backbone features at four resolutions
              [0] feat0: [B, 256,  H/4,  W/4]
              [1] feat1: [B, 512,  H/8,  W/8]
              [2] feat2: [B, 1024, H/16, W/16]
              [3] feat3: [B, 2048, H/32, W/32]
            flow_feat: [B, flow_in_channels, H/4, W/4] or None

        Returns:
            [B, num_classes, H/4, W/4]

        Side-effect:
            self.last_aux_logits = [aux1(H/32), aux2(H/16), aux3(H/8)]
        """
        feat0, feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2], inputs[3]

        # ── Stage 1: H/32 ─────────────────────────────────────────────────────
        x = self.stage1_conv(self.proj3(feat3))        # [B, m, H/32, W/32]
        aux1 = self.aux_seg1(x)

        # ── Stage 2: H/16 ─────────────────────────────────────────────────────
        x = self.stage2_conv(
            self._upsample_add(x, self.lateral2(feat2))
        )                                               # [B, m, H/16, W/16]
        aux2 = self.aux_seg2(x)

        # ── Stage 3: H/8 ──────────────────────────────────────────────────────
        x = self.stage3_conv(
            self._upsample_add(x, self.lateral1(feat1))
        )                                               # [B, m, H/8, W/8]
        aux3 = self.aux_seg3(x)

        self.last_aux_logits = [aux1, aux2, aux3]

        # ── Stage 4: H/4 ──────────────────────────────────────────────────────
        x = F.interpolate(x, size=feat0.shape[-2:], mode='bilinear',
                          align_corners=self.align_corners)  # [B, m, H/4, W/4]

        if self.use_edge_feat:
            x = x + self.edge_proj(self._compute_edge(feat0))

        if self.use_flow_feat and flow_feat is not None:
            use_flow = (not self.training) or (torch.rand(1).item() >= self.flow_drop_p)
            if use_flow:
                x = x + self.flow_proj(flow_feat)

        x = self.decode_conv1(torch.cat([x, feat0], dim=1))  # [B, m, H/4, W/4]
        x = self.decode_conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)
