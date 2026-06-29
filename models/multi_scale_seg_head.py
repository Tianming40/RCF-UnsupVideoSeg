"""
MultiScaleSegHead
=================
Segmentation head replacing the original FCNHead (decode_head2).

Original FCNHead design:
  feat0 [256, H/4] downsampled to H/8, concatenated with feat3 [2048, H/8] → output at H/8
  → resized to mask_size (typically 128×128)

This head design:
  1. feat1 / feat2 / feat3 (all at H/8) each projected to mid_channels via 1×1 conv
  2. Element-wise sum → [mid_ch, H/8]
  3. One 3×3 dilated conv (dilation=6) for spatial information exchange across scales
  4. 2× bilinear upsample → [mid_ch, H/4]
  5. Concat feat0 [feat0_ch, H/4] → [mid_ch+feat0_ch, H/4]
  6. Two 3×3 convs (no dilation, local refinement) → [mid_ch, H/4]
  7. conv_seg 1×1 → [num_classes, H/4]

Output resolution is H/4 (96×96 for 384×384 input), twice the original 48×48.
Combined with mask_size=[96, 96], warp_seg loss operates at the native 96×96 resolution.

Optional add-ons (disabled by default, backward-compatible):
  use_edge_feat: Sobel gradient of feat0 L2-norm → projected to mid_channels, added to fused
  use_flow_feat: optical flow features → projected to mid_channels, added to fused (training only,
                 randomly dropped with probability flow_drop_p to close the train/val gap)

Config example (decode_head2 field):
  decode_head2:
    type: MultiScaleSegHead
    num_classes: 5
    mid_channels: 256
    feat_channels: [256, 512, 1024, 2048]
    fuse_dilation: 6
    dropout_ratio: 0.1
    align_corners: false
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSegHead(nn.Module):
    """
    Args:
        num_classes      (int)  : number of segmentation channels, default 5
        mid_channels     (int)  : internal feature dimension, default 256
        feat_channels    (list) : [feat0_ch, feat1_ch, feat2_ch, feat3_ch], default [256,512,1024,2048]
        fuse_dilation    (int)  : dilation of fuse_conv (applied at H/8 resolution)
        dropout_ratio    (float): Dropout2d ratio; 0 disables dropout
        align_corners    (bool) : align_corners for F.interpolate
        use_edge_feat    (bool) : enable Sobel edge enhancement on feat0
        use_flow_feat    (bool) : enable optical flow guidance (training only)
        flow_in_channels (int)  : channels of the incoming flow feature (matches num_flow_feat_channels)
        flow_drop_p      (float): probability of dropping flow guidance during training
    """

    def __init__(
        self,
        num_classes: int = 5,
        mid_channels: int = 256,
        feat_channels=(256, 512, 1024, 2048),
        fuse_dilation: int = 6,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        use_edge_feat: bool = False,
        # ── optical flow guidance ─────────────────────────────────────────────
        use_flow_feat: bool = False,
        flow_in_channels: int = 64,   # output channels of flow_feat_before_agg
        flow_drop_p: float = 0.5,     # probability of dropping flow guidance during training
        # compatibility shims for fields that may appear in legacy configs
        in_channels=None,
        in_index=None,
        input_transform=None,
        channels=None,
        norm_cfg=None,
        loss_decode=None,
        concat_input=None,
        dilation=None,
        num_convs=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.use_edge_feat = use_edge_feat
        self.use_flow_feat = use_flow_feat
        self.flow_drop_p = flow_drop_p

        c0, c1, c2, c3 = feat_channels  # 256, 512, 1024, 2048

        # ── 1. independent projections: feat1 / feat2 / feat3 → mid_channels ─
        self.proj1 = nn.Sequential(
            nn.Conv2d(c1, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(c2, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(c3, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # ── 2. 3×3 dilated conv at H/8 to mix information across scales ───────
        pad = fuse_dilation
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3,
                      padding=pad, dilation=fuse_dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # ── 3. optional: edge enhancement (Sobel on feat0 → mid_channels additive skip)
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
            self.edge_proj = nn.Sequential(
                nn.Conv2d(1, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

        # ── 4. optional: flow guidance (flow_feat → mid_channels additive skip) ─
        if use_flow_feat:
            self.flow_proj = nn.Sequential(
                nn.Conv2d(flow_in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

        # ── 5. upsample to H/4, concat feat0, refine ─────────────────────────
        self.decode_conv1 = nn.Sequential(
            nn.Conv2d(mid_channels + c0, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.decode_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # ── 6. classifier ────────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(mid_channels, num_classes, 1)

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
        # small init for conv_seg to avoid extreme softmax at the start
        nn.init.normal_(self.conv_seg.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv_seg.bias)

    def _compute_edge(self, feat0):
        """Sobel on L2 norm of feat0 → [B, 1, H/4, W/4]"""
        norm = feat0.norm(dim=1, keepdim=True)          # [B, 1, H/4, W/4]
        gx = F.conv2d(norm, self.sobel_x, padding=1)
        gy = F.conv2d(norm, self.sobel_y, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-6).sqrt()       # [B, 1, H/4, W/4]

    def forward(self, inputs, flow_feat=None):
        """
        Args:
            inputs: list of backbone2 feature maps
              inputs[0] = feat0: [B, 256,  H/4, W/4]
              inputs[1] = feat1: [B, 512,  H/8, W/8]
              inputs[2] = feat2: [B, 1024, H/8, W/8]
              inputs[3] = feat3: [B, 2048, H/8, W/8]
            flow_feat: [B, flow_in_channels, H/4, W/4] or None
              Passed during training from gt_flow via rcf_model (.detach());
              always None at val/test time — the branch is skipped entirely.

        Returns: [B, num_classes, H/4, W/4]
        """
        feat0, feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2], inputs[3]

        # Step 1: independent projections + element-wise sum (at H/8)
        fused = self.proj1(feat1) + self.proj2(feat2) + self.proj3(feat3)
        # [B, mid_ch, H/8, W/8]

        # Step 2: 3×3 dilated conv for cross-scale spatial mixing
        fused = self.fuse_conv(fused)
        # [B, mid_ch, H/8, W/8]

        # Step 3: upsample to feat0's exact size (avoids ±1 mismatch with odd spatial dims)
        fused = F.interpolate(
            fused, size=feat0.shape[-2:], mode='bilinear',
            align_corners=self.align_corners,
        )
        # [B, mid_ch, H/4, W/4]

        # Step 4: optional edge enhancement — inject Sobel response of feat0 into fused
        if self.use_edge_feat:
            edge_feat = self.edge_proj(self._compute_edge(feat0))  # [B, mid_ch, H/4, W/4]
            fused = fused + edge_feat

        # Step 5: optional flow guidance — randomly dropped during training; skipped at val/test
        if self.use_flow_feat and flow_feat is not None:
            use_flow = (not self.training) or (torch.rand(1).item() >= self.flow_drop_p)
            if use_flow:
                fused = fused + self.flow_proj(flow_feat)  # [B, mid_ch, H/4, W/4]

        # Step 6: concat feat0 (fine-grained boundary detail)
        x = torch.cat([fused, feat0], dim=1)
        # [B, mid_ch + feat0_ch, H/4, W/4]

        # Step 7: two 3×3 convs for local refinement
        x = self.decode_conv1(x)
        x = self.decode_conv2(x)
        # [B, mid_ch, H/4, W/4]

        # Step 8: dropout + classifier
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)
        # [B, num_classes, H/4, W/4]
