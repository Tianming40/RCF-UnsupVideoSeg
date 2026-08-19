"""
MultiScaleSegHeadJoint4: MultiScaleSegHead extended with THREE additional
additive-skip injection points (feat1_joint, feat2_joint, feat3_joint),
mirroring the existing flow_feat slot (which only ever carries a
feat0-level joint feature, see models/rcf_joint_mask_model.py /
models/joint_frame_feat.py) but for the three deeper/coarser scales too.

Motivation: RCFJointMaskSoftTissueModel (v121/v122) only gives the mask
branch a joint two-frame feature at feat0 (the finest, most detail-rich
scale). This class extends that to all four scales -- the mask branch now
gets joint (cross-frame) information at every level of abstraction feat0
gives fine detail, feat3 gives coarse semantic content), not just the
finest one.

Pure subclass of MultiScaleSegHead -- reuses all its existing submodules
(proj1/proj2/proj3/fuse_conv/decode_conv1/decode_conv2/conv_seg) via
super().__init__(), only ADDS three new small 1x1-conv projectors
(feat1_joint_proj/feat2_joint_proj/feat3_joint_proj, mirroring flow_proj's
existing pattern: 64-channel input -> mid_channels, added into `fused`).
forward() replicates the base class's exact control flow with three new
optional kwargs (all default None -> bit-identical to base MultiScaleSegHead
when unset, so this class is a strict superset, safe to use even without
the new joints wired up).

Used by models/rcf_joint_mask_v2_model.py (RCFJointMaskV2SoftTissueModel).
Zero edits to multi_scale_seg_head.py or any v121/v122 file -- those keep
using the base MultiScaleSegHead unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_scale_seg_head import MultiScaleSegHead


class MultiScaleSegHeadJoint4(MultiScaleSegHead):
    def __init__(self, *args, feat1_joint_in_channels: int = 64,
                 feat2_joint_in_channels: int = 64,
                 feat3_joint_in_channels: int = 64,
                 **kwargs):
        super().__init__(*args, **kwargs)
        mid_channels = self.decode_conv2[0].out_channels  # read back from an existing submodule, avoids re-threading the ctor arg

        def _proj(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

        self.feat1_joint_proj = _proj(feat1_joint_in_channels)
        self.feat2_joint_proj = _proj(feat2_joint_in_channels)
        self.feat3_joint_proj = _proj(feat3_joint_in_channels)

    def forward_features(self, inputs, flow_feat=None, feat1_joint=None, feat2_joint=None, feat3_joint=None):
        """
        Identical to forward() up to (NOT including) the final conv_seg
        classifier -- returns the mid_channels-dim fused feature map conv_seg
        would otherwise consume, [B, mid_channels, H/4, W/4]. Added 260728
        for k-means-based conv_seg initialization (see
        rcf_selftaught_flow_model.py:kmeans_init_mask_head) -- clustering
        needs the ACTUAL feature space conv_seg operates on, not a proxy
        space (e.g. DINO's), so this must be extracted from this exact
        forward path rather than duplicated/approximated elsewhere.
        """
        feat0, feat1, feat2, feat3 = inputs[0], inputs[1], inputs[2], inputs[3]

        # Step 1: independent projections + element-wise sum (at H/8)
        fused = self.proj1(feat1) + self.proj2(feat2) + self.proj3(feat3)

        # Step 1b (NEW): inject feat1/feat2/feat3-level joint (cross-frame) features,
        # BEFORE fuse_conv so the dilated/ASPP conv can spatially mix them together
        # with the own-frame signal, not just tack them on afterward.
        if feat1_joint is not None:
            fused = fused + self.feat1_joint_proj(feat1_joint)
        if feat2_joint is not None:
            fused = fused + self.feat2_joint_proj(feat2_joint)
        if feat3_joint is not None:
            fused = fused + self.feat3_joint_proj(feat3_joint)

        # Step 2: 3x3 dilated conv (or ASPP) for cross-scale spatial mixing
        fused = self.fuse_conv(fused)

        # Step 3: upsample to feat0's exact size
        fused = F.interpolate(
            fused, size=feat0.shape[-2:], mode='bilinear',
            align_corners=self.align_corners,
        )

        # Step 4: optional edge enhancement (unchanged from base)
        if self.use_edge_feat:
            edge_feat = self.edge_proj(self._compute_edge(feat0))
            fused = fused + edge_feat

        # Step 5: feat0-level joint feature (same slot as the base class's flow_feat).
        # Deliberately NO flow_drop_p random-drop here (unlike the base class) -- see
        # models/rcf_joint_mask_model.py's flow_drop_p comment (v121 vs v122): that
        # regularizer was designed for GT-flow guidance with a real train/eval
        # availability gap, which doesn't apply to a joint two-frame feature that's
        # available at every stage (train/val/test) by construction.
        if self.use_flow_feat and flow_feat is not None:
            fused = fused + self.flow_proj(flow_feat)

        # Step 6: concat feat0 (fine-grained boundary detail, own-frame only)
        x = torch.cat([fused, feat0], dim=1)

        # Step 7: two 3x3 convs for local refinement
        x = self.decode_conv1(x)
        x = self.decode_conv2(x)

        # Step 8: dropout
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(self, inputs, flow_feat=None, feat1_joint=None, feat2_joint=None, feat3_joint=None):
        """
        Args:
            inputs: same as MultiScaleSegHead (feat0..feat3, own-frame features)
            flow_feat: [B, flow_in_channels, H/4, W/4] or None -- feat0-level
                joint feature (same slot/semantics as the base class)
            feat1_joint, feat2_joint, feat3_joint: [B, C, H/8, W/8] or None --
                joint two-frame features at the feat1/feat2/feat3 scale
                (H/8 resolution, matching proj1/proj2/proj3's input scale).
                All three default None -> this class behaves exactly like
                MultiScaleSegHead.

        Returns: [B, num_classes, H/4, W/4]
        """
        return self.conv_seg(self.forward_features(
            inputs, flow_feat=flow_feat, feat1_joint=feat1_joint,
            feat2_joint=feat2_joint, feat3_joint=feat3_joint))
