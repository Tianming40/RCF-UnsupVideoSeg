"""
RCFTripletJointMaskV2Model: extends RCFTripletJointMaskModel (v127, feat0-
only triplet joint feature) to all FOUR backbone scales -- "the 3-frame
version of v123" (v123 did the same feat0-only -> 4-scale extension for
the 2-frame case). Each frame's mask now gets its own 4 features PLUS 4
joint (3-way-concatenated) features, one per scale, instead of just feat0.

Reuses MultiScaleSegHeadJoint4 (models/multi_scale_seg_head_joint4.py) with
ZERO changes -- its forward() contract (4 optional joint-feature kwargs,
same shapes as the 2-frame case) doesn't care whether the tensor passed in
was built from 2 or 3 frames. Only the model-level computation of those 4
tensors differs (3-way concat instead of 2-way).

Subclasses RCFTripletModel (not RCFTripletJointMaskModel -- this is a
sibling extension, not a further subclass of it, to keep both variants
independently simple) and overrides ONLY _decode_head_forward, same
injection point every joint-mask variant this session uses.
"""
import torch.nn as nn

from models.rcf_triplet_model import RCFTripletModel


class RCFTripletJointMaskV2Model(RCFTripletModel):
    def __init__(self, *args,
                 joint_feat_channels: int = 64,
                 joint_feat_mid_channels: int = 128,
                 feat_channels=(256, 512, 1024, 2048),
                 **kwargs):
        super().__init__(*args, **kwargs)

        def _proj(c):
            return nn.Sequential(
                nn.Conv2d(c * 3, joint_feat_mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(joint_feat_mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(joint_feat_mid_channels, joint_feat_channels, kernel_size=3, padding=1),
            )

        c0, c1, c2, c3 = feat_channels
        self.joint_feat_proj = _proj(c0)   # feat0-level (same slot/name as v127, for flow_feat)
        self.feat1_joint_proj = _proj(c1)
        self.feat2_joint_proj = _proj(c2)
        self.feat3_joint_proj = _proj(c3)

    def _triplet_joint(self, feat, proj):
        """feat: [B*3, C, H, W] -> [B*3, joint_feat_channels, H, W], all
        three frames broadcast the SAME joint summary (symmetric, matching
        v127's own pattern)."""
        total = feat.shape[0]
        assert total % 3 == 0, f"requires im_num==3, got batch*im_num={total}"
        batch_size = total // 3
        triplet = feat.unflatten(0, (batch_size, 3)).flatten(1, 2)  # [B, 3C, H, W]
        joint = proj(triplet)                                        # [B, joint_feat_channels, H, W]
        return (joint.unsqueeze(1)
               .expand(-1, 3, -1, -1, -1)
               .reshape(total, *joint.shape[1:]))

    def _decode_head_forward(self, x, decode_head, flow_feat=None):
        if decode_head is self.decode_head2 and getattr(decode_head, 'use_flow_feat', False):
            feat0, feat1, feat2, feat3 = x[0], x[1], x[2], x[3]
            flow_feat = self._triplet_joint(feat0, self.joint_feat_proj)
            feat1_joint = self._triplet_joint(feat1, self.feat1_joint_proj)
            feat2_joint = self._triplet_joint(feat2, self.feat2_joint_proj)
            feat3_joint = self._triplet_joint(feat3, self.feat3_joint_proj)
            return decode_head.forward(x, flow_feat=flow_feat,
                                       feat1_joint=feat1_joint,
                                       feat2_joint=feat2_joint,
                                       feat3_joint=feat3_joint)
        return super()._decode_head_forward(x, decode_head, flow_feat=flow_feat)
