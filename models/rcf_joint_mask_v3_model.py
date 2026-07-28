"""
RCFJointMaskV3SoftTissueModel: same 4-scale joint-fusion mask branch as
RCFJointMaskV2SoftTissueModel (v123), but the joint feature at each scale
is now computed by DeformableCrossFrameAttention (models/
deformable_cross_frame_attention.py) instead of JointFrameFeatProjector's
naive channel-concat + conv. "Idea 3" from this session's discussion
(260720): let the network learn WHERE to look in the other frame (via
learned sampling offsets, Deformable-DETR-style), rather than comparing
same-position content only, and without depending on external GT flow at
all (train or eval) -- offsets are predicted purely from the query
feature itself.

Reuses MultiScaleSegHeadJoint4 (models/multi_scale_seg_head_joint4.py) with
ZERO changes -- that class's forward() contract (4 optional joint-feature
kwargs, same shapes as before) is unchanged; only WHAT computes those 4
tensors differs. v121/v122 (RCFJointMaskSoftTissueModel, base MultiScaleSegHead)
and v123 (RCFJointMaskV2SoftTissueModel, JointFrameFeatProjector) are
completely unaffected by this file.

Same single injection point as v1/v2: _decode_head_forward (see
rcf_joint_mask_model.py's docstring for why this is the one method common
to forward_train / forward_eval / main.py's _sliding_window_eval). Same
im_num==2 requirement, no graceful single-frame fallback.
"""
from models.rcf_soft_tissue_model import RCFSoftTissueModel
from models.deformable_cross_frame_attention import DeformableCrossFrameAttention


class RCFJointMaskV3SoftTissueModel(RCFSoftTissueModel):
    def __init__(self, *args,
                 joint_feat_channels: int = 64,
                 attn_heads: int = 8,
                 attn_num_points: int = 4,
                 feat_channels=(256, 512, 1024, 2048),
                 **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = feat_channels
        self.joint_attn0 = DeformableCrossFrameAttention(c0, joint_feat_channels, attn_heads, attn_num_points)
        self.joint_attn1 = DeformableCrossFrameAttention(c1, joint_feat_channels, attn_heads, attn_num_points)
        self.joint_attn2 = DeformableCrossFrameAttention(c2, joint_feat_channels, attn_heads, attn_num_points)
        self.joint_attn3 = DeformableCrossFrameAttention(c3, joint_feat_channels, attn_heads, attn_num_points)

    def _decode_head_forward(self, x, decode_head, flow_feat=None):
        if decode_head is self.decode_head2 and getattr(decode_head, 'use_flow_feat', False):
            feat0, feat1, feat2, feat3 = x[0], x[1], x[2], x[3]
            flow_feat = self.joint_attn0(feat0)
            feat1_joint = self.joint_attn1(feat1)
            feat2_joint = self.joint_attn2(feat2)
            feat3_joint = self.joint_attn3(feat3)
            return decode_head.forward(x, flow_feat=flow_feat,
                                       feat1_joint=feat1_joint,
                                       feat2_joint=feat2_joint,
                                       feat3_joint=feat3_joint)
        return super()._decode_head_forward(x, decode_head, flow_feat=flow_feat)
