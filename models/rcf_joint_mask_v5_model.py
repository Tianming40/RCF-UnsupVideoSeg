"""
RCFJointMaskV5SoftTissueModel: same 4-scale joint-fusion mask branch as
RCFJointMaskV2SoftTissueModel (v123) / RCFJointMaskV3SoftTissueModel (v124),
but the joint feature at each scale is now computed by
LocalCorrelationCrossFrameAttention (models/local_correlation_cross_frame_attention.py)
instead of either naive channel-concat (v123) or blind deformable-offset
regression (v124/v130). Discussed 260724: replaces "guess an offset from the
query's own appearance" with "explicitly check content similarity against a
local window of candidates in the other frame" -- the same correspondence-
finding principle classical/deep optical flow methods use (block matching,
FlowNet's Correlation Layer, PWC-Net's cost volume), consumed here as
attention weights over a local window rather than an explicit flow vector.

Reuses MultiScaleSegHeadJoint4 (models/multi_scale_seg_head_joint4.py) with
ZERO changes -- only WHAT computes the 4 joint-feature tensors differs from
v123/v124/v130.

Same single injection point as v1-v4: _decode_head_forward. Same im_num==2
requirement, no graceful single-frame fallback.
"""
from models.rcf_soft_tissue_model import RCFSoftTissueModel
from models.local_correlation_cross_frame_attention import LocalCorrelationCrossFrameAttention


class RCFJointMaskV5SoftTissueModel(RCFSoftTissueModel):
    def __init__(self, *args,
                 joint_feat_channels: int = 64,
                 corr_proj_channels: int = 32,
                 corr_radius: int = 4,
                 feat_channels=(256, 512, 1024, 2048),
                 **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = feat_channels
        self.joint_attn0 = LocalCorrelationCrossFrameAttention(c0, joint_feat_channels, corr_proj_channels, corr_radius)
        self.joint_attn1 = LocalCorrelationCrossFrameAttention(c1, joint_feat_channels, corr_proj_channels, corr_radius)
        self.joint_attn2 = LocalCorrelationCrossFrameAttention(c2, joint_feat_channels, corr_proj_channels, corr_radius)
        self.joint_attn3 = LocalCorrelationCrossFrameAttention(c3, joint_feat_channels, corr_proj_channels, corr_radius)

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
