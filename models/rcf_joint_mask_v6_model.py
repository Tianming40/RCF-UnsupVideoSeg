"""
RCFJointMaskV6SoftTissueModel: identical to RCFJointMaskV5SoftTissueModel
(v138, local-window content correlation) except
LocalCorrelationCrossFrameAttention -> LocalCorrelationCrossFrameAttentionPE
(models/local_correlation_cross_frame_attention_pe.py) -- adds a learned 2D
position embedding to the query side of the correlation, discussed 260724.
Same relationship as v130 is to v124 (DeformableCrossFrameAttention + PE),
just applied to the newer correlation-based mechanism instead.

Reuses MultiScaleSegHeadJoint4 with ZERO changes (same as v123/v124/v130/v138).
"""
from models.rcf_soft_tissue_model import RCFSoftTissueModel
from models.local_correlation_cross_frame_attention_pe import LocalCorrelationCrossFrameAttentionPE


class RCFJointMaskV6SoftTissueModel(RCFSoftTissueModel):
    def __init__(self, *args,
                 joint_feat_channels: int = 64,
                 corr_proj_channels: int = 32,
                 corr_radius: int = 4,
                 pos_max_len: int = 128,
                 feat_channels=(256, 512, 1024, 2048),
                 **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = feat_channels
        self.joint_attn0 = LocalCorrelationCrossFrameAttentionPE(c0, joint_feat_channels, corr_proj_channels, corr_radius, pos_max_len)
        self.joint_attn1 = LocalCorrelationCrossFrameAttentionPE(c1, joint_feat_channels, corr_proj_channels, corr_radius, pos_max_len)
        self.joint_attn2 = LocalCorrelationCrossFrameAttentionPE(c2, joint_feat_channels, corr_proj_channels, corr_radius, pos_max_len)
        self.joint_attn3 = LocalCorrelationCrossFrameAttentionPE(c3, joint_feat_channels, corr_proj_channels, corr_radius, pos_max_len)

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
