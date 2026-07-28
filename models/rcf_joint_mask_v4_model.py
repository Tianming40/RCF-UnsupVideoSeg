"""
RCFJointMaskV4SoftTissueModel: identical to RCFJointMaskV3SoftTissueModel
(v124) except DeformableCrossFrameAttention -> DeformableCrossFrameAttentionPE
(models/deformable_cross_frame_attention_pe.py) -- adds a learned 2D
position embedding to the query side of offset/weight prediction at each
scale, discussed 260720. See that file's docstring for the design
rationale (learned, not sinusoidal; query-side only, not value-side).

Reuses MultiScaleSegHeadJoint4 with ZERO changes (same as v123/v124) --
only WHAT computes the 4 joint-feature tensors differs.
"""
from models.rcf_soft_tissue_model import RCFSoftTissueModel
from models.deformable_cross_frame_attention_pe import DeformableCrossFrameAttentionPE


class RCFJointMaskV4SoftTissueModel(RCFSoftTissueModel):
    def __init__(self, *args,
                 joint_feat_channels: int = 64,
                 attn_heads: int = 8,
                 attn_num_points: int = 4,
                 pos_max_len: int = 128,
                 feat_channels=(256, 512, 1024, 2048),
                 **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = feat_channels
        self.joint_attn0 = DeformableCrossFrameAttentionPE(c0, joint_feat_channels, attn_heads, attn_num_points, pos_max_len)
        self.joint_attn1 = DeformableCrossFrameAttentionPE(c1, joint_feat_channels, attn_heads, attn_num_points, pos_max_len)
        self.joint_attn2 = DeformableCrossFrameAttentionPE(c2, joint_feat_channels, attn_heads, attn_num_points, pos_max_len)
        self.joint_attn3 = DeformableCrossFrameAttentionPE(c3, joint_feat_channels, attn_heads, attn_num_points, pos_max_len)

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
