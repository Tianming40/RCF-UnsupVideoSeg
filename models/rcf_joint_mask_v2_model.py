"""
RCFJointMaskV2SoftTissueModel: extends RCFJointMaskSoftTissueModel
(models/rcf_joint_mask_model.py, v121/v122's model_cls) from a single
feat0-level joint two-frame feature to all FOUR backbone scales
(feat0..feat3) -- "idea 1" from this session's discussion (260720): each
frame's own 4 features, PLUS 4 joint (concatenated cross-frame) features at
every scale, instead of only feat0.

Pairs with models/multi_scale_seg_head_joint4.py (MultiScaleSegHeadJoint4),
a new MultiScaleSegHead subclass with three additional additive-skip slots
(feat1_joint/feat2_joint/feat3_joint) alongside the existing feat0-level
flow_feat slot.

Zero edits to RCFModel/RCFDinoModel/RCFSoftTissueModel/MultiScaleSegHead/
RCFJointMaskSoftTissueModel/multi_scale_seg_head_joint4.py's parent class --
v121/v122 (which use the base MultiScaleSegHead + RCFJointMaskSoftTissueModel)
are completely unaffected. Only a config setting model_cls:
RCFJointMaskV2SoftTissueModel + decode_head2.type: MultiScaleSegHeadJoint4
exercises this code path.

Same single injection point as v1: _decode_head_forward, for the same
reason (the one method common to forward_train / forward_eval /
main.py's _sliding_window_eval -- see rcf_joint_mask_model.py's docstring
for why the alternative, forward_eval override, silently never engages
during real sliding-window training runs). Bypasses the base
RCFModel._decode_head_forward's single-flow_feat-argument contract
entirely when talking to decode_head2, calling decode_head.forward(...)
directly with all four joint tensors -- MultiScaleSegHeadJoint4's forward
accepts them as named kwargs, all defaulting to None.

Same requirement as v1: im_num==2 whenever decode_head2.use_flow_feat is
set (no graceful single-frame fallback) -- same paired train/val/test data
wiring as v121/v122 (CMC_grasp0_multigap_seq for training,
val_paired.txt/frame_num=2 for eval).
"""
from models.rcf_soft_tissue_model import RCFSoftTissueModel
from models.joint_frame_feat import JointFrameFeatProjector


class RCFJointMaskV2SoftTissueModel(RCFSoftTissueModel):
    def __init__(self, *args,
                 joint_feat_channels: int = 64,
                 joint_feat_mid_channels: int = 128,
                 feat_channels=(256, 512, 1024, 2048),
                 **kwargs):
        super().__init__(*args, **kwargs)
        c0, c1, c2, c3 = feat_channels
        self.joint_feat_proj = JointFrameFeatProjector(c0, joint_feat_channels, joint_feat_mid_channels)
        self.feat1_joint_proj = JointFrameFeatProjector(c1, joint_feat_channels, joint_feat_mid_channels)
        self.feat2_joint_proj = JointFrameFeatProjector(c2, joint_feat_channels, joint_feat_mid_channels)
        self.feat3_joint_proj = JointFrameFeatProjector(c3, joint_feat_channels, joint_feat_mid_channels)

    def _joint_feat(self, feat, proj):
        """feat: [B*2, C, H, W] (both frames' own-scale feature, batched) ->
        [B*2, joint_feat_channels, H, W] (same joint two-frame context
        broadcast to both frames, matching v1's _compute_joint_flow_feat)."""
        total = feat.shape[0]
        assert total % 2 == 0, (
            f"RCFJointMaskV2SoftTissueModel requires im_num==2 (paired frames) "
            f"whenever use_flow_feat is set, got batch*im_num={total} (odd)"
        )
        batch_size = total // 2
        feat_pair = feat.unflatten(0, (batch_size, 2)).flatten(1, 2)  # [B, 2*C, H, W]
        joint = proj(feat_pair)                                        # [B, joint_feat_channels, H, W]
        return (joint.unsqueeze(1)
               .expand(-1, 2, -1, -1, -1)
               .reshape(total, *joint.shape[1:]))

    def _decode_head_forward(self, x, decode_head, flow_feat=None):
        if decode_head is self.decode_head2 and getattr(decode_head, 'use_flow_feat', False):
            feat0, feat1, feat2, feat3 = x[0], x[1], x[2], x[3]
            flow_feat = self._joint_feat(feat0, self.joint_feat_proj)
            feat1_joint = self._joint_feat(feat1, self.feat1_joint_proj)
            feat2_joint = self._joint_feat(feat2, self.feat2_joint_proj)
            feat3_joint = self._joint_feat(feat3, self.feat3_joint_proj)
            return decode_head.forward(x, flow_feat=flow_feat,
                                       feat1_joint=feat1_joint,
                                       feat2_joint=feat2_joint,
                                       feat3_joint=feat3_joint)
        return super()._decode_head_forward(x, decode_head, flow_feat=flow_feat)
