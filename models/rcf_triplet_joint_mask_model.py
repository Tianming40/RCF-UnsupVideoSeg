"""
RCFTripletJointMaskModel: combines v121's joint-mask mechanism (mask
branch gets a forward-pass-visible feature built from OTHER frames'
appearance, via MultiScaleSegHead's existing use_flow_feat additive-skip
slot) with RCFTripletModel's (v126) 3-frame / 3-pairwise-flow-loss
supervision -- "build the joint version of 126" per this session's
discussion (260720).

Subclasses RCFTripletModel and overrides ONLY _decode_head_forward (same
injection point v121/v123/v124 all use, and for the same reason -- it's
the one method forward_train already calls generically for the mask
branch, with flow_feat=None by default). Direct 3-frame generalization of
v121's JointFrameFeatProjector: instead of concatenating 2 frames' feat0
(512 channels), concatenate all 3 (768 channels) into ONE joint feature,
broadcast identically to all three frames' mask computation (mask_i,
mask_j, mask_k all see the SAME joint summary of all 3 frames -- symmetric,
matching v121's own symmetric-broadcast pattern for the 2-frame case, not
per-direction).

decode_head2.type stays MultiScaleSegHead (the v121 base class, single
feat0-level slot) -- NOT MultiScaleSegHeadJoint4 (v123's 4-scale variant) --
this combines with the triplet-loss idea at the SAME scope v121 explored
first, not the more elaborate 4-scale/attention versions (v123/v124),
consistent with testing one new variable at a time.

decode_head2.flow_drop_p MUST be set to 0.0 in config (learned from v121's
bug, see rcf_joint_mask_model.py's flow_drop_p history) -- the default 0.5
would silently drop this triplet joint feature half of training steps too.
"""
import torch.nn as nn

from models.rcf_triplet_model import RCFTripletModel


class RCFTripletJointMaskModel(RCFTripletModel):
    def __init__(self, *args, joint_feat_channels: int = 64,
                 joint_feat_mid_channels: int = 128,
                 feat0_channels: int = 256,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_feat_proj = nn.Sequential(
            nn.Conv2d(feat0_channels * 3, joint_feat_mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(joint_feat_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(joint_feat_mid_channels, joint_feat_channels, kernel_size=3, padding=1),
        )

    def _decode_head_forward(self, x, decode_head, flow_feat=None):
        if decode_head is self.decode_head2 and getattr(decode_head, 'use_flow_feat', False):
            feat0 = x[0]  # [B*3, C, H, W]
            total = feat0.shape[0]
            assert total % 3 == 0, (
                f"RCFTripletJointMaskModel requires im_num==3, got batch*im_num={total}"
            )
            batch_size = total // 3
            feat0_triplet = feat0.unflatten(0, (batch_size, 3)).flatten(1, 2)  # [B, 3C, H, W]
            joint = self.joint_feat_proj(feat0_triplet)                        # [B, joint_feat_channels, H, W]
            # All three frames see the SAME joint summary of all three frames
            # (symmetric broadcast, matching v121's own pattern for the
            # 2-frame case -- not a per-direction/per-pair feature).
            flow_feat = (joint.unsqueeze(1)
                        .expand(-1, 3, -1, -1, -1)
                        .reshape(total, *joint.shape[1:]))
        return super()._decode_head_forward(x, decode_head, flow_feat=flow_feat)
