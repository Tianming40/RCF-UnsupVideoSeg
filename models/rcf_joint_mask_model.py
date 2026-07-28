"""
RCFJointMaskSoftTissueModel: gives the mask branch (decode_head2) genuine
forward-pass access to the PAIRED frame's appearance features, via a new
JointFrameFeatProjector (models/joint_frame_feat.py) feeding
MultiScaleSegHead's existing (previously unused by any config)
use_flow_feat additive-skip slot (`fused = fused + self.flow_proj(flow_feat)`).

Motivation (discussed at length 260717, see README): mask prediction is
currently completely per-frame in its forward computation -- frame_i and
frame_j's backbone features never interact (batched, not concatenated), so
mask_i has zero forward-pass access to frame_j's content. The only place
two-frame information has ever entered this architecture is the residual
branch (decode_head3, pair-conditioned) and, indirectly, mask's TRAINING
GRADIENT (via the rigid/affine flow-fit loss in FlowAggregationHead) --
never mask's own forward computation. This matters because some appearance
ambiguity (specular highlights, low-texture drift -- exactly the RAFT
failure modes visually confirmed in this session's flow-quality analysis)
can only be resolved by looking at the OTHER frame, and single-frame
appearance alone has no way to do that no matter how well trained.

This is a NEW model class (subclassing RCFSoftTissueModel) rather than a
modification to RCFModel/RCFSoftTissueModel/MultiScaleSegHead directly, so
every existing config (v102, v116-v120, etc.) is completely unaffected --
only a config that sets model_cls: RCFJointMaskSoftTissueModel exercises
this code path.

Single injection point: _decode_head_forward. This turned out to matter a
lot -- it's the ONE method common to all three real call paths:
  - forward_train (RCFModel): self._decode_head_forward(all_feat, self.decode_head2, flow_feat=_flow_feat)
  - forward_eval (RCFModel): self._decode_head_forward(all_feat, self.decode_head2)  [flow_feat always None]
  - _sliding_window_eval (main.py Model, used by every real config since
    they all set use_sliding_window: true, which makes test_step bypass
    forward_eval ENTIRELY): self.model._decode_head_forward(feat, head)
An earlier version of this file instead overrode _get_flow_feat_for_seg
(forward_train's own extension point) plus a duplicated forward_eval
override -- both real but INCOMPLETE, since sliding-window eval (what
every actual training run uses) calls neither. Overriding
_decode_head_forward instead covers all three uniformly, and receives
`x` (== all_feat, containing feat0 with the real cross-frame content)
directly as an argument, so no forward-hook capture trick is needed either.

Requirement: whenever decode_head2.use_flow_feat is True, x[0] (feat0)
must have a batch dim of size 2*B (i.e. im_num==2, paired frames) --
enforced by assertion. This means any eval source used with this model
needs a genuinely paired split (frame_num=2), e.g.
tools/build_paired_eval_split.py's val_paired.txt for
CMC_grasp0_continuous_bwdif/eval_instrument,eval_tissue, wired via
val_dataset_list's new per-entry `frame_num: 2` (main_tissue.py's
_make_val_loader/val_dataloader, extended this session, default 1 =
backward compatible for every other config). No graceful single-frame
fallback is attempted -- simpler and matches the fact that this project's
eval sources mostly already have a real paired frame available.

eval_on_ema is naturally unaffected: _sliding_window_eval passes
decode_head2_ema (a different module identity) when eval_on_ema is set, so
the `decode_head is self.decode_head2` guard below correctly no-ops for
that path without any special-casing.
"""
from models.rcf_soft_tissue_model import RCFSoftTissueModel
from models.joint_frame_feat import JointFrameFeatProjector


class RCFJointMaskSoftTissueModel(RCFSoftTissueModel):
    def __init__(self, *args, joint_feat_channels: int = 64,
                 joint_feat_mid_channels: int = 128,
                 joint_feat_in_channels_per_frame: int = 256,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_feat_proj = JointFrameFeatProjector(
            in_channels_per_frame=joint_feat_in_channels_per_frame,
            out_channels=joint_feat_channels,
            mid_channels=joint_feat_mid_channels,
        )

    def _decode_head_forward(self, x, decode_head, flow_feat=None):
        if decode_head is self.decode_head2 and getattr(decode_head, 'use_flow_feat', False):
            feat0 = x[0]  # [B*im_num, C, H, W]
            total = feat0.shape[0]
            assert total % 2 == 0, (
                f"RCFJointMaskSoftTissueModel requires im_num==2 (paired frames) "
                f"whenever use_flow_feat is set, got batch*im_num={total} (odd)"
            )
            batch_size = total // 2
            feat0_pair = feat0.unflatten(0, (batch_size, 2)).flatten(1, 2)  # [B, 2*C, H, W]
            joint_feat = self.joint_feat_proj(feat0_pair)                   # [B, joint_feat_channels, H, W]
            # Both frames in a pair see the SAME joint two-frame context.
            flow_feat = (joint_feat.unsqueeze(1)
                        .expand(-1, 2, -1, -1, -1)
                        .reshape(total, *joint_feat.shape[1:]))
        return super()._decode_head_forward(x, decode_head, flow_feat=flow_feat)
