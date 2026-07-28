"""
JointFrameFeatProjector: projects the concatenated backbone features of a
frame PAIR (both frames' feat0, channel-concatenated) down to a small
feature map that can be fed into MultiScaleSegHead's existing flow_feat
auxiliary-input slot (use_flow_feat=True, additive skip inside
MultiScaleSegHead.forward: `fused = fused + self.flow_proj(flow_feat)`).

That slot has existed since MultiScaleSegHead was written but every config
to date has only ever fed it a feature derived from GT optical flow
(RCFModel._get_flow_feat_for_seg) -- never raw cross-frame appearance. This
module lets the mask branch see the OTHER frame's actual image content for
the first time (discussed at length 260717: the mask branch's forward pass
is otherwise completely per-frame -- frame_i and frame_j's backbone
features never interact, since they're batched not concatenated -- so
appearance ambiguity that only the paired frame could resolve, e.g.
specular highlights or low-texture drift, currently has no forward-pass
channel to be resolved through; the residual branch is pair-conditioned
but only reaches mask indirectly, multiplied in as one additive term after
mask is already fixed).

See models/rcf_joint_mask_model.py for how this gets wired in via a
forward hook (no changes to any existing file).
"""
import torch
import torch.nn as nn


class JointFrameFeatProjector(nn.Module):
    """
    Input:  feat0 for BOTH frames of a pair, channel-concatenated:
            [B, 2*in_channels_per_frame, H, W]  (feat0 is [B*I, 256, 96, 96]
            from the ResNet backbone's first output scale for a 384x384
            crop -- concatenating frame0 and frame1 gives [B, 512, 96, 96])
    Output: [B, out_channels, H, W]  (same spatial size as feat0 -- matches
            MultiScaleSegHead's flow_feat contract exactly, no resize
            needed at the call site)
    """
    def __init__(self, in_channels_per_frame: int = 256, out_channels: int = 64, mid_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels_per_frame * 2, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, feat0_pair: torch.Tensor) -> torch.Tensor:
        return self.net(feat0_pair)
