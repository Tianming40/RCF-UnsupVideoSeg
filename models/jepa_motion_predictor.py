"""
JepaMotionPredictor: a BLIND predictor -- given ONLY the context (frame0)
features, predicts what the target encoder would produce for the other
frame, WITHOUT ever looking at the other frame's actual content. Discussed
260729, as part of a more faithful JEPA-style redesign (models/
rcf_jepa_predictor_model.py) than models/rcf_jepa_flow_model.py's earlier,
partial adoption (that version still did an explicit correlation SEARCH
against real target features -- i.e. it could "peek" -- and still produced
an interpretable (dx,dy) flow field for the existing rigid/affine
decode_head to consume).

This module is deliberately NOT a correlation/attention mechanism over the
target -- it never receives target-side tensors as input at all. This
blindness is the essential I-JEPA/V-JEPA property: since the predictor
cannot look up the answer, the only way to reduce its loss (compared
against the REAL target encoder's output, computed independently and never
backpropagated into) is to learn genuine, generalizable priors about how
scene content plausibly evolves across the frame gap (tissue deformation
patterns, typical camera motion, slow parallax, etc.). Content that does NOT
follow such learnable priors -- most saliently, the instrument's own motion,
which is not predictable from passively watching the background alone --
will have persistently higher residual prediction error even after the
predictor is well trained. That residual error map is the actual signal
this architecture hands to segmentation (see rcf_jepa_predictor_model.py's
_predict_error_consistency_loss), replacing the explicit-flow-field +
rigid/affine motion-clustering approach entirely for this model variant.

Architecture: a small stack of dilated 3x3 conv blocks (increasing dilation
for a wider receptive field without pooling -- the predictor may need
context well beyond a pixel's own neighbourhood to "guess" plausible
motion/deformation) followed by a 1x1 projection back to the target's own
channel dimension. Deliberately simple/small: this is a first version,
meant to be easy to swap out (a transformer-based predictor, closer to
I-JEPA/V-JEPA's own architecture, is a natural upgrade path if this proves
too weak -- kept in its own file for exactly that reason).
"""
import torch.nn as nn


class JepaMotionPredictor(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 256,
                 num_blocks: int = 4, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers = []
        in_ch = channels
        for i in range(num_blocks):
            d = dilations[i % len(dilations)]
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ))
            in_ch = hidden_channels
        self.blocks = nn.Sequential(*layers)
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, context_feat):
        """context_feat: [B, channels, H, W] (context/own-frame encoder
        output ONLY -- caller must never pass target-side tensors here).
        Returns predicted target features, same shape."""
        return self.out_proj(self.blocks(context_feat))
