"""
MultiScaleSegHeadDecoderPrior: MultiScaleSegHead + DINO graph-partitioning
eigenvectors injected at the DECODER stage, discussed 260801 -- a third
distinct injection POINT (not just fusion type) alongside v146/v149 (fuse
into backbone2's feat0/feat3, i.e. ENCODER-side, before decode_head2 even
runs) and v153 (attention-gate, also encoder-side at feat3).

Motivation ("Eigen其实更像mask prior...很多医学论文都会在decoder阶段加入
shape prior", discussed 260801): v146/v149/v153 all inject the eigenvector
signal into a backbone2 feature map, which then flows through decode_head2's
OWN multi-scale fusion (Step 1-2: proj1(feat1)+proj2(feat2)+proj3(feat3),
then a 3x3 dilated/ASPP conv) before ever reaching a decision. Injecting
instead AFTER that fusion (right after Step 3's upsample to feat0's
resolution, via the new _inject_decoder_prior hook added to
MultiScaleSegHead.forward_features, see that file's 260801 change) treats
the eigenvectors more literally as a "mask/shape prior" added directly onto
the network's own near-final multi-scale decision, the way many medical
segmentation papers add a shape prior at the decoder rather than the
encoder. Different tradeoff from the encoder-side lines: less opportunity
for the signal to be "digested"/deeply integrated into the semantic
reasoning (Step 1-2 never sees it), but also less opportunity for it to be
diluted/washed out by that same processing.

Mechanism: unlike v146/v149/v153 (which each bundle extraction + trainable
fusion into one object living outside decode_head2), this class keeps
extraction external (models/dino_graph_eigvec_extractor.py's
DinoGraphEigvecExtractor, built in RCFDinoModel, no trainable params) and
owns its OWN small trainable fusion module internally: eig_proj (num_eigvecs
-> hidden channels) + fuse_conv (mid_channels + hidden -> mid_channels,
LAST layer zero-initialized) -- operating at mid_channels (256 by default,
much smaller than feat3's 2048) and H/4 resolution (post-Step-3 upsample),
not feat3's H/8/2048ch. Same "exact, provable no-op at init" property as
every other DINO-graph mechanism this session: at init, fuse_conv's output
is exactly 0 regardless of eig_proj's (non-zero-init) output, so
fused_after == fused_before exactly.
"""
import torch
import torch.nn as nn

from models.multi_scale_seg_head import MultiScaleSegHead


class MultiScaleSegHeadDecoderPrior(MultiScaleSegHead):
    def __init__(self, *args, dino_prior_num_eigvecs: int = 10,
                 dino_prior_hidden_channels: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        mid_channels = self.decode_conv2[0].out_channels  # matches parent's mid_channels

        self.dino_prior_eig_proj = nn.Conv2d(dino_prior_num_eigvecs, dino_prior_hidden_channels, kernel_size=1)
        self.dino_prior_fuse_conv = nn.Conv2d(
            mid_channels + dino_prior_hidden_channels, mid_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.dino_prior_fuse_conv.weight)
        nn.init.zeros_(self.dino_prior_fuse_conv.bias)

    def _inject_decoder_prior(self, fused, dino_eigvecs=None):
        """
        Overrides the base class's no-op hook (see MultiScaleSegHead.
        forward_features's 260801 change). fused: [B, mid_channels, H/4, W/4]
        (right after Step 3's upsample). dino_eigvecs: [B, g, H', W'] or None
        (computed externally by DinoGraphEigvecExtractor, passed in via
        rcf_model.py's forward_train/forward_eval).
        """
        if dino_eigvecs is None:
            return fused

        eigvecs = dino_eigvecs
        if eigvecs.shape[-2:] != fused.shape[-2:]:
            eigvecs = torch.nn.functional.interpolate(
                eigvecs, size=fused.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        eigvecs = eigvecs.detach()

        proj = self.dino_prior_eig_proj(eigvecs)
        delta = self.dino_prior_fuse_conv(torch.cat([fused, proj], dim=1))  # zero-init -> 0 at init
        return fused + delta
