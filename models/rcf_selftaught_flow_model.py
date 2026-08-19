"""
RCFSelfTaughtFlowModel: removes this project's dependence on RAFT-precomputed
optical flow ENTIRELY (no gt_fw_flows/gt_bw_flows are loaded or consumed
anywhere in this model -- set train/test_dataset_kwargs.load_flow: false).
Motion is instead learned end-to-end, jointly with the mask, from THIS
dataset's own frame pairs via reconstruction. Discussed 260728.

CORRECTION (260728): job 506 (v140, gap1-only data) collapsed within ~20
steps -- flow_raw -> ~0 everywhere, mask -> near-uniform, all four loss
terms -> near-zero simultaneously, confirmed visually (flow_raw's colorized
viz panel was blank/white -- zero flow) and numerically. Root cause: the
soft-argmax correlation's output at/near random init is ~0 by construction
(a symmetric window's softmax, undiscriminating early on, averages to the
centre offset), and for small-motion gap1 pairs zero flow ALSO already
gives fairly low absolute reconstruction error -- two forces reinforcing the
same trivial point with very little gradient ever pushing the network away
from it. Fixed two ways (see forward_train and config headers): (1) training
data reverted to a gap1+2+3 mix (larger true motion makes zero flow a
genuinely bad, not merely mediocre, solution), (2) loss_flow_recon changed
from absolute reconstruction error to error IMPROVEMENT over a detached
zero-flow baseline, removing "predict zero" as a rewarded point at all.

Motivation: script/diagnose_v102_bottleneck.py (this session) found mask
miss-rate on true foreground pixels correlates strongly with RAFT's OWN
cycle-consistency confidence (3x+ gap between low/high-confidence bins) --
RAFT's synthetic-data domain-transfer error is a bigger source of trouble
here than anything wrong with the ResNet appearance features themselves.
Rather than trying to denoise/clean RAFT's output, this model removes RAFT
from the loop: motion can never inherit a domain gap it was never given.

Architecture (subclasses RCFJointMaskV5SoftTissueModel to inherit its
joint_attn0..3 cross-frame correlation modules and _decode_head_forward
override UNCHANGED -- mask still sees both frames every step, exactly as
v138):
  ResNet50 (shared) -> 4-scale features, as always
    -> joint_attn0..3 (v138, UNCHANGED)   -> feeds decode_head2 (mask)
    -> LocalCorrelationFlowHeadBidirectional (NEW, models/local_correlation_flow_head.py)
       coarse=feat2 (1024ch, 48x48), fine=feat0 (256ch, 96x96), coarse-to-fine
       warp-refine (PWC-Net style) -> flow_raw_fw, flow_raw_bw, in feat0's
       OWN 96x96-grid pixel units.
       Trained by THIS model's forward_train via:
         loss_flow_recon:  feature reconstruction, warp(feat0_j, flow) vs feat0_i
         loss_flow_smooth: edge-aware smoothness (regularises low-texture
                            interior, where reconstruction alone is ambiguous
                            -- the aperture problem)
         loss_flow_cycle:  forward/backward consistency
       Per-pixel reconstruction error also yields a confidence map, fed into
       decode_head (FlowAggregationHeadRaftFree) IN PLACE OF the old static
       boundary-angle detector (HQAM) -- dynamic, self-diagnosing, no
       hand-tuned angle threshold.
    -> decode_head3 (UNCHANGED) -> free per-pixel residual correction
    -> decode_head (FlowAggregationHeadRaftFree, models/flow_aggregation_head_raftfree.py):
       same per-channel rigid+affine+residual motion fitting as always,
       fit against flow_raw.detach() (detached: mask/decode_head must not be
       able to corrupt the flow estimate towards "whatever's easiest to
       explain" instead of "physically accurate" -- same collapse concern
       discussed for the C-LaV latent-denoising idea earlier this session).
       topk (LQCD) KEPT, config raises it 4->6 (relaxed, not removed --
       protects against a DIFFERENT failure mode, whole-frame content being
       fundamentally uninterpretable, that self-taught flow can't fix
       either).

Unit-convention note (easy to get wrong, see forward_train): flow_raw comes
out of LocalCorrelationFlowHeadBidirectional in feat0's OWN 96x96-grid pixel
units (needed for flow_warp against feat0-resolution tensors in the
reconstruction/smoothness/cycle losses). decode_head's calibrated constants
(clamp_flow_t=10, residual_adjustment_scale=10, pred_div_coeff=10) were all
tuned against the OLD convention -- RAFT flow kept in ORIGINAL crop-pixel
units even after spatial downsampling to mask_size (rcf_model.py's
_resize_gt_flow never rescales magnitude, only resamples the grid). To keep
those constants meaningful unchanged, flow_raw is rescaled by feat0's
downsample factor (crop_size / feat0_size, typically 384/96=4) ONLY on the
copy fed into decode_head; the native (unscaled) flow_raw is what
flow_warp/smoothness/cycle actually use.
"""
import os

import torch
import torch.nn.functional as F
import torchvision

import utils
from models.rcf_joint_mask_v5_model import RCFJointMaskV5SoftTissueModel
from models.local_correlation_flow_head import LocalCorrelationFlowHeadBidirectional
from utils.warp_utils import flow_warp, edge_aware_smoothness_loss

logger = utils.get_logger()


class RCFSelfTaughtFlowModel(RCFJointMaskV5SoftTissueModel):
    def __init__(self, *args,
                 feat_channels=(256, 512, 1024, 2048),
                 flow_proj_channels: int = 32,
                 flow_coarse_radius: int = 4,
                 flow_fine_radius: int = 3,
                 w_flow_recon: float = 1.0,
                 w_flow_smooth: float = 0.1,
                 w_flow_cycle: float = 0.1,
                 recon_conf_sigma: float = 0.5,
                 **kwargs):
        super().__init__(*args, feat_channels=feat_channels, **kwargs)
        c0, c1, c2, c3 = feat_channels
        self.flow_head = LocalCorrelationFlowHeadBidirectional(
            coarse_channels=c2, fine_channels=c0,
            proj_channels=flow_proj_channels,
            coarse_radius=flow_coarse_radius, fine_radius=flow_fine_radius)
        self.w_flow_recon = w_flow_recon
        self.w_flow_smooth = w_flow_smooth
        self.w_flow_cycle = w_flow_cycle
        self.recon_conf_sigma = recon_conf_sigma

    def forward(self, x, return_pred_vis_list=False):
        # Overrides RCFModel.forward: skip the gt_fw_flows/gt_bw_flows
        # stacking (main.py's Model.forward path never runs this -- self.model
        # IS this class, called directly) -- those keys don't exist in the
        # batch at all when train/test_dataset_kwargs.load_flow: false.
        imgs, seq_ids, seq_names, paths = x['imgs'], x['seq_ids'], x['seq_names'], x['paths']
        imgs = torch.stack(imgs, dim=1)
        if self.training:
            gaps = x.get('gap', None)
            return self.forward_train(imgs, seq_ids, seq_names, paths, gaps=gaps)
        else:
            return self.forward_eval(imgs, seq_ids, seq_names, paths, return_pred_vis_list=return_pred_vis_list)

    @torch.no_grad()
    def kmeans_init_mask_head(self, sample_batches, n_clusters=None, n_iters=30, temperature=None):
        """
        Initializes decode_head2.conv_seg from a k-means clustering of REAL
        pre-classifier features (models/multi_scale_seg_head_joint4.py's
        forward_features, added alongside this method) instead of the
        small-random-normal default (models/multi_scale_seg_head.py:
        std=0.01 -- designed only to avoid an extreme initial softmax,
        carries no information about the actual data).

        Added 260728, diagnosing why mask collapsed to one dominant channel
        by step ~600 of job 508: decode_head2's classifier starts from pure
        noise, and with the flow branch's own signal still weak/undeveloped
        early on (see forward_train's baseline-relative loss_flow_recon,
        deliberately slow to develop after job 506/507's zero-flow collapse
        fixes), there is nothing informative pulling conv_seg toward a
        genuine 5-way split for a while -- softmax's winner-take-all dynamic
        makes collapsing into one dominant channel the path of least
        resistance during that gap.

        DenseCL's pretrained backbone ALREADY carries real appearance-
        clustering structure (that is what dense contrastive pretraining is
        for) -- decode_head2 simply doesn't know how to READ it yet at
        random init. Running k-means directly in decode_head2's own
        pre-classifier feature space (mid_channels-dim, NOT a proxy space
        like DINO's -- must match exactly what conv_seg will consume) and
        initializing conv_seg to reproduce that clustering (soft nearest-
        centroid, via the standard identity
        -||x-c||^2 = 2x.c - |c|^2 - |x|^2, last term class-independent so it
        drops out of softmax -> weight=2c/T, bias=-|c|^2/T) gives mask a
        REAL starting point grounded in actual appearance structure, instead
        of noise -- without needing any training-schedule change.

        Called once from main_tissue.py's on_train_start (guarded by
        hasattr so every other model class is unaffected; skipped on resume
        so a checkpoint's already-learned conv_seg is never clobbered).

        sample_batches: list of raw batch dicts (as yielded by the train
        dataloader, i.e. batch['imgs'] etc, NOT yet stacked) -- a handful
        (~4 batches) is enough; k-means only needs a representative sample
        of the feature distribution, not the whole dataset.
        """
        n_clusters = n_clusters or self.mask_layer
        device = next(self.parameters()).device
        feats = []
        for batch in sample_batches:
            imgs = torch.stack(batch['imgs'], dim=1).to(device)
            B, I, C, H, W = imgs.shape
            img_3 = imgs.view(B * I, C, H, W)
            all_feat = self.extract_feat(img_3, self.backbone2)
            feat0, feat1, feat2, feat3 = all_feat
            flow_feat = self.joint_attn0(feat0)
            feat1_joint = self.joint_attn1(feat1)
            feat2_joint = self.joint_attn2(feat2)
            feat3_joint = self.joint_attn3(feat3)
            x = self.decode_head2.forward_features(
                all_feat, flow_feat=flow_feat, feat1_joint=feat1_joint,
                feat2_joint=feat2_joint, feat3_joint=feat3_joint)   # [B*I, mid_ch, h, w]
            feats.append(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))

        feats = torch.cat(feats, dim=0)   # [N_pixels, mid_ch]
        max_points = 50000   # keep k-means (O(N*K*iters)) cheap
        if feats.shape[0] > max_points:
            idx = torch.randperm(feats.shape[0], device=feats.device)[:max_points]
            feats = feats[idx]

        centroids = self._kmeans_plus_plus_init(feats, n_clusters)
        for _ in range(n_iters):
            dists = torch.cdist(feats, centroids)
            assign = dists.argmin(dim=1)
            new_centroids = torch.stack([
                feats[assign == k].mean(dim=0) if (assign == k).any() else centroids[k]
                for k in range(n_clusters)
            ])
            if torch.allclose(new_centroids, centroids, atol=1e-5):
                centroids = new_centroids
                break
            centroids = new_centroids

        if temperature is None:
            pdist = torch.cdist(centroids, centroids) ** 2
            temperature = pdist[pdist > 0].mean().clamp(min=1e-3).item()

        weight = (2.0 * centroids / temperature).view(n_clusters, -1, 1, 1)
        bias = -(centroids ** 2).sum(dim=1) / temperature

        conv_seg = self.decode_head2.conv_seg
        conv_seg.weight.data.copy_(weight.to(conv_seg.weight.dtype))
        conv_seg.bias.data.copy_(bias.to(conv_seg.bias.dtype))
        logger.info(f"RCFSelfTaughtFlowModel: k-means-initialized decode_head2.conv_seg "
                    f"from {feats.shape[0]} real pixel features, temperature={temperature:.4f}")

    @staticmethod
    def _kmeans_plus_plus_init(feats, k):
        N = feats.shape[0]
        centroids = [feats[torch.randint(N, (1,), device=feats.device)][0]]
        for _ in range(1, k):
            c = torch.stack(centroids)
            d2 = torch.cdist(feats, c).min(dim=1).values ** 2
            probs = d2 / d2.sum().clamp(min=1e-12)
            idx = torch.multinomial(probs, 1).item()
            centroids.append(feats[idx])
        return torch.stack(centroids)

    def forward_train(self, imgs, seq_ids, seq_names, paths, gaps=None):
        batch_size, im_num, num_channels, _h, _w = imgs.shape
        assert im_num == 2, "RCFSelfTaughtFlowModel requires paired frames (im_num=2)"

        img_3 = imgs.view(batch_size * im_num, num_channels, _h, _w)
        all_feat = self.extract_feat(img_3, self.backbone2)

        # mask branch -- inherited _decode_head_forward (RCFJointMaskV5SoftTissueModel,
        # unchanged) injects joint_attn0..3's cross-frame features automatically
        # whenever decode_head2.use_flow_feat is set.
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        if self.allow_mask_resize and all_pred_mask.shape[-2:] != self.mask_size:
            all_pred_mask = self.resize(all_pred_mask, self.mask_size)

        if self.separate_residual:
            all_pred_residual_fw, all_pred_residual_bw = self.pred_separate_residual(all_feat, batch_size, im_num)
        else:
            all_pred_residual_fw, all_pred_residual_bw = self.pred_joint_residual(
                all_feat[-1].unflatten(0, (batch_size, im_num)))

        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(batch_size, im_num, self.mask_layer, _feat_h, _feat_w)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)

        # ── self-taught flow (RAFT-free) ────────────────────────────────
        # native units: feat0's OWN grid (96x96 typically), required for
        # flow_warp against feat0-resolution tensors below.
        flow_raw_fw, flow_raw_bw = self.flow_head(all_feat[2], all_feat[0])

        feat0_hw = all_feat[0].shape[-2:]
        feat0 = all_feat[0].view(batch_size, im_num, *all_feat[0].shape[1:])
        feat0_i, feat0_j = feat0[:, 0], feat0[:, 1]

        recon_err_fw = (feat0_i - flow_warp(feat0_j, flow_raw_fw, pad='border')).abs().mean(dim=1, keepdim=True)
        recon_err_bw = (feat0_j - flow_warp(feat0_i, flow_raw_bw, pad='border')).abs().mean(dim=1, keepdim=True)

        # Baseline-relative reconstruction loss (added 260728, after job 506
        # collapsed to near-zero flow within ~20 steps): the soft-argmax
        # correlation's default output at/near random init is ~0 (a
        # symmetric window's softmax, undiscriminating early on, averages to
        # the centre offset by construction) and, for small-motion pairs,
        # zero flow ALSO already gives a fairly low absolute reconstruction
        # error -- two forces reinforcing the same trivial point, with very
        # little gradient ever pushing the network away from it. Penalising
        # ABSOLUTE reconstruction error rewards "predict zero, frames looked
        # similar anyway" exactly as much as genuine correspondence-finding.
        # Instead, score against the zero-flow BASELINE error (detached --
        # must not be gameable by widening the baseline itself via the
        # backbone) and only reward IMPROVEMENT over doing nothing:
        # baseline_err - recon_err > 0 rewarded (this flow explains the
        # pixel better than no motion at all), == 0 for the trivial zero
        # solution (no credit, not a minimum to settle into), regions where
        # baseline_err is already tiny (genuinely static content) contribute
        # ~nothing either way since there's nothing to gain there regardless
        # of what flow says -- self-weighting by how much real signal exists.
        baseline_err_fw = (feat0_i - feat0_j).abs().mean(dim=1, keepdim=True).detach()
        baseline_err_bw = baseline_err_fw  # identical pair, symmetric

        # Weight the per-pixel loss by baseline_err itself (added 260728,
        # after visually confirming flow/mask WERE developing real structure
        # by step ~200 but the scalar loss_flow_recon stayed pinned near 0):
        # a plain .mean() over the whole image divides by ALL 96x96 pixels,
        # but the large majority (static background/tissue with nothing to
        # explain) contribute ~0 signal regardless of flow quality -- this
        # doesn't just make the LOGGED number small, it dilutes the actual
        # GRADIENT for the pixels that matter (motion/instrument regions) by
        # the same factor, since .mean() scales every pixel's gradient
        # contribution by 1/N uniformly. Weighting by baseline_err.detach()
        # (large where the two frames already differ -- i.e. where there is
        # something to explain -- near-zero where genuinely static) is the
        # data-derived analogue of HQAM's boundary restriction: concentrates
        # gradient on informative pixels WITHOUT reintroducing RAFT or a
        # hand-tuned angle threshold, since baseline_err is computed directly
        # from feat0_i/feat0_j (no flow, no RAFT) and is naturally large
        # almost exactly where real motion/boundary structure exists.
        eps = 1e-6
        w_fw = baseline_err_fw
        w_bw = baseline_err_bw
        loss_flow_recon_fw = ((recon_err_fw - baseline_err_fw) * w_fw).sum(dim=(1, 2, 3)) / (w_fw.sum(dim=(1, 2, 3)) + eps)
        loss_flow_recon_bw = ((recon_err_bw - baseline_err_bw) * w_bw).sum(dim=(1, 2, 3)) / (w_bw.sum(dim=(1, 2, 3)) + eps)
        loss_flow_recon = loss_flow_recon_fw.mean() + loss_flow_recon_bw.mean()

        img0_small = self.resize(imgs[:, 0], feat0_hw)
        img1_small = self.resize(imgs[:, 1], feat0_hw)
        loss_flow_smooth = (edge_aware_smoothness_loss(flow_raw_fw, img0_small)
                             + edge_aware_smoothness_loss(flow_raw_bw, img1_small))

        cycle_err = (flow_raw_fw + flow_warp(flow_raw_bw, flow_raw_fw, pad='border')).abs().mean(dim=1, keepdim=True)
        loss_flow_cycle = cycle_err.mean()

        loss_flow_selftaught = (loss_flow_recon
                                 + self.w_flow_smooth * loss_flow_smooth
                                 + self.w_flow_cycle * loss_flow_cycle)

        # confidence from the flow head's OWN reconstruction error, replaces
        # HQAM's static boundary-angle detector (see
        # models/flow_aggregation_head_raftfree.py's docstring)
        conf_fw = torch.exp(-recon_err_fw.detach() / self.recon_conf_sigma)
        conf_bw = torch.exp(-recon_err_bw.detach() / self.recon_conf_sigma)
        self.decode_head.set_external_weights(conf_fw, conf_bw)

        # rescale to decode_head's calibrated (old RAFT-at-mask_size) unit
        # convention -- see module docstring -- and detach: decode_head/mask
        # must not be able to push the flow estimate towards "easiest to
        # explain", only towards "physically accurate" (that's L_recon's job).
        downsample_factor = _w / feat0_hw[-1]
        flow_raw_fw_in = (flow_raw_fw * downsample_factor).detach().unsqueeze(1)  # [B,1,2,h,w]
        flow_raw_bw_in = (flow_raw_bw * downsample_factor).detach().unsqueeze(1)

        pred_flows, loss_flow = self.decode_head(
            imgs, all_pred_mask, flow_raw_fw_in, flow_raw_bw_in,
            all_pred_residual_fw, all_pred_residual_bw, seq_names=seq_names, gaps=gaps)

        loss_warp_seg = loss_flow['seg']

        if self.train_iter % self.log_interval == 0:
            self._save_train_viz(all_pred_mask, imgs, img0_small, img1_small,
                                  flow_raw_fw, flow_raw_bw, recon_err_fw, recon_err_bw,
                                  conf_fw, conf_bw, batch_size, im_num,
                                  seq_names, seq_ids, paths)
        losses = {
            "loss_warp_seg": loss_warp_seg,
            "loss_flow_recon": loss_flow_recon,
            "loss_flow_smooth": loss_flow_smooth,
            "loss_flow_cycle": loss_flow_cycle,
        }
        loss = loss_warp_seg * self.w_seg + loss_flow_selftaught * self.w_flow_recon

        if self.w_dino > 0.0:
            l_dino = self._dino_consistency_loss(all_pred_mask[:, 0], imgs[:, 0])
            losses["loss_dino"] = l_dino
            loss = loss + self.w_dino * l_dino

        losses["loss"] = loss
        self.train_iter += 1
        return losses

    # ------------------------------------------------------------------ #
    # Visualization -- training (periodic, every log_interval steps) and #
    # eval (one image per validation epoch, see main_tissue.py's guarded #
    # hasattr(self.model, 'visualize_eval_sample') call).                #
    # ------------------------------------------------------------------ #
    def _compose_viz_grid(self, all_pred_mask, img0_small, img1_small,
                          flow_raw_fw, flow_raw_bw, recon_err_fw, recon_err_bw,
                          conf_fw, conf_bw, batch_size, im_num):
        """Builds one [2*B, 3, total_H, W] tensor: rows = frame0's B samples
        then frame1's B samples (matches mask_mat's convention below);
        columns (stacked along H, torchvision.utils.save_image then lays out
        rows into a grid) = mask ch0..4, frame images, flow_raw (colorized),
        reconstruction error (heatmap), confidence (heatmap)."""
        mask_mat = [
            torch.cat([all_pred_mask[:, it, c:c + 1].repeat(1, 3, 1, 1) for it in range(im_num)], dim=0)
            for c in range(self.mask_layer)
        ]  # each [2*B, 3, h, w]

        imgs_vis = torch.cat([(img0_small + 2.0) / 4.0, (img1_small + 2.0) / 4.0], dim=0)

        flow_color_fw = torch.cat([self.let_tensor_vis(flow_raw_fw[b:b + 1]) for b in range(batch_size)], dim=0) / 255.0
        flow_color_bw = torch.cat([self.let_tensor_vis(flow_raw_bw[b:b + 1]) for b in range(batch_size)], dim=0) / 255.0
        flow_vis_cat = torch.cat([flow_color_fw, flow_color_bw], dim=0)

        def _heat(x):
            x = x.repeat(1, 3, 1, 1)
            return x / (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

        err_vis = torch.cat([_heat(recon_err_fw), _heat(recon_err_bw)], dim=0)
        conf_vis = torch.cat([conf_fw.repeat(1, 3, 1, 1), conf_bw.repeat(1, 3, 1, 1)], dim=0)

        return torch.cat(mask_mat + [imgs_vis, flow_vis_cat, err_vis, conf_vis], dim=2)

    def _save_train_viz(self, all_pred_mask, imgs, img0_small, img1_small,
                        flow_raw_fw, flow_raw_bw, recon_err_fw, recon_err_bw,
                        conf_fw, conf_bw, batch_size, im_num, seq_names, seq_ids, paths):
        try:
            with torch.no_grad():
                tosave = self._compose_viz_grid(
                    all_pred_mask, img0_small, img1_small, flow_raw_fw, flow_raw_bw,
                    recon_err_fw, recon_err_bw, conf_fw, conf_bw, batch_size, im_num)
                img_frame_id = paths[0][0].split('/')[-1][:-4]
                fn_name = '{}/train_iter{:07d}_{}_{}_{}_img_pred_flow_recon.jpg'.format(
                    self.save_dir, self.train_iter, seq_names[0], seq_ids[0], img_frame_id)
                torchvision.utils.save_image(tosave, fn_name)
        except Exception:
            logger.warning("RCFSelfTaughtFlowModel: train viz save failed", exc_info=True)

    @torch.no_grad()
    def visualize_eval_sample(self, batch, save_dir, epoch):
        """Called by main_tissue.py's TissueModel.validation_step (guarded by
        hasattr, no-op for every other model class) once per validation
        epoch (first batch only) -- saves the SAME panel layout as training's
        viz (mask/images/flow_raw/recon-error/confidence), but on real
        held-out eval frames instead of a training crop, so flow_raw's
        quality can be checked directly against annotated data over the
        course of training, not just training-crop sanity checks."""
        device = next(self.parameters()).device
        imgs = torch.stack(batch['imgs'], dim=1).to(device)   # [B, im_num, 3, H, W]
        batch_size, im_num = imgs.shape[:2]
        if im_num != 2:
            return

        img_3 = imgs.view(batch_size * im_num, *imgs.shape[2:])
        all_feat = self.extract_feat(img_3, self.backbone2)
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        _, _, fh, fw_ = all_pred_mask.shape
        all_pred_mask = F.softmax(
            all_pred_mask.view(batch_size, im_num, self.mask_layer, fh, fw_), dim=2)

        flow_raw_fw, flow_raw_bw = self.flow_head(all_feat[2], all_feat[0])
        feat0_hw = all_feat[0].shape[-2:]
        feat0 = all_feat[0].view(batch_size, im_num, *all_feat[0].shape[1:])
        feat0_i, feat0_j = feat0[:, 0], feat0[:, 1]
        recon_err_fw = (feat0_i - flow_warp(feat0_j, flow_raw_fw, pad='border')).abs().mean(dim=1, keepdim=True)
        recon_err_bw = (feat0_j - flow_warp(feat0_i, flow_raw_bw, pad='border')).abs().mean(dim=1, keepdim=True)
        conf_fw = torch.exp(-recon_err_fw / self.recon_conf_sigma)
        conf_bw = torch.exp(-recon_err_bw / self.recon_conf_sigma)

        img0_small = self.resize(imgs[:, 0], feat0_hw)
        img1_small = self.resize(imgs[:, 1], feat0_hw)

        try:
            tosave = self._compose_viz_grid(
                all_pred_mask, img0_small, img1_small, flow_raw_fw, flow_raw_bw,
                recon_err_fw, recon_err_bw, conf_fw, conf_bw, batch_size, im_num)
            os.makedirs(save_dir, exist_ok=True)
            fn_name = os.path.join(save_dir, f'eval_flow_viz_epoch{epoch:04d}.jpg')
            torchvision.utils.save_image(tosave, fn_name)
        except Exception:
            logger.warning("RCFSelfTaughtFlowModel: eval viz save failed", exc_info=True)
