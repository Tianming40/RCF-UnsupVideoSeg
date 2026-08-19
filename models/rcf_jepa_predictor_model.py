"""
RCFJepaPredictorModel: a more faithful JEPA-style redesign, discussed
260729 as a deliberately separate, independently-swappable experimental
line (kept in its own new files, NOT modifying rcf_jepa_flow_model.py or
its running job) after diagnosing that RCFJepaFlowModel's EMA-target fix
(job 510) stabilised the flow branch's OWN collapse but did NOT stop the
mask branch collapsing on essentially the same timeline as before (one
channel dying, the rest merging together by ~step 300) -- job 510's own
mask-collapse pattern showed the EMA-target trick alone doesn't fix
everything; something more structural was still missing.

Reframing (session discussion): RCFJepaFlowModel was only a PARTIAL JEPA
adoption -- its flow_head still did an explicit correlation SEARCH against
real target-encoder features (i.e. it could "peek" at the target), and still
produced an interpretable (dx,dy) field for the existing rigid/affine
decode_head to consume. This model instead implements the full JEPA
principle:

  - context encoder = backbone2 (ResNet50, actively trained), processes
    BOTH frames -- unchanged, still also feeds the mask branch (joint_attn0..3
    -> decode_head2, completely untouched, same as every RAFT-free variant
    this session).
  - target encoder = backbone2_ema (EMA-updated, stop-gradient shadow of
    backbone2 -- same existing infra as rcf_jepa_flow_model.py), processes
    BOTH frames, produces the REAL target representation the predictor is
    scored against. NEVER receives gradient.
  - predictor = JepaMotionPredictor (models/jepa_motion_predictor.py, new,
    small dilated-conv stack) -- BLIND: given ONLY one frame's context
    features, predicts what the target encoder would produce for the OTHER
    frame, without ever seeing that other frame's actual content (no
    correlation, no search, no warping). This blindness is the essential
    JEPA property forcing the predictor to learn genuine transformation
    priors rather than being able to look up the answer.
  - NO explicit flow field is produced anywhere in this model. Mask is NOT
    trained via a rigid/affine motion-clustering loss against a flow target
    -- decode_head (FlowAggregationHeadRaftFree) and decode_head3 (the
    residual head) are both constructed (inherited from the base class
    config schema) but UNUSED/never called in forward_train. This is the
    architecture-level "decoder mode change" this session's discussion
    concluded is necessary for a full JEPA adoption (a rigid/affine flow-
    fitting decode_head has no way to consume an abstract predicted
    latent -- see rcf_jepa_flow_model.py's own docstring for why that
    model kept the explicit-flow-field compromise instead).

What replaces it: the predictor's residual error -- even once well trained,
content that doesn't follow the predictor's learned priors (most saliently
the instrument's own independent motion, unpredictable from passively
watching background/tissue deformation patterns alone) will keep producing
higher prediction error than content that does. This residual error map IS
the segmentation-relevant signal, consumed via
_predict_error_consistency_loss: mask channels are pushed to separate by
how predictable their region is (some channels specialise into low-error/
predictable content, others into high-error/unpredictable content),
directly usable without any notion of an explicit displacement vector.

Known open risk (flagged explicitly, not yet empirically resolved): this
family of method (BYOL/DINO/JEPA) is well documented in the literature to
have its OWN collapse mode -- target encoder and predictor could jointly
learn to map everything to a near-constant vector, trivially minimising
prediction error without learning anything real. A VICReg-style (Bardes et
al. 2022) per-channel variance regulariser is included as a proactive
defence (given this session's repeated collapse history, added upfront
rather than waiting to observe collapse first) -- applied to the CONTEXT
encoder's own output (the target/EMA side has no gradient, regularising it
directly would be a no-op; the EMA side follows backbone2 anyway).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import utils
from models.rcf_joint_mask_v5_model import RCFJointMaskV5SoftTissueModel
from models.jepa_motion_predictor import JepaMotionPredictor

logger = utils.get_logger()


class RCFJepaPredictorModel(RCFJointMaskV5SoftTissueModel):
    def __init__(self, *args,
                 feat_channels=(256, 512, 1024, 2048),
                 predictor_hidden_channels: int = 256,
                 predictor_num_blocks: int = 4,
                 w_predict: float = 1.0,
                 w_variance: float = 1.0,
                 w_predict_cluster: float = 0.1,
                 w_balance: float = 1.0,
                 w_balance_per_sample: float = 1.0,
                 **kwargs):
        super().__init__(*args, feat_channels=feat_channels, **kwargs)
        assert self.backbone2_ema is not None, (
            "RCFJepaPredictorModel requires an EMA target encoder -- set "
            "model_kwargs.backbone2.create_ema: true in the config.")
        c0 = feat_channels[0]
        self.predictor = JepaMotionPredictor(
            channels=c0, hidden_channels=predictor_hidden_channels,
            num_blocks=predictor_num_blocks)
        self.w_predict = w_predict
        self.w_variance = w_variance
        self.w_predict_cluster = w_predict_cluster
        self.w_balance = w_balance
        self.w_balance_per_sample = w_balance_per_sample

    def forward(self, x, return_pred_vis_list=False):
        # Same override as RCFSelfTaughtFlowModel.forward (not inherited --
        # this class subclasses RCFJointMaskV5SoftTissueModel directly, see
        # module docstring on why): skip gt_fw_flows/gt_bw_flows stacking,
        # those keys don't exist when train/test_dataset_kwargs.load_flow: false.
        imgs, seq_ids, seq_names, paths = x['imgs'], x['seq_ids'], x['seq_names'], x['paths']
        imgs = torch.stack(imgs, dim=1)
        if self.training:
            gaps = x.get('gap', None)
            return self.forward_train(imgs, seq_ids, seq_names, paths, gaps=gaps)
        else:
            return self.forward_eval(imgs, seq_ids, seq_names, paths, return_pred_vis_list=return_pred_vis_list)

    # ------------------------------------------------------------------ #
    # k-means init for decode_head2.conv_seg -- copied from                #
    # rcf_selftaught_flow_model.py rather than inherited, so this model    #
    # class has no dependency on that (flow-based) variant's internals --  #
    # see this session's discussion, these two JEPA variants are meant to  #
    # be independently swappable. Identical logic/rationale, see that      #
    # file's docstring for the full writeup.                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def kmeans_init_mask_head(self, sample_batches, n_clusters=None, n_iters=30, temperature=None):
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
                feat2_joint=feat2_joint, feat3_joint=feat3_joint)
            feats.append(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))

        feats = torch.cat(feats, dim=0)
        max_points = 50000
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
        logger.info(f"RCFJepaPredictorModel: k-means-initialized decode_head2.conv_seg "
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

    # ------------------------------------------------------------------ #
    # Mask <-> prediction-error alignment loss                            #
    # ------------------------------------------------------------------ #
    def _predict_error_consistency_loss(self, masks, err_map):
        """
        masks: [B, C, h, w] softmax. err_map: [B, 1, h, w] JEPA prediction
        error (high where content is genuinely unpredictable from blind
        context alone -- plausibly foreground/independently-moving content).

        Pushes mask channels to separate by how predictable their assigned
        region is: each channel's mask-weighted average error should differ
        from the others' (maximise the across-channel variance of per-
        channel average error), WITHOUT presupposing which channel index
        "should" end up high-error -- channel-identity-agnostic, same spirit
        as _dino_consistency_loss's own channel-agnostic design.

        Returns a loss to MINIMISE (negative variance -> minimising this
        maximises the real variance).
        """
        B, C, H, W = masks.shape
        if err_map.shape[-2:] != (H, W):
            err_map = F.interpolate(err_map, size=(H, W), mode='bilinear', align_corners=False)
        W_c = masks.sum(dim=(2, 3))                                   # [B, C]
        mu_c = (masks * err_map).sum(dim=(2, 3)) / (W_c + 1e-6)       # [B, C]
        var = mu_c.var(dim=1)                                          # [B]
        return (-var).mean()

    @staticmethod
    def _channel_balance_loss(masks):
        """
        Batch-level (marginal) channel-usage balance regulariser, added
        260729 after four different motion-signal designs (v140 self-
        referential flow, v140+kmeans-conv_seg-init, v142 EMA-target flow,
        v143 blind-predictor+variance-reg -- all tried this session) showed
        the SAME mask-collapse pattern on a similar timeline regardless of
        how different their upstream motion signal was. That consistency
        across four otherwise-unrelated designs isolates the problem to
        decode_head2's OWN training dynamics, not to motion-signal quality
        -- this regulariser targets THAT directly, independent of whichever
        motion-signal mechanism sits upstream of it.

        masks: [B, C, h, w] softmax. Computes p_bar = the AVERAGE
        probability per channel pooled over the whole batch+space, and
        pushes its entropy toward the maximum (uniform, 1/C each) --
        WITHOUT touching any individual pixel's own confidence (a pixel can
        stay fully sharp/confident about its own single-channel choice;
        only the aggregate population-level usage is constrained). Standard
        anti-collapse technique in unsupervised clustering (SwAV,
        DeepCluster, IIC) -- NOT the same thing as per-pixel entropy
        (w_sharpen/get_sharpen_loss already handles that, unrelated,
        untouched by this).

        Returns a loss to MINIMISE (negative entropy -> minimising this
        maximises the real marginal entropy, i.e. pushes p_bar uniform).
        """
        p_bar = masks.mean(dim=(0, 2, 3))              # [C]
        entropy = -(p_bar * torch.log(p_bar.clamp(min=1e-8))).sum()
        return -entropy

    @staticmethod
    def _channel_balance_loss_per_sample(masks):
        """
        Per-sample (within-image) companion to _channel_balance_loss, added
        260729 after job 512 showed EXACTLY the batch-level balance loss
        held perfectly flat at -ln(5) (never violated) while loss_dino
        still collapsed and the mask viz revealed WHY: mask learned to
        assign an entire image almost wholly to ONE channel (varying WHICH
        channel by image -- e.g. pinkish-toned frames -> channel A,
        yellowish-toned frames -> channel B), satisfying batch-level balance
        (each channel still gets ~1/5 of the IMAGES) and even giving DINO an
        easy time (each channel's "cluster" = a set of similar-toned WHOLE
        images) without ever learning genuine WITHIN-image, pixel-level
        structure (instrument vs tissue vs background in the SAME frame).
        This is a real gap in the batch-level loss: pooling batch AND
        spatial dims together (masks.mean(dim=(0,2,3))) cannot distinguish
        "each channel used equally often across different whole images"
        from "each channel used equally often WITHIN every image" -- only
        the latter is the genuine per-pixel segmentation this project wants.

        masks: [B, C, h, w] softmax. Computes p_b = each SAMPLE's own
        spatially-pooled channel distribution (batch dim kept separate this
        time, only h,w pooled) and pushes EVERY sample's own entropy toward
        the maximum too -- directly penalising the "whole image -> one
        channel" shortcut, independent of (and complementary to) the
        batch-level term above.
        """
        B, C, H, W = masks.shape
        p_b = masks.mean(dim=(2, 3))                                    # [B, C]
        entropy_b = -(p_b * torch.log(p_b.clamp(min=1e-8))).sum(dim=1)   # [B]
        return -entropy_b.mean()

    @staticmethod
    def _variance_loss(feat):
        """VICReg-style (Bardes et al. 2022) per-channel variance floor,
        proactive anti-collapse defence -- see module docstring. feat:
        [B, C, H, W]; std computed over (batch, spatial) per channel."""
        f = feat.permute(1, 0, 2, 3).reshape(feat.shape[1], -1)
        std = f.std(dim=1)
        return F.relu(1.0 - std).mean()

    def forward_train(self, imgs, seq_ids, seq_names, paths, gaps=None):
        batch_size, im_num, num_channels, _h, _w = imgs.shape
        assert im_num == 2, "RCFJepaPredictorModel requires paired frames (im_num=2)"

        img_3 = imgs.view(batch_size * im_num, num_channels, _h, _w)
        all_feat = self.extract_feat(img_3, self.backbone2)          # context (trainable)
        all_feat_ema = self.extract_feat(img_3, self.backbone2_ema)  # target (EMA, no grad)

        # mask branch -- UNCHANGED, purely on context-encoder features
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        if self.allow_mask_resize and all_pred_mask.shape[-2:] != self.mask_size:
            all_pred_mask = self.resize(all_pred_mask, self.mask_size)
        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(batch_size, im_num, self.mask_layer, _feat_h, _feat_w)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)

        feat0_hw = all_feat[0].shape[-2:]
        feat0 = all_feat[0].view(batch_size, im_num, *all_feat[0].shape[1:])
        feat0_i, feat0_j = feat0[:, 0], feat0[:, 1]                          # context, both frames
        feat0_ema = all_feat_ema[0].view(batch_size, im_num, *all_feat_ema[0].shape[1:])
        feat0_i_ema, feat0_j_ema = feat0_ema[:, 0], feat0_ema[:, 1]          # target, both frames

        # ── blind prediction, no access to the other frame's real content ──
        pred_j_from_i = self.predictor(feat0_i)   # "imagine" frame1 from frame0 context alone
        pred_i_from_j = self.predictor(feat0_j)   # "imagine" frame0 from frame1 context alone

        err_fw = (pred_j_from_i - feat0_j_ema).abs().mean(dim=1, keepdim=True)  # [B,1,h,w]
        err_bw = (pred_i_from_j - feat0_i_ema).abs().mean(dim=1, keepdim=True)
        loss_predict = err_fw.mean() + err_bw.mean()

        loss_variance = self._variance_loss(feat0_i) + self._variance_loss(feat0_j)

        err_avg = (err_fw + err_bw) / 2
        loss_predict_cluster = self._predict_error_consistency_loss(all_pred_mask[:, 0], err_avg)

        loss_balance = self._channel_balance_loss(all_pred_mask[:, 0])
        loss_balance_per_sample = self._channel_balance_loss_per_sample(all_pred_mask[:, 0])

        loss = (loss_predict * self.w_predict
                + loss_variance * self.w_variance
                + loss_predict_cluster * self.w_predict_cluster
                + loss_balance * self.w_balance
                + loss_balance_per_sample * self.w_balance_per_sample)

        losses = {
            "loss_predict": loss_predict,
            "loss_variance": loss_variance,
            "loss_predict_cluster": loss_predict_cluster,
            "loss_balance": loss_balance,
            "loss_balance_per_sample": loss_balance_per_sample,
        }

        if self.w_dino > 0.0:
            l_dino = self._dino_consistency_loss(all_pred_mask[:, 0], imgs[:, 0])
            losses["loss_dino"] = l_dino
            loss = loss + self.w_dino * l_dino

        losses["loss"] = loss

        if self.train_iter % self.log_interval == 0:
            self._save_train_viz(all_pred_mask, imgs, feat0_hw, err_fw, err_bw, batch_size, im_num,
                                  seq_names, seq_ids, paths)

        # EMA momentum update -- backbone2_ema never receives gradient
        # directly; this is its only update mechanism.
        utils.momentum_update_param_and_buffer(src=self.backbone2, dest=self.backbone2_ema, m=self.ema_m)

        self.train_iter += 1
        return losses

    # ------------------------------------------------------------------ #
    # Visualization -- distinct panel set from rcf_jepa_flow_model.py's    #
    # (no flow_raw/confidence -- this model produces no explicit flow),   #
    # so NOT reusing that class's _save_train_viz/_compose_viz_grid.       #
    # ------------------------------------------------------------------ #
    def _save_train_viz(self, all_pred_mask, imgs, feat0_hw, err_fw, err_bw,
                        batch_size, im_num, seq_names, seq_ids, paths):
        try:
            with torch.no_grad():
                mask_mat = [
                    torch.cat([all_pred_mask[:, it, c:c + 1].repeat(1, 3, 1, 1) for it in range(im_num)], dim=0)
                    for c in range(self.mask_layer)
                ]
                img0_small = self.resize(imgs[:, 0], feat0_hw)
                img1_small = self.resize(imgs[:, 1], feat0_hw)
                imgs_vis = torch.cat([(img0_small + 2.0) / 4.0, (img1_small + 2.0) / 4.0], dim=0)

                def _heat(x):
                    x = x.repeat(1, 3, 1, 1)
                    return x / (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

                err_vis = torch.cat([_heat(err_fw), _heat(err_bw)], dim=0)

                tosave = torch.cat(mask_mat + [imgs_vis, err_vis], dim=2)
                img_frame_id = paths[0][0].split('/')[-1][:-4]
                fn_name = '{}/train_iter{:07d}_{}_{}_{}_img_pred_err.jpg'.format(
                    self.save_dir, self.train_iter, seq_names[0], seq_ids[0], img_frame_id)
                torchvision.utils.save_image(tosave, fn_name)
        except Exception:
            logger.warning("RCFJepaPredictorModel: train viz save failed", exc_info=True)

    @torch.no_grad()
    def visualize_eval_sample(self, batch, save_dir, epoch):
        """Same guarded hook pattern as RCFSelfTaughtFlowModel -- see
        main_tissue.py's validation_step (hasattr-guarded, no-op for other
        model classes)."""
        import os
        device = next(self.parameters()).device
        imgs = torch.stack(batch['imgs'], dim=1).to(device)
        batch_size, im_num = imgs.shape[:2]
        if im_num != 2:
            return

        img_3 = imgs.view(batch_size * im_num, *imgs.shape[2:])
        all_feat = self.extract_feat(img_3, self.backbone2)
        all_feat_ema = self.extract_feat(img_3, self.backbone2_ema)
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        _, _, fh, fw_ = all_pred_mask.shape
        all_pred_mask = F.softmax(
            all_pred_mask.view(batch_size, im_num, self.mask_layer, fh, fw_), dim=2)

        feat0_hw = all_feat[0].shape[-2:]
        feat0 = all_feat[0].view(batch_size, im_num, *all_feat[0].shape[1:])
        feat0_i, feat0_j = feat0[:, 0], feat0[:, 1]
        feat0_ema = all_feat_ema[0].view(batch_size, im_num, *all_feat_ema[0].shape[1:])
        feat0_i_ema, feat0_j_ema = feat0_ema[:, 0], feat0_ema[:, 1]

        pred_j_from_i = self.predictor(feat0_i)
        pred_i_from_j = self.predictor(feat0_j)
        err_fw = (pred_j_from_i - feat0_j_ema).abs().mean(dim=1, keepdim=True)
        err_bw = (pred_i_from_j - feat0_i_ema).abs().mean(dim=1, keepdim=True)

        try:
            mask_mat = [
                torch.cat([all_pred_mask[:, it, c:c + 1].repeat(1, 3, 1, 1) for it in range(im_num)], dim=0)
                for c in range(self.mask_layer)
            ]
            img0_small = self.resize(imgs[:, 0], feat0_hw)
            img1_small = self.resize(imgs[:, 1], feat0_hw)
            imgs_vis = torch.cat([(img0_small + 2.0) / 4.0, (img1_small + 2.0) / 4.0], dim=0)

            def _heat(x):
                x = x.repeat(1, 3, 1, 1)
                return x / (x.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

            err_vis = torch.cat([_heat(err_fw), _heat(err_bw)], dim=0)
            tosave = torch.cat(mask_mat + [imgs_vis, err_vis], dim=2)
            os.makedirs(save_dir, exist_ok=True)
            fn_name = os.path.join(save_dir, f'eval_err_viz_epoch{epoch:04d}.jpg')
            torchvision.utils.save_image(tosave, fn_name)
        except Exception:
            logger.warning("RCFJepaPredictorModel: eval viz save failed", exc_info=True)
