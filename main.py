# This file references the pytorch lightning template, with the following license:
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from datetime import datetime
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import dataset
import models
import utils
import wandb

logger = utils.get_logger()
# test


def _guided_filter(guide_rgb, src, r=16, eps=1e-2):
    """Guided filter using grayscale version of guide_rgb.

    guide_rgb : [B, 3, H, W]  normalised image (same range as training)
    src       : [B, C, H, W]  soft probability maps to be filtered
    r         : box filter radius (window = 2r+1)
    eps       : edge threshold — smaller keeps harder edges, larger blurs more
    Returns   : [B, C, H, W]  smoothed probability maps, clamped to [0, 1]
    """
    I = (0.299 * guide_rgb[:, 0:1] + 0.587 * guide_rgb[:, 1:2]
         + 0.114 * guide_rgb[:, 2:3])  # [B, 1, H, W]
    k = 2 * r + 1

    def mean_f(x):
        return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='reflect'), k, stride=1, padding=0)

    mean_I  = mean_f(I)
    mean_p  = mean_f(src)
    mean_Ip = mean_f(I * src)
    cov_Ip  = mean_Ip - mean_I * mean_p
    var_I   = mean_f(I * I) - mean_I * mean_I

    a       = cov_Ip / (var_I + eps)
    b       = mean_p - a * mean_I

    return (mean_f(a) * I + mean_f(b)).clamp(0, 1)


def _apply_progressive_unfreeze(model, stage: dict, epoch: int) -> None:
    """Apply one step of a progressive-unfreeze schedule to `model`.

    stage keys:
      freeze_backbone_stages  int   0 = all backbone trainable; N = freeze stem+layer1..layerN
      freeze_flow_head        bool  True = freeze decode_head + decode_head3
      freeze_all_bn           bool  True = freeze ALL BN running stats globally (adapter excluded)

    freeze_all_bn rationale
    -----------------------
    Calling .eval() only on frozen backbone stages creates a mode mismatch:
    backbone uses Phase-1 running stats while FCNHead uses batch stats → worse
    than leaving everything in train mode.  The correct fix is to freeze ALL
    BN running_mean/var so the entire pipeline is consistent with Phase-1
    statistics at both train and val time.  BN affine params (gamma/beta) keep
    their requires_grad and still adapt via gradient.
    """
    freeze_backbone_stages = stage.get('freeze_backbone_stages', 0)
    freeze_flow_head       = stage.get('freeze_flow_head', False)
    freeze_all_bn          = stage.get('freeze_all_bn', False)

    # DINO encoder is always frozen — collect its param ids
    dino_ids: set = set()
    if hasattr(model, 'dino') and model.dino is not None:
        dino_ids = {id(p) for p in model.dino.parameters()}

    # Step 1: unfreeze everything except DINO (PL has already called model.train())
    for p in model.parameters():
        if id(p) not in dino_ids:
            p.requires_grad_(True)

    # Step 2: re-apply backbone stage freeze (requires_grad only — no eval here)
    if freeze_backbone_stages > 0 and hasattr(model, 'backbone2'):
        stem_names = {'conv1', 'bn1'}
        stage_names = {f'layer{i}' for i in range(1, freeze_backbone_stages + 1)}
        freeze_names = stem_names | stage_names
        for name, param in model.backbone2.named_parameters():
            if name.split('.')[0] in freeze_names:
                param.requires_grad_(False)

    # Step 3: re-apply flow-head freeze (requires_grad only)
    if freeze_flow_head:
        for attr in ('decode_head', 'decode_head3'):
            if hasattr(model, attr):
                for p in getattr(model, attr).parameters():
                    p.requires_grad_(False)

    # Step 4: freeze ALL BN running stats globally (adapter BN excluded)
    #   .eval() locks running_mean/var so the full pipeline is consistent with
    #   Phase-1 statistics at train AND val — no crop-vs-full-res drift.
    #   BN weight/bias (affine) still have requires_grad=True and keep updating.
    if freeze_all_bn:
        adapter_mod_ids: set = set()
        for name, m in model.named_modules():
            if 'adapter' in name:
                adapter_mod_ids.add(id(m))
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) and id(m) not in adapter_mod_ids:
                m.eval()

    trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    frozen_m    = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    logger.info(
        f"[ProgressiveUnfreeze] Epoch {epoch}: "
        f"freeze_backbone_stages={freeze_backbone_stages}, "
        f"freeze_flow_head={freeze_flow_head}, "
        f"freeze_all_bn={freeze_all_bn} → "
        f"trainable={trainable_m:.2f}M  frozen={frozen_m:.2f}M"
    )

class Model(pl.LightningModule):
    def __init__(self, args, trainer):
        super().__init__()
        if wandb.run is not None:
            wandb.run.log_code(".")

        preferred_linalg_library = getattr(args, "preferred_linalg_library", None)
        if preferred_linalg_library is not None and preferred_linalg_library != "do_not_set":
            torch.backends.cuda.preferred_linalg_library(args.preferred_linalg_library)
            logger.info(f"Setting preferred linalg library to {args.preferred_linalg_library}")

        if args.rank <= 0:
            os.makedirs(args.checkpoints_dir,
                        exist_ok=args.allow_overwriting_checkpoints_dir)
            if not args.test:
                os.makedirs(os.path.join(args.checkpoints_dir, "saved"),
                            exist_ok=args.allow_overwriting_checkpoints_dir)
            # We may have validation in training
            os.makedirs(os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_dir_name", "saved_eval")),
                        exist_ok=args.allow_overwriting_checkpoints_dir)
            os.makedirs(os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_export_dir_name", "saved_eval_export")),
                        exist_ok=args.allow_overwriting_checkpoints_dir)
        if args.rank >= 0:
            torch.cuda.set_device(args.rank)
            trainer.strategy.barrier()

            self.world_size = trainer.strategy.world_size
        else:
            self.world_size = 1

        dataset_cls_name = getattr(args, "dataset_cls", "VideoDataset")
        self.dataset_cls = dataset.__dict__[dataset_cls_name]
        logger.info(f"Using dataset: {dataset_cls_name}")

        self.save_hyperparameters(args, ignore="model")
        self.args = args
        self.model = models.__dict__[args.model_cls](args, **args.model_kwargs)

        if args.pretrained_model is not None and not getattr(args, 'resume_from_checkpoint', None):
            if "*" in args.pretrained_model:
                potential_matches = glob(args.pretrained_model)
                assert len(
                    potential_matches) == 1, f"{potential_matches} is not unique"
                args.pretrained_model = potential_matches[0]
            logger.info(
                f"Loading pretrained model from {args.pretrained_model}")
            ckpt = torch.load(args.pretrained_model, map_location="cpu")

            # If the checkpoint is the main model, we load the main model.
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            if getattr(self.args, "pretrained_model_backbone_only", False):
                state_dict = {k:v for k, v in state_dict.items() if "backbone" in k}
            example_key = list(state_dict.keys())[0]
            
            ema_loaded = False
            ema_in_model = hasattr(self.model, 'backbone2_ema') and self.model.backbone2_ema is not None
            if example_key.startswith("model."):
                # Main model
                # ema exists in the state_dict
                ema_in_state_dict = len([k for k in state_dict.keys() if '_ema' in k])
                if ema_in_model and not ema_in_state_dict:
                    assert self.model.decode_head2_ema is not None, "decode_head2_ema is not enabled with backbone2_ema enabled"
                    logger.info("Detected EMA in model but not in state_dict, loading state_dict to both the main model and ema")
                    ema_state_dict = {
                        k.replace("backbone2", "backbone2_ema").replace("decode_head2", "decode_head2_ema"): v for k, v in state_dict.items() if "backbone2" in k or "decode_head2" in k
                    }
                    state_dict_with_ema = {**state_dict, **ema_state_dict}
                else:
                    # Do not need to modify the state_dict
                    state_dict_with_ema = state_dict
                
                if getattr(self.args, "drop_head_decode_head2", False):
                    logger.info("Dropping decode_head2 (with ema)")
                    state_dict_with_ema = {k: v for k, v in state_dict_with_ema.items() if "decode_head2" not in k}
                
                mismatches = self.load_state_dict(state_dict_with_ema, strict=False)
                ema_loaded = True
            elif example_key.startswith("module."):
                # Moco model
                model_prefix = 'module.encoder_q'
                
                for k in list(state_dict.keys()):
                    # retain only student model up to before the embedding layer
                    if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
                        # remove prefix
                        new_key = k.replace(model_prefix + '.', "")
                        state_dict[new_key] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                mismatches = self.model.backbone2.load_state_dict(
                    state_dict, strict=False)
            elif 'conv1.weight' in state_dict.keys():
                # DenseCL model
                mismatches = self.model.backbone2.load_state_dict(state_dict, strict=False)
            elif 'backbone2.conv1.weight' in state_dict.keys():
                # Only the model in the main model
                mismatches = self.model.load_state_dict(state_dict, strict=False)
            else:
                raise ValueError("Unknown module in state_dict")

            if not ema_loaded:
                # To implement, call `self.model.init_ema()`
                assert (getattr(self.model, "backbone2_ema", None) is None) and (getattr(self.model, "decode_head2_ema", None) is None), "EMA is enabled but weights are not loaded"

            logger.info(f"Mismatches in the pretrained model: {mismatches}")
        else:
            logger.info("Not loading pretrained model")

        self.accumulate_training_loss = {}

        self.object_channel = args.object_channel if args.object_channel is not None else os.environ.get("OBJECT_CHANNEL", None)
        if isinstance(self.object_channel, str):
            self.object_channel = int(self.object_channel)
        # self.args.object_channel is for global use
        self.args.object_channel = self.object_channel
        logger.info(f"Using {self.object_channel} as object channel")
        # Per-sequence channels: {seq_name: [ch_idx, ...]}, set after set_object_channel_after_epoch
        # May contain multiple channels if instruments are split across channels (union at eval time)
        self.seq_object_channels = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        losses = self.forward(x)
        if isinstance(losses, dict):
            loss = losses['loss']
        else:
            loss = losses
            losses = dict(loss=loss)
        
        for loss_key, loss_value in losses.items():
            if 'loss' not in loss_key:
                continue
            self.accumulate_training_loss[loss_key] = self.accumulate_training_loss.get(loss_key, 0.) + loss_value.item()
            if (batch_idx + 1) % self.args.loss_log_interval == 0:
                self.log(f"train_{loss_key}", self.accumulate_training_loss[loss_key] /
                        self.args.loss_log_interval, sync_dist=True, reduce_fx="mean")
                self.accumulate_training_loss[loss_key] = 0.

        if torch.isnan(loss):
            raise Exception("loss is NaN")
        return loss

    @rank_zero_only
    def on_validation_start(self):
        self.on_test_start()

    @torch.no_grad()
    def _sliding_window_eval(self, x, window_size=384, stride=192):
        """Forward eval via sliding window so BN always sees window_size crops."""
        imgs = torch.stack(x['imgs'], dim=1)   # [B, im_num, C, H, W]
        B, im_num, C, H, W = imgs.shape

        mask_layer = self.args.model_kwargs['mask_layer']
        pred_accum = torch.zeros(B, mask_layer, H, W, device=imgs.device)
        count_map  = torch.zeros(1, 1, H, W, device=imgs.device)

        # Window positions — always include right/bottom edge for full coverage
        def _positions(length):
            if length <= window_size:
                return [0]
            pts = list(range(0, length - window_size, stride))
            if not pts or pts[-1] + window_size < length:
                pts.append(length - window_size)
            return sorted(set(pts))

        eval_on_ema = getattr(self.model, 'eval_on_ema', False)
        backbone = self.model.backbone2_ema if eval_on_ema else self.model.backbone2
        head     = self.model.decode_head2_ema if eval_on_ema else self.model.decode_head2

        for y in _positions(H):
            for xp in _positions(W):
                y2, x2 = min(y + window_size, H), min(xp + window_size, W)
                crop = imgs[:, :, :, y:y2, xp:x2]
                ch, cw = crop.shape[-2], crop.shape[-1]
                if ch < window_size or cw < window_size:
                    crop = F.pad(crop, (0, window_size - cw, 0, window_size - ch))

                img3 = crop.view(B * im_num, C, window_size, window_size)
                feat = self.model.extract_feat(img3, backbone)
                pred = self.model._decode_head_forward(feat, head)
                pred = pred.view(B, im_num, mask_layer, *pred.shape[-2:])
                pred = F.softmax(pred, dim=2)[:, 0]   # [B, C, fH, fW]
                pred = F.interpolate(pred, size=(window_size, window_size),
                                     mode='bilinear', align_corners=False)
                pred_accum[:, :, y:y2, xp:x2] += pred[:, :, :ch, :cw]
                count_map[:, :, y:y2, xp:x2] += 1

        return pred_accum / count_map.clamp(min=1)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        # We use max IoU channel in validation to be fast.
        # In test, we should only use one channel.
        self.test_step(batch, batch_idx)

    # Need to run it on non-zero rank to make sure we have val_miou key
    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs, name="val_miou", display_all=False)

    @rank_zero_only
    def on_test_start(self):
        self.iou_all_sequences = {}
        self.max_channel_freq = [0 for _ in range(self.args.model_kwargs["mask_layer"])]
        self.seq_channel_freq = {}  # {seq_name: np.array of shape [mask_layer]}

    @rank_zero_only
    def test_step(self, batch, batch_idx, always_use_max_iou_channel=False):
        x = batch
        if getattr(self.args, 'use_sliding_window', False):
            sw_size   = getattr(self.args, 'sliding_window_size', 384)
            sw_stride = getattr(self.args, 'sliding_window_stride', sw_size // 2)
            pred_masks_current_batch = self._sliding_window_eval(x, sw_size, sw_stride)
            # forward_eval was not called → no pending viz; disable eval_save for this step
            if hasattr(self.model, '_pending_eval_viz'):
                self.model._pending_eval_viz = None
        else:
            pred_masks_current_batch = self.forward(x)
        assert len(pred_masks_current_batch) == len(
            batch['ann']), f"{len(pred_masks_current_batch)} != {len(batch['ann'])}"
        pred_masks_current_batch_resize = utils.eval_utils._resize(
            pred_masks_current_batch, batch['ann'].shape[1:3])

        # Export mask when using sliding window (forward_eval is bypassed so we do it here)
        if getattr(self.args, 'use_sliding_window', False) and self.args.eval_export:
            if getattr(self.args, 'export_all_seg', False):
                pred_vis_list = [
                    pred_masks_current_batch_resize[:, i:i+1, :, :].repeat(1, 3, 1, 1)
                    for i in range(pred_masks_current_batch_resize.shape[1])
                ]
                self.model.export_all_seg(pred_vis_list, batch['paths'], batch['seq_ids'],
                                          batch['seq_names'], name='pred_seg', train_iter=0)
            else:
                ch = self.object_channel if self.object_channel is not None else 1
                export_tensor = (pred_masks_current_batch_resize[:, ch:ch+1] > self.args.eval_pos_th
                                 ).float().repeat(1, 3, 1, 1)
                self.model.export_seg(export_tensor, batch['paths'], batch['seq_ids'],
                                      batch['seq_names'], name='pred_seg', train_iter=0)

        # -1 means use hard max (turn masks into one hot)
        num_channels = pred_masks_current_batch_resize.shape[1]
        if self.args.eval_pos_th != -1:
            pred_masks_current_batch_resize = (
                pred_masks_current_batch_resize > self.args.eval_pos_th).long().cpu().numpy()
        else:
            pred_masks_current_batch_resize_max_idx = pred_masks_current_batch_resize.argmax(
                dim=1)
            pred_masks_current_batch_resize = F.one_hot(
                pred_masks_current_batch_resize_max_idx,
                num_classes=num_channels
            ).permute((0, 3, 1, 2)).long().cpu().numpy()

        ignore_locations = batch['ann'] == 128
        anns_np = (batch['ann'] > 128).long()
        anns_np[ignore_locations] = -1
        anns_np = anns_np.cpu().numpy()
        # Accumulate per-item GT and union masks for finalize_eval_save (locked mode)
        _eval_gt_list, _eval_union_list = [], []
        _in_oracle_mode = always_use_max_iou_channel or (self.object_channel is None)

        for batch_idx_item, (pred_mask_resize, ann, seq_name) in enumerate(
                zip(pred_masks_current_batch_resize, anns_np, batch['seq_names'])):
            # pred_mask_resize and ann are numpy array
            # index 1 selects the iou in the foreground
            if _in_oracle_mode:
                frame_ious = [
                    utils.iou(pred_mask_resize_item, ann, num_classes=2, ignore_index=-1)[1] for pred_mask_resize_item in pred_mask_resize]
                # Greedy union oracle: start with best single channel, greedily add
                # channels (in descending IoU order) if they improve the union IoU.
                # This correctly identifies all instrument channels even when one
                # instrument rarely "wins" head-to-head against another.
                best_ch = int(np.argmax(frame_ious))
                active_channels = [best_ch]
                best_iou = frame_ious[best_ch]
                remaining = sorted(
                    [k for k in range(num_channels) if k != best_ch],
                    key=lambda k: frame_ious[k], reverse=True)
                for ch in remaining:
                    candidate_mask = np.any(
                        np.stack([pred_mask_resize[c] for c in active_channels + [ch]]), axis=0
                    ).astype(pred_mask_resize.dtype)
                    candidate_iou = utils.iou(candidate_mask, ann, num_classes=2, ignore_index=-1)[1]
                    if candidate_iou > best_iou + 0.01:
                        active_channels.append(ch)
                        best_iou = candidate_iou
                seq_freq = self.seq_channel_freq.setdefault(seq_name, np.zeros(num_channels, dtype=int))
                for ch in active_channels:
                    self.max_channel_freq[ch] += 1
                    seq_freq[ch] += 1
                frame_iou = best_iou
            else:
                channels = self.seq_object_channels.get(seq_name, [self.object_channel])
                if len(channels) == 1:
                    union_mask = pred_mask_resize[channels[0]]
                else:
                    union_mask = np.any(
                        np.stack([pred_mask_resize[ch] for ch in channels]), axis=0
                    ).astype(pred_mask_resize.dtype)
                frame_iou = utils.iou(union_mask, ann, num_classes=2, ignore_index=-1)[1]
                # Collect per-item GT and union mask for batch-correct visualization
                if self.args.eval_save:
                    ann_vis = np.clip(ann, 0, 1).astype('float32')
                    if ann_vis.ndim == 3:
                        ann_vis = ann_vis[..., 0]
                    _eval_gt_list.append(ann_vis)
                    _eval_union_list.append(union_mask.astype('float32'))
            iou_current_sequence = self.iou_all_sequences.setdefault(
                seq_name, [])
            iou_current_sequence.append(frame_iou)

        # Call finalize_eval_save once after the loop with full-batch arrays
        if self.args.eval_save:
            if not getattr(self.args, 'use_sliding_window', False):
                if _in_oracle_mode:
                    self.model.finalize_eval_save([])
                elif _eval_gt_list:
                    self.model.finalize_eval_save([
                        np.stack(_eval_gt_list, axis=0),    # [B, H, W]
                        np.stack(_eval_union_list, axis=0), # [B, H, W]
                    ])
            elif not _in_oracle_mode and _eval_gt_list:
                # Sliding window: forward_eval bypassed → build 2-column visualization.
                # Left column : raw soft probability (no post-processing)
                # Right column: Guided Filter → binary threshold
                # Last row    : GT mask (left) | union mask (right)
                imgs = torch.stack(x['imgs'], dim=1)
                orig = imgs[:, 0]                                # [B, 3, H, W]
                B, C, H_img, W_img = pred_masks_current_batch.shape
                orig_d = ((orig + 2.0) / 4.0).clamp(0, 1)      # [0,1], full resolution
                gf_r   = getattr(self.args, 'sw_vis_gf_r',   16)
                gf_eps = getattr(self.args, 'sw_vis_gf_eps', 1e-2)
                pred_smooth = _guided_filter(orig_d, pred_masks_current_batch,
                                             r=gf_r, eps=gf_eps)
                th = self.args.eval_pos_th
                # Row 0: orig | orig
                rows = [torch.cat([orig_d, orig_d], dim=3)]
                # Rows 1…C: raw soft prob (left) | GF binary (right)
                for i in range(C):
                    raw = pred_masks_current_batch[:, i:i+1].clamp(0, 1).repeat(1, 3, 1, 1)
                    gf  = (pred_smooth[:, i:i+1] > th).float().repeat(1, 3, 1, 1)
                    rows.append(torch.cat([raw, gf], dim=3))
                # Row 6: GT | GT  (ground truth is the same for both columns)
                # Row 7: raw union (left) | GF union (right)
                def _np_to_row(masks_np):
                    t = torch.from_numpy(masks_np.astype('float32'))[:, None].repeat(1, 3, 1, 1)
                    return F.interpolate(t, (H_img, W_img), mode='nearest').to(orig.device)
                gt_t  = _np_to_row(np.stack(_eval_gt_list,    axis=0))
                uni_t = _np_to_row(np.stack(_eval_union_list,  axis=0))
                # GF union: re-derive from smoothed probs using same channel selection
                obj_chs = self.seq_object_channels.get(
                    x['seq_names'][0],
                    [self.object_channel] if self.object_channel is not None else [1])
                if len(obj_chs) == 1:
                    gf_uni = (pred_smooth[:, obj_chs[0]:obj_chs[0]+1] > th).float()
                else:
                    gf_uni = torch.stack(
                        [(pred_smooth[:, ch:ch+1] > th) for ch in obj_chs], dim=0
                    ).any(dim=0).float()
                rows.append(torch.cat([gt_t,  gt_t        ], dim=3))  # Row 6
                rows.append(torch.cat([uni_t, gf_uni.repeat(1, 3, 1, 1)], dim=3))  # Row 7
                tosave = torch.cat(rows, dim=2)                  # [B, 3, (C+2)*H, 2W]
                self.model.save_eval_visualizations(
                    tosave, x['paths'], x['seq_ids'], x['seq_names'])

    def test_epoch_end(self, outputs, name="test_miou", display_all=True):
        if (self.object_channel is None) and (not self.trainer.sanity_checking) and ((self.current_epoch >= getattr(self.args, "set_object_channel_after_epoch", 1) - 1) or self.trainer.testing):
            # Set object channel only once (if `always_use_max_iou_channel` is set, object channel will be ignored, otherwise this will always be used)
            if self.args.rank >= 0: # Distributed
                if self.args.rank == 0:
                    object_channel_current_rank = np.argmax(self.max_channel_freq)
                else:
                    object_channel_current_rank = 0

                object_channel = torch.tensor([object_channel_current_rank], device="cuda")
                dist.all_reduce(object_channel, op=dist.ReduceOp.SUM)
                object_channel = object_channel.item()
            else:
                object_channel = np.argmax(self.max_channel_freq)

            self.object_channel = object_channel
            self.args.object_channel = self.object_channel
            # Build per-sequence channel list: include channels with freq >= 30% of the best
            # channel's freq. This unions channels when instruments are split across channels.
            union_rel_th = getattr(self.args, "union_channel_rel_th", 0.3)
            self.seq_object_channels = {}
            for seq, freq in self.seq_channel_freq.items():
                freq_arr = np.array(freq)
                best_freq = freq_arr.max()
                if best_freq == 0:
                    self.seq_object_channels[seq] = [int(np.argmax(freq_arr))]
                else:
                    chs = [k for k in range(len(freq_arr))
                           if freq_arr[k] >= union_rel_th * best_freq]
                    self.seq_object_channels[seq] = chs
            if self.args.rank <= 0:
                print(f"Rank {self.args.rank}: Set object channel to {self.object_channel} (global fallback, Max channel distribution: {self.max_channel_freq})")
                for seq, chs in self.seq_object_channels.items():
                    freq_str = self.seq_channel_freq[seq].tolist()
                    print(f"  {seq}: channels {chs} (freq: {freq_str})")
            else:
                print(f"Rank {self.args.rank}: Set object channel to {self.object_channel}")
        
        if self.args.rank > 0:
            # Otherwise model checkpoint will not work in validation
            # Sync values to make model checkpoint correctly.
            self.log(name, 0., sync_dist=True, reduce_fx="sum")
            self.log(name + "_frame_avg", 0., sync_dist=True, reduce_fx="sum")
            return

        miou_each_sequence = {}
        iou_sum = 0.
        iou_num_frames = 0.
        for seq_name, miou_current_sequence in self.iou_all_sequences.items():
            miou = np.nanmean(miou_current_sequence).astype(np.float32)
            miou_each_sequence[seq_name] = miou

            iou_sum += np.nansum(miou_current_sequence).astype(np.float32)
            iou_num_frames += len(miou_current_sequence)

            if display_all:
                logger.info(f"{name}_{seq_name}: {miou * 100.:.2f}")
            # This is only computed and logged on rank 0, so do not sync.
            self.log(f'{name}_{seq_name}', miou, sync_dist=False)

        # Use nanmean: sequences with all-zero GT + all-zero pred produce IoU=0/0=NaN, skip them
        mean_miou_all_sequences = np.nanmean(list(miou_each_sequence.values())).astype(np.float32)

        logger.info(f"{name}: {mean_miou_all_sequences * 100.:.2f}")
        self.log(name, mean_miou_all_sequences, sync_dist=True, reduce_fx="sum")

        miou_frame_avg = iou_sum / iou_num_frames
        logger.info(f"{name}_frame_avg: {miou_frame_avg * 100.:.2f}")
        self.log(name + "_frame_avg", miou_frame_avg, sync_dist=True, reduce_fx="sum")

    def get_lr(self, epoch, power, base_lr, min_lr):
        coeff = (1 - epoch / self.args.epochs) ** power
        lr = (base_lr - min_lr) * coeff + min_lr
        return lr / base_lr

    def configure_optimizers(self):
        if hasattr(self.model, 'get_param_groups'):
            param_groups = self.model.get_param_groups(self.args.learning_rate)
            logger.info(f"Using model-defined param groups ({len(param_groups)} groups)")
        else:
            params = list(self.model.parameters())
            params_require_grad = [param for param in params if param.requires_grad]
            logger.info(f"Number of param tensors: {len(params)}. Number of param tensors (require grad): {len(params_require_grad)}.")
            param_groups = [{'params': params_require_grad, 'lr': self.args.learning_rate}]
        optimizer = torch.optim.__dict__[self.args.optimizer](
            param_groups,
            lr=self.args.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        def lr_lambda(epoch): return self.get_lr(
            epoch, base_lr=self.args.learning_rate, **self.args.lr_scheduler_kwargs)
        return [optimizer], [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)]

    def on_train_epoch_start(self):
        schedule = getattr(self.args, 'progressive_unfreeze', None)
        if schedule:
            epoch = self.current_epoch
            current_stage = None
            for stage in sorted(schedule, key=lambda s: s['epoch']):
                if epoch >= stage['epoch']:
                    current_stage = stage
            if current_stage is not None:
                _apply_progressive_unfreeze(self.model, current_stage, epoch)

        current_lr = self.optimizers(False).param_groups[0]['lr']

        logger.info(f"LR: {current_lr}")

        self.log(
            "lr",
            current_lr,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def train_dataloader(self):
        train_dataset = self.dataset_cls(
            self.args.data_path, training=True,
            transform=dataset.get_transform(self.args, training=True),
            **self.args.dataset_kwargs, **self.args.train_dataset_kwargs)
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True) if self.args.multi_gpu else None
        
        if getattr(self.args, 'force_no_shuffle', False):
            assert not self.args.multi_gpu, "force_no_shuffle is for visualization with one GPU"
            shuffle = False
        else:
            shuffle = sampler is None
        
        resolution_crop_configs = getattr(self.args, 'train_transform_kwargs', {}).get('resolution_crop_configs')
        if resolution_crop_configs and not self.args.multi_gpu:
            # Group same-resolution samples into each batch; interleave across batches.
            # Avoids zero-padding when ResolutionAwareCrop produces different crop sizes.
            batch_sampler = dataset.ResolutionGroupedBatchSampler(
                train_dataset,
                batch_size=self.hparams.batch_size,
                resolution_configs=resolution_crop_configs,
            )
            return torch.utils.data.DataLoader(
                train_dataset,
                num_workers=self.args.workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
            )
        collate_fn = dataset.pad_collate if resolution_crop_configs else None
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

    def val_dataloader(self, subsample_frame_interval=5):
        # subsample_frames for fast evaluation
        test_data_path = self.args.test_data_path if getattr(self.args, 'test_data_path', None) else self.args.data_path
        val_dataset = self.dataset_cls(
            test_data_path, training=False,
            transform=dataset.get_transform(self.args, training=False),
            subsample_frame_interval=subsample_frame_interval,
            **self.args.dataset_kwargs, **self.args.test_dataset_kwargs)
        
        val_batch_size = getattr(self.hparams, 'val_batch_size', self.hparams.batch_size)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            sampler=torch.utils.data.SequentialSampler(val_dataset),
            shuffle=False
        )

    def test_dataloader(self):
        test_data_path = self.args.test_data_path if getattr(self.args, 'test_data_path', None) else self.args.data_path
        test_dataset = self.dataset_cls(
            test_data_path, training=False,
            transform=dataset.get_transform(self.args, training=False),
            **self.args.dataset_kwargs, **self.args.test_dataset_kwargs)
        val_batch_size = getattr(self.hparams, 'val_batch_size', self.hparams.batch_size)
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            sampler=torch.utils.data.SequentialSampler(test_dataset),
            shuffle=False
        )


exp_name = None


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items["v_num"] = exp_name
        return items


def main():
    global logger, exp_name

    parser = argparse.ArgumentParser(description='Train segmentation.')
    parser.add_argument('config', metavar='C', type=str, nargs='?',
                        help='path to config', default='configs/rcf/rcf_stage1.yaml')
    parser.add_argument('--test', help='test only',
                        default=False, action="store_true")
    parser.add_argument('--test-override-pretrained', help='override pretrained model and checkpoints directory at test',
                        default=None, type=str)
    parser.add_argument('--test-override-object-channel', help='override object channel at test',
                        default=None, type=int)
    parser.add_argument('--no-test', help='no test at the end of training',
                        default=False, action="store_true")
    parser.add_argument('--resume', help='resume training from checkpoint (restores model weights, optimizer, lr scheduler, and epoch)',
                        default=None, type=str)
    # From detectron2
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    cli_args = parser.parse_args()
    config_path = cli_args.config
    test = cli_args.test
    no_test = cli_args.no_test

    rank = int(os.environ.get("LOCAL_RANK", "-1"))

    if rank <= 0:
        utils.set_loglevel(debug=True)
    else:
        utils.set_loglevel(debug=False)

    logger.info(f"Loading config from {config_path}")

    args = utils.load_args(config_path, cli_opts=cli_args.opts)
    logger.info(str(args))

    # Use config file name as the experiment name
    # exp_name = config_path.split(
    #     "/")[-1][:-5] + "_" + datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Use checkpoints_dir as the experiment name because we may override checkpoints_dir with CLI options
    exp_name = args.checkpoints_dir.split(
        "/")[-1] + "_" + datetime.now().strftime("%y%m%d_%H%M%S")
    if args.disable_wandb:
        wandb_logger = pl.loggers.TensorBoardLogger(  
            save_dir=args.checkpoints_dir,
            name='tensorboard_logs',
            version=None,
            default_hp_metric=True
        )
    else:
        wandb_logger = pl.loggers.WandbLogger(      
            project="RCF", mode=None, name=exp_name, settings=wandb.Settings(start_method="thread"))
        # save_on_train_epoch_end should be set to True if there is no validation set
    # even on rank non-zero we need to have the monitor key (saving is ignored in Stratrgy class so other code that requires the monitor key will run on rank non-zero)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir, save_on_train_epoch_end=False, every_n_epochs=1, monitor='val_miou_frame_avg',
        save_top_k=5, save_last=True, mode='max', auto_insert_metric_name=True)

    trainer_cfg = {
        "logger": wandb_logger,
        # override_max_epochs is for early stopping without influencing the learning rate scheduler
        "max_epochs": getattr(args, "override_max_epochs", args.epochs),
        "accelerator": "gpu",
        "replace_sampler_ddp": False,
        "callbacks": [CustomProgressBar(), checkpoint_callback],
        **args.trainer_kwargs
    }

    args.config_path = config_path
    args.test = test
    args.rank = rank
    args.resume_from_checkpoint = cli_args.resume
    args.multi_gpu = rank > -1

    if args.multi_gpu:
        assert not test, "Testing with multi-GPUs is unsupported."
        trainer_cfg.update(dict(strategy="ddp_find_unused_parameters_false"))
    else:
        trainer_cfg.update(dict(devices=1))

    if test:
        if cli_args.test_override_pretrained is not None:
            args.pretrained_model = cli_args.test_override_pretrained
            args.checkpoints_dir = os.path.dirname(args.pretrained_model)
            logger.info(f"Overriding pretrained_model to {args.pretrained_model}")
        if cli_args.test_override_object_channel is not None:
            args.object_channel = cli_args.test_override_object_channel
            logger.info(f"Overriding object channel to {args.object_channel}")

    zero_ann = args.test_dataset_kwargs.get('zero_ann', False)

    trainer = pl.Trainer(**trainer_cfg, default_root_dir=args.checkpoints_dir)
    model = Model(args, trainer)

    logger.info(f"{model}")

    def _two_pass_test(ckpt_path=None):
        """Pass 1: subsampled oracle via validate() to lock per-seq channels.
           Pass 2: full test() with locked channels to compute real mIoU."""
        # set_object_channel_after_epoch=0 ensures channels are locked even at
        # epoch 0 (trainer.validate() runs outside of training, current_epoch=0)
        args.set_object_channel_after_epoch = 0
        # Pass 1: channel selection only — disable eval_save to avoid writing
        # oracle images (no GT rows) that would clutter the output directory.
        _eval_save = args.eval_save
        args.eval_save = False
        logger.info("Pass 1/2: oracle channel selection (subsampled) ...")
        trainer.validate(model=model, ckpt_path=ckpt_path)
        # Pass 2: locked channel eval — restore eval_save, saves GT + union rows.
        args.eval_save = _eval_save
        logger.info("Pass 2/2: locked channel mIoU evaluation (full dataset) ...")
        trainer.test(model=model, ckpt_path=ckpt_path)

    if not test:
        trainer.fit(model=model, ckpt_path=args.resume_from_checkpoint)
        if not no_test:
            args.saved_eval_dir_name = 'saved_eval_test'
            args.eval_pos_th = -1
            if zero_ann or model.object_channel is not None:
                trainer.test(model=model, ckpt_path='best')
            else:
                _two_pass_test(ckpt_path='best')
    else:
        if zero_ann or model.object_channel is not None:
            trainer.test(model=model)
        else:
            _two_pass_test()


if __name__ == "__main__":
    main()
