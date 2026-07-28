"""
main_tissue.py — entry point for tissue-role-aware fine-tuning.

Steps (in order):
  1. Register FlowAggregationHeadWithResidualV2 into rcf_model namespace.
  2. Register RCFDinoModel + RCFSoftTissueModel into models namespace.
     Also registers RCFTissueModel as an alias so old configs still work.
  3. Subclass main.py's Model (PL LightningModule) to track per-epoch train
     loss and log 'epoch_train_loss' — used by checkpoint callback.
  4. Replace ModelCheckpoint: monitor='epoch_train_loss', mode='min',
     save_on_train_epoch_end=True, save_top_k=3.
     Val_miou on data_medical is not meaningful for CMC tissue training.
  5. Patch both replacements into main module namespace so main() picks them up.
  6. Call main().

Usage:
  python main_tissue.py configs/instrument/rcf_cmc_tissue_v2.yaml
  python main_tissue.py configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml
"""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
import dataset as _ds_mod
from metrics import empty_prf_bucket, mean_prf as _mean_prf

# ── 1. Register V2 flow head + MultiScaleSegHeadJoint4 ─────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
from models.multi_scale_seg_head_joint4 import MultiScaleSegHeadJoint4
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2
_rcf_mod.MultiScaleSegHeadJoint4 = MultiScaleSegHeadJoint4  # decode_head2.type: MultiScaleSegHeadJoint4 (v123+)

# ── 2. Register RCFDinoModel + RCFSoftTissueModel ─────────────────────────────
import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
from models.rcf_soft_tissue_model import RCFSoftTissueModel
from models.rcf_joint_mask_model import RCFJointMaskSoftTissueModel
from models.rcf_joint_mask_v2_model import RCFJointMaskV2SoftTissueModel
from models.rcf_joint_mask_v3_model import RCFJointMaskV3SoftTissueModel
from models.rcf_joint_mask_v4_model import RCFJointMaskV4SoftTissueModel
from models.rcf_joint_mask_v5_model import RCFJointMaskV5SoftTissueModel
from models.rcf_joint_mask_v6_model import RCFJointMaskV6SoftTissueModel
from models.rcf_triplet_model import RCFTripletModel

_models_pkg.RCFDinoModel              = RCFDinoModel               # type: ignore[attr-defined]
_models_pkg.RCFSoftTissueModel        = RCFSoftTissueModel         # type: ignore[attr-defined]
_models_pkg.RCFTissueModel            = RCFSoftTissueModel         # type: ignore[attr-defined]
_models_pkg.RCFJointMaskV2SoftTissueModel = RCFJointMaskV2SoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFJointMaskV3SoftTissueModel = RCFJointMaskV3SoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFJointMaskV4SoftTissueModel = RCFJointMaskV4SoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFJointMaskV5SoftTissueModel = RCFJointMaskV5SoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFJointMaskV6SoftTissueModel = RCFJointMaskV6SoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFJointMaskSoftTissueModel = RCFJointMaskSoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFTripletModel            = RCFTripletModel           # type: ignore[attr-defined]

# ── 3. Subclass Model to track epoch-level train loss ─────────────────────────
import main as _main_module
from main import Model as _BaseModel

logger = logging.getLogger(__name__)


class TissueModel(_BaseModel):
    """
    Extends the base PL Model with per-epoch train-loss logging.
    Logs 'epoch_train_loss' at the end of each training epoch so that
    ModelCheckpoint can monitor it and save the best checkpoint by
    minimum training loss.
    """

    def __init__(self, args, trainer):
        super().__init__(args, trainer)

    def on_train_start(self) -> None:
        super().on_train_start()
        if getattr(self.model, 'reset_full_decode_head', False):
            self.model._reset_full_decode_head2()
        elif getattr(self.model, 'reset_non_instrument_heads', False):
            self.model._reset_non_instrument_mask_heads()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        warmup = getattr(self.model, 'distill_warmup_epochs', 0)
        if warmup > 0 and self.current_epoch >= warmup:
            if hasattr(self.model, 'set_distill_cool'):
                self.model.set_distill_cool()
        tv_start = getattr(self.model, 'flow_tv_start_epoch', 0)
        if tv_start > 0 and self.current_epoch >= tv_start:
            self.model._flow_tv_active = True
        ce_start = getattr(self.model, 'flow_cluster_ce_start_epoch', 0)
        if ce_start > 0 and self.current_epoch >= ce_start:
            self.model._flow_cluster_ce_active = True

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        # on_epoch=True → PL aggregates mean across all steps in the epoch;
        # ModelCheckpoint can then find 'epoch_train_loss' at epoch end.
        self.log('epoch_train_loss', loss.detach(),
                 on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True, reduce_fx='mean')
        return loss

    # ── 双 val：instrument + tissue，各自 oracle，mIoU 相加 ────────────────────
    def _make_val_loader(self, data_path, split, frame_num=1, gap_options=None):
        # gap_options (e.g. [1]) forces a deterministic gap for frame_num>1 eval
        # splits -- needed once frame_num>=2: VideoDataset.__getitem__ draws a
        # random gap UNCONDITIONALLY (training or not); without pinning it,
        # eval would sometimes select frames at the wrong spacing (e.g. 0,2,4
        # instead of 0,1,2), inconsistent with what training actually used.
        extra = {}
        if gap_options is not None:
            extra['gap_options'] = gap_options
            extra['gap_probabilities'] = [1.0 / len(gap_options)] * len(gap_options)
            extra['gap_flow_suffixes'] = ['_NewCT'] * len(gap_options)  # unused (load_flow=False), only needed to satisfy VideoDataset's length assertion

        ds = self.dataset_cls(
            data_path, training=False,
            transform=_ds_mod.get_transform(self.args, training=False),
            subsample_frame_interval=None,
            frame_num=frame_num, load_flow=False, split=split, zero_ann=False,
            **extra, **self.args.dataset_kwargs,
        )

        if frame_num > 1:
            # VideoDataset's __len__ visits sum(seq_lens) raw indices per split
            # (e.g. 3 per case for a 3-frame split line), but only the FIRST raw
            # index of each case is guaranteed to resolve to frame_ind=0 (the
            # boundary-shift formula `frame_ind -= (frame_num-1)` only
            # self-corrects exactly for frame_num==2; for frame_num==3 with a
            # tight 3-frame split line it produces frame_ind=-1 for the middle
            # raw index -- verified this session, silently reads the wrong
            # annotation file). Restrict to each case's first raw index
            # (seq_len_cumsum's own start offsets) -- always safe regardless of
            # frame_num, and as a side effect also removes the harmless-but-
            # wasteful ~2x duplicate-visit overhead frame_num=2 splits had.
            import torch as _torch
            first_indices = ds.seq_len_cumsum[:-1].tolist()
            ds = _torch.utils.data.Subset(ds, first_indices)

        return DataLoader(ds, batch_size=1, num_workers=self.args.workers,
                          pin_memory=True, sampler=SequentialSampler(ds), shuffle=False)

    def test_dataloader(self):
        # Same frame_num>1 fix as _make_val_loader above, applied to the
        # POST-TRAINING final test pass (main.py's own test_dataloader,
        # used by trainer.test()/_two_pass_test()) -- this path is separate
        # from val_dataset_list/_make_val_loader and would otherwise hit the
        # exact same frame_ind=-1 bug for frame_num>=3 test_dataset_kwargs
        # (e.g. v127's val_triplet.txt).
        test_kw = dict(self.args.test_dataset_kwargs)
        frame_num = test_kw.get('frame_num', 1)
        test_data_path = self.args.test_data_path if getattr(self.args, 'test_data_path', None) else self.args.data_path
        ds = self.dataset_cls(
            test_data_path, training=False,
            transform=_ds_mod.get_transform(self.args, training=False),
            **self.args.dataset_kwargs, **test_kw,
        )
        if frame_num > 1:
            import torch as _torch
            first_indices = ds.seq_len_cumsum[:-1].tolist()
            ds = _torch.utils.data.Subset(ds, first_indices)
        val_batch_size = getattr(self.hparams, 'val_batch_size', self.hparams.batch_size)
        return DataLoader(ds, batch_size=val_batch_size, num_workers=self.args.workers,
                          pin_memory=True, sampler=SequentialSampler(ds), shuffle=False)

    def val_dataloader(self):
        # config 里 val_dataset_list: [{data_path, split, name, frame_num?}, ...]；没配则退回单 val
        # frame_num 默认 1（向后兼容，所有既有 config 行为不变）；RCFJointMaskSoftTissueModel
        # 等需要 eval 时也拿到配对帧的模型，可以在对应 entry 里显式加 frame_num: 2
        # （配合 dataset/data.py 里放宽过的 eval frame_num>1 支持，以及一个 2 帧/行的 split 文件）。
        vlist = getattr(self.args, 'val_dataset_list', None)
        if not vlist:
            return super().val_dataloader()
        self._val_names = [v.get('name', f'val{i}') for i, v in enumerate(vlist)]
        return [self._make_val_loader(v['data_path'], v['split'], v.get('frame_num', 1), v.get('gap_options'))
               for v in vlist]

    def _fresh_val_bucket(self):
        n = self.args.model_kwargs["mask_layer"]
        return {'iou': {}, 'freq': [0] * n, 'seqfreq': {}, 'prf': empty_prf_bucket()}

    def on_validation_epoch_start(self):
        self._val_buckets = {}
        self._val_saved_eval = getattr(self.args, 'eval_save', False)
        self.args.eval_save = False        # val 不存图，省时

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if not getattr(self.args, 'val_dataset_list', None):
            return super().validation_step(batch, batch_idx)
        if dataloader_idx not in self._val_buckets:
            self._val_buckets[dataloader_idx] = self._fresh_val_bucket()
        b = self._val_buckets[dataloader_idx]
        # 把累积状态指向当前 dataloader 的 bucket（PL 串行跑 dl0 全部再 dl1，安全）
        self.iou_all_sequences = b['iou']
        self.max_channel_freq  = b['freq']
        self.seq_channel_freq  = b['seqfreq']
        self._prf_all_sequences = b['prf']
        self.object_channel    = None      # per-frame greedy union oracle
        # tissue val：贪心 union 排除 instrument channel(s)，避免器械并进 tissue；
        # instrument val：不排除（instrument 本就在 ch1）
        nm = self._val_names[dataloader_idx] if dataloader_idx < len(self._val_names) else None
        if nm is not None and 'tissue' in nm:
            self.args.oracle_exclude_channels = list(self.args.model_kwargs.get('instrument_channels', [1]))
        else:
            self.args.oracle_exclude_channels = []
        self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        if not getattr(self.args, 'val_dataset_list', None):
            return super().validation_epoch_end(outputs)
        self.args.eval_save = self._val_saved_eval
        names = getattr(self, '_val_names', None) or \
                [f'val{i}' for i in range(len(self._val_buckets))]
        total = 0.0
        for idx in sorted(self._val_buckets.keys()):
            b = self._val_buckets[idx]
            iou_sum, nfr = 0.0, 0
            for seq, lst in b['iou'].items():
                iou_sum += np.nansum(lst); nfr += len(lst)
            avg = float(iou_sum / max(nfr, 1))
            nm = names[idx] if idx < len(names) else f'val{idx}'
            # channel 分布：每个 channel 在 greedy union 里被选中的累计帧数
            ch_dist = [int(x) for x in b['freq']]
            p, r, f1 = _mean_prf(b['prf']) if b['prf'] else (0., 0., 0.)
            logger.info(f'val_miou_{nm}: {avg * 100.:.2f}  ({nfr} frames)  '
                        f'precision={p * 100.:.2f}  recall={r * 100.:.2f}  f1={f1 * 100.:.2f}  '
                        f'greedy_channel_dist[ch0..ch{len(ch_dist)-1}]={ch_dist}')
            self.log(f'val_miou_{nm}', avg, sync_dist=True, prog_bar=True)
            total += avg
        logger.info(f'val_miou_sum: {total * 100.:.2f}')
        self.log('val_miou_sum', total, sync_dist=True, prog_bar=True)


# ── 4. Replace ModelCheckpoint: save top-3 by min train loss ─────────────────
from pytorch_lightning.callbacks import ModelCheckpoint as _OrigMC


class _TrainLossCheckpoint(_OrigMC):
    """ModelCheckpoint monitoring val_miou_sum (instrument + tissue), keep top-3."""

    def __init__(self, *args, **kwargs):
        kwargs['monitor']                 = 'val_miou_sum'
        kwargs['mode']                    = 'max'
        kwargs['save_on_train_epoch_end'] = False   # 在 val 之后存
        kwargs['save_top_k']              = 3
        kwargs['save_last']               = True
        super().__init__(*args, **kwargs)


# ── 5. Patch into main module namespace ───────────────────────────────────────
_main_module.Model           = TissueModel        # main() does Model(args, trainer)
_main_module.ModelCheckpoint = _TrainLossCheckpoint  # main() does ModelCheckpoint(...)

# ── 6. Call original main ─────────────────────────────────────────────────────
from main import main

if __name__ == "__main__":
    main()
