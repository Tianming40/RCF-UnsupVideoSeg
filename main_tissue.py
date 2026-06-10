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

# ── 1. Register V2 flow head ──────────────────────────────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

# ── 2. Register RCFDinoModel + RCFSoftTissueModel ─────────────────────────────
import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
from models.rcf_soft_tissue_model import RCFSoftTissueModel

_models_pkg.RCFDinoModel       = RCFDinoModel        # type: ignore[attr-defined]
_models_pkg.RCFSoftTissueModel = RCFSoftTissueModel  # type: ignore[attr-defined]
_models_pkg.RCFTissueModel     = RCFSoftTissueModel  # type: ignore[attr-defined]

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

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        # on_epoch=True → PL aggregates mean across all steps in the epoch;
        # ModelCheckpoint can then find 'epoch_train_loss' at epoch end.
        self.log('epoch_train_loss', loss.detach(),
                 on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True, reduce_fx='mean')
        return loss


# ── 4. Replace ModelCheckpoint: save top-3 by min train loss ─────────────────
from pytorch_lightning.callbacks import ModelCheckpoint as _OrigMC


class _TrainLossCheckpoint(_OrigMC):
    """ModelCheckpoint that monitors epoch_train_loss (min) instead of val_miou."""

    def __init__(self, *args, **kwargs):
        kwargs['monitor']              = 'epoch_train_loss'
        kwargs['mode']                 = 'min'
        kwargs['save_on_train_epoch_end'] = True
        kwargs['save_top_k']           = 3
        super().__init__(*args, **kwargs)


# ── 5. Patch into main module namespace ───────────────────────────────────────
_main_module.Model           = TissueModel        # main() does Model(args, trainer)
_main_module.ModelCheckpoint = _TrainLossCheckpoint  # main() does ModelCheckpoint(...)

# ── 6. Call original main ─────────────────────────────────────────────────────
from main import main

if __name__ == "__main__":
    main()
