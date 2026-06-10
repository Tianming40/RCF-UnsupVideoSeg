"""
main_grasp0.py — entry point for grasp0 soft-tissue fine-tuning.

Steps:
  1. Register FlowAggregationHeadWithResidualV2 into rcf_model namespace.
  2. Register RCFDinoModel (base class of RCFSoftTissueModel).
  3. Register RCFSoftTissueModel into models namespace.
  4. Subclass main.py's Model to log per-epoch train loss so that
     ModelCheckpoint can monitor it (val mIoU is meaningless on grasp0).
  5. Patch into main module namespace and call main().

Usage:
  python main_grasp0.py configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml

  # Override any key:
  python main_grasp0.py configs/instrument/rcf_cmc_grasp0_tissue_ft.yaml \\
      --opts checkpoints_dir saved/my_run allow_overwriting_checkpoints_dir True
"""

import logging

# ── 1. Register V2 flow head ──────────────────────────────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

# ── 2. Register RCFDinoModel ──────────────────────────────────────────────────
import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
_models_pkg.RCFDinoModel = RCFDinoModel              # type: ignore[attr-defined]

# ── 3. Register RCFSoftTissueModel ────────────────────────────────────────────
from models.rcf_soft_tissue_model import RCFSoftTissueModel
_models_pkg.RCFSoftTissueModel = RCFSoftTissueModel  # type: ignore[attr-defined]

# ── 4. Track per-epoch train loss ─────────────────────────────────────────────
import main as _main_module
from main import Model as _BaseModel

logger = logging.getLogger(__name__)


class Grasp0Model(_BaseModel):
    """Logs epoch_train_loss so ModelCheckpoint can monitor it."""

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log('epoch_train_loss', loss.detach(),
                 on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True, reduce_fx='mean')
        return loss


# ── 5. Replace ModelCheckpoint ────────────────────────────────────────────────
from pytorch_lightning.callbacks import ModelCheckpoint as _OrigMC


class _TrainLossCheckpoint(_OrigMC):
    def __init__(self, *args, **kwargs):
        kwargs['monitor']                 = 'epoch_train_loss'
        kwargs['mode']                    = 'min'
        kwargs['save_on_train_epoch_end'] = True
        kwargs['save_top_k']              = 3
        super().__init__(*args, **kwargs)


_main_module.Model           = Grasp0Model
_main_module.ModelCheckpoint = _TrainLossCheckpoint

from main import main

if __name__ == "__main__":
    main()
