"""
main_triplet.py — entry point for RCFTripletModel training (3 consecutive
frames per sample, genuine supervision for all 3 pairwise flow
relationships -- see models/rcf_triplet_model.py).

Reuses everything from main_tissue.py (model/decode-head registration,
val_dataloader / validation_step / checkpoint callback -- eval is still
plain single-frame VideoDataset via val_dataset_list, completely
unmodified) except train_dataloader, which is overridden here to use
TripletVideoDataset + build_triplet_transform (dataset/triplet_data.py)
instead of the standard VideoDataset + get_transform pipeline (which only
supports one flow pair per sample).

Usage:
  python main_triplet.py configs/instrument/rcf_cmc_grasp0_tissue_ft_v126.yaml
"""
import torch

import main_tissue  # runs all registration side effects (models, decode heads, TissueModel/checkpoint patch)
from main_tissue import TissueModel
import main as _main_module
from main import main

from dataset.triplet_data import TripletVideoDataset, build_triplet_transform

# Register RCFTripletModel + RCFTripletJointMaskModel (model_cls resolution
# uses models.__dict__[args.model_cls], same mechanism main_tissue.py uses
# for RCFSoftTissueModel etc.)
import models as _models_pkg
from models.rcf_triplet_model import RCFTripletModel
from models.rcf_triplet_joint_mask_model import RCFTripletJointMaskModel
from models.rcf_triplet_joint_mask_v2_model import RCFTripletJointMaskV2Model
_models_pkg.RCFTripletModel            = RCFTripletModel             # type: ignore[attr-defined]
_models_pkg.RCFTripletJointMaskModel   = RCFTripletJointMaskModel    # type: ignore[attr-defined]
_models_pkg.RCFTripletJointMaskV2Model = RCFTripletJointMaskV2Model  # type: ignore[attr-defined]


class TripletModel(TissueModel):
    def train_dataloader(self):
        transform = build_triplet_transform(
            **getattr(self.args, 'train_transform_kwargs', {})
        )
        train_dataset = TripletVideoDataset(
            self.args.data_path, transform=transform, training=True,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            shuffle=True,
        )


# Overrides main_tissue.py's own Model patch (which runs at import time above) --
# this assignment executes after, so it wins.
_main_module.Model = TripletModel

if __name__ == "__main__":
    main()
