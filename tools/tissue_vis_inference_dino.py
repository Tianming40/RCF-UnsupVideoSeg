"""
tissue_vis_inference_dino.py — tissue_vis_inference wrapper for DINO Phase-1 checkpoints.



Usage:
  python tools/tissue_vis_inference_dino.py \\
    --config  configs/instrument/rcf_cmc_dino_phase1.yaml \\
    --ckpt    saved/cmc_dino_phase1_260605_143205/epoch=7-step=1800.ckpt \\
    --output  analysis/dino_val_vis \\
    --split   fold1_val.txt \\
    --tissue_channel 0 \\
    --instrument_channels 1
"""

import sys
import copy
import argparse as _ap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Register V2 flow head ────────────────────────────────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

# ── Import vis module (runs its own module-level registrations) ───────────────
import tools.tissue_vis_inference as _vis_mod
from models.rcf_model import RCFModel

# ── DINO-only and tissue-only kwargs that RCFModel doesn't accept ─────────────
_STRIP = frozenset({
    'w_dino', 'dino_checkpoint', 'dino_arch', 'dino_patch_size', 'dino_input_size',
    'instrument_channels', 'tissue_channel', 'grasping_channel', 'bg_channels',
    'w_rigid', 'w_grasp_conv', 'w_deform', 'w_align', 'w_motion',
    'motion_margin', 'deform_margin', 'min_grasp_frac',
})


def _build_base_model(cfg: dict, output_dir: str = "/tmp") -> RCFModel:
    """Drop-in replacement for build_tissue_model using plain RCFModel."""
    kwargs = copy.deepcopy(cfg["model_kwargs"])
    for k in _STRIP:
        kwargs.pop(k, None)
    kwargs["allow_mask_resize"] = True
    fake_args = _ap.Namespace(
        checkpoints_dir=output_dir,
        eval_save=False, eval_export=False, export_all_seg=False,
        eval_pos_th=cfg.get("eval_pos_th", 0.35),
        object_channel=None, log_interval=9999,
    )
    return RCFModel(args=fake_args, **kwargs)


# Patch before main() is called
_vis_mod.build_tissue_model = _build_base_model

# ── Optional --data_path override (intercept before tissue_vis main parses) ──
# Remove it from sys.argv so tissue_vis_inference.main() doesn't choke on it
import sys as _sys

_data_path_override = None
_zero_ann_override  = False
_new_argv = [_sys.argv[0]]
_i = 1
while _i < len(_sys.argv):
    if _sys.argv[_i] == "--data_path" and _i + 1 < len(_sys.argv):
        _data_path_override = _sys.argv[_i + 1]
        _i += 2
    elif _sys.argv[_i] == "--zero_ann":
        _zero_ann_override = True
        _i += 1
    else:
        _new_argv.append(_sys.argv[_i])
        _i += 1
_sys.argv = _new_argv

if _data_path_override is not None:
    _orig_load_config = _vis_mod.load_config

    def _load_config_with_override(path):
        cfg = _orig_load_config(path)
        cfg["data_path"] = _data_path_override
        if _zero_ann_override:
            cfg.setdefault("test_dataset_kwargs", {})["zero_ann"] = True
        return cfg

    _vis_mod.load_config = _load_config_with_override

if __name__ == "__main__":
    _vis_mod.main()
