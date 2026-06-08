"""
channel_role_analysis_dino.py — wrapper to analyse RCFDinoModel checkpoints.

maa_union_inference.build_model hardcodes RCFModel() and rejects DINO-specific
kwargs (w_dino, dino_checkpoint, …).  This wrapper strips those kwargs before
build_model is called so a plain RCFModel is built for inference; the DINO
weights in the checkpoint are simply ignored via strict=False.

All CLI flags are identical to channel_role_analysis.py.

Usage:
  python tools/channel_role_analysis_dino.py \\
    --config  configs/instrument/rcf_cmc_dino_phase1.yaml \\
    --ckpt    saved/cmc_dino_phase1_260605_143205/epoch=7-step=1800.ckpt \\
    --output  analysis/dino_phase1_channels \\
    --split   trainval.txt \\
    --flow_suffix _NewCT \\
    --n_samples 60
"""

import sys
import copy as _copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Register V2 flow head (must happen before any rcf_model import) ──────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

# ── Import the analysis module (runs its module-level imports) ───────────────
import tools.channel_role_analysis as _analysis_mod

# ── Patch build_model to strip DINO-only kwargs before RCFModel() call ───────
# channel_role_analysis.main() resolves `build_model` from the module's global
# dict at call time, so replacing the module attribute here takes effect.

_DINO_ONLY_KWARGS = frozenset({
    'w_dino', 'dino_checkpoint', 'dino_arch', 'dino_patch_size', 'dino_input_size',
})
_orig_build_model = _analysis_mod.build_model


def _build_model_strip_dino(cfg, output_dir):
    cfg_clean = _copy.deepcopy(cfg)
    mk = cfg_clean.get('model_kwargs', {})
    for k in _DINO_ONLY_KWARGS:
        mk.pop(k, None)
    return _orig_build_model(cfg_clean, output_dir)


_analysis_mod.build_model = _build_model_strip_dino

# ── Delegate to original main() ───────────────────────────────────────────────
if __name__ == "__main__":
    _analysis_mod.main()
