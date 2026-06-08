"""
main_dino.py — entry point for DINO-guided Phase 1 training (Method B).

Steps (in order):
  1. Register FlowAggregationHeadWithResidualV2 into rcf_model's namespace
     (same pattern as main_v2.py).
  2. Register RCFDinoModel into models namespace so main() can instantiate it
     via model_cls: RCFDinoModel in the config.
  3. Delegate to the original main() function — all CLI flags are identical.

Usage:
  python main_dino.py configs/instrument/rcf_cmc_dino_phase1.yaml

  # Override any config key via --opts:
  python main_dino.py configs/instrument/rcf_cmc_dino_phase1.yaml \\
      --opts checkpoints_dir saved/my_run allow_overwriting_checkpoints_dir True
"""

# ── 1. Register V2 flow head into rcf_model namespace ─────────────────────────
import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2

_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

# ── 2. Register RCFDinoModel into models namespace ────────────────────────────
import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel

_models_pkg.RCFDinoModel = RCFDinoModel

# ── 3. Delegate to original main() ────────────────────────────────────────────
from main import main

if __name__ == "__main__":
    main()
