"""
main_dino_adapter.py — Entry point for adapter-based fine-tuning.

Registers RCFDinoAdapterModel in addition to RCFDinoModel,
then delegates to the standard main().
"""

import models.rcf_model as _rcf_mod
from models.flow_aggregation_head_with_residual_v2 import FlowAggregationHeadWithResidualV2
_rcf_mod.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

import models as _models_pkg
from models.rcf_dino_model import RCFDinoModel
from models.rcf_dino_adapter_model import RCFDinoAdapterModel

_models_pkg.RCFDinoModel = RCFDinoModel
_models_pkg.RCFDinoAdapterModel = RCFDinoAdapterModel

from main import main

if __name__ == "__main__":
    main()
