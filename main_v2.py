"""
main_v2.py  –  like main.py, but also registers V2 model components.

Why this file exists
--------------------
`rcf_model.py` instantiates the decode head via

    self.decode_head = globals()[decode_head.pop('type')](...)

`globals()` there refers to the *module-level namespace of rcf_model.py*,
so any new head class must live in that namespace.

Rather than editing rcf_model.py (which we want to leave untouched), we
inject the V2 class into that namespace right at import time, then delegate
to the original main() function.

Usage: replace `python main.py` with `python main_v2.py` in your scripts.
All CLI flags are identical.
"""

# ── 1. Register V2 components into rcf_model's namespace ──────────────────
import models.rcf_model as _rcf_model_module
from models.flow_aggregation_head_with_residual_v2 import (
    FlowAggregationHeadWithResidualV2,
)

_rcf_model_module.FlowAggregationHeadWithResidualV2 = FlowAggregationHeadWithResidualV2

# ── 2. Delegate to the original main() ────────────────────────────────────
from main import main

if __name__ == "__main__":
    main()
