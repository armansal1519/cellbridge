"""Semi-synthetic ground truth for v0.5: gating, criterion, composition, small n."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import complete_stage, freeze_implementation, write_json
from .measurement_value import measurement_benefit
from .response_stress_v040 import ANCHORS, TARGETS, _pool, observe_pool
from .response_v050 import GatedAbundanceResponse


SCENARIOS = (
    {"name": "reference", "cells": 128},
    {"name": "composition_20_to_80pct", "cells": 128, "composition": (0.2, 0.8)},
    {"name": "hidden_only_response_plus1", "cells": 128, "hidden_shift": 1.0},
    {"name": "adt_stim_depth_25pct", "cells": 128, "stim_adt_depth": 0.25},
    {"name": "small_n", "cells": 128, "training_donors": 8},
)


def stress_protocol(seed=20261007):
    return {"schema_version": 1, "version": "0.5.0", "seed": int(seed),
            "training_donors": 16, "query_donors": 24, "scenarios": list(SCENARIOS),
            "model": {"alpha": 1.0, "shrinkage": 0.5, "n_hvg": 24},
            "selection_on_results": False,
            "interpretation": "Controlled synthetic checks of gating and the measurement-value criterion; not biological evidence"}


def compute_stress(seed=20261007):
    protocol = stress_protocol(seed)
    n_train = protocol["training_donors"]
    tp = _pool(seed + 1, n_train, "train")
    qp = _pool(seed + 2, protocol["query_donors"], "query")
    train = observe_pool(tp, SCENARIOS[0], seed + 3)
    kwargs = {k: train[k] for k in ("abundance_x", "abundance_y", "abundance_donors", "abundance_weights")}
    model = GatedAbundanceResponse(**protocol["model"]).fit(
        train["x"], train["measured_y"], tp["donors"], **kwargs, anchors=ANCHORS, gate_targets=TARGETS)
    rows = []
    for scenario in SCENARIOS:
        if scenario["name"] == "small_n":
            keep = tp["donors"][: scenario["training_donors"]]
            mask = np.isin(tp["donors"], keep)
            small = GatedAbundanceResponse(**protocol["model"]).fit(
                train["x"][mask], train["measured_y"][mask], tp["donors"][mask],
                abundance_x=train["abundance_x"][np.isin(train["abundance_donors"], keep)],
                abundance_y=train["abundance_y"][np.isin(train["abundance_donors"], keep)],
                abundance_donors=train["abundance_donors"][np.isin(train["abundance_donors"], keep)],
                abundance_weights=train["abundance_weights"][np.isin(train["abundance_donors"], keep)],
                anchors=ANCHORS, gate_targets=TARGETS)
            fitted = small
            n_used = int(scenario["training_donors"])
        else:
            fitted = model
            n_used = n_train
        observed = observe_pool(qp, scenario, seed + 4)
        pred = fitted.predict(observed["x"], observed["anchor_values"], ANCHORS)
        delta = pred["prediction"][:, TARGETS] - observed["truth"][:, TARGETS]
        benefit = measurement_benefit(fitted.inner.covariance, ANCHORS, TARGETS, n_used, protocol["model"]["shrinkage"])
        rows.append({"scenario": scenario["name"], "query_mae_targets": float(np.mean(np.abs(delta))),
                     "gate_mean_anchor_weight": float(fitted.gate_weights[:, 2].mean()),
                     "mean_target_benefit": float(benefit["benefit"].mean()),
                     "n_train": n_used})
    checks = {"training_query_disjoint": not bool(set(tp["donors"]) & set(qp["donors"])),
              "gating_weights_simplex": bool(np.allclose(model.gate_weights.sum(1), 1) and np.all(model.gate_weights >= -1e-12)),
              "parameter_selection_on_scenarios": False}
    return {"protocol": protocol, "summary": rows, "checks": checks}


def run_stress(output, seed=20261007):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", stress_protocol(seed), immutable=True)
    freeze_implementation(output)
    result = compute_stress(seed)
    write_json(output / "checks.json", result["checks"], immutable=True)
    write_json(output / "summary.json", result["summary"], immutable=True)
    complete_stage(output, {"version": "0.5.0"})
    return result
