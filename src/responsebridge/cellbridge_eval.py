"""Fixed-sequence tests and the two-cohort meta-analysis for CellBridge."""
from __future__ import annotations

import numpy as np

from .reliability import paired_gain_bootstrap
from .response_downstream_v050 import sign_flip_p
from .response_evaluate_v050 import random_effects_mean


def fixed_sequence(loss, hypotheses, donors, *, alpha, seed, metadata):
    """Test hypotheses in order. Stop after the first non-rejection.

    A missing comparator is skipped and does not stop the sequence.
    ``loss`` maps (method, endpoint) to a vector of donor losses.
    """
    tests = []
    stop = False
    for hypothesis in hypotheses:
        comparator, endpoint = hypothesis["comparator"], hypothesis["endpoint"]
        record = {"id": hypothesis["id"], "endpoint": endpoint, "model": "cellbridge",
                  "comparator": comparator}
        key = (comparator, endpoint)
        if key not in loss or not np.isfinite(loss[key]).all():
            record["status"] = "comparator_not_available_removed_by_protocol"
            tests.append(record)
            continue
        baseline, model = loss[key], loss[("cellbridge", endpoint)]
        boot = paired_gain_bootstrap(baseline, model, donors, n_bootstrap=10000, seed=seed,
                                     independent_test=True, models_frozen=True, model_metadata=metadata)
        p_value = sign_flip_p(baseline - model)
        gain = float(np.mean(baseline - model))
        rejected = (not stop) and p_value is not None and p_value < alpha and gain > 0
        record.update({
            "mean_loss_comparator": float(baseline.mean()),
            "mean_loss_cellbridge": float(model.mean()),
            "mean_difference": gain,
            "relative_gain": boot["relative_gain"],
            "difference_ci95": boot["difference_ci95"],
            "sign_flip_p_two_sided": p_value,
            "donor_wins": int(np.sum(model < baseline)),
            "n_donors": int(len(baseline)),
            "tested_in_sequence": not stop,
            "rejected_null": bool(rejected),
        })
        if not stop and not rejected:
            stop = True
        tests.append(record)
    return tests


def cohort_effect(differences):
    """Mean paired gain and its sampling variance."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        raise ValueError("A cohort effect needs at least two donors")
    mean = float(values.mean())
    variance = float(values.var(ddof=1) / n)
    return {"mean": mean, "variance": max(variance, 1e-12), "n_donors": int(n)}


def meta_analyze(cohorts):
    """Fixed-effect and DerSimonian-Laird means of cohort-level gains.

    ``cohorts`` is a list of {"cohort", "mean", "variance"} dicts.
    """
    effects = np.array([c["mean"] for c in cohorts], dtype=float)
    variances = np.array([c["variance"] for c in cohorts], dtype=float)
    weights = 1.0 / variances
    fixed = float(np.sum(weights * effects) / np.sum(weights))
    fixed_se = float(np.sqrt(1.0 / np.sum(weights)))
    random = random_effects_mean(effects, variances)
    return {"cohorts": cohorts, "fixed_effect_mean": fixed, "fixed_effect_se": fixed_se,
            "random_effects": random}
