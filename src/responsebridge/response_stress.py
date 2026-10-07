"""Prespecified synthetic stress tests, not independent biological evidence."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np

from .artifacts import complete_stage, freeze_implementation, write_json
from .shrinkage import ResponseShrinkage


ANCHORS = np.array([0, 1, 2, 3])
TARGETS = np.array([4, 5])
PROTEINS = ("anchor_1", "anchor_2", "anchor_3", "anchor_4", "CD25", "CD69")
SCENARIOS = (
    {"name": "reference", "cells": 128},
    {"name": "cells_64", "cells": 64},
    {"name": "cells_16", "cells": 16},
    {"name": "cells_4", "cells": 4},
    {"name": "rna_adt_depth_25pct", "cells": 128, "depth": .25},
    {"name": "adt_stim_depth_25pct", "cells": 128, "stim_adt_depth": .25},
    {"name": "adt_overdispersion_shape2", "cells": 128, "adt_gamma_shape": 2.},
    {"name": "composition_20_to_80pct", "cells": 128, "composition": (.2, .8)},
    {"name": "hidden_only_response_plus1", "cells": 128, "hidden_shift": 1.},
)


def stress_protocol(seed=20261004):
    return {"schema_version": 1, "seed": int(seed), "training_donors": 24,
            "query_donors": 48, "cells_per_state_pool": 128, "rna_genes": 24,
            "proteins": list(PROTEINS), "anchors": ANCHORS.tolist(),
            "targets": TARGETS.tolist(), "scenarios": list(SCENARIOS),
            "model": {"alpha": 1., "shrinkage": .5, "n_hvg": 24},
            "training": "Fit once to reference-condition synthetic training donors; no test tuning or scenario refits",
            "estimand": "Paired mean cellwise log1p response; evaluation truth is a finite latent-intensity reference, not noisy held-out target counts",
            "truth": "State-weighted mean log1p noiseless per-cell intensity, using all reference pool cells in each state",
            "composition": "Mixture-specific truth changes with state proportions; this is an estimand/distribution shift, not fixed-truth noise",
            "uncertainty": "Model-conditional residual variances only; no calibrated coverage or biological generalization claim",
            "interpretation": "Controlled software/mechanism simulations; not independent biology or proof of superiority"}


def _pool(seed, donors, prefix):
    rng = np.random.default_rng(seed)
    genes, proteins, per_state = 24, 6, 128
    states = np.repeat([0, 1], per_state)
    z = rng.normal(size=(donors, 3))
    rna_load = np.column_stack((np.linspace(-.4, .4, genes), np.cos(np.arange(genes)) * .3))
    protein_load = np.array([[.4, .1, .5], [.1, .5, -.3], [-.3, .2, .4],
                             [.2, -.3, .5], [.4, .3, .6], [-.2, .4, -.5]])
    protein_state = np.array([.9, -.5, .7, -.4, 1.1, -.7])
    rna_state = np.sin(np.arange(genes) / 2) * .8
    rna_base = 3.2 + rng.normal(scale=.15, size=(donors, 1, genes))
    adt_base = 3.5 + rng.normal(scale=.15, size=(donors, 1, proteins))
    rx, ry = [], []
    for arm in (0, 1):
        rx.append(np.exp(rna_base + states[None, :, None] * rna_state
                         + arm * (.15 + z[:, :2] @ rna_load.T)[:, None, :]
                         + rng.normal(scale=.2, size=(donors, len(states), genes))))
        ry.append(np.exp(adt_base + states[None, :, None] * protein_state
                         + arm * (.35 + z @ protein_load.T)[:, None, :]
                         + rng.normal(scale=.2, size=(donors, len(states), proteins))))
    rna_rates, adt_rates = np.stack(rx, axis=1), np.stack(ry, axis=1)
    return {"donors": np.array([f"{prefix}_{d:03d}" for d in range(donors)]),
            "states": states, "rna_rates": rna_rates, "adt_rates": adt_rates,
            "rna_counts": rng.poisson(rna_rates), "adt_counts": rng.poisson(adt_rates)}


def _reference(pool, composition):
    arm_means = []
    latent = np.log1p(pool["adt_rates"])
    for arm, p in enumerate(composition):
        arm_means.append((1 - p) * latent[:, arm, pool["states"] == 0].mean(axis=1)
                         + p * latent[:, arm, pool["states"] == 1].mean(axis=1))
    return arm_means[1] - arm_means[0]


def observe_pool(pool, scenario, seed=20261004):
    """Return observable RNA/anchor responses and a separate evaluation oracle.

Shared reference counts are used across thinning/composition scenarios. Cell
selection uses nested prefixes within each state, without replacement. The
hidden-only scenario changes the oracle and nothing observable.
"""
    rng = np.random.default_rng(seed)
    rna, adt = pool["rna_counts"], pool["adt_counts"]
    if "depth" in scenario:
        rna = rng.binomial(rna, scenario["depth"])
        adt = rng.binomial(adt, scenario["depth"])
    if "stim_adt_depth" in scenario:
        adt = adt.copy()
        adt[:, 1] = rng.binomial(adt[:, 1], scenario["stim_adt_depth"])
    if "adt_gamma_shape" in scenario:
        shape = scenario["adt_gamma_shape"]
        adt = rng.poisson(pool["adt_rates"] * rng.gamma(shape, 1 / shape, adt.shape))
    raw_cells = scenario["cells"]
    if isinstance(raw_cells, bool) or int(raw_cells) != raw_cells or raw_cells < 1:
        raise ValueError("cells must be a positive integer")
    cells = int(raw_cells)
    composition = scenario.get("composition", (.5, .5))
    if len(composition) != 2 or any(not np.isfinite(p) or not 0 <= p <= 1 for p in composition):
        raise ValueError("composition must give two finite state fractions in [0, 1]")
    state_positions = [np.flatnonzero(pool["states"] == s) for s in (0, 1)]
    # Identical selection seeds across scenarios ensure genuine paired thinning.
    select_rng = np.random.default_rng(seed + 7919)
    x, measured_y, selections = [], [], []
    for d in range(len(rna)):
        xx, yy, selected = [], [], []
        for arm, p in enumerate(composition):
            n1 = round(cells * p)
            sizes = [cells - n1, n1]
            if min(sizes) < 0 or any(n > len(pos) for n, pos in zip(sizes, state_positions)):
                raise ValueError("Requested cells/state proportions exceed the fixed reference pool")
            idx = np.concatenate([select_rng.permutation(pos)[:n] for n, pos in zip(sizes, state_positions)])
            selected.append(idx)
            xx.append(np.log1p(rna[d, arm, idx]).mean(axis=0))
            yy.append(np.log1p(adt[d, arm, idx]).mean(axis=0))
        x.append(xx[1] - xx[0])
        measured_y.append(yy[1] - yy[0])
        selections.append(selected)
    # Use the actually sampled proportions, avoiding rounding mismatches.
    actual_composition = tuple(round(cells * p) / cells for p in composition)
    truth = _reference(pool, actual_composition)
    truth[:, TARGETS] += scenario.get("hidden_shift", 0.)
    measured_y = np.asarray(measured_y)
    return {"x": np.asarray(x), "anchor_values": measured_y[:, ANCHORS],
            "measured_y": measured_y, "truth": truth,
            "cell_indices": np.asarray(selections), "composition": actual_composition}


def _numeric_checks(model, x, anchors):
    full = model.predict(x, anchors, ANCHORS)
    no_anchor = model.predict(x, np.empty((len(x), 0)), np.array([], int))
    checks = {
        "decomposition_max_abs_error": float(np.max(np.abs(full["prediction"] -
            (full["base_response"] + full["rna_contribution"] + full["anchor_contributions"].sum(-1))))),
        "no_anchor_mean_max_abs_error": float(np.max(np.abs(no_anchor["prediction"] -
            model.predict_rna(x) - model.residual_mean))),
        "no_anchor_variance_max_abs_error": float(np.max(np.abs(no_anchor["model_variance"] - np.diag(model.covariance)))),
        "exact_additive_anchor_ablation_max_abs_error": 0.,
        "drop_anchor_reconditioning_max_abs_error": 0.,
        "minimum_variance_increase_after_dropping_anchor": 0.,
    }
    full_mean = model.predict_rna(x) + model.residual_mean
    variance_increases = []
    for j, anchor in enumerate(ANCHORS):
        changed = anchors.copy()
        changed[:, j] = full_mean[:, anchor]
        ablated = model.predict(x, changed, ANCHORS)["prediction"]
        exact = full["prediction"] - full["anchor_contributions"][:, :, j]
        checks["exact_additive_anchor_ablation_max_abs_error"] = max(
            checks["exact_additive_anchor_ablation_max_abs_error"], float(np.max(np.abs(ablated-exact))))
        keep = ANCHORS[ANCHORS != anchor]
        dropped = model.predict(x, anchors[:, ANCHORS != anchor], keep)
        operator = np.linalg.solve(model.covariance[np.ix_(keep, keep)], model.covariance[keep]).T
        expected = full_mean + (anchors[:, ANCHORS != anchor] - full_mean[:, keep]) @ operator.T
        checks["drop_anchor_reconditioning_max_abs_error"] = max(
            checks["drop_anchor_reconditioning_max_abs_error"], float(np.max(np.abs(dropped["prediction"]-expected))))
        variance_increases.extend((dropped["model_variance"][TARGETS] - full["model_variance"][TARGETS]).tolist())
    checks["minimum_variance_increase_after_dropping_anchor"] = float(min(variance_increases))
    diagonal = deepcopy(model)._set_shrinkage(1.)
    corrected = diagonal.predict(x, anchors, ANCHORS)
    checks["diagonal_covariance_hidden_correction_max_abs"] = float(np.max(np.abs(corrected["anchor_contributions"][:, TARGETS])))
    # A separate constant/collinear fixture tests the numerical floor, not performance.
    singular = ResponseShrinkage().fit(np.ones((6, 3)),
        np.arange(6.)[:, None] * np.ones((1, 6)), np.arange(6), alpha=None, shrinkage=0)
    singular_pred = singular.predict(np.ones((2, 3)), np.ones((2, 4)), ANCHORS)
    checks["singular_covariance_min_eigenvalue"] = float(np.linalg.eigvalsh(singular.covariance).min())
    checks["singular_predictions_finite"] = bool(np.isfinite(singular_pred["prediction"]).all())
    checks["singular_variances_finite_nonnegative"] = bool(np.isfinite(singular_pred["model_variance"]).all()
                                                         and (singular_pred["model_variance"] >= 0).all())
    return checks


def compute_stress(seed=20261004):
    """Run the small fixed suite in memory; never select parameters on results."""
    protocol = stress_protocol(seed)
    train_pool = _pool(seed + 1, protocol["training_donors"], "train")
    query_pool = _pool(seed + 2, protocol["query_donors"], "query")
    train = observe_pool(train_pool, SCENARIOS[0], seed + 3)
    model = ResponseShrinkage().fit(train["x"], train["measured_y"], train_pool["donors"], **protocol["model"])
    scale = model.train_scales[TARGETS]
    rows, per_donor, arrays, predictions = [], [], {}, {}
    for scenario in SCENARIOS:
        observed = observe_pool(query_pool, scenario, seed + 4)
        result = model.predict(observed["x"], observed["anchor_values"], ANCHORS)
        no_anchor = model.predict(observed["x"], np.empty((len(observed["x"]), 0)), np.array([], int))
        methods = {"response_shrinkage": result["prediction"],
                   "response_no_anchors": no_anchor["prediction"],
                   "rna_only": model.predict_rna(observed["x"]),
                   "mean_response_template": np.broadcast_to(model.cohort_mean, result["prediction"].shape)}
        for method, pred in methods.items():
            err = (pred[:, TARGETS] - observed["truth"][:, TARGETS]) / scale
            donor_mae = np.abs(err).mean(axis=1)
            rows.append({"scenario": scenario["name"], "method": method,
                         "n_donors": len(pred), "cells_per_arm": scenario["cells"],
                         "standardized_mae": float(donor_mae.mean()),
                         "standardized_rmse": float(np.sqrt(np.mean(err**2))),
                         "prediction_available_fraction": float(np.isfinite(pred[:, TARGETS]).mean()),
                         "model_sd_mean": float(np.sqrt(result["model_variance"][TARGETS]).mean())
                         if method == "response_shrinkage" else None})
            per_donor.extend({"scenario": scenario["name"], "method": method,
                              "donor": str(d), "standardized_mae": float(e)}
                             for d, e in zip(query_pool["donors"], donor_mae))
        predictions[scenario["name"]] = result
        for key in ("x", "anchor_values", "truth", "cell_indices"):
            arrays[scenario["name"] + "__" + key] = observed[key]
        arrays[scenario["name"] + "__prediction"] = result["prediction"]
    ref = predictions["reference"]
    hidden = predictions["hidden_only_response_plus1"]
    checks = _numeric_checks(model, arrays["reference__x"], arrays["reference__anchor_values"])
    checks.update({"hidden_only_prediction_identical": bool(np.array_equal(ref["prediction"], hidden["prediction"])),
                   "hidden_only_model_variance_identical": bool(np.array_equal(ref["model_variance"], hidden["model_variance"])),
                   "hidden_only_input_rna_identical": bool(np.array_equal(arrays["reference__x"], arrays["hidden_only_response_plus1__x"])),
                   "hidden_only_input_anchors_identical": bool(np.array_equal(arrays["reference__anchor_values"], arrays["hidden_only_response_plus1__anchor_values"])),
                   "training_query_donors_disjoint": not bool(set(train_pool["donors"]) & set(query_pool["donors"])),
                   "composition_changes_reference_target_response_mean_abs": float(np.mean(np.abs(
                       arrays["composition_20_to_80pct__truth"][:, TARGETS] - arrays["reference__truth"][:, TARGETS]))),
                   "observed_inputs_only": True})
    arrays.update(train_x=train["x"], train_y=train["measured_y"],
                  training_donors=train_pool["donors"], query_donors=query_pool["donors"],
                  training_target_scale=scale)
    return {"protocol": protocol, "summary": rows, "donor_errors": per_donor,
            "checks": checks, "arrays": arrays, "model": model}


def run_stress(output, seed=20261004):
    """Save a new immutable simulated report with protocol, model and receipts."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", stress_protocol(seed), immutable=True)
    freeze_implementation(output)
    result = compute_stress(seed)
    result["model"].save(output / "model.npz")
    np.savez_compressed(output / "synthetic_arrays.npz", **result["arrays"])
    summary = pd.DataFrame(result["summary"])
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(result["donor_errors"]).to_csv(output / "donor_errors.csv", index=False)
    write_json(output / "checks.json", result["checks"], immutable=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [s["name"] for s in SCENARIOS]
    methods = list(dict.fromkeys(summary.method))
    yy = np.arange(len(names))
    for j, method in enumerate(methods):
        values = summary[summary.method == method].set_index("scenario").loc[names, "standardized_mae"]
        ax.barh(yy + (j-1.5)*.18, values, height=.17, label=method.replace("_", " "))
    ax.set_yticks(yy, [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Mean standardized absolute error; equal synthetic donors")
    ax.set_title("Controlled simulation: every prespecified stress condition")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "stress_comparison.png", dpi=160)
    fig.savefig(output / "stress_comparison.pdf")
    plt.close(fig)
    lines = ["# Prespecified response robustness simulations", "",
        "These are controlled synthetic experiments, not independent biological evidence.", "",
        "A single fixed model is trained on 24 synthetic donors and evaluated on the same 48 separate synthetic donors under all nine conditions. No scenario result selects model parameters.", "",
        "| Scenario | Method | Standardized MAE |", "| --- | --- | ---: |"]
    lines += [f"| {r['scenario']} | {r['method']} | {r['standardized_mae']:.4f} |" for r in result["summary"]]
    lines += ["", "Cell thinning changes sampling precision; low depth and overdispersion change measured inputs. Composition shifts also change the mean-response estimand, so the oracle follows the declared mixture. No condition is required to improve a method's error.", "",
        "Hidden-only shifts leave both predictions and model variances unchanged by construction: the observable inputs carry no information about the added target response. This is an impossibility example, not a successful shift detector.", "",
        "Anchor contribution removal holds the conditioning set fixed and sets one residual to zero. Actually dropping an anchor recomputes the conditional operator; these are distinct checks. With no anchors, the response is the RNA prediction plus residual mean, with the full marginal covariance diagonal.", "",
        "Conditional model variances are neither calibrated prediction intervals nor confidence intervals for method gains. This simulation does not test the full RNA preprocessing/real-assay pipeline and contains no donor-population inference.", "",
        "Numerical checks are recorded in checks.json; all simulated inputs, oracles and predictions are preserved in synthetic_arrays.npz."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    complete_stage(output, {"kind": "controlled_synthetic_response_stress", "seed": int(seed),
                            "scenarios": len(SCENARIOS), "biological_validation": False})
    return {"output": str(output.resolve()), "scenarios": len(SCENARIOS),
            "status": "complete_synthetic_only", "checks": result["checks"]}
