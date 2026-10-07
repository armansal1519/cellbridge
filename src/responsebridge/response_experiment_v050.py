"""v0.5 experiment stages: development LODO or sealed independent splits.

Queries never include hidden target outcomes. Independent evaluation requires
a prediction seal before opening the outcome vault.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .artifacts import complete_stage, require_receipt, sha256, write_json
from .benchmark_data import GroupData, balanced_weights
from .cell_comparators_v050 import record_cell_adapter_receipts
from .comparators_v050 import (CELL_ADAPTERS, ElasticNetResponse, KernelRidgeResponse,
                               LightGBMResponse, PLSResponse, TrainingMean, default_grids,
                               joint_ridge_factory, nested_select, two_penalty_factory)
from .measurement_value import greedy_panel
from .response_downstream_v050 import gate_g1, sign_flip_p
from .response_evaluate_v050 import evaluate_task, holm, interval_report, random_effects_mean
from .response_v050 import (GatedAbundanceResponse, interval_from_scores, mondrian_calibrate,
                            prediction_powered_mean)


def _paired(data: GroupData, lineage, control, perturbed, min_cells):
    obs = data.obs
    keep_lineage = obs.lineage.astype(str) == str(lineage)
    ctrl = keep_lineage & obs.is_control.astype(str).str.lower().isin(["true", "1"])
    if str(control) not in ("", "Baseline", "CTRL", "non-targeting"):
        if "condition" in obs:
            ctrl = ctrl & (obs.condition.astype(str) == str(control))
        else:
            ctrl = ctrl & obs.perturbation.astype(str).isin(["non-targeting", str(control)])
    stim = keep_lineage & (obs.perturbation.astype(str) == str(perturbed))
    rows = []
    for donor in sorted(set(obs.donor.astype(str))):
        c = np.flatnonzero(ctrl & (obs.donor.astype(str) == donor))
        s = np.flatnonzero(stim & (obs.donor.astype(str) == donor))
        if not len(c) or not len(s):
            continue
        if obs.n_cells.iloc[c].sum() < min_cells or obs.n_cells.iloc[s].sum() < min_cells:
            continue
        cw, sw = balanced_weights(obs.iloc[c]), balanced_weights(obs.iloc[s])
        x = data.x[s].T @ (sw / sw.sum()) - data.x[c].T @ (cw / cw.sum())
        y = data.y[s].T @ (sw / sw.sum()) - data.y[c].T @ (cw / cw.sum())
        ax = np.vstack([data.x[c], data.x[s]])
        ay = np.vstack([data.y[c], data.y[s]])
        rows.append({"donor": donor, "x": x, "y": y, "ax": ax, "ay": ay,
                     "aw": np.concatenate([cw / cw.sum() * 0.5, sw / sw.sum() * 0.5])})
    if len(rows) < 3:
        raise ValueError("Fewer than three paired donors after lineage/arm filters")
    return rows


def _stack(rows, key):
    return np.stack([r[key] for r in rows])


def protein_index(proteins, names):
    index = {str(p): i for i, p in enumerate(proteins)}
    missing = [n for n in names if n not in index]
    if missing:
        raise ValueError(f"Proteins absent from prepared roster: {missing}")
    return np.array([index[n] for n in names], dtype=int)


def initialize(groups, run, protocol, *, donor_split=None):
    run = Path(run)
    data = GroupData.load(groups)
    rows = _paired(data, protocol["cohort_lineage"], protocol.get("control_condition", "Baseline"),
                   protocol["primary_stimulation"], protocol.get("minimum_cells_per_arm", 30))
    donors = [r["donor"] for r in rows]
    y = _stack(rows, "y")
    x = _stack(rows, "x")
    donor_arr = np.asarray(donors, dtype=str)
    if donor_split is None:
        tasks = [{"name": "lodo", "train": [d for d in donors if d != held], "test": [held],
                  "calibration": []} for held in donors]
        mode = "development_lodo"
    else:
        roles = donor_split if isinstance(donor_split, dict) else json.loads(Path(donor_split).read_text())
        tasks = [{"name": "independent", "train": list(roles["train"]), "calibration": list(roles["calibration"]),
                  "test": list(roles["test"])}]
        mode = "independent_test"
        if set(roles["test"]) & set(donors) != set(roles["test"]):
            raise ValueError("Locked test donors missing from the paired arm table")
        test_mask = np.isin(donor_arr, roles["test"])
        np.savez_compressed(run / "outcome_vault.npz", y=y[test_mask], donors=donor_arr[test_mask],
                            proteins=np.asarray(data.proteins, dtype=str))
        write_json(run / "outcome_vault.json", {
            "hidden_outcomes": "test_donor_protein_responses",
            "test_donors": list(roles["test"]),
            "rule": "Do not read test-donor target ADT values until prediction_seal.json exists and unblind=True",
            "vault_sha256": sha256(run / "outcome_vault.npz")}, immutable=True)
        y = y.copy()
        y[test_mask] = np.nan
    write_json(run / "task.json", {"mode": mode, "donors": donors, "n": len(donors),
                                   "proteins": data.proteins.tolist()}, immutable=True)
    np.savez_compressed(run / "paired.npz", x=x, y=y, donors=donor_arr)
    abundance_x = np.vstack([r["ax"] for r in rows])
    abundance_y = np.vstack([r["ay"] for r in rows])
    abundance_d = np.concatenate([np.repeat(r["donor"], len(r["aw"])) for r in rows])
    abundance_w = np.concatenate([r["aw"] for r in rows])
    # Abundance for test donors is permitted (RNA and observed proteins in both arms)
    # but target columns of test rows in paired.npz remain NaN in independent mode.
    np.savez_compressed(run / "abundance.npz", x=abundance_x, y=abundance_y,
                        donors=abundance_d, weights=abundance_w)
    write_json(run / "folds.json", tasks, immutable=True)
    complete_stage(run / "initialize", {"mode": mode, "n_donors": len(donors)})
    return tasks


def _load_arrays(run):
    with np.load(Path(run) / "paired.npz", allow_pickle=False) as data:
        paired = {k: data[k] for k in data.files}
    with np.load(Path(run) / "abundance.npz", allow_pickle=False) as data:
        abundance = {k: data[k] for k in data.files}
    return paired, abundance


def _test_anchor_values(run, paired, test, anchors, proteins):
    y_test = paired["y"][test]
    if np.isfinite(y_test).any():
        return y_test[:, anchors]
    vault_path = Path(run) / "outcome_vault.npz"
    if not vault_path.exists():
        raise ValueError("Independent queries need an outcome vault for observed anchors")
    with np.load(vault_path, allow_pickle=False) as vault:
        vault_y, vault_d = vault["y"], vault["donors"].astype(str)
    order = {d: i for i, d in enumerate(vault_d)}
    test_donors = paired["donors"][test]
    rows = np.array([order[str(d)] for d in test_donors])
    return vault_y[rows][:, anchors]


def _test_outcomes(run, paired, test):
    y_test = paired["y"][test]
    if np.isfinite(y_test).all():
        return y_test
    with np.load(Path(run) / "outcome_vault.npz", allow_pickle=False) as vault:
        vault_y, vault_d = vault["y"], vault["donors"].astype(str)
    order = {d: i for i, d in enumerate(vault_d)}
    rows = np.array([order[str(d)] for d in paired["donors"][test]])
    return vault_y[rows]


def fit_predict(run, protocol, proteins):
    run = Path(run)
    require_receipt(run / "initialize")
    paired, abundance = _load_arrays(run)
    tasks = json.loads((run / "folds.json").read_text())
    targets = protein_index(proteins, protocol["primary_targets"])
    forbidden = set(protocol.get("forbidden_anchors", protocol["primary_targets"]))
    candidate_names = [p for p in proteins if p not in forbidden]
    candidate_idx = protein_index(proteins, candidate_names)
    predictions = {}
    seals = []
    independent = tasks[0]["name"] == "independent"
    for i, task in enumerate(tasks):
        train = np.isin(paired["donors"], task["train"])
        test = np.isin(paired["donors"], task["test"])
        atrain = np.isin(abundance["donors"], task["train"])
        n_hvg = int(protocol.get("rna_hvg", 2000))
        probe = GatedAbundanceResponse(alpha=1.0, shrinkage=0.5, n_hvg=n_hvg)
        probe.fit(paired["x"][train], paired["y"][train], paired["donors"][train],
                  abundance_x=abundance["x"][atrain], abundance_y=abundance["y"][atrain],
                  abundance_donors=abundance["donors"][atrain],
                  abundance_weights=abundance["weights"][atrain],
                  anchors=protein_index(proteins, protocol["fixed_panel_8"]),
                  gate_targets=targets)
        designed = greedy_panel(probe.inner.covariance, candidate_idx, targets,
                                len(task["train"]), 0.5, min(8, len(candidate_idx)))
        anchors = np.array(designed["panel"] or protein_index(proteins, protocol["fixed_panel_8"]), dtype=int)
        model = GatedAbundanceResponse(alpha=1.0, shrinkage=0.5, n_hvg=n_hvg)
        model.fit(paired["x"][train], paired["y"][train], paired["donors"][train],
                  abundance_x=abundance["x"][atrain], abundance_y=abundance["y"][atrain],
                  abundance_donors=abundance["donors"][atrain],
                  abundance_weights=abundance["weights"][atrain],
                  anchors=anchors, gate_targets=targets)
        query_anchors = _test_anchor_values(run, paired, test, anchors, proteins)
        pred = model.predict(paired["x"][test], query_anchors, anchors)
        mean = TrainingMean().fit(paired["x"][train], paired["y"][train], paired["y"][train][:, anchors],
                                  donors=paired["donors"][train])
        methods = {"gated_abundance_response": pred["prediction"],
                   "training_mean": mean.predict(paired["x"][test]),
                   "rna_only": pred["rna_only_prediction"],
                   "ungated": pred["ungated_prediction"]}
        scale = model.train_scales
        nested_log = {}
        if protocol.get("fit_nested_comparators", True):
            grids = default_grids(n_hvg)
            factories = (("elastic_net", ElasticNetResponse), ("pls", PLSResponse),
                         ("kernel_ridge", KernelRidgeResponse), ("lightgbm", LightGBMResponse))
            for name, factory in factories:
                selected = nested_select(factory, grids[name], paired["x"][train], paired["y"][train],
                                         paired["y"][train][:, anchors], paired["donors"][train], targets, scale)
                nested_log[name] = {k: selected[k] for k in selected if k != "model"}
                if selected["status"] == "selected":
                    methods[name] = selected["model"].predict(paired["x"][test], query_anchors)
                else:
                    methods[name] = np.full((int(test.sum()), paired["y"].shape[1]), np.nan)
            two = nested_select(two_penalty_factory, grids["two_penalty_ridge"], paired["x"][train], paired["y"][train],
                                paired["y"][train][:, anchors], paired["donors"][train], targets, scale)
            nested_log["two_penalty_ridge"] = {k: two[k] for k in two if k != "model"}
            if two["status"] == "selected":
                methods["two_penalty_ridge"] = two["model"].predict(paired["x"][test], query_anchors)
            joint = nested_select(joint_ridge_factory, grids["joint_ridge"], paired["x"][train], paired["y"][train],
                                  paired["y"][train][:, anchors], paired["donors"][train], targets, scale)
            nested_log["joint_ridge"] = {k: joint[k] for k in joint if k != "model"}
            if joint["status"] == "selected":
                methods["joint_ridge"] = joint["model"].predict(paired["x"][test], query_anchors)
        else:
            joint = joint_ridge_factory(alpha=1.0, n_hvg=n_hvg).fit(
                paired["x"][train], paired["y"][train], paired["y"][train][:, anchors],
                donors=paired["donors"][train])
            methods["joint_ridge"] = joint.predict(paired["x"][test], query_anchors)
            two = two_penalty_factory(alpha_rna=1.0, alpha_anchor=1.0, n_hvg=n_hvg).fit(
                paired["x"][train], paired["y"][train], paired["y"][train][:, anchors],
                donors=paired["donors"][train])
            methods["two_penalty_ridge"] = two.predict(paired["x"][test], query_anchors)
        folder = run / f"fold_{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "predictions.npz", **{k: np.asarray(v) for k, v in methods.items()})
        write_json(folder / "panel.json", {"anchors": anchors.tolist(), "designed": designed}, immutable=True)
        write_json(folder / "gate.json", {"weights": model.gate_weights.tolist()}, immutable=True)
        write_json(folder / "comparator_log.json", json.loads(json.dumps(nested_log, default=str)), immutable=True)
        record_cell_adapter_receipts(folder)
        cal_ids = task.get("calibration") or []
        if cal_ids:
            cal = np.isin(paired["donors"], cal_ids)
            cal_anchors = paired["y"][cal][:, anchors]
            cal_pred = model.predict(paired["x"][cal], cal_anchors, anchors)["prediction"]
            calibration = mondrian_calibrate(paired["y"][cal], cal_pred, scale, paired["donors"][cal],
                                             alpha=float(protocol.get("uncertainty", {}).get("alpha", 0.1)))
            intervals = interval_from_scores(pred["prediction"], scale, calibration)
            np.savez_compressed(folder / "intervals.npz", lower=intervals["lower"], upper=intervals["upper"],
                                available=intervals["available"])
            write_json(folder / "calibration.json", {"n": int(cal.sum()), "alpha": 0.1}, immutable=True)
        complete_stage(folder / "predictions")
        key = task["name"] if task["name"] != "lodo" else f"lodo_{task['test'][0]}"
        predictions[key] = methods
        seals.append({"fold": folder.name, "predictions_sha256": sha256(folder / "predictions.npz")})
    write_json(run / "prediction_seal.json", {"folds": seals, "cell_adapters": CELL_ADAPTERS,
                                              "hidden_outcomes_opened": False,
                                              "independent": independent}, immutable=True)
    return predictions


def evaluate(run, protocol, proteins, *, unblind=False):
    run = Path(run)
    if not (run / "prediction_seal.json").exists():
        raise ValueError("Predictions must be sealed before evaluation")
    paired, _ = _load_arrays(run)
    tasks = json.loads((run / "folds.json").read_text())
    independent = tasks[0]["name"] == "independent"
    if independent and not unblind:
        raise ValueError("Independent outcomes remain in the vault until unblind=True after P4 freeze")
    targets = protein_index(proteins, protocol["primary_targets"])
    summaries = []
    effects, variances, pvalues = [], [], []
    coverage_rows = []
    for i, task in enumerate(tasks):
        test = np.isin(paired["donors"], task["test"])
        with np.load(run / f"fold_{i:02d}/predictions.npz", allow_pickle=False) as data:
            methods = {k: data[k] for k in data.files}
        y = _test_outcomes(run, paired, test)
        train = np.isin(paired["donors"], task["train"])
        mean = np.average(paired["y"][train], axis=0)
        scale = np.maximum(np.sqrt(np.average((paired["y"][train] - mean)**2, axis=0)), 1e-6)
        y_t = y[:, targets]
        methods_t = {k: v[:, targets] for k, v in methods.items()}
        baseline = "joint_ridge" if "joint_ridge" in methods_t else "two_penalty_ridge"
        if baseline not in methods_t:
            baseline = "training_mean"
        result = evaluate_task(y_t, methods_t, scale[targets], mean[targets], paired["donors"][test],
                               independent=independent, baseline=baseline)
        result["task"] = task["name"] if task["name"] != "lodo" else f"lodo_{task['test'][0]}"
        interval_path = run / f"fold_{i:02d}/intervals.npz"
        if interval_path.exists():
            with np.load(interval_path, allow_pickle=False) as iv:
                report = interval_report(y_t, iv["lower"][:, targets], iv["upper"][:, targets],
                                         alpha=float(protocol.get("uncertainty", {}).get("alpha", 0.1)))
            result["intervals"] = report
            coverage_rows.append(report)
        if independent and result["paired_inference"] and result["paired_inference"].get("donor_differences") is not None:
            diffs = result["paired_inference"]["donor_differences"]
            result["p_sign_flip"] = sign_flip_p(diffs)
            labeled = y_t
            pred = methods_t["gated_abundance_response"]
            result["ppi"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                             for k, v in prediction_powered_mean(labeled, pred, pred).items()}
            if result["paired_inference"].get("difference_ci95"):
                lo, hi = result["paired_inference"]["difference_ci95"]
                effects.append(result["paired_inference"]["mean_difference"])
                variances.append(max(((hi - lo) / 3.92) ** 2, 1e-12))
                pvalues.append(result["p_sign_flip"])
        summaries.append(result)
    g1 = gate_g1(summaries)
    decision = {"version": "0.5.0", "independent": independent, "unblinded": bool(unblind),
                "publication_ready": False, "summaries": summaries, "gate_g1": g1,
                "coverage": coverage_rows}
    if effects:
        decision["meta"] = random_effects_mean(effects, variances)
        decision["holm"] = holm(pvalues).tolist()
        decision["p_sign_flip"] = pvalues
    write_json(run / "evaluation/decision.json", {k: v for k, v in decision.items() if k != "summaries"},
               immutable=True)
    write_json(run / "evaluation/summaries.json", summaries, immutable=True)
    complete_stage(run / "evaluation")
    return decision
