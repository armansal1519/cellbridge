"""Descriptive plots after the seals. Does not refit or rewrite a decision.

Reads sealed predictions and the outcomes that the single unblind already opened.
Writes only under runs/cellbridge_v100/*_posthoc/ and marks every record post_hoc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cellbridge_dev_v100 import paired_rows  # noqa: E402
from responsebridge.emtab_adapter import load_arm  # noqa: E402
from responsebridge.response_baselines import response_scale  # noqa: E402


def _json(path):
    return json.loads(Path(path).read_text())


def _write(dest: Path, frame: pd.DataFrame | None, payload: dict):
    if not any(part.endswith("_posthoc") for part in dest.resolve().parts):
        raise PermissionError(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if frame is not None:
        frame.to_csv(dest / "table.csv", index=False)
    body = {"post_hoc": True, **payload}
    (dest / "record.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def _check(summary, method, endpoint, value, tol=1e-6):
    row = summary[(summary.method == method) & (summary.endpoint == endpoint)].iloc[0]
    got = float(row.standardized_mae)
    if abs(got - value) > tol:
        raise SystemExit(f"{method} {endpoint}: recomputed {value} != sealed {got}")


def gse():
    protocol = _json(ROOT / "configs/protocol_cellbridge_v100.json")
    freeze = _json(ROOT / "runs/cellbridge_v100/gse334503/freeze.json")
    prepared = ROOT / "runs/gse334503_zenodo_workflow_v040_v2/prepared"
    roles = freeze["roles"]
    train, queries = roles["train"], list(roles["calibration"]) + list(roles["test"])
    train_rows, proteins, _ = paired_rows(
        prepared / "groups", protocol["lineage"], protocol["control"], protocol["perturbed"], roles["train"])
    query_rows, _, _ = paired_rows(
        prepared / "groups", protocol["lineage"], protocol["control"], protocol["perturbed"], queries)
    proteins = list(proteins)
    ytr = np.stack([row["y"] for row in train_rows])
    scale = response_scale(ytr)
    truth = {row["donor"]: row["y"] for row in query_rows}
    with np.load(ROOT / "runs/cellbridge_v100/gse334503/predictions/core.npz") as blob:
        donors = list(blob["donors"].astype(str))
        pred_proteins = list(blob["proteins"].astype(str))
        predicted = blob["cellbridge"].astype(float)
    if pred_proteins != proteins or donors != queries:
        raise SystemExit("GSE prediction order does not match the prepared responses")
    observed = np.stack([truth[donor] for donor in donors])
    is_test = np.array([donor in set(roles["test"]) for donor in donors])
    primary = np.array([proteins.index(name) for name in protocol["primary_targets"]])
    anchors = {proteins.index(name) for name in protocol["anchors"]}
    targets = np.array([i for i in range(len(proteins)) if i not in anchors])
    err = np.abs(predicted - observed) / scale
    _check(pd.read_csv(ROOT / "runs/cellbridge_v100/gse334503/evaluation/summary.csv"),
           "cellbridge", "E1_primary", float(err[np.ix_(is_test, primary)].mean()))
    out = ROOT / "runs/cellbridge_v100/gse334503_posthoc"
    per = pd.DataFrame({
        "protein": [proteins[i] for i in targets],
        "primary": [bool(i in set(primary)) for i in targets],
        "standardized_mae": err[is_test][:, targets].mean(0),
    })
    _write(out / "per_protein", per, {"cohort": "GSE334503", "method": "cellbridge",
                                      "note": "Recomputed from the sealed predictions. Matches the sealed E1 mean."})
    points = []
    for donor, obs, pred, flag in zip(donors, observed, predicted, is_test):
        if not flag:
            continue
        for name, col in zip(protocol["primary_targets"], primary):
            points.append({"donor": donor, "protein": name, "observed": float(obs[col]), "predicted": float(pred[col])})
    _write(out / "predicted_observed", pd.DataFrame(points),
           {"cohort": "GSE334503", "method": "cellbridge", "role": "test"})
    model = np.load(ROOT / "runs/cellbridge_v100/gse334503/fit/cellbridge_model.npz")
    features = list(model["feature_names"].astype(str))
    hidden = list(model["target_names"].astype(str))
    query = list(model["query_donors"].astype(str))
    anchor_features = [i for i, name in enumerate(features) if name in protocol["anchors"]]
    coef, mz = model["coef"], model["mz"]
    dz = model["query_dz"]
    test_rows = [i for i, donor in enumerate(query) if donor in set(roles["test"])]
    shares = []
    for t, protein in enumerate(hidden):
        parts = (dz[test_rows] - mz) * coef[:, t]
        total = np.abs(parts).sum(1)
        anchor = np.abs(parts[:, anchor_features]).sum(1)
        shares.append({"protein": protein, "anchor_share": float(np.mean(anchor / np.maximum(total, 1e-12))),
                       "primary": protein in protocol["primary_targets"]})
    _write(out / "contribution_shares", pd.DataFrame(shares),
           {"cohort": "GSE334503", "note": "Mean over test donors of the absolute-contribution share."})
    protein = "CD69"
    column = hidden.index(protein)
    cd = pd.DataFrame(points)
    cd = cd[cd.protein == protein].copy()
    cd["abs_error"] = (cd.predicted - cd.observed).abs()
    chosen = cd.iloc[(cd.abs_error - cd.abs_error.median()).abs().argmin()].donor
    row = query.index(chosen)
    parts = (dz[row] - mz) * coef[:, column]
    order = np.argsort(-np.abs(parts))
    top = order[:12]
    waterfall = pd.DataFrame({
        "feature": [features[i] for i in top] + ["other"],
        "contribution": [float(parts[i]) for i in top] + [float(parts[np.setdiff1d(np.arange(len(parts)), top)].sum())],
    })
    waterfall["donor"] = chosen
    waterfall["protein"] = protein
    _write(out / "waterfall_cd69", waterfall, {
        "cohort": "GSE334503", "donor": chosen, "protein": protein,
        "rule": "test donor whose absolute CD69 error is closest to the median",
        "observed": float(cd.loc[cd.donor == chosen, "observed"].iloc[0]),
        "predicted": float(cd.loc[cd.donor == chosen, "predicted"].iloc[0]),
    })
    print("gse posthoc", chosen)


def emtab():
    protocol = _json(ROOT / "configs/protocol_cellbridge_v100_emtab9357.json")
    freeze = _json(ROOT / "runs/cellbridge_v100/emtab9357/freeze.json")
    manifest = _json(ROOT / "runs/cellbridge_v100/emtab9357_prepared/manifest.json")
    groups = freeze["groups"]
    prepared = ROOT / "runs/cellbridge_v100/emtab9357_prepared"
    vault = prepared / "vault"
    columns = _json(vault / "columns.json")["targets"]
    proteins = manifest["proteins"]
    targets = np.asarray(manifest["target_index"], dtype=int)
    primary = np.array([proteins.index(name) for name in protocol["primary_targets_deposited"]])

    def response(patient, full):
        loaded_p = load_arm(prepared / "arms" / f"{patient}__perturbed.npz")
        loaded_c = load_arm(prepared / "arms" / f"{patient}__control.npz")
        if full:
            return loaded_p["protein"].mean(0) - loaded_c["protein"].mean(0)
        change = np.zeros(len(columns))
        for arm, sign in (("control", -1.0), ("perturbed", 1.0)):
            change += sign * np.load(vault / f"{patient}__{arm}.npy").mean(0)
        out = np.full(len(proteins), np.nan)
        for name, value in zip(columns, change):
            out[proteins.index(name)] = value
        return out

    ytr = np.vstack([response(patient, True) for patient in groups["train"]])
    scale = response_scale(ytr)
    queries = list(groups["calibration"]) + list(groups["test"])
    observed = np.vstack([response(patient, False) for patient in queries])
    with np.load(ROOT / "runs/cellbridge_v100/emtab9357/predictions/core.npz") as blob:
        donors = list(blob["donors"].astype(str))
        predicted = blob["cellbridge"].astype(float)
        pred_proteins = list(blob["proteins"].astype(str))
    if donors != queries or pred_proteins != proteins:
        raise SystemExit("E-MTAB prediction order does not match the freeze")
    is_test = np.array([donor in set(groups["test"]) for donor in donors])
    err = np.abs(predicted - observed) / scale
    _check(pd.read_csv(ROOT / "runs/cellbridge_v100/emtab9357/evaluation/summary.csv"),
           "cellbridge", "E1_primary", float(np.nanmean(err[np.ix_(is_test, primary)])))
    out = ROOT / "runs/cellbridge_v100/emtab9357_posthoc"
    per = pd.DataFrame({
        "protein": [proteins[i] for i in targets],
        "primary": [bool(i in set(primary)) for i in targets],
        "standardized_mae": err[is_test][:, targets].mean(0),
    })
    _write(out / "per_protein", per, {"cohort": "E-MTAB-9357", "method": "cellbridge",
                                      "note": "Descriptive. The sealed decision is unchanged."})
    points = []
    for donor, obs, pred, flag in zip(donors, observed, predicted, is_test):
        if not flag:
            continue
        for name in protocol["primary_targets_deposited"]:
            col = proteins.index(name)
            points.append({"donor": donor, "protein": name, "observed": float(obs[col]), "predicted": float(pred[col])})
    _write(out / "predicted_observed", pd.DataFrame(points),
           {"cohort": "E-MTAB-9357", "method": "cellbridge", "role": "test"})
    print("emtab posthoc", len(points))


if __name__ == "__main__":
    gse()
    emtab()
