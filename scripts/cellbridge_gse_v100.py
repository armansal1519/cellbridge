"""ResponseBridge 1.0 (CellBridge): single independent test on GSE334503.

Stages (each refuses to run out of order):
  freeze   lock protocol, roles and implementation hashes
  fit      fit every method on the 11 training donors; predict calibration and
           test donors from RNA and anchor ADTs only
  seal     hash every prediction file (CellBridge, baselines, external adapters)
  unblind  verify the seal, open test/calibration outcomes once, evaluate
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "6")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from responsebridge.artifacts import complete_stage, sha256, write_json

DRY_RUN = os.environ.get("CELLBRIDGE_DRY_RUN") == "1"
RUN = ROOT / ("runs/cellbridge_v100/gse334503_dryrun" if DRY_RUN else "runs/cellbridge_v100/gse334503")
PROTOCOL = ROOT / "configs/protocol_cellbridge_v100.json"
PREPARED = ROOT / "runs/gse334503_zenodo_workflow_v040_v2/prepared"
V050 = ROOT / "runs/gse334503_response_v050"
LOCKED_SOURCES = [
    "src/responsebridge/cellbridge.py", "src/responsebridge/cellbridge_data.py",
    "src/responsebridge/response_baselines.py", "src/responsebridge/abundance_response.py",
    "src/responsebridge/shrinkage.py", "src/responsebridge/rna.py",
    "src/responsebridge/benchmark_data.py", "src/responsebridge/response_experiment_v050.py",
    "src/responsebridge/reliability.py", "src/responsebridge/response_downstream_v050.py",
    "src/responsebridge/artifacts.py", "scripts/cellbridge_gse_v100.py",
    "scripts/cellbridge_dev_v100.py", "scripts/cellbridge_totalvi_v100.py",
]


def _protocol():
    return json.loads(PROTOCOL.read_text())


def _roles():
    roles = json.loads((PREPARED / "cells/manifest.json").read_text())["roles"]
    if DRY_RUN:
        # Code-path rehearsal inside the training donors only; never the sealed donors.
        train = sorted(roles["train"])
        return {"train": train[:8], "calibration": train[8:9], "test": train[9:]}
    return roles


def _lock_value():
    p = _protocol()
    manifest = json.loads((PREPARED / "cells/manifest.json").read_text())
    return {
        "protocol_sha256": sha256(PROTOCOL),
        "sources_sha256": {s: sha256(ROOT / s) for s in LOCKED_SOURCES},
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pandas", "scikit-learn")},
        "prepared_manifest_sha256": sha256(PREPARED / "cells/manifest.json"),
        "groups_receipt_sha256": sha256(PREPARED / "complete.json"),
        "role_lock_receipt_sha256": manifest["role_lock_receipt_sha256"],
        "roles": _roles(),
        "protocol_version": p["version"],
        "dry_run": DRY_RUN,
    }


def freeze(_):
    p = _protocol()
    roles = _roles()
    for key in ("train", "calibration", "test"):
        if not DRY_RUN and sorted(roles[key]) != sorted(p["roles"][key]):
            raise ValueError(f"Protocol {key} roles differ from the locked manifest")
    value = _lock_value()
    value["frozen_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    value["hidden_outcomes_opened"] = False
    write_json(RUN / "freeze.json", value, immutable=True)
    print("frozen", RUN / "freeze.json")


def verify_freeze():
    frozen = json.loads((RUN / "freeze.json").read_text())
    now = _lock_value()
    for key, value in now.items():
        if frozen[key] != value:
            raise ValueError(f"Frozen {key} changed after freeze")
    return frozen


def fit(_):
    from responsebridge.cellbridge import CellBridge
    from responsebridge.cellbridge_data import CellTable, build_stats, cognate_gene, feature_layout
    from cellbridge_dev_v100 import baselines_queries, paired_rows

    verify_freeze()
    dest = RUN / "fit"
    if (dest / "complete.json").exists():
        print("fit already complete"); return
    p = _protocol()
    roles = _roles()
    train, cal, test = roles["train"], roles["calibration"], roles["test"]
    queries = list(cal) + list(test)
    t0 = time.time()
    table = CellTable(PREPARED / "cells", p["lineage"], p["control"], p["perturbed"], hidden_donors=queries)
    counts = table.obs.groupby(["donor", "arm"]).size().unstack(fill_value=0)
    ineligible = sorted(d for d in train + queries if d not in counts.index or counts.loc[d].min() < p["minimum_cells_per_arm"])
    if ineligible:
        raise ValueError(f"Locked donors below the per-arm cell minimum: {ineligible}")
    anchors = table.protein_index(p["anchors"])
    targets = np.array([i for i in range(len(table.proteins)) if i not in set(anchors)])
    forced = [g for g in (cognate_gene(q, table.genes) for q in table.proteins) if g]
    genes = table.select_genes(p["cellbridge"]["n_genes"], donors=train, forced=forced, rule=p["cellbridge"]["gene_rule"])
    ks = tuple(p["cellbridge"]["metacell_sizes"])
    stats, _ = build_stats(table, train, genes, anchors, targets, ks=ks, seed=p["seed"])
    _, dzq = build_stats(table, queries, genes, anchors, targets, ks=ks, with_outcomes=False)
    load_seconds = time.time() - t0
    fs = feature_layout(len(genes), len(anchors))
    grid = dict(ks=ks, mixes=tuple(tuple(m) for m in p["cellbridge"]["mixes"]),
                lambdas=tuple(p["cellbridge"]["lambdas"]), strategy=p["cellbridge"]["strategy"],
                top=p["cellbridge"]["top"])
    zq = np.stack([dzq[d] for d in queries])
    P = len(table.proteins)
    preds = {}
    timing = {"load_seconds": load_seconds}

    def full(values):
        out = np.full((len(queries), P), np.nan)
        out[:, targets] = values
        return out

    t1 = time.time()
    model = CellBridge(fs, **grid).fit(stats)
    timing["cellbridge_fit_seconds"] = time.time() - t1
    preds["cellbridge"] = full(model.predict(zq))
    t1 = time.time()
    rna_model = CellBridge({"rna": fs["rna"]}, **grid).fit(stats)
    timing["cellbridge_rna_only_fit_seconds"] = time.time() - t1
    preds["cellbridge_rna_only"] = full(rna_model.predict(zq))

    rows, gproteins, _ = paired_rows(PREPARED / "groups", p["lineage"], p["control"], p["perturbed"], train)
    if not np.array_equal(gproteins, table.proteins):
        raise ValueError("Group and cell protein rosters differ")
    if [r["donor"] for r in rows] != sorted(train):
        raise ValueError("Training paired rows do not match the locked training donors")
    xq = np.stack([table.pseudobulk_rna_response(d) for d in queries])
    aq = np.stack([dzq[d][len(genes) + 2:] for d in queries])
    t1 = time.time()
    base, seconds = baselines_queries(rows, xq, aq, anchors, targets)
    timing.update(seconds)
    for key, value in base.items():
        preds[key] = np.where(np.isin(np.arange(P), targets)[None], value, np.nan)

    dest.mkdir(parents=True, exist_ok=True)
    pred_dir = RUN / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / "core.npz", donors=np.array(queries), proteins=table.proteins,
                        roles=np.array(["calibration"] * len(cal) + ["test"] * len(test)), **preds)
    feature_names = np.array(list(table.genes[genes]) + ["log_rna_umis", "log_rna_genes"] + list(p["anchors"]))
    np.savez_compressed(dest / "cellbridge_model.npz", coef=model.coef, mz=model.mz, my=model.my,
                        feature_names=feature_names, target_names=table.proteins[targets],
                        rna_only_coef=rna_model.coef, query_dz=zq, query_donors=np.array(queries))
    write_json(dest / "selection.json", {"cellbridge": model.selected, "cellbridge_rna_only": rna_model.selected,
                                         "targets": table.proteins[targets].tolist()})
    write_json(dest / "fit.json", {
        "training_donors": sorted(train), "query_donors": queries, "n_genes": int(len(genes)),
        "n_cells": {d: list(map(int, s.n_cells)) for d, s in zip(train, stats)},
        "query_cells": {d: [int(len(table.cells(d, "control"))), int(len(table.cells(d, "perturbed")))] for d in queries},
        "timing_seconds": timing, "methods": sorted(preds),
        "outcome_access": "Target ADTs read only for training donors; queries use RNA and anchor ADTs"})
    complete_stage(dest)
    print(json.dumps(timing, indent=1))


def seal(_):
    verify_freeze()
    if (RUN / "prediction_seal.json").exists():
        raise FileExistsError("Predictions already sealed")
    if not (RUN / "fit/complete.json").exists():
        raise ValueError("Fit stage incomplete")
    files = sorted((RUN / "predictions").glob("*.npz"))
    value = {"files": {f.name: sha256(f) for f in files},
             "fit_receipt_sha256": sha256(RUN / "fit/complete.json"),
             "freeze_sha256": sha256(RUN / "freeze.json"),
             "v050_prediction_seal_sha256": sha256(V050 / "prediction_seal.json"),
             "v050_predictions_sha256": sha256(V050 / "fold_00/predictions.npz"),
             "sealed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "hidden_outcomes_opened": False}
    write_json(RUN / "prediction_seal.json", value, immutable=True)
    print(json.dumps(value, indent=1))


def _verify_seal():
    seal_value = json.loads((RUN / "prediction_seal.json").read_text())
    present = sorted(f.name for f in (RUN / "predictions").glob("*.npz"))
    if present != sorted(seal_value["files"]):
        raise ValueError("Prediction files differ from the seal")
    for name, digest in seal_value["files"].items():
        if sha256(RUN / "predictions" / name) != digest:
            raise ValueError(f"Sealed prediction {name} changed")
    if sha256(V050 / "fold_00/predictions.npz") != seal_value["v050_predictions_sha256"]:
        raise ValueError("v0.5 sealed predictions changed")
    return seal_value


def _load_predictions():
    methods, donors, proteins = {}, None, None
    for f in sorted((RUN / "predictions").glob("*.npz")):
        with np.load(f, allow_pickle=False) as v:
            d = v["donors"].astype(str)
            pr = v["proteins"].astype(str)
            if donors is None:
                donors, proteins = d, pr
            if not (np.array_equal(d, donors) and np.array_equal(pr, proteins)):
                raise ValueError(f"{f.name} donor/protein order differs")
            for k in v.files:
                if k not in ("donors", "proteins", "roles"):
                    methods[k] = v[k].astype(float)
    folds = json.loads((V050 / "folds.json").read_text())[0]
    with np.load(V050 / "fold_00/predictions.npz", allow_pickle=False) as v:
        order = {d: i for i, d in enumerate(folds["test"])}
        for k in v.files:
            arr = np.full((len(donors), len(proteins)), np.nan)
            for i, d in enumerate(donors):
                if d in order:
                    arr[i] = v[k][order[d]]
            methods[f"v050_{k}"] = arr
    return methods, donors, proteins


def unblind(_):
    from responsebridge.reliability import paired_gain_bootstrap
    from responsebridge.response_baselines import response_scale
    from responsebridge.response_downstream_v050 import sign_flip_p
    from cellbridge_dev_v100 import paired_rows

    frozen = verify_freeze()
    seal_value = _verify_seal()
    dest = RUN / "evaluation"
    if (dest / "complete.json").exists():
        print("already evaluated"); return
    p = _protocol()
    roles = _roles()
    methods, donors, proteins = _load_predictions()
    if "cellbridge" in methods and "v040_abundance_shrinkage" in methods:
        methods["ensemble_cellbridge_v040"] = 0.5 * (methods["cellbridge"] + methods["v040_abundance_shrinkage"])
    train_rows, _, _ = paired_rows(PREPARED / "groups", p["lineage"], p["control"], p["perturbed"], roles["train"])
    ytr = np.stack([r["y"] for r in train_rows])
    scale = response_scale(ytr)
    mu = ytr.mean(0)
    query_rows, _, _ = paired_rows(PREPARED / "groups", p["lineage"], p["control"], p["perturbed"],
                                   list(roles["calibration"]) + list(roles["test"]))
    truth = {r["donor"]: r["y"] for r in query_rows}
    y = np.stack([truth[d] for d in donors])
    anchors = [list(proteins).index(a) for a in p["anchors"]]
    targets = np.array([i for i in range(len(proteins)) if i not in set(anchors)])
    primary = np.array([list(proteins).index(t) for t in p["primary_targets"]])
    is_test = np.isin(donors, roles["test"])
    is_cal = np.isin(donors, roles["calibration"])
    endpoints = {"E1_primary": primary, "E2_all_hidden": targets}
    err = {m: np.abs(v - y) / scale for m, v in methods.items()}
    rows = []
    loss = {}
    for m, e in err.items():
        for ep, cols in endpoints.items():
            per = e[np.ix_(is_test, cols)].mean(1)
            loss[(m, ep)] = per
            sq = ((methods[m] - y) ** 2)[np.ix_(is_test, cols)].sum()
            tsq = ((mu - y) ** 2)[np.ix_(is_test, cols)].sum()
            rows.append({"method": m, "endpoint": ep, "test_donors": int(np.isfinite(per).sum()),
                         "standardized_mae": float(np.nanmean(per)) if np.isfinite(per).any() else None,
                         "complete": bool(np.isfinite(per).all()),
                         "skill_vs_training_mean": float(1 - sq / tsq) if np.isfinite(per).all() else None})
    summary = pd.DataFrame(rows).sort_values(["endpoint", "standardized_mae"])
    dest.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dest / "summary.csv", index=False)
    meta = {"freeze": sha256(RUN / "freeze.json"), "seal": sha256(RUN / "prediction_seal.json")}
    tests = []
    stop = False
    for h in p["hypotheses"]:
        comparator, ep = h["comparator"], h["endpoint"]
        record = {"id": h["id"], "endpoint": ep, "model": "cellbridge", "comparator": comparator}
        if (comparator, ep) not in loss or not np.isfinite(loss[(comparator, ep)]).all():
            record["status"] = "comparator_not_available_removed_by_protocol"
            tests.append(record); continue
        b, mdl = loss[(comparator, ep)], loss[("cellbridge", ep)]
        boot = paired_gain_bootstrap(b, mdl, donors[is_test], n_bootstrap=10000, seed=p["seed"],
                                     independent_test=True, models_frozen=True, model_metadata=meta)
        pval = sign_flip_p(b - mdl)
        rejected = (not stop) and pval is not None and pval < p["alpha"] and float(np.mean(b - mdl)) > 0
        record.update({"mean_loss_comparator": float(b.mean()), "mean_loss_cellbridge": float(mdl.mean()),
                       "mean_difference": float(np.mean(b - mdl)), "relative_gain": boot["relative_gain"],
                       "difference_ci95": boot["difference_ci95"], "sign_flip_p_two_sided": pval,
                       "donor_wins": int(np.sum(mdl < b)), "n_donors": int(len(b)),
                       "tested_in_sequence": not stop, "rejected_null": bool(rejected)})
        if not stop and not rejected:
            stop = True
        tests.append(record)
    noninf = []
    for ep in endpoints:
        if ("v040_abundance_shrinkage", ep) in loss:
            b, mdl = loss[("v040_abundance_shrinkage", ep)], loss[("cellbridge", ep)]
            boot = paired_gain_bootstrap(b, mdl, donors[is_test], n_bootstrap=10000, seed=p["seed"],
                                         independent_test=True, models_frozen=True, model_metadata=meta)
            upper = -boot["difference_ci95"][0]
            noninf.append({"endpoint": ep, "margin": p["noninferiority_margin_vs_v040"],
                           "mean_excess_loss_cellbridge": float(np.mean(mdl - b)),
                           "upper_ci95_excess_loss": float(upper),
                           "noninferior": bool(upper < p["noninferiority_margin_vs_v040"])})
    per_protein = []
    for m in methods:
        if not np.isfinite(err[m][np.ix_(is_test, targets)]).all():
            continue
        for t in targets:
            per_protein.append({"method": m, "protein": proteins[t], "primary": bool(t in primary),
                                "standardized_mae": float(err[m][is_test, t].mean())})
    per_protein = pd.DataFrame(per_protein)
    per_protein.to_csv(dest / "per_protein.csv", index=False)
    wins = {}
    cb = per_protein[per_protein.method == "cellbridge"].set_index("protein").standardized_mae
    for m in per_protein.method.unique():
        if m == "cellbridge":
            continue
        other = per_protein[per_protein.method == m].set_index("protein").standardized_mae
        wins[m] = float((cb < other.reindex(cb.index)).mean())
    alpha = p["conformal_alpha"]
    conformal = {}
    for m in ("cellbridge", "v040_abundance_shrinkage", "rna_ridge", "joint_ridge"):
        if m not in err:
            continue
        scores = err[m][np.ix_(is_cal, targets)].ravel()
        n = len(scores)
        q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
        te = err[m][np.ix_(is_test, targets)]
        conformal[m] = {"pooled_standardized_quantile": q, "test_coverage_all_hidden": float((te <= q).mean()),
                        "test_coverage_primary": float((err[m][np.ix_(is_test, primary)] <= q).mean()),
                        "standardized_half_width": q, "n_calibration_scores": n}
    donors_frame = pd.DataFrame([{"donor": d, "role": "test" if t else "calibration", "method": m, "endpoint": ep,
                                  "loss": float(err[m][i, cols].mean())}
                                 for m in err for ep, cols in endpoints.items()
                                 for i, (d, t) in enumerate(zip(donors, is_test))])
    donors_frame.to_csv(dest / "donor_losses.csv", index=False)
    decision = {
        "version": p["version"], "hypotheses": tests, "noninferiority_vs_v040": noninf,
        "per_protein_win_fraction_vs": wins, "conformal": conformal,
        "unblinded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "freeze_frozen_at": frozen["frozen_at"], "sealed_at": seal_value["sealed_at"],
        "primary_claim_supported": bool(tests and all(t.get("rejected_null") for t in tests
                                                      if t["id"] in p["primary_claim_hypotheses"])),
    }
    write_json(dest / "decision.json", decision)
    complete_stage(dest)
    print(summary.to_string())
    print(json.dumps(decision, indent=1))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=["freeze", "fit", "seal", "unblind"])
    args = a.parse_args()
    {"freeze": freeze, "fit": fit, "seal": seal, "unblind": unblind}[args.stage](args)


if __name__ == "__main__":
    main()
