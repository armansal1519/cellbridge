"""ResponseBridge 1.0 (CellBridge) development benchmark.

Development tasks only: Lawlor arms (previously exposed) and leave-one-donor-out
inside the 11 GSE334503 *training* donors. GSE334503 calibration and test
donors are never loaded here (cells or groups).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "6")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from responsebridge.artifacts import write_json
from responsebridge.benchmark_data import GroupData
from responsebridge.cellbridge import CellBridge
from responsebridge.cellbridge_data import CellTable, build_stats, cognate_gene, feature_layout
from responsebridge.response_baselines import tune_linear
from responsebridge.response_experiment_v050 import _paired

DATA = ROOT.parent / "data/immune_protein"
GSE = ROOT / "runs/gse334503_zenodo_workflow_v040_v2/prepared"
ANCHORS = ["CD38", "ICOS", "PD-1", "HLA-DR", "CD127", "CD27", "CD28", "CD45RO"]
PRIMARY = ["CD25", "CD69"]
ALPHAS = [0.1, 1, 10, 100, None]
SHRINKAGES = [0.25, 0.5, 0.75, 1]

TASKS = {
    "Lawlor_CD3_CD28_T": ("Lawlor_v2", "T", "Baseline", "CD3_CD28"),
    "Lawlor_LPS_Monocyte": ("Lawlor_v2", "Monocyte", "Baseline", "LPS"),
    "Lawlor_LPS_T": ("Lawlor_v2", "T", "Baseline", "LPS"),
    "Lawlor_CD3_CD28_Monocyte": ("Lawlor_v2", "Monocyte", "Baseline", "CD3_CD28"),
    "Lawlor_IgM_IgG_B": ("Lawlor_IgM_v050", "B", "Baseline", "IgM_IgG"),
    "Lawlor_IgM_IgG_T": ("Lawlor_IgM_v050", "T", "Baseline", "IgM_IgG"),
    "Lawlor_IgM_IgG_NK": ("Lawlor_IgM_v050", "NK", "Baseline", "IgM_IgG"),
    "Lawlor_IgM_IgG_Monocyte": ("Lawlor_IgM_v050", "Monocyte", "Baseline", "IgM_IgG"),
    "GSE334503_train_T": ("GSE334503", "T", "D0", "TyphimVi_day7"),
}


def task_sources(dataset):
    if dataset == "GSE334503":
        roles = json.loads((GSE / "cells/manifest.json").read_text())["roles"]
        return GSE / "cells", GSE / "groups", list(roles["train"])
    return DATA / "prepared" / dataset, DATA / "groups" / dataset, None


def paired_rows(groups, lineage, control, perturbed, allowed):
    data = GroupData.load(groups)
    if allowed is not None:
        data = data.subset(np.flatnonzero(data.obs.donor.astype(str).isin(allowed).to_numpy()))
    rows = _paired(data, lineage, control, perturbed, 30)
    return rows, data.proteins.astype(str), data.genes.astype(str)


def baselines_queries(train_rows, xq, aq, anchors, targets):
    """Tuned baselines fit on training rows; queries supply RNA and anchor responses only."""
    from responsebridge.abundance_response import select_abundance_shrinkage
    x = np.stack([r["x"] for r in train_rows]); y = np.stack([r["y"] for r in train_rows])
    d = np.array([r["donor"] for r in train_rows])
    out = {"mean": np.repeat(y.mean(0)[None], len(xq), 0)}
    seconds = {}
    for kind in ("rna", "joint"):
        t0 = time.time()
        model, _, _ = tune_linear(x, y, d, y[:, anchors], targets, kind, ALPHAS, n_hvg=2000)
        out[f"{kind}_ridge"] = model.predict(xq, aq)
        seconds[f"{kind}_ridge_fit_seconds"] = time.time() - t0
    t0 = time.time()
    ax = np.vstack([r["ax"] for r in train_rows]); ay = np.vstack([r["ay"] for r in train_rows])
    ad = np.concatenate([np.repeat(r["donor"], len(r["aw"])) for r in train_rows])
    aw = np.concatenate([r["aw"] for r in train_rows])
    model, _ = select_abundance_shrinkage(x, y, d, anchors, targets, abundance_x=ax, abundance_y=ay,
                                          abundance_donors=ad, abundance_weights=aw, alphas=ALPHAS,
                                          shrinkages=SHRINKAGES, n_hvg=2000)
    out["v040_abundance_shrinkage"] = model.predict(xq, aq, anchors)["prediction"]
    seconds["v040_fit_seconds"] = time.time() - t0
    return out, seconds


def baselines(rows, train, test, anchors, targets):
    xq = np.stack([rows[i]["x"] for i in test]); aq = np.stack([rows[i]["y"][anchors] for i in test])
    out, seconds = baselines_queries([rows[i] for i in train], xq, aq, anchors, targets)
    return out, seconds["v040_fit_seconds"]


def run_task(name, out_root, n_genes, ks, gene_rule="cell", baseline_root=None, mixes=None):
    dataset, lineage, control, perturbed = TASKS[name]
    cells, groups, allowed = task_sources(dataset)
    dest = out_root / name
    reuse = None
    if baseline_root is not None:
        with np.load(Path(baseline_root) / name / "tensors.npz") as v:
            reuse = {k: v[k] for k in v.files if k.startswith("baseline__")}
            reuse_donors = v["donors"].astype(str)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "complete.json").exists():
        print(name, "already complete"); return
    t0 = time.time()
    rows, gproteins, _ = paired_rows(groups, lineage, control, perturbed, allowed)
    table = CellTable(cells, lineage, control, perturbed, donors=allowed)
    if not np.array_equal(table.proteins, gproteins):
        raise ValueError("Cell and group protein rosters differ")
    counts = table.obs.groupby(["donor", "arm"]).size().unstack(fill_value=0)
    eligible = set(counts.index[(counts.min(axis=1) >= 30)].astype(str))
    rows = [r for r in rows if r["donor"] in eligible]
    donors = [r["donor"] for r in rows]
    anchors = table.protein_index(ANCHORS)
    targets = np.array([i for i in range(len(table.proteins)) if i not in set(anchors)])
    primary = np.array([list(targets).index(i) for i in table.protein_index(PRIMARY)])
    forced = [g for g in (cognate_gene(p, table.genes) for p in table.proteins) if g]
    genes = table.select_genes(n_genes, donors=donors, forced=forced, rule=gene_rule)
    stats, dz = build_stats(table, donors, genes, anchors, targets, ks=ks)
    load_time = time.time() - t0
    fs = feature_layout(len(genes), len(anchors))
    model = CellBridge(fs, ks=ks) if mixes is None else CellBridge(fs, ks=ks, mixes=mixes)
    if reuse is not None and list(reuse_donors) != donors:
        raise ValueError("Baseline donors differ from this variant")
    D = len(donors)
    truth = np.stack([r["y"] for r in rows])
    outer = np.empty((D,) + model.grid_shape + (len(targets),), dtype=np.float32)
    inner = np.empty((D,) + model.grid_shape + (D - 1, len(targets)), dtype=np.float32)
    inner_truth = np.empty((D, D - 1, len(targets)))
    base = {}
    timing = {"cellbridge_fit_seconds": [], "v040_fit_seconds": []}
    for j, donor in enumerate(donors):
        train = [i for i in range(D) if i != j]
        tr = [stats[i] for i in train]
        t1 = time.time()
        pred, dy = model.inner_tensor(tr)
        outer[j] = model.path_tensor(tr, dz[donor][None])[..., 0, :]
        timing["cellbridge_fit_seconds"].append(time.time() - t1)
        inner[j], inner_truth[j] = pred, dy
        if reuse is None:
            b, tv = baselines(rows, train, [j], anchors, targets)
            timing["v040_fit_seconds"].append(tv)
            for key, value in b.items():
                base.setdefault(key, np.empty((D, len(table.proteins))))[j] = value[0]
        print(f"{name} fold {j+1}/{D} cellbridge {timing['cellbridge_fit_seconds'][-1]:.1f}s", flush=True)
    base_arrays = reuse if reuse is not None else {f"baseline__{k}": v for k, v in base.items()}
    np.savez_compressed(dest / "tensors.npz", outer=outer, inner=inner, inner_truth=inner_truth,
                        truth=truth, donors=np.array(donors), targets=targets, primary=primary,
                        proteins=table.proteins, cell_truth=np.stack([s.dy for s in stats]),
                        **base_arrays)
    write_json(dest / "task.json", {
        "task": name, "dataset": dataset, "lineage": lineage, "control": control, "perturbed": perturbed,
        "donors": donors, "n_cells": {d: list(map(int, s.n_cells)) for d, s in zip(donors, stats)},
        "n_genes": int(len(genes)), "gene_rule": gene_rule, "genes": table.genes[genes].tolist(), "anchors": ANCHORS,
        "targets": table.proteins[targets].tolist(), "primary": PRIMARY,
        "grid": {"feature_sets": list(fs), "ks": list(ks), "mixes": [list(m) for m in model.mixes],
                 "lambdas": list(model.lambdas)},
        "load_seconds": load_time, **timing,
        "gse_scope": "training donors only" if allowed is not None else "not applicable"})
    write_json(dest / "complete.json", {"status": "complete"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "runs/cellbridge_v100/development"))
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--genes", type=int, default=1000)
    p.add_argument("--ks", type=int, nargs="*", default=[1, 5, 20, 60])
    p.add_argument("--gene-rule", default="cell", choices=["cell", "union"])
    p.add_argument("--baselines-from", default=None)
    a = p.parse_args()
    for name in a.tasks:
        run_task(name, Path(a.out), a.genes, tuple(a.ks), a.gene_rule, a.baselines_from)


if __name__ == "__main__":
    main()
