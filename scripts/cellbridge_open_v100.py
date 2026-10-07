"""Open-data CellBridge checks. Does not read GSE334503 calibration or test outcomes.

Writes runs/cellbridge_v100/open_ablations/. The development tensors stay read-only;
subgrid numbers are copied out of the locked analysis summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cellbridge_dev_v100 import ANCHORS, PRIMARY, TASKS, paired_rows, task_sources
from responsebridge.cellbridge import CellBridge
from responsebridge.cellbridge_data import CellTable, build_stats, cognate_gene
from responsebridge.cellbridge_open import (OPEN_LAMBDAS, TAU0_MIXES, conformal_coverage_rate,
                                            scaling_point, selected_lodo, simulate_metacell_attenuation)

OUT = ROOT / "runs/cellbridge_v100/open_ablations"
GENE_SIZES = (100, 250, 1000, 2000)
PANEL_SIZES = (0, 1, 2, 4, 8)
INTEREST = (
    "cellbridge[shared_top05]",
    "cellbridge[shared_top05|rna_only]",
    "cellbridge[shared_top05|anchors_required]",
    "cellbridge[shared_top05|rho0]",
    "cellbridge[shared_top05|rho_max]",
    "cellbridge[shared_top05|k1]",
    "joint_ridge",
    "rna_ridge",
    "mean",
    "v040_abundance_shrinkage",
)


def subgrid_table():
    frame = pd.read_csv(ROOT / "runs/cellbridge_v100/analysis/development/summary.csv")
    kept = frame[frame.method.isin(INTEREST)]
    means = kept.groupby("method")[["primary_mae", "all_mae"]].mean().reset_index()
    means.to_csv(OUT / "subgrid_means.csv", index=False)
    kept.to_csv(OUT / "subgrid_by_task.csv", index=False)
    return means


def _load(name, n_genes):
    dataset, lineage, control, perturbed = TASKS[name]
    cells, groups, allowed = task_sources(dataset)
    rows, _, _ = paired_rows(groups, lineage, control, perturbed, allowed)
    table = CellTable(cells, lineage, control, perturbed, donors=allowed,
                      hidden_donors=() if allowed is None else [])
    counts = table.obs.groupby(["donor", "arm"]).size().unstack(fill_value=0)
    eligible = set(counts.index[(counts.min(axis=1) >= 30)].astype(str))
    rows = [r for r in rows if r["donor"] in eligible]
    donors = [r["donor"] for r in rows]
    anchors = table.protein_index(ANCHORS)
    targets = np.array([i for i in range(len(table.proteins)) if i not in set(anchors)])
    primary = np.array([list(targets).index(i) for i in table.protein_index(PRIMARY)])
    forced = [g for g in (cognate_gene(p, table.genes) for p in table.proteins) if g]
    genes = table.select_genes(n_genes, donors=donors, forced=forced, rule="cell")
    stats, _ = build_stats(table, donors, genes, anchors, targets, ks=(1,))
    names = table.genes[genes].tolist() + ["log1p_umis", "log1p_genes"] + list(ANCHORS)
    return stats, primary, names, table, donors, targets


def _layouts(n_genes, anchor_positions):
    rna = np.arange(n_genes + 2)
    extra = [n_genes + 2 + j for j in anchor_positions]
    return {"panel": np.array(list(rna) + extra, dtype=int)}


def panel_and_genes():
    gene_rows, panel_rows, anchor_rows = [], [], []
    for name in TASKS:
        print(f"open {name}", flush=True)
        stats, primary, names, table, donors, targets = _load(name, 1000)
        n_genes = len(names) - 2 - len(ANCHORS)
        base = selected_lodo(stats, _layouts(n_genes, range(len(ANCHORS))), primary=primary)
        rna = selected_lodo(stats, {"panel": np.arange(n_genes + 2)}, primary=primary)
        only = selected_lodo(stats, {"panel": np.arange(n_genes + 2, n_genes + 2 + len(ANCHORS))}, primary=primary)
        for label, result in (("rna_plus_anchors", base), ("rna_only", rna), ("anchors_only", only)):
            anchor_rows.append({"task": name, "features": label, **result})
        print(f"  features {anchor_rows[-3:]}", flush=True)

        chosen = []
        remaining = list(range(len(ANCHORS)))
        recorded = {0: ("", rna)}
        while remaining and len(chosen) < max(PANEL_SIZES):
            trials = []
            for j in remaining:
                trial = chosen + [j]
                score = selected_lodo(stats, _layouts(n_genes, trial), primary=primary)
                trials.append((score["all_mae"], j, score))
            trials.sort(key=lambda item: item[0])
            _, best, score = trials[0]
            chosen.append(best)
            remaining.remove(best)
            if len(chosen) in PANEL_SIZES:
                recorded[len(chosen)] = (",".join(ANCHORS[j] for j in chosen), score)
            print(f"  panel {len(chosen)} {ANCHORS[best]} {score['all_mae']:.3f}", flush=True)
        for size in PANEL_SIZES:
            label, score = recorded[size]
            panel_rows.append({"task": name, "n_anchors": size, "anchors": label, **score})

        for n in GENE_SIZES:
            if n == 1000:
                gene_rows.append({"task": name, "n_genes": n_genes, **base})
                continue
            stats_n, primary_n, names_n, *_ = _load(name, n)
            n_g = len(names_n) - 2 - len(ANCHORS)
            score = selected_lodo(stats_n, _layouts(n_g, range(len(ANCHORS))), primary=primary_n)
            gene_rows.append({"task": name, "n_genes": n_g, **score})
            print(f"  genes {n_g} {score['all_mae']:.3f}", flush=True)
            del stats_n
        del stats, table
    pd.DataFrame(gene_rows).to_csv(OUT / "gene_curve.csv", index=False)
    pd.DataFrame(panel_rows).to_csv(OUT / "panel_curve.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(OUT / "anchor_only.csv", index=False)


def attribution(n_bootstrap=20):
    """Bootstrap training donors of the open GSE training task. Test donors are not loaded."""
    name = "GSE334503_train_T"
    stats, primary, names, table, donors, targets = _load(name, 1000)
    wanted = ["CD25", "CD69", "CD154", "CD137"]
    target_names = table.proteins[targets]
    present = [p for p in wanted if p in set(target_names)]
    columns = {p: int(np.where(target_names == p)[0][0]) for p in present}
    n_genes = len(names) - 2 - len(ANCHORS)
    layout = _layouts(n_genes, range(len(ANCHORS)))
    rng = np.random.default_rng(20261007)
    ranks = []
    dz = np.stack([s.dz for s in stats])
    for b in range(n_bootstrap):
        draw = rng.integers(0, len(stats), len(stats))
        sample = [stats[i] for i in draw]
        # Rename so a repeated donor does not collide in logs; the solver uses positions.
        model = CellBridge(layout, ks=(1,), mixes=TAU0_MIXES, lambdas=OPEN_LAMBDAS,
                           strategy="shared_top", top=0.05).fit(sample)
        query = dz.mean(0)
        for protein, col in columns.items():
            parts = np.abs(model.contributions(query, col)[0])
            order = np.argsort(-parts)
            top = [names[i] for i in order[:5]]
            anchor_share = float(parts[-len(ANCHORS):].sum() / parts.sum())
            ranks.append({"bootstrap": b, "protein": protein, "top": ",".join(top),
                          "anchor_share": anchor_share, "top_is_anchor": top[0] in ANCHORS})
        print(f"  bootstrap {b+1}/{n_bootstrap}", flush=True)
    frame = pd.DataFrame(ranks)
    frame.to_csv(OUT / "attribution_bootstrap.csv", index=False)
    summary = frame.groupby("protein").agg(anchor_share=("anchor_share", "mean"),
                                           top_is_anchor=("top_is_anchor", "mean")).reset_index()
    summary.to_csv(OUT / "attribution_summary.csv", index=False)


def synthetic():
    meta = simulate_metacell_attenuation()
    cover = conformal_coverage_rate()
    points = []
    for n in (10_000, 100_000, 1_000_000):
        print(f"scale {n}", flush=True)
        points.append(scaling_point(n))
    points.append({
        "method": "totalVI",
        "n_cells_training": 89906,
        "fit_seconds": 2349.99,
        "source": "runs/cellbridge_v100/gse334503/totalvi_log.json",
        "note": "One prespecified fit on the real training cells. Not re-run.",
    })
    (OUT / "simulation.json").write_text(json.dumps({"metacells": meta, "conformal": cover, "scaling": points}, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subgrid_table()
    synthetic()
    panel_and_genes()
    attribution()
    (OUT / "complete.json").write_text(json.dumps({
        "status": "complete",
        "exploratory": True,
        "scipenn": "not run; it is not installed, and installing it into the locked environment would change the results-lock freeze",
        "grid": "k=1, tau=0, 8 lambdas. The full-grid subgrid table is subgrid_means.csv.",
        "seconds": time.time() - t0,
    }, indent=2) + "\n")
    print("open ablations done", flush=True)


if __name__ == "__main__":
    main()
