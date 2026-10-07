"""totalVI comparator for the sealed GSE334503 test (run in .venv_totalvi).

One prespecified fit, no tuning on any outcome: training donors contribute all
ADTs; calibration/test donors contribute RNA and the eight anchor ADTs, and
their remaining ADTs are encoded as missing proteins for their batch (donor).
Imputed protein counts use training-donor batches (transform_batch) and are
aggregated as mean log1p per arm, matching the estimand transform.
"""
from __future__ import annotations

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
import pandas as pd
from scipy import sparse

from responsebridge.cellbridge_data import CellTable, cognate_gene

DRY_RUN = os.environ.get("CELLBRIDGE_DRY_RUN") == "1"
RUN = ROOT / ("runs/cellbridge_v100/gse334503_dryrun" if DRY_RUN else "runs/cellbridge_v100/gse334503")
PREPARED = ROOT / "runs/gse334503_zenodo_workflow_v040_v2/prepared"


def main():
    import anndata as ad
    import scvi
    import torch

    p = json.loads((ROOT / "configs/protocol_cellbridge_v100.json").read_text())
    tv = p["totalvi"]
    if not (RUN / "freeze.json").exists() or (RUN / "prediction_seal.json").exists():
        raise ValueError("totalVI must run after freeze and before the prediction seal")
    out = RUN / "predictions/totalvi.npz"
    if out.exists():
        print("exists"); return
    torch.set_num_threads(6)
    scvi.settings.seed = p["seed"]
    roles = json.loads((PREPARED / "cells/manifest.json").read_text())["roles"]
    if DRY_RUN:
        donors = sorted(roles["train"])
        roles = {"train": donors[:8], "calibration": donors[8:9], "test": donors[9:]}
        tv = {**tv, "max_epochs": 2, "max_training_cells_per_donor_arm": 200, "n_samples": 2}
    train, queries = roles["train"], list(roles["calibration"]) + list(roles["test"])
    t0 = time.time()
    table = CellTable(PREPARED / "cells", p["lineage"], p["control"], p["perturbed"], hidden_donors=queries)
    anchors = table.protein_index(p["anchors"])
    targets = np.array([i for i in range(len(table.proteins)) if i not in set(anchors)])
    forced = [g for g in (cognate_gene(q, table.genes) for q in table.proteins) if g]
    genes = table.select_genes(tv["n_genes"], donors=train, forced=forced)
    raw = sparse.load_npz(PREPARED / "cells/rna_counts.npz").tocsr()
    rng = np.random.default_rng(p["seed"])
    parts, adts, obs = [], [], []
    for donor in train + queries:
        for arm in ("control", "perturbed"):
            local = table.cells(donor, arm)
            parts.append(raw[table.rows[local]][:, genes])
            if donor in train:
                a = table.adt_counts(donor, arm, np.arange(len(table.proteins)), anchors=anchors)
            else:
                a = np.zeros((len(local), len(table.proteins)), dtype=np.float32)
                a[:, anchors] = table.adt_counts(donor, arm, anchors, anchors=anchors)
            adts.append(a)
            obs.append(pd.DataFrame({"donor": donor, "arm": arm, "role": "train" if donor in train else "query"},
                                    index=[f"{donor}|{arm}|{i}" for i in range(len(local))]))
    del raw
    obs = pd.concat(obs)
    adata = ad.AnnData(sparse.vstack(parts).tocsr().astype(np.float32), obs=obs)
    adata.var_names = table.genes[genes]
    adata.obsm["protein"] = pd.DataFrame(np.vstack(adts), index=obs.index, columns=table.proteins)
    load_seconds = time.time() - t0
    keep = []
    for (donor, arm), idx in adata.obs.groupby(["donor", "arm"], observed=True).indices.items():
        keep.extend(rng.choice(idx, size=min(len(idx), tv["max_training_cells_per_donor_arm"]), replace=False))
    fit_data = adata[np.sort(keep)].copy()
    scvi.model.TOTALVI.setup_anndata(fit_data, protein_expression_obsm_key="protein", batch_key="donor")
    model = scvi.model.TOTALVI(fit_data, latent_distribution="normal")
    t1 = time.time()
    model.train(max_epochs=tv["max_epochs"], early_stopping=True, accelerator="cpu",
                batch_size=tv["batch_size"])
    fit_seconds = time.time() - t1
    history = {k: [float(x) for x in v.values.ravel()] for k, v in model.history.items() if k in ("elbo_train", "elbo_validation")}
    pred = np.full((len(queries), len(table.proteins)), np.nan)
    t2 = time.time()
    query = adata[adata.obs.role == "query"].copy()
    _, protein = model.get_normalized_expression(adata=query, n_samples=tv["n_samples"], transform_batch=train,
                                                 include_protein_background=True, sample_protein_mixing=False,
                                                 return_mean=True)
    logp = np.log1p(np.asarray(protein.values, dtype=np.float64))
    for i, donor in enumerate(queries):
        s = ((query.obs.donor == donor) & (query.obs.arm == "perturbed")).to_numpy()
        c = ((query.obs.donor == donor) & (query.obs.arm == "control")).to_numpy()
        pred[i, targets] = (logp[s].mean(0) - logp[c].mean(0))[targets]
    predict_seconds = time.time() - t2
    np.savez_compressed(out, donors=np.array(queries), proteins=table.proteins, totalvi=pred)
    (RUN / "totalvi_log.json").write_text(json.dumps({
        "load_seconds": load_seconds, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "training_cells": int(fit_data.n_obs), "query_cells": int(query.n_obs), "genes": int(len(genes)),
        "epochs_run": len(history.get("elbo_train", [])), "history": history,
        "versions": {"scvi-tools": scvi.__version__, "torch": torch.__version__}}, indent=1))
    print("totalVI done", fit_seconds)


if __name__ == "__main__":
    main()
