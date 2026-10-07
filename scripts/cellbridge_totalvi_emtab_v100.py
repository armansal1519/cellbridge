"""totalVI comparator for the sealed E-MTAB-9357 run (use .venv_totalvi).

One prespecified fit. Training patients contribute every protein. Calibration
and test patients contribute RNA and the eight anchors; their other proteins
are zeros. Inputs are CPM, the expm1 of the deposited log1p matrices. Query
predictions are the mean log1p imputed protein, AC minus BL.
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

from responsebridge.emtab_adapter import cognate_for_deposited, load_arm, select_genes_from_moments

DRY = os.environ.get("CELLBRIDGE_DRY_RUN") == "1"
RUN = ROOT / "runs/cellbridge_v100" / ("emtab9357_dryrun" if DRY else "emtab9357")
PREPARED = ROOT / "runs/cellbridge_v100/emtab9357_prepared"


def _arm(patient, arm):
    return load_arm(PREPARED / "arms" / f"{patient}__{arm}.npz")


def _counts(cpm):
    """Integer counts per 10^4. Raw CPM made the totalVI encoder non-finite."""
    rounded = np.rint(np.maximum(np.asarray(cpm, dtype=np.float64) / 100.0, 0))
    return rounded.astype(np.float32)


def _genes(data, train, n_genes):
    names = data["genes"]
    total = np.zeros(len(names))
    total_sq = np.zeros(len(names))
    detected = np.zeros(len(names))
    n_cells = 0
    for patient in train:
        for arm in ("control", "perturbed"):
            rna = _arm(patient, arm)["rna"]
            total += np.asarray(rna.sum(axis=0)).ravel()
            squared = rna.copy()
            squared.data = np.square(squared.data.astype(np.float64))
            total_sq += np.asarray(squared.sum(axis=0)).ravel()
            detected += np.asarray((rna > 0).sum(axis=0)).ravel()
            n_cells += rna.shape[0]
    forced = [names.index(gene) for gene in (cognate_for_deposited(name, names) for name in data["proteins"]) if gene]
    return select_genes_from_moments(total, total_sq, n_cells, detected, n_genes, forced)


def main():
    import anndata as ad
    import scvi
    import torch

    spec = json.loads((ROOT / "configs/protocol_cellbridge_v100_emtab9357.json").read_text())
    tv = dict(spec["totalvi"])
    if DRY:
        tv.update(max_epochs=2, max_training_cells_per_donor_arm=200, n_samples=2)
    if not (RUN / "freeze.json").exists() or (RUN / "prediction_seal.json").exists():
        raise SystemExit("totalVI must run after freeze and before the seal")
    freeze = json.loads((RUN / "freeze.json").read_text())
    if freeze.get("dry_run") != DRY:
        raise SystemExit("Dry-run mode does not match the freeze")
    out = RUN / "predictions" / "totalvi.npz"
    if out.exists():
        print("exists")
        return
    torch.set_num_threads(6)
    scvi.settings.seed = spec["seed"]
    data = json.loads((PREPARED / "manifest.json").read_text())
    groups = freeze["groups"]
    train, queries = groups["train"], list(groups["calibration"]) + list(groups["test"])
    proteins = data["proteins"]
    anchors = np.asarray(data["anchor_index"], dtype=int)
    targets = np.asarray(data["target_index"], dtype=int)
    genes = _genes(data, train, tv["n_genes"])
    rng = np.random.default_rng(spec["seed"])
    t0 = time.time()
    parts, adts, obs_rows, keep_local = [], [], [], []
    cursor = 0
    for donor in train + queries:
        for arm in ("control", "perturbed"):
            loaded = _arm(donor, arm)
            rna = loaded["rna"][:, genes].copy()
            rna.data = _counts(np.expm1(rna.data.astype(np.float64)))
            rna.eliminate_zeros()
            if loaded["full_protein"]:
                protein = _counts(np.expm1(np.asarray(loaded["protein"], dtype=np.float64)))
            else:
                protein = np.zeros((rna.shape[0], len(proteins)), dtype=np.float32)
                protein[:, anchors] = _counts(np.expm1(np.asarray(loaded["protein_anchors"], dtype=np.float64)))
            n = rna.shape[0]
            parts.append(rna.tocsr())
            adts.append(protein)
            obs_rows.append(pd.DataFrame({
                "donor": donor, "arm": arm, "role": "train" if donor in set(train) else "query",
            }, index=[f"{donor}|{arm}|{i}" for i in range(n)]))
            chosen = rng.choice(n, size=min(n, tv["max_training_cells_per_donor_arm"]), replace=False)
            keep_local.extend((cursor + int(i) for i in chosen))
            cursor += n
            del loaded
    obs = pd.concat(obs_rows)
    adata = ad.AnnData(sparse.vstack(parts).tocsr().astype(np.float32), obs=obs)
    adata.var_names = [data["genes"][i] for i in genes]
    adata.obsm["protein"] = pd.DataFrame(np.vstack(adts), index=obs.index, columns=proteins)
    del parts, adts
    fit_data = adata[np.sort(np.asarray(keep_local))].copy()
    library = np.asarray(fit_data.X.sum(1)).ravel()
    keep_cells = np.isfinite(library) & (library > 0)
    protein_ok = np.isfinite(fit_data.obsm["protein"].to_numpy()).all(1)
    fit_data = fit_data[keep_cells & protein_ok].copy()
    if fit_data.n_obs < 100:
        raise SystemExit("totalVI training matrix has too few finite cells")
    print(f"totalVI cells {fit_data.n_obs} dropped {int((~keep_cells).sum())}", flush=True)
    epochs = int(os.environ.get("CELLBRIDGE_TOTALVI_EPOCHS", tv["max_epochs"]))
    # scvi's default 4e-3 made the encoder non-finite during epoch 1, and 1e-3 during epoch 2.
    # 4e-4 stayed finite across an 8-epoch probe. Chosen before unblinding, from training loss only.
    learning_rate = float(os.environ.get("CELLBRIDGE_TOTALVI_LR", "0.0004"))
    scvi.model.TOTALVI.setup_anndata(fit_data, protein_expression_obsm_key="protein", batch_key="donor")
    model = scvi.model.TOTALVI(fit_data, latent_distribution="normal")
    t1 = time.time()
    model.train(
        max_epochs=epochs, early_stopping=epochs > 2, accelerator="cpu",
        batch_size=tv["batch_size"], lr=learning_rate,
    )
    fit_seconds = time.time() - t1
    history = {k: [float(x) for x in v.values.ravel()] for k, v in model.history.items()
               if k in ("elbo_train", "elbo_validation", "train_loss_epoch")}
    if any(not np.isfinite(value) for series in history.values() for value in series):
        raise SystemExit("totalVI training loss was non-finite; predictions were not written")
    pred = np.full((len(queries), len(proteins)), np.nan)
    t2 = time.time()
    for i, donor in enumerate(queries):
        query = adata[adata.obs.donor == donor].copy()
        _, protein = model.get_normalized_expression(
            adata=query, n_samples=tv["n_samples"], transform_batch=train,
            include_protein_background=True, sample_protein_mixing=False, return_mean=True,
        )
        logp = np.log1p(np.asarray(protein.values, dtype=np.float64))
        stimulated = (query.obs.arm == "perturbed").to_numpy()
        control = (query.obs.arm == "control").to_numpy()
        pred[i, targets] = (logp[stimulated].mean(0) - logp[control].mean(0))[targets]
        del query
    np.savez_compressed(out, donors=np.asarray(queries), proteins=np.asarray(proteins), totalvi=pred)
    (RUN / "totalvi_log.json").write_text(json.dumps({
        "load_seconds": time.time() - t0, "fit_seconds": fit_seconds, "predict_seconds": time.time() - t2,
        "training_cells": int(fit_data.n_obs), "genes": int(len(genes)),
        "epochs_run": len(history.get("elbo_train", [])), "history": history, "dry_run": DRY,
        "input": "nearest integer of expm1(deposited log1p CPM) / 100; unscaled CPM made epoch 1 non-finite",
        "learning_rate": learning_rate,
        "learning_rate_note": "Below scvi-tools' default 4e-3. That rate, and 1e-3, made the encoder non-finite within two epochs. 4e-4 remained finite on an 8-epoch probe. Set before unblinding from training stability only.",
        "versions": {"scvi-tools": scvi.__version__, "torch": torch.__version__},
    }, indent=1))
    print("totalVI done", round(fit_seconds, 1))


if __name__ == "__main__":
    main()
