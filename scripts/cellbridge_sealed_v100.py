#!/usr/bin/env python3
"""Frozen CellBridge v1.0 on E-MTAB-9357.

    python scripts/cellbridge_sealed_v100.py freeze|fit|seal|unblind

CELLBRIDGE_DRY_RUN=1 splits the eligible training patients and writes
runs/cellbridge_v100/emtab9357_dryrun. It does not open the outcome vault.
The real run writes runs/cellbridge_v100/emtab9357 and opens the vault once.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from responsebridge.artifacts import sha256, write_json  # noqa: E402
from responsebridge.cellbridge import CellBridge, donor_stats  # noqa: E402
from responsebridge.cellbridge_data import feature_layout  # noqa: E402
from responsebridge.cellbridge_eval import fixed_sequence  # noqa: E402
from responsebridge.emtab_adapter import (  # noqa: E402
    assert_gse_frozen_sources, cognate_for_deposited, load_arm, select_genes_from_moments,
)
from responsebridge.reliability import paired_gain_bootstrap  # noqa: E402
from responsebridge.response_baselines import response_scale  # noqa: E402
from cellbridge_dev_v100 import baselines_queries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DRY = os.environ.get("CELLBRIDGE_DRY_RUN") == "1"
RUN = ROOT / "runs/cellbridge_v100" / ("emtab9357_dryrun" if DRY else "emtab9357")
PREPARED = ROOT / "runs/cellbridge_v100/emtab9357_prepared"
PROTOCOL = ROOT / "configs/protocol_cellbridge_v100_emtab9357.json"
PREREG = ROOT / "docs/prereg_emtab9357.md"
VAULT = PREPARED / "vault"


def _json(path):
    return json.loads(Path(path).read_text())


def protocol():
    return _json(PROTOCOL)


def manifest():
    path = PREPARED / "manifest.json"
    if not path.exists():
        raise SystemExit("E-MTAB-9357 is not prepared")
    return _json(path)


def roles_of(data):
    eligible = data["eligible"]
    spec = protocol()
    for key in ("train", "calibration", "test"):
        extra = set(eligible[key]) - set(spec["roles_metadata"][key])
        if extra:
            raise SystemExit(f"Eligible {key} patients are outside the metadata lock: {sorted(extra)[:5]}")
    if DRY:
        names = sorted(eligible["train"])
        n_train, n_cal = spec["dry_run_split"]["train"], spec["dry_run_split"]["calibration"]
        if len(names) <= n_train + n_cal:
            raise SystemExit("Too few eligible training patients for the dry-run split")
        return {"train": names[:n_train], "calibration": names[n_train:n_train + n_cal],
                "test": names[n_train + n_cal:]}
    return {key: list(eligible[key]) for key in ("train", "calibration", "test")}


def _check():
    receipt = _json(RUN / "freeze.json")
    if receipt.get("dry_run") != DRY:
        raise SystemExit("This run directory belongs to a different dry-run mode")
    if sha256(PROTOCOL) != receipt["protocol_sha256"] or sha256(PREREG) != receipt["preregistration_sha256"]:
        raise SystemExit("Protocol or preregistration changed after the freeze")
    if sha256(PREPARED / "manifest.json") != receipt["manifest_sha256"]:
        raise SystemExit("Prepared manifest changed after the freeze")
    for rel, digest in receipt["sources_sha256"].items():
        if sha256(ROOT / rel) != digest:
            raise SystemExit(f"Frozen source changed: {rel}")
    return receipt


def freeze():
    if (RUN / "freeze.json").exists():
        raise SystemExit(f"Refusing to replace {RUN / 'freeze.json'}")
    sources = assert_gse_frozen_sources(ROOT)
    data = manifest()
    if data.get("protocol_sha256") != sha256(PROTOCOL) or data.get("prereg_sha256") != sha256(PREREG):
        raise SystemExit("Protocol or preregistration changed after preparation")
    groups = roles_of(data)
    spec = protocol()
    write_json(RUN / "freeze.json", {
        "status": "frozen", "version": spec["version"], "dry_run": DRY,
        "cohort": "E-MTAB-9357", "seed": spec["seed"],
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL),
        "preregistration_sha256": sha256(PREREG),
        "preregistration_osf_url": spec["preregistration"]["osf_url"],
        "manifest_sha256": sha256(PREPARED / "manifest.json"),
        "sources_sha256": sources,
        "groups": groups,
        "anchors": data["anchors"], "primary": data["primary"] if "primary" in data else spec.get("primary_targets_deposited"),
        "hidden_outcomes_opened": False,
    })
    print(f"frozen {'dry run' if DRY else 'real run'} -> {RUN}")


def _arm(patient, arm):
    return load_arm(PREPARED / "arms" / f"{patient}__{arm}.npz")


def _matrix(patient, arm, gene_index):
    loaded = _arm(patient, arm)
    rna = np.asarray(loaded["rna"][:, gene_index].todense(), dtype=np.float64)
    technical = np.column_stack([np.log1p(loaded["library_cpm"]), np.log1p(loaded["detected"])])
    if loaded["full_protein"]:
        anchors = np.asarray(loaded["protein"][:, manifest_anchors()], dtype=np.float64)
        protein = np.asarray(loaded["protein"], dtype=np.float64)
    else:
        anchors = np.asarray(loaded["protein_anchors"], dtype=np.float64)
        protein = None
    return np.hstack([rna, technical, anchors]), protein


def manifest_anchors():
    return np.asarray(manifest()["anchor_index"], dtype=int)


def gene_pool(data, train, n_genes):
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
    forced = []
    for name in data["proteins"]:
        gene = cognate_for_deposited(name, names)
        if gene is not None:
            forced.append(names.index(gene))
    return select_genes_from_moments(total, total_sq, n_cells, detected, n_genes, forced)


def fit():
    dest = RUN / "fit"
    if (dest / "complete.json").exists():
        raise SystemExit("Refusing to replace a completed fit")
    receipt = _check()
    if (RUN / "evaluation" / "complete.json").exists() or (RUN / "prediction_seal.json").exists():
        raise SystemExit("Refusing to fit after the seal")
    data = manifest()
    spec = protocol()
    grid_spec = spec["cellbridge"]
    groups = receipt["groups"]
    genes = np.asarray(data["selected_gene_index"], dtype=int)
    anchors = np.asarray(data["anchor_index"], dtype=int)
    targets = np.asarray(data["target_index"], dtype=int)
    stats = []
    for offset, patient in enumerate(groups["train"]):
        control_z, control_y = _matrix(patient, "control", genes)
        perturbed_z, perturbed_y = _matrix(patient, "perturbed", genes)
        if control_y is None or perturbed_y is None:
            raise SystemExit(f"Training patient {patient} is missing target proteins")
        stats.append(donor_stats(
            patient, control_z, control_y[:, targets], perturbed_z, perturbed_y[:, targets],
            ks=tuple(grid_spec["metacell_sizes"]), rna_columns=np.arange(len(genes)),
            seed=spec["seed"] + 101 * offset,
        ))
        print(f"  donor {patient}", flush=True)
    queries = list(groups["calibration"]) + list(groups["test"])
    dz = []
    for patient in queries:
        control_z, _ = _matrix(patient, "control", genes)
        perturbed_z, _ = _matrix(patient, "perturbed", genes)
        dz.append(perturbed_z.mean(0) - control_z.mean(0))
    zq = np.vstack(dz)
    layout = feature_layout(len(genes), len(anchors))
    grid = dict(ks=tuple(grid_spec["metacell_sizes"]), mixes=tuple(tuple(m) for m in grid_spec["mixes"]),
                lambdas=tuple(grid_spec["lambdas"]), strategy=grid_spec["strategy"], top=grid_spec["top"])
    model = CellBridge(layout, **grid).fit(stats)
    rna_model = CellBridge({"rna": layout["rna"]}, **grid).fit(stats)
    pool = gene_pool(data, groups["train"], spec["v040_n_genes"])
    rows = []
    rng = np.random.default_rng(spec["seed"])
    cap = int(spec["v040_cells_per_arm"])
    for patient in groups["train"]:
        control, perturbed = _arm(patient, "control"), _arm(patient, "perturbed")
        x = np.asarray(perturbed["rna"].mean(0)).ravel()[pool] - np.asarray(control["rna"].mean(0)).ravel()[pool]
        y = perturbed["protein"].mean(0) - control["protein"].mean(0)
        pieces = []
        for arm_name, loaded in (("control", control), ("perturbed", perturbed)):
            n = int(loaded["rna"].shape[0])
            take = rng.choice(n, size=min(n, cap), replace=False)
            pieces.append((
                np.asarray(loaded["rna"][take][:, pool].todense(), dtype=np.float64),
                np.asarray(loaded["protein"][take], dtype=np.float64),
                np.full(len(take), 0.5 / len(take)),
            ))
        rows.append({
            "donor": patient, "x": np.asarray(x, dtype=np.float64), "y": np.asarray(y, dtype=np.float64),
            "ax": np.vstack([p[0] for p in pieces]), "ay": np.vstack([p[1] for p in pieces]),
            "aw": np.concatenate([p[2] for p in pieces]),
        })
    xq = np.vstack([
        np.asarray(_arm(patient, "perturbed")["rna"].mean(0)).ravel()[pool]
        - np.asarray(_arm(patient, "control")["rna"].mean(0)).ravel()[pool]
        for patient in queries
    ])
    aq = zq[:, len(genes) + 2:]
    baselines, _seconds = baselines_queries(rows, xq, aq, anchors, targets)
    n_proteins = len(data["proteins"])

    def place(values):
        out = np.full((len(queries), n_proteins), np.nan)
        out[:, targets] = values
        return out

    predictions = {
        "cellbridge": place(model.predict(zq)),
        "cellbridge_rna_only": place(rna_model.predict(zq)),
    }
    mask = np.isin(np.arange(n_proteins), targets)
    for key, value in baselines.items():
        if value.shape != (len(queries), n_proteins):
            raise SystemExit(f"{key} predictions have shape {value.shape}")
        predictions[key] = np.where(mask[None], value, np.nan)
    pred_dir = RUN / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / "core.npz", donors=np.asarray(queries), proteins=np.asarray(data["proteins"]),
        roles=np.asarray(["calibration"] * len(groups["calibration"]) + ["test"] * len(groups["test"])),
        **predictions,
    )
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "selection.json", {"cellbridge": model.selected, "cellbridge_rna_only": rna_model.selected,
                                         "targets": [data["proteins"][i] for i in targets]})
    write_json(dest / "complete.json", {"status": "fit_complete", "n_train": len(groups["train"]),
                                        "n_queries": len(queries), "dry_run": DRY})
    print(f"fit complete: {len(groups['train'])} training, {len(queries)} query")


def seal():
    if (RUN / "prediction_seal.json").exists():
        raise SystemExit("Refusing to replace the seal")
    _check()
    if not (RUN / "fit" / "complete.json").exists():
        raise SystemExit("Fit stage incomplete")
    files = sorted((RUN / "predictions").glob("*.npz"))
    names = [path.name for path in files]
    if "core.npz" not in names:
        raise SystemExit("core predictions are missing")
    if "totalvi.npz" not in names:
        raise SystemExit("totalVI must be run before the seal")
    write_json(RUN / "prediction_seal.json", {
        "status": "sealed", "dry_run": DRY,
        "sealed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {path.name: sha256(path) for path in files},
        "hidden_outcomes_opened": False,
    })
    print(f"sealed {RUN}")


def _predictions():
    methods, donors, proteins = {}, None, None
    for path in sorted((RUN / "predictions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as blob:
            found = blob["donors"].astype(str)
            names = blob["proteins"].astype(str)
            if donors is None:
                donors, proteins = found, names
            if not (np.array_equal(found, donors) and np.array_equal(names, proteins)):
                raise SystemExit(f"{path.name} donor or protein order differs")
            for key in blob.files:
                if key not in ("donors", "proteins", "roles"):
                    methods[key] = blob[key].astype(float)
    return methods, donors, proteins


def _truth(queries, proteins, groups):
    columns = _json(VAULT / "columns.json")["targets"]
    truth = np.full((len(queries), len(proteins)), np.nan)
    index = {name: i for i, name in enumerate(proteins)}
    for i, patient in enumerate(queries):
        if DRY:
            values = _arm(patient, "perturbed")["protein"].mean(0) - _arm(patient, "control")["protein"].mean(0)
            truth[i] = values
            continue
        if patient in set(groups["train"]):
            raise SystemExit("The real unblind must not score training patients")
        change = np.zeros(len(columns))
        for arm, sign in (("control", -1.0), ("perturbed", 1.0)):
            cells = np.load(VAULT / f"{patient}__{arm}.npy")
            change += sign * cells.mean(0)
        for name, value in zip(columns, change):
            truth[i, index[name]] = value
    return truth


def unblind():
    dest = RUN / "evaluation"
    if (dest / "complete.json").exists():
        raise SystemExit("Refusing to unblind twice")
    receipt = _check()
    seal_value = _json(RUN / "prediction_seal.json")
    if seal_value.get("hidden_outcomes_opened"):
        raise SystemExit("The seal says outcomes were already opened")
    present = sorted(path.name for path in (RUN / "predictions").glob("*.npz"))
    if present != sorted(seal_value["files"]):
        raise SystemExit("Prediction files differ from the seal")
    for name, digest in seal_value["files"].items():
        if sha256(RUN / "predictions" / name) != digest:
            raise SystemExit(f"Sealed prediction changed: {name}")
    data = manifest()
    groups = receipt["groups"]
    methods, donors, proteins = _predictions()
    donors, proteins = list(donors), list(proteins)
    if donors != list(groups["calibration"]) + list(groups["test"]):
        raise SystemExit("Prediction donors do not match the freeze")
    if proteins != data["proteins"]:
        raise SystemExit("Prediction proteins do not match the manifest")
    spec = protocol()
    ytr = np.vstack([
        _arm(patient, "perturbed")["protein"].mean(0) - _arm(patient, "control")["protein"].mean(0)
        for patient in groups["train"]
    ])
    scale = response_scale(ytr)
    truth = _truth(donors, proteins, groups)
    targets = np.asarray(data["target_index"], dtype=int)
    primary_names = data.get("primary") or spec["primary_targets_deposited"]
    primary = np.array([proteins.index(name) for name in primary_names])
    is_test = np.array([donor in set(groups["test"]) for donor in donors])
    is_cal = np.array([donor in set(groups["calibration"]) for donor in donors])
    endpoints = {"E1_primary": primary, "E2_all_hidden": targets}
    err = {name: np.abs(value - truth) / scale for name, value in methods.items()}
    loss = {}
    rows = []
    mu = ytr.mean(0)
    for name, matrix in err.items():
        for endpoint, cols in endpoints.items():
            per = matrix[np.ix_(is_test, cols)].mean(1)
            loss[(name, endpoint)] = per
            rows.append({"method": name, "endpoint": endpoint, "standardized_mae": float(np.mean(per)),
                         "n_donors": int(len(per))})
    tests = fixed_sequence(
        loss, spec["hypotheses"], np.asarray(donors)[is_test], alpha=spec["alpha"], seed=spec["seed"],
        metadata={"protocol": spec["version"], "model_hash": receipt["sources_sha256"]["src/responsebridge/cellbridge.py"]},
    )
    noninferior = []
    for endpoint in endpoints:
        if ("v040_abundance_shrinkage", endpoint) not in loss:
            continue
        baseline, model = loss[("v040_abundance_shrinkage", endpoint)], loss[("cellbridge", endpoint)]
        boot = paired_gain_bootstrap(baseline, model, np.asarray(donors)[is_test], n_bootstrap=10000,
                                     seed=spec["seed"], independent_test=True, models_frozen=True,
                                     model_metadata={"protocol": spec["version"]})
        upper = -boot["difference_ci95"][0]
        margin = spec["noninferiority_margin_vs_v040"]
        noninferior.append({"endpoint": endpoint, "margin": margin,
                            "mean_excess_loss_cellbridge": float(np.mean(model - baseline)),
                            "upper_ci95_excess_loss": float(upper),
                            "noninferior": bool(upper < margin)})
    alpha = spec["conformal_alpha"]
    conformal = {}
    for name in ("cellbridge", "v040_abundance_shrinkage", "rna_ridge", "joint_ridge", "totalvi"):
        if name not in err or not np.isfinite(err[name][np.ix_(is_cal, targets)]).all():
            continue
        scores = err[name][np.ix_(is_cal, targets)].ravel()
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhat = float(np.quantile(scores, level, method="higher"))
        conformal[name] = {"pooled_standardized_quantile": qhat,
                           "test_coverage_all_hidden": float((err[name][np.ix_(is_test, targets)] <= qhat).mean()),
                           "test_coverage_primary": float((err[name][np.ix_(is_test, primary)] <= qhat).mean())}
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "summary.csv", index=False)
    donor_rows = [{"donor": donor, "role": "test" if flag else "calibration", "method": name,
                   "endpoint": endpoint, "loss": float(err[name][i, cols].mean())}
                  for name in err for endpoint, cols in endpoints.items()
                  for i, (donor, flag) in enumerate(zip(donors, is_test))]
    pd.DataFrame(donor_rows).to_csv(dest / "donor_losses.csv", index=False)
    claim = [row["id"] for row in tests if row["id"] in spec["primary_claim_hypotheses"]]
    decision = {
        "version": spec["version"], "dry_run": DRY, "hypotheses": tests,
        "noninferiority_vs_v040": noninferior, "conformal": conformal,
        "primary_claim_supported": bool(claim) and all(
            row.get("rejected_null") for row in tests if row["id"] in spec["primary_claim_hypotheses"]),
        "sequence_stopped": next((row["id"] for row in tests if row.get("tested_in_sequence") and not row.get("rejected_null")), None),
        "unblinded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_osf_url": spec["preregistration"]["osf_url"],
        "training_mean_sse_reference": float(np.sum((mu[targets] - truth[np.ix_(is_test, targets)]) ** 2)),
    }
    write_json(dest / "decision.json", decision)
    write_json(dest / "complete.json", {"status": "unblinded_once", "dry_run": DRY, "n_test": int(is_test.sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"unblinded once: {int(is_test.sum())} test donors; primary claim {decision['primary_claim_supported']}")


if __name__ == "__main__":
    {"freeze": freeze, "fit": fit, "seal": seal, "unblind": unblind}[sys.argv[1]]()
