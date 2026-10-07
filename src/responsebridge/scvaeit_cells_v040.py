"""ResponseBridge 0.4 donor-held-out, cell-level official scVAEIT comparator.

The coordinator alone reads development validation outcomes. Fit subprocesses
receive development cells and RNA/anchor-only queries. Outer test target values
are never indexed, validated, normalized or scored by this module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import ResourceGuard, ResourceLimit, exclusive_run, fingerprint, sha256, write_json
from .scvaeit import ScVAEITRegressor


FIXED_4 = ("CD38", "ICOS", "PD-1", "HLA-DR")
FIXED_8 = (*FIXED_4, "CD127", "CD27", "CD28", "CD45RO")
PRIMARY = ("CD25", "CD69")
ARMS = ("Baseline", "CD3_CD28")


@dataclass(frozen=True)
class CohortSpec:
    """Exact assay labels, frozen in the caller's manifest; no relabeling."""
    lineage: str = "T"
    control_condition: str = "Baseline"
    stimulated_condition: str = "CD3_CD28"

    def __post_init__(self):
        for value in (self.lineage, self.control_condition, self.stimulated_condition):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("Cohort labels must be nonempty exact strings without surrounding whitespace")
        if self.control_condition == self.stimulated_condition:
            raise ValueError("Control and stimulated condition labels must differ")

    @property
    def arms(self):
        return (self.control_condition, self.stimulated_condition)

    @classmethod
    def from_manifest(cls, manifest):
        keys = ("control_condition", "stimulated_condition")
        if any(key in manifest for key in keys) and not all(key in manifest for key in keys):
            raise ValueError("Declare both control_condition and stimulated_condition explicitly")
        result = cls(lineage=manifest.get("cohort_lineage", "T"),
                     control_condition=manifest.get("control_condition", ARMS[0]),
                     stimulated_condition=manifest.get("stimulated_condition", ARMS[1]))
        if "primary_stimulation" in manifest and manifest["primary_stimulation"] != result.stimulated_condition:
            raise ValueError("primary_stimulation disagrees with stimulated_condition")
        return result


def _validated_arms(arms):
    if isinstance(arms, str) or len(arms) != 2:
        raise ValueError("Exactly two ordered condition labels required")
    return CohortSpec(control_condition=arms[0], stimulated_condition=arms[1]).arms


@dataclass(frozen=True)
class Settings:
    n_hvg: int = 2000
    batch_size: int = 64
    hidden: int = 128
    latent: int = 16
    learning_rate: float = 3e-4
    mc_samples: int = 32
    threads: int = 6


@dataclass
class CellData:
    rna: sparse.csr_matrix
    proteins: np.ndarray
    obs: pd.DataFrame
    genes: list[str]
    protein_names: list[str]
    input_sha256: dict
    eligibility: list[dict]
    cohort: CohortSpec = CohortSpec()


def _source_hashes():
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in ("scvaeit_cells_v040.py", "scvaeit.py", "artifacts.py")}


def _versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "pandas", "scVAEIT", "tensorflow", "tensorflow-probability")}


def _verify_hashes(entries):
    for path, expected in entries.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f"Input integrity failure: {path}")


def _verify_outputs(folder, entries):
    for relative, expected in entries.items():
        path = Path(folder) / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Output integrity failure: {path}")


def _files(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(Path(folder).rglob("*"))
            if p.is_file() and p.name != "receipt.json"}


def normalized_rna(counts, target_sum=10000.):
    """Cellwise RNA-only library normalization; no fitted query statistics."""
    counts = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        raise ValueError("RNA counts must be finite and nonnegative")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Eligible cells must have positive RNA library sizes")
    counts = counts.multiply((target_sum / totals)[:, None]).tocsr()
    np.log1p(counts.data, out=counts.data)
    return counts


def load_cells(prepared, minimum_cells=30, cohort=None):
    """Verify canonical files without scanning hidden protein/isotype values.

    load_prepared().validate() intentionally scans every count, including query
    targets. Here immutable checksums establish identity and RNA/metadata are
    checked separately; only role-authorized protein slices are read later.
    """
    cohort = cohort or CohortSpec()
    arms = cohort.arms
    prepared = Path(prepared).resolve()
    path = prepared / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("counts") != "raw_integer" or manifest.get("orientation") != "cells_by_features":
        raise ValueError("Canonical raw cells-by-features counts required")
    names = ("rna_counts.npz", "protein_counts.npy", "obs.csv", "genes.json", "proteins.json")
    hashes = {str(path): sha256(path)}
    hashes.update({str(prepared / name): manifest["files"][name]["sha256"] for name in names})
    _verify_hashes(hashes)
    obs = pd.read_csv(prepared / "obs.csv", keep_default_na=False,
                      dtype={"donor": str, "cell_id": str})
    required = {"donor", "cell_id", "lineage", "condition", "is_control"}
    if required - set(obs) or not obs.cell_id.is_unique:
        raise ValueError("Unique cell identifiers and donor/lineage/condition/control metadata required")
    obs["prepared_row"] = np.arange(len(obs), dtype=np.int64)
    rna = sparse.load_npz(prepared / "rna_counts.npz").tocsr()
    proteins = np.load(prepared / "protein_counts.npy", mmap_mode="r", allow_pickle=False)
    genes = json.loads((prepared / "genes.json").read_text())
    protein_names = json.loads((prepared / "proteins.json").read_text())
    if rna.shape != (len(obs), len(genes)) or proteins.shape != (len(obs), len(protein_names)):
        raise ValueError("Prepared dimensions disagree")
    if len(set(genes)) != len(genes) or len(set(protein_names)) != len(protein_names):
        raise ValueError("Feature names must be unique")
    selected = obs.lineage.eq(cohort.lineage) & obs.condition.isin(arms)
    obs = obs.loc[selected].copy()
    control = obs.is_control.astype(str).str.lower().map({"true": True, "false": False})
    if control.isna().any() or not np.array_equal(control.to_numpy(), obs.condition.eq(arms[0]).to_numpy()):
        raise ValueError("Condition/control labels disagree")
    eligibility = []
    eligible_donors = []
    for donor, rows in obs.groupby("donor", sort=True):
        counts = {arm: int(rows.condition.eq(arm).sum()) for arm in arms}
        eligible = min(counts.values()) >= minimum_cells
        eligibility.append({"donor": donor, **counts, "eligible": eligible})
        if eligible:
            eligible_donors.append(donor)
    obs = obs.loc[obs.donor.isin(eligible_donors)].reset_index(drop=True)
    if len(eligible_donors) < 4:
        raise ValueError("At least four eligible paired donors are required")
    rna = normalized_rna(rna[obs.prepared_row.to_numpy()])
    return CellData(rna, proteins, obs, genes, protein_names, hashes, eligibility, cohort)


def donor_roles(donors, test_donor, seed, manifest=None):
    """Three deterministic donor folds; validation never crosses training."""
    manifest = manifest or {}
    def donor_list(value):
        return [] if value is None else [str(value)] if isinstance(value, str) else list(map(str, value))
    donors = sorted(set(map(str, donors)))
    requested = donor_list(test_donor)
    declared = donor_list(manifest.get("test", manifest.get("test_donors", manifest.get("test_donor"))))
    if requested and declared and sorted(requested) != sorted(declared):
        raise ValueError("Test donors disagree with manifest")
    test = declared or requested
    calibration = donor_list(manifest.get("calibration", manifest.get("calibration_donors")))
    if not test or len(set(test)) != len(test) or set(test) - set(donors):
        raise ValueError("Test donors must be unique and eligible")
    if len(set(calibration)) != len(calibration) or set(calibration) - set(donors) or set(calibration) & set(test):
        raise ValueError("Calibration donors must be unique, eligible and disjoint from test")
    if "train" in manifest and "development_donors" in manifest and sorted(manifest["train"]) != sorted(manifest["development_donors"]):
        raise ValueError("Conflicting development donor declarations")
    development = donor_list(manifest.get("train", manifest.get("development_donors",
                                  [d for d in donors if d not in test + calibration])))
    if len(development) < 3 or len(set(development)) != len(development) or set(development) & set(test + calibration) or set(development) - set(donors):
        raise ValueError("Development donors must be unique, eligible and exclude test/calibration")
    development = sorted(development)
    if "inner_folds" in manifest:
        folds = manifest["inner_folds"]
    else:
        shuffled = np.asarray(development)[np.random.default_rng(seed).permutation(len(development))]
        folds = [{"train_donors": sorted(set(development) - set(part.tolist())),
                  "validation_donors": sorted(part.tolist())} for part in np.array_split(shuffled, 3)]
    if len(folds) != 3:
        raise ValueError("Exactly three inner donor folds required")
    seen = []
    for fold in folds:
        train, val = fold["train_donors"], fold["validation_donors"]
        if not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ValueError("Inner roles must be nonempty and unique")
        if set(train) & set(val) or set(train) | set(val) != set(development):
            raise ValueError("Inner train and validation must partition outer development")
        seen.extend(val)
    if sorted(seen) != development:
        raise ValueError("Each development donor must be validated exactly once")
    return {"test_donor": test[0] if len(test) == 1 else None,
            "test_donors": sorted(test), "calibration_donors": sorted(calibration),
            "development_donors": development, "inner_folds": folds}


def cell_weights(obs, arms=ARMS):
    """Equal mass per donor, then arm, then cell within that donor-arm."""
    arms = _validated_arms(arms)
    sizes = obs.groupby(["donor", "condition"], sort=False).donor.transform("size").to_numpy()
    pairs = obs[["donor", "condition"]].drop_duplicates()
    if not pairs.groupby("donor").size().eq(2).all() or not set(obs.condition).issubset(arms):
        raise ValueError("Training donors must have both prespecified arms")
    weights = 1. / sizes.astype(np.float64)
    return weights / weights.sum()


def sparse_hvgs(x, weights, n_hvg):
    """Weighted training-only feature variances without dense all-gene arrays."""
    weights = np.asarray(weights, dtype=np.float64)
    if n_hvg < 1 or weights.shape != (x.shape[0],) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Valid positive aligned training weights/HVG budget required")
    weights = weights / weights.sum()
    x64 = x.astype(np.float64)
    mean = np.asarray(x64.T @ weights).ravel()
    variance = np.maximum(np.asarray(x64.power(2).T @ weights).ravel() - mean * mean, 0)
    eligible = np.flatnonzero(variance > 1e-12)
    if not len(eligible):
        raise ValueError("No variable development RNA features")
    return eligible[np.argsort(-variance[eligible], kind="stable")[:n_hvg]]


def aggregate_responses(prediction, donor, condition, arms=ARMS):
    """Mean cellwise transformed ADT, then matched stimulation minus baseline."""
    arms = _validated_arms(arms)
    prediction = np.asarray(prediction, dtype=np.float64)
    donor, condition = np.asarray(donor, dtype=str), np.asarray(condition, dtype=str)
    if prediction.ndim != 2 or len(prediction) != len(donor) or len(donor) != len(condition) or not np.isfinite(prediction).all():
        raise ValueError("Finite aligned cell predictions required")
    names = np.asarray(sorted(set(donor)), dtype=str)
    if not len(names) or set(condition) - set(arms):
        raise ValueError("Only the two declared arms may be aggregated")
    means, counts = {}, {}
    for arm in arms:
        masks = [(donor == d) & (condition == arm) for d in names]
        if any(not mask.any() for mask in masks):
            raise ValueError("Each response requires matched control/stimulation cells")
        means[arm] = np.stack([prediction[m].mean(axis=0) for m in masks])
        counts[arm] = np.asarray([m.sum() for m in masks], dtype=np.int64)
    return {"donor": names, "control_mean": means[arms[0]], "stimulated_mean": means[arms[1]],
            "prediction": means[arms[1]] - means[arms[0]],
            "n_control": counts[arms[0]], "n_stimulated": counts[arms[1]]}


def _log_counts(values, *, dtype=np.float32):
    values = np.asarray(values, dtype=dtype)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Authorized protein counts must be finite and nonnegative")
    return np.log1p(values)


def _panels(protein_names, manifest):
    panels = {"fixed_4": list(manifest.get("fixed_panel_4", FIXED_4)),
              "fixed_8": list(manifest.get("fixed_panel_8", FIXED_8))}
    primary = list(manifest.get("primary_targets", PRIMARY))
    if primary != list(PRIMARY):
        raise ValueError("The declared primary endpoint is CD25/CD69")
    for name, panel in panels.items():
        if len(panel) != int(name[-1]) or len(set(panel)) != len(panel) or set(panel) & set(primary):
            raise ValueError("Fixed panels must contain exactly4/8 unique non-target markers")
        if set(panel + primary) - set(protein_names):
            raise ValueError("A prespecified target/anchor is absent from the measured panel")
    return panels, primary


def prepare_inputs(data, folder, train_donors, query_donors, panels, settings, *, development_validation):
    """Freeze a reusable cell input cache, with no query target columns."""
    folder = Path(folder)
    contract = {"train_donors": sorted(train_donors), "query_donors": sorted(query_donors),
                "panels": panels, "settings": asdict(settings), "input_sha256": data.input_sha256,
                "cohort": asdict(data.cohort),
                "development_validation": development_validation,
                "primary_validation_scaling": "inner-training donor response SD; ddof=0; floor1e-6" if development_validation else None}
    if set(train_donors) & set(query_donors):
        raise ValueError("Training and query donor roles overlap")
    if (folder / "receipt.json").exists():
        receipt = json.loads((folder / "receipt.json").read_text())
        if receipt["contract"] != contract:
            raise ValueError("Input-cache contract changed; use a new run")
        _verify_outputs(folder, receipt["output_sha256"])
        return receipt
    if folder.exists() and any(folder.iterdir()):
        index = 0
        while folder.with_name(folder.name + f".incomplete_{index:03d}").exists():
            index += 1
        folder.rename(folder.with_name(folder.name + f".incomplete_{index:03d}"))
    folder.mkdir(parents=True, exist_ok=True)
    tr = np.flatnonzero(data.obs.donor.isin(train_donors))
    qr = np.flatnonzero(data.obs.donor.isin(query_donors))
    if set(data.obs.iloc[tr].donor) != set(train_donors) or set(data.obs.iloc[qr].donor) != set(query_donors):
        raise ValueError("Missing requested donor cells")
    train_obs, query_obs = data.obs.iloc[tr], data.obs.iloc[qr]
    weights = cell_weights(train_obs, data.cohort.arms)
    features = sparse_hvgs(data.rna[tr], weights, settings.n_hvg)
    # The second adapter HVG pass only reorders this development-selected set.
    x = data.rna[tr][:, features].toarray()
    y = _log_counts(data.proteins[np.ix_(train_obs.prepared_row.to_numpy(), np.arange(len(data.protein_names)))])
    np.savez_compressed(folder / "train.npz", x=x, y=y, weights=weights,
                        prepared_rows=train_obs.prepared_row.to_numpy())
    del x, y
    union = sorted({data.protein_names.index(p) for panel in panels.values() for p in panel})
    observed = _log_counts(data.proteins[np.ix_(query_obs.prepared_row.to_numpy(), union)])
    np.savez_compressed(folder / "query.npz", x=data.rna[qr][:, features].toarray(),
                        anchor_values=observed, anchor_indices=np.asarray(union, dtype=np.int64),
                        donor=query_obs.donor.to_numpy(dtype=str), condition=query_obs.condition.to_numpy(dtype=str),
                        prepared_rows=query_obs.prepared_row.to_numpy())
    provenance = data.obs.iloc[np.r_[tr, qr]][["cell_id", "prepared_row", "donor", "condition"]].copy()
    provenance["role"] = ["training"] * len(tr) + ["query"] * len(qr)
    provenance["training_probability"] = np.r_[weights, np.zeros(len(qr))]
    provenance.to_csv(folder / "cell_roles.csv", index=False)
    np.savez_compressed(folder / "rna_selection.npz", gene_indices=features,
                        gene_names=np.asarray(data.genes, dtype=str)[features])
    if development_validation:
        targets = [data.protein_names.index(p) for p in PRIMARY]
        training_truth = _log_counts(data.proteins[np.ix_(train_obs.prepared_row.to_numpy(), targets)], dtype=np.float64)
        training_response = aggregate_responses(training_truth, train_obs.donor, train_obs.condition, data.cohort.arms)
        scales = np.maximum(np.std(training_response["prediction"], axis=0, ddof=0), 1e-6)
        truth = _log_counts(data.proteins[np.ix_(query_obs.prepared_row.to_numpy(), targets)], dtype=np.float64)
        truth = aggregate_responses(truth, query_obs.donor, query_obs.condition, data.cohort.arms)
        np.savez_compressed(folder / "development_truth.npz", donor=truth["donor"],
                            response=truth["prediction"], primary_targets=np.asarray(PRIMARY),
                            primary_training_response_scales=scales,
                            training_response_donors=training_response["donor"],
                            primary_training_responses=training_response["prediction"])
    receipt = {"contract": contract, "training_cells": len(tr), "query_cells": len(qr),
               "protein_names": data.protein_names, "selected_rna_features": len(features),
               "output_sha256": _files(folder)}
    write_json(folder / "receipt.json", receipt, immutable=True)
    return receipt


def fit_job(spec_path, model_factory=ScVAEITRegressor):
    """Subprocess entry: only training NPZ and anchor-only query NPZ are opened."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    output = Path(spec["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fit output must be new or empty")
    _verify_hashes(spec["input_sha256"])
    if spec["source_sha256"] != _source_hashes():
        raise ValueError("Worker source differs from frozen specification")
    settings = Settings(**spec["settings"])
    cohort = CohortSpec(**spec.get("cohort", asdict(CohortSpec())))
    model = model_factory(epochs=spec["epochs"], seed=spec["seed"], **asdict(settings))
    started = time.monotonic()
    with np.load(spec["train"], allow_pickle=False) as train:
        if set(train.files) != {"x", "y", "weights", "prepared_rows"}:
            raise ValueError("Unexpected training NPZ schema")
        model.fit(train["x"], train["y"], weights=train["weights"])
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model")
    with np.load(spec["query"], allow_pickle=False) as query:
        if set(query.files) != {"x", "anchor_values", "anchor_indices", "donor", "condition", "prepared_rows"}:
            raise ValueError("Query schema contains unexpected (possibly hidden) features")
        union = query["anchor_indices"].tolist()
        for name, indices in spec["panel_indices"].items():
            columns = [union.index(i) for i in indices]
            prediction = model.predict(query["x"], query["anchor_values"][:, columns], indices)
            result = aggregate_responses(prediction, query["donor"], query["condition"], cohort.arms)
            np.savez_compressed(output / f"{name}.npz", **result,
                                protein_names=np.asarray(spec["protein_names"], dtype=str))
    _verify_hashes(spec["input_sha256"])
    if spec["source_sha256"] != _source_hashes():
        raise ValueError("Worker source changed during fit")
    receipt = {"schema_version": 1, "spec_sha256": sha256(spec_path), "source_sha256": spec["source_sha256"],
               "input_sha256": spec["input_sha256"], "output_sha256": _files(output),
               "epochs": spec["epochs"], "seed": spec["seed"], "settings": asdict(settings),
               "wall_seconds": time.monotonic() - started, "training_metadata": model.metadata(),
               "fit_unit": "cells", "cohort": asdict(cohort),
               "response_unit": f"mean cellwise log1p ADT {cohort.stimulated_condition} minus matched {cohort.control_condition}",
               "query_target_values_read": False, "query_adaptation": False}
    write_json(output / "receipt.json", receipt, immutable=True)
    return receipt


def _monitor_job(spec_path, log_path, guard, pause_files=()):
    env = os.environ.copy()
    threads = json.loads(Path(spec_path).read_text())["settings"]["threads"]
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(threads)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    with Path(log_path).open("x") as log:
        child = subprocess.Popen([sys.executable, "-m", "responsebridge.scvaeit_cells_v040", "--fit-job", str(spec_path)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        previous_pids = guard.extra_pids
        guard.extra_pids = (*previous_pids, child.pid)
        try:
            while child.poll() is None:
                if any(Path(p).exists() for p in pause_files):
                    raise ResourceLimit("External PAUSE file requested a stop")
                guard.check()
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"scVAEIT cell fit failed ({child.returncode}); inspect {log_path}")
        except BaseException as error:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            write_json(Path(spec_path).parent / "interrupted.json", {"error": type(error).__name__,
                       "message": str(error), "child_returncode": child.returncode,
                       "action": "Incomplete attempt preserved; resume starts a new attempt"}, immutable=True)
            raise
        finally:
            guard.extra_pids = previous_pids


def _run_job(cache, folder, epochs, seed, settings, guard, pause_files=(), runner=_monitor_job):
    cache, folder = Path(cache).resolve(), Path(folder).resolve()
    cache_receipt = json.loads((cache / "receipt.json").read_text())
    panels = cache_receipt["contract"]["panels"]
    proteins = cache_receipt["protein_names"]
    common = {"train": str(cache / "train.npz"), "query": str(cache / "query.npz"),
              "input_sha256": {str(cache / name): sha256(cache / name) for name in ("train.npz", "query.npz", "receipt.json")},
              "source_sha256": _source_hashes(), "settings": asdict(settings), "epochs": int(epochs), "seed": int(seed),
              "cohort": cache_receipt["contract"]["cohort"],
              "protein_names": proteins,
              "panel_indices": {name: [proteins.index(p) for p in panel] for name, panel in panels.items()}}
    finished = folder / "complete.json"
    if finished.exists():
        saved = json.loads(finished.read_text())
        if saved["contract"] != common:
            raise ValueError("Completed job contract changed; use a new run")
        attempt = folder / saved["attempt"]
        receipt = json.loads((attempt / "output/receipt.json").read_text())
        if sha256(attempt / "output/receipt.json") != saved["receipt_sha256"]:
            raise ValueError("Job receipt integrity failure")
        _verify_outputs(attempt / "output", receipt["output_sha256"])
        return attempt / "output"
    folder.mkdir(parents=True, exist_ok=True)
    # Interrupted attempts are retained; a completed fit is never overwritten.
    index = 0
    while (folder / f"attempt_{index:03d}").exists():
        index += 1
    attempt = folder / f"attempt_{index:03d}"
    attempt.mkdir()
    spec = {**common, "output": str(attempt / "output")}
    write_json(attempt / "spec.json", spec, immutable=True)
    guard.check()
    runner(attempt / "spec.json", attempt / "worker.log", guard, pause_files)
    receipt = json.loads((attempt / "output/receipt.json").read_text())
    if receipt["spec_sha256"] != sha256(attempt / "spec.json"):
        raise ValueError("Job specification integrity failure")
    _verify_outputs(attempt / "output", receipt["output_sha256"])
    write_json(finished, {"contract": common, "attempt": attempt.name,
                         "receipt_sha256": sha256(attempt / "output/receipt.json")}, immutable=True)
    return attempt / "output"


def _save_arrays_immutable(path, arrays):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError(f"Existing prediction differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _write_role_predictions(output, role, donors, jobs, seeds, panels, proteins, epochs):
    """Split predictions only; no target outcomes are required for either role."""
    if not donors:
        return
    name = "predictions" if role == "test" else "calibration_predictions"
    combined = {"seeds": np.asarray(seeds, dtype=np.int64),
                "protein_names": np.asarray(proteins, dtype=str),
                "donor_ids": np.asarray(donors, dtype=str), "selected_epochs": np.asarray(epochs)}
    for panel in panels:
        predictions = []
        for seed, job in zip(seeds, jobs):
            with np.load(job / f"{panel}.npz", allow_pickle=False) as saved:
                order = saved["donor"].tolist()
                rows = [order.index(d) for d in donors]
                if not np.array_equal(saved["protein_names"], combined["protein_names"]):
                    raise ValueError("Final output protein order mismatch")
                arrays = {key: saved[key][rows] for key in
                          ("donor", "control_mean", "stimulated_mean", "prediction", "n_control", "n_stimulated")}
                arrays["protein_names"] = saved["protein_names"]
            _save_arrays_immutable(Path(output) / name / f"seed_{seed}_{panel}.npz", arrays)
            predictions.append(arrays["prediction"])
        combined[panel] = np.stack(predictions)
    _save_arrays_immutable(Path(output) / f"{name}.npz", combined)


def validate_cell_receipt(folder):
    """Validate a completed cell fold without importing TensorFlow or outcomes.

    Return False only for an unfinished fold; tampering, changed inputs or
    scientific source raise ValueError. Runtime package versions remain recorded
    in the frozen lock; this validator can run in the separate core environment.
    """
    folder = Path(folder).resolve()
    if not (folder / "receipt.json").exists():
        return False
    receipt = json.loads((folder / "receipt.json").read_text())
    contract = json.loads((folder / "fold_lock.json").read_text())
    if receipt.get("status") != "completed" or receipt.get("contract_sha256") != fingerprint(contract):
        raise ValueError("Cell fold contract integrity failure")
    if contract["source_sha256"] != _source_hashes():
        raise ValueError("Cell comparator scientific source changed since fitting")
    _verify_hashes(contract["input_sha256"])
    _verify_outputs(folder, receipt["output_sha256"])
    required = {"fold_lock.json", "eligibility.json"}
    if contract["pilot_epochs"] is not None:
        jobs = ["runtime_pilot"]
        required.add("runtime_pilot.json")
    else:
        jobs = [f"inner_{i:02d}_epoch_{epoch}" for i in range(3) for epoch in contract["epoch_grid"]]
        jobs += [f"final_seed_{seed}" for seed in contract["seeds"]]
        required.update(("development_tuning.csv", "epoch_selection.json", "predictions.npz"))
        if contract["roles"].get("calibration_donors"):
            required.add("calibration_predictions.npz")
    for name in jobs:
        job = folder / "jobs" / name
        relative = str((job / "complete.json").relative_to(folder))
        required.add(relative)
        complete = json.loads((job / "complete.json").read_text())
        attempt = job / complete["attempt"]
        spec_path = attempt / "spec.json"
        spec = json.loads(spec_path.read_text())
        common = {k: v for k, v in spec.items() if k != "output"}
        if complete["contract"] != common or common["source_sha256"] != contract["source_sha256"]:
            raise ValueError("Nested fit contract/source integrity failure")
        if Path(spec["output"]).resolve() != (attempt / "output").resolve():
            raise ValueError("Nested fit output path mismatch")
        _verify_hashes(common["input_sha256"])
        output = attempt / "output"
        fit_receipt = json.loads((output / "receipt.json").read_text())
        if complete["receipt_sha256"] != sha256(output / "receipt.json") or fit_receipt["spec_sha256"] != sha256(spec_path):
            raise ValueError("Nested fit receipt/specification integrity failure")
        if fit_receipt["input_sha256"] != common["input_sha256"] or fit_receipt["source_sha256"] != common["source_sha256"]:
            raise ValueError("Nested fit input/source integrity failure")
        _verify_outputs(output, fit_receipt["output_sha256"])
        required.update(str(p.relative_to(folder)) for p in [spec_path, output / "receipt.json"])
        required.update(str((output / name).relative_to(folder)) for name in fit_receipt["output_sha256"])
    if required - set(receipt["output_sha256"]):
        raise ValueError("Cell fold receipt omits required completed scientific artifacts")
    return True


def run_cell_fold(prepared, output, test_donor=None, *, epoch_grid=(100, 300, 1000),
                  seeds=(20261004, 20261005, 20261006), threads=6, manifest=None,
                  settings=None, minimum_cells=30, max_rss_gib=32, minimum_free_gib=30,
                  max_hours=168, pause_files=(), pilot_epochs=None, runner=_monitor_job):
    """Freeze roles, select epoch on development donors, then predict test cells.

    pilot_epochs is an execution-cost pilot on all outer development cells; it
    opens no validation truth and must never be used for accuracy selection.
    """
    output = Path(output).resolve()
    prepared = Path(prepared).resolve()
    settings = settings or Settings(threads=threads)
    manifest_path = Path(manifest).resolve() if manifest is not None else None
    roles_manifest = json.loads(manifest_path.read_text()) if manifest_path else {}
    epoch_grid, seeds = sorted(set(map(int, epoch_grid))), list(map(int, seeds))
    if not epoch_grid or min(epoch_grid) < 1 or len(seeds) != 3 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Positive epoch budgets and exactly three unique nonnegative seeds required")
    if pilot_epochs is not None and pilot_epochs < 1:
        raise ValueError("Pilot epoch count must be positive")
    with exclusive_run(output):
        guard = ResourceGuard(output, max_rss_gib, minimum_free_gib, max_hours)
        guard.check()
        cohort = CohortSpec.from_manifest(roles_manifest)
        data = load_cells(prepared, minimum_cells, cohort)
        if data.cohort != cohort:
            raise ValueError("Prepared cell loader cohort differs from frozen manifest")
        roles = donor_roles(data.obs.donor, test_donor, seeds[0], roles_manifest)
        query_donors = sorted(roles["test_donors"] + roles["calibration_donors"])
        panels, primary = _panels(data.protein_names, roles_manifest)
        input_hashes = dict(data.input_sha256)
        if manifest_path:
            input_hashes[str(manifest_path)] = sha256(manifest_path)
        contract = {"schema_version": 2, "model_version": "0.4.0", "prepared": str(prepared), "input_sha256": input_hashes,
                    "source_sha256": _source_hashes(), "versions": _versions(), "roles": roles,
                    "panels": panels, "primary_targets": primary, "settings": asdict(settings),
                    "cohort": asdict(cohort),
                    "epoch_grid": epoch_grid, "seeds": seeds, "minimum_cells": minimum_cells,
                    "pilot_epochs": pilot_epochs,
                    "weighting": "equal donor, equal condition, equal cell within donor-condition",
                    "selection": "mean donor-by-target development response MAE divided by inner-training donor response SD (ddof0, floor1e-6); fixed8; first seed; ties choose fewer epochs",
                    "source_data_exposure": roles_manifest.get("source_data_exposure",
                        "No untouched-data status inferred by worker; Lawlor_v2 outcomes were previously exposed in this project")}
        write_json(output / "fold_lock.json", contract, immutable=True)
        if (output / "receipt.json").exists():
            receipt = json.loads((output / "receipt.json").read_text())
            if receipt["contract_sha256"] != fingerprint(contract):
                raise ValueError("Completed fold contract changed")
            validate_cell_receipt(output)
            return receipt
        write_json(output / "eligibility.json", data.eligibility, immutable=True)
        if pilot_epochs is not None:
            # Fit all outer-development cells. Query inference uses only the
            # held-out RNA/anchors; no target metrics inform this timing pilot.
            cache = output / "inputs/final"
            prepare_inputs(data, cache, roles["development_donors"], query_donors, panels,
                           settings, development_validation=False)
            job = _run_job(cache, output / "jobs/runtime_pilot", pilot_epochs, seeds[0],
                           settings, guard, pause_files, runner)
            metadata = json.loads((job / "receipt.json").read_text())
            train_seconds = metadata["training_metadata"]["training_seconds"]
            # Inner training sets contain ~2/3 as many cells as the outer fit.
            budget = {str(final_epochs): 10 * (3 * (2 / 3) * sum(epoch_grid) + len(seeds) * final_epochs)
                      for final_epochs in epoch_grid}
            summary = {"purpose": "execution cost only; not accuracy tuning", "pilot_epochs": pilot_epochs,
                       "training_seconds": train_seconds, "per_epoch_seconds_including_initial_tracing": train_seconds / pilot_epochs,
                       "estimated_suite_training_hours": {k: v * train_seconds / pilot_epochs / 3600 for k, v in budget.items()},
                       "estimate_limits": "Approximate: includes first-epoch tracing, excludes repeated process/import and prediction costs; cell counts vary by donor fold",
                       "test_outcome_metrics_computed": False}
            write_json(output / "runtime_pilot.json", summary, immutable=True)
        else:
            tuning = []
            for i, fold in enumerate(roles["inner_folds"]):
                guard.check()
                cache = output / f"inputs/inner_{i:02d}"
                prepare_inputs(data, cache, fold["train_donors"], fold["validation_donors"],
                               {"fixed_8": panels["fixed_8"]}, settings, development_validation=True)
                with np.load(cache / "development_truth.npz", allow_pickle=False) as truth:
                    truth_donors, truth_values = truth["donor"], truth["response"]
                    training_scales = truth["primary_training_response_scales"]
                for epochs in epoch_grid:
                    job = _run_job(cache, output / f"jobs/inner_{i:02d}_epoch_{epochs}", epochs, seeds[0],
                                   settings, guard, pause_files, runner)
                    with np.load(job / "fixed_8.npz", allow_pickle=False) as prediction:
                        if not np.array_equal(truth_donors, prediction["donor"]):
                            raise ValueError("Development truth/prediction donor order mismatch")
                        target_columns = [prediction["protein_names"].tolist().index(p) for p in primary]
                        error = np.abs(prediction["prediction"][:, target_columns] - truth_values)
                    for d, donor in enumerate(truth_donors):
                        for p, protein in enumerate(primary):
                            tuning.append({"inner_fold": i, "epochs": epochs, "donor": donor,
                                           "protein": protein, "absolute_error": float(error[d, p]),
                                           "training_response_scale": float(training_scales[p]),
                                           "standardized_absolute_error": float(error[d, p] / training_scales[p])})
            table = pd.DataFrame(tuning)
            scores = table.groupby("epochs", sort=True).standardized_absolute_error.mean()
            selected = int(scores.idxmin())
            table.to_csv(output / "development_tuning.csv", index=False)
            write_json(output / "epoch_selection.json", {"selected_epochs": selected,
                       "donor_target_mean_standardized_mae": {str(k): float(v) for k, v in scores.items()},
                       "scaling": "inner-training donor response SD; ddof=0; floor1e-6",
                       "panel": "fixed_8", "seed": seeds[0], "ties": "smaller epoch budget",
                       "validation_donors": roles["development_donors"]}, immutable=True)
            cache = output / "inputs/final"
            prepare_inputs(data, cache, roles["development_donors"], query_donors, panels,
                           settings, development_validation=False)
            jobs = []
            for seed in seeds:
                guard.check()
                job = _run_job(cache, output / f"jobs/final_seed_{seed}", selected, seed,
                               settings, guard, pause_files, runner)
                jobs.append(job)
            _write_role_predictions(output, "test", roles["test_donors"], jobs, seeds,
                                    panels, data.protein_names, selected)
            _write_role_predictions(output, "calibration", roles["calibration_donors"], jobs, seeds,
                                    panels, data.protein_names, selected)
        del data
        gc.collect()
        _verify_hashes(input_hashes)
        if contract["source_sha256"] != _source_hashes():
            raise ValueError("Scientific source changed during cell comparator run")
        guard.check()
        # Resource monitoring and interrupted attempts are mutable diagnostics;
        # every completed fit's scientific files have separate immutable receipts.
        outputs = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                   if p.is_file() and p.name not in {"resource_usage.json", ".run.lock", "PAUSE", "stopped.json"}}
        receipt = {"status": "completed", "mode": "runtime_pilot" if pilot_epochs else "cell_level_donor_fold",
                   "contract_sha256": fingerprint(contract), "output_sha256": outputs,
                   "test_donor": roles["test_donor"], "test_donors": roles["test_donors"],
                   "calibration_donors": roles["calibration_donors"], "query_target_values_read": False,
                   "heldout_accuracy_evaluated": False,
                   "resource_snapshot": json.loads((output / "resource_usage.json").read_text())}
        write_json(output / "receipt.json", receipt, immutable=True)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-donor")
    parser.add_argument("--manifest", type=Path, help="Optional explicit donor roles and fixed panels JSON")
    parser.add_argument("--epoch-grid", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261004, 20261005, 20261006])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minimum-cells", type=int, default=30)
    parser.add_argument("--max-rss-gib", type=float, default=32)
    parser.add_argument("--minimum-free-gib", type=float, default=30)
    parser.add_argument("--max-hours", type=float, default=168)
    parser.add_argument("--pause-file", type=Path, action="append", default=[])
    parser.add_argument("--runtime-pilot-epochs", type=int)
    parser.add_argument("--fit-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fit_job:
        fit_job(args.fit_job)
        return
    manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
    prepared = args.prepared or manifest.get("prepared")
    test_donor = args.test_donor or manifest.get("test_donor") or manifest.get("test_donors") or manifest.get("test")
    if not prepared or not test_donor or not args.output:
        parser.error("--prepared, --test-donor and --output are required (first two may be in --manifest)")
    run_cell_fold(prepared, args.output, test_donor, epoch_grid=args.epoch_grid, seeds=args.seeds,
                  threads=args.threads, manifest=args.manifest,
                  settings=Settings(threads=args.threads, n_hvg=args.n_hvg, batch_size=args.batch_size),
                  minimum_cells=args.minimum_cells, max_rss_gib=args.max_rss_gib,
                  minimum_free_gib=args.minimum_free_gib, max_hours=args.max_hours,
                  pause_files=args.pause_file, pilot_epochs=args.runtime_pilot_epochs)


if __name__ == "__main__":
    main()
