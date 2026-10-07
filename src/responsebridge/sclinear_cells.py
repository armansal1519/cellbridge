"""Cell-level scLinear published-core comparator, with explicit assay adaptation.

The pinned original core is executed, rather than a reimplementation labelled
official. Only its preprocessing and linear fit/predict methods are loaded; the
unused deep-learning imports and feature-importance code are not executed.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional, Tuple
import warnings

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

from .artifacts import ResourceGuard, ResourceLimit, exclusive_run, fingerprint, sha256, write_json
from . import scvaeit_cells_v040 as cells

VENDOR = Path(__file__).parent / "_vendor/sclinear"
CORE_METHODS = {"__init__", "fit", "_filter_gex_names", "predict"}


def provenance():
    source = json.loads((VENDOR / "SOURCE.json").read_text())
    for name, entry in source["files"].items():
        if sha256(VENDOR / name) != entry["sha256"]:
            raise ValueError(f"Pinned scLinear source integrity failure: {name}")
    return source


def _source_hashes():
    provenance()
    paths = [Path(__file__), Path(cells.__file__), Path(__file__).with_name("artifacts.py")]
    paths += [VENDOR / name for name in ("SOURCE.json", "prediction.py", "preprocessing.py", "LICENSE.md")]
    return {str(p.relative_to(Path(__file__).parent)): sha256(p) for p in paths}


def official_core():
    """Load exact pinned AST nodes; adapters do not alter their bodies."""
    provenance()
    import anndata as ad
    import scanpy as sc
    ns = {"np": np, "ad": ad, "sc": sc, "AnnData": ad.AnnData,
          "Optional": Optional, "Tuple": Tuple, "warnings": warnings,
          "TruncatedSVD": TruncatedSVD, "LinearRegression": LinearRegression}
    pre = ast.parse((VENDOR / "preprocessing.py").read_text())
    selected = [n for n in pre.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in {"GEXPreprocessor", "zscore_normalization"}]
    if len(selected) != 2:
        raise ValueError("Pinned scLinear preprocessing core not found")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(VENDOR / "preprocessing.py"), "exec"), ns)
    tree = ast.parse((VENDOR / "prediction.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ADTPredictor")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in CORE_METHODS]
    if {n.name for n in cls.body} != CORE_METHODS:
        raise ValueError("Pinned scLinear prediction core not found")
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(VENDOR / "prediction.py"), "exec"), ns)
    return ns["ADTPredictor"]


def prepare_inputs(data, folder, train_donors, query_donors):
    """No query protein values, including anchors, are accessed for RNA-only."""
    folder = Path(folder)
    if set(train_donors) & set(query_donors):
        raise ValueError("Training and query donor roles overlap")
    contract = {"train_donors": sorted(train_donors), "query_donors": sorted(query_donors),
                "cohort": asdict(data.cohort), "input_sha256": data.input_sha256,
                "preprocessing": "all RNA genes; cellwise log1p library-normalized RNA; independent log1p ADT"}
    if (folder / "receipt.json").exists():
        saved = json.loads((folder / "receipt.json").read_text())
        if saved["contract"] != contract:
            raise ValueError("Input-cache contract changed; use a new run")
        cells._verify_outputs(folder, saved["output_sha256"])
        return saved
    if folder.exists() and any(folder.iterdir()):
        i = 0
        while folder.with_name(folder.name + f".incomplete_{i:03d}").exists():
            i += 1
        folder.rename(folder.with_name(folder.name + f".incomplete_{i:03d}"))
    folder.mkdir(parents=True, exist_ok=True)
    # Canonical cell order makes the randomized SVD insensitive to input row order.
    tr = data.obs.loc[data.obs.donor.isin(train_donors)].sort_values("cell_id").index.to_numpy()
    qr = data.obs.loc[data.obs.donor.isin(query_donors)].sort_values("cell_id").index.to_numpy()
    tro, qro = data.obs.loc[tr], data.obs.loc[qr]
    if set(tro.donor) != set(train_donors) or set(qro.donor) != set(query_donors):
        raise ValueError("Missing requested donor cells")
    cells.cell_weights(tro, data.cohort.arms)  # validate pairs; official core uses equal-cell fitting
    sparse.save_npz(folder / "train_rna.npz", data.rna[tr])
    sparse.save_npz(folder / "query_rna.npz", data.rna[qr])
    y = cells._log_counts(data.proteins[np.ix_(tro.prepared_row.to_numpy(), np.arange(len(data.protein_names)))])
    np.save(folder / "train_proteins.npy", y)
    np.savez_compressed(folder / "query_metadata.npz", donor=qro.donor.to_numpy(dtype=str),
                        condition=qro.condition.to_numpy(dtype=str), prepared_rows=qro.prepared_row.to_numpy())
    rows = data.obs.loc[np.r_[tr, qr], ["cell_id", "prepared_row", "donor", "condition"]].copy()
    rows["role"] = ["training"] * len(tr) + ["query"] * len(qr)
    rows.to_csv(folder / "cell_roles.csv", index=False)
    write_json(folder / "features.json", {"genes": data.genes, "proteins": data.protein_names}, immutable=True)
    receipt = {"contract": contract, "output_sha256": cells._files(folder),
               "training_cells": len(tr), "query_cells": len(qr), "query_protein_values_read": False}
    write_json(folder / "receipt.json", receipt, immutable=True)
    return receipt


def fit_job(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    out = Path(spec["output"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit output must be new or empty")
    cells._verify_hashes(spec["input_sha256"])
    if spec["source_sha256"] != _source_hashes():
        raise ValueError("scLinear source differs from frozen specification")
    cache = Path(spec["cache"])
    x = sparse.load_npz(cache / "train_rna.npz").tocsr()
    y = np.load(cache / "train_proteins.npy", allow_pickle=False)
    if not np.isfinite(x.data).all() or not np.isfinite(y).all() or len(y) != x.shape[0]:
        raise ValueError("Finite aligned training arrays required")
    components = min(int(spec["n_components"]), x.shape[1], x.shape[0] - 1)
    if components < 2:
        raise ValueError("scLinear needs at least two components for row z-score")
    features = json.loads((cache / "features.json").read_text())
    model = official_core()(do_log1p=False, n_components=components, do_tsvd_before_zscore=True)
    model.gex_preprocessor.tsvd.set_params(random_state=int(spec["seed"]))
    started = time.monotonic()
    model.fit(x, y, gex_test=None, gex_names=np.asarray(features["genes"]), adt_names=np.asarray(features["proteins"]))
    training_seconds = time.monotonic() - started
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "model.npz", components=model.gex_preprocessor.tsvd.components_,
                        coefficients=model.model.coef_, intercept=model.model.intercept_,
                        explained_variance_ratio=model.gex_preprocessor.tsvd.explained_variance_ratio_)
    qx = sparse.load_npz(cache / "query_rna.npz")
    cohort = cells.CohortSpec(**spec["cohort"])
    with np.load(cache / "query_metadata.npz", allow_pickle=False) as q:
        if set(q.files) != {"donor", "condition", "prepared_rows"}:
            raise ValueError("Query metadata schema contains unexpected features")
        pred, names = model.predict(qx)
        if not np.array_equal(names, np.asarray(features["proteins"])):
            raise ValueError("scLinear output protein order mismatch")
        result = cells.aggregate_responses(pred, q["donor"], q["condition"], cohort.arms)
    np.savez_compressed(out / "responses.npz", **result, protein_names=names)
    cells._verify_hashes(spec["input_sha256"])
    if spec["source_sha256"] != _source_hashes():
        raise ValueError("scLinear source changed during fit")
    receipt = {"spec_sha256": sha256(spec_path), "input_sha256": spec["input_sha256"],
               "source_sha256": spec["source_sha256"], "output_sha256": cells._files(out),
               "published_core": provenance(), "requested_components": spec["n_components"],
               "actual_components": components, "seed": spec["seed"], "training_seconds": training_seconds,
               "query_target_values_read": False, "query_protein_values_read": False,
               "transductive_preprocessing": False, "heldout_accuracy_evaluated": False,
               "versions": {k: importlib.metadata.version(k) for k in
                            ("numpy", "scipy", "scikit-learn", "anndata", "scanpy")},
               "weighting": "published equal-cell fit; equal-donor response evaluation",
               "uncertainty": "no intrinsic calibrated interval; external donor calibration required"}
    write_json(out / "receipt.json", receipt, immutable=True)
    return receipt


def _runner(spec_path, log_path, guard, pause_files):
    env = os.environ.copy()
    threads = str(json.loads(Path(spec_path).read_text())["threads"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = threads
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    with Path(log_path).open("x") as log:
        child = subprocess.Popen([sys.executable, "-m", "responsebridge.sclinear_cells", "--fit-job", str(spec_path)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
        old = guard.extra_pids
        guard.extra_pids = (*old, child.pid)
        try:
            while child.poll() is None:
                if any(Path(p).exists() for p in pause_files):
                    raise ResourceLimit("External PAUSE file requested a stop")
                guard.check()
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"scLinear cell fit failed ({child.returncode}); inspect {log_path}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            guard.extra_pids = old


def validate_cell_receipt(folder):
    folder = Path(folder)
    if not (folder / "receipt.json").exists():
        return False
    receipt = json.loads((folder / "receipt.json").read_text())
    contract = json.loads((folder / "fold_lock.json").read_text())
    if receipt.get("status") != "completed" or receipt["contract_sha256"] != fingerprint(contract) or contract["source_sha256"] != _source_hashes():
        raise ValueError("scLinear cell fold/source integrity failure")
    cells._verify_hashes(contract["input_sha256"])
    cells._verify_outputs(folder, receipt["output_sha256"])
    required = {"fold_lock.json", "predictions.npz", "eligibility.json", "job_complete.json"}
    if contract["roles"]["calibration_donors"]:
        required.add("calibration_predictions.npz")
    if required - set(receipt["output_sha256"]):
        raise ValueError("scLinear receipt omits required artifacts")
    complete = json.loads((folder / "job_complete.json").read_text())
    attempt = folder / receipt["attempt"]
    spec = json.loads((attempt / "spec.json").read_text())
    fit_receipt = json.loads((attempt / "output/receipt.json").read_text())
    if (fit_receipt["source_sha256"] != contract["source_sha256"]
            or complete["receipt_sha256"] != sha256(attempt / "output/receipt.json")
            or fit_receipt["spec_sha256"] != sha256(attempt / "spec.json")
            or complete["contract"] != {k: v for k, v in spec.items() if k != "output"}
            or complete["attempt"] != receipt["attempt"]):
        raise ValueError("scLinear fit source integrity failure")
    cells._verify_hashes(spec["input_sha256"])
    cells._verify_outputs(attempt / "output", fit_receipt["output_sha256"])
    return True


def run_cell_fold(prepared, output, test_donor=None, *, manifest=None, minimum_cells=30,
                  seed=20261004, n_components=300, threads=6, max_rss_gib=32,
                  minimum_free_gib=30, max_hours=168, pause_files=(), runner=_runner):
    output, prepared = Path(output).resolve(), Path(prepared).resolve()
    if n_components < 2 or threads < 1 or seed < 0:
        raise ValueError("Valid component count, thread budget and seed required")
    manifest_path = Path(manifest).resolve() if manifest is not None else None
    declared = json.loads(manifest_path.read_text()) if manifest_path else {}
    with exclusive_run(output):
        guard = ResourceGuard(output, max_rss_gib, minimum_free_gib, max_hours)
        guard.check()
        cohort = cells.CohortSpec.from_manifest(declared)
        data = cells.load_cells(prepared, minimum_cells, cohort)
        roles = cells.donor_roles(data.obs.donor, test_donor, seed, declared)
        panels, _ = cells._panels(data.protein_names, declared)
        hashes = dict(data.input_sha256)
        if manifest_path:
            hashes[str(manifest_path)] = sha256(manifest_path)
        contract = {"schema_version": 1, "model_version": "0.4.0", "method": "sclinear_cell",
                    "prepared": str(prepared), "input_sha256": hashes, "source_sha256": _source_hashes(),
                    "roles": roles, "cohort": asdict(cohort), "seed": seed, "n_components": n_components,
                    "minimum_cells": minimum_cells, "panels": panels, "threads": threads,
                    "versions": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn")},
                    "provenance": provenance(), "selection": "no hyperparameter selection"}
        write_json(output / "fold_lock.json", contract, immutable=True)
        if (output / "receipt.json").exists():
            validate_cell_receipt(output)
            return json.loads((output / "receipt.json").read_text())
        write_json(output / "eligibility.json", data.eligibility, immutable=True)
        query = sorted(roles["test_donors"] + roles["calibration_donors"])
        cache = output / "inputs"
        prepare_inputs(data, cache, roles["development_donors"], query)
        common = {"cache": str(cache), "cohort": asdict(cohort),
                  "input_sha256": {str(p): sha256(p) for p in cache.iterdir() if p.is_file()},
                  "source_sha256": _source_hashes(), "seed": seed, "n_components": n_components, "threads": threads}
        if (output / "job_complete.json").exists():
            saved = json.loads((output / "job_complete.json").read_text())
            if saved["contract"] != common:
                raise ValueError("Completed scLinear fit contract changed")
            attempt = output / saved["attempt"]
            if sha256(attempt / "output/receipt.json") != saved["receipt_sha256"]:
                raise ValueError("scLinear worker receipt integrity failure")
        else:
            # A fit receipt written immediately before interruption is reusable.
            finished = []
            for candidate in sorted(output.glob("attempt_*")):
                if (candidate / "output/receipt.json").exists():
                    old_spec = json.loads((candidate / "spec.json").read_text())
                    if {k: v for k, v in old_spec.items() if k != "output"} != common:
                        raise ValueError("Prior scLinear attempt contract changed")
                    finished.append(candidate)
            if finished:
                attempt = finished[-1]
            else:
                i = 0
                while (output / f"attempt_{i:03d}").exists():
                    i += 1
                attempt = output / f"attempt_{i:03d}"
                attempt.mkdir()
                write_json(attempt / "spec.json", {**common, "output": str(attempt / "output")}, immutable=True)
                if any(Path(p).exists() for p in pause_files):
                    raise ResourceLimit("External PAUSE file requested a stop")
                guard.check()
                runner(attempt / "spec.json", attempt / "worker.log", guard, pause_files)
        job_receipt = json.loads((attempt / "output/receipt.json").read_text())
        if job_receipt["spec_sha256"] != sha256(attempt / "spec.json"):
            raise ValueError("scLinear worker specification integrity failure")
        cells._verify_outputs(attempt / "output", job_receipt["output_sha256"])
        write_json(output / "job_complete.json", {"contract": common, "attempt": attempt.name,
                   "receipt_sha256": sha256(attempt / "output/receipt.json")}, immutable=True)
        with np.load(attempt / "output/responses.npz", allow_pickle=False) as pred:
            for role, names in (("test", roles["test_donors"]), ("calibration", roles["calibration_donors"])):
                if not names:
                    continue
                indices = [pred["donor"].tolist().index(d) for d in names]
                arrays = {"seeds": np.asarray([seed]), "protein_names": pred["protein_names"],
                          "donor_ids": np.asarray(names),
                          **{p: pred["prediction"][indices][None, ...] for p in panels}}
                filename = "predictions.npz" if role == "test" else "calibration_predictions.npz"
                cells._save_arrays_immutable(output / filename, arrays)
        cells._verify_hashes(hashes)
        if contract["source_sha256"] != _source_hashes():
            raise ValueError("scLinear source changed during fold")
        guard.check()
        outputs = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                   if p.is_file() and p.name not in {"resource_usage.json", ".run.lock", "PAUSE", "stopped.json"}}
        receipt = {"status": "completed", "contract_sha256": fingerprint(contract), "output_sha256": outputs,
                   "attempt": attempt.name, "method": "sclinear_cell", "query_target_values_read": False,
                   "heldout_accuracy_evaluated": False, "published_core": provenance(),
                   "resource_snapshot": json.loads((output / "resource_usage.json").read_text())}
        write_json(output / "receipt.json", receipt, immutable=True)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "output", "manifest", "fit-job"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--test-donor")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--n-components", type=int, default=300)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=30)
    parser.add_argument("--max-rss-gib", type=float, default=32)
    parser.add_argument("--minimum-free-gib", type=float, default=30)
    parser.add_argument("--max-hours", type=float, default=168)
    parser.add_argument("--pause-file", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.fit_job:
        fit_job(args.fit_job)
        return
    if not args.prepared or not args.output:
        parser.error("--prepared and --output are required")
    run_cell_fold(args.prepared, args.output, args.test_donor, manifest=args.manifest, seed=args.seed,
                  n_components=args.n_components, threads=args.threads, minimum_cells=args.minimum_cells,
                  max_rss_gib=args.max_rss_gib, minimum_free_gib=args.minimum_free_gib,
                  max_hours=args.max_hours, pause_files=args.pause_file)


if __name__ == "__main__":
    main()
