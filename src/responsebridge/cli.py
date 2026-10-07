"""Command line for independent, resumable ResponseBridge stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent


def parser():
    p = argparse.ArgumentParser(description="ResponseBridge: sparse-panel protein response research")
    p.add_argument("--version", action="version", version="ResponseBridge 0.5.0")
    sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Inspect raw assay metadata without fitting")
    audit.add_argument("--dataset", choices=["gse278572", "lawlor"], default="gse278572")
    audit.add_argument("--raw", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    prep = sub.add_parser("prepare", help="Prepare RNA-QC cells and disjoint control group means")
    prep.add_argument("--dataset", choices=["gse278572", "lawlor"], required=True)
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--groups", type=Path)
    dl = sub.add_parser("download-lawlor", help="Checksum-verified public data acquisition")
    dl.add_argument("--output", type=Path, default=WORKSPACE/"data/immune_protein/raw/lawlor")
    group = sub.add_parser("aggregate", help="Aggregate a canonical prepared dataset")
    group.add_argument("--prepared", type=Path, required=True)
    group.add_argument("--output", type=Path, required=True)
    group.add_argument("--control-blocks", type=int, default=12)
    sim = sub.add_parser("simulate", help="Execute synthetic information-limit and confounding tests")
    sim.add_argument("--output", type=Path, default=PROJECT/"runs/simulation_v1")
    stress = sub.add_parser("response-stress", help="Controlled donor-response robustness simulations")
    stress.add_argument("--output", type=Path, required=True)
    stress.add_argument("--model", choices=["response-shrinkage", "abundance-response-shrinkage", "gated-abundance-response"], default="response-shrinkage")
    for command in ["fit-rna", "fit-response", "select-panel", "make-query", "predict-response", "calibrate", "explain", "sclinear-benchmark", "scvaeit-benchmark", "diagnose", "evaluate", "report", "benchmark"]:
        cmd = sub.add_parser(command)
        cmd.add_argument("--run", type=Path, required=True)
        cmd.add_argument("--model", choices=["legacy", "response-shrinkage", "abundance-response-shrinkage", "gated-abundance-response"], default="legacy")
        cmd.add_argument("--protocol", type=Path)
        if command in {"benchmark", "scvaeit-benchmark", "sclinear-benchmark"}:
            cmd.add_argument("--prepared", type=Path, help="Canonical cell-level inputs for published comparators")
        if command in {"fit-rna", "fit-response", "benchmark"}:
            cmd.add_argument("--donor-split", type=Path, help="Frozen independent donor roles, metadata allocated before outcomes")
        if command in {"evaluate", "benchmark"}:
            cmd.add_argument("--unblind", action="store_true", help="Open independent test outcomes after prediction seal (v0.5 only)")
        if command == "benchmark":
            cmd.add_argument("--skip-scvaeit", action="store_true", help="Explicit incomplete comparator run; advancement remains blocked")
            cmd.add_argument("--skip-sclinear", action="store_true", help="v0.4 incomplete diagnostic only; published linear comparator omitted")
        if command in {"fit-rna", "benchmark"}:
            cmd.add_argument("--groups", type=Path, required=True)
            cmd.add_argument("--axis", choices=["perturbation_gene", "donor", "context"], default=None)
            cmd.add_argument("--limit-folds", type=int, help="Explicitly labeled pilot, never a full benchmark claim")
            cmd.add_argument("--lineage", help="Restrict to this prespecified lineage before partitioning; omission fits a shared basis")
        if command == "fit-response":
            cmd.add_argument("--groups", type=Path, help="Initialize a new response-native run, or omit after fit-rna")
            cmd.add_argument("--limit-folds", type=int)
    return p


def _aggregate(prepared, output, control_blocks=12):
    from .datasets import load_prepared
    from .benchmark_data import aggregate
    from .artifacts import complete_stage, validate_receipt
    if validate_receipt(output): return
    data = load_prepared(prepared)
    result = aggregate(data.rna, data.proteins, data.obs, data.gene_names, data.protein_names,
        control_blocks=control_blocks, technical_counts=data.isotypes, technical_names=data.isotype_names)
    result.save(output)
    complete_stage(output, {"source": str(Path(prepared).resolve()), "control_blocks": control_blocks,
        "normalization": "Mean cellwise independent log1p ADT, RNA log1p after 1e4 per-cell normalization",
        "model_unit": "Donor/context/perturbation means; affine prediction commutes with averaging"})
    print(json.dumps({"groups": len(result.obs), "genes": len(result.genes), "proteins": len(result.proteins)}))


def main(argv=None):
    args = parser().parse_args(argv)
    # Bound BLAS parallelism before importing numpy/scipy.
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(key, "6")
    from .artifacts import ResourceGuard, freeze_protocol, freeze_implementation, exclusive_run, write_json
    if args.command == "download-lawlor":
        from .datasets import download_lawlor
        print(json.dumps(download_lawlor(args.output), indent=2)); return
    if args.command == "audit":
        from . import datasets
        fn = getattr(datasets, "audit_gse278572" if args.dataset == "gse278572" else "audit_lawlor")
        result = fn(args.raw)
        if args.output: write_json(args.output, result)
        print(json.dumps(result, indent=2)); return
    if args.command == "prepare":
        from .prepare import prepare_gse278572, prepare_lawlor
        fn = prepare_gse278572 if args.dataset == "gse278572" else prepare_lawlor
        print(json.dumps(fn(args.raw, args.output), indent=2))
        if args.groups: _aggregate(args.output, args.groups)
        return
    if args.command == "aggregate":
        _aggregate(args.prepared, args.output, args.control_blocks); return
    if args.command == "simulate":
        from .simulation import run_simulations
        result = run_simulations(seed=20261004)
        write_json(args.output / "simulation.json", result, immutable=True)
        print(json.dumps(result, indent=2)); return
    if args.command == "response-stress":
        if args.model == "gated-abundance-response":
            from .response_stress_v050 import run_stress
            seed = 20261007
        elif args.model == "abundance-response-shrinkage":
            from .response_stress_v040 import run_stress
            seed = 20261005
        else:
            from .response_stress import run_stress
            seed = 20261004
        print(json.dumps(run_stress(args.output, seed=seed), indent=2)); return
    if args.protocol is None:
        args.protocol = PROJECT/"configs"/({"response-shrinkage":"protocol_response_v030.json",
            "abundance-response-shrinkage":"protocol_response_v040.json",
            "gated-abundance-response":"protocol_response_v050.json"}.get(args.model,"protocol_v4.json"))
    if args.model == "gated-abundance-response":
        from .response_cli_v050 import run_response
        return run_response(args)
    if args.model == "abundance-response-shrinkage":
        from .response_cli_v040 import run_response
        return run_response(args)
    if args.model == "response-shrinkage":
        from .response_cli import run_response
        return run_response(args)
    if args.command in {"calibrate", "explain"}:
        raise ValueError("This stage requires --model response-shrinkage")
    protocol = freeze_protocol(args.protocol, args.run)
    from threadpoolctl import threadpool_limits
    from . import experiment as exp
    # One experiment coordinator per project; the 168-hour accounting survives
    # resume and is shared across run directories rather than reset per fold.
    with exclusive_run(PROJECT/"runs/compute_budget"), exclusive_run(args.run), threadpool_limits(limits=protocol["resources"]["threads"]):
        freeze_implementation(args.run)
        guard = ResourceGuard(PROJECT/"runs/compute_budget", **{k: v for k, v in protocol["resources"].items() if k != "threads"})
        guard.check()
        commands = ["fit-rna", "fit-response", "select-panel", "make-query", "predict-response", "scvaeit-benchmark", "evaluate", "report"] if args.command == "benchmark" else [args.command]
        if args.command == "benchmark" and args.skip_scvaeit: commands.remove("scvaeit-benchmark")
        for command in commands:
            if (args.run / "PAUSE").exists(): raise RuntimeError("Run PAUSE marker requested stop")
            guard.check()
            print(json.dumps({"stage": command, "status": "running", "run": str(args.run)}), flush=True)
            if command == "fit-rna":
                from .benchmark_data import GroupData
                from .artifacts import require_receipt
                require_receipt(args.groups)
                data = GroupData.load(args.groups)
                lineage = args.lineage or protocol.get("cohort_lineage")
                if args.lineage and protocol.get("cohort_lineage") and args.lineage != protocol["cohort_lineage"]:
                    raise ValueError("Lineage override conflicts with the frozen protocol")
                if lineage:
                    import numpy as np
                    indices = np.flatnonzero(data.obs.lineage.astype(str).to_numpy() == lineage)
                    if not len(indices): raise ValueError(f"Unknown or empty lineage: {lineage}")
                    data = data.subset(indices)
                axis = args.axis or protocol["split_axes"][0]
                if axis not in protocol["split_axes"]: raise ValueError("Split axis is not enabled by the protocol")
                exp.fit_rna_stage(data, args.run, protocol, axis=axis, limit=args.limit_folds, guard=guard)
            elif command == "fit-response": exp.fit_response_stage(args.run, protocol, guard)
            elif command == "select-panel": exp.select_panel_stage(args.run, protocol)
            elif command == "make-query": exp.mask_queries_stage(args.run)
            elif command == "predict-response": exp.predict_stage(args.run, protocol, guard)
            elif command == "scvaeit-benchmark":
                import subprocess
                import time
                worker = PROJECT/".venv-scvaeit/bin/python"
                if not worker.exists(): raise RuntimeError("Missing optional scVAEIT environment; see docs/scvaeit.md")
                for fold in sorted(args.run.glob("fold_*")):
                    from .artifacts import require_receipt, validate_scvaeit
                    for prerequisite in ["rna", "panels", "queries"]: require_receipt(fold/prerequisite)
                    destination = fold / "scvaeit"
                    if validate_scvaeit(destination): continue
                    command_args = [str(worker), "-m", "responsebridge.scvaeit", "--benchmark-fold", str(fold),
                        "--output", str(destination), "--epochs", str(protocol["scvaeit"]["epochs"]),
                        "--n-hvg", str(protocol["preprocessing"]["rna_hvg"]), "--threads", str(protocol["resources"]["threads"]),
                        "--seed", str(protocol["seed"])]
                    env = dict(os.environ, PYTHONPATH=str(PROJECT/"src"))
                    with (fold/"scvaeit_worker.log").open("a") as log:
                        process = subprocess.Popen(command_args, env=env, stdout=log, stderr=subprocess.STDOUT)
                        guard.extra_pids = (process.pid,)
                        try:
                            while process.poll() is None:
                                guard.check()
                                if (args.run/"PAUSE").exists(): raise RuntimeError("Run PAUSE marker requested stop")
                                time.sleep(2)
                            if process.returncode: raise RuntimeError(f"scVAEIT worker failed ({process.returncode}); inspect {fold/'scvaeit_worker.log'}")
                        finally:
                            if process.poll() is None:
                                process.terminate()
                                try: process.wait(timeout=10)
                                except subprocess.TimeoutExpired: process.kill(); process.wait()
                            guard.extra_pids = ()
            elif command == "evaluate": exp.evaluate_stage(args.run, protocol)
            elif command == "report":
                from .report import render_report
                print(render_report(args.run))
            guard.check()
            print(json.dumps({"stage": command, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
