"""CLI stages for gated-abundance-response (v0.5). Separate budget ledger."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import ResourceGuard, exclusive_run, freeze_implementation, freeze_protocol, require_receipt, write_json
from . import response_experiment_v050 as exp

PROJECT = Path(__file__).resolve().parents[2]


def run_response(args):
    args.run = args.run.resolve()
    protocol = freeze_protocol(args.protocol, args.run)
    if protocol.get("model") != "gated-abundance-response" or protocol.get("version") != "0.5.0":
        raise ValueError("v0.5 CLI requires gated-abundance-response 0.5.0")
    ledger = PROJECT / protocol["resources"].get("budget_ledger", "runs/compute_budget_v050")
    commands = ["initialize", "fit-predict", "evaluate"] if args.command == "benchmark" else [args.command]
    with exclusive_run(ledger), exclusive_run(args.run):
        freeze_implementation(args.run)
        guard = ResourceGuard(ledger, **{k: v for k, v in protocol["resources"].items()
                                         if k in {"max_rss_gib", "minimum_free_gib", "max_hours"}})
        guard.check()
        if "initialize" in commands or args.command == "benchmark":
            if args.groups is None:
                raise ValueError("--groups is required to initialize a v0.5 run")
            exp.initialize(args.groups, args.run, protocol, donor_split=getattr(args, "donor_split", None))
            print(json.dumps({"stage": "initialize", "status": "complete"}), flush=True)
        proteins = json.loads((args.run / "task.json").read_text())["proteins"]
        if "fit-predict" in commands or args.command in {"benchmark", "predict-response", "fit-response"}:
            require_receipt(args.run / "initialize")
            exp.fit_predict(args.run, protocol, proteins)
            print(json.dumps({"stage": "fit-predict", "status": "complete"}), flush=True)
        if "evaluate" in commands or args.command == "evaluate":
            unblind = protocol.get("evaluation_mode") != "independent_test" or bool(getattr(args, "unblind", False))
            if protocol.get("evaluation_mode") == "independent_test" and not unblind:
                write_json(args.run / "evaluation" / "waiting.json",
                           {"status": "sealed_awaiting_unblind", "rule": "Open outcomes only after P4 freeze"})
                print(json.dumps({"stage": "evaluate", "status": "sealed_waiting"}), flush=True)
            else:
                exp.evaluate(args.run, protocol, proteins, unblind=unblind)
                print(json.dumps({"stage": "evaluate", "status": "complete"}), flush=True)
        guard.check()
