#!/usr/bin/env python3
"""Two-cohort meta-analysis of donor-level CellBridge gains.

Reads the locked GSE334503 donor losses and the E-MTAB-9357 evaluation.
Does not refit either model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from responsebridge.artifacts import write_json
from responsebridge.cellbridge_eval import cohort_effect, meta_analyze

ROOT = Path(__file__).resolve().parents[1]
GSE = ROOT / "runs/cellbridge_v100/gse334503/evaluation/donor_losses.csv"
EMTAB = ROOT / "runs/cellbridge_v100/emtab9357/evaluation"
PROTOCOL = ROOT / "configs/protocol_cellbridge_v100_emtab9357.json"


def _effects(frame, contrast, endpoint):
    test = frame[frame.role == "test"]
    model = test[(test.method == "cellbridge") & (test.endpoint == endpoint)].set_index("donor").loss
    other = test[(test.method == contrast) & (test.endpoint == endpoint)].set_index("donor").loss
    aligned = other.reindex(model.index)
    if not aligned.notna().all() or not len(model):
        return None
    return cohort_effect((aligned - model).to_numpy())


def main():
    decision = json.loads((EMTAB / "decision.json").read_text())
    if decision.get("dry_run"):
        raise SystemExit("The meta-analysis uses the real unblind, not the dry run")
    spec = json.loads(PROTOCOL.read_text())["meta_analysis"]
    gse = pd.read_csv(GSE)
    emtab = pd.read_csv(EMTAB / "donor_losses.csv")
    results = []
    for endpoint in spec["endpoints"]:
        for contrast in spec["contrasts"]:
            cohorts = []
            for name, frame in (("GSE334503", gse), ("E-MTAB-9357", emtab)):
                effect = _effects(frame, contrast, endpoint)
                if effect is not None:
                    cohorts.append({"cohort": name, **effect})
            if len(cohorts) == 2:
                results.append({"endpoint": endpoint, "contrast": contrast, **meta_analyze(cohorts)})
    out = ROOT / "runs/cellbridge_v100/meta"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "meta.json", {"role": spec["role"], "results": results})
    print(f"meta-analysis wrote {len(results)} contrasts")


if __name__ == "__main__":
    main()
