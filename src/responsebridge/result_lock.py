"""Tamper checks for the sealed GSE334503 CellBridge v1.0 result.

The lock records file hashes and the headline numbers regenerated from the
evaluation tables. It does not cover files created after the lock.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

LOCK_REL = Path("analyses/cellbridge_v100/results_lock.json")
SEALED_REL = Path("runs/cellbridge_v100/gse334503")
POSTHOC_REL = Path("runs/cellbridge_v100/gse334503_posthoc")
DEVELOPMENT_RELS = (
    Path("runs/cellbridge_v100/development"),
    Path("runs/cellbridge_v100/development_tau"),
    Path("runs/cellbridge_v100/development_tau_union"),
    Path("runs/cellbridge_v100/analysis"),
    Path("runs/cellbridge_v100/gse334503_dryrun"),
)

# Byte-identical to scripts/cellbridge_gse_v100.py LOCKED_SOURCES. Listed here
# so the verifier does not import that script.
LOCKED_SOURCES = [
    "src/responsebridge/cellbridge.py",
    "src/responsebridge/cellbridge_data.py",
    "src/responsebridge/response_baselines.py",
    "src/responsebridge/abundance_response.py",
    "src/responsebridge/shrinkage.py",
    "src/responsebridge/rna.py",
    "src/responsebridge/benchmark_data.py",
    "src/responsebridge/response_experiment_v050.py",
    "src/responsebridge/reliability.py",
    "src/responsebridge/response_downstream_v050.py",
    "src/responsebridge/artifacts.py",
    "scripts/cellbridge_gse_v100.py",
    "scripts/cellbridge_dev_v100.py",
    "scripts/cellbridge_totalvi_v100.py",
]

# Rounded claims printed in analyses/cellbridge_v100/report.md.
REPORT_MAE = {
    ("cellbridge", "E1_primary"): 0.284,
    ("cellbridge", "E2_all_hidden"): 0.322,
    ("v040_abundance_shrinkage", "E1_primary"): 0.315,
    ("v040_abundance_shrinkage", "E2_all_hidden"): 0.359,
    ("totalvi", "E1_primary"): 0.393,
    ("totalvi", "E2_all_hidden"): 0.395,
    ("cellbridge_rna_only", "E1_primary"): 0.376,
    ("cellbridge_rna_only", "E2_all_hidden"): 0.461,
    ("mean", "E1_primary"): 0.511,
    ("mean", "E2_all_hidden"): 0.643,
    ("joint_ridge", "E1_primary"): 0.756,
    ("joint_ridge", "E2_all_hidden"): 0.722,
    ("rna_ridge", "E1_primary"): 0.799,
    ("rna_ridge", "E2_all_hidden"): 0.755,
}
REPORT_STRINGS = ("0.284", "0.322", "0.154", "0.04999", "primary claim")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _root_from(here: Path | None) -> Path:
    if here is not None:
        return Path(here)
    return Path(__file__).resolve().parents[2]


def locked_paths(root: Path) -> list[Path]:
    """Files covered by the results lock, relative to the package root."""
    paths = [p for p in (root / SEALED_REL).rglob("*") if p.is_file()]
    paths += [root / rel for rel in LOCKED_SOURCES]
    paths.append(root / "configs/protocol_cellbridge_v100.json")
    paths.append(root / "analyses/cellbridge_v100/report.md")
    paths.append(root / "analyses/cellbridge_v100/posthoc_policy.json")
    paths.append(root / "manuscript/v100/figure_main.png")
    paths += sorted(p for p in (root / "manuscript/v100/source_data").iterdir() if p.is_file())
    analysis = root / "runs/cellbridge_v100/analysis"
    paths += sorted(p for p in analysis.rglob("*.csv") if p.is_file())
    paths.append(root / "analyses/cellbridge_v100/env_venv.txt")
    paths.append(root / "analyses/cellbridge_v100/env_venv_totalvi.txt")
    unique = sorted({p.resolve() for p in paths})
    return unique


def headlines(root: Path) -> dict:
    """Regenerate the sealed-test headline numbers from the evaluation tables."""
    summary_path = root / SEALED_REL / "evaluation/summary.csv"
    decision_path = root / SEALED_REL / "evaluation/decision.json"
    methods: dict[str, dict] = {}
    with summary_path.open() as handle:
        for row in csv.DictReader(handle):
            methods.setdefault(row["method"], {})[row["endpoint"]] = {
                "standardized_mae": float(row["standardized_mae"]),
                "skill_vs_training_mean": float(row["skill_vs_training_mean"]),
                "test_donors": int(row["test_donors"]),
            }
    decision = json.loads(decision_path.read_text())
    hypotheses = []
    for item in decision["hypotheses"]:
        hypotheses.append({
            "id": item["id"],
            "endpoint": item["endpoint"],
            "comparator": item["comparator"],
            "donor_wins": item["donor_wins"],
            "n_donors": item["n_donors"],
            "mean_difference": item["mean_difference"],
            "sign_flip_p_two_sided": item["sign_flip_p_two_sided"],
            "rejected_null": item["rejected_null"],
            "tested_in_sequence": item["tested_in_sequence"],
        })
    return {
        "methods": methods,
        "hypotheses": hypotheses,
        "noninferiority_vs_v040": decision["noninferiority_vs_v040"],
        "primary_claim_supported": decision["primary_claim_supported"],
        "sealed_at": decision["sealed_at"],
    }


def _check_report_claims(root: Path, numbers: dict) -> list[str]:
    errors = []
    report = (root / "analyses/cellbridge_v100/report.md").read_text()
    for (method, endpoint), expected in REPORT_MAE.items():
        got = round(numbers["methods"][method][endpoint]["standardized_mae"], 3)
        if got != expected:
            errors.append(f"{method} {endpoint} rounds to {got}, report claims {expected}")
    for text in REPORT_STRINGS:
        if text not in report:
            errors.append(f"report.md is missing {text!r}")
    e1 = next(x for x in numbers["noninferiority_vs_v040"] if x["endpoint"] == "E1_primary")
    if not e1["noninferior"] or e1["upper_ci95_excess_loss"] >= 0.05:
        errors.append("E1 non-inferiority bound is no longer below the 0.05 margin")
    if not f"{e1['upper_ci95_excess_loss']:.8f}".startswith("0.04999"):
        errors.append("E1 upper bound no longer prints as 0.04999")
    h9 = next(h for h in numbers["hypotheses"] if h["id"] == "H9")
    if h9["rejected_null"] or h9["donor_wins"] != 7:
        errors.append("H9 outcome changed")
    if not numbers["primary_claim_supported"]:
        errors.append("primary claim is no longer supported")
    rejected = [h["id"] for h in numbers["hypotheses"] if h["rejected_null"]]
    if rejected != [f"H{i}" for i in range(1, 9)]:
        errors.append(f"rejected hypotheses are {rejected}")
    return errors


def verify(root: Path | None = None, *, require_readonly: bool = False) -> dict:
    root = _root_from(root)
    lock = json.loads((root / LOCK_REL).read_text())
    errors = []
    recorded = lock["files_sha256"]
    current = {str(p.relative_to(root)): sha256(p) for p in locked_paths(root)}
    for rel, digest in recorded.items():
        if current.get(rel) != digest:
            errors.append(f"hash mismatch: {rel}")
    extra = sorted(set(current) - set(recorded))
    missing = sorted(set(recorded) - set(current))
    if extra:
        errors.append(f"unrecorded locked paths: {extra}")
    if missing:
        errors.append(f"missing locked paths: {missing}")
    numbers = headlines(root)
    if numbers != lock["headlines"]:
        errors.append("regenerated headlines differ from the lock")
    errors.extend(_check_report_claims(root, numbers))
    frozen_sources = json.loads((root / SEALED_REL / "freeze.json").read_text())["sources_sha256"]
    if sorted(frozen_sources) != sorted(LOCKED_SOURCES):
        errors.append("LOCKED_SOURCES drifted from freeze.json")
    for rel, digest in frozen_sources.items():
        if sha256(root / rel) != digest:
            errors.append(f"frozen source changed after the GSE freeze: {rel}")
    if require_readonly:
        probe = root / SEALED_REL / "freeze.json"
        if os.access(probe, os.W_OK):
            errors.append(f"{probe} is still writable")
    if errors:
        raise AssertionError("\n".join(errors))
    return {"files": len(recorded), "sealed_at": numbers["sealed_at"]}


def _pip_freeze(python: Path) -> str:
    # The project virtualenvs have no pip module. Read installed distributions directly.
    code = (
        "import importlib.metadata as m\n"
        "rows=[]\n"
        "for dist in m.distributions():\n"
        "    name=dist.metadata['Name']\n"
        "    if name:\n"
        "        rows.append(f'{name}=={dist.version}')\n"
        "print('\\n'.join(sorted(rows, key=str.lower)))\n"
    )
    proc = subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True)
    return proc.stdout


def write_environment(root: Path) -> None:
    dest = root / "analyses/cellbridge_v100"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "env_venv.txt").write_text(_pip_freeze(root / ".venv/bin/python"))
    (dest / "env_venv_totalvi.txt").write_text(_pip_freeze(root / ".venv_totalvi/bin/python"))
    policy = {
        "sealed_run": str(SEALED_REL),
        "posthoc_run": str(POSTHOC_REL),
        "rule": (
            "Any later analysis of the GSE334503 test writes only under posthoc_run. "
            "Every JSON record there sets post_hoc to true. The sealed run stays read-only."
        ),
        "timestamp_note": (
            "prediction_seal.json uses the laptop clock. It is not a public timestamp "
            "until the author uploads archive/cellbridge_v1.0.0.tar.gz."
        ),
    }
    (dest / "posthoc_policy.json").write_text(json.dumps(policy, indent=2) + "\n")


def write_lock(root: Path | None = None) -> Path:
    root = _root_from(root)
    files = {str(p.relative_to(root)): sha256(p) for p in locked_paths(root)}
    payload = {
        "version": "1.0.0",
        "purpose": "Tamper-evident lock of the GSE334503 CellBridge v1.0 sealed test.",
        "python": sys.version,
        "platform": platform.platform(),
        "files_sha256": files,
        "headlines": headlines(root),
    }
    path = root / LOCK_REL
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def protect(root: Path | None = None) -> None:
    """Remove write permission from the sealed run and the development runs it selected on."""
    root = _root_from(root)
    for rel in (SEALED_REL, *DEVELOPMENT_RELS):
        base = root / rel
        if not base.exists():
            raise FileNotFoundError(base)
        for dirpath, dirnames, filenames in os.walk(base):
            for name in dirnames + filenames:
                os.chmod(Path(dirpath) / name, 0o555 if name in dirnames or Path(dirpath, name).is_dir() else 0o444)
            os.chmod(dirpath, 0o555)


def write_posthoc(root: Path, name: str, payload: dict) -> Path:
    """Write a post-hoc GSE334503 analysis record. Refuses the sealed directory."""
    root = Path(root)
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise PermissionError(f"post-hoc name must be a relative path inside {POSTHOC_REL}")
    dest = (root / POSTHOC_REL / name).resolve()
    sealed = (root / SEALED_REL).resolve()
    if sealed == dest or sealed in dest.parents:
        raise PermissionError(f"refusing to write inside the sealed run: {dest}")
    if (root / POSTHOC_REL).resolve() not in dest.parents:
        raise PermissionError(f"post-hoc records must live under {POSTHOC_REL}")
    body = dict(payload)
    body["post_hoc"] = True
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return dest
