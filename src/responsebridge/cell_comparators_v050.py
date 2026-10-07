"""Cell-level comparator adapters for v0.5. Missing runtimes are skips."""
from __future__ import annotations

from pathlib import Path

from .artifacts import write_json
from .comparators_v050 import CELL_ADAPTERS


def adapter_status(name):
    spec = CELL_ADAPTERS[name]
    missing = None
    if name == "totalvi_masked":
        try:
            import scvi  # noqa: F401
        except ImportError:
            missing = "scvi-tools_not_installed"
    elif name in {"scvaeit_retuned", "scvaeit_v040_locked"}:
        try:
            import scvaeit  # noqa: F401
        except ImportError:
            missing = "scvaeit_not_installed"
    elif name == "sclinear":
        try:
            from responsebridge._vendor.sclinear import sclinear  # noqa: F401
        except Exception as error:  # noqa: BLE001
            missing = f"sclinear_unavailable:{type(error).__name__}"
    elif name == "seurat_v4_mapping":
        from shutil import which
        if which("Rscript") is None:
            missing = "Rscript_not_on_path"
    if missing:
        return {**spec, "kind": name, "status": "skipped", "reason": missing}
    return {**spec, "kind": name, "status": "runtime_present_not_executed_in_this_call"}


def record_cell_adapter_receipts(folder, names=None):
    folder = Path(folder)
    names = list(names or CELL_ADAPTERS)
    rows = [adapter_status(name) for name in names]
    write_json(folder / "cell_adapter_status.json", {"adapters": rows}, immutable=True)
    return rows
