import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import anndata_tutorial


def test_panel_curve_runs_on_the_tutorial_object():
    out = anndata_tutorial.run()
    assert out["container"] in {"anndata", "plain object"}
    assert len(out["predict"]) == 1
    assert np.isfinite(out["predict"]).all()
    counts = [row["n_anchors"] for row in out["panel_curve"]]
    assert counts == [0, 1, 2]
    assert all(np.isfinite(row["mae"]) for row in out["panel_curve"])
