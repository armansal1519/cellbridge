"""AnnData tutorial on synthetic cells.

Scanpy is not required. If AnnData is installed, the same arrays are wrapped
in an AnnData object and passed to ``fit_anndata``. The numbers are a
software check, not a cohort result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cellbridge import fit_anndata, panel_curve


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    blocks_x = []
    blocks_y = []
    for donor in ("d0", "d1", "d2", "d3"):
        for arm, shift in (("control", 0.0), ("perturbed", 0.5)):
            n = 12
            genes = rng.normal(size=(n, 4))
            technical = rng.normal(size=(n, 2))
            anchors = rng.normal(size=(n, 2)) + shift
            protein = anchors[:, :1] + 0.3 * shift + rng.normal(scale=0.1, size=(n, 1))
            frames.append(pd.DataFrame({"donor": donor, "arm": arm}, index=range(n)))
            blocks_x.append(np.hstack([genes, technical, anchors]))
            blocks_y.append(protein)
    obs = pd.concat(frames, ignore_index=True)
    return obs, np.vstack(blocks_x), np.vstack(blocks_y)


class _Object:
    def __init__(self, x, obs, protein):
        self.X = x
        self.obs = obs
        self.obsm = {"protein": protein}


def as_anndata(x, obs, protein):
    try:
        import anndata as ad
    except ImportError:
        return _Object(x, obs, protein), "plain object"
    return ad.AnnData(x, obs=obs, obsm={"protein": protein}), "anndata"


def run():
    obs, x, protein = synthetic()
    adata, kind = as_anndata(x, obs, protein)
    model = fit_anndata(
        adata, n_genes=4, n_anchors=2, ks=(1,), mixes=((0.0, 0.0),),
        lambdas=(1.0,), strategy="shared_top", top=1.0, seed=0,
    )
    # Hold out no sealed donor. The query is the mean arm difference of d3,
    # which was also used for fitting. This is a software path check.
    d3 = obs.donor == "d3"
    stimulated = d3 & (obs.arm == "perturbed")
    control = d3 & (obs.arm == "control")
    query = x[stimulated.to_numpy()].mean(0) - x[control.to_numpy()].mean(0)
    point = model.predict(query)
    donors = []
    for name, group in obs.groupby("donor"):
        arms = {}
        for label in ("control", "perturbed"):
            rows = group.index[group.arm == label].to_numpy()
            arms[label] = (x[rows], protein[rows])
        donors.append({
            "donor": name,
            "z_control": arms["control"][0], "y_control": arms["control"][1],
            "z_perturbed": arms["perturbed"][0], "y_perturbed": arms["perturbed"][1],
        })
    curve = panel_curve(
        donors[:3], query, point, n_genes=4, anchor_counts=(0, 1, 2),
        ks=(1,), mixes=((0.0, 0.0),), lambdas=(1.0,), strategy="shared_top", top=1.0, seed=0,
    )
    return {"container": kind, "predict": point.reshape(-1).tolist(), "panel_curve": curve}


if __name__ == "__main__":
    print(run())
