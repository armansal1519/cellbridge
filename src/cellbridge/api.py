"""Array and AnnData wrappers around the frozen CellBridge solver."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from responsebridge.cellbridge import CellBridge, donor_stats
from responsebridge.cellbridge_data import feature_layout


@dataclass
class FittedCellBridge:
    """A fitted model. Predictions are an exact linear function of the query."""

    model: CellBridge
    donors: list
    n_genes: int
    n_anchors: int

    def predict(self, dz):
        """Predict every target from donor-level feature differences."""
        return np.asarray(self.model.predict(np.atleast_2d(dz)), dtype=float)

    def contributions(self, dz, target=0):
        """Per-feature contribution to one target. Columns sum to the prediction shift."""
        return np.asarray(self.model.contributions(np.atleast_2d(dz), int(target)), dtype=float)

    def intervals(self, dz, calibration_scores, alpha=0.1):
        """Split-conformal intervals from absolute calibration scores.

        ``calibration_scores`` are absolute errors on the same scale as
        ``predict``. The quantile uses the higher method.
        """
        scores = np.asarray(calibration_scores, dtype=float).ravel()
        if not len(scores) or np.any(scores < 0) or not np.isfinite(scores).all():
            raise ValueError("Calibration scores must be finite and nonnegative")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        half = float(np.quantile(scores, level, method="higher"))
        point = self.predict(dz)
        return point - half, point + half


def fit(donors, *, n_genes, n_anchors, ks=(1,), mixes=((0.0, 0.0), (1.0, 0.0)),
        lambdas=(1.0,), strategy="shared_top", top=0.05, seed=0):
    """Fit CellBridge from per-donor arm matrices.

    Each donor mapping needs ``z_control``, ``y_control``, ``z_perturbed`` and
    ``y_perturbed``. ``z`` already contains genes, two technical columns and
    the anchor proteins, in that order.
    """
    if len(donors) < 3:
        raise ValueError("CellBridge selection needs at least three donors")
    def columns(value):
        y = np.asarray(value, dtype=float)
        return y[:, None] if y.ndim == 1 else y

    stats = []
    for i, donor in enumerate(donors):
        stats.append(donor_stats(
            str(donor.get("donor", i)), donor["z_control"], columns(donor["y_control"]),
            donor["z_perturbed"], columns(donor["y_perturbed"]), ks=tuple(ks),
            rna_columns=np.arange(n_genes), seed=seed + 101 * i,
        ))
    layout = feature_layout(n_genes, n_anchors)
    model = CellBridge(
        layout, ks=tuple(ks), mixes=tuple(tuple(m) for m in mixes), lambdas=tuple(lambdas),
        strategy=strategy, top=top,
    ).fit(stats)
    return FittedCellBridge(model, [s.donor for s in stats], n_genes, n_anchors)


def _table(adata):
    obs = adata.obs
    if not isinstance(obs, pd.DataFrame):
        obs = pd.DataFrame(obs)
    x = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=float)
    return obs.reset_index(drop=True), x


def fit_anndata(adata, *, donor="donor", arm="arm", protein="protein", control="control",
                perturbed="perturbed", n_genes, n_anchors, **kwargs):
    """Fit from an AnnData object.

    ``X`` is the model matrix. ``obsm[protein]`` holds the protein matrix in
    the same cell order. ``obs`` names the donor and the arm.
    """
    if protein not in getattr(adata, "obsm", {}):
        raise ValueError(f"obsm[{protein!r}] is required")
    obs, x = _table(adata)
    y = np.asarray(adata.obsm[protein], dtype=float)
    if len(obs) != len(x) or len(y) != len(x):
        raise ValueError("Cells, features and proteins must be aligned")
    donors = []
    for name, group in obs.groupby(donor, sort=True):
        arms = {}
        for label in (control, perturbed):
            rows = group.index[group[arm].astype(str) == label].to_numpy()
            if len(rows) < 2:
                raise ValueError(f"{name} needs at least two cells in {label}")
            arms[label] = (x[rows], y[rows])
        donors.append({
            "donor": name,
            "z_control": arms[control][0], "y_control": arms[control][1],
            "z_perturbed": arms[perturbed][0], "y_perturbed": arms[perturbed][1],
        })
    return fit(donors, n_genes=n_genes, n_anchors=n_anchors, **kwargs)


def panel_curve(donors, query, observed, *, n_genes, anchor_counts, **fit_kwargs):
    """Refit with the first k anchors and return the mean absolute error.

    ``z`` columns are genes, two technical columns, then anchors. ``query`` is
    the donor-level difference of that matrix. ``observed`` is the matching
    protein response. ``anchor_counts`` lists how many leading anchors to keep.
    The solver is called as it stands. This function only slices columns.
    """
    query = np.atleast_2d(np.asarray(query, dtype=float))
    observed = np.atleast_2d(np.asarray(observed, dtype=float))
    if query.shape[0] != observed.shape[0]:
        raise ValueError("Query donors and observed responses must be aligned")
    width = int(np.asarray(donors[0]["z_control"]).shape[1])
    rows = []
    for k in anchor_counts:
        k = int(k)
        if k < 0 or n_genes + 2 + k > width:
            raise ValueError(f"{k} anchors do not fit in the feature matrix")
        cols = np.arange(n_genes + 2 + k)
        sliced = []
        for donor in donors:
            sliced.append({
                **donor,
                "z_control": np.asarray(donor["z_control"], dtype=float)[:, cols],
                "z_perturbed": np.asarray(donor["z_perturbed"], dtype=float)[:, cols],
            })
        model = fit(sliced, n_genes=n_genes, n_anchors=k, **fit_kwargs)
        predicted = model.predict(query[:, cols])
        rows.append({"n_anchors": k, "mae": float(np.mean(np.abs(predicted - observed)))})
    return rows
