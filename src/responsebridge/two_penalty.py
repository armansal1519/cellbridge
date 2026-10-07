"""Donor-response ridge with independently penalized RNA and anchor blocks."""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from sklearn.linear_model import Ridge

from .response_baselines import LinearResponse, donor_weights, response_scale
from .shrinkage import _arrays, _check, _indices, _row_ids, _source_hash


def _penalty(alpha):
    if alpha is None:
        return None
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Block penalty must be None or a finite positive number")
    return float(alpha)


@dataclass
class TwoPenaltyResponse(LinearResponse):
    """Same feature selection/scales as LinearResponse('joint'), no block norms.

    None removes that block's slope, while retaining the response intercept.
    Predictors are reparameterized by sqrt(penalty), giving an ordinary unit
    ridge solve and avoiding a potentially large primal gene-space matrix.
    """
    kind: str = "joint"
    alpha: float | None = None
    n_hvg: int = 2000
    alpha_rna: float | None = 1.
    alpha_anchor: float | None = 1.

    def fit(self, x, y, anchors, *, donors=None):
        self.alpha_rna, self.alpha_anchor = _penalty(self.alpha_rna), _penalty(self.alpha_anchor)
        if self.kind != "joint":
            raise ValueError("TwoPenaltyResponse always uses the joint interface")
        if isinstance(self.n_hvg, bool) or not isinstance(self.n_hvg, (int, np.integer)) or self.n_hvg < 1:
            raise ValueError("n_hvg must be a positive integer")
        x, y, anchors = (np.asarray(v, dtype=float) for v in (x, y, anchors))
        # Parent alpha=None fits exactly the parent preprocessing and intercept.
        self.alpha = None
        super().fit(x, y, anchors, donors=donors)
        w = donor_weights(donors, len(y))
        penalties = np.array([np.inf if self.alpha_rna is None else self.alpha_rna] * len(self.features)
                             + [np.inf if self.alpha_anchor is None else self.alpha_anchor] * self.anchor_width)
        active = np.isfinite(penalties)
        if np.any(active):
            design = (self._combine(x, anchors)-self.x_mean) / self.x_scale
            factor = np.sqrt(penalties[active])
            reg = Ridge(alpha=1., fit_intercept=False, solver="cholesky")
            reg.fit(design[:, active] / factor, (y-self.y_mean)/self.y_scale, sample_weight=w)
            self.coef[:, active] = reg.coef_ / factor
        self.training_error_sd = np.maximum(np.sqrt(np.average(
            (y-self.predict(x, anchors))**2, axis=0, weights=w)), 1e-6)
        return self

    def save(self, path):
        metadata = {"format_version": 1, "kind": "two-penalty-ridge", "n_hvg": int(self.n_hvg),
            "alpha_rna": self.alpha_rna, "alpha_anchor": self.alpha_anchor,
            "rna_width": self.rna_width, "anchor_width": self.anchor_width}
        with open(path, "wb") as f:
            np.savez_compressed(f, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                **{k: getattr(self, k) for k in ("features", "x_mean", "x_scale", "y_mean", "y_scale",
                                                "coef", "training_error_sd")})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["format_version"] != 1 or metadata["kind"] != "two-penalty-ridge":
                raise ValueError("Unsupported two-penalty model format")
            obj = cls(n_hvg=metadata["n_hvg"], alpha_rna=metadata["alpha_rna"],
                      alpha_anchor=metadata["alpha_anchor"])
            obj.rna_width, obj.anchor_width = metadata["rna_width"], metadata["anchor_width"]
            for key in ("features", "x_mean", "x_scale", "y_mean", "y_scale", "coef", "training_error_sd"):
                setattr(obj, key, data[key].copy())
            return obj


def tune_two_penalty(x, y, donors, anchors, targets,
        alphas=(.01, .1, 1., 10., 100., None), *, n_hvg=2000, guard=None):
    """Train-only LODO target-response MAE tuning over the Cartesian penalty grid."""
    x, y, donors = _arrays(x, y, donors)
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 2 or len(anchors) != len(y) or not np.isfinite(anchors).all():
        raise ValueError("Anchors must be a finite training matrix aligned to responses")
    unique = np.unique(donors)
    if len(unique) < 2:
        raise ValueError("Two-penalty tuning requires at least two donors")
    targets = _indices(targets, y.shape[1], "targets", nonempty=True)
    grid = [_penalty(a) for a in alphas]
    if not grid:
        raise ValueError("Two-penalty grid must not be empty")
    records, predictions = [], []
    for ar in grid:
        for aa in grid:
            errors, fits = [], []
            oof = np.empty_like(y)
            for donor in unique:
                _check(guard)
                tr, va = donors != donor, donors == donor
                model = TwoPenaltyResponse(n_hvg=n_hvg, alpha_rna=ar, alpha_anchor=aa).fit(
                    x[tr], y[tr], anchors[tr], donors=donors[tr])
                oof[va] = model.predict(x[va], anchors[va])
                errors.append(float(np.mean(np.abs((oof[va]-y[va]) /
                                                    response_scale(y[tr], donors[tr]))[:, targets])))
                fits.append({"validation_donor": str(donor), "training_donors": np.unique(donors[tr]).tolist(),
                    "training_source_hash": _source_hash(_row_ids(np.column_stack([x[tr], anchors[tr]]), y[tr], donors[tr])),
                    "features": model.features.tolist(), "target_scales": model.y_scale[targets].tolist()})
            records.append({"alpha_rna": ar, "alpha_anchor": aa,
                "mae": float(np.mean(errors)), "donor_errors": dict(zip(unique.tolist(), errors)),
                "inner_fits": fits})
            predictions.append(oof)
    minimum = min(v["mae"] for v in records)
    tied = [i for i, v in enumerate(records) if np.isclose(v["mae"], minimum, atol=1e-12, rtol=1e-10)]
    # Fixed order for joint ties: strongest anchor penalty, then RNA penalty.
    strength = lambda a: np.inf if a is None else a
    best = max(tied, key=lambda i: (strength(records[i]["alpha_anchor"]), strength(records[i]["alpha_rna"])))
    chosen = records[best]
    _check(guard)
    model = TwoPenaltyResponse(n_hvg=n_hvg, alpha_rna=chosen["alpha_rna"],
        alpha_anchor=chosen["alpha_anchor"]).fit(x, y, anchors, donors=donors)
    return model, {"selected_alpha_rna": chosen["alpha_rna"], "selected_alpha_anchor": chosen["alpha_anchor"],
        "selected_index": best, "scores": records,
        "selection": "Equal-donor LODO target MAE standardized by training response SD",
        "tie_rule": "Within atol=1e-12/rtol=1e-10 of minimum: greatest anchor penalty then greatest RNA penalty; None strongest"}, predictions[best]
