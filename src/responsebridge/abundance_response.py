"""Abundance-trained RNA slope with donor-cross-fitted response correction.

Only whole donors may be removed from the supplied abundance design. Its
weights must already balance conditions/contexts within a donor; removing a
whole donor preserves those masses. Every nested fit receives only its own
training donors, including for abundance feature selection and scaling.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
from sklearn.linear_model import Ridge

from .rna import RNARegressor
from .shrinkage import (ResponseShrinkage, _alpha, _arrays, _check, _indices,
                        _row_ids, _shrinkage, _source_hash, _weights)


def _abundance_arrays(x, y, donors, ax, ay, ad, aw):
    ax, ay, ad = _arrays(ax, ay, ad)
    aw = np.asarray(aw, dtype=float)
    if ax.shape[1] != x.shape[1] or ay.shape[1] != y.shape[1]:
        raise ValueError("Abundance and response feature/protein widths must match")
    if set(ad) != set(donors):
        raise ValueError("Abundance and response training donors must match exactly")
    if aw.shape != (len(ax),) or not np.isfinite(aw).all() or np.any(aw <= 0):
        raise ValueError("Abundance weights must be finite, positive and aligned")
    aw = aw / aw.sum()
    masses = np.array([aw[ad == d].sum() for d in np.unique(ad)])
    if not np.allclose(masses, 1 / len(masses), atol=1e-12, rtol=1e-10):
        raise ValueError("Abundance weights must assign equal total mass to donors")
    # Weight is part of the provenance: identical arrays with a different
    # within-donor design must not share a training-source identity.
    ids = _row_ids(np.column_stack([ax, aw]), ay, ad)
    order = np.lexsort((ids, ad))
    return ax[order], ay[order], ad[order], aw[order], ids[order]


def _slope_model(ax, ay, aw, x, y, donors, alpha, n_hvg, formulation):
    """Fit train-only abundance transforms and the chosen training objective."""
    # Fit with a harmless finite penalty to retain transforms in the zero-slope
    # case. Its estimated coefficients are then discarded, never used.
    model = RNARegressor(1. if alpha is None else alpha, n_hvg).fit(ax, ay, aw)
    if formulation == "abundance":
        # g(X_stim)-g(X_control) cancels all affine location terms exactly.
        model.x_mean = np.zeros_like(model.x_mean)
        model.y_mean = np.zeros_like(model.y_mean)
        model.intercept = np.zeros_like(model.intercept)
        if alpha is None:
            model.coef = np.zeros_like(model.coef)
    else:
        # A matched response-objective ablation: retain the *same abundance*
        # features and x/y scales, while fitting a response intercept and slope.
        w = _weights(donors)
        model.x_mean = np.average(x[:, model.features], axis=0, weights=w)
        model.y_mean = np.average(y, axis=0, weights=w)
        model.intercept = np.zeros(y.shape[1])
        model.coef = np.zeros((y.shape[1], len(model.features)))
        if alpha is not None and len(model.features):
            xx = (x[:, model.features] - model.x_mean) / model.x_scale
            yy = (y - model.y_mean) / model.y_scale
            reg = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky")
            reg.fit(xx, yy, sample_weight=w)
            model.coef = reg.coef_
    return model


def _transform_record(model):
    return {"features": model.features.tolist(), "x_scale": model.x_scale.tolist(),
            "y_scale": model.y_scale.tolist()}


class AbundanceResponseShrinkage(ResponseShrinkage):
    """ResponseShrinkage prediction interface with abundance RNA training.

    ``x`` is the RNA response (stimulated minus control). ``abundance_x`` and
    ``abundance_y`` hold separate condition means, without held-out donors.
    ``formulation='matched-response'`` is the prespecified objective ablation.
    """

    def fit(self, x, y, donors, *, abundance_x, abundance_y, abundance_donors,
            abundance_weights, alpha=1., shrinkage=.5, n_hvg=2000,
            formulation="abundance", guard=None):
        x, y, donors = _arrays(x, y, donors)
        alpha, shrinkage = _alpha(alpha), _shrinkage(shrinkage)
        if formulation not in {"abundance", "matched-response"}:
            raise ValueError("Unknown abundance-response formulation")
        if isinstance(n_hvg, (bool, np.bool_)) or not isinstance(n_hvg, (int, np.integer)) or n_hvg < 1:
            raise ValueError("n_hvg must be a positive integer")
        unique = np.unique(donors)
        if len(unique) < 2:
            raise ValueError("Response residual fitting requires at least two donors")
        ax, ay, ad, aw, abundance_ids = _abundance_arrays(
            x, y, donors, abundance_x, abundance_y, abundance_donors, abundance_weights)
        response_ids = _row_ids(x, y, donors)
        order = np.lexsort((response_ids, donors))
        x, y, donors, response_ids = x[order], y[order], donors[order], response_ids[order]
        oof = np.empty_like(y)
        folds = []
        for donor in unique:
            _check(guard)
            train, val, atrain = donors != donor, donors == donor, ad != donor
            fitted = _slope_model(ax[atrain], ay[atrain], aw[atrain], x[train],
                y[train], donors[train], alpha, int(n_hvg), formulation)
            oof[val] = fitted.predict(x[val])
            folds.append({"validation_donor": str(donor),
                "training_donors": np.unique(donors[train]).tolist(),
                "training_source_hash": _source_hash(response_ids[train]),
                "abundance_training_donors": np.unique(ad[atrain]).tolist(),
                "abundance_source_hash": _source_hash(abundance_ids[atrain]),
                "validation_row_ids": response_ids[val].tolist(),
                "transforms": _transform_record(fitted)})
        _check(guard)
        final = _slope_model(ax, ay, aw, x, y, donors, alpha, int(n_hvg), formulation)
        abundance_hash = _source_hash(abundance_ids)
        response_hash = _source_hash(response_ids)
        source_hash = hashlib.sha256(json.dumps([response_hash, abundance_hash]).encode()).hexdigest()
        provenance = {"formulation": formulation, "residual_folds": folds,
            "abundance_source_hash": abundance_hash,
            "abundance_training_donors": np.unique(ad).tolist(),
            "abundance_training_rows": len(ad), "transforms": _transform_record(final),
            "abundance_weighting": "Supplied metadata-only condition-balanced weights; equal donor masses verified; only whole-donor partitions",
            "rna_objective": "Weighted abundance fit, then exact affine differencing" if formulation == "abundance"
                else "Response fit with matched abundance-selected features and abundance x/y scales",
            "zero_slope": alpha is None}
        fitted = ResponseShrinkage.from_components(x, y, donors, rna_model=final,
            oof_residuals=y-oof, shrinkage=shrinkage, provenance=provenance)
        self.__dict__.update(fitted.__dict__)
        self.alpha = alpha
        self.provenance.update(provenance)
        self.provenance["response_source_hash"] = response_hash
        self.provenance["source_hash"] = source_hash
        return self

    def predict_rna(self, x):
        # Unlike a response-mean model, alpha=None abundance predicts a zero
        # *difference*. Retain the explicit zero-slope affine component.
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in or not np.isfinite(x).all():
            raise ValueError("Query RNA must be finite and match training feature count")
        return self.response_rna.predict(x)

    def predict(self, x, anchor_values, anchor_indices):
        result = super().predict(x, anchor_values, anchor_indices)
        # Exact and interpretable decomposition for a slope-only RNA response:
        # constant residual mean + B delta_X + individual anchor innovations.
        origin = self.response_rna.predict(np.zeros((1, self.n_features_in)))[0]
        result["base_response"] = np.broadcast_to(origin + self.residual_mean,
                                                    result["prediction"].shape).copy()
        result["rna_contribution"] = result["rna_prediction"] - origin
        result["prediction"] = (result["base_response"] + result["rna_contribution"]
                                  + result["anchor_contributions"].sum(-1))
        return result

    @classmethod
    def load(cls, path):
        obj = super().load(path)
        if "formulation" not in obj.provenance:
            raise ValueError("Not an abundance-response model artifact")
        # The shared format historically used alpha=None to omit its affine
        # model. Here a zero-slope mapping is explicit and always persisted.
        if obj.response_rna is None:
            with np.load(path, allow_pickle=False) as data:
                model = RNARegressor(1., obj.n_hvg)
                model.n_features_in = obj.n_features_in
                for key in ("features", "x_mean", "x_scale", "y_mean", "y_scale", "coef", "intercept"):
                    setattr(model, key, data["rna_" + key].copy())
                obj.response_rna = model
        return obj


def select_abundance_shrinkage(x, y, donors, anchor_indices, targets, *,
        abundance_x, abundance_y, abundance_donors, abundance_weights,
        alphas=(.1, 1., 10., 100., None), shrinkages=(.25, .5, .75, 1.),
        n_hvg=2000, guard=None):
    """Fully donor-nested selection; fixed8 is supplied by the protocol caller.

    No residual cache crosses training partitions. Numerical ties (absolute
    1e-12, relative 1e-10) prefer greatest shrinkage, then greatest penalty,
    with None representing an infinite RNA penalty.
    """
    x, y, donors = _arrays(x, y, donors)
    ax, ay, ad, aw, _ = _abundance_arrays(x, y, donors, abundance_x, abundance_y,
                                        abundance_donors, abundance_weights)
    unique = np.unique(donors)
    if len(unique) < 3:
        raise ValueError("Nested abundance selection requires at least three donors")
    anchors = _indices(anchor_indices, y.shape[1], "anchor_indices")
    targets = _indices(targets, y.shape[1], "targets", nonempty=True)
    if np.intersect1d(anchors, targets).size:
        raise ValueError("Selection targets must be hidden, outside the anchor panel")
    alphas, shrinkages = [_alpha(a) for a in alphas], [_shrinkage(s) for s in shrinkages]
    if not alphas or not shrinkages:
        raise ValueError("Selection grids must not be empty")
    records, inner_fits = [], []
    for alpha in alphas:
        losses = [[] for _ in shrinkages]
        for donor in unique:
            _check(guard)
            train, val, atrain = donors != donor, donors == donor, ad != donor
            model = AbundanceResponseShrinkage().fit(x[train], y[train], donors[train],
                abundance_x=ax[atrain], abundance_y=ay[atrain], abundance_donors=ad[atrain],
                abundance_weights=aw[atrain], alpha=alpha, shrinkage=shrinkages[0],
                n_hvg=n_hvg, guard=guard)
            predictions = []
            for k, shrinkage in enumerate(shrinkages):
                _check(guard)
                result = model._set_shrinkage(shrinkage).predict(x[val], y[val][:, anchors], anchors)
                loss = float(np.mean(np.abs((result["prediction"][:, targets]-y[val][:, targets])
                                            / model.train_scales[targets])))
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite nested validation loss")
                losses[k].append(loss)
                predictions.append({"shrinkage": shrinkage,
                    "target_predictions": result["prediction"][:, targets].tolist()})
            inner_fits.append({"alpha": alpha, "validation_donor": str(donor),
                "validation_row_ids": _row_ids(x[val], y[val], donors[val]).tolist(),
                "training_donors": model.provenance["training_donors"],
                "training_source_hash": model.provenance["source_hash"],
                "abundance_training_donors": model.provenance["abundance_training_donors"],
                "abundance_source_hash": model.provenance["abundance_source_hash"],
                "parameter_hash_before_shrinkage": model._parameter_hash(),
                "residual_folds": deepcopy(model.provenance["residual_folds"]),
                "target_scales": model.train_scales[targets].tolist(),
                "predictions": predictions})
        for shrinkage, errors in zip(shrinkages, losses):
            records.append({"alpha": alpha, "shrinkage": shrinkage,
                "mean_donor_standardized_mae": float(np.mean(errors)),
                "donor_losses": dict(zip(unique.tolist(), errors))})
    minimum = min(v["mean_donor_standardized_mae"] for v in records)
    tied = [i for i, v in enumerate(records) if np.isclose(v["mean_donor_standardized_mae"],
                                                         minimum, rtol=1e-10, atol=1e-12)]
    best = max(tied, key=lambda i: (records[i]["shrinkage"],
                                   np.inf if records[i]["alpha"] is None else records[i]["alpha"]))
    chosen = records[best]
    model = AbundanceResponseShrinkage().fit(x, y, donors, abundance_x=ax, abundance_y=ay,
        abundance_donors=ad, abundance_weights=aw, alpha=chosen["alpha"],
        shrinkage=chosen["shrinkage"], n_hvg=n_hvg, guard=guard)
    return model, {"selected_alpha": chosen["alpha"], "selected_shrinkage": chosen["shrinkage"],
        "selected_index": best, "candidates": records, "inner_fits": inner_fits,
        "anchor_indices": anchors.tolist(), "target_indices": targets.tolist(),
        "training_source_hash": model.provenance["source_hash"], "training_donors": unique.tolist(),
        "selection_loss": "Equal-donor target mean absolute response error standardized by inner-training response SD",
        "tie_rule": "Within atol=1e-12/rtol=1e-10 of minimum: greatest shrinkage then greatest alpha; None strongest",
        "panel_policy": "Supplied fixed8 primary; same selected parameters for fixed4"}
