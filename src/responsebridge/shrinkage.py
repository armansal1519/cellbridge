"""Donor-response ridge and conditional prediction from residual covariance.

The covariance describes observed, cross-fitted response errors, including
sampling noise. It is not a noise-free biological covariance or a calibrated
prediction interval. All repeated rows of a donor share that donor's mass.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .rna import RNARegressor


SCALE_FLOOR = 1e-6
COVARIANCE_FLOOR = 1e-10


def _arrays(x, y, donors):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    raw = np.asarray(donors)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x)
            or not x.shape[1] or not y.shape[1]):
        raise ValueError("x and y must be aligned nonempty two-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training responses and RNA must be finite")
    if raw.ndim != 1 or len(raw) != len(x):
        raise ValueError("donors must identify every training row")
    if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)) for v in raw):
        raise ValueError("donor identities must not be missing")
    donors = raw.astype(str)
    if np.any(donors == ""):
        raise ValueError("donor identities must not be empty")
    return x, y, donors


def _weights(donors):
    unique, inverse, counts = np.unique(donors, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])


def _row_ids(x, y, donors):
    result = []
    for xi, yi, donor in zip(x, y, donors):
        digest = hashlib.sha256()
        digest.update(json.dumps([str(donor), len(xi), len(yi)]).encode())
        digest.update(np.asarray(xi, dtype="<f8").tobytes())
        digest.update(np.asarray(yi, dtype="<f8").tobytes())
        result.append(digest.hexdigest())
    return np.asarray(result, dtype=str)


def _source_hash(row_ids):
    return hashlib.sha256("\n".join(sorted(row_ids)).encode()).hexdigest()


def _indices(values, p, name, *, nonempty=False):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    values = values.astype(int)
    if ((nonempty and not len(values)) or len(np.unique(values)) != len(values)
            or np.any(values < 0) or np.any(values >= p)):
        raise ValueError(f"{name} must contain distinct valid protein indices")
    return values


def _alpha(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError("alpha must be None or a finite nonnegative number")
    return float(value)


def _shrinkage(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("shrinkage must be a finite number in [0, 1]")
    return float(value)


def _check(guard):
    if guard is not None:
        guard()


class ResponseShrinkage:
    """A fixed-alpha RNA model plus LODO residual covariance conditioning.

    ``fit`` requires at least two donors; it does not select hyperparameters.
    ``select_shrinkage`` performs the additional donor-nested selection layer.
    """

    def fit(self, x, y, donors, *, alpha=1.0, shrinkage=.5, n_hvg=2000, guard=None):
        x, y, donors = _arrays(x, y, donors)
        alpha, shrinkage = _alpha(alpha), _shrinkage(shrinkage)
        if isinstance(n_hvg, (bool, np.bool_)) or not isinstance(n_hvg, (int, np.integer)) or n_hvg < 1:
            raise ValueError("n_hvg must be a positive integer")
        unique = np.unique(donors)
        if len(unique) < 2:
            raise ValueError("Residual LODO fitting requires at least two donors")
        # Canonical ordering makes numerical calculations independent of input
        # row order, while row digests retain duplicates for an auditable source.
        row_ids = _row_ids(x, y, donors)
        order = np.lexsort((row_ids, donors))
        x, y, donors, row_ids = x[order], y[order], donors[order], row_ids[order]
        weights = _weights(donors)
        self.alpha, self.n_hvg = alpha, int(n_hvg)
        self.n_features_in, self.n_proteins = x.shape[1], y.shape[1]
        self.train_donors, self.train_row_ids = donors.copy(), row_ids.copy()
        self.weights = weights
        self.cohort_mean = np.average(y, axis=0, weights=weights)
        self.train_scales = np.maximum(np.sqrt(np.average(
            (y - self.cohort_mean)**2, axis=0, weights=weights)), SCALE_FLOOR)
        self.oof_prediction = np.empty_like(y)
        residual_folds = []
        for donor in unique:
            _check(guard)
            val = donors == donor
            train = ~val
            tw = _weights(donors[train])
            if alpha is None:
                prediction = np.broadcast_to(np.average(y[train], axis=0, weights=tw), y[val].shape)
            else:
                prediction = RNARegressor(alpha, n_hvg).fit(x[train], y[train], tw).predict(x[val])
            self.oof_prediction[val] = prediction
            residual_folds.append({
                "validation_donor": str(donor),
                "training_donors": np.unique(donors[train]).tolist(),
                "training_source_hash": _source_hash(row_ids[train]),
                "validation_row_ids": row_ids[val].tolist(),
            })
        self.oof_residuals = y - self.oof_prediction
        self.residual_mean = np.average(self.oof_residuals, axis=0, weights=weights)
        centered = self.oof_residuals - self.residual_mean
        # Population moment for the equal-donor/within-donor-row distribution.
        # No row-count Bessel correction: replicating a donor's complete row
        # set must not manufacture independent donor information.
        self.empirical_covariance = (centered * weights[:, None]).T @ centered
        self.empirical_covariance = (self.empirical_covariance + self.empirical_covariance.T) / 2
        if not np.isfinite(self.empirical_covariance).all():
            raise ValueError("Residual covariance overflowed")
        _check(guard)
        self.response_rna = None if alpha is None else RNARegressor(alpha, n_hvg).fit(x, y, weights)
        self.provenance = {
            "source_hash": _source_hash(row_ids), "training_donors": unique.tolist(),
            "training_row_ids": row_ids.tolist(), "residual_folds": residual_folds,
            "weighting": "Equal donors; equal rows within each donor, recomputed in every partition",
            "covariance_estimand": "Donor-balanced population moment of observed OOF response residuals; includes sampling noise",
            "uncertainty": "Model-conditional residual variance; excludes parameter uncertainty and is not a calibrated interval",
        }
        return self._set_shrinkage(shrinkage)

    @classmethod
    def from_components(cls, x, y, donors, *, rna_model, oof_residuals,
                        shrinkage=.5, provenance):
        """Build a matched ablation from an externally fitted affine RNA model.

        Residuals must align with the supplied input rows. The caller is
        responsible for donor-disjoint component fitting and must provide its
        provenance; this constructor does not certify those external fits.
        A supplied RNARegressor must already predict responses from response
        inputs (an abundance intercept must not survive differencing).
        """
        x, y, donors = _arrays(x, y, donors)
        residuals = np.asarray(oof_residuals, dtype=float)
        if residuals.shape != y.shape or not np.isfinite(residuals).all():
            raise ValueError("oof_residuals must be finite and align with training y")
        if len(np.unique(donors)) < 2:
            raise ValueError("Residual components require at least two donors")
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("External components require an explicit provenance dictionary")
        if not isinstance(rna_model, RNARegressor):
            raise ValueError("rna_model must be a fitted response-input RNARegressor")
        if rna_model.predict(x[:1]).shape != (1, y.shape[1]):
            raise ValueError("RNA component output dimension must match responses")
        obj = cls()
        row_ids = _row_ids(x, y, donors)
        order = np.lexsort((row_ids, donors))
        x, y, donors, row_ids, residuals = x[order], y[order], donors[order], row_ids[order], residuals[order]
        obj.alpha, obj.n_hvg = float(rna_model.alpha), int(rna_model.n_hvg)
        obj.n_features_in, obj.n_proteins = x.shape[1], y.shape[1]
        obj.train_donors, obj.train_row_ids = donors.copy(), row_ids.copy()
        obj.weights = _weights(donors)
        obj.cohort_mean = np.average(y, axis=0, weights=obj.weights)
        obj.train_scales = np.maximum(np.sqrt(np.average(
            (y - obj.cohort_mean)**2, axis=0, weights=obj.weights)), SCALE_FLOOR)
        obj.oof_residuals, obj.oof_prediction = residuals.copy(), y - residuals
        obj.residual_mean = np.average(residuals, axis=0, weights=obj.weights)
        centered = residuals - obj.residual_mean
        obj.empirical_covariance = (centered * obj.weights[:, None]).T @ centered
        obj.empirical_covariance = (obj.empirical_covariance + obj.empirical_covariance.T) / 2
        if not np.isfinite(obj.empirical_covariance).all():
            raise ValueError("Residual covariance overflowed")
        obj.response_rna = deepcopy(rna_model)
        obj.provenance = {
            "source_hash": _source_hash(row_ids), "training_donors": np.unique(donors).tolist(),
            "training_row_ids": row_ids.tolist(), "external_components": deepcopy(provenance),
            "residual_folds": deepcopy(provenance.get("residual_folds", [])),
            "weighting": "Equal donors; equal rows within each donor",
            "covariance_estimand": "Donor-balanced population moment of supplied OOF response residuals; includes sampling noise",
            "uncertainty": "Model-conditional residual variance; excludes parameter uncertainty and is not a calibrated interval",
        }
        return obj._set_shrinkage(shrinkage)

    def _set_shrinkage(self, shrinkage):
        self.shrinkage = _shrinkage(shrinkage)
        diagonal = np.diag(self.empirical_covariance)
        self.covariance_floor = COVARIANCE_FLOOR * max(1.0, float(np.mean(diagonal)))
        self.covariance = ((1 - self.shrinkage) * self.empirical_covariance
                           + self.shrinkage * np.diag(diagonal)
                           + self.covariance_floor * np.eye(self.n_proteins))
        return self

    def predict_rna(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in or not np.isfinite(x).all():
            raise ValueError("Query RNA must be finite and match training feature count")
        return (np.broadcast_to(self.cohort_mean, (len(x), self.n_proteins)).copy()
                if self.response_rna is None else self.response_rna.predict(x))

    def predict(self, x, anchor_values, anchor_indices):
        """Use only query RNA and measured anchors; never hidden query outcomes.

        prediction = base_response + rna_contribution + sum(anchor_contributions).
        Anchor contributions are exact additive model associations, not causal
        or uniquely allocated information when anchors are correlated.
        """
        rna = self.predict_rna(x)
        anchors = _indices(anchor_indices, self.n_proteins, "anchor_indices")
        observed = np.asarray(anchor_values, dtype=float)
        if observed.shape != (len(rna), len(anchors)) or not np.isfinite(observed).all():
            raise ValueError("anchor_values must be finite with shape (query rows, number of anchors)")
        if len(anchors):
            aa = self.covariance[np.ix_(anchors, anchors)]
            operator = np.linalg.solve(aa, self.covariance[anchors, :]).T
        else:
            operator = np.empty((self.n_proteins, 0))
        anchor_residuals = observed - rna[:, anchors] - self.residual_mean[anchors]
        contributions = anchor_residuals[:, None, :] * operator[None, :, :]
        base = np.broadcast_to(self.cohort_mean + self.residual_mean, rna.shape).copy()
        rna_contribution = rna - self.cohort_mean
        prediction = base + rna_contribution + contributions.sum(axis=-1)
        variance = np.diag(self.covariance) - np.sum(operator * self.covariance[:, anchors], axis=1)
        return {"prediction": prediction, "model_variance": np.maximum(variance, 0),
                "base_response": base, "rna_contribution": rna_contribution,
                "anchor_contributions": contributions, "anchor_operator": operator,
                "anchor_residuals": anchor_residuals, "rna_prediction": rna}

    def _parameter_hash(self):
        digest = hashlib.sha256()
        digest.update(json.dumps({"alpha": self.alpha, "n_hvg": self.n_hvg,
            "n_features_in": self.n_features_in, "n_proteins": self.n_proteins}, sort_keys=True).encode())
        arrays = [(key, getattr(self, key)) for key in
                  ("cohort_mean", "train_scales", "residual_mean", "empirical_covariance")]
        if self.response_rna is not None:
            arrays += [("rna_" + key, getattr(self.response_rna, key)) for key in
                       ("features", "x_mean", "x_scale", "y_mean", "y_scale", "coef", "intercept")]
        for key, array in arrays:
            digest.update(json.dumps([key, list(np.asarray(array).shape)]).encode())
            digest.update(np.asarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()

    def save(self, path):
        """Save numeric arrays and JSON metadata; no pickle/object arrays."""
        metadata = {"format_version": 1, "alpha": self.alpha, "shrinkage": self.shrinkage,
                    "n_hvg": self.n_hvg, "n_features_in": self.n_features_in,
                    "n_proteins": self.n_proteins, "provenance": self.provenance}
        arrays = {key: getattr(self, key) for key in (
            "cohort_mean", "train_scales", "residual_mean", "empirical_covariance",
            "train_donors", "train_row_ids", "weights", "oof_prediction", "oof_residuals")}
        if self.response_rna is not None:
            for key in ("features", "x_mean", "x_scale", "y_mean", "y_scale", "coef", "intercept"):
                arrays["rna_" + key] = getattr(self.response_rna, key)
        np.savez_compressed(Path(path), metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported ResponseShrinkage format version")
            obj = cls()
            for key in ("alpha", "n_hvg", "n_features_in", "n_proteins", "provenance"):
                setattr(obj, key, metadata[key])
            for key in ("cohort_mean", "train_scales", "residual_mean", "empirical_covariance",
                        "train_donors", "train_row_ids", "weights", "oof_prediction", "oof_residuals"):
                setattr(obj, key, data[key].copy())
            obj.response_rna = None
            if obj.alpha is not None:
                obj.response_rna = RNARegressor(obj.alpha, obj.n_hvg)
                obj.response_rna.n_features_in = obj.n_features_in
                for key in ("features", "x_mean", "x_scale", "y_mean", "y_scale", "coef", "intercept"):
                    setattr(obj.response_rna, key, data["rna_" + key].copy())
            return obj._set_shrinkage(metadata["shrinkage"])


def select_shrinkage(x, y, donors, anchor_indices, targets,
                     alphas=(.1, 1, 10, 100, None), shrinkages=(.25, .5, .75, 1),
                     n_hvg=2000, guard=None):
    """Select fixed-panel parameters by fully nested leave-one-donor-out MAE.

    Each validation donor is absent from the final inner RNA fit AND every
    residual cross-fit used for its covariance. Fits may be reused across
    shrinkages only within that same alpha/training-donor partition.
    """
    x, y, donors = _arrays(x, y, donors)
    unique = np.unique(donors)
    if len(unique) < 3:
        raise ValueError("Nested selection requires at least three donors")
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
            train, val = donors != donor, donors == donor
            model = ResponseShrinkage().fit(x[train], y[train], donors[train], alpha=alpha,
                                           shrinkage=shrinkages[0], n_hvg=n_hvg, guard=guard)
            predictions = []
            for k, shrinkage in enumerate(shrinkages):
                _check(guard)
                result = model._set_shrinkage(shrinkage).predict(x[val], y[val][:, anchors], anchors)
                error = (result["prediction"][:, targets] - y[val][:, targets]) / model.train_scales[targets]
                loss = float(np.mean(np.abs(error)))
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite nested validation loss")
                losses[k].append(loss)
                predictions.append({"shrinkage": shrinkage,
                                    "target_predictions": result["prediction"][:, targets].tolist()})
            inner_fits.append({"alpha": alpha, "validation_donor": str(donor),
                               "validation_row_ids": _row_ids(x[val], y[val], donors[val]).tolist(),
                               "training_source_hash": model.provenance["source_hash"],
                               "training_donors": model.provenance["training_donors"],
                               "training_row_ids": model.provenance["training_row_ids"],
                               "parameter_hash_before_shrinkage": model._parameter_hash(),
                               "residual_folds": model.provenance["residual_folds"],
                               "target_scales": model.train_scales[targets].tolist(),
                               "predictions": predictions})
        for shrinkage, loss in zip(shrinkages, losses):
            records.append({"alpha": alpha, "shrinkage": shrinkage,
                            "mean_donor_standardized_mae": float(np.mean(loss)),
                            "donor_losses": dict(zip(unique.tolist(), loss))})
    best = min(range(len(records)), key=lambda i: records[i]["mean_donor_standardized_mae"])
    chosen = records[best]
    _check(guard)
    fitted = ResponseShrinkage().fit(x, y, donors, alpha=chosen["alpha"],
                                    shrinkage=chosen["shrinkage"], n_hvg=n_hvg, guard=guard)
    selection = {"selected_alpha": chosen["alpha"], "selected_shrinkage": chosen["shrinkage"],
                 "selected_index": best, "candidates": records, "inner_fits": inner_fits,
                 "anchor_indices": anchors.tolist(), "target_indices": targets.tolist(),
                 "training_source_hash": fitted.provenance["source_hash"],
                 "training_donors": unique.tolist(),
                 "selection_loss": "Equal-donor mean absolute error, standardized by inner-training response SD (floor 1e-6)",
                 "tie_rule": "First candidate in the prespecified alpha/shrinkage grid",
                 "panel_policy": "Supplied fixed panel; any learned panel requires its own nested selection"}
    return fitted, selection
