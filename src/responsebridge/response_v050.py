"""Gated abundance-response estimator (ResponseBridge 1.0 / v0.5).

Builds on v0.4 abundance training and residual shrinkage. Adds training-only
do-no-harm stacking, optional empirical-Bayes slope shrinkage, composition
aggregation, Mondrian/jackknife-plus conformal scores, and PPI summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from .abundance_response import AbundanceResponseShrinkage
from .measurement_value import greedy_panel, measurement_benefit
from .reliability import apply_calibration, split_conformal
from .rna import RNARegressor
from .shrinkage import _arrays, _weights


SIMPLEX = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
           (0.5, 0.5, 0.0), (0.7, 0.3, 0.0), (0.3, 0.7, 0.0),
           (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (1 / 3, 1 / 3, 1 / 3))


def simplex_weights(oof, y, donors, benefit=None):
    """Per-target nonnegative weights on (mean, rna, anchor) from OOF MAE.

    Mixes with a positive anchor weight are eligible only when the training
    measurement-value benefit for that target is positive (or unevaluated).
    """
    oof = np.stack(oof, axis=-1)
    if oof.shape[:2] != y.shape:
        raise ValueError("OOF candidate predictions must align with y")
    w = _weights(donors)
    chosen = np.zeros((y.shape[1], 3))
    benefit = None if benefit is None else np.asarray(benefit, dtype=float)
    for t in range(y.shape[1]):
        allow_anchor = benefit is None or (t < len(benefit) and benefit[t] > 0)
        best, best_err = (1.0, 0.0, 0.0), np.inf
        for mix in SIMPLEX:
            if mix[2] > 0 and not allow_anchor:
                continue
            pred = oof[:, t] @ np.asarray(mix)
            err = float(np.average(np.abs(y[:, t] - pred), weights=w))
            if err < best_err - 1e-15:
                best, best_err = mix, err
        chosen[t] = best
    return chosen


def empirical_bayes_prior(coefs):
    """Pool standardized RNA slopes across development tasks."""
    stacked = np.concatenate([np.asarray(c, dtype=float).ravel() for c in coefs]) if coefs else np.array([0.])
    return {"mean": float(np.mean(stacked)), "var": float(max(np.var(stacked), 1e-8)), "n": int(len(stacked))}


def shrink_coef(coef, prior, local_var):
    mean, var = prior["mean"], prior["var"]
    local_var = max(float(local_var), 1e-8)
    weight = var / (var + local_var)
    return weight * np.asarray(coef, dtype=float) + (1 - weight) * mean


def mix_composition(subtype_pred, fractions):
    """Aggregate subtype-level responses with query composition fractions."""
    pred = np.asarray(subtype_pred, dtype=float)
    frac = np.asarray(fractions, dtype=float)
    if pred.ndim != 3 or frac.ndim != 2 or pred.shape[0] != frac.shape[0] or pred.shape[1] != frac.shape[1]:
        raise ValueError("subtype_pred (donors, subtypes, proteins) must match fractions")
    if np.any(frac < 0) or not np.allclose(frac.sum(axis=1), 1, atol=1e-6):
        raise ValueError("fractions must be nonnegative and sum to one per donor")
    return np.sum(pred * frac[:, :, None], axis=1)


def mondrian_calibrate(y, pred, scale, donors, alpha=0.1):
    """Per-target split conformal using the maximum score within each donor."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    scale = np.broadcast_to(np.asarray(scale, float), y.shape)
    donors = np.asarray(donors, str)
    result = []
    for t in range(y.shape[1]):
        scores = {}
        for d in sorted(set(donors)):
            rows = donors == d
            value = np.abs(y[rows, t] - pred[rows, t]) / scale[rows, t]
            scores[d] = float(np.max(np.where(np.isfinite(value), value, np.inf)))
        result.append(split_conformal(scores, alpha))
    return result


def jackknife_plus_radius(train_oof_scores, alpha=0.1):
    """Finite-sample rank quantile of training leave-one-donor scores."""
    return split_conformal(np.asarray(train_oof_scores, dtype=float), alpha)


def prediction_powered_mean(labeled_y, labeled_pred, unlabeled_pred):
    """PPI estimator for a mean outcome (Angelopoulos et al. 2023 form)."""
    labeled_y, labeled_pred, unlabeled_pred = (np.asarray(v, float) for v in (labeled_y, labeled_pred, unlabeled_pred))
    if labeled_y.shape != labeled_pred.shape:
        raise ValueError("Labeled outcomes and predictions must align")
    rectifier = unlabeled_pred.mean(axis=0)
    residual = (labeled_y - labeled_pred).mean(axis=0)
    labeled_only = labeled_y.mean(axis=0)
    return {"ppi": rectifier + residual, "labeled_only": labeled_only,
            "rectifier": rectifier, "residual_mean": residual,
            "n_labeled": int(len(labeled_y)), "n_unlabeled": int(len(unlabeled_pred))}


@dataclass
class GatedAbundanceResponse:
    alpha: float | None = 1.0
    shrinkage: float = 0.5
    n_hvg: int = 2000
    slope_prior: dict | None = None
    gate_weights: np.ndarray | None = None
    inner: AbundanceResponseShrinkage = field(default_factory=AbundanceResponseShrinkage)
    rna_only: AbundanceResponseShrinkage = field(default_factory=AbundanceResponseShrinkage)
    train_mean: np.ndarray | None = None
    train_scales: np.ndarray | None = None
    n_proteins: int = 0

    def fit(self, x, y, donors, *, abundance_x, abundance_y, abundance_donors,
            abundance_weights, anchors, gate_targets=None):
        x, y, donors = _arrays(x, y, donors)
        unique = np.unique(donors)
        if len(unique) < 3:
            raise ValueError("Gating requires at least three training donors")
        self.inner.fit(x, y, donors, abundance_x=abundance_x, abundance_y=abundance_y,
                       abundance_donors=abundance_donors, abundance_weights=abundance_weights,
                       alpha=self.alpha, shrinkage=self.shrinkage, n_hvg=self.n_hvg)
        self.rna_only.fit(x, y, donors, abundance_x=abundance_x, abundance_y=abundance_y,
                          abundance_donors=abundance_donors, abundance_weights=abundance_weights,
                          alpha=self.alpha, shrinkage=1.0, n_hvg=self.n_hvg)
        if self.slope_prior is not None and self.inner.response_rna is not None:
            local = np.var(self.inner.response_rna.coef) if self.inner.response_rna.coef.size else 1.0
            self.inner.response_rna.coef = shrink_coef(self.inner.response_rna.coef, self.slope_prior, local)
        self.train_mean = np.average(y, axis=0, weights=_weights(donors))
        self.train_scales = np.maximum(np.sqrt(np.average((y - self.train_mean)**2, axis=0, weights=_weights(donors))), 1e-6)
        self.n_proteins = y.shape[1]
        oof_mean = np.broadcast_to(self.train_mean, y.shape).copy()
        oof_rna = np.empty_like(y)
        oof_full = np.empty_like(y)
        empty = np.zeros((len(x), 0))
        for donor in unique:
            train = donors != donor
            fitted = AbundanceResponseShrinkage().fit(
                x[train], y[train], donors[train],
                abundance_x=abundance_x[np.isin(abundance_donors, np.unique(donors[train]))],
                abundance_y=abundance_y[np.isin(abundance_donors, np.unique(donors[train]))],
                abundance_donors=abundance_donors[np.isin(abundance_donors, np.unique(donors[train]))],
                abundance_weights=abundance_weights[np.isin(abundance_donors, np.unique(donors[train]))],
                alpha=self.alpha, shrinkage=self.shrinkage, n_hvg=self.n_hvg)
            rna = AbundanceResponseShrinkage().fit(
                x[train], y[train], donors[train],
                abundance_x=abundance_x[np.isin(abundance_donors, np.unique(donors[train]))],
                abundance_y=abundance_y[np.isin(abundance_donors, np.unique(donors[train]))],
                abundance_donors=abundance_donors[np.isin(abundance_donors, np.unique(donors[train]))],
                abundance_weights=abundance_weights[np.isin(abundance_donors, np.unique(donors[train]))],
                alpha=self.alpha, shrinkage=1.0, n_hvg=self.n_hvg)
            val = donors == donor
            observed = y[val][:, anchors] if len(anchors) else empty[:int(val.sum())]
            oof_full[val] = fitted.predict(x[val], observed, anchors)["prediction"]
            oof_rna[val] = rna.predict(x[val], empty[:int(val.sum())], [])["prediction"]
        benefit = None
        if getattr(self.inner, "covariance", None) is not None and len(anchors):
            benefit = measurement_benefit(self.inner.covariance, anchors, np.arange(y.shape[1]),
                                          len(unique), self.shrinkage)["benefit"]
        self.gate_weights = simplex_weights((oof_mean, oof_rna, oof_full), y, donors, benefit=benefit)
        if gate_targets is not None:
            mask = np.ones(y.shape[1], dtype=bool)
            mask[np.asarray(gate_targets, dtype=int)] = True
        return self

    def predict(self, x, anchor_values, anchor_indices):
        full = self.inner.predict(x, anchor_values, anchor_indices)
        rna = self.rna_only.predict(x, np.zeros((len(x), 0)), [])
        mean = np.broadcast_to(self.train_mean, rna["prediction"].shape).copy()
        stacked = np.stack([mean, rna["prediction"], full["prediction"]], axis=-1)
        prediction = np.einsum("dtk,tk->dt", stacked, self.gate_weights)
        result = dict(full)
        result["prediction"] = prediction
        result["ungated_prediction"] = full["prediction"]
        result["rna_only_prediction"] = rna["prediction"]
        result["training_mean_prediction"] = mean
        result["gate_weights"] = self.gate_weights
        return result


def hierarchical_residual_covariance(covariances, weights=None):
    """Average residual covariances across lineages or tasks."""
    mats = [np.asarray(c, dtype=float) for c in covariances]
    if not mats or any(m.shape != mats[0].shape for m in mats):
        raise ValueError("Covariances must share shape")
    w = np.ones(len(mats)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    pooled = sum(wi * m for wi, m in zip(w, mats))
    return (pooled + pooled.T) / 2


def interval_from_scores(pred, scale, calibration, targets=None):
    if isinstance(calibration, list):
        lowers, uppers, available = [], [], []
        for t, cal in enumerate(calibration):
            applied = apply_calibration(np.asarray(pred)[:, t:t + 1], np.broadcast_to(np.asarray(scale), np.asarray(pred).shape)[:, t:t + 1], cal)
            lowers.append(applied["lower"][:, 0])
            uppers.append(applied["upper"][:, 0])
            available.append(applied["available"][:, 0])
        return {"lower": np.stack(lowers, axis=1), "upper": np.stack(uppers, axis=1),
                "available": np.stack(available, axis=1), "calibration": calibration}
    return apply_calibration(pred, scale, calibration, targets=targets)


def design_panel(sigma, n_donors, shrinkage, candidates, targets, budget, forbidden=()):
    benefit = measurement_benefit(sigma, [], targets, n_donors, shrinkage)
    designed = greedy_panel(sigma, candidates, targets, n_donors, shrinkage, budget, forbidden)
    return {"empty_benefit": benefit, "designed": designed}
