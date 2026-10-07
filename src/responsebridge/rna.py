"""Train-only preprocessing and nested group cross-fitted multioutput ridge."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


@dataclass
class RNARegressor:
    alpha: float = 10
    n_hvg: int = 2000

    def fit(self, x, y, weights=None):
        """Fit weighted-mean loss + alpha * squared standardized coefficients.

        Normalizing weights to sum one fixes the meaning of alpha across folds
        and makes the fit invariant to arbitrary sample-weight rescaling.
        All feature selection and scaling use this training partition alone.
        """
        x, y, weights = _training_arrays(x, y, weights)
        if not np.isscalar(self.alpha) or not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be a finite nonnegative scalar")
        if isinstance(self.n_hvg, (bool, np.bool_)) or not isinstance(self.n_hvg, (int, np.integer)) or self.n_hvg < 1:
            raise ValueError("n_hvg must be a positive integer")
        weights = weights / weights.sum()
        self.n_features_in = x.shape[1]
        mean = np.average(x, axis=0, weights=weights)
        variance = np.average((x - mean)**2, axis=0, weights=weights)
        eligible = np.flatnonzero(variance > 1e-12)
        self.features = eligible[np.argsort(-variance[eligible], kind="stable")[:self.n_hvg]]
        self.x_mean = mean[self.features]
        self.x_scale = np.sqrt(variance[self.features])
        self.y_mean = np.average(y, axis=0, weights=weights)
        self.y_scale = np.sqrt(np.average((y-self.y_mean)**2, axis=0, weights=weights))
        self.y_scale = np.maximum(self.y_scale, 1e-6)
        xs = (x[:, self.features] - self.x_mean) / self.x_scale
        ys = (y - self.y_mean) / self.y_scale
        if not len(self.features):
            # A constant-RNA fold still supports a training-mean baseline.
            self.coef = np.empty((y.shape[1], 0))
            self.intercept = np.zeros(y.shape[1])
            return self
        # n group means << n genes: the dual Cholesky ridge solve is inexpensive.
        model = Ridge(alpha=self.alpha, solver="cholesky").fit(xs, ys, sample_weight=weights)
        self.coef, self.intercept = model.coef_, model.intercept_
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not np.all(np.isfinite(x)):
            raise ValueError("prediction RNA must be a finite two-dimensional array")
        if getattr(self, "n_features_in", None) is not None and x.shape[1] != self.n_features_in:
            raise ValueError("prediction RNA feature count does not match training")
        xs = (x[:, self.features] - self.x_mean) / self.x_scale
        return (xs @ self.coef.T + self.intercept) * self.y_scale + self.y_mean

    def save(self, path):
        np.savez_compressed(path, alpha=self.alpha, n_hvg=self.n_hvg, features=self.features,
            x_mean=self.x_mean, x_scale=self.x_scale, y_mean=self.y_mean, y_scale=self.y_scale,
            coef=self.coef, intercept=self.intercept, n_features_in=self.n_features_in)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            obj = cls(float(d["alpha"]), int(d["n_hvg"]))
            for key in ["features", "x_mean", "x_scale", "y_mean", "y_scale", "coef", "intercept"]: setattr(obj, key, d[key])
            obj.n_features_in = int(d["n_features_in"]) if "n_features_in" in d else None
        return obj


def _training_arrays(x, y, weights=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x) or not x.shape[1] or not y.shape[1]:
        raise ValueError("x and y must be aligned nonempty two-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Nonfinite training values")
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(x),) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive total")
    return x, y, weights


def _cv_arrays(x, y, groups, weights, alphas):
    x, y, weights = _training_arrays(x, y, weights)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(x):
        raise ValueError("groups must identify every training row")
    if np.any(weights <= 0):
        raise ValueError("Cross-validation weights must be strictly positive; exclude zero-weight rows explicitly")
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or not len(alphas) or not np.all(np.isfinite(alphas)) or np.any(alphas < 0):
        raise ValueError("alphas must be a nonempty finite nonnegative vector")
    return x, y, groups, weights, alphas


def _splits(groups, folds=3):
    count = min(folds, len(np.unique(groups)))
    if count < 2: raise ValueError("Cross fitting needs at least two groups")
    return GroupKFold(count).split(np.zeros(len(groups)), groups=groups)


def _partition_weights(weights, indices, weight_fn):
    """Recompute design weights from this partition's metadata when requested."""
    result = weights[indices] if weight_fn is None else np.asarray(weight_fn(indices), dtype=float)
    if result.shape != (len(indices),) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Partition weights must be aligned, finite and strictly positive")
    return result


def choose_alpha(x, y, groups, weights, alphas=(.1, 1, 10, 100), n_hvg=2000, *, weight_fn=None):
    """Select alpha without query outcomes or query-derived transforms.

    With weight_fn, every training and validation partition gets freshly
    computed metadata-only design weights; normalized validation losses are
    averaged equally across folds. Omitting it preserves legacy fixed-source
    weights and the original pooled weighted score.
    """
    x, y, groups, weights, alphas = _cv_arrays(x, y, groups, weights, alphas)
    scores = []
    for alpha in alphas:
        weighted_error, fold_scores = 0.0, []
        for train, val in _splits(groups):
            tw = _partition_weights(weights, train, weight_fn)
            vw = _partition_weights(weights, val, weight_fn)
            model = RNARegressor(alpha, n_hvg).fit(x[train], y[train], tw)
            error = np.mean(np.abs((model.predict(x[val])-y[val])/model.y_scale), axis=1)
            weighted_error += float(np.sum(error * vw))
            fold_scores.append(float(np.average(error, weights=vw)))
        scores.append(float(np.mean(fold_scores)) if weight_fn is not None else weighted_error / float(weights.sum()))
    return float(alphas[int(np.argmin(scores))]), scores


def cross_fit(x, y, groups, weights, alphas=(.1, 1, 10, 100), n_hvg=2000, *, weight_fn=None):
    """OOF predictions include alpha/HVG/scaling fitting inside each fold."""
    x, y, groups, weights, alphas = _cv_arrays(x, y, groups, weights, alphas)
    predictions = np.full_like(y, np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    chosen = []
    for fold, (train, val) in enumerate(_splits(groups)):
        tw = _partition_weights(weights, train, weight_fn)
        if len(np.unique(groups[train])) >= 2:
            # Nested indices are relative to the outer training partition.
            nested_weights = None if weight_fn is None else lambda ids: weight_fn(train[ids])
            alpha, _ = choose_alpha(x[train], y[train], groups[train], tw, alphas, n_hvg,
                weight_fn=nested_weights)
        else:
            alpha = 10.0  # Prespecified fallback, never chosen using val outcomes.
        model = RNARegressor(alpha, n_hvg).fit(x[train], y[train], tw)
        predictions[val] = model.predict(x[val]); fold_ids[val] = fold; chosen.append(alpha)
    if not np.isfinite(predictions).all(): raise ValueError("Incomplete OOF coverage")
    return predictions, fold_ids, chosen
