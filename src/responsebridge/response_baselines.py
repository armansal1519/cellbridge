"""Small, equally tuned response baselines; all transformations are training-only."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge


def donor_weights(donors, n):
    if donors is None:
        return np.full(n, 1 / n)
    donors = np.asarray(donors, dtype=str)
    if donors.shape != (n,) or np.any(donors == ''):
        raise ValueError('Donor identities must align with training rows')
    _, inv, counts = np.unique(donors, return_inverse=True, return_counts=True)
    return 1 / (len(counts) * counts[inv])


def response_scale(y, donors=None):
    """Training donor-response SD, with a numerical floor, not an assay threshold."""
    y = np.asarray(y, dtype=float)
    w = donor_weights(donors, len(y))
    mean = np.average(y, axis=0, weights=w)
    return np.maximum(np.sqrt(np.average((y-mean)**2, axis=0, weights=w)), 1e-6)


@dataclass
class LinearResponse:
    kind: str = "joint"
    alpha: float | None = 1.0
    n_hvg: int = 2000

    def fit(self, x, y, anchors, *, donors=None):
        x, y, anchors = (np.asarray(v, dtype=float) for v in (x, y, anchors))
        if any(v.ndim != 2 or not np.isfinite(v).all() for v in (x, y, anchors)):
            raise ValueError("Finite training matrices required")
        if len(x) != len(y) or len(x) != len(anchors) or not len(x):
            raise ValueError("Training row mismatch")
        if self.kind not in {"mean", "rna", "anchor", "joint", "residual"}:
            raise ValueError("Unknown baseline")
        if self.alpha is not None and (not np.isfinite(self.alpha) or self.alpha < 0):
            raise ValueError('alpha must be None or nonnegative and finite')
        w = donor_weights(donors, len(y))
        variance = np.average((x-np.average(x, axis=0, weights=w))**2, axis=0, weights=w)
        eligible = np.flatnonzero(variance > 1e-12)
        self.features = eligible[np.argsort(-variance[eligible], kind="stable")[:self.n_hvg]]
        self.rna_width = x.shape[1]
        self.anchor_width = anchors.shape[1]
        xx = self._combine(x, anchors)
        self.x_mean = np.average(xx, axis=0, weights=w)
        self.x_scale = np.maximum(np.sqrt(np.average((xx-self.x_mean)**2, axis=0, weights=w)), 1e-6)
        self.y_mean = np.average(y, axis=0, weights=w)
        self.y_scale = response_scale(y, donors)
        self.coef = np.zeros((y.shape[1], xx.shape[1]))
        if self.kind != "mean" and self.alpha is not None and xx.shape[1]:
            reg = Ridge(alpha=float(self.alpha), fit_intercept=False, solver="cholesky")
            reg.fit((xx-self.x_mean)/self.x_scale, (y-self.y_mean)/self.y_scale,
                    sample_weight=w)
            self.coef = reg.coef_
        self.training_error_sd = np.maximum(np.sqrt(np.average((y-self.predict(x, anchors))**2, axis=0, weights=w)), 1e-6)
        return self

    def _combine(self, x, anchors):
        if x.shape[1] != self.rna_width or anchors.shape[1] != self.anchor_width:
            raise ValueError("Feature width mismatch")
        parts = []
        if self.kind in {"rna", "joint"}: parts.append(x[:, self.features])
        if self.kind in {"anchor", "joint", "residual"}: parts.append(anchors)
        return np.column_stack(parts) if parts else np.empty((len(x), 0))

    def predict(self, x, anchors):
        x, anchors = np.asarray(x, dtype=float), np.asarray(anchors, dtype=float)
        if x.ndim != 2 or anchors.ndim != 2 or len(x) != len(anchors):
            raise ValueError("Query shape mismatch")
        if not np.isfinite(x).all() or not np.isfinite(anchors).all():
            raise ValueError("Observed query inputs must be finite")
        xx = self._combine(x, anchors)
        return self.y_mean + (((xx-self.x_mean)/self.x_scale) @ self.coef.T)*self.y_scale

    def save(self, path):
        with open(path, "wb") as f:
            np.savez_compressed(f, kind=self.kind, alpha=np.nan if self.alpha is None else self.alpha,
                n_hvg=self.n_hvg, features=self.features, rna_width=self.rna_width,
                anchor_width=self.anchor_width, x_mean=self.x_mean, x_scale=self.x_scale,
                y_mean=self.y_mean, y_scale=self.y_scale, coef=self.coef,
                training_error_sd=self.training_error_sd)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as v:
            a=float(v["alpha"])
            obj=cls(str(v["kind"]), None if np.isnan(a) else a, int(v["n_hvg"]))
            for k in ["features", "x_mean", "x_scale", "y_mean", "y_scale", "coef", "training_error_sd"]:
                setattr(obj,k,v[k])
            obj.rna_width=int(v["rna_width"]); obj.anchor_width=int(v["anchor_width"])
        return obj


def tune_linear(x, y, donors, anchors, targets, kind, alphas, *, n_hvg=2000, guard=None):
    """Select one alpha by donor-heldout target MAE; fit all development donors."""
    donors = np.asarray(donors, dtype=str)
    x, y, anchors = (np.asarray(v, dtype=float) for v in (x, y, anchors))
    if donors.shape != (len(y),) or len(np.unique(donors)) < 2:
        raise ValueError('Baseline tuning requires at least two aligned donors')
    scores=[]; predictions=[]
    for alpha in ([None] if kind == "mean" else alphas):
        error=[]; oof=np.empty_like(y, dtype=float)
        for donor in sorted(set(donors)):
            if guard: guard()
            tr, va = donors != donor, donors == donor
            model=LinearResponse(kind,alpha,n_hvg).fit(x[tr], y[tr], anchors[tr], donors=donors[tr])
            pred=model.predict(x[va],anchors[va]); oof[va]=pred
            error.append(float(np.mean(np.abs((pred-y[va])/response_scale(y[tr],donors[tr]))[:,targets])))
        scores.append({"alpha":alpha,"mae":float(np.mean(error)),"donor_errors":error})
        predictions.append(oof)
    if not scores: raise ValueError('Baseline alpha grid must not be empty')
    best=min(range(len(scores)),key=lambda i:scores[i]["mae"])
    model=LinearResponse(kind,scores[best]["alpha"],n_hvg).fit(x,y,anchors,donors=donors)
    return model, {"selected_alpha":scores[best]["alpha"],"scores":scores,
        "selection":"LODO on development donors only; lower MAE, ties use first prespecified candidate"}, predictions[best]
