"""Preregistered evaluation statistics for v0.5. No model selection here."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .reliability import interval_score, paired_gain_bootstrap


def standardized_mae(y, pred, scale):
    y, pred, scale = (np.asarray(v, float) for v in (y, pred, scale))
    err = np.abs(y - pred) / scale
    err = np.where(np.isfinite(err), err, np.nan)
    return {"per_row": err, "mean": float(np.nanmean(err)) if np.isfinite(err).any() else np.nan,
            "availability": float(np.isfinite(err).mean())}


def skill_vs_mean(y, pred, mean):
    y, pred, mean = (np.asarray(v, float) for v in (y, pred, mean))
    num = np.nansum((y - pred)**2)
    den = np.nansum((y - mean)**2)
    if den <= 0:
        return np.nan
    return float(1 - num / den)


def direction_error(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    usable = np.isfinite(y) & np.isfinite(pred)
    if not usable.any():
        return np.nan
    return float(np.mean(np.sign(y[usable]) != np.sign(pred[usable])))


def noise_ceiling(split_a, split_b, scale):
    """Split-cell reproducibility ceiling: MAE between disjoint cell-half estimates."""
    return standardized_mae(split_a, split_b, scale)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adj[idx] = running
    return adj


def random_effects_mean(effects, variances):
    """DerSimonian-Laird random-effects mean of paired differences."""
    effects, variances = np.asarray(effects, float), np.asarray(variances, float)
    if len(effects) != len(variances) or not len(effects):
        raise ValueError("effects and variances must be aligned and nonempty")
    w = 1 / variances
    fixed = np.sum(w * effects) / np.sum(w)
    q = np.sum(w * (effects - fixed)**2)
    df = len(effects) - 1
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau = max(0.0, (q - df) / c) if df > 0 and c > 0 else 0.0
    w_star = 1 / (variances + tau)
    mean = np.sum(w_star * effects) / np.sum(w_star)
    se = np.sqrt(1 / np.sum(w_star))
    return {"mean": float(mean), "se": float(se), "tau2": float(tau),
            "q": float(q), "df": int(df),
            "p_heterogeneity": float(1 - chi2.cdf(q, df)) if df > 0 else None}


def coverage(y, lower, upper):
    y, lower, upper = (np.asarray(v, float) for v in (y, lower, upper))
    usable = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if not usable.any():
        return {"coverage": None, "n": 0, "mean_width": None}
    covered = (y[usable] >= lower[usable]) & (y[usable] <= upper[usable])
    return {"coverage": float(covered.mean()), "n": int(usable.sum()),
            "mean_width": float(np.mean(upper[usable] - lower[usable]))}


def evaluate_task(y, methods, scale, mean, donors, *, independent=False, baseline="two_penalty_ridge"):
    """methods maps name -> prediction array with y's shape."""
    rows = []
    for name, pred in methods.items():
        mae = standardized_mae(y, pred, scale)
        rows.append({"method": name, "standardized_mae": mae["mean"],
                     "availability": mae["availability"],
                     "skill_vs_mean": skill_vs_mean(y, pred, mean),
                     "direction_error": direction_error(y, pred)})
    inference = None
    if independent and baseline in methods:
        donor_base, donor_model = [], []
        for d in sorted(set(map(str, donors))):
            rows_d = np.asarray(donors, str) == d
            donor_base.append(float(np.mean(np.abs(y[rows_d] - methods[baseline][rows_d]) / scale)))
            donor_model.append(float(np.mean(np.abs(y[rows_d] - methods["gated_abundance_response"][rows_d]) / scale)))
        inference = paired_gain_bootstrap(np.asarray(donor_base), np.asarray(donor_model),
                                          sorted(set(map(str, donors))), independent_test=True,
                                          models_frozen=True, model_metadata={"estimator": "gated_abundance_response"})
    return {"summary": rows, "paired_inference": inference}


def interval_report(y, lower, upper, alpha=0.1):
    cov = coverage(y, lower, upper)
    score = interval_score(y, lower, upper, alpha=alpha)
    return {**cov, "interval_score_mean": score["mean"], "interval_status": score["status"]}
