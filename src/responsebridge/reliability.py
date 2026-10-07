"""Donor-level reliability safeguards; no automatic inference from CV errors.

Split-conformal coverage is marginal over exchangeable calibration/test donors,
conditional on a model and score rule fixed before calibration. These helpers do
not establish exchangeability or transform inspected outcomes into fresh data.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from scipy.stats import norm


def _alpha(alpha):
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one")
    return alpha


def _donors(values):
    values = list(values)
    if any(v is None or not str(v).strip() or str(v).lower() == "nan" for v in values):
        raise ValueError("donor IDs must be nonempty and nonmissing")
    result = np.asarray([str(v) for v in values], dtype=str)
    # Numeric and string identities must not be silently merged.
    originals = {}
    for old, new in zip(values, result):
        key = (type(old).__name__, repr(old))
        if new in originals and originals[new] != key:
            raise ValueError("distinct donor IDs collide after string conversion")
        originals[new] = key
    return result


def validate_donor_splits(fit_donors, calibration_donors, test_donors):
    """Validate complete donor grouping, allowing repeated samples within a split."""
    groups = {name: set(_donors(ids)) for name, ids in (
        ("fit", fit_donors), ("calibration", calibration_donors), ("test", test_donors))}
    for left, right in (("fit", "calibration"), ("fit", "test"), ("calibration", "test")):
        overlap = sorted(groups[left] & groups[right])
        if overlap:
            raise ValueError(f"donor overlap between {left} and {right}: {overlap}")
    return {"status": "disjoint", "n_donors": {k: len(v) for k, v in groups.items()},
            "donors": {k: sorted(v) for k, v in groups.items()}}


def split_conformal(scores_by_donor, alpha=.1):
    """Return the finite-sample order statistic, retaining unavailable scores.

One scalar per independent donor is required. NaN/+inf scores become +inf;
they are never dropped. Mapping keys identify donors; an array asserts that
its entries already represent distinct donors. No interpolation is used.
"""
    alpha = _alpha(alpha)
    if isinstance(scores_by_donor, Mapping):
        _donors(scores_by_donor.keys())
        scores = np.asarray(list(scores_by_donor.values()), dtype=float)
    else:
        scores = np.asarray(scores_by_donor, dtype=float)
    if scores.ndim != 1 or np.any(scores < 0):
        raise ValueError("scores must be a one-dimensional nonnegative donor vector")
    unavailable = ~np.isfinite(scores)
    scores = np.where(unavailable, np.inf, scores)
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if not n or k > n:
        quantile, status = np.inf, "insufficient_calibration_donors"
    else:
        quantile = float(np.partition(scores, k - 1)[k - 1])
        status = "calibrated" if np.isfinite(quantile) else "unavailable_calibration_quantile"
    return {"quantile": float(quantile), "status": status, "alpha": alpha,
            "n_donors": n, "order_statistic": k,
            "unavailable_donor_scores": int(unavailable.sum()),
            "guarantee": "marginal_joint_donor_coverage_under_exchangeability",
            "requires": "fixed_model_and_score_rule_before_independent_calibration"}


def _matrix(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape (rows, targets)")
    return result


def _scale(scale, shape):
    scale = np.asarray(scale, dtype=float)
    try:
        scale = np.broadcast_to(scale, shape)
    except ValueError as exc:
        raise ValueError("scale must broadcast to (rows, targets)") from exc
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scale must be positive, finite and fixed without calibration outcomes")
    return scale


def fit_joint_calibration(y, pred, scale, donors, targets, alpha=.1):
    """Calibrate the maximum normalized error across each donor's rows/targets.

All required outcomes for a donor must be available. A missing outcome or
prediction gives that donor an infinite score rather than removing the donor.
"""
    y, pred = _matrix(y, "y"), _matrix(pred, "pred")
    ids = _donors(donors)
    targets = [str(t) for t in targets]
    if y.shape != pred.shape or len(ids) != len(y):
        raise ValueError("y, pred and donors must have matching rows/shapes")
    if len(targets) != y.shape[1] or len(set(targets)) != len(targets) or not targets:
        raise ValueError("targets must uniquely label every target column")
    scores = np.abs(y - pred) / _scale(scale, y.shape)
    scores = np.where(np.isfinite(scores), scores, np.inf)
    by_donor = {d: float(np.max(scores[ids == d])) for d in sorted(set(ids))}
    result = split_conformal(by_donor, alpha)
    result.update({"targets": targets, "donor_scores": by_donor,
                   "score": "maximum_absolute_normalized_error_across_donor_rows_and_targets",
                   "n_rows": len(y)})
    return result


def apply_calibration(pred, scale, calibration, targets=None):
    """Apply a frozen joint calibrator; uninformative intervals stay infinite."""
    pred = _matrix(pred, "pred")
    expected = calibration.get("targets")
    if expected is not None and pred.shape[1] != len(expected):
        raise ValueError("prediction columns do not match calibration targets")
    if targets is not None and expected is not None and list(targets) != expected:
        raise ValueError("target ordering differs from the frozen calibrator")
    q = float(calibration["quantile"])
    if np.isnan(q) or q < 0:
        raise ValueError("invalid calibration quantile")
    width = q * _scale(scale, pred.shape)
    lower, upper = pred - width, pred + width
    valid_pred = np.isfinite(pred)
    lower = np.where(valid_pred, lower, np.nan)
    upper = np.where(valid_pred, upper, np.nan)
    available = valid_pred & np.isfinite(lower) & np.isfinite(upper)
    status = np.full(pred.shape, "calibrated", dtype=object)
    status[~available] = "uninformative_interval"
    status[~valid_pred] = "prediction_unavailable"
    return {"lower": lower, "upper": upper, "available": available,
            "status": status, "calibration_status": calibration["status"],
            "alpha": calibration["alpha"], "guarantee": calibration.get("guarantee")}


def conditional_gaussian_intervals(pred, sd, alpha=.1):
    """Model-conditional Gaussian intervals, explicitly without coverage guarantee."""
    alpha = _alpha(alpha)
    pred = np.asarray(pred, dtype=float)
    sd = np.broadcast_to(np.asarray(sd, dtype=float), pred.shape)
    if np.any(sd < 0):
        raise ValueError("Gaussian standard deviations cannot be negative")
    available = np.isfinite(pred) & np.isfinite(sd)
    width = norm.ppf(1 - alpha / 2) * sd
    return {"lower": np.where(available, pred - width, np.nan),
            "upper": np.where(available, pred + width, np.nan),
            "available": available, "alpha": alpha,
            "status": "conditional_gaussian_not_coverage_guaranteed",
            "guarantee": None}


def interval_score(y, lower, upper, alpha=.1):
    """Proper interval score with refused finite-truth cases penalized by infinity.

Missing truths cannot be evaluated and are counted separately. The finite-only
mean is descriptive and must not replace the full evaluated-cohort mean.
"""
    alpha = _alpha(alpha)
    y, lower, upper = np.broadcast_arrays(np.asarray(y, float), np.asarray(lower, float),
                                        np.asarray(upper, float))
    if np.any(lower > upper):
        raise ValueError("lower interval bound exceeds upper bound")
    truth_available = np.isfinite(y)
    available = np.isfinite(lower) & np.isfinite(upper)
    usable = truth_available & available
    scores = np.full(y.shape, np.nan)
    scores[truth_available] = np.inf
    scores[usable] = (upper[usable] - lower[usable]
                      + 2 / alpha * np.maximum(lower[usable] - y[usable], 0)
                      + 2 / alpha * np.maximum(y[usable] - upper[usable], 0))
    n = int(truth_available.sum())
    return {"per_row": scores, "mean": float(scores[truth_available].mean()) if n else None,
            "finite_subset_mean": float(scores[usable].mean()) if usable.any() else None,
            "n_evaluable": n, "n_missing_truth": int((~truth_available).sum()),
            "n_refused": int((truth_available & ~available).sum()),
            "availability": float(usable.sum() / n) if n else None,
            "status": "complete" if usable.sum() == n and n else "incomplete_or_refused"}


def paired_gain_bootstrap(loss_baseline, loss_model, donors, n_bootstrap=10000,
                          seed=20261004, independent_test=False, models_frozen=False,
                          model_metadata=None):
    """Paired donor bootstrap only for independent tests of explicitly frozen fits.

The approximate percentile interval conditions on the fitted models. It does
not estimate uncertainty from retraining, selection, or overlapping CV fits.
Within-donor rows are averaged equally; callers must first encode any required
context/target weighting. Missing losses are never silently dropped.
"""
    b, m = np.asarray(loss_baseline, float), np.asarray(loss_model, float)
    ids = _donors(donors)
    if b.ndim != 1 or b.shape != m.shape or len(ids) != len(b):
        raise ValueError("losses and donors must be equally sized one-dimensional vectors")
    if np.any(b < 0) or np.any(m < 0):
        raise ValueError("losses must be nonnegative")
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    unique = sorted(set(ids))
    db = np.asarray([b[ids == d].mean() for d in unique])
    dm = np.asarray([m[ids == d].mean() for d in unique])
    complete = len(unique) > 0 and np.all(np.isfinite(db)) and np.all(np.isfinite(dm))
    delta = db - dm
    result = {"status": "descriptive_only_dependent_or_unverified_test",
              "n_donors": len(unique), "donors": unique,
              "donor_differences": delta.tolist(),
              "mean_difference": float(delta.mean()) if complete else None,
              "relative_gain": float(1 - dm.mean() / db.mean()) if complete and db.mean() > 0 else None,
              "difference_ci95": None, "relative_gain_ci95": None,
              "independent_test": bool(independent_test), "models_frozen": bool(models_frozen),
              "model_metadata": dict(model_metadata or {}),
              "conditioning": "fixed_models; no_training_or_model_selection_uncertainty"}
    if not complete:
        result["status"] = "inference_refused_incomplete_losses"
        return result
    if not (independent_test and models_frozen and model_metadata):
        return result
    if len(unique) < 2:
        result["status"] = "inference_refused_insufficient_test_donors"
        return result
    rng = np.random.default_rng(seed)
    diff, gain = [], []
    for start in range(0, int(n_bootstrap), 1000):
        idx = rng.integers(len(unique), size=(min(1000, int(n_bootstrap) - start), len(unique)))
        bb, mm = db[idx].mean(axis=1), dm[idx].mean(axis=1)
        diff.extend(bb - mm)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain.extend(1 - mm / bb)
    result.update({"status": "approximate_conditional_independent_test_bootstrap",
                   "difference_ci95": np.quantile(diff, [.025, .975]).tolist(),
                   "relative_gain_ci95": (np.quantile(gain, [.025, .975]).tolist()
                                           if np.all(np.isfinite(gain)) else None),
                   "n_bootstrap": int(n_bootstrap), "seed": int(seed),
                   "small_sample_warning": "coverage_is_approximate_not_a_finite_sample_guarantee"})
    return result


def allocate_external_donor_splits(donors, seed=20261004):
    """Stable metadata-only allocation; this does not certify power or freshness."""
    ids = _donors(donors)
    unique = np.asarray(sorted(set(ids)), dtype=str)
    n = len(unique)
    n_cal, n_test = max(9, n // 4), max(10, n // 4)
    n_fit = n - n_cal - n_test
    if n_fit < 8:
        return {"status": "insufficient_donors", "n_donors": n,
                "required_fit": 8, "required_calibration": n_cal,
                "required_test": n_test, "seed": int(seed), "splits": None}
    shuffled = np.random.default_rng(seed).permutation(unique)
    groups = {"fit": sorted(shuffled[:n_fit].tolist()),
              "calibration": sorted(shuffled[n_fit:n_fit+n_cal].tolist()),
              "test": sorted(shuffled[n_fit+n_cal:].tolist())}
    validate_donor_splits(groups["fit"], groups["calibration"], groups["test"])
    assignment = {d: split for split, ds in groups.items() for d in ds}
    return {"status": "allocated_metadata_only", "n_donors": n,
            "n_samples": len(ids), "seed": int(seed), "splits": groups,
            "sample_splits": [assignment[d] for d in ids],
            "power_validated": False, "freshness_verified": False,
            "note": "test>=10 is an allocation minimum, not evidence of adequate power"}
