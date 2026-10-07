"""Explicitly separated basis refitting and held-out-context error calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ResponseBasis, NumericalRankError, _matrix


@dataclass
class ContextCalibration:
    radius: np.ndarray
    alpha: float
    group_count: int
    quantile_level: float
    interpretation: str


def calibrate_context_error(errors, groups, alpha=0.05, simultaneous=True) -> ContextCalibration:
    """Finite-sample rank quantile of held-out group errors.

    Repeated rows within a donor/context group are reduced by the maximum.
    simultaneous=True also takes the maximum across proteins; callers should
    standardize errors first if equalized protein scales are desired. The
    exchangeable calibration-group requirement does not follow from masking.
    Too few groups for the requested rank returns infinity, not false precision.
    """
    errors = _matrix(errors, "errors")
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(errors) or not len(groups):
        raise ValueError("groups must identify each calibration row")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    unique = np.unique(groups)
    scores = np.stack([np.max(np.abs(errors[groups == g]), axis=0) for g in unique])
    if simultaneous:
        scores = np.max(scores, axis=1, keepdims=True)
    m = len(scores)
    order = int(np.ceil((m + 1) * (1 - alpha)))
    radius = (np.full(scores.shape[1], np.inf) if order > m
              else np.sort(scores, axis=0)[order - 1])
    if simultaneous:
        radius = np.repeat(radius, errors.shape[1])
    return ContextCalibration(radius, alpha, m, min(order / m, 1.0),
                              "Rank-calibrated for exchangeable independent groups only; no arbitrary-context-shift guarantee.")


@dataclass
class BootstrapBasisDraws:
    draws: list[ResponseBasis | None]
    valid: np.ndarray
    failures: list[dict]


def bootstrap_basis(residuals, groups, rank, n_bootstrap=100, sample_weight=None,
                    protein_names=None, scales=None, seed=0, failure_policy="raise") -> list[ResponseBasis] | BootstrapBasisDraws:
    """Resample entire supplied groups, keeping paired/context rows together.

    This refits the basis only. It does not refit RNA models or estimate all
    pipeline uncertainty; callers requiring that must bootstrap upstream too.
    Default behavior raises on numerical-rank failure. Explicit record mode
    retains every failed slot and reason; it never replaces or drops a draw.
    """
    residuals = _matrix(residuals, "residuals")
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(residuals):
        raise ValueError("groups must identify every residual row")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two independent groups are required")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    if failure_policy not in {"raise", "record"}:
        raise ValueError("failure_policy must be raise or record")
    weight = np.ones(len(residuals)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weight.shape != (len(residuals),):
        raise ValueError("sample_weight length mismatch")
    # Validate the requested source model and metadata before recording any
    # resampling failure. Malformed inputs must never become missing draws.
    ResponseBasis.fit(residuals, rank, weight, protein_names, scales)
    rng = np.random.default_rng(seed)
    draws, failures = [], []
    for draw_index in range(n_bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        try:
            draws.append(ResponseBasis.fit(residuals[index], rank, weight[index], protein_names, scales))
        except NumericalRankError as exc:
            if failure_policy == "raise":
                raise NumericalRankError(rank, exc.numerical_rank,
                    f"Bootstrap basis draw {draw_index} failed at fixed rank {rank}; "
                    "failed draws cannot be silently omitted or replaced by lower ranks. "
                    f"Original error: {exc}") from exc
            draws.append(None)
            failures.append({"draw_index": draw_index, "requested_rank": int(rank),
                "numerical_rank": exc.numerical_rank, "reason": str(exc),
                "sampled_groups": [str(group) for group in chosen]})
    if failure_policy == "record":
        return BootstrapBasisDraws(draws, np.array([draw is not None for draw in draws]), failures)
    return draws
