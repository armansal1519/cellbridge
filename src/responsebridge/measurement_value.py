"""Training-only measurement-value criterion and greedy panel design.

For disjoint anchors A and targets T and lambda in [0, 1), the v0.4 residual
ridge operator is the algebraic identity

    K = S_TA [ S_AA + lambda/(1-lambda) D_A + eps/(1-lambda) I ]^{-1}

where D_A = diag(S_AA). This module does not introduce a new function class.
It estimates, from training residuals only, whether measuring A is expected to
reduce target residual error after accounting for estimation variance that
grows with panel size over donor count.
"""
from __future__ import annotations

import numpy as np

from .shrinkage import COVARIANCE_FLOOR


def regularized_anchor_precision(sigma, anchors, shrinkage, floor=None):
    """Return the regularized precision of the anchor block for lambda<1."""
    anchors = np.asarray(anchors, dtype=int)
    if shrinkage >= 1:
        n = len(anchors)
        return np.zeros((n, n)), np.zeros((sigma.shape[0], n))
    aa = np.asarray(sigma, dtype=float)[np.ix_(anchors, anchors)]
    diagonal = np.diag(aa)
    floor = COVARIANCE_FLOOR * max(1.0, float(np.mean(np.diag(sigma)))) if floor is None else float(floor)
    scale = shrinkage / (1 - shrinkage)
    regularized = aa + scale * np.diag(diagonal) + (floor / (1 - shrinkage)) * np.eye(len(anchors))
    precision = np.linalg.solve(regularized, np.eye(len(anchors)))
    operator = sigma[:, anchors] @ precision
    return precision, operator


def explained_residual(sigma, anchors, shrinkage):
    """Per-protein explained residual variance from the shrunk conditional."""
    _, operator = regularized_anchor_precision(sigma, anchors, shrinkage)
    if operator.size == 0:
        return np.zeros(sigma.shape[0])
    explained = np.sum(operator * sigma[:, np.asarray(anchors, dtype=int)], axis=1)
    return np.maximum(explained, 0.0)


def estimation_penalty(sigma, anchors, n_donors, shrinkage):
    """Finite-sample cost: residual variance times p_anchors / n_donors.

    The leading term is the usual covariance estimation rate. It is a training
    diagnostic, not a proof of out-of-sample optimality.
    """
    n_donors = int(n_donors)
    if n_donors < 2:
        raise ValueError("Measurement-value penalty requires at least two training donors")
    p = len(np.asarray(anchors, dtype=int))
    residual = np.maximum(np.diag(sigma), 0.0)
    return residual * (p / n_donors) * max(1.0 - float(shrinkage), 1e-6)


def measurement_benefit(sigma, anchors, targets, n_donors, shrinkage):
    """Expected benefit for each target: explained residual minus estimation cost."""
    anchors = np.asarray(anchors, dtype=int)
    targets = np.asarray(targets, dtype=int)
    explained = explained_residual(sigma, anchors, shrinkage)[targets]
    penalty = estimation_penalty(sigma, anchors, n_donors, shrinkage)[targets]
    benefit = explained - penalty
    return {"explained": explained, "penalty": penalty, "benefit": benefit,
            "helpful": benefit > 0, "n_anchors": int(len(anchors)),
            "n_donors": int(n_donors), "shrinkage": float(shrinkage)}


def greedy_panel(sigma, candidates, targets, n_donors, shrinkage, budget, forbidden=()):
    """Greedy submodular-style addition of anchors maximizing summed positive benefit.

    Selection uses training covariance only. Ties take the smaller candidate index.
    """
    remaining = [int(i) for i in candidates if int(i) not in set(map(int, forbidden)) | set(map(int, targets))]
    chosen = []
    budget = int(budget)
    if budget < 0:
        raise ValueError("Panel budget must be nonnegative")
    history = []
    for _ in range(min(budget, len(remaining))):
        best, best_score = None, -np.inf
        for idx in remaining:
            trial = chosen + [idx]
            score = float(np.sum(np.maximum(
                measurement_benefit(sigma, trial, targets, n_donors, shrinkage)["benefit"], 0)))
            if score > best_score + 1e-15 or (abs(score - best_score) <= 1e-15 and (best is None or idx < best)):
                best, best_score = idx, score
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
        history.append({"anchor": best, "sum_positive_benefit": best_score, "panel": list(chosen)})
    return {"panel": chosen, "history": history, "budget": budget}
