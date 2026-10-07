"""Fixed-target panel design; difficult targets cannot become panel anchors."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .model import ResponseBasis, _indices, anchor_geometry


@dataclass
class PanelSelection:
    indices: np.ndarray
    score: float
    target_indices: np.ndarray
    target_ambiguity: np.ndarray
    target_noise: np.ndarray
    observable: np.ndarray
    method: str = "fixed"


def evaluate_panel(basis: ResponseBasis, anchor_indices, target_indices,
                   anchor_noise=None, score_norm_bound=1.0, noise_weight=1.0,
                   rcond=1e-8) -> PanelSelection:
    """Worst standardized null bound plus propagated anchor-noise SD.

    anchor_noise contains per-protein measurement standard deviations in raw
    response units (independent-noise design approximation). Defaults to the
    fixed source scales. The target set must be specified before selection.
    """
    p = len(basis.scales)
    anchors = _indices(anchor_indices, p, "anchor_indices")
    targets = _indices(target_indices, p, "target_indices")
    if np.intersect1d(anchors, targets).size:
        raise ValueError("evaluation targets cannot be used as anchors")
    noise = basis.scales if anchor_noise is None else np.asarray(anchor_noise, dtype=float)
    if noise.shape != (p,) or not np.all(np.isfinite(noise)) or np.any(noise < 0):
        raise ValueError("anchor_noise must be nonnegative finite standard deviations, one per protein")
    if not np.isfinite(score_norm_bound) or score_norm_bound < 0 or not np.isfinite(noise_weight) or noise_weight < 0:
        raise ValueError("score_norm_bound and noise_weight must be nonnegative finite scalars")
    geometry = anchor_geometry(basis, anchors, rcond)
    ambiguity = geometry.null_ambiguity[targets] / basis.scales[targets]
    propagated = np.linalg.norm(geometry.operator[targets] * noise[anchors], axis=1) / basis.scales[targets]
    score = float(np.max(score_norm_bound * ambiguity + noise_weight * propagated))
    return PanelSelection(anchors, score, targets, ambiguity, propagated, geometry.observable[targets])


def _candidates(basis, targets, candidate_indices, excluded_indices):
    p = len(basis.scales)
    candidates = np.arange(p) if candidate_indices is None else _indices(candidate_indices, p, "candidate_indices")
    exclusions = set(int(x) for x in targets)
    if excluded_indices is not None and len(excluded_indices):
        exclusions.update(_indices(excluded_indices, p, "excluded_indices").tolist())
    # Primary activation outcomes are protected even if omitted from targets.
    exclusions.update(i for i, name in enumerate(basis.protein_names)
                      if re.search(r"(?<![A-Z0-9])CD(?:25|69)(?![0-9])", name.upper()))
    return np.asarray(sorted(set(candidates.tolist()) - exclusions), dtype=int)


def select_panel(basis: ResponseBasis, size: int, target_indices,
                 candidate_indices=None, anchor_noise=None, excluded_indices=None,
                 score_norm_bound=1.0, noise_weight=1.0, rcond=1e-8,
                 max_swap_passes=20) -> PanelSelection:
    """Deterministic greedy construction followed by single-swap improvement."""
    targets = _indices(target_indices, len(basis.scales), "target_indices")
    candidates = _candidates(basis, targets, candidate_indices, excluded_indices)
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or not 1 <= size <= len(candidates):
        raise ValueError("size must be an integer within the eligible candidate count")
    if not isinstance(max_swap_passes, (int, np.integer)) or max_swap_passes < 0:
        raise ValueError("max_swap_passes must be a nonnegative integer")

    def assess(indices):
        return evaluate_panel(basis, sorted(indices), targets, anchor_noise,
                              score_norm_bound, noise_weight, rcond)

    selected = []
    for _ in range(size):
        proposals = [assess(selected + [int(c)]) for c in candidates if c not in selected]
        best = min(proposals, key=lambda x: (x.score, tuple(x.indices)))
        selected = best.indices.tolist()
    for _ in range(max_swap_passes):
        current = assess(selected)
        best = current
        for old in selected:
            for new in candidates:
                if new not in selected:
                    proposal = assess([x for x in selected if x != old] + [int(new)])
                    if proposal.score < best.score - 1e-12:
                        best = proposal
        if best.score >= current.score - 1e-12:
            break
        selected = best.indices.tolist()
    result = assess(selected)
    result.method = "greedy_single_swap_minimax"
    return result


def random_panel(basis: ResponseBasis, size: int, target_indices,
                 candidate_indices=None, excluded_indices=None, seed=0, **kwargs):
    targets = _indices(target_indices, len(basis.scales), "target_indices")
    candidates = _candidates(basis, targets, candidate_indices, excluded_indices)
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or not 1 <= size <= len(candidates):
        raise ValueError("size must be an integer within the eligible candidate count")
    chosen = np.sort(np.random.default_rng(seed).choice(candidates, size, replace=False))
    result = evaluate_panel(basis, chosen, targets, **kwargs)
    result.method = "random"
    return result
