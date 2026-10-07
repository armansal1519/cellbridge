"""Small, executable identification and assay-confounding demonstrations.

These are controlled mathematical experiments, not evidence of superiority on
biological data. No file I/O, training cohorts, or hidden outcomes enter predict.
"""

from __future__ import annotations

import numpy as np

from .model import ObservableInput, ResponseBasis, ResponseBridge


def run_simulations(seed=0, n=4000) -> dict:
    if not isinstance(n, (int, np.integer)) or n < 100:
        raise ValueError("n must be an integer of at least 100 for the noise demonstration")
    rng = np.random.default_rng(seed)
    loadings = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
    basis = ResponseBasis(loadings, protein_names=("anchor_x", "anchor_y", "CD25", "CD69"))
    score = rng.normal(size=(n, 2))
    truth = score @ loadings.T
    bridge = ResponseBridge(basis)
    full = bridge.predict(ObservableInput(np.zeros_like(truth), truth[:, :2], np.array([0, 1])),
                          anchor_covariance=np.zeros((2, 2)))
    missing = bridge.predict(ObservableInput(np.zeros_like(truth), truth[:, :1], np.array([0])),
                             anchor_covariance=np.zeros((1, 1)))
    bounded = bridge.predict(ObservableInput(np.zeros_like(truth), truth[:, :1], np.array([0])),
                             anchor_covariance=np.zeros((1, 1)), score_norm_bound=np.linalg.norm(score, axis=1))

    ill_loadings = np.array([[1., 0.], [1., 1e-4], [0., 1.]])
    ill_basis = ResponseBasis(ill_loadings)
    ill_truth = score @ ill_loadings.T
    sigma = 0.01
    noisy_anchors = ill_truth[:, :2] + rng.normal(scale=sigma, size=(n, 2))
    ill = ResponseBridge(ill_basis).predict(
        ObservableInput(np.zeros_like(ill_truth), noisy_anchors, np.array([0, 1])),
        anchor_covariance=np.eye(2) * sigma**2)
    empirical_variance = float(np.var(ill.prediction[:, 2] - ill_truth[:, 2], ddof=1))
    expected_variance = float(ill.measurement_variance[0, 2])

    shifted_truth = truth.copy()
    shifted_truth[:, 2] += 3.0
    unchanged_input = ObservableInput(np.zeros_like(truth), shifted_truth[:, :2], np.array([0, 1]))
    shifted_prediction = bridge.predict(unchanged_input)

    counts = np.array([100., 100., 100.])
    hidden_shift = np.array([100., 100., 10000.])
    full_clr = lambda x: np.log1p(x) - np.mean(np.log1p(x))
    anchor_clr = lambda x: np.log1p(x[:2]) - np.mean(np.log1p(x[:2]))
    return {
        "full_rank": {"max_absolute_error": float(np.max(np.abs(full.prediction - truth))),
                      "all_targets_observable": bool(full.observable.all())},
        "missing_direction": {"observable": missing.observable.tolist(),
                              "unbounded_target_intervals_are_nan": bool(np.isnan(missing.lower[:, 2:]).all()),
                              "bounded_target_coverage": float(np.mean((truth[:, 2:] >= bounded.lower[:, 2:]) &
                                                                        (truth[:, 2:] <= bounded.upper[:, 2:]))),
                              "score_bound_is_oracle_for_demonstration": True},
        "nearly_singular": {"target_observable": bool(ill.observable[2]),
                            "condition_number": float(ill.geometry.condition_number),
                            "target_noise_amplification": float(ill.noise_amplification[2]),
                            "analytic_target_variance": expected_variance,
                            "empirical_target_variance": empirical_variance,
                            "empirical_to_analytic_variance": empirical_variance / expected_variance},
        "hidden_only_shift": {"predictions_identical": bool(np.allclose(full.prediction, shifted_prediction.prediction)),
                              "hidden_target_mean_error": float(np.mean(shifted_prediction.prediction[:, 2] - shifted_truth[:, 2])),
                              "interpretation": "Anchor observations cannot identify a target-only response outside the fitted span."},
        "adt_depth_and_composition": {
            "raw_log_anchor_shift_under_doubled_depth": float(np.log1p(2 * counts[0]) - np.log1p(counts[0])),
            "full_panel_clr_anchor_change_from_hidden_only_change": float(full_clr(hidden_shift)[0] - full_clr(counts)[0]),
            "accessible_anchor_clr_change_from_hidden_only_change": float(anchor_clr(hidden_shift)[0] - anchor_clr(counts)[0]),
            "accessible_per_protein_log_change_from_hidden_only_change": float(np.log1p(hidden_shift[0]) - np.log1p(counts[0])),
            "interpretation": "Full-panel denominators leak hidden proteins; accessible-only transforms still require technical-depth control."}}


run_simulation_suite = run_simulations


def make_group_fixture(seed=53, donors=2, contexts=2, perturbations=12,
                       control_blocks=8, genes=12):
    """Return small synthetic GroupData for end-to-end software verification.

    Every donor/context has independent control blocks and all interventions.
    Responses lie exactly in a known two-dimensional protein span. This is an
    intentionally favorable identification fixture, not a realistic benchmark
    or evidence that the method outperforms a biological comparator.
    """
    from .benchmark_data import GroupData
    import pandas as pd

    values = [donors, contexts, perturbations, control_blocks, genes]
    if any(not isinstance(x, (int, np.integer)) or x < 1 for x in values):
        raise ValueError("fixture dimensions must be positive integers")
    if genes < perturbations:
        raise ValueError("genes must cover the one-hot synthetic interventions")
    rng = np.random.default_rng(seed)
    names = np.array(["CD38", "ICOS", "PD-1", "HLA-DR", "CD127", "CD27",
                      "CD28", "CD45RO", "CD3", "CD4", "CD25", "CD69"])
    loadings = np.array([[1., 0.], [0., 1.], [1., .2], [.2, 1.], [.7, .3],
                         [.3, .7], [1., -.3], [-.3, 1.], [.2, .4], [.4, .2],
                         [1.2, .6], [-.5, 1.1]])
    intervention_scores = rng.normal(scale=.6, size=(perturbations, 2))
    rna_coefficients = rng.normal(scale=.05, size=(genes, 2))
    x, y, metadata = [], [], []
    for donor in range(donors):
        for context in range(contexts):
            base_x = 2 + rng.normal(scale=.1, size=genes)
            for control in range(control_blocks):
                row_x = base_x + rng.normal(scale=.01, size=genes)
                x.append(row_x)
                y.append(8 + (row_x @ rna_coefficients) @ loadings.T)
                metadata.append({"donor": f"d{donor}", "context": f"c{context}",
                                 "lineage": "CD4_T", "perturbation": "NTC", "is_control": True,
                                 "block": f"control_{control:02d}", "n_cells": 80,
                                 "group_id": f"d{donor}_c{context}_control_{control:02d}"})
            for perturbation in range(perturbations):
                row_x = base_x.copy()
                row_x[perturbation] += .5
                response = intervention_scores[perturbation] * (1 + .1 * donor + .15 * context)
                x.append(row_x)
                y.append(8 + (row_x @ rna_coefficients + response) @ loadings.T)
                metadata.append({"donor": f"d{donor}", "context": f"c{context}",
                                 "lineage": "CD4_T", "perturbation": f"GENE{perturbation:02d}",
                                 "is_control": False, "block": f"GENE{perturbation:02d}", "n_cells": 80,
                                 "group_id": f"d{donor}_c{context}_GENE{perturbation:02d}"})
    x, y = np.asarray(x), np.asarray(y)
    return GroupData(x, y, np.full_like(y, 1e-5), pd.DataFrame(metadata),
                     np.array([f"RNA{i:02d}" for i in range(genes)]), names)
