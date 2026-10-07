"""Low-rank response transfer using only RNA predictions and observed anchors.

Observability is conditional on the learned response span. A target-only change
outside this span cannot be diagnosed from anchors, regardless of sample size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import norm


class NumericalRankError(ValueError):
    """Valid response data cannot support the requested fixed basis rank."""

    def __init__(self, requested_rank, numerical_rank, message=None):
        self.requested_rank = int(requested_rank)
        self.numerical_rank = int(numerical_rank)
        super().__init__(message or
            f"requested rank {self.requested_rank} exceeds the numerical rank {self.numerical_rank} of the residual responses")


def _matrix(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return value


def _indices(value, p: int, name: str = "indices") -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be a nonempty integer vector")
    value = raw.astype(int, copy=True)
    if np.any(value < 0) or np.any(value >= p) or len(np.unique(value)) != len(value):
        raise ValueError(f"{name} must contain unique indices in [0, {p})")
    return value


@dataclass
class ResponseBasis:
    """Plain-array model state; loadings are in raw protein-response units.

    ``fit`` creates orthonormal latent directions in the training-standardized
    protein space. Explicitly constructed loadings retain the supplied latent
    coordinate system, which also defines any score-norm bound. No intercept
    is removed: zero response remains the origin.
    """

    loadings: np.ndarray
    scales: np.ndarray | None = None
    singular_values: np.ndarray | None = None
    protein_names: Sequence[str] | None = None

    def __post_init__(self):
        self.loadings = _matrix(self.loadings, "loadings").copy()
        p, k = self.loadings.shape
        if not p or not k or k > p:
            raise ValueError("loadings must have 1 <= rank <= number of proteins")
        self.scales = np.ones(p) if self.scales is None else np.asarray(self.scales, dtype=float).copy()
        if self.scales.shape != (p,) or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("scales must be a positive finite vector, one per protein")
        self.singular_values = (np.empty(0) if self.singular_values is None
                                else np.asarray(self.singular_values, dtype=float).copy())
        if self.singular_values.ndim != 1 or not np.all(np.isfinite(self.singular_values)) or np.any(self.singular_values < 0):
            raise ValueError("singular_values must be a finite nonnegative vector")
        self.protein_names = tuple(str(x) for x in (range(p) if self.protein_names is None else self.protein_names))
        if len(self.protein_names) != p or len(set(self.protein_names)) != p:
            raise ValueError("protein_names must be unique and match the protein dimension")

    @property
    def rank(self) -> int:
        return self.loadings.shape[1]

    @classmethod
    def fit(cls, residuals, rank: int, sample_weight=None, protein_names=None, scales=None):
        """Fit weighted *uncentered* SVD to OOF donor/condition residuals.

        The caller supplies leakage-free residuals, source-only scales, and any
        donor/context balancing weights. Replicate rows are not independent
        biological observations merely because they appear in this matrix.
        """
        residuals = _matrix(residuals, "residuals")
        n, p = residuals.shape
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
            raise ValueError("rank must be an integer")
        weight = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weight.shape != (n,) or not np.all(np.isfinite(weight)) or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and have positive sum")
        scale = np.ones(p) if scales is None else np.asarray(scales, dtype=float)
        if scale.shape != (p,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scales must be finite and positive with one entry per protein")
        if rank < 1 or rank > p:
            raise ValueError("rank must lie between one and the protein dimension")
        if rank > np.count_nonzero(weight):
            raise NumericalRankError(rank, min(np.count_nonzero(weight), p), "rank exceeds the positive-weight data dimensions")
        weighted = (residuals / scale) * np.sqrt(weight / weight.sum())[:, None]
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        numerical_rank = np.count_nonzero(singular > np.finfo(float).eps * max(weighted.shape) * singular[0])
        if rank > numerical_rank:
            raise NumericalRankError(rank, numerical_rank)
        return cls(vt[:rank].T * scale[:, None], scale, singular, protein_names)


@dataclass
class ObservableInput:
    """The complete inference interface: no hidden target outcomes are accepted."""

    baseline: np.ndarray
    anchor_residuals: np.ndarray
    anchor_indices: np.ndarray

    def __post_init__(self):
        self.baseline = _matrix(self.baseline, "baseline").copy()
        self.anchor_residuals = _matrix(self.anchor_residuals, "anchor_residuals").copy()
        self.anchor_indices = _indices(self.anchor_indices, self.baseline.shape[1], "anchor_indices")
        if self.anchor_residuals.shape != (self.baseline.shape[0], len(self.anchor_indices)):
            raise ValueError("anchor_residuals must have shape (n_responses, n_anchors)")


@dataclass
class AnchorGeometry:
    operator: np.ndarray
    pseudoinverse: np.ndarray
    observable: np.ndarray
    null_ambiguity: np.ndarray
    noise_amplification: np.ndarray
    singular_values: np.ndarray
    effective_rank: int
    condition_number: float


def anchor_geometry(basis: ResponseBasis, anchor_indices, rcond=1e-8, observability_tolerance=1e-8) -> AnchorGeometry:
    """Target-specific observability, retaining the SVD directions above rcond."""
    if not 0 < rcond < 1 or not 0 < observability_tolerance < 1:
        raise ValueError("rcond and observability_tolerance must lie strictly between zero and one")
    indices = _indices(anchor_indices, len(basis.scales), "anchor_indices")
    la = basis.loadings[indices]
    singular = np.linalg.svd(la, compute_uv=False)
    rank = int(np.count_nonzero(singular > (singular[0] * rcond)))
    pinv = np.linalg.pinv(la, rcond=rcond)
    operator = basis.loadings @ pinv
    null = basis.loadings - operator @ la
    ambiguity = np.linalg.norm(null, axis=1)
    row_norm = np.linalg.norm(basis.loadings, axis=1)
    observable = ambiguity <= observability_tolerance * np.maximum(row_norm, np.finfo(float).eps)
    # Tiny roundoff in an identified direction is not substantive ambiguity.
    ambiguity = np.where(observable, 0.0, ambiguity)
    condition = float(singular[0] / singular[-1]) if rank == basis.rank else float("inf")
    return AnchorGeometry(operator, pinv, observable, ambiguity,
                          np.linalg.norm(operator, axis=1), singular, rank, condition)


def _measurement_variance(covariance, operator, n):
    p, a = operator.shape
    if covariance is None:
        return np.full((n, p), np.nan)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape == (a,):
        covariance = np.diag(covariance)
    if covariance.shape == (a, a):
        covariance = np.broadcast_to(covariance, (n, a, a))
    if covariance.shape != (n, a, a) or not np.all(np.isfinite(covariance)):
        raise ValueError("anchor_covariance must be (A,), (A,A), or (n,A,A)")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-8, atol=1e-10):
        raise ValueError("anchor_covariance must be symmetric")
    tolerance = 1e-10 * np.maximum(1.0, np.max(np.abs(covariance), axis=(-1, -2)))
    if np.any(np.linalg.eigvalsh(covariance)[:, 0] < -tolerance):
        raise ValueError("anchor_covariance must be positive semidefinite")
    return np.maximum(0.0, np.einsum("pa,nab,pb->np", operator, covariance, operator))


@dataclass
class PredictionResult:
    prediction: np.ndarray
    residual_prediction: np.ndarray
    scores: np.ndarray
    observable: np.ndarray
    null_ambiguity: np.ndarray
    noise_amplification: np.ndarray
    measurement_variance: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    geometry: AnchorGeometry
    basis_error_radius: np.ndarray
    context_error_radius: np.ndarray
    structural_error_radius: np.ndarray
    raw_prediction: np.ndarray
    raw_residual_prediction: np.ndarray
    accepted: np.ndarray
    availability_status: np.ndarray
    metadata: dict = field(default_factory=dict)


class ResponseBridge:
    def __init__(self, basis: ResponseBasis, rcond=1e-8, observability_tolerance=1e-8):
        self.basis = basis
        self.rcond = rcond
        self.observability_tolerance = observability_tolerance

    def predict(self, inputs: ObservableInput, anchor_covariance=None,
                score_norm_bound=None, basis_draws=None,
                context_error_quantiles=None, alpha=0.05) -> PredictionResult:
        """Return predictions and explicitly model-conditional error envelopes.

        Covariance propagation is exact for the linear operator. Its normal
        radius additionally assumes approximately Gaussian measurement error.
        Supplied bootstrap bases give a descriptive refitting-error radius;
        supplied context quantiles require their own calibration assumptions.
        Summing these components is not an unconditional confidence guarantee.
        No error envelope is supplied without an uncertainty source or an
        explicit score bound. Nonobservable targets have NaN point predictions;
        an explicit bound can provide a partial-identification interval, not an
        identified point. ``raw_prediction`` is the minimum-norm diagnostic
        without refusal and must not be presented as an identified estimate.
        """
        if inputs.baseline.shape[1] != self.basis.loadings.shape[0]:
            raise ValueError("baseline protein dimension does not match the basis")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie between zero and one")
        n, p = inputs.baseline.shape
        geometry = anchor_geometry(self.basis, inputs.anchor_indices, self.rcond, self.observability_tolerance)
        residual = inputs.anchor_residuals @ geometry.operator.T
        prediction = inputs.baseline + residual
        variance = _measurement_variance(anchor_covariance, geometry.operator, n)
        measurement_radius = (np.zeros((n, p)) if anchor_covariance is None
                              else norm.ppf(1 - alpha / 2) * np.sqrt(variance))
        structural = np.zeros((n, p))
        if score_norm_bound is not None:
            bound = np.asarray(score_norm_bound, dtype=float)
            if bound.ndim == 0:
                bound = np.full(n, float(bound))
            if bound.shape != (n,) or not np.all(np.isfinite(bound)) or np.any(bound < 0):
                raise ValueError("score_norm_bound must be nonnegative, scalar or length n")
            structural = bound[:, None] * geometry.null_ambiguity[None, :]
        context = np.zeros((n, p))
        if context_error_quantiles is not None:
            try:
                context = np.broadcast_to(np.asarray(context_error_quantiles, dtype=float), (n, p)).copy()
            except ValueError as exc:
                raise ValueError("context_error_quantiles must broadcast to (n,P)") from exc
            if np.any(context < 0) or np.any(np.isnan(context)):
                raise ValueError("context_error_quantiles must be nonnegative and not NaN; infinity denotes insufficient calibration")
        basis_radius = np.zeros((n, p))
        all_observable = geometry.observable.copy()
        draws = [] if basis_draws is None else list(basis_draws)
        unavailable_draws = []
        if draws:
            if len(draws) < 2:
                raise ValueError("at least two bootstrap basis draws are required")
            differences = []
            for draw_index, basis in enumerate(draws):
                if basis is None:
                    unavailable_draws.append(draw_index)
                    continue
                if not isinstance(basis, ResponseBasis):
                    raise ValueError("basis draws must be ResponseBasis objects or explicit unavailable None slots")
                if basis.loadings.shape[0] != p or not np.array_equal(basis.scales, self.basis.scales):
                    raise ValueError("basis draws must use the same proteins and fixed source scales")
                if basis.protein_names != self.basis.protein_names:
                    raise ValueError("basis draws must preserve the protein names and order")
                draw_geometry = anchor_geometry(basis, inputs.anchor_indices, self.rcond, self.observability_tolerance)
                differences.append(np.abs(inputs.anchor_residuals @ draw_geometry.operator.T - residual))
                all_observable &= draw_geometry.observable
                if score_norm_bound is not None:
                    structural = np.maximum(structural, bound[:, None] * draw_geometry.null_ambiguity[None, :])
            if unavailable_draws:
                # A successful-only quantile would condition away genuine
                # bootstrap failures. No basis radius can be estimated here.
                basis_radius[:] = np.inf
            else:
                basis_radius = np.quantile(differences, 1 - alpha, axis=0, method="higher")
        observed_in_available_draws = all_observable.copy()
        status = np.full(p, "available", dtype="U48")
        status[~geometry.observable] = "base_unobservable"
        status[geometry.observable & ~all_observable] = "bootstrap_unobservable"
        if unavailable_draws:
            hidden = np.ones(p, dtype=bool); hidden[inputs.anchor_indices] = False
            all_observable[hidden] = False
            status[hidden] = "bootstrap_draw_unavailable"
            status[inputs.anchor_indices] = "anchor_reconstruction_basis_incomplete"
        has_uncertainty = (anchor_covariance is not None or score_norm_bound is not None
                           or bool(draws) or context_error_quantiles is not None)
        radius = measurement_radius + structural + context + basis_radius
        lower, upper = prediction - radius, prediction + radius
        if not has_uncertainty:
            lower[:] = np.nan
            upper[:] = np.nan
        elif score_norm_bound is None:
            lower[:, ~all_observable] = np.nan
            upper[:, ~all_observable] = np.nan
        if unavailable_draws:
            # Even an explicit score bound cannot identify an uncomputed
            # bootstrap basis; no complete-ensemble envelope is supplied.
            lower[:] = np.nan
            upper[:] = np.nan
        identified_prediction = prediction.copy()
        identified_residual = residual.copy()
        identified_prediction[:, ~all_observable] = np.nan
        identified_residual[:, ~all_observable] = np.nan
        return PredictionResult(prediction=identified_prediction, residual_prediction=identified_residual,
                                scores=inputs.anchor_residuals @ geometry.pseudoinverse.T,
                                observable=geometry.observable, null_ambiguity=geometry.null_ambiguity,
                                noise_amplification=geometry.noise_amplification,
                                measurement_variance=variance, lower=lower, upper=upper,
                                geometry=geometry, basis_error_radius=basis_radius,
                                context_error_radius=context, structural_error_radius=structural,
                                raw_prediction=prediction, raw_residual_prediction=residual,
                                accepted=all_observable,
                                availability_status=status,
                                metadata={"interval_kind": "model_conditional_error_envelope",
                                 "nominal_component_alpha": alpha,
                                 "basis_draw_count": len(draws),
                                 "basis_draw_valid_count": len(draws)-len(unavailable_draws),
                                 "unavailable_basis_draw_indices": unavailable_draws,
                                 "basis_quantile_available": bool(draws) and not bool(unavailable_draws),
                                 "observable_in_all_basis_draws": None if unavailable_draws else all_observable.tolist(),
                                 "observable_in_all_available_basis_draws": observed_in_available_draws.tolist(),
                                 "bootstrap_failure_policy": "Any unavailable draw refuses hidden targets; no successful-only quantile or rank reduction",
                                 "structural_assumption": "residual response lies in fitted basis, or supplied misspecification bound holds",
                                 "guarantee": "No universal coverage under context shift; hidden-target-only drift is undetectable from anchors."})
