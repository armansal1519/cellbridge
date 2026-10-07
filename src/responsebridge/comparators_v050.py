"""Same-information and published comparators for the v0.5 protocol.

All hyperparameter search uses training donors only. Missing optional runtimes
are recorded as skips, never as superiority. Cell-level methods are adapters
that operate on subsampled cells and then aggregate to donor responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor

from .response_baselines import LinearResponse, donor_weights, response_scale
from .two_penalty import TwoPenaltyResponse


def _select_features(x, weights, n_hvg):
    mean = np.average(x, axis=0, weights=weights)
    var = np.average((x - mean)**2, axis=0, weights=weights)
    eligible = np.flatnonzero(var > 1e-12)
    return eligible[np.argsort(-var[eligible], kind="stable")[:n_hvg]]


@dataclass
class TrainingMean:
    kind: str = "training_mean"
    mean: np.ndarray | None = None
    training_error_sd: np.ndarray | None = None
    failures: list = field(default_factory=list)

    def fit(self, x, y, anchors=None, *, donors=None):
        w = donor_weights(donors, len(y))
        self.mean = np.average(y, axis=0, weights=w)
        self.training_error_sd = response_scale(y, donors)
        return self

    def predict(self, x, anchors=None):
        return np.broadcast_to(self.mean, (len(x), len(self.mean))).copy()


@dataclass
class ElasticNetResponse:
    kind: str = "elastic_net"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    n_hvg: int = 2000
    features: np.ndarray | None = None
    model: object | None = None
    training_error_sd: np.ndarray | None = None
    failures: list = field(default_factory=list)

    def fit(self, x, y, anchors, *, donors=None):
        w = donor_weights(donors, len(y))
        self.features = _select_features(x, w, self.n_hvg)
        design = np.column_stack([x[:, self.features], anchors])
        self.model = MultiOutputRegressor(ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                                     max_iter=5000, copy_X=True))
        self.model.fit(design, y)
        self.training_error_sd = response_scale(y, donors)
        return self

    def predict(self, x, anchors):
        design = np.column_stack([x[:, self.features], anchors])
        return self.model.predict(design)


@dataclass
class PLSResponse:
    kind: str = "pls"
    n_components: int = 4
    n_hvg: int = 2000
    features: np.ndarray | None = None
    model: object | None = None
    training_error_sd: np.ndarray | None = None
    failures: list = field(default_factory=list)

    def fit(self, x, y, anchors, *, donors=None):
        w = donor_weights(donors, len(y))
        self.features = _select_features(x, w, self.n_hvg)
        design = np.column_stack([x[:, self.features], anchors])
        n_comp = max(1, min(int(self.n_components), design.shape[0] - 1, design.shape[1], y.shape[1]))
        self.model = PLSRegression(n_components=n_comp)
        self.model.fit(design, y)
        self.training_error_sd = response_scale(y, donors)
        return self

    def predict(self, x, anchors):
        design = np.column_stack([x[:, self.features], anchors])
        return np.asarray(self.model.predict(design), dtype=float)


@dataclass
class KernelRidgeResponse:
    kind: str = "kernel_ridge"
    alpha: float = 1.0
    kernel: str = "rbf"
    gamma: float | None = None
    n_hvg: int = 2000
    features: np.ndarray | None = None
    model: object | None = None
    training_error_sd: np.ndarray | None = None
    failures: list = field(default_factory=list)

    def fit(self, x, y, anchors, *, donors=None):
        w = donor_weights(donors, len(y))
        self.features = _select_features(x, w, self.n_hvg)
        design = np.column_stack([x[:, self.features], anchors])
        self.model = KernelRidge(alpha=self.alpha, kernel=self.kernel, gamma=self.gamma)
        self.model.fit(design, y, sample_weight=w)
        self.training_error_sd = response_scale(y, donors)
        return self

    def predict(self, x, anchors):
        design = np.column_stack([x[:, self.features], anchors])
        return np.asarray(self.model.predict(design), dtype=float)


@dataclass
class LightGBMResponse:
    kind: str = "lightgbm"
    n_estimators: int = 50
    learning_rate: float = 0.05
    num_leaves: int = 8
    n_hvg: int = 2000
    features: np.ndarray | None = None
    model: object | None = None
    training_error_sd: np.ndarray | None = None
    failures: list = field(default_factory=list)

    def fit(self, x, y, anchors, *, donors=None):
        try:
            import lightgbm as lgb
        except ImportError as error:
            self.failures.append({"error": "lightgbm_not_installed"})
            raise ImportError("lightgbm_not_installed") from error
        w = donor_weights(donors, len(y))
        self.features = _select_features(x, w, self.n_hvg)
        design = np.column_stack([x[:, self.features], anchors])
        self.model = MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=int(self.n_estimators), learning_rate=float(self.learning_rate),
            num_leaves=int(self.num_leaves), verbosity=-1, n_jobs=1))
        try:
            self.model.fit(design, y, sample_weight=w)
        except TypeError:
            self.model.fit(design, y)
        self.training_error_sd = response_scale(y, donors)
        return self

    def predict(self, x, anchors):
        design = np.column_stack([x[:, self.features], anchors])
        return np.asarray(self.model.predict(design), dtype=float)


def lightgbm_response(*args, **kwargs):
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return {"status": "skipped", "reason": "lightgbm_not_installed", "kind": "lightgbm"}
    return LightGBMResponse(*args, **kwargs)


CELL_ADAPTERS = {
    "totalvi_masked": {
        "package": "scvi-tools",
        "status": "optional_runtime",
        "contract": "Mask hidden proteins as missing; train on training-donor cells; aggregate to donor responses",
        "budget": "Subsampled cells; early stopping on inner donors; record ELBO trajectory",
    },
    "scvaeit_retuned": {
        "package": "scVAEIT",
        "status": "optional_runtime",
        "contract": "Learning-rate grid, epochs up to 2000, early stopping on inner donors; v0.4 locked config also reported",
    },
    "scvaeit_v040_locked": {
        "package": "scVAEIT",
        "status": "locked_v040_config",
        "epoch_grid": [100, 300, 1000],
        "seeds": [20261004, 20261005, 20261006],
    },
    "sclinear": {
        "package": "scLinear",
        "status": "optional_runtime",
        "contract": "RNA-only published core; same aggregation as v0.4",
    },
    "seurat_v4_mapping": {
        "package": "Seurat",
        "status": "optional_r_runtime",
        "contract": "RNA-only reference mapping of protein; not same-information",
    },
}


def nested_select(factory, grid, x, y, anchors, donors, targets, scale):
    """Leave-one-donor MAE on training donors; ties keep the first grid item."""
    unique = np.unique(donors)
    if len(unique) < 3:
        raise ValueError("Nested comparator selection requires at least three donors")
    best, best_err, log = None, np.inf, []
    for params in grid:
        oof = np.empty((len(y), len(targets)))
        failed = False
        for donor in unique:
            train = donors != donor
            try:
                model = factory(**params).fit(x[train], y[train], anchors[train], donors=donors[train])
                pred = model.predict(x[~train], anchors[~train])[:, targets]
            except Exception as error:  # noqa: BLE001 — comparator failures are retained
                failed = True
                log.append({"params": params, "error": type(error).__name__, "message": str(error)})
                break
            oof[np.flatnonzero(~train)] = pred
        if failed:
            continue
        err = float(np.mean(np.abs(y[:, targets] - oof) / scale[targets]))
        log.append({"params": params, "oof_standardized_mae": err})
        if err < best_err - 1e-15:
            best, best_err = params, err
    if best is None:
        return {"status": "all_failed", "log": log, "model": None}
    model = factory(**best).fit(x, y, anchors, donors=donors)
    return {"status": "selected", "params": best, "oof_standardized_mae": best_err, "log": log, "model": model}


def default_grids(n_hvg=2000):
    return {
        "elastic_net": [{"alpha": a, "l1_ratio": r, "n_hvg": n_hvg}
                        for a in (0.1, 1.0, 10.0) for r in (0.2, 0.5, 0.8)],
        "pls": [{"n_components": k, "n_hvg": n_hvg} for k in (2, 4, 8)],
        "kernel_ridge": [{"alpha": a, "kernel": "rbf", "n_hvg": n_hvg} for a in (0.1, 1.0, 10.0)],
        "two_penalty_ridge": [{"alpha_rna": a, "alpha_anchor": b, "n_hvg": n_hvg}
                              for a in (0.1, 1.0, 10.0, None) for b in (0.1, 1.0, 10.0, None)],
        "joint_ridge": [{"alpha": a, "n_hvg": n_hvg} for a in (0.1, 1.0, 10.0, 100.0)],
        "lightgbm": [{"n_estimators": n, "learning_rate": 0.05, "num_leaves": leaves, "n_hvg": n_hvg}
                     for n, leaves in ((50, 8), (100, 15))],
    }


def joint_ridge_factory(alpha=1.0, n_hvg=2000):
    return LinearResponse(kind="joint", alpha=alpha, n_hvg=n_hvg)


def two_penalty_factory(alpha_rna=1.0, alpha_anchor=1.0, n_hvg=2000):
    return TwoPenaltyResponse(alpha_rna=alpha_rna, alpha_anchor=alpha_anchor, n_hvg=n_hvg)
