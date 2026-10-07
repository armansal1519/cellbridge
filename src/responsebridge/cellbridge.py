"""CellBridge: closed-form bridging of single-cell and donor-response information.

For target t, cell i of donor d in arm a (control or perturbed) with observed
features z_i (log-normalised RNA, RNA technical covariates and, optionally,
anchor ADTs):

    y_it = c_dt + gamma_t [a = perturbed] + z_i' beta_t + e_it

with unpenalised donor effects c_dt and response intercept gamma_t. Profiling
both out splits the arm-balanced squared loss into a within-arm (cell-state)
term and a between-arm (donor-response) term. ``rho`` reweights the latter:

* rho = 0   slopes come only from within-arm cell-state covariation;
* rho = 1   ordinary pooled cell-level least squares with donor and arm effects;
* rho -> oo slopes come only from donor responses (pseudobulk response ridge).

``tau`` adds a third, between-donor (abundance) term: donor arm-averaged means
(ybar_d - ybar) ~ (zbar_d - zbar)' beta. The three variance components (cell
state, donor response, donor abundance) thus share one slope vector; a
configuration is the pair (rho, tau).

The donor response is linear in the arm means, so the prediction for a query
donor is exact: m_y + (dz_q - m_z)' beta, where m_* are training-donor mean
responses. With lambda -> oo every target falls back to the training mean.
Metacells (size k, RNA-only k-means inside each donor-arm) reduce
errors-in-variables attenuation of cell-level slopes before they are applied
to low-noise arm means.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import numpy as np
from scipy import sparse


@dataclass
class DonorStats:
    donor: str
    within: dict
    dz: np.ndarray
    dy: np.ndarray
    n_cells: tuple = ()
    lz: np.ndarray | None = None
    ly: np.ndarray | None = None


def metacell_labels(embedding, k, seed):
    from sklearn.cluster import KMeans
    n = len(embedding)
    clusters = max(1, n // int(k))
    if clusters >= n:
        return np.arange(n)
    if clusters == 1:
        return np.zeros(n, dtype=int)
    return KMeans(clusters, n_init=1, max_iter=100, random_state=seed).fit_predict(embedding)


def _arm_scatter(z, y, labels):
    n = len(z)
    if labels is None:
        zm, ym, w = z, y, np.full(n, 1.0 / n)
    else:
        _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
        s = sparse.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(len(counts), n))
        zm = np.asarray(s @ z) / counts[:, None]
        ym = np.asarray(s @ y) / counts[:, None]
        w = counts / n
    zc = zm - w @ zm
    yc = ym - w @ ym
    zw = zc * w[:, None]
    return 0.5 * (zw.T @ zc), 0.5 * (zw.T @ yc)


def rna_embedding(z_rna, n_components=30, seed=0):
    from sklearn.decomposition import PCA
    z = np.asarray(z_rna, dtype=float)
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-6)
    m = min(n_components, z.shape[0] - 1, z.shape[1])
    if m < 1:
        return z[:, :1]
    return PCA(m, svd_solver="randomized", random_state=seed).fit_transform(z)


def donor_stats(donor, z_control, y_control, z_perturbed, y_perturbed, *, ks=(1,),
                rna_columns=None, seed=0):
    """Sufficient statistics for one donor; arms receive equal total weight."""
    z_control, z_perturbed = (np.asarray(v, dtype=float) for v in (z_control, z_perturbed))
    y_control, y_perturbed = (np.asarray(v, dtype=float) for v in (y_control, y_perturbed))
    if len(z_control) < 2 or len(z_perturbed) < 2:
        raise ValueError("Each arm needs at least two cells")
    cols = np.arange(z_control.shape[1]) if rna_columns is None else np.asarray(rna_columns)
    within = {}
    embed = {}
    for k in ks:
        w_total = 0
        u_total = 0
        for arm, (z, y) in enumerate([(z_control, y_control), (z_perturbed, y_perturbed)]):
            if k == 1:
                labels = None
            else:
                if arm not in embed:
                    embed[arm] = rna_embedding(z[:, cols], seed=seed + arm)
                labels = metacell_labels(embed[arm], k, seed + 7 * arm + int(k))
            w, u = _arm_scatter(z, y, labels)
            w_total = w_total + w
            u_total = u_total + u
        within[int(k)] = {"W": w_total, "u": u_total}
    return DonorStats(str(donor), within, z_perturbed.mean(0) - z_control.mean(0),
                      y_perturbed.mean(0) - y_control.mean(0),
                      (len(z_control), len(z_perturbed)),
                      0.5 * (z_perturbed.mean(0) + z_control.mean(0)),
                      0.5 * (y_perturbed.mean(0) + y_control.mean(0)))


class _Pooled:
    """Running sums over donors so that inner folds subtract one donor."""

    def __init__(self, stats):
        self.stats = list(stats)
        self.ks = sorted(self.stats[0].within)
        self.W = {k: sum(s.within[k]["W"] for s in self.stats) for k in self.ks}
        self.u = {k: sum(s.within[k]["u"] for s in self.stats) for k in self.ks}

    def without(self, j, k):
        s = self.stats[j].within[k]
        return self.W[k] - s["W"], self.u[k] - s["u"]


def _prepare(W, u, dz, dy, cols, n_donors, lz=None, ly=None):
    W = W[np.ix_(cols, cols)] / n_donors
    u = u[cols] / n_donors
    var = np.diag(W).copy()
    keep = var > 1e-10 * max(var.max(), 1e-300)
    cols, W, u, var = cols[keep], W[np.ix_(keep, keep)], u[keep], var[keep]
    sd = np.sqrt(var)
    Wt = W / np.outer(sd, sd)
    Wt = 0.5 * (Wt + Wt.T)
    lam, Q = np.linalg.eigh(Wt)
    lam = np.clip(lam, 0, None)
    dzc = dz[:, cols]
    mz = dzc.mean(0)
    my = dy.mean(0)
    U = ((dzc - mz) / sd).T / (2 * np.sqrt(n_donors))
    V = (dy - my) / (2 * np.sqrt(n_donors))
    f = {"cols": cols, "sd": sd, "lam": lam, "Q": Q, "a": Q.T @ (u / sd[:, None]),
         "P": Q.T @ U, "mz": mz, "my": my}
    f["PV"] = f["P"] @ V
    if lz is not None:
        lzc = lz[:, cols]
        Ul = ((lzc - lzc.mean(0)) / sd).T / np.sqrt(n_donors)
        Vl = (ly - ly.mean(0)) / np.sqrt(n_donors)
        f["Pl"] = Q.T @ Ul
        f["PlVl"] = f["Pl"] @ Vl
    return f


def _coef_eigen(f, rho, tau, lam):
    dinv = 1.0 / (f["lam"] + lam)
    c = f["a"].copy()
    parts = []
    if rho > 0:
        c += rho * f["PV"]
        parts.append(np.sqrt(rho) * f["P"])
    if tau > 0:
        if "Pl" not in f:
            raise ValueError("Between-donor term requested without donor level means")
        c += tau * f["PlVl"]
        parts.append(np.sqrt(tau) * f["Pl"])
    t = dinv[:, None] * c
    if not parts:
        return t
    P = np.hstack(parts)
    M = np.eye(P.shape[1]) + P.T @ (dinv[:, None] * P)
    return t - dinv[:, None] * (P @ np.linalg.solve(M, P.T @ t))


def _path(f, queries, mixes, lambdas):
    q = f["Q"].T @ ((queries[:, f["cols"]] - f["mz"]) / f["sd"]).T
    out = np.empty((len(mixes), len(lambdas), len(queries), len(f["my"])))
    for r, (rho, tau) in enumerate(mixes):
        for l, lam in enumerate(lambdas):
            out[r, l] = f["my"] + q.T @ _coef_eigen(f, rho, tau, lam)
    return out


def _coef_original(f, rho, tau, lam, width):
    beta = np.zeros((width, len(f["my"])))
    beta[f["cols"]] = (f["Q"] @ _coef_eigen(f, rho, tau, lam)) / f["sd"][:, None]
    return beta, f["mz"]


DEFAULT_RHOS = (0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0)
DEFAULT_TAUS = (0.0, 1.0, 16.0, 256.0)
DEFAULT_MIXES = tuple((r, t) for t in DEFAULT_TAUS for r in DEFAULT_RHOS)
DEFAULT_LAMBDAS = tuple(float(v) for v in np.logspace(6, -2, 25))
STRATEGIES = ("per_target", "shared", "shared_top", "per_target_top")


def selection_weights(inner_pred, truth, strategy="shared_top", top=0.05):
    """Configuration weights per target from leave-one-donor-out predictions.

    inner_pred has shape (*grid, D, T) and truth (D, T). Shared strategies pool
    targets after standardising by the training-donor response SD, which keeps
    selection noise low when only a handful of donors are available. ``*_top``
    averages the best ``top`` fraction of configurations; an average of linear
    fits is itself linear, so the exact additive decomposition is preserved.
    """
    grid = inner_pred.shape[:-2]
    T = truth.shape[1]
    scale = np.maximum(truth.std(0), 1e-6)
    err = np.abs(inner_pred - truth).mean(axis=-2) / scale  # (*grid, T)
    flat = err.reshape(-1, T)
    n = flat.shape[0]
    weights = np.zeros((n, T))
    m = max(1, int(round(top * n)))
    if strategy == "per_target":
        weights[flat.argmin(0), np.arange(T)] = 1
    elif strategy == "per_target_top":
        order = np.argsort(flat, axis=0, kind="stable")[:m]
        for t in range(T):
            weights[order[:, t], t] = 1.0 / m
    elif strategy in ("shared", "shared_top"):
        pooled = flat.mean(1)
        order = np.argsort(pooled, kind="stable")
        chosen = order[:1] if strategy == "shared" else order[:m]
        weights[chosen] = 1.0 / len(chosen)
    else:
        raise ValueError(f"Unknown selection strategy {strategy}")
    return weights.reshape(*grid, T), err


@dataclass
class CellBridge:
    feature_sets: dict
    ks: tuple = (1,)
    mixes: tuple = DEFAULT_MIXES
    lambdas: tuple = DEFAULT_LAMBDAS
    strategy: str = "shared_top"
    top: float = 0.05
    selection: dict = field(default_factory=dict)

    @property
    def grid_shape(self):
        return (len(self.feature_sets), len(self.ks), len(self.mixes), len(self.lambdas))

    @staticmethod
    def _arrays(stats, keep):
        dz = np.stack([s.dz for s in stats])[keep]
        dy = np.stack([s.dy for s in stats])[keep]
        if any(s.lz is None for s in stats):
            return dz, dy, None, None
        return dz, dy, np.stack([s.lz for s in stats])[keep], np.stack([s.ly for s in stats])[keep]

    def path_tensor(self, stats, queries, pooled=None, drop=None):
        """Predictions of every configuration, shape (F, K, M, L, Q, T)."""
        pooled = pooled or _Pooled(stats)
        D = len(pooled.stats)
        keep = np.ones(D, dtype=bool)
        if drop is not None:
            keep[drop] = False
        dz, dy, lz, ly = self._arrays(pooled.stats, keep)
        queries = np.atleast_2d(queries)
        out = np.empty(self.grid_shape + (len(queries), dy.shape[1]))
        for fi, name in enumerate(self.feature_sets):
            cols = np.asarray(self.feature_sets[name], dtype=int)
            for ki, k in enumerate(self.ks):
                W, u = (pooled.W[k], pooled.u[k]) if drop is None else pooled.without(drop, k)
                f = _prepare(W, u, dz, dy, cols, int(keep.sum()), lz, ly)
                out[fi, ki] = _path(f, queries, self.mixes, self.lambdas)
        return out

    def inner_tensor(self, stats):
        """Leave-one-donor-out predictions for every configuration."""
        if len(stats) < 3:
            raise ValueError("Nested selection needs at least three training donors")
        pooled = _Pooled(stats)
        dz = np.stack([s.dz for s in stats])
        dy = np.stack([s.dy for s in stats])
        pred = np.empty(self.grid_shape + (len(stats), dy.shape[1]))
        for j in range(len(stats)):
            pred[..., j, :] = self.path_tensor(stats, dz[[j]], pooled, drop=j)[..., 0, :]
        return pred, dy

    def fit(self, stats, inner=None):
        self.train_donors = [s.donor for s in stats]
        pred, dy = inner if inner is not None else self.inner_tensor(stats)
        w, err = selection_weights(pred, dy, self.strategy, self.top)
        self.weights, self.inner_error = w, err
        pooled = _Pooled(stats)
        dz, dy, lz, ly = self._arrays(stats, np.ones(len(stats), dtype=bool))
        width, T = dz.shape[1], dy.shape[1]
        self.coef = np.zeros((width, T))
        self.my = dy.mean(0)
        self.mz = dz.mean(0)
        names = list(self.feature_sets)
        for fi, ki in itertools.product(range(len(names)), range(len(self.ks))):
            block = w[fi, ki]
            if not block.any():
                continue
            k = self.ks[ki]
            f = _prepare(pooled.W[k], pooled.u[k], dz, dy,
                         np.asarray(self.feature_sets[names[fi]], dtype=int), len(stats), lz, ly)
            for mi, li in zip(*np.nonzero(block.any(axis=-1))):
                rho, tau = self.mixes[mi]
                beta, _ = _coef_original(f, rho, tau, self.lambdas[li], width)
                self.coef += beta * block[mi, li]
        flat = w.reshape(-1, T)
        self.selected = []
        for t in range(T):
            idx = np.flatnonzero(flat[:, t])
            cfg = [np.unravel_index(i, self.grid_shape) for i in idx]
            wt = [flat[i, t] for i in idx]
            self.selected.append({
                "n_configurations": len(idx),
                "rna_anchor_weight": float(sum(v for v, c in zip(wt, cfg) if names[c[0]] != "rna")),
                "mean_log10_lambda": float(sum(v * np.log10(self.lambdas[c[3]]) for v, c in zip(wt, cfg))),
                "rho_weights": {str(r): float(sum(v for v, c in zip(wt, cfg) if self.mixes[c[2]][0] == r))
                                for r in sorted({m[0] for m in self.mixes})},
                "tau_weights": {str(r): float(sum(v for v, c in zip(wt, cfg) if self.mixes[c[2]][1] == r))
                                for r in sorted({m[1] for m in self.mixes})},
                "k_weights": {str(k): float(sum(v for v, c in zip(wt, cfg) if self.ks[c[1]] == k)) for k in self.ks}})
        return self

    def predict(self, dz):
        dz = np.atleast_2d(np.asarray(dz, dtype=float))
        return self.my + (dz - self.mz) @ self.coef

    def contributions(self, dz, t):
        """Exact additive decomposition of the predicted response of target t."""
        dz = np.atleast_2d(np.asarray(dz, dtype=float))
        return (dz - self.mz) * self.coef[:, t]


def fixed_config_predictions(stats, queries, cols, k, rho, tau, lambdas):
    """Diagnostic path for one feature set/k/(rho, tau) on fixed training donors."""
    pooled = _Pooled(stats)
    dz, dy, lz, ly = CellBridge._arrays(stats, np.ones(len(stats), dtype=bool))
    f = _prepare(pooled.W[k], pooled.u[k], dz, dy, np.asarray(cols, dtype=int), len(stats), lz, ly)
    return _path(f, np.atleast_2d(queries), ((rho, tau),), lambdas)[0]
