"""Exploratory CellBridge checks on open or synthetic data.

Nothing here refits the sealed GSE334503 test, and nothing here changes the
frozen CellBridge solver.
"""
from __future__ import annotations

import time

import numpy as np

from responsebridge.cellbridge import CellBridge, donor_stats, selection_weights
from responsebridge.reliability import split_conformal
from responsebridge.response_baselines import response_scale

TAU0_MIXES = tuple((rho, 0.0) for rho in (0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0))
OPEN_LAMBDAS = tuple(float(v) for v in np.logspace(4, -1, 8))


def selected_lodo(stats, feature_sets, *, ks=(1,), mixes=TAU0_MIXES, lambdas=OPEN_LAMBDAS,
                  primary=None):
    """Leave-one-donor-out standardized MAE of the shared-top-5% model."""
    model = CellBridge(feature_sets, ks=ks, mixes=mixes, lambdas=lambdas,
                       strategy="shared_top", top=0.05)
    pred, truth = model.inner_tensor(stats)
    weights, _ = selection_weights(pred, truth, "shared_top", 0.05)
    selected = (weights[..., None, :] * pred).sum(axis=tuple(range(pred.ndim - 2)))
    rows = []
    primary_rows = []
    for j in range(len(stats)):
        train = np.arange(len(stats)) != j
        scale = response_scale(truth[train])
        err = np.abs(selected[j] - truth[j]) / scale
        rows.append(float(err.mean()))
        if primary is not None and len(primary):
            primary_rows.append(float(err[np.asarray(primary)].mean()))
    return {
        "all_mae": float(np.mean(rows)),
        "primary_mae": float(np.mean(primary_rows)) if primary_rows else None,
        "n_donors": len(stats),
    }


def simulate_metacell_attenuation(sigmas=(1.0, 2.0), *, n_donors=6, n_states=25, repeats=12,
                                  p=4, seed=0):
    """Errors-in-variables when several cells share a latent state.

    Each state is observed ``repeats`` times with independent feature noise.
    Averaging the true state's repeats shrinks slopes less than using every
    noisy cell. K-means on those same noisy features is reported too: it does
    not know the states, and it is not assumed to help.
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=p)
    true_norm = float(np.linalg.norm(beta))
    rows = []
    for sigma in sigmas:
        for mode in ("cells", "kmeans", "oracle_states"):
            stats = []
            for d in range(n_donors):
                arms, outcomes = [], []
                for arm in range(2):
                    latent = rng.normal(size=(n_states, p))
                    observed = latent[:, None, :] + rng.normal(scale=sigma, size=(n_states, repeats, p))
                    flat = observed.reshape(-1, p)
                    response = np.repeat(latent @ beta, repeats)
                    if mode == "cells":
                        z, y = flat, response
                    elif mode == "oracle_states":
                        z, y = observed.mean(1), latent @ beta
                    else:
                        labels = KMeans(n_states, n_init=1, random_state=seed + d + arm).fit_predict(flat)
                        z = np.vstack([flat[labels == c].mean(0) for c in range(n_states)])
                        y = np.array([response[labels == c].mean() for c in range(n_states)])
                    arms.append(z)
                    outcomes.append(y[:, None])
                stats.append(donor_stats(f"d{d}", arms[0], outcomes[0], arms[1], outcomes[1], ks=(1,), seed=d))
            model = CellBridge({"all": np.arange(p)}, ks=(1,), mixes=((0.0, 0.0),),
                               lambdas=(1e-6,), strategy="shared").fit(stats)
            rows.append({
                "sigma": float(sigma),
                "grouping": mode,
                "slope_norm_ratio": float(np.linalg.norm(model.coef[:, 0]) / true_norm),
            })
    return rows


def conformal_coverage_rate(*, n_repeats=200, n_cal=40, n_features=4, alpha=0.1, seed=0):
    """Fraction of exchangeable test donors covered by split-conformal intervals."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=n_features)
    hits = 0
    for _ in range(n_repeats):
        x = rng.normal(size=(n_cal + 1, n_features))
        y = x @ beta + rng.normal(size=n_cal + 1)
        xt, yt = x[:-1], y[:-1]
        gram = xt.T @ xt + np.eye(n_features)
        coef = np.linalg.solve(gram, xt.T @ yt)
        scores = np.abs(yt - xt @ coef)
        quantile = split_conformal(scores, alpha)["quantile"]
        hits += float(np.abs(y[-1] - x[-1] @ coef) <= quantile)
    return {"coverage": hits / n_repeats, "n_repeats": n_repeats, "alpha": alpha, "nominal": 1 - alpha}


def scaling_point(n_cells, *, n_donors=8, p=30, seed=0):
    """Seconds and peak RSS for building donor stats and one nested fit."""
    import resource
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=(p, 2))
    t0 = time.perf_counter()
    stats = []
    for d in range(n_donors):
        zc = rng.normal(size=(n_cells, p))
        zp = zc + rng.normal(scale=0.3, size=(n_cells, p))
        yc = zc @ beta + rng.normal(scale=0.2, size=(n_cells, 2))
        yp = zp @ beta + rng.normal(scale=0.2, size=(n_cells, 2))
        stats.append(donor_stats(f"d{d}", zc, yc, zp, yp, ks=(1,), seed=d))
        del zc, zp, yc, yp
    build = time.perf_counter() - t0
    model = CellBridge({"all": np.arange(p)}, ks=(1,), mixes=((1.0, 0.0),),
                       lambdas=(1.0, 0.1), strategy="shared")
    t1 = time.perf_counter()
    model.fit(stats)
    fit = time.perf_counter() - t1
    import platform
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    return {"n_cells_per_arm": int(n_cells), "n_donors": n_donors, "n_features": p,
            "build_seconds": build, "fit_seconds": fit, "peak_rss_bytes": int(rss_bytes)}
