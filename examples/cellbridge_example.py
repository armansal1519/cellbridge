"""Seeded synthetic example. The expected numbers are in expected_output.json."""
from __future__ import annotations

import numpy as np

from cellbridge import fit


def donors(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(4):
        zc = rng.normal(size=(20, 6))
        zp = rng.normal(size=(20, 6))
        rows.append({
            "donor": f"d{i}",
            "z_control": zc,
            "y_control": zc[:, :1],
            "z_perturbed": zp,
            "y_perturbed": zp[:, :1] + 0.4,
        })
    return rows


def run():
    model = fit(
        donors(), n_genes=4, n_anchors=0, ks=(1,), mixes=((0.0, 0.0),),
        lambdas=(1.0,), strategy="shared_top", top=0.05, seed=0,
    )
    query = np.zeros(6)
    point = model.predict(query)
    parts = model.contributions(query, target=0)
    low, high = model.intervals(query, np.full(12, 0.25), alpha=0.1)
    return {
        "predict": np.asarray(point, dtype=float).reshape(-1).tolist(),
        "contributions": np.asarray(parts, dtype=float).reshape(-1).tolist(),
        "low": np.asarray(low, dtype=float).reshape(-1).tolist(),
        "high": np.asarray(high, dtype=float).reshape(-1).tolist(),
    }


if __name__ == "__main__":
    print(run())
