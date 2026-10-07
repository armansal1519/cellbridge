import numpy as np
import pandas as pd

from cellbridge import fit, fit_anndata


def _donors():
    rng = np.random.default_rng(0)
    beta = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.0])
    donors = []
    for i in range(4):
        zc = rng.normal(size=(24, 6))
        zp = rng.normal(size=(24, 6))
        donors.append({
            "donor": f"d{i}",
            "z_control": zc,
            "y_control": zc @ beta + rng.normal(scale=0.05, size=24),
            "z_perturbed": zp,
            "y_perturbed": zp @ beta + 0.4 + rng.normal(scale=0.05, size=24),
        })
    return donors


def test_fit_predict_contributions_and_intervals():
    model = fit(_donors(), n_genes=4, n_anchors=0, ks=(1,), mixes=((0.0, 0.0),), lambdas=(1e-3,),
                strategy="shared", top=1.0)
    point = model.predict(np.zeros(6))
    parts = model.contributions(np.zeros(6), 0)
    assert point.shape == (1, 1)
    assert parts.shape == (1, 6)
    assert abs(float(parts.sum() + model.model.my[0] - point[0, 0])) < 1e-8
    low, high = model.intervals(np.zeros(6), np.full(20, 0.2), alpha=0.1)
    assert float(low[0, 0]) <= float(point[0, 0]) <= float(high[0, 0])


def test_anndata_interface_uses_obs_and_obsm():
    rng = np.random.default_rng(1)
    rows = []
    blocks = []
    proteins = []
    for donor in ("a", "b", "c"):
        for arm in ("control", "perturbed"):
            z = rng.normal(size=(8, 6))
            blocks.append(z)
            proteins.append(z[:, :1] + (0.3 if arm == "perturbed" else 0.0))
            rows.extend({"donor": donor, "arm": arm} for _ in range(8))

    class Table:
        X = np.vstack(blocks)
        obs = pd.DataFrame(rows)
        obsm = {"protein": np.vstack(proteins)}

    model = fit_anndata(Table(), n_genes=4, n_anchors=0, ks=(1,), mixes=((1.0, 0.0),), lambdas=(1.0,),
                        strategy="shared", top=1.0)
    assert model.predict(np.ones(6)).shape == (1, 1)
