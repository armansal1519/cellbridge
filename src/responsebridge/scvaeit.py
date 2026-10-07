"""Optional official scVAEIT baseline with an observed-anchor-only query API.

This is a Gaussian, train-standardized adapter for the project's independently
transformed RNA and ADT inputs. Group means are allowed but are an adaptation of
the cell-level published method, not an exact reproduction of paper preprocessing.
TensorFlow is imported lazily; use the isolated .venv-scvaeit environment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np


def _matrix(values, name, *, width=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"{name} has {values.shape[1]} columns; expected {width}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite observed values")
    return values


def _indices(indices, width):
    original = np.asarray(indices)
    if original.ndim != 1 or (original.size and not np.issubdtype(original.dtype, np.integer)):
        raise ValueError("Anchor indices must be a one-dimensional integer array")
    values = original.astype(np.int64)
    if len(set(values.tolist())) != len(values) or np.any(values < 0) or np.any(values >= width):
        raise ValueError("Anchor indices must be unique and within the protein panel")
    return values


def _probabilities(weights, rows):
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all() or np.any(values <= 0) or not np.isfinite(values.sum()):
        raise ValueError("Training weights must be aligned, finite and strictly positive")
    return values / values.sum()


def weighted_epoch_indices(probabilities, sample_count, seed, epoch):
    """One finite, independently seeded replacement sample of development rows."""
    probabilities = _probabilities(probabilities, len(probabilities))
    if sample_count < 1 or epoch < 0:
        raise ValueError("Positive sample_count and nonnegative epoch required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 81037, epoch]))
    return rng.choice(len(probabilities), size=sample_count, replace=True, p=probabilities)


def _weighted_gaussian_stats(data, probabilities, block_sizes):
    """Use supported upstream config fields to avoid unweighted recentering."""
    values = np.asarray(data, dtype=np.float64)
    mean = np.average(values, axis=0, weights=probabilities)
    std = np.sqrt(np.average((values-mean)**2, axis=0, weights=probabilities))
    low, high = np.empty_like(mean), np.empty_like(mean)
    start = 0
    for size in block_sizes:
        stop = start + size
        low[start:stop] = min(np.min(mean[start:stop]-3*std[start:stop]), np.min(values[:, start:stop]))
        high[start:stop] = max(np.max(mean[start:stop]+3*std[start:stop]), np.max(values[:, start:stop]))
        start = stop
    return {"mean_vals": mean.astype(np.float32), "min_vals": low.astype(np.float32), "max_vals": high.astype(np.float32)}


@dataclass
class TrainTransform:
    n_hvg: int = 2000

    def fit(self, x, y, weights=None):
        x, y = _matrix(x, "training RNA"), _matrix(y, "training ADT")
        if len(x) != len(y) or len(x) < 4 or not y.shape[1]:
            raise ValueError("Training needs at least four aligned observations and one protein")
        if self.n_hvg < 1:
            raise ValueError("n_hvg must be positive")
        probability = None if weights is None else _probabilities(weights, len(x))
        if probability is None:
            variance = np.var(x.astype(np.float64), axis=0)
        else:
            x64 = x.astype(np.float64)
            mean = np.average(x64, axis=0, weights=probability)
            variance = np.average((x64-mean)**2, axis=0, weights=probability)
        eligible = np.flatnonzero(variance > 1e-12)
        if not len(eligible):
            raise ValueError("No variable RNA features in development data")
        self.features = eligible[np.argsort(-variance[eligible], kind="stable")[:self.n_hvg]]
        self.rna_width, self.protein_width = x.shape[1], y.shape[1]
        if probability is None:
            self.x_mean = np.mean(x[:, self.features], axis=0)
            self.x_scale = np.maximum(np.std(x[:, self.features], axis=0), 1e-6)
            self.y_mean = np.mean(y, axis=0)
            self.y_scale = np.maximum(np.std(y, axis=0), 1e-6)
        else:
            self.x_mean = mean[self.features]
            self.x_scale = np.maximum(np.sqrt(variance[self.features]), 1e-6)
            y64 = y.astype(np.float64)
            self.y_mean = np.average(y64, axis=0, weights=probability)
            self.y_scale = np.maximum(np.sqrt(np.average((y64-self.y_mean)**2, axis=0, weights=probability)), 1e-6)
        return self

    def training(self, x, y):
        x = _matrix(x, "training RNA", width=self.rna_width)
        y = _matrix(y, "training ADT", width=self.protein_width)
        if len(x) != len(y):
            raise ValueError("Training RNA/ADT row mismatch")
        return np.concatenate(((x[:, self.features] - self.x_mean) / self.x_scale,
                               (y - self.y_mean) / self.y_scale), axis=1).astype(np.float32)

    def query(self, x, anchor_values, anchor_indices):
        """Hidden protein columns are never accepted, normalized or inspected."""
        x = _matrix(x, "query RNA", width=self.rna_width)
        anchors = _indices(anchor_indices, self.protein_width)
        observed = _matrix(anchor_values, "observed ADT anchors", width=len(anchors))
        if len(x) != len(observed):
            raise ValueError("Query RNA/anchor row mismatch")
        width = len(self.features) + self.protein_width
        inputs = np.zeros((len(x), width), dtype=np.float32)
        masks = np.full_like(inputs, -1.0)  # upstream -1 means genuinely missing
        inputs[:, :len(self.features)] = (x[:, self.features] - self.x_mean) / self.x_scale
        masks[:, :len(self.features)] = 0.0
        cols = len(self.features) + anchors
        inputs[:, cols] = (observed - self.y_mean[anchors]) / self.y_scale[anchors]
        masks[:, cols] = 0.0
        return inputs, masks

    def save(self, path):
        np.savez_compressed(path, n_hvg=self.n_hvg, features=self.features,
                            rna_width=self.rna_width, protein_width=self.protein_width,
                            x_mean=self.x_mean, x_scale=self.x_scale,
                            y_mean=self.y_mean, y_scale=self.y_scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(int(data["n_hvg"]))
            for key in ["features", "x_mean", "x_scale", "y_mean", "y_scale"]:
                setattr(result, key, data[key])
            result.rna_width, result.protein_width = int(data["rna_width"]), int(data["protein_width"])
        return result


def _runtime(threads):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        from scVAEIT.VAEIT import VAEIT
    except ImportError as error:
        raise ImportError("scVAEIT is optional; run this adapter with .venv-scvaeit/bin/python (see docs/scvaeit.md)") from error
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        if tf.config.threading.get_intra_op_parallelism_threads() not in range(1, threads + 1):
            raise RuntimeError("Start a separate scVAEIT worker to enforce its thread limit")
    return tf, VAEIT


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def _upstream_sources():
    import scVAEIT
    root = Path(scVAEIT.__file__).resolve().parent
    return {path.name: _sha(path) for path in sorted(root.glob("*.py"))}


def _jsonable(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass
class ScVAEITRegressor:
    epochs: int = 100
    batch_size: int = 64
    n_hvg: int = 2000
    hidden: int = 128
    latent: int = 16
    learning_rate: float = 3e-4
    mc_samples: int = 32
    seed: int = 20261004
    threads: int = 4

    def fit(self, x, y, weights=None):
        """Fit only development rows. No random cell split or query adaptation."""
        if min(self.epochs, self.batch_size, self.hidden, self.latent, self.mc_samples, self.threads) < 1 or self.learning_rate <= 0:
            raise ValueError("Training and inference dimensions/budgets must be positive")
        tf, VAEIT = _runtime(self.threads)
        tf.keras.utils.set_random_seed(self.seed)
        self.transform = TrainTransform(self.n_hvg).fit(x, y, weights)
        data = self.transform.training(x, y)
        self.training_probabilities = None if weights is None else _probabilities(weights, len(data))
        self.sampling_indices = []
        rna_width, protein_width = len(self.transform.features), self.transform.protein_width
        self.upstream_config = {
            "dim_input_arr": np.array([rna_width, protein_width], dtype=np.int32),
            "dimensions": np.array([self.hidden], dtype=np.int32),
            "dim_latent": self.latent,
            "dim_block": np.array([rna_width, protein_width], dtype=np.int32),
            "dim_block_enc": np.array([self.hidden, min(32, self.hidden)], dtype=np.int32),
            "dim_block_dec": np.array([self.hidden, min(32, self.hidden)], dtype=np.int32),
            "dim_block_embed": np.array([8, 8], dtype=np.int32),
            "dist_block": np.array(["Gaussian", "Gaussian"]),
            "block_names": np.array(["RNA", "ADT"]),
            "uni_block_names": np.array(["RNA", "ADT"]),
            "p_modal": np.array([.5, .5], dtype=np.float32),
            "p_feat": .2,
            "beta_modal": np.array([1.0, 1.0], dtype=np.float32),
            "beta_kl": 2.0, "beta_unobs": .5, "beta_reverse": 0.0,
            "skip_conn": False, "gamma": 0.0,
        }
        if self.training_probabilities is not None:
            self.upstream_config.update(_weighted_gaussian_stats(data, self.training_probabilities, [rna_width, protein_width]))
        self.model = VAEIT(self.upstream_config.copy(), data, np.zeros_like(data))
        started = time.monotonic()
        self.effective_batch_size = min(self.batch_size, len(data))
        self.steps_per_epoch = max(1, len(data) // self.effective_batch_size)
        if self.training_probabilities is None:
            self.history = self.model.train(valid=False, learning_rate=self.learning_rate,
                                           num_epoch=self.epochs, batch_size=self.effective_batch_size,
                                           L=1, verbose=False, checkpoint_dir=None,
                                           early_stopping_patience=self.epochs + 1)
        else:
            # The official high-level wrapper has no sample_weight argument.
            # Its own trainer accepts a dataset. Sampling changes its input
            # distribution while leaving the VAE loss/optimizer source intact.
            from scVAEIT.train import train as official_train
            batch, width = self.effective_batch_size, data.shape[1]
            samples = batch * self.steps_per_epoch
            def epoch_batches():
                epoch = len(self.sampling_indices)
                if epoch >= self.epochs:
                    raise RuntimeError("Weighted dataset iterated beyond the fixed epoch budget")
                indices = weighted_epoch_indices(self.training_probabilities, samples, self.seed, epoch)
                self.sampling_indices.append(indices)
                for start in range(0, samples, batch):
                    values = data[indices[start:start+batch]]
                    yield (values, np.empty((batch, 0), np.float32), np.zeros_like(values), np.zeros((batch, 1), np.int32))
            signature = (tf.TensorSpec((batch, width), tf.float32), tf.TensorSpec((batch, 0), tf.float32),
                         tf.TensorSpec((batch, width), tf.float32), tf.TensorSpec((batch, 1), tf.int32))
            dataset = tf.data.Dataset.from_generator(epoch_batches, output_signature=signature)
            options = tf.data.Options(); options.threading.private_threadpool_size = 1
            dataset = dataset.with_options(options).prefetch(1)
            self.model.dataset_train, self.model.dataset_valid = dataset, None
            # num_step_per_epoch does not truncate upstream iteration. This
            # dataset must be finite; an infinite repeat would never end.
            self.model.vae, self.history = official_train(dataset, None, self.model.vae, None,
                learning_rate=self.learning_rate, L=1, num_epoch=self.epochs,
                num_step_per_epoch=self.steps_per_epoch, save_every_epoch=self.epochs,
                es_patience=self.epochs+1, full_masks=True, verbose=False)
            if len(self.sampling_indices) != self.epochs:
                raise RuntimeError("Weighted training did not consume the fixed finite epoch budget")
        self.training_seconds = time.monotonic() - started
        self.training_rows = len(data)
        self.fitted = True
        return self

    def predict(self, x, anchor_values, anchor_indices):
        if not getattr(self, "fitted", False):
            raise ValueError("Fit the scVAEIT model before prediction")
        tf, _ = _runtime(self.threads)
        inputs, masks = self.transform.query(x, anchor_values, anchor_indices)
        # Reset the official sampling layer's generator. Repeated calls use the
        # same Monte Carlo noise, making the hidden-target invariance test exact.
        generator = self.model.vae.encoder.sampling.g
        generator_state = generator.state.numpy().copy()
        generator.reset_from_seed(self.seed + 917)
        data = tf.data.Dataset.from_tensor_slices((inputs, np.empty((len(inputs), 0), np.float32),
                                                   masks, np.arange(len(inputs), dtype=np.int32)))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        data = data.with_options(options).batch(self.batch_size)
        try:
            reconstruction = self.model.vae.get_recon(data, len(inputs), full_masks=True,
                                                       zero_out=True, return_mean=True,
                                                       L=self.mc_samples, training=False)
        finally:
            generator.reset(generator_state)
        predictions = reconstruction[:, len(self.transform.features):]
        predictions = predictions * self.transform.y_scale + self.transform.y_mean
        if predictions.shape != (len(inputs), self.transform.protein_width) or not np.isfinite(predictions).all():
            raise RuntimeError("scVAEIT returned invalid protein predictions")
        return predictions

    def predict_masked(self, x, full_panel, anchor_indices):
        """Audit convenience: select anchors before inspecting any ADT values."""
        anchors = _indices(anchor_indices, self.transform.protein_width)
        full_panel = np.asarray(full_panel)
        if full_panel.ndim != 2 or full_panel.shape[1] != self.transform.protein_width:
            raise ValueError("Full-panel array has the wrong shape")
        return self.predict(x, full_panel[:, anchors], anchors)

    def metadata(self):
        return {"method": "scVAEIT_Gaussian_adapter", "upstream": "https://github.com/JinHongDu-Lab/scVAEIT",
                "settings": asdict(self), "training_rows": self.training_rows,
                "training_seconds": self.training_seconds,
                "versions": {name: importlib.metadata.version(name) for name in ["scVAEIT", "tensorflow", "tensorflow-probability", "numpy"]},
                "upstream_source_sha256": _upstream_sources(),
                "preprocessing": "Already independently normalized RNA and log1p ADT; development-only HVGs and feature standardization",
                "query_input": "RNA and observed anchor values only; all other ADT entries zero with upstream missing mask -1",
                "query_training": False,
                "exact_published_preprocessing": False,
                "sample_weighting": "Equal weight per supplied row" if self.training_probabilities is None else
                    "Exact weighted development preprocessing; fresh finite-epoch replacement sampling with supplied row probabilities",
                "sampling_objective": "Legacy equal-row empirical distribution" if self.training_probabilities is None else
                    "Official VAE objective under the weighted empirical development distribution, approximated by seeded minibatch sampling; not exact weighted gradients",
                "sampling_steps_per_epoch": self.steps_per_epoch,
                "sampling_rows_per_epoch": self.steps_per_epoch*self.effective_batch_size,
                "training_weight_sha256": None if self.training_probabilities is None else
                    hashlib.sha256(self.training_probabilities.tobytes()).hexdigest(),
                "training_weight_provenance": "Caller-supplied development-only row weights; benchmark/CLI receipts record the complete input-file hash" if self.training_probabilities is not None else None,
                "hyperparameter_selection": "Fixed settings; any tuning must be performed by an outer development-only grouped protocol"}

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.transform.save(folder / "transform.npz")
        if self.training_probabilities is not None:
            indices = np.stack(self.sampling_indices)
            np.savez_compressed(folder / "training_sampling.npz", probabilities=self.training_probabilities,
                sampled_indices=indices, row_counts=np.bincount(indices.ravel(), minlength=self.training_rows),
                seed=self.seed, seed_stream=81037)
        self.model.save_model(str(folder / "official_checkpoint"))
        (folder / "metadata.json").write_text(json.dumps(self.metadata(), indent=2) + "\n")
        (folder / "training_history.json").write_text(json.dumps(_jsonable(self.history), indent=2) + "\n")
        config = {}
        for key, value in vars(self.model.config).items():
            if hasattr(value, "numpy"):
                value = value.numpy()
            config[key] = value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
        (folder / "upstream_config.json").write_text(json.dumps(config, indent=2) + "\n")


def _model_from_args(args):
    return ScVAEITRegressor(epochs=args.epochs, batch_size=args.batch_size,
                            n_hvg=args.n_hvg, threads=args.threads, seed=args.seed)


def _load_query(path):
    with np.load(path, allow_pickle=False) as query:
        if set(query.files) != {"x", "anchor_values", "anchor_indices"}:
            raise ValueError("Query NPZ must contain exactly x, anchor_values and anchor_indices; do not pass held-out ADT")
        return query["x"], query["anchor_values"], query["anchor_indices"]


def _load_training(path):
    with np.load(path, allow_pickle=False) as train:
        if set(train.files) not in ({"x", "y"}, {"x", "y", "weights"}):
            raise ValueError("Development file must contain exactly x/y or x/y/weights")
        weights = None if "weights" not in train.files else train["weights"]
        if weights is not None:
            _probabilities(weights, len(train["x"]))
        return train["x"], train["y"], weights


def validate_benchmark_receipt(output, *, check_inputs=True, check_current_adapter=True):
    """Validate a finished fold before reuse; no TensorFlow import is required."""
    output = Path(output).resolve()
    receipt_path = output / "receipt.json"
    if not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "completed" or receipt.get("mode") != "benchmark_fold":
        raise ValueError("Not a completed scVAEIT benchmark receipt")
    for name, expected in receipt["output_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"scVAEIT output integrity failure: {name}")
    for panel_id in receipt["panels"]:
        if f"{panel_id}.npz" not in receipt["output_sha256"]:
            raise ValueError("scVAEIT receipt omits a requested panel")
    if check_inputs:
        for path, expected in receipt["input_sha256"].items():
            if not Path(path).is_file() or _sha(path) != expected:
                raise ValueError(f"scVAEIT input integrity failure: {path}")
    if check_current_adapter and receipt["adapter_sha256"] != _sha(__file__):
        raise ValueError("scVAEIT adapter changed since this benchmark; use a new output directory")
    return True


def run_benchmark_fold(fold, output, model):
    """Fit once, predict every panel and form contrasts without opening outcomes."""
    fold, output = Path(fold), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    train_path = fold / "rna" / "development_groups.npz"
    panels_path = fold / "panels" / "panels.json"
    contrast_path = fold / "queries" / "contrasts.npz"
    panels = json.loads(panels_path.read_text())
    if not isinstance(panels, list) or not panels:
        raise ValueError("Benchmark fold needs a nonempty panel list")
    ids = [panel["id"] for panel in panels]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i or
                                       any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in i) for i in ids):
        raise ValueError("Panel IDs must be unique safe file-name components")
    with np.load(contrast_path, allow_pickle=False) as data:
        if set(data.files) != {"contrast"}:
            raise ValueError("Contrast file must contain exactly the contrast matrix")
        contrast = np.asarray(data["contrast"], dtype=np.float64)
        _matrix(contrast, "contrast")
    paths = [train_path, panels_path, contrast_path]
    queries = {}
    for panel_id in ids:
        query_path = fold / "queries" / f"{panel_id}_scvaeit.npz"
        query = _load_query(query_path)
        if contrast.shape[1] != len(query[0]):
            raise ValueError("Contrast columns do not match query group rows")
        queries[panel_id] = query_path
        del query
        paths.append(query_path)
    # Hash inputs before fitting and check them again before completion. No
    # outcome directory, query full-panel data or held-out metric is read.
    input_hashes = {str(path.resolve()): _sha(path) for path in paths}
    adapter_sha = _sha(__file__)
    tx, ty, weights = _load_training(train_path)
    if weights is None:
        model.fit(tx, ty)
    else:
        model.fit(tx, ty, weights=weights)
    output.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for panel_id in ids:
        group_prediction = model.predict(*_load_query(queries[panel_id]))
        prediction = contrast @ group_prediction
        path = output / f"{panel_id}.npz"
        np.savez_compressed(path, prediction=prediction)
        output_hashes[path.name] = _sha(path)
    model.save(output / "model")
    for path in sorted((output / "model").rglob("*")):
        if path.is_file():
            output_hashes[str(path.relative_to(output))] = _sha(path)
    for path, expected in input_hashes.items():
        if _sha(path) != expected:
            raise RuntimeError("Benchmark input changed during scVAEIT fitting/prediction")
    if _sha(__file__) != adapter_sha:
        raise RuntimeError("scVAEIT adapter changed during fitting/prediction")
    receipt = {**model.metadata(), "status": "completed", "mode": "benchmark_fold",
               "fit_unit": "development group means", "prediction_unit": "matched-control group contrasts",
               "panels": ids, "input_sha256": input_hashes, "output_sha256": output_hashes,
               "adapter_sha256": adapter_sha, "outcomes_read": False,
               "scientific_comparison_completed": False}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train", type=Path, help="NPZ containing development x/y and optional weights")
    source.add_argument("--benchmark-fold", type=Path, help="Prepared fold containing development groups, panels and anchor-only queries")
    parser.add_argument("--query", type=Path, help="NPZ containing x, anchor_values, anchor_indices only; required with --train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    if args.train is not None and args.query is None:
        parser.error("--query is required with --train")
    if args.benchmark_fold is not None and args.query is not None:
        parser.error("--query cannot be combined with --benchmark-fold")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty")
    if args.benchmark_fold is not None:
        run_benchmark_fold(args.benchmark_fold, args.output, _model_from_args(args))
        return
    tx, ty, weights = _load_training(args.train)
    model = _model_from_args(args).fit(tx, ty, weights=weights)
    prediction = model.predict(*_load_query(args.query))
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model")
    np.savez_compressed(args.output / "predictions.npz", prediction=prediction)
    receipt = {**model.metadata(), "status": "completed", "train_sha256": _sha(args.train),
               "query_sha256": _sha(args.query), "predictions_sha256": _sha(args.output / "predictions.npz"),
               "adapter_sha256": _sha(__file__)}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
