"""Outcome-aware technical diagnostics: these are never prediction inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import write_json
from .experiment import _read, _ridge_prediction
from .benchmark_data import balanced_weights


def technical_audit(run, protocol):
    records = []
    for fold in sorted(Path(run).glob('fold_*')):
        train = _read(fold / 'rna/train.npz')
        test = _read(fold / 'outcomes/test.npz')
        obs = pd.read_csv(fold / 'rna/train.csv')
        if train['technical'].shape[1] == 0:
            continue
        residual = train['y'] - train['baseline']
        predicted = _ridge_prediction(train['technical'], residual, test['technical'], balanced_weights(obs), 1.0)
        intercept = np.average(residual, axis=0, weights=balanced_weights(obs))
        measured = test['y'] - test['baseline']
        for j, protein in enumerate(train['proteins']):
            mse = np.mean((measured[:, j] - predicted[:, j])**2)
            null = np.mean(measured[:, j]**2)
            intercept_mse = np.mean((measured[:, j]-intercept[j])**2)
            records.append({'fold': fold.name, 'protein': str(protein), 'n_response_groups': len(measured),
                            'prediction_error_reduction_vs_zero': 1-mse/null if null > 0 else np.nan,
                            'prediction_error_reduction_vs_training_intercept': 1-mse/intercept_mse if intercept_mse > 0 else np.nan,
                            'technical_prediction_rms': float(np.sqrt(np.mean(predicted[:, j]**2))),
                            'residual_rms': float(np.sqrt(null))})
    out = Path(run) / 'evaluation'
    pd.DataFrame(records, columns=['fold','protein','n_response_groups','prediction_error_reduction_vs_zero',
        'prediction_error_reduction_vs_training_intercept',
        'technical_prediction_rms','residual_rms']).to_csv(out / 'technical_diagnostics.csv', index=False)
    write_json(out / 'technical_diagnostics.json', {
        'diagnostic_only': True, 'additional_controls_used_for_prediction': False,
        'method': 'Train-only ridge of RNA-unrecovered responses on isotype response contrasts; score on outer-heldout groups.',
        'interpretation': 'Only improvement over the training-intercept baseline indicates predictive information beyond average residual bias. Neither metric is a causal adjustment or biological discovery.',
        'limitations': ['Does not remove every ADT depth or staining effect', 'RNA-model error remains in the residual',
                       'A low technical association is not evidence of a post-transcriptional mechanism',
                       'Shared control groups induce dependent contrasts; no cell-level significance tests are reported'],
        'biological_validation_complete': False, 'protein_fold_records_scored': len(records)})
