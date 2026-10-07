"""Fixed synthetic perturbations of the actual v0.4 abundance-response model.

The v0.3 scenario design is retained, with four prespecified extra anchors for
the v0.4 primary panel. No scenario outcome selects model parameters. These
checks do not establish biological benefit or uncertainty calibration.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from .abundance_response import AbundanceResponseShrinkage
from .two_penalty import TwoPenaltyResponse
from .response_stress import SCENARIOS
from .artifacts import complete_stage, freeze_implementation, write_json


PROTEINS = ('CD38', 'ICOS', 'PD-1', 'HLA-DR', 'CD127', 'CD27', 'CD28', 'CD45RO', 'CD25', 'CD69')
ANCHORS = np.arange(8)
TARGETS = np.array([8, 9])
MODEL_KEYS = ('prediction', 'model_variance', 'base_response', 'rna_contribution',
              'anchor_contributions', 'anchor_operator', 'anchor_residuals', 'rna_prediction')


def stress_protocol(seed=20261005):
    return {'schema_version': 1, 'version': '0.4.0', 'seed': int(seed),
        'training_donors': 24, 'query_donors': 48, 'rna_genes': 24,
        'cells_per_state_pool': 128, 'proteins': list(PROTEINS),
        'anchors': ANCHORS.tolist(), 'targets': TARGETS.tolist(),
        'scenarios': list(SCENARIOS),
        'abundance_model': {'alpha': 1., 'shrinkage': .5, 'n_hvg': 24},
        'two_penalty_model': {'alpha_rna': 1., 'alpha_anchor': 1., 'n_hvg': 24},
        'training': 'One fixed fit on reference training donors; no scenario tuning or refitting',
        'weighting': 'Equal donors; equal control and stimulated condition weights within each donor',
        'oracle': 'Mean log1p noiseless cell intensity in fixed state pools, weighted by actual sampled state proportions',
        'estimand': 'Stimulated minus control mean cellwise log1p protein abundance',
        'composition': 'Changes both observable inputs and target mixture estimand; not a fixed-truth noise perturbation',
        'uncertainty': 'Conditional observed-response covariance; no calibrated coverage or generalization guarantee',
        'interpretation': 'Controlled synthetic software/mechanism experiment; not independent biological evidence',
        'selection_on_results': False}


def _pool(seed, donors, prefix):
    rng = np.random.default_rng(seed)
    genes, proteins, per_state = 24, 10, 128
    states = np.repeat([0, 1], per_state)
    z = rng.normal(size=(donors, 3))
    rna_load = np.column_stack((np.linspace(-.4, .4, genes), np.cos(np.arange(genes))*.3))
    # First four anchor/last two target loadings retain the previous design.
    # Additional rows are frozen simulation coefficients, not biological claims.
    protein_load = np.array([[.4, .1, .5], [.1, .5, -.3], [-.3, .2, .4], [.2, -.3, .5],
        [-.2, .4, .2], [.3, -.2, -.4], [.2, .2, .3], [-.1, -.3, .4], [.4, .3, .6], [-.2, .4, -.5]])
    protein_state = np.array([.9, -.5, .7, -.4, .3, -.6, .5, -.2, 1.1, -.7])
    rna_state = np.sin(np.arange(genes)/2)*.8
    rna_base = 3.2+rng.normal(scale=.15, size=(donors, 1, genes))
    adt_base = 3.5+rng.normal(scale=.15, size=(donors, 1, proteins))
    rx, ry = [], []
    for arm in (0, 1):
        rx.append(np.exp(rna_base+states[None, :, None]*rna_state
            +arm*(.15+z[:, :2]@rna_load.T)[:, None, :]
            +rng.normal(scale=.2, size=(donors, len(states), genes))))
        ry.append(np.exp(adt_base+states[None, :, None]*protein_state
            +arm*(.35+z@protein_load.T)[:, None, :]
            +rng.normal(scale=.2, size=(donors, len(states), proteins))))
    rna_rates, adt_rates = np.stack(rx, axis=1), np.stack(ry, axis=1)
    return {'donors': np.array([f'{prefix}_{d:03d}' for d in range(donors)]),
        'states': states, 'rna_rates': rna_rates, 'adt_rates': adt_rates,
        'rna_counts': rng.poisson(rna_rates), 'adt_counts': rng.poisson(adt_rates)}


def _reference(pool, composition):
    latent = np.log1p(pool['adt_rates'])
    means = [(1-p)*latent[:, arm, pool['states']==0].mean(1)
             +p*latent[:, arm, pool['states']==1].mean(1)
             for arm, p in enumerate(composition)]
    return means[1]-means[0]


def observe_pool(pool, scenario, seed=20261005):
    """Paired arm means plus a separate oracle; no full-panel normalization."""
    rng = np.random.default_rng(seed)
    rna, adt = pool['rna_counts'], pool['adt_counts']
    if 'depth' in scenario:
        rna = rng.binomial(rna, scenario['depth'])
        adt = rng.binomial(adt, scenario['depth'])
    if 'stim_adt_depth' in scenario:
        adt = adt.copy()
        adt[:, 1] = rng.binomial(adt[:, 1], scenario['stim_adt_depth'])
    if 'adt_gamma_shape' in scenario:
        shape = scenario['adt_gamma_shape']
        adt = rng.poisson(pool['adt_rates']*rng.gamma(shape, 1/shape, adt.shape))
    raw_cells = scenario['cells']
    if isinstance(raw_cells, bool) or int(raw_cells) != raw_cells or raw_cells < 1:
        raise ValueError('cells must be a positive integer')
    cells = int(raw_cells)
    composition = scenario.get('composition', (.5, .5))
    if len(composition) != 2 or any(not np.isfinite(p) or not 0 <= p <= 1 for p in composition):
        raise ValueError('composition must give two state fractions in [0, 1]')
    positions = [np.flatnonzero(pool['states']==s) for s in (0, 1)]
    select_rng = np.random.default_rng(seed+7919)
    abundance_x, abundance_y, selections = [], [], []
    for d in range(len(rna)):
        xx, yy, indices = [], [], []
        for arm, p in enumerate(composition):
            n1 = round(cells*p)
            counts = [cells-n1, n1]
            if any(n > len(pos) for n, pos in zip(counts, positions)):
                raise ValueError('Requested cells exceed fixed reference state pool')
            idx = np.concatenate([select_rng.permutation(pos)[:n] for n, pos in zip(counts, positions)])
            indices.append(idx)
            xx.append(np.log1p(rna[d, arm, idx]).mean(0))
            yy.append(np.log1p(adt[d, arm, idx]).mean(0))
        abundance_x.append(xx); abundance_y.append(yy); selections.append(indices)
    ax, ay = np.asarray(abundance_x), np.asarray(abundance_y)
    actual = tuple(round(cells*p)/cells for p in composition)
    truth = _reference(pool, actual)
    truth[:, TARGETS] += scenario.get('hidden_shift', 0.)
    measured_y = ay[:, 1]-ay[:, 0]
    return {'x': ax[:, 1]-ax[:, 0], 'anchor_values': measured_y[:, ANCHORS],
        'measured_y': measured_y, 'truth': truth, 'cell_indices': np.asarray(selections),
        'composition': actual, 'abundance_x': ax.reshape(-1, ax.shape[-1]),
        'abundance_y': ay.reshape(-1, ay.shape[-1]),
        'abundance_donors': np.repeat(pool['donors'], 2),
        'abundance_weights': np.full(2*len(ax), 1/(2*len(ax)))}


def compute_stress(seed=20261005, guard=None):
    """Fit fixed candidates once; report every donor in every fixed scenario."""
    protocol = stress_protocol(seed)
    if guard: guard()
    tp = _pool(seed+1, protocol['training_donors'], 'train')
    qp = _pool(seed+2, protocol['query_donors'], 'query')
    train = observe_pool(tp, SCENARIOS[0], seed+3)
    kwargs = {k: train[k] for k in ('abundance_x', 'abundance_y', 'abundance_donors', 'abundance_weights')}
    model = AbundanceResponseShrinkage().fit(train['x'], train['measured_y'], tp['donors'],
        **kwargs, **protocol['abundance_model'], guard=guard)
    matched = AbundanceResponseShrinkage().fit(train['x'], train['measured_y'], tp['donors'],
        **kwargs, **protocol['abundance_model'], formulation='matched-response', guard=guard)
    baseline = TwoPenaltyResponse(**protocol['two_penalty_model']).fit(train['x'], train['measured_y'],
        train['anchor_values'], donors=tp['donors'])
    scale = model.train_scales[TARGETS]
    rows, errors, arrays, predictions = [], [], {}, {}
    for scenario in SCENARIOS:
        if guard: guard()
        name = scenario['name']
        observed = observe_pool(qp, scenario, seed+4)
        x, a = observed['x'], observed['anchor_values']
        result = model.predict(x, a, ANCHORS)
        no_anchor = model.predict(x, np.empty((len(x), 0)), np.array([], dtype=int))
        direct = matched.predict(x, a, ANCHORS)
        methods = {'abundance_response_shrinkage': result['prediction'],
            'two_penalty_ridge': baseline.predict(x, a), 'matched_response': direct['prediction'],
            'no_anchor_correction': no_anchor['prediction'],
            'mean_response_template': np.broadcast_to(model.cohort_mean, result['prediction'].shape)}
        for method, pred in methods.items():
            delta = (pred[:, TARGETS]-observed['truth'][:, TARGETS])/scale
            mae = np.abs(delta).mean(1)
            rows.append({'scenario': name, 'method': method, 'n_donors': len(x),
                'cells_per_arm': scenario['cells'], 'standardized_mae': float(mae.mean()),
                'standardized_rmse': float(np.sqrt(np.mean(delta**2))),
                'prediction_available_fraction': float(np.isfinite(pred[:, TARGETS]).mean()),
                'model_sd_mean': float(np.sqrt(result['model_variance'][TARGETS]).mean())
                    if method=='abundance_response_shrinkage' else None})
            errors.extend({'scenario': name, 'method': method, 'donor': str(d),
                'standardized_mae': float(e)} for d, e in zip(qp['donors'], mae))
            arrays[f'{name}__{method}'] = pred
        for key in ('x', 'anchor_values', 'truth', 'cell_indices'):
            arrays[f'{name}__{key}'] = observed[key]
        for key in MODEL_KEYS:
            arrays[f'{name}__{key}'] = result[key]
        predictions[name] = result
    ref, hidden = predictions['reference'], predictions['hidden_only_response_plus1']
    hidden_checks = {key: bool(np.array_equal(ref[key], hidden[key])) for key in MODEL_KEYS}
    checks = {'training_query_donors_disjoint': not bool(set(tp['donors']) & set(qp['donors'])),
        'hidden_only_outputs_identical': hidden_checks,
        'hidden_only_all_outputs_identical': all(hidden_checks.values()),
        'hidden_only_baseline_identical': bool(np.array_equal(arrays['reference__two_penalty_ridge'],
                                                arrays['hidden_only_response_plus1__two_penalty_ridge'])),
        'decomposition_max_abs_error': float(max(np.max(np.abs(v['prediction']-
            (v['base_response']+v['rna_contribution']+v['anchor_contributions'].sum(-1)))) for v in predictions.values())),
        'composition_changes_oracle_mean_abs': float(np.mean(np.abs(arrays['composition_20_to_80pct__truth'][:, TARGETS]
                                                                  -arrays['reference__truth'][:, TARGETS]))),
        'parameter_selection_on_scenarios': False,
        'model_variance_updates_with_query_noise': False,
        'uncertainty_limitation': 'The fixed conditional covariance does not adapt to unobserved query noise or hidden-only shifts'}
    arrays.update({f'train__{k}': train[k] for k in ('x', 'measured_y', 'abundance_x',
        'abundance_y', 'abundance_donors', 'abundance_weights')})
    arrays.update(training_donors=tp['donors'], query_donors=qp['donors'], training_target_scale=scale)
    return {'protocol': protocol, 'summary': rows, 'donor_errors': errors, 'arrays': arrays,
        'checks': checks, 'model': model, 'baseline': baseline, 'matched_model': matched}


def run_stress(output, seed=20261005, guard=None):
    """Write a new immutable result; reject existing output directories."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    output = Path(output)
    if guard: guard()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', stress_protocol(seed), immutable=True)
    freeze_implementation(output)
    result = compute_stress(seed, guard=guard)
    result['model'].save(output/'abundance_response_shrinkage.npz')
    result['baseline'].save(output/'two_penalty_ridge.npz')
    result['matched_model'].save(output/'matched_response.npz')
    np.savez_compressed(output/'synthetic_arrays.npz', **result['arrays'])
    summary = pd.DataFrame(result['summary'])
    summary.to_csv(output/'summary.csv', index=False)
    pd.DataFrame(result['donor_errors']).to_csv(output/'donor_errors.csv', index=False)
    write_json(output/'checks.json', result['checks'], immutable=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    names = [s['name'] for s in SCENARIOS]
    methods = list(dict.fromkeys(summary.method))
    positions = np.arange(len(names))
    for j, method in enumerate(methods):
        vals = summary[summary.method==method].set_index('scenario').loc[names, 'standardized_mae']
        ax.barh(positions+(j-2)*.15, vals, height=.14, label=method.replace('_', ' '))
    ax.set_yticks(positions, [n.replace('_', ' ') for n in names])
    ax.invert_yaxis(); ax.set_xlabel('Mean standardized absolute error; equal synthetic donors')
    ax.set_title('ResponseBridge 0.4: fixed synthetic stress scenarios')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(output/'stress_comparison.png', dpi=160)
    fig.savefig(output/'stress_comparison.pdf'); plt.close(fig)
    lines = ['# ResponseBridge 0.4: fixed synthetic stress tests', '',
        'These are controlled synthetic experiments, not independent biological evidence.', '',
        'All candidates are fitted once on 24 training donors with prespecified parameters. Every condition retains all 48 disjoint query donors. The baseline penalties are fixed, not tuned on these simulations; this is not a benchmark claim against a fully tuned baseline.', '',
        '| Scenario | Method | Standardized MAE |', '|---|---|---:|']
    lines.extend(f"| {r['scenario']} | {r['method']} | {r['standardized_mae']:.4f} |" for r in result['summary'])
    lines.extend(['', '![Fixed stress comparison](stress_comparison.png)', '',
        'Cell reductions use nested samples without replacement. Depth and overdispersion perturb observed inputs while preserving the noiseless oracle. Composition changes the target mean-response estimand, so its oracle follows the sampled state proportions.', '',
        'The hidden-only shift leaves predictions, variances and every explanatory component exactly unchanged. This demonstrates lack of information about the added hidden response, not successful detection.', '',
        'Conditional covariance is fixed after training and does not automatically widen under lower query cell counts or depth. It is not a calibrated interval or a guarantee under measurement shifts. No perturbation is required to worsen or improve every method.', '',
        'All scenarios, inputs, oracle targets, paired abundance training groups, predictions and contributions are preserved. No result selects model parameters or supports a biological superiority claim.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    if guard: guard()
    complete_stage(output, {'kind': 'controlled_synthetic_response_stress_v040',
        'seed': int(seed), 'scenarios': len(SCENARIOS), 'biological_validation': False})
    return {'output': str(output.resolve()), 'status': 'complete_synthetic_only',
        'scenarios': len(SCENARIOS), 'checks': result['checks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    import json
    print(json.dumps(run_stress(args.output, args.seed), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
