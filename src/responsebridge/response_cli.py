"""Coordinator for the response-native v0.3 protocol; old stages stay available."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import signal
import time

from .artifacts import (ResourceGuard, exclusive_run, freeze_implementation,
                        freeze_protocol, require_receipt, write_json)

PROJECT = Path(__file__).resolve().parents[2]


def _cell_comparator(args, protocol, guard):
    from .scvaeit_cells import validate_cell_receipt
    worker = PROJECT / '.venv-scvaeit/bin/python'
    if not worker.is_file():
        raise RuntimeError('Missing cell-level scVAEIT runtime; see docs/scvaeit_cells.md')
    prepared = getattr(args, 'prepared', None)
    if prepared is None:
        raise ValueError('--prepared is required for cell-level scVAEIT')
    prepared = prepared.resolve()
    write_json(args.run / 'cell_input_lock.json', {'prepared': str(prepared)}, immutable=True)
    pending = list(sorted(args.run.glob('fold_*')))
    active = {}
    concurrency = int(protocol['scvaeit'].get('parallel_folds', 1))
    if not 1 <= concurrency <= 2:
        raise ValueError('At most two independent cell comparator folds per laptop coordinator')
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise SystemExit('Comparator coordinator received SIGTERM; preserving partial work')
    signal.signal(signal.SIGTERM, terminate)
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                fold = pending.pop(0)
                require_receipt(fold / 'development')
                destination = fold / 'scvaeit_cells'
                if validate_cell_receipt(destination):
                    continue
                cmd = [str(worker), '-m', 'responsebridge.scvaeit_cells',
                       '--prepared', str(prepared), '--output', str(destination),
                       '--manifest', str(fold / 'roles.json'),
                       '--epoch-grid', *map(str, protocol['scvaeit']['epoch_grid']),
                       '--seeds', *map(str, protocol['scvaeit']['seeds']),
                       '--threads', str(protocol['resources']['threads']),
                       '--pause-file', str(args.run / 'PAUSE')]
                env = dict(os.environ, PYTHONPATH=str(PROJECT / 'src'))
                for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                    env[key] = str(protocol['resources']['threads'])
                stream = (fold / 'scvaeit_cell_worker.log').open('a')
                try:
                    process = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    stream.close(); raise
                active[process.pid] = (process, fold, stream)
                print(json.dumps({'cell_fold': fold.name, 'status': 'running', 'pid': process.pid}), flush=True)
            pids = set(active)
            # Nested coordinators explicitly report the TensorFlow subprocesses
            # they monitor when sandbox restrictions prevent process-tree scans.
            for process, fold, _ in active.values():
                resource = fold / 'scvaeit_cells/resource_usage.json'
                if resource.exists():
                    snapshot = json.loads(resource.read_text())
                    if snapshot.get('pid') == process.pid:
                        pids.update(snapshot.get('monitored_pids', []))
            guard.extra_pids = tuple(pids)
            guard.check()
            if (args.run / 'PAUSE').exists():
                raise RuntimeError('Run PAUSE marker requested stop')
            for pid, (process, fold, stream) in list(active.items()):
                if process.poll() is None:
                    continue
                if process.returncode:
                    raise RuntimeError(f'Cell comparator failed ({process.returncode}); inspect {fold / "scvaeit_cell_worker.log"}')
                if not validate_cell_receipt(fold / 'scvaeit_cells'):
                    raise RuntimeError('Cell comparator exited without a verified complete receipt')
                stream.close(); del active[pid]
                print(json.dumps({'cell_fold': fold.name, 'status': 'complete'}), flush=True)
            if active:
                time.sleep(2)
    finally:
        for process, _, stream in active.values():
            # A leader may have exited while its TensorFlow child is alive.
            # Always clean the owned session, then kill any remaining members.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            stream.close()
        guard.extra_pids = ()
        signal.signal(signal.SIGTERM, previous_sigterm)


def run_response(args):
    from threadpoolctl import threadpool_limits
    from . import response_experiment as exp
    args.run = args.run.resolve()
    protocol = freeze_protocol(args.protocol, args.run)
    if protocol.get('model') != 'response-shrinkage' or protocol.get('version') != '0.3.0':
        raise ValueError('response-shrinkage requires its v0.3.0 protocol')
    if getattr(args, 'axis', None) not in (None, 'donor'):
        raise ValueError('The response-native protocol leaves donors out; other split axes remain legacy analyses')
    if getattr(args, 'lineage', None) not in (None, protocol['cohort_lineage']):
        raise ValueError('Lineage override conflicts with the frozen protocol')
    if args.command == 'benchmark' and not args.skip_scvaeit and args.prepared is None:
        raise ValueError('--prepared is required; --skip-scvaeit explicitly requests an incomplete diagnostic run')
    with exclusive_run(PROJECT / 'runs/compute_budget'), exclusive_run(args.run), threadpool_limits(limits=protocol['resources']['threads']):
        freeze_implementation(args.run)
        guard = ResourceGuard(PROJECT / 'runs/compute_budget', **{k: v for k, v in protocol['resources'].items() if k != 'threads'})
        commands = (['fit-rna', 'fit-response', 'predict-response', 'explain',
                     'scvaeit-benchmark', 'calibrate', 'evaluate', 'report'] if args.command == 'benchmark' else [args.command])
        if args.command == 'benchmark' and args.skip_scvaeit:
            commands.remove('scvaeit-benchmark')
        def check():
            guard.check()
            if (args.run / 'PAUSE').exists():
                raise RuntimeError('Run PAUSE marker requested stop')
        for command in commands:
            check()
            print(json.dumps({'stage': command, 'status': 'running', 'model': 'response-shrinkage', 'run': str(args.run)}), flush=True)
            if command == 'fit-rna' or (command == 'fit-response' and getattr(args, 'groups', None) is not None):
                if getattr(args, 'groups', None) is None:
                    raise ValueError('--groups is required to initialize a response-native run')
                exp.initialize(args.groups, args.run, protocol,
                               donor_split=getattr(args, 'donor_split', None), limit=getattr(args, 'limit_folds', None))
            if command == 'fit-response':
                exp.fit_stage(args.run, protocol, guard=check)
            elif command in ('select-panel', 'make-query'):
                # Fixed panels and privileged masking are frozen together with
                # donor roles; these stages verify their immutable artifacts.
                require_receipt(args.run / 'cohort')
                for fold in sorted(args.run.glob('fold_*')):
                    for role in ('test', 'calibration'):
                        require_receipt(fold / 'queries' / role)
            elif command == 'predict-response':
                exp.predict_stage(args.run, protocol, guard=check)
            elif command == 'calibrate':
                exp.calibrate_stage(args.run, protocol)
            elif command == 'explain':
                from .response_diagnostics import training_sensitivity
                training_sensitivity(args.run, protocol, guard=check)
                exp.explain_stage(args.run, protocol)
            elif command == 'scvaeit-benchmark':
                _cell_comparator(args, protocol, guard)
            elif command == 'evaluate':
                exp.evaluate_stage(args.run, protocol)
            elif command == 'report':
                print(exp.report_stage(args.run, protocol), flush=True)
            check()
            print(json.dumps({'stage': command, 'status': 'complete'}), flush=True)
