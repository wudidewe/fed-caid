"""JSON-only round and resource diagnostics from one completed experiment.

Missing historical observations remain unavailable and are never reconstructed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiment_report_data import ReportDataError, _metrics


ROUND_METRICS = ('weighted_f1',)
ROUND_TIMERS = ('adapter_fit_seconds', 'aggregation_seconds',
                'cached_personalization_seconds', 'cached_val_decision_seconds')


def _finite(value, label, *, nullable=False, maximum=None):
    if value is None and nullable:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or
            (maximum is not None and value > maximum)):
        raise ReportDataError(f'{label}: expected a finite nonnegative measurement')
    return float(value)


def _measurement(value):
    """Keep the report schema for a single recorded value, without statistics."""
    return {'mean': value, 'sample_sd': None,
            'n': int(value is not None), 'values': [value]}


def _resource(raw, label):
    if raw is None:
        return {'cpu_percent_mean': None, 'ram_peak_mib': None,
                'sample_count': 0, 'scope': 'Resource sampling was not recorded.'}
    if not isinstance(raw, dict):
        raise ReportDataError(label + ': expected an object')
    count = raw.get('sample_count', 0)
    if type(count) is not int or count < 0:
        raise ReportDataError(label + ': invalid resource sample count')
    scope = raw.get('scope')
    if not isinstance(scope, str) or not scope:
        raise ReportDataError(label + ': resource scope must be explicit')
    cpu = _finite(raw.get('cpu_percent_mean'), label + ':CPU', nullable=True)
    ram = _finite(raw.get('ram_peak_mib'), label + ':RAM', nullable=True)
    if count == 0 and (cpu is not None or ram is not None):
        raise ReportDataError(label + ': measurements without resource samples')
    return {'cpu_percent_mean': cpu, 'ram_peak_mib': ram,
            'sample_count': count, 'scope': scope}


def _read(path):
    payload = path.read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReportDataError(f'{path.name}: duplicate JSON key {key}')
            result[key] = value
        return result
    def constant(value):
        raise ReportDataError(f'{path.name}: nonfinite JSON value {value}')
    try:
        raw = json.loads(payload.decode('utf-8-sig'), object_pairs_hook=pairs,
                         parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReportDataError(f'{path.name}: malformed observations') from error
    if not isinstance(raw, dict):
        raise ReportDataError(f'{path.name}: expected an object')
    return raw, hashlib.sha256(payload).hexdigest()


def load_observations(paths, data):
    """Validate optional JSON observations from exactly one completed run."""
    paths = list(paths)
    if len(paths) != 1 or data['run_count'] != 1 or len(data['runs']) != 1:
        raise ReportDataError('Round observations require exactly one completed run')
    root = Path(paths[0]).resolve()
    run = data['runs'][0]
    if root.name != run['run_id']:
        raise ReportDataError('Round observations do not match the completed run directory')
    summary, provenance = {}, []
    for dataset in data['datasets']:
        relative = f'private_DO_NOT_UPLOAD/{dataset}/round_observability_private.json'
        path = root / relative
        if not path.is_file():
            summary[dataset] = {'available': False,
                                'reason': 'Round/resource observations were not recorded in this run.'}
            continue
        raw, digest = _read(path)
        expected = {'schema': 'round_observability_v1', 'dataset': dataset,
                    'training_seed': run['training_seed'], 'partition_seed': run['partition_seed'],
                    'evaluation': 'VAL development',
                    'class_names': run['datasets'][dataset]['class_names'],
                    'rows': run['datasets'][dataset]['rows']}
        for key, value in expected.items():
            if raw.get(key) != value or (key.endswith('_seed') and type(raw.get(key)) is not int):
                raise ReportDataError(f'{dataset}: round observation {key} does not match the completed run')
        rounds = raw.get('rounds')
        if not isinstance(rounds, list) or not rounds:
            raise ReportDataError(dataset + ': no measured round observations')
        indices = [row.get('round') if isinstance(row, dict) else None for row in rounds]
        completed_rounds = run['datasets'][dataset]['aggregation']['completed_rounds']
        expected_indices = list(range(1, completed_rounds + 1))
        if indices not in (expected_indices, [0] + expected_indices) or any(type(i) is not int for i in indices):
            raise ReportDataError(dataset + ': round indices must contain every configured round, optionally preceded by baseline zero')
        clean = []
        for row in rounds:
            metrics = row.get('metrics')
            if not isinstance(metrics, dict):
                raise ReportDataError(dataset + ': missing round detection metrics')
            if 'class_names' in metrics and metrics['class_names'] != expected['class_names']:
                raise ReportDataError(dataset + ': inconsistent round class order')
            if 'rows' in metrics and metrics['rows'] != expected['rows']:
                raise ReportDataError(dataset + ': inconsistent round VAL support')
            validated = _metrics(metrics, dataset + ':round metrics')
            if validated['support'] != run['datasets'][dataset]['support']:
                raise ReportDataError(dataset + ': round VAL support differs from the completed run')
            values = {key: _finite(validated['metrics'].get(key), dataset + ':' + key, maximum=1)
                      for key in ROUND_METRICS}
            clean.append({'round': row['round'], 'metrics': values,
                          **{key: _finite(row.get(key), dataset + ':' + key, nullable=True)
                             for key in ROUND_TIMERS},
                          'resources': _resource(row.get('resources'), dataset + ':round resource')})
        resource = _resource(raw.get('resources_summary'), dataset + ':workflow resource')
        for key in ROUND_METRICS:
            if not math.isclose(clean[-1]['metrics'][key], run['datasets'][dataset]['metrics'][key],
                                rel_tol=1e-9, abs_tol=1e-10):
                raise ReportDataError(dataset + ': final round metrics disagree with the completed result')
        grouped = []
        for row in clean:
            grouped.append({'round': row['round'],
                            'metrics': {key: _measurement(row['metrics'][key]) for key in ROUND_METRICS},
                            'timing': {key: _measurement(row[key]) for key in ROUND_TIMERS},
                            'resources': {key: _measurement(row['resources'][key])
                                          for key in ('cpu_percent_mean', 'ram_peak_mib')},
                            'resource_scope': row['resources']['scope']})
        summary[dataset] = {'available': True, 'rounds': grouped,
                            'records': [{'seed': run['training_seed'], 'rounds': clean,
                                         'resources_summary': resource}],
                            'resources_summary': {key: _measurement(resource[key])
                                                  for key in ('cpu_percent_mean', 'ram_peak_mib')},
                            'resource_scope': resource['scope']}
        provenance.append({'run_id': run['run_id'], 'file': relative, 'sha256': digest})
    return {'datasets': summary, 'provenance': provenance}
