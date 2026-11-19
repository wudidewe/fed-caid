"""Score the published 16 February 2018 capture with the frozen FedCAID Hulk models."""

import csv
import json
from collections import Counter
from hashlib import file_digest
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd

from .data_io import apply_recipes
from .revision_external import CRITERIA, FIXED_CRITERIA
from .revision_external_preflight import preflight_hulk, run_verified_hulk


DEFAULT_CAPTURE = Path('dataset/CSE-CIC-IDS2018/02-16-2018.csv')
RECIPE = Path(__file__).with_name('fedcaid_resources') / 'preparation/CIC-IDS-2017/recipe.json'

SOURCE_SHA256 = '1a4919faa0c49c7af97230b0c2d076eba23ee6dd81103a3801d51ac316355d8b'
SOURCE_URL = 'https://registry.opendata.aws/cse-cic-ids2018/'
CAPTURE_ID = 'cse-cic-ids2018-2018-02-16'
SEEDS = [42, 43, 44, 45, 46]
LABELS = {
    'Benign': 'BENIGN',
    'DoS attacks-Hulk': 'DoS Hulk',
    'DoS attacks-SlowHTTPTest': 'DoS Slowhttptest',
}
EXPECTED_COUNTS = {'BENIGN': 446_772, 'DoS Hulk': 461_912, 'DoS Slowhttptest': 139_890}
ALIASES = {
    'Destination Port': 'Dst Port',
    'Total Fwd Packets': 'Tot Fwd Pkts',
    'Total Backward Packets': 'Tot Bwd Pkts',
    'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    'Flow Bytes/s': 'Flow Byts/s',
    'Flow Packets/s': 'Flow Pkts/s',
    'Fwd IAT Total': 'Fwd IAT Tot',
    'Bwd IAT Total': 'Bwd IAT Tot',
    'Fwd Header Length': 'Fwd Header Len',
    'Bwd Header Length': 'Bwd Header Len',
    'Fwd Packets/s': 'Fwd Pkts/s',
    'Bwd Packets/s': 'Bwd Pkts/s',
    'Min Packet Length': 'Pkt Len Min',
    'Max Packet Length': 'Pkt Len Max',
    'Packet Length Mean': 'Pkt Len Mean',
    'Packet Length Std': 'Pkt Len Std',
    'Packet Length Variance': 'Pkt Len Var',
    'Average Packet Size': 'Pkt Size Avg',
    'Avg Fwd Segment Size': 'Fwd Seg Size Avg',
    'Avg Bwd Segment Size': 'Bwd Seg Size Avg',
    'Subflow Fwd Packets': 'Subflow Fwd Pkts',
    'Subflow Bwd Packets': 'Subflow Bwd Pkts',
    'Subflow Fwd Bytes': 'Subflow Fwd Byts',
    'Subflow Bwd Bytes': 'Subflow Bwd Byts',
    'Init_Win_bytes_forward': 'Init Fwd Win Byts',
    'Init_Win_bytes_backward': 'Init Bwd Win Byts',
    'act_data_pkt_fwd': 'Fwd Act Data Pkts',
    'min_seg_size_forward': 'Fwd Seg Size Min',
    **{f'{side} Packet Length {metric}': f'{side} Pkt Len {metric}'
       for side in ('Fwd', 'Bwd') for metric in ('Max', 'Min', 'Mean', 'Std')},
    **{f'{flag} Flag Count': f'{flag} Flag Cnt' for flag in ('FIN', 'SYN', 'PSH', 'ACK', 'URG')},
}


def sha256_file(path):
    """Hash a file without loading it into memory."""
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def prepare_capture(source, destination):
    """Align all valid rows to the original 2017 features and recipe."""
    recipe = json.loads(RECIPE.read_text(encoding='utf-8'))
    selected = recipe['pipeline']['selected_original_features']
    derived = recipe['pipeline']['engineered_features']
    features = [*selected, *(step['name'] for step in derived)]
    columns = [ALIASES.get(name, name) for name in selected]
    header = pd.read_csv(source, nrows=0).columns
    if features != recipe['feature_names'] or not {*columns, 'Label'} <= set(header):
        raise ValueError('The source columns are incompatible with the frozen feature recipe.')

    counts = Counter()
    staging = destination.with_suffix('.partial')
    with staging.open('x', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerow([*features, 'label', 'group_id'])
        for frame in pd.read_csv(source, usecols=[*columns, 'Label'], chunksize=65_536, low_memory=False):
            frame = frame.loc[frame['Label'] != 'Label']
            labels = frame['Label'].map(LABELS)
            if labels.isna().any():
                raise ValueError('The published capture contains an unexpected label.')
            raw = frame[columns].apply(pd.to_numeric, errors='raise').to_numpy(np.float32, copy=True)
            raw[~np.isfinite(raw)] = np.nan
            raw[raw == 0] = 0.0
            values = np.column_stack((raw, apply_recipes(raw, selected, derived)))
            prepared = pd.DataFrame(values, columns=features)
            prepared['label'] = labels.to_numpy()
            prepared['group_id'] = CAPTURE_ID
            prepared.to_csv(handle, header=False, index=False, float_format='%.9g', lineterminator='\n')
            counts.update(labels)

    if counts != EXPECTED_COUNTS or sha256_file(source) != SOURCE_SHA256:
        raise ValueError('The source capture identity or class counts changed during preparation.')
    staging.rename(destination)
    return [*features, 'label', 'group_id']


def write_manifest(path, source, prepared):
    """Declare one capture and retain the original confirmation thresholds."""
    manifest = {
        'version': 1,
        'dataset': 'CIC-IDS-2017',
        'target_class': 'DoS Hulk',
        'attestation': {
            'custodian': 'Communications Security Establishment and Canadian Institute for Cybersecurity public dataset',
            'unseen_in_development': True,
            'basis': 'Official 2018 capture identity checked against the frozen 2017 development sources.',
        },
        'capture_dataset': 'CSE-CIC-IDS2018',
        'capture_source': SOURCE_URL,
        'evaluation_scope': 'Exploratory evaluation on one published capture. Session identifiers are unavailable.',
        'exact_overlap_policy': 'reject',
        'captures': [{
            'capture_id': CAPTURE_ID,
            'prepared_csv': prepared.name,
            'sha256': sha256_file(prepared),
            'raw_files': [{'path': str(source), 'sha256': SOURCE_SHA256}],
        }],
        'criteria': {**CRITERIA, **FIXED_CRITERIA},
    }
    with path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')


def write_summary(run, result):
    """Report observed performance without treating seeds as independent captures."""
    tables = run / 'score/result'
    with (tables / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['capture_id'] == 'ALL_CAPTURES']
    methods = {
        'base_threshold': 'Frozen TabM',
        'head_local': 'Local subtype head',
        'head_donor_single': 'Single donor',
        'head_donor': 'Donor pool',
        'head_federated': 'Federated subtype head',
        'base_reference_1pct': 'Frozen TabM at 1% CAL reference',
        'guarded_federated_reference_1pct': 'Federated head at 1% CAL reference',
    }
    lines = [
        '# Exploratory Hulk evaluation on CSE-CIC-IDS2018',
        '',
        'One published capture on 16 February 2018, scored with five frozen CIC-IDS-2017 models.',
        'The capture is one conservative group because the CSV has no verified session identifiers.',
        'Means describe five seeds on this capture. There is no interval across acquisitions.',
        'All three source classes were retained. Frozen weights and alarm settings were used.',
        '',
        f'Confirmation assessment: {result["status"]}. See the original criteria in the manifest.',
        '',
        '| Method | Hulk precision (%) | Hulk recall (%) | Hulk F1 (%) | Macro F1 (%) | FAR (%) | FNR (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for mode in dict.fromkeys(row['mode'] for row in rows):
        selected = [row for row in rows if row['mode'] == mode]
        values = [f'{100 * fmean(float(row[key]) for row in selected):.3f}'
                  for key in ('target_precision', 'target_recall', 'target_f1', 'macro_f1', 'far', 'fnr')]
        lines.append('| ' + ' | '.join([methods.get(mode, mode), *values]) + ' |')
    lines.extend(('', '[Capture support](score/result/capture_support.csv) and '
                  '[paired seed comparisons](score/result/paired_intervals.csv).', '',
                  '[Frozen alarm diagnosis and CAL threshold replay](diagnostics/SUMMARY.md).', ''))
    (run / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def validate_hulk_2018_source(source_csv):
    """Check the published capture against the expected digest before any scoring."""
    source = Path(source_csv).resolve(strict=True)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError('Select the original 02-16-2018 CSV with the recorded SHA256.')
    return source


def run_hulk_2018(output, source_csv, frozen_source, project):
    """Score one capture and diagnose its frozen and source CAL operating points."""
    from .revision_hulk_2018_diagnostics import run_hulk_2018_diagnostics

    source = validate_hulk_2018_source(source_csv)
    frozen_source = Path(frozen_source).resolve(strict=True)
    project = Path(project).resolve(strict=True)
    run = Path(output).absolute()
    if not frozen_source.is_dir() or not project.is_dir():
        raise ValueError('The frozen source and project must be existing directories.')
    if (
        run.resolve() != run
        or run.is_relative_to(frozen_source)
        or frozen_source.is_relative_to(run)
        or source.is_relative_to(run)
    ):
        raise ValueError('The 2018 output must be disjoint from the frozen source and capture.')
    run.mkdir(parents=True, exist_ok=False)
    prepared, manifest = run / 'capture.csv', run / 'manifest.json'
    print(f'Preparing the published 2018 capture in {run}', flush=True)
    columns = prepare_capture(source, prepared)
    write_manifest(manifest, source, prepared)
    checked = preflight_hulk(frozen_source, manifest)
    if checked['status'] != 'ready':
        raise ValueError(checked['reason'])
    if checked['expected_prepared_columns'] != columns or checked['seeds'] != SEEDS:
        raise ValueError('The capture schema or seed inventory differs from the frozen 2017 run.')

    print('Scoring one 2018 capture with five frozen models.', flush=True)
    result = run_verified_hulk(run / 'score', frozen_source, manifest)
    if result['status'] != 'completed':
        raise RuntimeError(result.get('reason', result['status']))
    assessment = json.loads((run / 'score/result/complete.json').read_text(encoding='utf-8'))
    diagnostics = run_hulk_2018_diagnostics(run / 'diagnostics', run, frozen_source, project)
    write_summary(run, assessment)
    receipt = {
        'status': 'completed', 'scope': 'exploratory_single_public_capture',
        'source': str(source), 'source_sha256': SOURCE_SHA256,
        'frozen_source': str(frozen_source), 'seeds': SEEDS,
        'assessment': assessment['status'], 'diagnostics': diagnostics,
        'summary_path': str(run / 'SUMMARY.md'),
        'receipt_path': str(run / 'complete.json'),
        'artifacts_sha256': {
            name: sha256_file(run / name) for name in (
                'manifest.json', 'score/receipt.json', 'score/result/complete.json', 'diagnostics/complete.json',
                'SUMMARY.md', 'diagnostics/SUMMARY.md',
            )
        },
    }
    with (run / 'complete.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    return receipt
