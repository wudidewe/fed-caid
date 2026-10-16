"""FedCAID experiments with checked datasets and automatic package downloads.

The default reader uses dataset/<name>/ and rebuilds missing prepared data
from pinned Kaggle CSV files before training begins.
All shared numerical methods live in Utils.py. The four datasets remain
independent federations with their own features, labels and checkpoints.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

from _iov_layout import project_root

ROOT=project_root(__file__)
DATASETS=('CIC-IoV-2024','Car-Hacking','CIC-IDS-2017','CIC-UNSW-NB15')

# These are the previously validated parameters. Each dataset has an independent
# copy so a later deliberate experiment can change one profile without changing
# the others. No new tuning or adaptive learning-rate rule is introduced.
_REFERENCE_PARAMETERS={
    'clients':3, 'rounds':3, 'base_epochs':5, 'adapter_epochs':10,
    'adapter_lr':.1, 'batch_size':1024, 'rows_per_client':0, 'alpha':.8,
    'prior_strength':.5, 'adapter_prox':1., 'target_far':.01,
    'threshold_floor':.5, 'cal_fit_fraction':.5, 'clip':.1,
    'secure_scale':1000000, 'secure_timeout':20.,
    'detector_lr':.002, 'detector_weight_decay':.0001,
}
DATASET_PARAMETERS={name:dict(_REFERENCE_PARAMETERS) for name in DATASETS}


def non_numerical_option(key):
    return key.startswith('_') or key in {
        'dataset','data','download_manifest','offline','no_report','no_observe',
    }


def selected_datasets(args):
    selection=getattr(args,'dataset',None)
    data=getattr(args,'data',None)
    scope=getattr(args,'scope','all')
    allowed=DATASETS if scope=='all' else DATASETS[:2] if scope=='can' else DATASETS[2:]
    if data:
        if getattr(args,'smoke',False):
            raise ValueError('--data cannot be combined with synthetic --smoke')
        if selection in (None,'auto'):
            from dataset_access import identify_dataset
            selection=identify_dataset(data)
        elif selection=='all':
            raise ValueError('A single --data archive cannot select all datasets')
    if selection=='auto':
        raise ValueError('--dataset auto requires --data PATH_TO_DEVELOPMENT_NPZ')
    if selection in (None,'all'):
        return tuple(allowed)
    if selection not in allowed:
        raise ValueError('Dataset conflicts with --scope or is not supported')
    return (selection,)


def resolve_dataset_parameters(args,name):
    """Return a fresh namespace; CLI override > dataset profile > reference.

    Calls through the historical parser have no explicit-option set, so all
    their existing arguments retain precedence exactly as before.
    """
    if name not in DATASET_PARAMETERS:
        raise ValueError('Unknown dataset profile')
    result=argparse.Namespace(**vars(args))
    explicit=getattr(args,'_explicit_options',None)
    for key,value in DATASET_PARAMETERS[name].items():
        if explicit is not None and key not in explicit or not hasattr(result,key):
            setattr(result,key,value)
    for key in ('clients','rounds','base_epochs','adapter_epochs','batch_size'):
        if type(getattr(result,key)) is not int or getattr(result,key)<1:
            raise ValueError(f'{name}: {key} must be a positive integer')
    if result.mechanism=='mpc' and result.clients<3:
        raise ValueError('MPC requires at least three clients')
    for key in ('alpha','adapter_lr','adapter_prox','clip','secure_timeout','detector_lr'):
        if not math.isfinite(getattr(result,key)) or getattr(result,key)<=0:
            raise ValueError(f'{name}: {key} must be positive and finite')
    if not math.isfinite(result.detector_weight_decay) or result.detector_weight_decay<0:
        raise ValueError('Weight decay must be nonnegative and finite')
    if result.rows_per_client<0 or not 0<result.target_far<1 or not 0<result.cal_fit_fraction<1 or not 0<=result.threshold_floor<=1 or not 0<=result.prior_strength<=1:
        raise ValueError(f'{name}: invalid training or calibration profile')
    if type(result.secure_scale) is not int or result.secure_scale<1:
        raise ValueError('Secure fixed-point scale must be a positive integer')
    return result


def numerical_parameters(args,name):
    resolved=resolve_dataset_parameters(args,name)
    return {key:getattr(resolved,key) for key in _REFERENCE_PARAMETERS}


def load_local_for_dataset(args,name):
    """Compatible hook used by the runner; repository reading is the default."""
    from dataset_access import read_local_dataset, read_repository_dataset
    if getattr(args,'data',None):
        return read_local_dataset(Path(args.data),dataset=name)
    return read_repository_dataset(name,dataset_dir=args.data_root,
        download_manifest=getattr(args,'download_manifest',None),
        allow_download=not getattr(args,'offline',False))


def prepare_selected_datasets(args,names):
    """Resolve all missing packages before spending time on model fitting."""
    if args.smoke or args.data:
        return
    from dataset_access import check_repository_source, read_repository_dataset
    missing=[]
    for name in names:
        if not (Path(args.data_root)/name).exists():
            check_repository_source(name,args.data_root,args.download_manifest,
                                    allow_download=not args.offline)
            missing.append(name)
    for name in missing:
        print(f'DATASET DOWNLOAD: {name}',flush=True)
        loaded=read_repository_dataset(name,dataset_dir=args.data_root,
            download_manifest=args.download_manifest,allow_download=not args.offline)
        del loaded
        print(f'DATASET READY: {name}',flush=True)


def parser():
    from privacy_run import parser as experiment_parser
    p=experiment_parser()
    p.description=__doc__
    p.add_argument('--dataset',choices=('all','auto')+DATASETS,default=None,
                   help='One known dataset, all, or auto with --data; default keeps --scope')
    p.add_argument('--data',type=Path,help='One local prepared development.npz; no download is performed')
    p.add_argument('--download-manifest',type=Path,
                   help='Prepared package sources; default is <data-root>/download_sources.json')
    p.add_argument('--offline',action='store_true',help='Read existing datasets without network access')
    p.add_argument('--detector-lr',type=float,default=.002)
    p.add_argument('--detector-weight-decay',type=float,default=.0001)
    p.add_argument('--no-observe',action='store_true',help='Disable additional private round/resource observations')
    p.add_argument('--no-report',action='store_true',help='Train without automatic postprocessing')
    return p


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0]=='--privacy':
        arguments=arguments[1:]
    if arguments==['--check']:
        if not (ROOT/'tests/test_unified.py').is_file():
            print('Run --check from the source checkout containing tests/.')
            return 2
        environment=dict(os.environ)
        existing=environment.get('PYTHONPATH','')
        environment['PYTHONPATH']=str(ROOT/'src')+(os.pathsep+existing if existing else '')
        return subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,env=environment).returncode
    args=parser().parse_args(arguments)
    args._explicit_options={token[2:].split('=',1)[0].replace('-','_') for token in arguments if token.startswith('--')}
    try:
        chosen=selected_datasets(args)
        for name in chosen:
            resolve_dataset_parameters(args,name)
        prepare_selected_datasets(args,chosen)
        args._observe_rounds=not args.no_observe
        from privacy_run import execute
        code=execute(args)
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print(f'IOV STOP: {error}',flush=True)
        return 2
    if code==0 and not args.no_report and args.mechanism=='mpc' and args.out is not None:
        try:
            from experiment_report import generate_report
            report_dir=Path(args.out).parent/(Path(args.out).name+'_report')/'private_DO_NOT_UPLOAD'
            result=generate_report([args.out],report_dir)
            print(f'REPORT COMPLETE: {len(result["figures"])} figures; VAL development')
            print(f'Report: {report_dir.resolve()/"report.html"}')
        except (ImportError,ValueError,OSError) as error:
            print(f'Experiment completed; report unavailable: {error}',flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
