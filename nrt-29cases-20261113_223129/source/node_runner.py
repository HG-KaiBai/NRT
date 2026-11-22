"""Check an allocated CPU/GPU node, then run an NRT entry point.

Launcher-only checks leave scientific source fingerprints and cached fits intact.
Exit 75 asks runtime.sh to requeue the current Slurm job/array element.
"""
import argparse
import importlib
from pathlib import Path
import re
import sys
import tempfile
import traceback

import cuda_retry


MODULES = {
    'tuning.main', 'testing.simulation', 'tuning.prepare_cases',
    'regularization.tuning.main', 'regularization.testing.simulation', 'realworld.run',
}


def options(module, arguments):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--device')
    parser.add_argument('--methods', default='all')
    parser.add_argument('--method')
    args, _ = parser.parse_known_args(arguments)
    if args.device is not None and not re.fullmatch(r'(cpu|cuda)(:[0-9]+)?', args.device):
        parser.error('--device must be cpu or cuda[:INDEX]')
    args.cpu_only = (module == 'tuning.prepare_cases' or '--summarize' in arguments
                     or arguments[:1] in (['prepare'], ['summarize']))
    return args


def check_temporary_storage():
    with tempfile.TemporaryDirectory(prefix='nrt-node-check-') as directory:
        path = Path(directory) / 'probe'
        payload = b'NRT node temporary-storage probe\n'
        path.write_bytes(payload)
        if path.read_bytes() != payload:
            raise OSError('Temporary-storage read/write mismatch')


def precheck(module, args):
    import numpy as np
    import pandas  # noqa: F401 - every entry point uses the pinned runtime
    import scipy  # noqa: F401
    import torch
    from workflow import check_device, node_preflight

    check_temporary_storage()
    # GPU tasks also need a working CPU and BLAS runtime.
    check_device('cpu')
    matrix = np.arange(16., dtype=np.float64).reshape(4, 4)
    product = matrix @ matrix.T
    expected = np.array([[sum(matrix[i, k] * matrix[j, k] for k in range(4))
                          for j in range(4)] for i in range(4)])
    np.testing.assert_array_equal(product, expected)
    if args.cpu_only:
        return
    device = args.device
    if device is None:
        device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if module == 'realworld.run' else 'cuda:0'
    if torch.device(device).type == 'cuda':
        check_device(device)

    if module == 'tuning.main':
        # The driver validates unsupported/missing method names as configuration errors.
        methods = [args.method] if args.method in ('NNCDE', 'DeepSurv') else []
        node_preflight('cpu', methods, 'tuning')
    elif module == 'testing.simulation':
        from workflow import METHODS
        methods = list(METHODS) if args.methods == 'all' else args.methods.split(',')
        node_preflight('cpu', [m for m in methods if m in METHODS], 'testing')
    elif module == 'regularization.tuning.main':
        importlib.import_module('optuna')
    elif module == 'realworld.run':
        # These are required by realworld.run.environment(), including CPU jobs.
        for name in ('optuna', 'lifelines', 'sksurv'):
            importlib.import_module(name)
        methods = {'Ctree', 'ICcforest'} if args.methods == 'all' else set(args.methods.split(','))
        node_preflight('cpu', [name.lower() for name in ('Ctree', 'ICcforest') if name in methods], 'testing')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in MODULES:
        raise SystemExit('Usage: python -m node_runner NRT_MODULE [ARGS ...]')
    module, *arguments = argv
    args = options(module, arguments)
    if not any(flag in arguments for flag in ('--help', '-h', '--dry-run', '--list-cases')):
        try:
            precheck(module, args)
        except Exception as error:
            traceback.print_exc()
            print('CPU/GPU node precheck failed; exiting 75 for Slurm requeue.', file=sys.stderr, flush=True)
            raise SystemExit(cuda_retry.RETRY_EXIT_CODE) from error
    cuda_retry.main([module, *arguments])


if __name__ == '__main__':
    main()
