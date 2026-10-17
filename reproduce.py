"""Run an isolated, small current-NRT reproduction check from any directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke'])
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='An unused directory; production outputs are never overwritten')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    space = output / 'search-space.json'
    space.write_text(json.dumps(dict(
        search=dict(depth_min=2, depth_max=2, widths=[8], shared_layers_max=1,
                    private_layers_max=1, strength_max=1.),
        baseline=dict(depth=2, shared_width=8, private_width=8, private_layers=1,
                      regularization_strength=1.)), indent=2) + '\n')
    commands = [
        ['regularization.tuning.main', '--cases', 'po_interval_q0.2',
         '--output-dir', str(output / 'tuning'), '--n-samples', '128',
         '--device', 'cpu', '--threads', '1', '--trials', '2', '--initial-trials', '2',
         '--folds', '2', '--epochs', '3', '--warmup-epochs', '1', '--validation-every', '1',
         '--batch-size', '64', '--grid-size', '8', '--no-cuda-graphs', '--search-space', str(space)],
        ['regularization.testing.simulation', 'run', '--cases', 'po_interval_q0.2',
         '--hyper-dir', str(output / 'tuning/hyper_para'), '--output-dir', str(output / 'testing'),
         '--n-samples', '128', '--replications', '2', '--device', 'cpu', '--threads', '1',
         '--grid-size', '8'],
    ]
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for command in commands:
        subprocess.run([sys.executable, '-m', *command], cwd=ROOT, check=True)
    results = [json.loads(p.read_text()) for p in (output / 'testing').glob('*/*/replication_*.json')]
    if len(results) != 6 or any(r['status'] != 'ok' for r in results):
        raise RuntimeError('Expected six successful simulation fits')
    table = output / 'testing/tables/po_interval_q0.2.csv'
    if not table.is_file():
        raise RuntimeError('Simulation summary was not generated')
    print(f'Completed all three NRT methods, two replications each. Summary: {table}')


if __name__ == '__main__':
    main()
