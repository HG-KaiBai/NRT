"""Prepare and submit the 24 task groups, followed by an automatic summary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from . import ROOT
from .run import tasks, main as run
from workflow import atomic_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/'full_co_recur_death')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--nrt-search-space',type=Path,
                   help='JSON overrides; resolved settings are frozen in the output directory')
    args = p.parse_args(argv)
    output = args.output.resolve()
    if (output/'submission.json').exists():
        raise SystemExit('Already submitted: inspect submission.json before submitting duplicate jobs')
    prepare_args=['prepare','--output',str(output)]
    if args.nrt_search_space is not None:
        prepare_args += ['--nrt-search-space',str(args.nrt_search_space.resolve())]
    run(prepare_args)
    logs=output/'logs'; logs.mkdir(parents=True,exist_ok=True)
    common=['sbatch','--parsable',f'--output={logs}/%x-%A_%a.out',f'--error={logs}/%x-%A_%a.err']
    submitted=[]
    for kind in ('cpu','gpu'):
        indices=[str(t['task_id']) for t in tasks() if (t['device']=='cpu') == (kind=='cpu')]
        command=common+[f'--job-name=realworld_{kind}',f'--array={",".join(indices)}']
        if kind=='gpu':
            command+=['--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1']
        command+=[str(ROOT/'job.sh')]
        env={**os.environ,'REALWORLD_OUTPUT':str(output),'DEVICE':'cpu' if kind=='cpu' else 'cuda:0',
             'PROFILE':'full','MODE':'run'}
        print(' '.join(command),flush=True)
        job=None if args.dry_run else subprocess.check_output(command,env=env,text=True).strip().split(';')[0]
        submitted.append(dict(kind=kind,job_id=job,task_ids=indices,command=command))
        if not args.dry_run:
            atomic_json(output/'submission.json',dict(status='submitting',jobs=submitted))
    if not args.dry_run:
        dependency='afterany:'+':'.join(j['job_id'] for j in submitted)
        command=common+['--job-name=realworld_summary','--time=00:15:00','--cpus-per-task=1','--mem=4G',
                        f'--dependency={dependency}',str(ROOT/'job.sh')]
        job=subprocess.check_output(command,env={**os.environ,'MODE':'summarize','DEVICE':'cpu',
            'PROFILE':'full','REALWORLD_OUTPUT':str(output)},text=True).strip().split(';')[0]
        submitted.append(dict(kind='summary',job_id=job,command=command))
        atomic_json(output/'submission.json',dict(status='submitted',jobs=submitted))
        print(json.dumps(submitted,indent=2))


if __name__=='__main__':
    main()
