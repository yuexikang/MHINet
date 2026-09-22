"""Two-step real-resource replay audit; explicitly select a free CUDA device."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch


def compare(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(f'{path}: tensor differs')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(f'{path}: keys differ')
        for k in a:compare(a[k],b[k],f'{path}.{k}')
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError(path)
        for k,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}.{k}')
    elif a!=b:raise AssertionError(f'{path}: {a!r} != {b!r}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    common=[sys.executable,'-m','mhinet.downstream.train','--config',args.config,
        '--checkpoint',args.checkpoint,'--limit-train','8','--limit-val','1']
    full=args.output/'full';split=args.output/'split'
    for folder,stop,resume in ((full,2,False),(split,1,False),(split,2,True)):
        command=common+['--output',str(folder),'--max-steps',str(stop)]
        if resume:command+=['--resume',str(folder/'latest.pt')]
        with (args.output/f'{folder.name}_{stop}.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(f'completed {folder.name} step {stop}',flush=True)
    a=torch.load(full/'latest.pt',weights_only=True,map_location='cpu',mmap=True)
    b=torch.load(split/'latest.pt',weights_only=True,map_location='cpu',mmap=True)
    for key in ('model','optimizer','scheduler','rng','progress'):
        if key not in a:raise KeyError(f'Checkpoint missing {key}')
        compare(a[key],b[key],key)
    report=dict(status='passed',comparison='bitwise equal model/optimizer/scheduler/RNG/progress',
        full=str(full/'latest.pt'),resumed=str(split/'latest.pt'))
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
