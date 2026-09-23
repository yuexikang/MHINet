"""Launch the registered, user-authorized two-GPU tier1 learning-rate comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    source=ROOT/'outputs/shared_stable_v2_full_tier3_seed0/shared_descriptor.pt'
    payload=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    metadata=payload['metadata']
    if payload['format']!='mhinet.shared-descriptor.v1' or metadata['config']['tier']!=3:
        raise ValueError('Require completed tier3 shared bundle')
    if metadata['step']!=metadata['total_steps'] or metadata['step']!=8663:
        raise ValueError('Tier3 has not completed the registered budget')
    if metadata['validation']['step']!=8663:raise ValueError('Missing final tier3 validation')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(line.split(',')[0]):int(line.split(',')[1]) for line in memory.splitlines()}
    jobs=[]
    for name,gpu in (('a',1),('b',2)):
        if used[gpu]>256:raise RuntimeError(f'GPU {gpu} no longer free: {used[gpu]} MiB')
        output=ROOT/f'outputs/semidense_stable_v2_tier1_lr_{name}_seed0'
        log=output.with_suffix('.console.log')
        if output.exists() or log.exists():raise FileExistsError(output)
        config=ROOT/f'configs/semidense_tier1_lr_{name}.json'
        cfg=json.loads(config.read_text())
        runtime=json.loads(Path(cfg['runtime']).read_text())
        jobs.append(dict(name=name,gpu=gpu,output=str(output),log=str(log),config=str(config),
            shared_lr=cfg['shared_lr'],head_lr=cfg['head_lr'],data_root=runtime['data_root'],
            command=[sys.executable,'-u','-m','mhinet.downstream.train','--config',str(config),
                     '--checkpoint',str(source),'--output',str(output)]))
    data=Path(jobs[0]['data_root'])
    registration=dict(created_unix=time.time(),source=str(source),source_sha256=sha(source),source_step=8663,
        source_validation=str(source.parent/'validation/step_008663/summary.json'),
        tier=1,epochs=1,effective_batch=4,seed=0,train_pairs=51978,val_pairs=5772,
        train_sha256=sha(data/'train/pairs.jsonl'),val_sha256=sha(data/'val/pairs.jsonl'),
        freeze=['DINOv3','MVT','GHIM/H0 head'],train=['VGG','CGMDP decoder','temperatures','QRRU'],
        loss='Lc + Lf + Lq',comparison='Learning rates only; same seed, checkpoint, order and val cohort',jobs=jobs)
    target=ROOT/'artifacts/semidense_tier1_lr_compare_registration.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(registration,indent=2)+'\n')
    for job in jobs:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with open(job['log'],'w') as stream:
            process=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        job['pid']=process.pid;job['state']='launched'
        target.write_text(json.dumps(registration,indent=2)+'\n')
        print(json.dumps(job),flush=True)


if __name__=='__main__':main()
