"""Run the registered 200-step 32/128 comparison and paired validation evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mhinet.config import sha256_file

OUT=ROOT/'outputs/semidense_chunk32_128_short200_20260928'
SOURCE=ROOT/'outputs/semidense_stable_v2_tier1_lr_c_seed0/latest.pt'
MANIFEST=Path('/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2/val/pairs.jsonl')
REGISTRY=ROOT/'artifacts/semidense_chunk32_128_short200_registration.json'


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def worker(chunk):
    started=time.time();folder=OUT/f'chunk{chunk}'
    train_command=[sys.executable,'-u',str(ROOT/'scripts/train_semidense_chunk_experiment.py'),
        '--config',str(ROOT/f'configs/semidense_train_chunk{chunk}_short200.json'),
        '--checkpoint',str(SOURCE),'--output',str(folder),'--max-steps','200','--limit-val','128','--device','cuda:0']
    subprocess.run(train_command,cwd=ROOT,check=True)
    trained=time.time()
    eval_command=[sys.executable,'-u',str(ROOT/'scripts/evaluate_semidense_timed.py'),
        '--checkpoint',str(folder/'latest.pt'),'--manifest',str(MANIFEST),'--tier','3',
        '--limit','128','--inference-window-chunk','1024','--output',str(folder/'cascaded_val128'),
        '--device','cuda:0']
    subprocess.run(eval_command,cwd=ROOT,check=True)
    write(folder/'completed.json',dict(status='completed',steps=200,training_pairs=800,
        validation_pairs=128,train_command=train_command,evaluation_command=eval_command,
        training_and_validation_wall_seconds=trained-started,evaluation_wall_seconds=time.time()-trained))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=(32,128));args=parser.parse_args()
    os.chdir(ROOT)
    if args.worker is not None:return worker(args.worker)
    if OUT.exists() or REGISTRY.exists():raise FileExistsError('Experiment already registered')
    used={int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    if any(used[gpu]>256 for gpu in (0,1)):raise RuntimeError('GPU0/1 must be free')
    configs=[json.loads((ROOT/f'configs/semidense_train_chunk{c}_short200.json').read_text()) for c in (32,128)]
    configs[1]['matcher']['window_chunk']=32
    if configs[0]!=configs[1]:raise ValueError('Configs differ beyond training window_chunk')
    OUT.mkdir(parents=True)
    registration=dict(status='running',started_unix=time.time(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source=str(SOURCE),source_sha256=sha256_file(SOURCE),val_manifest_sha256=sha256_file(MANIFEST),
        steps=200,accumulation=4,seed=0,shared_lr=8e-5,head_lr=8e-4,
        schedule='Original full tier3 cosine horizon; diagnostic stop at 200; fresh AdamW',
        scope='Single-seed short training; first128 tier3 validation pairs; not full-set convergence evidence',
        inference_window_chunk=1024,runner_sha256=sha256_file(Path(__file__)),jobs=[])
    write(REGISTRY,registration)
    processes=[]
    for chunk,gpu in ((32,0),(128,1)):
        log=OUT/f'chunk{chunk}.console.log'
        with log.open('w') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(chunk)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process)
        registration['jobs'].append(dict(chunk=chunk,gpu=gpu,pid=process.pid,state='running',log=str(log),
            config_sha256=sha256_file(ROOT/f'configs/semidense_train_chunk{chunk}_short200.json')))
        write(REGISTRY,registration)
    while any(process.poll() is None for process in processes):
        time.sleep(10)
        for process,job in zip(processes,registration['jobs']):
            code=process.poll()
            if code is not None:job.update(state='completed' if code==0 else 'failed',exit_code=code)
        write(REGISTRY,registration)
    registration.update(status='completed' if all(p.returncode==0 for p in processes) else 'failed',finished_unix=time.time())
    write(REGISTRY,registration)
    if registration['status']=='failed':raise RuntimeError('Worker failed; inspect registered logs')


if __name__=='__main__':main()
