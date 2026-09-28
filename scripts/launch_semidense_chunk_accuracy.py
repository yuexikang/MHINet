"""Schedule two full-val shards on free GPUs without competing with running tests."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    registry=ROOT/'artifacts/semidense_chunk_accuracy_registration.json'
    output=ROOT/'outputs/semidense_chunk_accuracy_val3848'
    if registry.exists() or output.exists():raise FileExistsError('Chunk accuracy run already exists')
    jobs=[]
    for shard,gpu in [(0,0),(1,1)]:
        folder=output/f'shard_{shard}'
        command=[sys.executable,'-u','scripts/evaluate_semidense_chunk_accuracy.py','--checkpoint',str(ROOT/'outputs/semidense_stable_v2_tier3_lr_c_seed0/latest.pt'),'--manifest','/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2/val/pairs.jsonl','--output',str(folder),'--shard',str(shard)]
        jobs.append(dict(shard=shard,gpu=gpu,command=command,log=str(ROOT/f'outputs/semidense_chunk_accuracy_shard{shard}.console.log'),state='waiting_for_free_gpu'))
    registration=dict(created_unix=time.time(),baseline_commit='be9025c',baseline_tag='baseline/pre-chunk-accuracy-20260928',chunks=[32,128,256,512,1024],total_pairs=3848,status='running',jobs=jobs,
        comparison='same C tier3 weights, full tier3 val, changes only inference window_chunk')
    def save():
        tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(registration,indent=2)+'\n');tmp.replace(registry)
    save()
    with open(ROOT/'outputs/semidense_chunk_accuracy_summary.console.log','w') as f:
        summary=subprocess.Popen([sys.executable,'-u','scripts/summarize_semidense_chunk_accuracy.py','--watch'],cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    registration['summary_pid']=summary.pid;save();processes={}
    while True:
        used={int(l.split(',')[0]):int(l.split(',')[1]) for l in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
        for job in jobs:
            shard=job['shard']
            if job['state']=='waiting_for_free_gpu' and used[job['gpu']]<256:
                with open(job['log'],'w') as f:
                    process=subprocess.Popen(job['command'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                processes[shard]=process;job.update(state='running',pid=process.pid,started_unix=time.time());save();print(json.dumps(job),flush=True)
            if job['state']=='running':
                code=processes[shard].poll()
                if code is not None:
                    done=output/f'shard_{shard}'/'completed.json'
                    job.update(state='completed' if code==0 and done.exists() else 'failed',exit_code=code,finished_unix=time.time());save();print(json.dumps(job),flush=True)
        if all(j['state'] in ('completed','failed') for j in jobs):
            registration['status']='completed' if all(j['state']=='completed' for j in jobs) else 'failed';save()
            if registration['status']=='failed':
                summary.terminate();raise RuntimeError('One or more validation shards failed; inspect logs')
            break
        time.sleep(30)


if __name__=='__main__':main()
