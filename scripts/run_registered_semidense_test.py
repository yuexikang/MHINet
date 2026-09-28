"""Run one registered evaluation and sample its physical GPU until completion."""
from pathlib import Path
import argparse,json,os,subprocess,time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True);args=parser.parse_args()
    path=args.registration.resolve();r=json.loads(path.read_text());root=Path(__file__).resolve().parents[1]
    if r['state']!='planned':raise ValueError('Evaluation already launched')
    output=Path(r['output']);log=Path(r['log']);gpu=str(r['gpu'])
    if output.exists() or log.exists():raise FileExistsError(output)
    used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>256:raise RuntimeError('Selected GPU is busy')
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(path)
    with log.open('w') as stream:
        proc=subprocess.Popen(r['command'],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    r.update(state='running',pid=proc.pid,started_unix=time.time());save();print(r,flush=True)
    with Path(r['telemetry']).open('w') as stream:
        while True:
            row=dict(unix=time.time(),pid=proc.pid)
            for key,query in [('gpu','--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem'),('processes','--query-compute-apps=gpu_uuid,pid,process_name,used_memory')]:
                try:row[key]=subprocess.check_output(['nvidia-smi','-i',gpu,query,'--format=csv,noheader'],text=True,timeout=10).strip()
                except Exception as e:row[key+'_error']=str(e)
            stream.write(json.dumps(row)+'\n');stream.flush()
            code=proc.poll()
            if code is not None:break
            time.sleep(30)
    r.update(exit_code=code,finished_unix=time.time(),state='failed')
    if code==0 and (output/'report.json').exists():
        report=json.loads((output/'report.json').read_text())
        if report['pairs']==r['expected_pairs'] and report['manifest_sha256']==r['manifest_sha256'] and report['checkpoint_sha256']==r['checkpoint_sha256']:
            r.update(state='completed',pairs=report['pairs'])
    save();print(r['state'],flush=True)
    if r['state']!='completed':raise RuntimeError('Evaluation incomplete')


if __name__=='__main__':main()
