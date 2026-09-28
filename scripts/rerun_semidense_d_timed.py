"""Replace only D's registered test, monitor GPU state, and refresh the C/D report."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_semidense_cd_test import main as summarize


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def refresh():
    summarize()
    data=json.loads((ROOT/'outputs/semidense_tier3_cd_test_comparison/comparison.json').read_text())
    save(ROOT/'artifacts/semidense_cd_timed_test_results.json',data)
    return data


def document(data,status):
    lines=['# C/D 第三档计时测试结果','',f'状态：{status}。D按用户要求在原目录重测覆盖，C保持原结果。GPU1（与C此前使用的物理卡相同），权重、数据和计时代码不变。','',
        '同一独立合成测试集，共2000对，有同母图精确真值；不是原始跨时相配准测试。CUDA event区间包含CPU提交间隙，QRRU包含投影与输出整理；读图、预热、GT评估、绘图和写盘不计入。','',
        f"进度：C {data['C']['pairs']}/2000；D {data['D']['pairs']}/2000。未完成时不作最终比较。",'',
        'GPU状态和同卡进程每30秒采样至 outputs/semidense_tier3_lr_d_test2000.gpu.jsonl，不能据此保证两次采样间完全没有其他负载。','']
    if data['complete']:
        lines+=['| 模块平均耗时(ms/对) | C | D |','|---|---:|---:|']
        for key,label in data['C']['registration']['module_labels'].items():
            lines.append(f"| {label} | {data['C']['timing']['module_ms'][key]['mean']:.3f} | {data['D']['timing']['module_ms'][key]['mean']:.3f} |")
        for key in ['mean','p50','p95','p99']:
            lines.append(f"| 总耗时 {key} | {data['C']['timing']['wall_total_ms'][key]:.3f} | {data['D']['timing']['wall_total_ms'][key]:.3f} |")
        if data.get('timing_warning'):lines+=['',data['timing_warning']]
    lines+=['','逐对记录在原测试目录；完整汇总为 artifacts/semidense_cd_timed_test_results.json。启动、覆盖范围与完成校验见 artifacts/semidense_cd_timed_test_registration.json。']
    (ROOT/'docs/semidense_cd_timed_test_results.md').write_text('\n'.join(lines)+'\n')


def main():
    os.chdir(ROOT)
    registry=ROOT/'artifacts/semidense_cd_timed_test_registration.json';reg=json.loads(registry.read_text())
    if reg.get('d_rerun'):raise RuntimeError('D rerun already registered; inspect its state before another launch')
    d=next(j for j in reg['jobs'] if j['name']=='d');folder=Path(d['output']);log=Path(d['log'])
    old=json.loads((folder/'report.json').read_text());assert old['pairs']==2000
    assert sha(Path(d['checkpoint']))==d['checkpoint_sha256'] and sha(Path(reg['manifest']))==reg['manifest_sha256']
    croot=ROOT/'outputs/semidense_tier3_lr_c_test2000'
    cfiles={str(p):sha(p) for p in croot.rglob('*') if p.is_file()}
    used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>256:raise RuntimeError(f'GPU1 is busy: {used} MiB')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                if target==str(log) or target.startswith(str(folder)+'/'):raise RuntimeError(f'Test still open by {proc.name}')
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    telemetry=folder.with_suffix('.gpu.jsonl')
    reg['d_rerun']=dict(created_unix=time.time(),gpu=1,reason='User requested D-only overwrite after abnormal previous latency',
        old_report_sha256=sha(folder/'report.json'),C_files_sha256=cfiles,telemetry=str(telemetry),
        evaluator_sha256=sha(ROOT/'scripts/evaluate_semidense_timed.py'),profiler_sha256=sha(ROOT/'scripts/semidense_timing.py'))
    reg['status']='D_rerun_starting';reg['timing_warning']=None
    d.update(gpu=1,state='starting');d.pop('report_sha256',None);d.pop('pairs',None)
    save(registry,reg)
    shutil.rmtree(folder);log.unlink(missing_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('w') as f:
        process=subprocess.Popen(d['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    d.update(pid=process.pid,state='running');reg['status']='D_rerun_running';save(registry,reg)
    print(f'D rerun PID={process.pid}, GPU1, replacing {folder}',flush=True)
    data=refresh();document(data,'D重测进行中')
    with telemetry.open('w') as stream:
        while True:
            sample={'unix':time.time(),'eval_pid':process.pid}
            for key,command in [('gpu',['nvidia-smi','-i','1','--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader']),
                ('processes',['nvidia-smi','-i','1','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])]:
                try:sample[key]=subprocess.check_output(command,text=True,timeout=10).strip()
                except Exception as e:sample[key+'_error']=str(e)
            stream.write(json.dumps(sample)+'\n');stream.flush()
            code=process.poll()
            data=refresh()
            if code is not None:break
            time.sleep(30)
    c_unchanged=all(Path(p).exists() and sha(Path(p))==digest for p,digest in cfiles.items())
    reg['d_rerun'].update(exit_code=code,finished_unix=time.time(),C_unchanged=c_unchanged)
    if code==0 and data['complete'] and c_unchanged:
        report=json.loads((folder/'report.json').read_text())
        assert report['protocol']=='semidense_qrru_timed_v1' and report['timing']['max_accounting_error_ms']<.1
        d.update(state='completed',pairs=2000,report_sha256=sha(folder/'report.json'))
        reg['status']='completed';reg['timing_warning']=data.get('timing_warning')
        document(data,'D重测完成，已覆盖此前结果')
    else:
        d['state']='failed';reg['status']='D_rerun_failed';document(data,'D重测失败，查看日志；当前记录未完成')
    save(registry,reg);print(reg['status'],flush=True)
    if reg['status']!='completed':raise RuntimeError('D rerun did not complete successfully')


if __name__=='__main__':main()
