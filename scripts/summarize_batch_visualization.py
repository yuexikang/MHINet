"""Archive measured batch probes and completed engineering visualization smoke."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    output = ROOT/'outputs/engineering_visualization_seven_smoke'
    run = json.loads((output/'run.json').read_text())
    assert run['status'] == 'completed' and run['optimizer_steps'] == 2
    manifest = json.loads((output/'visualizations/step_0000002/pair_0000/manifest.json').read_text())
    assert [r['iteration'] for r in manifest['images']] == list(range(7))
    pictures = [{'path': r['path'], 'sha256': digest(Path(r['path']))} for r in manifest['images']]
    probes = [json.loads((ROOT/f'artifacts/batch_probe_heads_bs{b}.json').read_text()) for b in (1,2,4)]
    summary = run['last_validation']['summary']
    runtime = json.loads((ROOT/'configs/runtime_paths.server.json').read_text())
    resources = {k: {'path':runtime[k], 'sha256':digest(Path(runtime[k]))}
                 for k in ('dino_checkpoint','selected_stage1_checkpoint','pyramid_checkpoint')}
    report = {
        'status':'engineering_smoke_passed_not_model_validation', 'formal_training_started':False,
        'recommended_real_batch':1, 'gradient_accumulation':4,
        'larger_batch_alignment_passed':False,
        'batch_isolation':json.loads((ROOT/'artifacts/batch_forward_isolation.json').read_text()),
        'precision_diagnosis':json.loads((ROOT/'outputs/batch_precision_diagnosis_fp32_position.log').read_text()),
        'batch_probes':probes, 'resources':resources,
        'checkpoint':run['last_checkpoint'], 'resume':run['resume'],
        'visualization_pair':manifest['pair_id'], 'pictures':pictures,
        'validation_summary':summary,
        'data_root':runtime['data_root'], 'runtime_config':str(ROOT/'configs/runtime_paths.server.json'),
        'environment':{'python':'/root/miniconda3/envs/loma-repro/bin/python',
                       'torch':probes[0]['torch'], 'gpu':probes[0]['device']},
        'source_files':{str(p.relative_to(ROOT)):digest(p) for p in
                        [ROOT/'mhinet/engine/train.py',ROOT/'mhinet/models/feature_provider.py',
                         ROOT/'mhinet/visualization/visualization.py',ROOT/'mhinet/engine/evaluate.py']},
        'test_used':False,
        'warning':'Two validation pairs and two engineering optimizer steps do not establish accuracy or generalization.'}
    destination = ROOT/'artifacts/batch_visualization_summary.json'
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(destination)
    print('checkpoint_sha256',run['last_checkpoint']['sha256'])
    print('trajectory_means',[r['mean'] for r in summary['trajectory']['mace_input_px_conditional_valid']])
    print('failure_rate',summary['failure_rate'],'latency_ms',summary['latency_ms_per_pair_single_pass'])

if __name__ == '__main__':
    main()
