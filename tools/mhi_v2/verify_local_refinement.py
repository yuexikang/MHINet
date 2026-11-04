"""Compare the released local refiner with its trained checkpoint calculation."""
import json
from pathlib import Path

import torch

from mhinet.v2.pipeline import load_mhi_v2
from mhinet.v2.qrru import QRRUV2
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.pair_matching.metrics import measure, totals


@torch.no_grad()
def main():
    assets = Path('/home/disk1/MHINet')
    out = Path(__file__).resolve().parents[2]/'reports/qrru_local_release'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = load_mhi_v2(device='cuda:2', assets_root=assets,
        runtime=json.loads((assets/'configs/mhi_v2/runtime.rgb512.server.json').read_text()))
    reference = QRRUV2(iterations=1).to('cuda:2').eval()
    reference.load_state_dict(model.refiner.state_dict(), strict=True)
    rgb = Path('/home/disk1/Data/datasets/MHINet_RGB_512/test')
    hz = Path('/home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4')
    records = []
    maximum = 0.
    for name, root, manifest in [('rgb', rgb, rgb/'pairs.jsonl'), ('hangzhou', hz, hz/'manifests/test_real.jsonl')]:
        rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        indices = torch.linspace(0, len(rows)-1, 16).round().long().tolist()
        data = GeometryPairDataset(root, [rows[i] for i in indices])
        for batch in data:
            shared = model.shared(batch['images'].to('cuda:2'), pyramid_scales=(8,4,2))
            features = shared['pyramid'][2][0].float()
            projected = reference.project(features)
            gt = batch['geometry'].to('cuda:2')
            for mutual in (False, True):
                prediction = model.match_shared(shared, diagnostics=True, mutual=mutual)
                trace = prediction['diagnostics']
                a = prediction['points_a']
                b = trace['points_b_d2']
                if mutual:
                    b = b[trace['output_mask']]
                updates = []
                for start in range(0,len(a),512):
                    r = reference(projected[0],projected[1],(a[start:start+512]+.5)/2-.5,
                        (b[start:start+512]+.5)/2-.5,projected=True,history=False)
                    updates.append(r['flows'][0].mean((-2,-1))*2)
                expected = b+torch.cat(updates) if updates else b
                delta = float((prediction['points_b']-expected).abs().max()) if len(b) else 0.
                maximum = max(maximum,delta)
                torch.testing.assert_close(prediction['points_b'],expected,atol=1e-5,rtol=0)
                current = measure(a,prediction['points_b'],gt)[0]
                trained = measure(a,expected,gt)[0]
                assert all(current[k]==trained[k] for k in ('n','visible','correct1','correct3','correct5','correct10'))
                records.append(dict(dataset=name,pair_id=batch['pair_id'][0],mutual=mutual,
                    coordinate_max_difference=delta,**current))
            print(name,batch['pair_id'][0],flush=True)
    summary = {f'{name}_{mode}':totals([r for r in records if r['dataset']==name and r['mutual']==mode])
               for name in ('rgb','hangzhou') for mode in (False,True)}
    result = dict(release=model.release,input_pairs=32,coordinate_max_difference=maximum,
                  metric_counts_identical=True,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  summary=summary,records=records)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','summary')}),flush=True)


if __name__=='__main__':
    main()
