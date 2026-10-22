"""Evaluate original epoch-30 U/B/C on the exact new held-out rectangles."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from mhinet.config import RuntimePaths
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.d16_experiment.train import write
from .evaluation import validate


def main():
    c=json.loads(Path('mhinet/v2/local_occlusion/config.json').read_text())
    torch.set_num_threads(2);torch.manual_seed(c['seed'])
    model=build_fresh_shared(replace(RuntimePaths.from_json(c['runtime']),device='cuda:0'))
    dataset=GeometryPairDataset(c['dataset'],json.loads(Path(c['selection']).read_text())['val'])
    for variant in ['U','B','C']:
        output=Path(c['output'])/'original_baselines'/variant;output.mkdir(parents=True,exist_ok=False)
        checkpoint=Path('outputs/d16_correspondence_20261022')/variant/'latest.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['epoch']==30
        model.mvt.load_state_dict(state['mvt']);model.stage1_head.load_state_dict(state['head'])
        result=validate(model,dataset,c,output,30,full=True);result['checkpoint']=str(checkpoint)
        write(output/'epoch_030.json',result)
        print(json.dumps(dict(variant=variant,occlusion=result['occlusion'])),flush=True)


if __name__=='__main__':main()
