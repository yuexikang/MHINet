"""Non-mutating checkpoint evaluation: full val, H0 only, no descriptor decoding."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.pretraining.data import SharedPairDataset
from mhinet.pretraining.overlap_metrics import overlap_projection_error,summarize_overlap


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--tiers',type=int,nargs='+',default=[1,2])
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    runtime=replace(RuntimePaths.from_json(state['metadata']['config']['runtime']),device='cuda:0')
    model,_=build_shared_network(runtime,lora=state['metadata']['config']['lora'])
    model.load_state_dict(state['model'],strict=True);model.eval()
    sha=sha256_file(args.checkpoint)
    for tier in args.tiers:
        data=SharedPairDataset(runtime.data_root/'val/pairs.jsonl',tier=tier)
        rows=[];start=time.perf_counter()
        with (output/f'tier{tier}_pairs.jsonl').open('w') as stream:
            for batch in tqdm(DataLoader(data,batch_size=1,num_workers=args.workers),desc=f'H0 val tier{tier}',mininterval=10):
                result=model(batch['images'].cuda(),pyramid_scales=())
                record=overlap_projection_error(result['H0_norm'],batch['H_gt_norm'].cuda(),
                    batch['mask_A_overlap'].cuda(),batch['mask_B_overlap'].cuda(),
                    fit_valid=bool(result['stage1_valid'][0]))
                rows.append(record)
                stream.write(json.dumps(dict(pair_id=batch['pair_id'][0],**record))+'\n')
        summary=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha,
            source_step=state['progress']['optimizer_step'],tier=tier,split='val',
            manifest_sha256=sha256_file(data.manifest),elapsed_seconds=time.perf_counter()-start,
            H0_overlap=summarize_overlap(rows))
        (output/f'tier{tier}_summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
