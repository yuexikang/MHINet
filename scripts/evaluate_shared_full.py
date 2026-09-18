"""Full validation of an existing shared checkpoint on a specified data tier."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.pretraining.data import SharedPairDataset
from mhinet.pretraining.train import validate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--tier',type=int,required=True,choices=[1,2,3])
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    runtime=replace(RuntimePaths.from_json(state['metadata']['config']['runtime']),device='cuda:0')
    model,_=build_shared_network(runtime,lora=state['metadata']['config']['lora'])
    model.load_state_dict(state['model'],strict=True)
    data=SharedPairDataset(runtime.data_root/'val/pairs.jsonl',tier=args.tier)
    summary=validate(model,data,runtime.device,output,state['progress']['optimizer_step'],
                     args.workers,state['metadata']['config']['queries'])
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256_file(args.checkpoint),
        tier=args.tier,split='val',manifest_sha256=sha256_file(data.manifest),summary=summary)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
