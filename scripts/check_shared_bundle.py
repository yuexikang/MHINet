"""Verify exported shared tensors and inference against their training checkpoint."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from mhinet.config import RuntimePaths, sha256_file
from mhinet.pretraining.model import load_shared
from mhinet.pretraining.data import SharedPairDataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.run)
    runtime=replace(RuntimePaths.from_json('configs/runtime_paths.quadrant.server.json'),device='cuda:0')
    model=load_shared(runtime,root/'shared_descriptor.pt').eval()
    checkpoint=torch.load(root/'latest.pt',map_location='cpu',weights_only=True)
    current=model.state_dict()
    assert set(current)==set(checkpoint['model'])
    mismatches=[k for k,v in current.items() if not torch.equal(v.cpu(),checkpoint['model'][k])]
    assert not mismatches, mismatches
    sample=SharedPairDataset(runtime.data_root/'val/pairs.jsonl',tier=1,max_pairs=1)[0]
    with torch.no_grad():
        before=model(sample['images'][None].cuda())
        refs={k:v.cpu() for k,v in before['pyramid'].items()}
        h0=before['H0_norm'].cpu()
        model.load_state_dict(checkpoint['model'],strict=True)
        after=model(sample['images'][None].cuda())
    errors={f'D{k}':float((v.cpu()-refs[k]).abs().max()) for k,v in after['pyramid'].items()}
    errors['H0']=float((after['H0_norm'].cpu()-h0).abs().max())
    assert all(v==0 for v in errors.values()),errors
    report=dict(passed=True,lora=model.lora_enabled,state_tensors=len(current),
        inference_max_abs=errors,bundle=str(root/'shared_descriptor.pt'),
        bundle_sha256=sha256_file(root/'shared_descriptor.pt'),
        checkpoint_sha256=sha256_file(root/'latest.pt'))
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
