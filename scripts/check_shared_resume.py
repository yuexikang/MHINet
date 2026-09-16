"""Compare uninterrupted and split/resumed runs at the same optimizer boundary."""
import argparse
import json
from pathlib import Path
import torch
from mhinet.config import sha256_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',required=True)
    parser.add_argument('--continuous',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    a=torch.load(Path(args.split)/'latest.pt',map_location='cpu',weights_only=True,mmap=True)
    b=torch.load(Path(args.continuous)/'latest.pt',map_location='cpu',weights_only=True,mmap=True)
    assert a['metadata']==b['metadata']
    assert a['progress']==b['progress']
    assert a['scheduler']==b['scheduler']
    differences=[]
    for key,x in a['model'].items():
        y=b['model'][key]
        if not torch.equal(x,y):
            differences.append((key,float((x.float()-y.float()).abs().max())))
    maximum=max((v for _,v in differences),default=0.)
    assert maximum < 1e-5, (maximum,differences[:10])
    opt_max=0.
    for key,state in a['optimizer']['state'].items():
        for name,x in state.items():
            y=b['optimizer']['state'][key][name]
            if isinstance(x,torch.Tensor):
                opt_max=max(opt_max,float((x.float()-y.float()).abs().max()))
            else:
                assert x==y
    assert opt_max < 1e-5,opt_max
    report=dict(passed=True,progress=a['progress'],max_parameter_difference=maximum,
        differing_tensors=len(differences),max_optimizer_difference=opt_max,
        split=str(args.split),continuous=str(args.continuous),
        split_checkpoint_sha256=sha256_file(Path(args.split)/'latest.pt'),
        continuous_checkpoint_sha256=sha256_file(Path(args.continuous)/'latest.pt'),
        tolerance=1e-5,note='Pinned environment with strict deterministic kernels; cross-platform bitwise identity is not guaranteed.')
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
