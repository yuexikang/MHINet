"""Explicit, auditable pretrained ablation; never randomize DINO or VGG."""
import hashlib
import torch
from torch import nn


def group_hashes(model):
    result={}
    for name,params in model._all_parameter_groups().items():
        h=hashlib.sha256()
        for p in params:
            h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name]=h.hexdigest()
    return result


def initialize_ablation(model, mode, seed):
    if mode not in ('pretrained','random_except_dino_vgg'):
        raise ValueError(f'Unknown initialization: {mode}')
    before=group_hashes(model)
    report={'mode':mode,'seed':seed,'before':before}
    if mode=='random_except_dino_vgg':
        groups=model._all_parameter_groups()
        target={id(p):p for name in ('mvt','dedode','stage1_head_parameters') for p in groups[name]}
        covered=set()
        device=next(model.parameters()).device
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for module in model.modules():
                direct=list(module.parameters(recurse=False))
                if not direct or not all(id(p) in target for p in direct):continue
                if hasattr(module,'reset_parameters'):
                    module.reset_parameters()
                    covered.update(id(p) for p in direct)
                else:
                    for name,p in module.named_parameters(recurse=False):
                        if id(p) in covered:continue
                        with torch.no_grad():
                            if p.ndim>=2:nn.init.xavier_uniform_(p)
                            elif name=='weight' and 'norm' in type(module).__name__.lower():nn.init.ones_(p)
                            elif name=='bias':nn.init.zeros_(p)
                            else:nn.init.normal_(p,std=.02)
                        covered.add(id(p))
        if covered!=set(target):raise RuntimeError('Random initialization missed parameters')
        # Adapter/MHIR already newly initialized by build_model, identical to A.
        # Preserve their custom normalization and zero prediction layer.
        report['reset_parameter_tensors']=len(covered)
    report['after']=group_hashes(model)
    for name in ('dino','vgg','new_modules'):
        if before[name]!=report['after'][name]:raise AssertionError(f'Unexpected reset: {name}')
    for decoder in model.refinement_decoders.values():
        if torch.count_nonzero(decoder.out_conv.weight) or torch.count_nonzero(decoder.out_conv.bias):
            raise AssertionError('MHIR prediction layer must start at zero')
    return report
