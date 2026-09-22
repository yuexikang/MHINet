"""Explicit second-stage ownership and strict first-stage loading."""
from pathlib import Path
import torch
from torch import nn
from mhinet.pretraining.model import build_shared_network,load_shared
from mhinet.config import sha256_file
from .semidense import SemidenseMatcher,SemidenseConfig


class SemidenseSystem(nn.Module):
    def __init__(self,shared,matcher):
        super().__init__()
        self.shared=shared
        self.matcher=matcher
        shared.set_training_groups(mvt=False,vgg=True,dedode=True,ghim_head=False)

    def forward(self,images):
        return self.shared(images)

    def optimizer_groups(self,shared_lr=1e-5,head_lr=1e-4):
        return [dict(name='cgmdp',params=[p for p in self.shared.parameters() if p.requires_grad],lr=shared_lr),
                dict(name='semidense_qrru',params=list(self.matcher.parameters()),lr=head_lr)]


def load_first_stage(runtime,path,config):
    """Accept shared export, shared training checkpoint, or full MHINet checkpoint.

    Full MHINet extraction explicitly selects feature_provider; no matcher weights
    or optimizer state migrate. Missing shared keys always fail strictly.
    """
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if payload.get('format')=='mhinet.shared-descriptor.v1':
        if payload.get('lora'):raise ValueError('Mainline requires frozen non-LoRA DINO')
        shared=load_shared(runtime,path)
    elif payload.get('format')=='mhinet.training' and isinstance(payload.get('model'),dict):
        if payload.get('metadata',{}).get('config',{}).get('lora',False):
            raise ValueError('LoRA checkpoint is outside this protocol')
        expected=payload.get('metadata',{}).get('dino_sha256')
        if expected and expected!=sha256_file(runtime.dino_checkpoint):raise ValueError('DINO identity mismatch')
        shared,_=build_shared_network(runtime,lora=False)
        state=payload['model']
        if any(k.startswith('feature_provider.') for k in state):
            state={k[len('feature_provider.'):]:v for k,v in state.items() if k.startswith('feature_provider.')}
        shared.load_state_dict(state,strict=True)
    else:
        raise ValueError('Unrecognized first-stage checkpoint format')
    return SemidenseSystem(shared,SemidenseMatcher(config)).to(runtime.device)
