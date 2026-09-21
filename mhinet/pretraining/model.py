"""Shared descriptor model and dependency-free, masked-QKV-safe LoRA."""
from contextlib import nullcontext
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mhinet.models.feature_provider import SharedFeatureProvider, build_feature_provider


class LoRALinear(nn.Module):
    """Keep the original linear forward (including DINO's masked K bias)."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha / rank
        for p in base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = self.base(x)
        residual = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + residual.to(base.dtype) * self.scale


class SharedDescriptorNetwork(SharedFeatureProvider):
    """Same shared components as MHINet; adapter path explicitly enables autograd."""
    def configure(self, lora=False):
        self.set_training_groups(mvt=True, vgg=True, dedode=True, ghim_head=True)
        self.lora_enabled = bool(lora)
        if lora:
            if len(self.dino.blocks) != 18:
                raise ValueError('Expected pinned DINO blocks 0..17')
            for block in self.dino.blocks[8:18]:
                block.attn.qkv = LoRALinear(block.attn.qkv)
        return self

    def _extract_batched_descriptors(self, images):
        if not self.lora_enabled:
            return super()._extract_batched_descriptors(images)
        flat = images.flatten(0, 1)
        enc = self.shared_encoder
        normalized = (flat - enc.imagenet_mean.to(flat)) / enc.imagenet_std.to(flat)
        with torch.autocast(flat.device.type, dtype=torch.bfloat16, enabled=flat.is_cuda):
            with torch.no_grad():
                tokens, (h, w) = self.dino.prepare_tokens_with_masks(normalized)
            features = []
            for i, block in enumerate(self.dino.blocks):
                rope = self.dino.rope_embed(H=h, W=w)
                with torch.no_grad() if i < 8 else nullcontext():
                    if i >= 8 and self.training and torch.is_grad_enabled():
                        tokens = checkpoint(block, tokens, rope, use_reentrant=False)
                    else:
                        tokens = block(tokens, rope)
                if i in enc.layer_indices:
                    patches = self.dino.norm(tokens)[:, self.dino.n_storage_tokens + 1:]
                    features.append(patches.reshape(images.shape[0], 2, h, w, -1))
            return torch.cat(features, -1), (h, w)

    def optimizer_groups(self):
        rates = {'mvt': 1e-6, 'vgg': 5e-6, 'dedode': 1e-5,
                 'stage1_head_parameters': 1e-6, 'dino': 1e-5}
        return [dict(name=k, params=[p for p in v if p.requires_grad], lr=rates[k])
                for k, v in self.parameter_groups().items() if any(p.requires_grad for p in v)]


def build_shared_network(runtime, *, lora=False):
    original, report = build_feature_provider(runtime)
    model = SharedDescriptorNetwork(original.descriptor, original.stage1_head).configure(lora)
    return model.to(runtime.device), report


def export_shared(model, path, metadata):
    """Portable bundle excludes frozen DINO base; requires hash-matched base weights.

    Consumers in either project use load_shared rather than assuming legacy keys.
    No MHIR or LoMa dense matcher state is included.
    """
    from pathlib import Path
    from mhinet.engine.checkpointing import _atomic_torch_save
    _atomic_torch_save({'format': 'mhinet.shared-descriptor.v1', 'lora': model.lora_enabled,
                'metadata': metadata, 'state': {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                    if not k.startswith('descriptor.encoder.frozen_dinov3.model.') or '.lora_' in k
                }}, Path(path))


def load_shared(runtime, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format'] != 'mhinet.shared-descriptor.v1':
        raise ValueError('Not a shared descriptor bundle')
    from mhinet.config import sha256_file
    expected = payload['metadata']['dino_sha256']
    if sha256_file(runtime.dino_checkpoint) != expected:
        raise ValueError('Frozen DINO checkpoint hash mismatch')
    model, _ = build_shared_network(runtime, lora=payload['lora'])
    missing, unexpected = model.load_state_dict(payload['state'], strict=False)
    if unexpected or any(not k.startswith('descriptor.encoder.frozen_dinov3.model.')
                         or '.lora_' in k for k in missing):
        raise ValueError(f'Incomplete shared bundle: {missing}, {unexpected}')
    return model
