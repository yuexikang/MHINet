"""Versioned, GT-free MHI-v2: Flow0 → local D8 → continuous Pair-D2."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from mhinet.config import sha256_file
from .feature_provider import build_fresh_shared
from .config import SemidenseConfigV2
from .pair_matching.local import LocalPairModulation, extract_windows, match_windows, center_indices

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/mhi_v2/releases/flow_pair_v2.json'


class MatchingTemperatures(nn.Module):
    """Only the two trained temperatures; no QRRU/DGFO modules are constructed."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.log_tau_c = nn.Parameter(torch.tensor(math.log(config.temperature)))
        self.log_tau_f = nn.Parameter(torch.tensor(math.log(config.temperature)))

    @property
    def tau_c(self):
        return self.log_tau_c.exp().clamp_min(1e-4)

    @property
    def tau_f(self):
        return self.log_tau_f.exp().clamp_min(1e-4)


def asset_path(path, root):
    path = Path(os.path.expandvars(str(path))).expanduser()
    return path if path.is_absolute() else root / path


def build_shared(runtime, device, root=ROOT):
    """Inference needs only source dependencies and DINO, not training manifests."""
    settings = {k: asset_path(runtime[k], root) for k in ('loma_root', 'loretta_source_root', 'dino_checkpoint')}
    return build_fresh_shared(SimpleNamespace(**settings, device=device))


def load_components(shared, matcher, adapter, flow, d2):
    """Load disjoint checkpoint components strictly, never merge partial state blindly."""
    shared.mvt.load_state_dict(flow['mvt'], strict=True)
    shared.stage1_head.load_state_dict(flow['head'], strict=True)
    shared.vgg.load_state_dict(d2['vgg'], strict=True)
    shared.dedode.load_state_dict(d2['dedode'], strict=True)
    adapter.load_state_dict(d2['adapter'], strict=True)
    matcher.load_state_dict(d2['temperatures'], strict=True)


class MHIV2(nn.Module):
    """match accepts one 2×3×512×512 RGB pair in [0,1]; coordinates are pixels."""
    def __init__(self, shared, matcher, adapter, max_parents=128):
        super().__init__()
        self.shared, self.matcher, self.adapter = shared, matcher, adapter
        self.max_parents = max_parents

    @torch.inference_mode()
    def match(self, images, *, diagnostics=False):
        if images.shape == (2, 3, 512, 512):
            images = images.unsqueeze(0)
        if images.shape != (1, 2, 3, 512, 512):
            raise ValueError('match expects a single pair: 2x3x512x512 or 1x2x3x512x512')
        self.eval()
        device = next(self.shared.parameters()).device
        shared = self.shared(images.to(device), pyramid_scales=(8, 4, 2))
        return self.match_shared(shared, diagnostics=diagnostics)

    @torch.inference_mode()
    def match_shared(self, shared, *, diagnostics=False):
        """Reuse already computed features; preserve parent-major query ordering."""
        windows = extract_windows(shared, self.matcher, self.max_parents)
        fine = match_windows(self.adapter, windows, self.matcher.tau_f)
        n = len(windows['parents'])
        parent = torch.arange(n, device=windows['source'].device).repeat_interleave(16)
        ids = ((windows['parents'] + .5) / 8 - .5).round().long()
        parent_ids = ids[:, 1]*64 + ids[:, 0]
        valid = windows['valid_a'][:, center_indices(parent.device)] & windows['valid_b'].any(-1)[:, None]
        result = dict(points_a=windows['source'].reshape(-1, 2),
                      points_b=fine['points_b'].reshape(-1, 2),
                      confidence=(fine['confidence']*windows['score8'][:, None]).sqrt().flatten(),
                      valid=valid.flatten(), parent=parent, parent_ids=parent_ids[parent])
        # Do not discard invalid rows or re-sort: historical metrics count every output.
        if diagnostics:
            result['diagnostics'] = dict(flow16_norm=shared['flow16_norm'],
                flow16_target_norm=shared['flow16_target_norm'],
                flow16_matchability=shared['flow16_matchability'], flow16_valid=shared['flow16_valid'],
                parent_ids=parent_ids, parents=windows['parents'], centers=windows['centers'],
                windows=windows, candidate_logp=fine['logp'], candidate_valid=fine['valid'],
                call_counts=shared['call_counts'])
        return result


def load_mhi_v2(config=DEFAULT_CONFIG, device='cuda:0', *, runtime=None, assets_root=None, variant='best'):
    """Paths are relative to assets_root (MHI_V2_ASSETS or repository by default).

    runtime may override just the three external DINO/source paths. Checkpoint
    hashes bind D2's recorded locator to its actual Flow content after relocation.
    """
    settings = json.loads(Path(config).read_text()) if not isinstance(config, dict) else dict(config)
    root = Path(assets_root or os.environ.get('MHI_V2_ASSETS', ROOT)).expanduser().resolve()
    if settings['architecture'] != 'flow_d8_pair_d2_v1':
        raise ValueError('Unsupported MHI-v2 release architecture')
    if settings['qrru'] or settings['dgfo'] or settings['global_modulation'] or settings['dustbin_rejection']:
        raise ValueError('Experimental refinement/rejection is not part of this release')
    if not 1 <= settings['max_parents'] <= 128:
        raise ValueError('max_parents must be 1..128 (16 outputs per parent, at most 2048)')
    manifest = json.loads(asset_path(settings['manifest'], ROOT).read_text())
    fs, ds = manifest['flow'], manifest['d2'][variant]
    checkpoints = []
    for spec in (fs, ds):
        path = asset_path(spec['path'], root)
        if sha256_file(path) != spec['sha256']:
            raise ValueError(f'Checkpoint SHA256 mismatch: {path}')
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if state['stage'] != spec['stage'] or state['epoch'] != spec['epoch']:
            raise ValueError(f'Checkpoint metadata mismatch: {path}')
        checkpoints.append(state)
    flow, d2 = checkpoints
    if d2['locator_checkpoint'] != ds['recorded_locator_checkpoint'] or ds['flow_sha256'] != fs['sha256']:
        raise ValueError('D2 checkpoint Flow provenance mismatch')
    c = SemidenseConfigV2(**d2['matcher_config'])
    if c.geometry_mode != 'flow_d16' or c.use_dgfo or c.use_discrepancy:
        raise ValueError('Checkpoint is not the validated Flow-only locator path')
    shared = build_shared(runtime or settings['runtime'], device, root)
    matcher = MatchingTemperatures(c).to(device)
    adapter = LocalPairModulation('pair').to(device)
    load_components(shared, matcher, adapter, flow, d2)
    model = MHIV2(shared, matcher, adapter, settings['max_parents']).requires_grad_(False).eval()
    model.release = dict(version=settings['version'], variant=variant, flow_epoch=fs['epoch'], d2_epoch=ds['epoch'])
    return model
