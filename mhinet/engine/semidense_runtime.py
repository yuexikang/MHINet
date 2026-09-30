"""Optimizer-boundary persistence and control-plane work for semidense DDP."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

from .checkpointing import capture_rng_state, restore_rng_state, save_checkpoint

# Only these audited orchestration revisions can migrate automatically. Model,
# loss, data, schedule and all other implementation hashes remain strict.
LEGACY_ORCHESTRATION = {
    'train.py': '448fac6dd3547bed7183cb4e466469c84e75ca72ef4f9d0c81abe940d43890db',
    'visualize.py': 'bf748a424c6edfa71be63bc8c02c8405a15190f34e06c30260f2a894ce704a45',
}


def _pack(value):
    stream=io.BytesIO();torch.save(value,stream)
    return stream.getvalue()


def _unpack(value):
    return torch.load(io.BytesIO(value),map_location='cpu',weights_only=True)


def model_buffers(model):
    return {name:value.detach().cpu().clone() for name,value in model.named_buffers()}


@torch.no_grad()
def restore_model_buffers(model, buffers):
    current=dict(model.named_buffers())
    if current.keys()!=buffers.keys():raise ValueError('Checkpoint model-buffer names differ')
    for name,value in current.items():value.copy_(buffers[name])


def restore_rank_buffers(payload, model, rank):
    buffers=(payload.get('auxiliary_state') or {}).get('rank_buffers')
    if buffers is not None:restore_model_buffers(model,buffers[rank])


def validation_buffers(model, *, rank, control_group):
    """Use rank-zero BN statistics for all validation shards, then restore."""
    if control_group is None:return None
    original=model_buffers(model)
    values=[_pack(original) if rank==0 else None]
    dist.broadcast_object_list(values,src=0,group=control_group)
    if rank!=0:restore_model_buffers(model,_unpack(values[0]))
    return original


def resume_expectations(current, saved):
    expected = deepcopy(current)
    legacy = saved.get('implementation', {}).get('train.py') == LEGACY_ORCHESTRATION['train.py']
    if legacy and 'training_runtime_sha256' not in saved:
        expected.pop('training_runtime_sha256', None)
        expected.pop('checkpoint_io_sha256', None)
        expected.pop('shared_data_loader_sha256', None)
    for name, digest in LEGACY_ORCHESTRATION.items():
        if legacy and saved.get('implementation', {}).get(name) == digest:
            expected['implementation'][name] = digest
    return expected


def rank_zero_work(callback, *, rank=0, control_group=None):
    """Wait on CPU/Gloo during long IO/plots; propagate rank-zero failures."""
    status = [None]
    result = None
    if rank == 0:
        try:
            result = callback()
        except Exception as error:
            if control_group is None:
                raise
            status[0] = f'{type(error).__name__}: {error}'
    if control_group is not None:
        dist.broadcast_object_list(status, src=0, group=control_group)
    if status[0] is not None:
        raise RuntimeError(f'Rank-zero operation failed: {status[0]}')
    return result


def save_training_checkpoint(path, *, model, optimizer, scheduler, step, cursor,
                             epoch, metadata, rank=0, world_size=1, control_group=None,
                             execution_settings=None):
    local_rng = capture_rng_state(current_cuda_only=True)
    local_buffers = model_buffers(model)
    rank_states = [None] * world_size if rank == 0 else None
    if control_group is not None:
        # Gather bytes instead of pickling Tensor storage through object
        # collectives (incompatible with some pinned PyTorch storage versions).
        dist.gather_object(_pack(dict(rng=local_rng,buffers=local_buffers)), rank_states, dst=0, group=control_group)
        if rank == 0:
            rank_states=[_unpack(value) for value in rank_states]
    else:
        rank_states = [dict(rng=local_rng,buffers=local_buffers)]

    def save():
        return save_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            optimizer_step=step, data_progress=dict(cursor=cursor, epoch=epoch),
            metadata=metadata, rng_state=rank_states[0]['rng'],
            auxiliary_state=dict(rank_rng_states=[state['rng'] for state in rank_states],
                                 rank_buffers=[state['buffers'] for state in rank_states],world_size=world_size,
                                 execution=execution_settings or {}),
        )
    return rank_zero_work(save, rank=rank, control_group=control_group)


def restore_training_rng(payload, *, rank, world_size, seed):
    auxiliary = payload.get('auxiliary_state') or {}
    rank_states = auxiliary.get('rank_rng_states')
    if rank_states is not None:
        if auxiliary.get('world_size') != world_size or len(rank_states) != world_size:
            raise ValueError('Rank RNG world-size mismatch')
        restore_rng_state(rank_states[rank])
        return 'exact_per_rank'
    if world_size == 1 or rank == 0:
        state = deepcopy(payload['rng'])
        # Legacy DDP rank 0 saved all visible devices, but only its own stream
        # is meaningful. Avoid initializing unused CUDA contexts when loading it.
        if world_size > 1 and state.get('cuda_initialized'):
            state['torch_cuda'] = [state['torch_cuda'][0]]
            state['cuda_device_count'] = 1
            state['cuda_current_device_only'] = True
        restore_rng_state(state)
    else:
        # Missing rank states cannot be reconstructed. Explicit deterministic
        # fallback is recorded in resume_events.jsonl, never called "exact".
        value = seed + payload['progress']['optimizer_step'] * world_size + rank
        random.seed(value); np.random.seed(value % (2**32)); torch.manual_seed(value)
    return 'legacy_missing_rank_rng' if world_size > 1 else 'exact_single_rank'


def validation_indices(length, sample_count=None):
    if length < 1 or (sample_count is not None and sample_count < 1):
        raise ValueError('Validation length and sample count must be positive')
    count = length if sample_count is None else min(sample_count, length)
    return np.linspace(0, length - 1, num=count, dtype=int).tolist()


def boundary_actions(step, end, config):
    vis = config.get('visualization', {})
    validation = step % config['validate_every'] == 0 or step == end
    snapshot = bool(vis.get('enabled')) and (step % vis.get('every', 1000) == 0 or step == end)
    curves = bool(vis.get('enabled')) and (step % vis.get('curves_every', 100) == 0 or step == end)
    save = step % config['save_every'] == 0 or validation or snapshot or step == end
    return dict(save=save, validation=validation, snapshot=snapshot, curves=curves)


def reconcile_training_log(output, step):
    """Atomically rewind unsaved rows; retain originals for provenance."""
    root = Path(output)
    path = root / 'train.jsonl'
    if not path.exists():
        return 0
    lines = path.read_text().splitlines(keepends=True)
    kept, discarded = [], []
    last = 0
    for line in lines:
        try:
            row = json.loads(line)
            value = int(row['step'])
        except (ValueError, KeyError, TypeError):
            discarded.append(line)
            continue
        if value <= step and value > last:
            kept.append(line if line.endswith('\n') else line + '\n')
            last = value
        else:
            discarded.append(line)
    archive = root / 'resume_history' / f'{step:07d}_{time.time_ns()}'
    if discarded:
        archive.mkdir(parents=True)
        (archive / 'discarded_train.jsonl').write_text(''.join(discarded))
        fd, temporary = tempfile.mkstemp(prefix='.train.', dir=root)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.writelines(kept); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    for area in ('validation', 'visualizations'):
        for item in (root / area).glob('step_*'):
            try:
                value = int(item.stem.split('_')[1])
            except (ValueError, IndexError):
                continue
            if value > step:
                target = archive / area / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                item.rename(target)
    return len(discarded)
