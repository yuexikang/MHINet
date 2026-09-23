"""Start a new data tier from a completed semidense model, with fresh optimizer.

Separate entry point keeps the implementation hashes of running jobs unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mhinet.config import sha256_file
from mhinet.downstream import train
from mhinet.downstream.training import load_first_stage


def load_completed_semidense(runtime, path, config):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    metadata = payload['metadata']
    if payload.get('format') != 'mhinet.training' or metadata.get('task') != 'semidense_qrru_v1':
        raise ValueError('Expected a semidense training checkpoint')
    if payload['progress']['optimizer_step'] != metadata['total_steps']:
        raise ValueError('Source training has not completed')
    if metadata['matcher'] != config.__dict__:
        raise ValueError('Matcher configuration differs from source')
    if metadata['dino_sha256'] != sha256_file(runtime.dino_checkpoint):
        raise ValueError('DINO identity differs from source')
    source = Path(metadata['source_checkpoint'])
    if sha256_file(source) != metadata['source_sha256']:
        raise ValueError('Original shared source checksum differs')
    system = load_first_stage(runtime, source, config)
    system.load_state_dict(payload['model'], strict=True)
    print(f'Loaded full semidense weights from {path}, step {payload["progress"]["optimizer_step"]}; fresh optimizer and schedule', flush=True)
    return system


if __name__ == '__main__':
    train.load_first_stage = load_completed_semidense
    raise SystemExit(train.main())
