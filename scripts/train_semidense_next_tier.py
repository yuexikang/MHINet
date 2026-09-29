"""Start a new data tier from a completed semidense model, with fresh optimizer.

Separate entry point keeps the implementation hashes of running jobs unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mhinet.config import sha256_file
from mhinet.downstream import train
from mhinet.downstream.training import SemidenseSystem
from mhinet.pretraining.model import build_shared_network
from mhinet.downstream.semidense import SemidenseConfig,SemidenseMatcher
from dataclasses import replace


def initialize_system(runtime,config):
    shared,_=build_shared_network(runtime,lora=False)
    return SemidenseSystem(shared,SemidenseMatcher(config)).to(runtime.device)


def load_completed_semidense(runtime, path, config):
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    metadata = payload['metadata']
    if payload.get('format') != 'mhinet.training' or metadata.get('task') != 'semidense_qrru_v1':
        raise ValueError('Expected a semidense training checkpoint')
    if payload['progress']['optimizer_step'] != metadata['total_steps']:
        raise ValueError('Source training has not completed')
    # Normalize legacy metadata; inference batching is not a training parameter.
    source_config = SemidenseConfig(**metadata['matcher'])
    if replace(source_config, inference_window_chunk=config.inference_window_chunk,input_size=config.input_size) != config:
        raise ValueError('Matcher configuration differs from source')
    if metadata['dino_sha256'] != sha256_file(runtime.dino_checkpoint):
        raise ValueError('DINO identity differs from source')
    source = Path(metadata['source_checkpoint'])
    if sha256_file(source) != metadata['source_sha256']:
        raise ValueError('Original shared source checksum differs')
    # Full state below supplies all shared and matcher weights; source may itself
    # be a semidense checkpoint. Resolution changes start a fresh optimizer.
    system = initialize_system(runtime,config)
    system.load_state_dict(payload['model'], strict=True)
    print(f'Loaded full semidense weights from {path}, step {payload["progress"]["optimizer_step"]}; fresh optimizer and schedule', flush=True)
    return system


if __name__ == '__main__':
    train.load_first_stage = load_completed_semidense
    raise SystemExit(train.main())
