"""Branch-only training chunk ablation; production continuation checks stay strict."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mhinet.config import sha256_file
from mhinet.downstream import train
from mhinet.pretraining.data import SharedPairDataset
from scripts.train_semidense_next_tier import load_completed_semidense


def load_chunk_variant(runtime,path,config):
    if config.window_chunk not in (32,128):raise ValueError('Experiment permits only chunks 32/128')
    # Load the unchanged source protocol strictly, then change only execution batching.
    system=load_completed_semidense(runtime,path,replace(config,window_chunk=32))
    system.matcher.config=config
    return system


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--limit-val',required=True,type=int)
    args,_=parser.parse_known_args(argv)
    config=json.loads(args.config.read_text())
    if config['experiment_entry_sha256']!=sha256_file(Path(__file__)):
        raise ValueError('Experiment entry identity mismatch')
    original_batch_loss=train.batch_loss
    def batch_loss(system,batch,device):
        loss,records=original_batch_loss(system,batch,device)
        if system.training:
            for row,pair_id in zip(records,batch['pair_id']):row['pair_id']=pair_id
        return loss,records
    def load(runtime,path,matcher_config):
        system=load_chunk_variant(runtime,path,matcher_config)
        if not (args.output/'validation/step_0000000.json').exists():
            val=SharedPairDataset(runtime.data_root/'val/pairs.jsonl',tier=config['tier'],max_pairs=args.limit_val)
            train.validate(system,val,runtime.device,args.output,0)
        return system
    train.load_first_stage=load;train.batch_loss=batch_loss
    return train.main(argv)


if __name__=='__main__':raise SystemExit(main())
