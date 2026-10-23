"""Resolve observed historical GE exposure before opening final test predictions."""
import json
from pathlib import Path
from .prepare import ROOT,RGB,read,balanced
from mhinet.v2.d16_experiment.train import write


def main():
    path=ROOT/'test_selection.json'
    if path.exists():raise FileExistsError(path)
    selected=json.loads((ROOT/'selection.json').read_text())
    historical=set()
    for split in ('train','val'):
        for r in read(Path('/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2')/split/'pairs.jsonl'):
            historical.add(Path(r.get('source','')).name)
    ge=[r for r in selected['rgb_test'] if r['source_dataset']=='googleearth_current']
    overlap=[r['pair_id'] for r in ge if Path(r['source_relative_path']).name in historical]
    rows=balanced([r for r in read(RGB/'test/pairs.jsonl') if r['source_dataset'] in ('flair_rgb','whu')],256,20261023,key='source_dataset')
    used={(r['source_dataset'],r['source_relative_path']) for split in ('train','val') for r in read(RGB/split/'pairs.jsonl')}
    assert not used & {(r['source_dataset'],r['source_relative_path']) for r in rows}
    for r in rows:
        for k in ('image_A','image_B','mask_A_overlap','mask_B_overlap'):r[k]='test/'+r[k]
        r['G']=r['source_dataset']
    write(path,dict(roots=selected['roots'],hz_test=selected['hz_test'],rgb_test=rows))
    write(ROOT/'test_provenance.json',dict(original_selection_preserved=True,original_ge_count=len(ge),historical_ge_overlap_ids=overlap,
        final_rgb_domains={'flair_rgb':128,'whu':128},reason='All 85 selected GE source basenames occur in historical shared-descriptor train/val. SharedDescriptorNetwork.configure trains MVT and stage1 head. Exclude GE before any test inference.',
        chain=['shared_stable_v2_full_tier3_seed0/shared_descriptor.pt','semidense_stable_v2_tier1_lr_c_seed0/latest.pt','semidense_stable_v2_tier3_lr_e_seed0.pt','semidense_rgb512_h0joint_from_e_lr8e5_seed42_5ep.pt'],
        scope='Source identity separation from audited local supervised training/selection. External foundation pretraining and geographic proximity not certified.',
        timing='After development decision locked, before any sealed-test model inference; no test outcomes used.'))

if __name__=='__main__':main()
