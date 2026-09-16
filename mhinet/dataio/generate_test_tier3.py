"""Independent same-image tier-3 synthetic test; never use cross-parent labels."""
import argparse
import csv
import json
from pathlib import Path
from tqdm import tqdm
from mhinet.config import sha256_file
from mhinet.dataio.generate_temporal import load_generator
from mhinet.dataio.quadrant_tiers import generate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/home/disk1/Data/datasets/GoogleEarth/evaluation_data'))
    p.add_argument('--output-dir',type=Path,default=Path('/home/disk1/Data/datasets/GoogleEarth_test_single_tier3_v1'))
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--generate',action='store_true')
    args=p.parse_args()
    with (args.source/'test_pairs.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    parents={}
    for index,row in enumerate(rows):
        for domain in ('Source','Target'):
            parents.setdefault(row[domain],index)
    entries=sorted(parents.items())
    if args.smoke:entries=entries[:2]
    for name,_ in entries:
        if not (args.source/name).is_file():raise FileNotFoundError(name)
    summary={'version':'single_parent_test_tier3_v1','status':'planned','smoke_only':args.smoke,
        'seed':args.seed,'source':str(args.source),'parents':len(entries),'planned_pairs':2*len(entries),
        'label_provenance':'exact_synthetic_same_parent','tier':3,
        'test_role':'sealed synthetic geometry/appearance robustness test; not real cross-temporal accuracy',
        'csv_sha256':sha256_file(args.source/'test_pairs.csv'),
        'driver_sha256':sha256_file(Path(__file__)),
        'generator_sha256':sha256_file(Path(__file__).with_name('quadrant_tiers.py'))}
    print(json.dumps(summary,indent=2),flush=True)
    if not args.generate:return
    args.output_dir.mkdir(parents=True,exist_ok=False)
    target=args.output_dir/'test'
    for name in ('images','masks','metadata'):(target/name).mkdir(parents=True)
    summary_path=args.output_dir/'dataset_summary.json'
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    legacy=load_generator(Path('/home/disk1/LoMa'))
    count=0
    with (target/'pairs.jsonl').open('w') as stream:
        for index,(name,original_index) in enumerate(tqdm(entries,desc='Tier3 synthetic test',unit='parent')):
            image=legacy.read_image(args.source/name,1)
            for j in range(2):
                pair_id=f'{Path(name).parent.name}_{Path(name).stem}_tier3_{j}'
                metadata={'split':'test','source':name,'parent_image_A':name,'parent_image_B':name,
                    'input_pair_row_index':original_index,'input_mode':'single_image','tier':3,
                    'resolution_ratio':(.8,.6,.4)[(index+j)%3],
                    'label_provenance':'exact_synthetic_same_parent','has_ground_truth':True}
                row=generate(image,image,target,pair_id,j,
                    legacy.derive_seed(args.seed,'test',name,j),None,metadata)
                stream.write(json.dumps(row)+'\n');count+=1
            stream.flush()
    summary.update(status='completed',generated_pairs=count,
                   manifest_sha256=sha256_file(target/'pairs.jsonl'))
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
