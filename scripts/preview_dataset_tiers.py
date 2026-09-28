"""Render reproducible random examples from the generated three-tier dataset."""
import argparse
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1'))
    parser.add_argument('--split', choices=['train','val'], default='val')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('outputs/previews/quadrant_tiers_val_seed42'))
    args=parser.parse_args()
    groups={(t,r):[] for t in (1,2,3) for r in (.8,.6,.4)}
    with (args.root/args.split/'pairs.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            groups[(row['tier'],row['resolution_ratio'])].append(row)
    rng=random.Random(args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for tier in (1,2,3):
        canvas=Image.new('RGB',(1024,3*564),'#181818')
        draw=ImageDraw.Draw(canvas)
        for index,ratio in enumerate((.8,.6,.4)):
            row=rng.choice(groups[(tier,ratio)])
            base=index*564
            draw.text((8,base+4),f"Tier {tier} | ratio 1:{ratio} | {row['pair_id']}",fill='white')
            visible=row['visible_overlap_fractions']
            draw.text((8,base+22),f"Visible overlap A/B: {visible[0]:.1%} / {visible[1]:.1%}",fill='white')
            for side,col in [('A',0),('B',1)]:
                path=args.root/args.split/row[f'image_{side}']
                with Image.open(path) as image:
                    image=image.convert('RGB')
                    draw.text((col*512+8,base+40),f'{side}: {image.width} x {image.height}',fill='white')
                    image.thumbnail((504,504),Image.Resampling.LANCZOS)
                    canvas.paste(image,(col*512+4,base+60))
            records.append({'tier':tier,'ratio':ratio,'pair_id':row['pair_id'],
                            'metadata':str(args.root/args.split/row['metadata']),
                            'image_A':str(args.root/args.split/row['image_A']),
                            'image_B':str(args.root/args.split/row['image_B'])})
        canvas.save(args.output/f'tier_{tier}.jpg',quality=92)
    (args.output/'selection.json').write_text(json.dumps({'seed':args.seed,'split':args.split,'pairs':records},indent=2)+'\n')
    print(args.output.resolve())


if __name__=='__main__':
    main()
