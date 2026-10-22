"""Contact sheets for every high-confidence >8px query (no selected subset)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from mhinet.v2.data import GeometryPairDataset


def main():
    c=json.loads(Path('mhinet/v2/local_occlusion/config.json').read_text())
    dataset=GeometryPairDataset(c['dataset'],json.loads(Path(c['selection']).read_text())['val'])
    root=Path(c['output'])/'diagnostics'
    for batch in dataset:
        pid=batch['pair_id'][0]
        images=[Image.fromarray((a.permute(1,2,0).numpy()*255).round().astype('uint8')) for a in batch['images'][0]]
        for variant in ('U','B','C'):
            path=root/variant/(pid+'.npz')
            with np.load(path) as a:
                queries=np.where(a['valid'] & (a['positive_mass']>=.9) & (a['epe']>8))[0]
                for page,start in enumerate(range(0,len(queries),24)):
                    selected=queries[start:start+24];sheet=Image.new('RGB',(4*200,((len(selected)+3)//4)*100),'white')
                    draw=ImageDraw.Draw(sheet)
                    for i,q in enumerate(selected):
                        x=(i%4)*200;y=(i//4)*100
                        for col,(image,xy) in enumerate([(images[0],a['source'][q]),(images[1],a['target'][q]),(images[1],a['prediction'][q])]):
                            cx,cy=np.round(xy).astype(int);crop=image.crop((cx-32,cy-32,cx+32,cy+32))
                            ImageDraw.Draw(crop).ellipse((29,29,35,35),outline='yellow',width=1)
                            sheet.paste(crop,(x+col*66,y+30))
                        dx,dy=a['delta'][q]
                        draw.text((x,y),f'q{q} E={a["epe"][q]:.1f} dx={dx:.1f} dy={dy:.1f}',fill='black')
                        draw.text((x,y+14),'A source | B reference | B prediction',fill='black')
                    sheet.save(root/variant/f'{pid}_all_high_confidence_{page:02d}.jpg',quality=90)
    print('Exported all high-confidence query patches; outside-image crop padding is black.')


if __name__=='__main__':main()
