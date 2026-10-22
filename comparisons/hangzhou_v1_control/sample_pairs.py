"""Randomly show actual training/validation rows without synthesized labels."""
import json
import random
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path('/home/disk1/MHINet')
RUN = ROOT/'outputs/mhi_v2/hangzhou_fresh_joint_20261021'
OUT = ROOT/'outputs/hangzhou_v1_control/samples'

def main():
    config = json.loads((RUN/'config.json').read_text())
    selection = json.loads((RUN/'selection.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = {}
    rng = random.Random(20261022)
    for split in ('train', 'val'):
        # One random row per geometry group, no quality-based hand selection.
        rows = [rng.choice([r for r in selection[split] if r['G'] == group])
                for group in ('G0','G1','G2','G3')]
        chosen[split] = rows
        fig, axes = plt.subplots(4, 2, figsize=(11, 21), dpi=110)
        for i, row in enumerate(rows):
            for j, side in enumerate('AB'):
                image = Image.open(Path(config['dataset'])/row[f'image_{side}']).convert('RGB')
                axes[i,j].imshow(image)
                source = row.get(f'source_{side}', {})
                axes[i,j].set_title(f"{row['G']} | {side} | {source.get('sensor','')} | {source.get('date','')}", fontsize=12)
                axes[i,j].axis('off')
            axes[i,0].text(0, -0.035, row['pair_id'], transform=axes[i,0].transAxes, fontsize=8)
        fig.suptitle(f'{split.upper()} - random sample of actual selected pairs\nOne per G0-G3; seed 20261022; original A/B; no true point annotations', fontsize=14)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(OUT/f'{split}_pairs.png'); plt.close(fig)
    (OUT/'selection.json').write_text(json.dumps(chosen, indent=2, ensure_ascii=False))
    print(OUT)

if __name__ == '__main__':
    main()
