"""Read-only old-data retention and new-data acceptance audit."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from mhinet.dataio.stable_geometry import check_geometry, POLICY


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    groups=defaultdict(list)
    for line in args.manifest.open():
        r=json.loads(line)
        accepted=[]
        for bound in (5.,10.,20.):
            ok,_=check_geometry(r['H_A_to_B'],r['size_A'],r['size_B'],{**POLICY,'max_jacobian_bound':bound})
            accepted.append(ok)
        groups[r['tier']].append([*accepted,abs(r['rotation_B_degrees']),r['attempts']])
    report={}
    for tier,values in groups.items():
        a=np.asarray(values); kept=a[:,1].astype(bool)
        report[tier]=dict(pairs=len(a),retention_by_jacobian_bound=dict(zip(('5','10','20'),a[:,:3].mean(0).tolist())),
            absolute_rotation_p50_p90_before=np.quantile(a[:,3],[.5,.9]).tolist(),
            absolute_rotation_p50_p90_kept=np.quantile(a[kept,3],[.5,.9]).tolist() if kept.any() else None,
            attempts_p50_p90_max=np.quantile(a[:,4],[.5,.9,1]).tolist())
    result=dict(manifest=str(args.manifest.resolve()),policy=POLICY,tiers=report)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
