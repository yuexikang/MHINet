"""Apply the registered gates after complete epoch-30 development results."""
import json
from pathlib import Path
from mhinet.v2.d16_experiment.train import write
ROOT=Path('outputs/d16_final_round')


def read(path):return json.loads(path.read_text())


def main():
    base=read(Path('outputs/d16_local_occlusion/B0/epoch_030.json'))['clean']
    combined=read(Path('outputs/d16_local_occlusion/N2/epoch_030.json'))['clean']
    epaths={v:ROOT/'ablation'/v/'epoch_030.json' for v in ('E1','E2')}
    if all(p.exists() for p in epaths.values()):
        e={v:read(p)['clean'] for v,p in epaths.items()}
        fraction=(base['epe']-e['E1']['epe'])/(base['epe']-combined['epe'])
        needs_keep=fraction>=.8 and abs(e['E1']['flow_p3']-combined['flow_p3'])<=.005
        interaction={k:combined[k]-e['E1'][k]-e['E2'][k]+base[k] for k in ('epe','flow_p3','epe_gt32')}
        write(ROOT/'keep_gate.json',dict(keep_additions_required=needs_keep,e1_fraction_of_combined_epe_gain=fraction,
            e1_minus_n2_p3=e['E1']['flow_p3']-combined['flow_p3'],interaction=interaction,
            note='Single-seed diagnostic only. Additional E_geo/E_cls_mat only if both predefined keep gates pass.'))
        if needs_keep and not all((ROOT/'ablation'/v/'epoch_030.json').exists() for v in ('E_geo','E_cls_mat')):
            print('Keep-specific conditional runs are required before final configuration selection.');return
    else:print('Waiting for E1/E2 completion.');return
    paths={v:ROOT/'decoder'/f'{v}_seed42'/'epoch_030.json' for v in ('D00','D01','D10','D11')}
    if not all(p.exists() for p in paths.values()):print('Waiting for four decoder development runs.');return
    rows={v:read(p) for v,p in paths.items()};fallback=read(Path('outputs/d16_local_occlusion/U0/epoch_030.json'))['clean']
    checks={};eligible=[]
    for v,b in [('D01','D00'),('D10','D00'),('D11','D01')]:
        h=rows[v]['hangzhou']['overall'];old=rows[b]['hangzhou']['overall'];rgb=rows[v]['rgb']['overall'];rgb_old=rows[b]['rgb']['overall']
        gates=dict(hangzhou_gain=h['flow_p3']>=old['flow_p3']+.01,rgb_gain=rgb['flow_p3']>=rgb_old['flow_p3']+.01,
            epe=h['epe']<=old['epe']*1.05,tail_vs_control=h['epe_gt32']<=old['epe_gt32']+.005,
            groups=all(rows[v]['hangzhou']['groups'][g]['flow_p3']>=rows[b]['hangzhou']['groups'][g]['flow_p3']-.01 for g in ('G1','G3')),
            reaches_u0=h['flow_p3']>=fallback['flow_p3'],tail_vs_u0=h['epe_gt32']<=fallback['epe_gt32'])
        checks[v]=dict(control=b,checks=gates,passed=all(gates.values()))
        if all(gates.values()):eligible.append(v)
    candidate=max(eligible,key=lambda v:rows[v]['hangzhou']['overall']['flow_p3']) if eligible else None
    # If within 0.1pp, prefer unchanged legacy architecture.
    if candidate and 'D01' in eligible and rows[candidate]['hangzhou']['overall']['flow_p3']-rows['D01']['hangzhou']['overall']['flow_p3']<=.001:candidate='D01'
    control=checks[candidate]['control'] if candidate else None
    result=dict(configuration_locked=True,candidate=candidate,control=control,checks=checks,
        fallback='U0_epoch30',confirmation_seeds=[43,44] if candidate else [],
        reason='Candidate passes all development gates; confirm before opening sealed tests.' if candidate else 'No candidate passes registered gates; keep U0, no additional seed training, stop architecture expansion.',
        interpretation='For D01 vs D00, data-substitution benefit must pass the same conservative two-domain +1pp gate. Fixed B0 features in all D runs; confirmation seeds concern coordinate training only.')
    write(ROOT/'decision.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
