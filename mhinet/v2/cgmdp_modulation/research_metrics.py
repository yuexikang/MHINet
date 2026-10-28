"""Same-window descriptor diagnostics and rematched end-to-end correspondence metrics."""
import torch
import torch.nn.functional as F
from mhinet.v2.d2_modulation.model import center_indices, match_windows
from mhinet.v2.d2_modulation.experiment import counts
from mhinet.v2.joint_single.path import sample, coarse


def resample(shared, record):
    source = record['source'].mean(1)
    axis = (torch.arange(8, device=source.device)-3.5)*2
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    points = source[:, None]+torch.stack((x, y), -1).reshape(1, 64, 2)
    d2 = shared['pyramid'][2][0].float()
    return dict(record, a=sample(d2[0], points), b=sample(d2[1], record['candidates']))


def labels(record, gt):
    truth, known, visible = gt.query((record['source']+.5)/256-1)
    return dict(record, truth=(truth+1)*256-.5, labeled=known & visible)


def spatial_similarity(features, valid):
    f = F.normalize(features, dim=-1)
    mask = valid[:, :, None] & valid[:, None, :]
    mask &= ~torch.eye(f.shape[1], device=f.device, dtype=torch.bool)[None]
    return float((f @ f.transpose(-1, -2))[mask].mean()) if mask.any() else 0.


def quality(adapter, record):
    a, b = adapter(record['a'], record['b'], record['valid_a'], record['valid_b'])
    ia = center_indices(a.device)
    similarity = F.normalize(a[:, ia], dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)
    positive = (record['candidates'][:, None]-record['truth'][:, :, None]).norm(dim=-1)<=3
    positive &= record['valid_b'][:, None]
    negative = ~positive & record['valid_b'][:, None]
    usable = record['labeled'] & positive.any(-1) & negative.any(-1)
    margin = similarity.masked_fill(~positive, -torch.inf).amax(-1)-similarity.masked_fill(~negative, -torch.inf).amax(-1)
    stats = dict(similarity_margin=float(margin[usable].mean()) if usable.any() else 0.,
                 margin_queries=int(usable.sum()),
                 spatial_cos_before=.5*(spatial_similarity(record['a'], record['valid_a'])+spatial_similarity(record['b'], record['valid_b'])),
                 spatial_cos_after=.5*(spatial_similarity(a, record['valid_a'])+spatial_similarity(b, record['valid_b'])))
    if adapter.mode != 'none':
        sa, sb = adapter.mean(record['a'], record['valid_a']), adapter.mean(record['b'], record['valid_b'])
        own, other = torch.cat((sa, sb)), torch.cat((sb, sa))
        gamma, beta = adapter.condition(torch.cat((own, other, (own-other).abs()), -1)).chunk(2, -1)
        norms = torch.cat([((record[k].norm(dim=-1)*record['valid_'+k]).sum(-1)/record['valid_'+k].sum(-1).clamp_min(1)) for k in ('a', 'b')])
        stats.update(pair_gamma_abs=float(gamma.abs().mean()), pair_beta_ratio=float((beta.norm(dim=-1)/norms.clamp_min(1e-8)).mean()))
    return stats


def comparison(record, result, baseline, baseline_result):
    def keys(r):
        xy = r['source'].reshape(-1, 2)
        return (xy[:, 1]+.5).long()*512+(xy[:, 0]+.5).long()
    before_keys, after_keys = keys(baseline), keys(record)
    sorted_keys, order = before_keys.sort()
    index = torch.searchsorted(sorted_keys, after_keys).clamp(max=len(sorted_keys)-1)
    common = sorted_keys[index]==after_keys
    index = order[index]
    before_error = (baseline_result['points_b']-baseline['truth']).norm(dim=-1).flatten()
    after_error = (result['points_b']-record['truth']).norm(dim=-1).flatten()
    usable = common & baseline['labeled'].flatten()[index] & record['labeled'].flatten()
    good = before_error[index]<=3
    after_good = after_error<=3
    return dict(common=int(common.sum()), added=int((~common).sum()), lost=int(len(before_keys)-common.sum()),
                before_good=int((usable&good).sum()), before_bad=int((usable&~good).sum()),
                damaged=int((usable&good&~after_good).sum()), rescued=int((usable&~good&after_good).sum()))


def metrics(shared, matcher, adapter, record, result, gt, baseline, baseline_result):
    c = counts(record, result)
    r = coarse(shared, matcher, record['source'].mean(1))
    truth, known, visible = gt.query((r['source']+.5)/256-1)
    truth = (truth+1)*256-.5; valid = known&visible
    e = (r['p8']-truth).norm(dim=-1)
    # D8 spacing is 8 px: half-cell diagonal is the geometric coverage tolerance.
    covered = ((r['candidates8']-truth[:, None]).norm(dim=-1)<=8/2**.5) & r['valid8']
    return dict(d2=c, d8=dict(n=len(e), visible=int(valid.sum()), covered=int((valid&covered.any(-1)).sum()),
                             **{f'correct{k}':int((valid&(e<=k)).sum()) for k in (3,5,10)}),
                changes=comparison(record, result, baseline, baseline_result), quality=quality(adapter, record))


def aggregate(rows):
    def total(section): return {k:sum(r[section][k] for r in rows) for k in rows[0][section]}
    d2, d8, changes = [total(k) for k in ('d2', 'd8', 'changes')]
    d2.update({f'p{k}':d2[f'correct{k}']/max(1,d2['n']) for k in (1,3,5,10)})
    d2.update(epe=d2['error_sum']/max(1,d2['visible']), coverage=d2['covered3']/max(1,d2['visible']),
              rank1_rate=d2['rank1']/max(1,d2['covered3']), rank5_rate=d2['rank5']/max(1,d2['covered3']))
    d8.update({f'p{k}':d8[f'correct{k}']/max(1,d8['n']) for k in (3,5,10)})
    d8['coverage'] = d8['covered']/max(1,d8['visible'])
    changes.update(damaged_rate=changes['damaged']/max(1,changes['before_good']), rescued_rate=changes['rescued']/max(1,changes['before_bad']))
    q = {k:sum(r['quality'][k] for r in rows)/len(rows) for k in rows[0]['quality']}
    return dict(d2=d2, d8=d8, changes=changes, quality_pair_mean=q)
