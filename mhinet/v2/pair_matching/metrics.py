"""Evaluation uses geometry externally; inference has no GT dependency."""

def measure(a,b,gt):
    target,known,visible = gt.query((a+.5)/256-1)
    good = known & visible
    error = ((target+1)*256-.5-b).norm(dim=-1)
    result = dict(n=len(a), visible=int(good.sum()), error_sum=float(error[good].sum()))
    result.update({f'correct{t}':int((good & (error<=t)).sum()) for t in (1,3,5,10)})
    return result, error, good




def totals(records):
    result = {k: sum(r[k] for r in records) for k in ('n', 'visible', 'error_sum', 'correct1', 'correct3', 'correct5', 'correct10')}
    result.update(epe=result['error_sum']/max(1, result['visible']),
                  p3_visible=result['correct3']/max(1, result['visible']))
    result.update({f'p{t}': result[f'correct{t}']/max(1, result['n']) for t in (1, 3, 5, 10)})
    return result
