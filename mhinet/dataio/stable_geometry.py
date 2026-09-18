"""Scale-invariant, bidirectional full-image projective safety checks."""
import numpy as np


POLICY = dict(min_horizon_distance=0.1, max_centered_extent=2.5,
              max_jacobian_bound=10.0)


def check_geometry(H, size_a, size_b, policy=None):
    policy = POLICY if policy is None else policy
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or not np.max(np.abs(H)):
        return False, {'reason': 'invalid_matrix'}
    H = H / np.max(np.abs(H))
    if min(*size_a, *size_b) <= 1:
        return False, {'reason': 'invalid_size'}
    A = np.diag([size_a[0]-1, size_a[1]-1, 1.])
    B = np.diag([1/(size_b[0]-1), 1/(size_b[1]-1), 1.])
    forward = B @ H @ A
    try:
        backward = np.linalg.inv(forward)
    except np.linalg.LinAlgError:
        return False, {'reason': 'singular_matrix'}
    corners = np.array([[0.,0,1], [1,0,1], [1,1,1], [0,1,1]])
    records = []
    for matrix in (forward, backward):
        matrix = matrix / np.max(np.abs(matrix))
        projected = corners @ matrix.T
        z = projected[:, 2]
        if not np.isfinite(projected).all() or not (np.all(z>1e-12) or np.all(z<-1e-12)):
            return False, {'reason': 'horizon_crossing_or_zero'}
        # Division happens only after checking the whole rectangular domain.
        xy = projected[:, :2] / z[:, None]
        normal = np.linalg.norm(matrix[2, :2])
        distance = float(np.min(np.abs(z))/normal) if normal else None
        extent = float(np.max(np.abs(xy-.5)))
        # Each Jacobian numerator is affine: its Frobenius norm is bounded
        # by the maximum at corners. min |z| is also attained at a corner.
        numerators = (matrix[:2,:2][None]*z[:,None,None]
                      - projected[:,:2,None]*matrix[2,:2][None,None,:])
        bound = float(np.linalg.norm(numerators, axis=(1,2)).max()/np.min(np.abs(z))**2)
        records.append(dict(horizon_distance=distance, centered_extent=extent,
                            jacobian_upper_bound=bound))
    accepted = all((r['horizon_distance'] is None or r['horizon_distance']>=policy['min_horizon_distance'])
                   and r['centered_extent']<=policy['max_centered_extent']
                   and r['jacobian_upper_bound']<=policy['max_jacobian_bound'] for r in records)
    return accepted, dict(policy=policy, directions=records, accepted=accepted)
