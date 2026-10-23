"""Conservative full-cell labels in native 512 pixel coordinates."""
import numpy as np
import torch
from mhinet.v2.d16_experiment.supervision import grid
from mhinet.v2.geometry_gt import GeometryGT
from mhinet.ops.geometry import pixel_homography_to_normalized

MATCH, NO_MATCH, UNKNOWN = 0, 1, 2


def project(points, H):
    q = np.concatenate((points, np.ones((*points.shape[:-1], 1))), -1) @ np.asarray(H).T
    return q[..., :2] / q[..., 2:]


def support_labels(H, rectangle=None, reverse=False):
    """Images are fully valid. Boundary/intersection ambiguity is ignored.

    A transformed cell is a convex quadrilateral. Its bounding box being
    disjoint is sufficient for a certain negative; other boundary cells are
    ignored conservatively, never asserted absent from a centre alone.
    Rectangle is B pixel index range [x0,y0,x1,y1), converted to pixel edges.
    """
    source = grid(torch.device('cpu')).numpy()
    corners = source[:, None] + np.array([[-8,-8],[8,-8],[8,8],[-8,8]])
    H = np.linalg.inv(H) if reverse else np.asarray(H)
    mapped = project(corners, H)
    low, high = mapped.min(1), mapped.max(1)
    positive = (low >= -.5-1e-7).all(-1) & (high <= 511.5+1e-7).all(-1)
    negative = ((high <= -.5).any(-1) | (low >= 511.5).any(-1))
    base_positive, base_negative = positive.copy(), negative.copy()
    reason = np.zeros(1024, np.int8)
    reason[negative] = 1  # known geometric outside
    if rectangle is not None:
        a = np.asarray(rectangle[:2])-.5
        b = np.asarray(rectangle[2:])-.5
        rect_low, rect_high = (corners.min(1), corners.max(1)) if reverse else (low, high)
        covered = (rect_low >= a).all(-1) & (rect_high <= b).all(-1)
        disjoint = (rect_high <= a).any(-1) | (rect_low >= b).any(-1)
        texture = covered if reverse else covered & positive
        positive &= disjoint
        # A source support crossing the donor boundary is unknown in reverse.
        if reverse:
            negative &= disjoint
            reason[~disjoint] = 0
        negative |= texture
        reason[texture] = 2
        if reverse:
            # A donor patch outside the original overlap is still NO_MATCH,
            # but is not evidence of solving the harder texture-only task.
            reason[texture & base_negative] = 1
            reason[texture & ~(base_positive | base_negative)] = 3
    state = np.full(1024, UNKNOWN, np.int8)
    state[positive], state[negative] = MATCH, NO_MATCH
    # Use the existing geometry interface for centre coordinates. Its visibility
    # alone is insufficient for the full-cell state, which was computed above.
    hn = pixel_homography_to_normalized(torch.tensor(H, dtype=torch.float64), (512,512), (512,512)).float()
    gt = GeometryGT('homography', H=hn)
    target, _, _ = gt.query(torch.from_numpy((source+.5)/256-1))
    target = ((target+1)*256-.5).numpy()
    target[state != MATCH] = np.nan
    return dict(state=state, negative_kind=reason, source_xy=source, gt_target_xy=target)
