"""Geometry for semidense supervision. Coordinates are normalized unless named uv."""
import torch
import torch.nn.functional as F
from mhinet.pretraining.loss import sample
from .loma_reference.matching_utils import normalized_cell_centers, project_normalized


def to_uv(points, hw):
    h, w = hw
    return (points + 1) * points.new_tensor([w, h]) / 2 - .5


def to_norm(points, hw):
    h, w = hw
    return 2 * (points + .5) / points.new_tensor([w, h]) - 1


def sample_uv(feature, points):
    shape = points.shape[:-1]
    return sample(feature, to_norm(points.reshape(-1, 2), feature.shape[-2:])).reshape(*shape, feature.shape[0])


def valid_points(mask, points):
    """Conservative bilinear validity; points outside cell centers are invalid."""
    h, w = mask.shape[-2:]
    uv = to_uv(points, (h, w))
    inside = torch.isfinite(points).all(-1) & (uv >= 0).all(-1) & (uv <= uv.new_tensor([w-1, h-1])).all(-1)
    safe = torch.where(inside[..., None], points, torch.zeros_like(points))
    return inside & (sample(mask.reshape(1, h, w), safe.reshape(-1, 2))[:, 0].reshape(inside.shape) > .999)


def legal_h(H):
    if not bool(torch.isfinite(H).all()) or float(H.abs().max()) == 0:
        return False
    H = H.float() / H.abs().max()
    s = torch.linalg.svdvals(H)
    if bool(s[-1] <= s[0] * 1e-10):
        return False
    corners = H.new_tensor([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    z = corners @ H[2]
    return bool((z > 1e-8).all() or (z < -1e-8).all())


@torch.no_grad()
def coarse_labels(H, mask_a, mask_b, hw):
    """Nearest-cell correspondences with inverse roundtrip and mask checks."""
    if not legal_h(H):
        raise ValueError('Invalid GT homography')
    h, w = hw
    grid = normalized_cell_centers(h, w, device=H.device).reshape(-1, 2)
    target, ok = project_normalized(grid, H)
    uv = to_uv(target, hw).round().long()
    inside = (uv >= 0).all(-1) & (uv < uv.new_tensor([w,h])).all(-1)
    j = uv[:, 1].clamp(0,h-1)*w + uv[:, 0].clamp(0,w-1)
    back, back_ok = project_normalized(grid[j], torch.linalg.inv(H.float()))
    back_uv = to_uv(back, hw).round().long()
    source_uv = to_uv(grid, hw).round().long()
    good = ok & inside & back_ok & (back_uv == source_uv).all(-1)
    good &= valid_points(mask_a, grid) & valid_points(mask_b, target) & valid_points(mask_b, grid[j])
    i = good.nonzero().flatten()
    return i, j[i]


def fine_grids(i, j, H0, coarse_hw, fine_hw):
    """Exact D8 child lattice and H-residual target lattice."""
    hc, wc = coarse_hw
    hf, wf = fine_hw
    if (hf, wf) != (hc*4, wc*4):
        raise ValueError('Require D8→D2 scale ratio four')
    grid = normalized_cell_centers(hc,wc,device=i.device).reshape(-1,2)
    off = torch.arange(4,device=i.device)
    oy, ox = torch.meshgrid(off,off,indexing='ij')
    uv = torch.stack(((i%wc)[:,None]*4+ox.flatten(), (i//wc)[:,None]*4+oy.flatten()),-1).float()
    a = to_norm(uv, fine_hw)
    ah, _ = project_normalized(grid[i], H0)
    projected, _ = project_normalized(a.reshape(-1,2), H0)
    b = projected.reshape_as(a) + (grid[j]-ah)[:,None]
    return a,b


@torch.no_grad()
def fine_labels(a, b, H, mask_a, mask_b, fine_hw, tolerance=1.5):
    """Mutual nearest GT geometry on the actual warped target lattice."""
    shape = a.shape[:-1]
    truth, ok = project_normalized(a.reshape(-1,2),H)
    truth = truth.reshape_as(a)
    va = valid_points(mask_a,a) & ok.reshape(shape) & valid_points(mask_b,truth)
    vb = valid_points(mask_b,b)
    dist = torch.cdist(to_uv(truth,fine_hw),to_uv(b,fine_hw))
    dist = dist.masked_fill(~(va[:,:,None] & vb[:,None,:]),float('inf'))
    nearest = dist.argmin(-1)
    reverse = dist.argmin(-2)
    ids = torch.arange(a.shape[1],device=a.device)[None].expand(shape)
    good = (dist.gather(-1,nearest[...,None]).squeeze(-1) <= tolerance)
    good &= reverse.gather(-1,nearest)==ids
    # An independent inverse projection rejects nearest-neighbour ambiguities.
    back, ok_back = project_normalized(b.reshape(-1,2),torch.linalg.inv(H.float()))
    back = to_uv(back.reshape_as(b),fine_hw)
    source = to_uv(a,fine_hw)
    inverse_dist = torch.cdist(source,back).masked_fill(~(va[:,:,None] & vb[:,None,:]),float('inf'))
    good &= inverse_dist.argmin(-2).gather(-1,nearest)==ids
    good &= ok_back.reshape(shape).gather(-1,nearest)
    m,k = good.nonzero(as_tuple=True)
    return m,k,nearest[m,k],va,vb


def offsets(device, control=False):
    axis = torch.tensor([-1.,1.] if control else [-1.5,-.5,.5,1.5],device=device)
    y,x = torch.meshgrid(axis,axis,indexing='ij')
    return torch.stack((x,y),-1)


def interpolate_controls(flow, positions):
    """Bilinear polynomial on actual ±1 quadrant centers, including extrapolation."""
    x,y = positions.unbind(-1)
    weights = torch.stack(((1-x)*(1-y),(1+x)*(1-y),(1-x)*(1+y),(1+x)*(1+y)),-1)/4
    return torch.einsum('...k,mck->m...c',weights,flow.flatten(2))


@torch.no_grad()
def qrru_targets(a_uv, b_uv, H, mask_a, mask_b, hw):
    control_a = a_uv[:,None,None,:]+offsets(a_uv.device,True)
    target, ok = project_normalized(to_norm(control_a,hw).reshape(-1,2),H)
    target = target.reshape_as(control_a)
    valid = ok.reshape(control_a.shape[:-1]) & valid_points(mask_a,to_norm(control_a,hw)) & valid_points(mask_b,target)
    flow = to_uv(target,hw)-(b_uv[:,None,None,:]+offsets(a_uv.device,True))
    center, legal = project_normalized(to_norm(a_uv,hw),H)
    center_valid = legal & valid_points(mask_a,to_norm(a_uv,hw)) & valid_points(mask_b,center)
    return flow.permute(0,3,1,2), valid, to_uv(center,hw), center_valid
