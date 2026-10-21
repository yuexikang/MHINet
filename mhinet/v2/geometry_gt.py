"""Single-pair geometry labels in normalized xy, align_corners=False.

Dense fields contain absolute B coordinates on the A pixel-center lattice,
not displacement. ``known`` distinguishes missing labels from known occlusion.
Sparse queries only match annotated source positions; no spatial extrapolation.
"""
from dataclasses import dataclass, fields

import torch

from mhinet.ops.geometry import safe_project_points
from mhinet.pretraining.loss import sample
from mhinet.downstream.supervision import valid_points, to_norm, to_uv, offsets
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers


@dataclass
class GeometryGT:
    kind: str
    hw_a: tuple = (512, 512)
    hw_b: tuple = (512, 512)
    H: torch.Tensor | None = None
    map_ab: torch.Tensor | None = None
    map_ba: torch.Tensor | None = None
    known_a: torch.Tensor | None = None
    known_b: torch.Tensor | None = None
    visible_a: torch.Tensor | None = None
    visible_b: torch.Tensor | None = None
    points_a: torch.Tensor | None = None
    points_b: torch.Tensor | None = None
    point_visible: torch.Tensor | None = None
    mask_a: torch.Tensor | None = None
    mask_b: torch.Tensor | None = None
    exact: bool = True
    tolerance_px: float = 0.

    def __post_init__(self):
        if self.kind not in ('homography', 'dense_flow', 'sparse_points', 'unlabeled'):
            raise ValueError(f'Unknown geometry kind: {self.kind}')

    def to(self, device):
        return GeometryGT(**{f.name: getattr(self, f.name).to(device)
            if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name)
            for f in fields(self)})

    @property
    def has_reverse(self):
        return self.kind in ('homography', 'sparse_points') or self.map_ba is not None

    def query(self, points, *, reverse=False):
        """Return (target xy, known annotation, visible correspondence), same shape."""
        shape = points.shape
        p = points.reshape(-1, 2).float()
        target = torch.zeros_like(p)
        known = torch.zeros(len(p), dtype=torch.bool, device=p.device)
        visible = known.clone()
        if self.kind == 'homography':
            H = torch.linalg.inv(self.H.float()) if reverse else self.H.float()
            target, legal, _ = safe_project_points(H, p)
            known = torch.isfinite(p).all(-1)
            visible = known & legal
        elif self.kind == 'dense_flow':
            mapping = self.map_ba if reverse else self.map_ab
            annotated = self.known_b if reverse else self.known_a
            visibility = self.visible_b if reverse else self.visible_a
            if mapping is not None:
                if mapping.ndim != 3 or mapping.shape[0] != 2:
                    raise ValueError('Dense mapping must be 2xHxW absolute normalized xy')
                finite = torch.isfinite(mapping).all(0)
                labels = finite if annotated is None else finite & annotated.reshape(finite.shape).bool()
                safe_p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
                target = sample(torch.where(labels[None], mapping, torch.zeros_like(mapping)), safe_p)
                known = valid_points(labels.float(), p)
                visible = known.clone()
                if visibility is not None:
                    visible &= valid_points(visibility.float(), p)
        elif self.kind == 'sparse_points':
            source, destination = (self.points_b, self.points_a) if reverse else (self.points_a, self.points_b)
            if source is not None and len(source):
                # Exact source coordinates, allowing only float conversion roundoff.
                # Never assign a neighbouring annotation's displacement to a query.
                for start in range(0, len(p), 1024):
                    distance = (p[start:start+1024, None]-source[None]).abs().amax(-1)
                    annotated_finite = torch.isfinite(source).all(-1) & torch.isfinite(destination).all(-1)
                    distance = distance.masked_fill(~annotated_finite[None], float('inf'))
                    delta, index = distance.min(-1)
                    annotated = delta <= 1e-6
                    target[start:start+1024] = destination[index]
                    known[start:start+1024] = annotated
                    good = annotated & torch.isfinite(destination[index]).all(-1)
                    if self.point_visible is not None:
                        good &= self.point_visible[index].bool()
                    visible[start:start+1024] = good
        source_hw, target_hw = (self.hw_b, self.hw_a) if reverse else (self.hw_a, self.hw_b)
        source_mask, target_mask = (self.mask_b, self.mask_a) if reverse else (self.mask_a, self.mask_b)
        def inside(x, hw):
            uv = to_uv(x, hw)
            return torch.isfinite(x).all(-1) & (uv >= 0).all(-1) & (uv <= uv.new_tensor([hw[1]-1, hw[0]-1])).all(-1)
        visible &= inside(p, source_hw) & inside(target, target_hw)
        if source_mask is not None:
            visible &= valid_points(source_mask, p)
        if target_mask is not None:
            visible &= valid_points(target_mask, target)
        target = torch.where((known & torch.isfinite(target).all(-1))[:, None], target, torch.zeros_like(target))
        return target.reshape(shape), known.reshape(shape[:-1]), visible.reshape(shape[:-1])

    def sources(self, hw, device):
        if self.kind == 'sparse_points':
            source = self.points_a.to(device)
            _, known, visible = self.query(source)
            return source[known & visible]
        if self.kind == 'unlabeled':
            return torch.empty((0, 2), device=device)
        grid = normalized_cell_centers(*hw, device=device).reshape(-1, 2)
        _, known, visible = self.query(grid)
        return grid[known & visible]


def as_geometry(value, mask_a=None, mask_b=None):
    if isinstance(value, GeometryGT):
        return value
    return GeometryGT('homography', H=value, mask_a=mask_a, mask_b=mask_b)


@torch.no_grad()
def qrru_targets(a_uv, b_uv, geometry, mask_a=None, mask_b=None, hw=(256, 256)):
    gt = as_geometry(geometry, mask_a, mask_b)
    if gt.kind == 'homography' and gt.exact:
        # Preserve the historical exact-H formula, rounding and boundary behavior.
        from mhinet.downstream.supervision import qrru_targets as exact_h_targets
        ma = gt.mask_a if gt.mask_a is not None else a_uv.new_ones((1, *gt.hw_a))
        mb = gt.mask_b if gt.mask_b is not None else a_uv.new_ones((1, *gt.hw_b))
        return exact_h_targets(a_uv, b_uv, gt.H, ma, mb, hw)
    center, known, visible = gt.query(to_norm(a_uv, hw))
    center = to_uv(center, hw)
    controls_a = a_uv[:, None, None]+offsets(a_uv.device, True)
    target, control_known, control_visible = gt.query(to_norm(controls_a, hw))
    valid = control_known & control_visible
    # A sparse center label never licenses its four surrounding control labels.
    if gt.kind == 'sparse_points' or not gt.exact:
        valid = torch.zeros_like(valid)
    flow = to_uv(target, hw)-(b_uv[:, None, None]+offsets(a_uv.device, True))
    return flow.permute(0, 3, 1, 2), valid, center, known & visible
