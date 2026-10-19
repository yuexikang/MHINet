"""Explicit Flow-head/QRRU ownership and reproducible mixed QRRU inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mhinet.downstream.supervision import coarse_labels, qrru_targets, to_uv
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers, project_normalized
from .losses import flow_head_loss


class SemidenseSystemV2(nn.Module):
    def __init__(self, shared, matcher, stage="flow_head"):
        super().__init__()
        self.shared, self.matcher = shared, matcher
        self.configure_stage(stage)

    def configure_stage(self, stage):
        if stage not in ("flow_head", "qrru"):
            raise ValueError("Expected flow_head or qrru training stage")
        self.stage = stage
        self.shared.set_training_groups(mvt=False, vgg=False, dedode=False, ghim_head=stage == "flow_head")
        self.matcher.requires_grad_(False)
        if stage == "qrru":
            self.matcher.qrru.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.shared.train(mode if self.stage == "flow_head" else False)
        self.matcher.train(mode if self.stage == "qrru" else False)
        return self

    def forward(self, images):
        if images.ndim != 5 or images.shape[1:] != (2, 3, 512, 512):
            raise ValueError("MHI v2 requires Bx2x3x512x512")
        if self.stage == "flow_head":
            return self.shared(images, pyramid_scales=())
        with torch.no_grad():
            return self.shared(images)

    def optimizer_groups(self, learning_rate):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("Empty training group")
        return [dict(name=self.stage, params=parameters, lr=learning_rate)]

    def training_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]


def assert_frozen_gradients(system):
    bad = [name for name, p in system.named_parameters() if not p.requires_grad and p.grad is not None]
    if bad:
        raise RuntimeError(f"Frozen parameters received gradients: {bad[:5]}")


def mixed_qrru_loss(system, shared, tensors, pair_id, *, predicted_fraction=.5,
                    correction_limit_d2=6., cache_root=None, source_identity=""):
    """GT filters labels only. Predicted D2 correspondences never receive GT in infer."""
    config = system.matcher.config
    import math
    if (isinstance(predicted_fraction, bool) or not isinstance(predicted_fraction, (int, float))
            or not math.isfinite(predicted_fraction) or not 0 <= predicted_fraction <= 1
            or isinstance(correction_limit_d2, bool) or not isinstance(correction_limit_d2, (int, float))
            or not math.isfinite(correction_limit_d2) or not 0 < correction_limit_d2 <= 6):
        raise ValueError("Invalid QRRU input mixture/correction range")
    if cache_root and not source_identity:
        raise ValueError("D2 cache requires an explicit source identity")
    H = tensors["H_gt_norm"][0]
    ma, mb = tensors["mask_A_overlap"][0], tensors["mask_B_overlap"][0]
    d2 = shared["pyramid"][2][0].float()
    hw = d2.shape[-2:]
    with torch.no_grad():
        cache_file = None
        if cache_root:
            cache_file = Path(cache_root)/f"{hashlib.sha256((source_identity+pair_id).encode()).hexdigest()}.npz"
        if cache_file and cache_file.is_file():
            with np.load(cache_file, allow_pickle=False) as cached:
                pa_pred = torch.from_numpy(cached["a"].copy()).to(d2.device)
                pb_pred = torch.from_numpy(cached["b"].copy()).to(d2.device)
        else:
            prediction = system.matcher.infer(shared, refine=False)[0]
            # The uncapped candidate fields precede QRRU and do not depend on its weights.
            aa = prediction.get("fine_all_a", d2.new_empty((0, 2)))
            bb = prediction.get("fine_all_b", d2.new_empty((0, 2)))
            confidence = prediction.get("fine_all_confidence", d2.new_empty((0,)))
            keep = torch.argsort(confidence, descending=True, stable=True)[:512]
            pa_pred = (aa[keep]+.5)/2-.5
            pb_pred = (bb[keep]+.5)/2-.5
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_file.with_suffix(".temporary.npz")
                np.savez_compressed(temporary, a=pa_pred.cpu().numpy(), b=pb_pred.cpu().numpy())
                temporary.replace(cache_file)
        if len(pa_pred):
            controls, control_valid, truth, visible = qrru_targets(pa_pred, pb_pred, H, ma, mb, hw)
            eligible = visible & ((pb_pred-truth).norm(dim=-1) <= correction_limit_d2)
            eligible &= ((controls.abs() <= correction_limit_d2) | ~control_valid[:, None]).flatten(1).all(1)
            keep = eligible.nonzero().flatten()
            keep = keep[torch.randperm(len(keep), device=d2.device)[:round(config.qrru_queries*predicted_fraction)]]
            pa_pred, pb_pred = pa_pred[keep], pb_pred[keep]
        count_noise = config.qrru_queries-len(pa_pred)
        i, _ = coarse_labels(H, ma, mb, hw)
        take = i[torch.randperm(len(i), device=i.device)[:count_noise]]
        grid = normalized_cell_centers(*hw, device=d2.device).reshape(-1, 2)
        pa_noise = to_uv(grid[take], hw)
        truth, _ = project_normalized(grid[take], H)
        pb_noise = to_uv(truth, hw)+torch.randn_like(pa_noise).mul(config.noise_std).clamp(-config.noise_bound, config.noise_bound)
        pa = torch.cat((pa_noise, pa_pred), 0); pb = torch.cat((pb_noise, pb_pred), 0)
        target, valid, center, center_valid = qrru_targets(pa, pb, H, ma, mb, hw)
    if not len(pa):
        raise ValueError("No usable QRRU training inputs")
    projected = system.matcher.qrru.project(d2)
    total = projected.sum()*0
    control_count, center_count = max(1, int(valid.sum())), max(1, int(center_valid.sum()))
    errors_before, errors_after = [], []
    for start in range(0, len(pa), config.window_chunk):
        end = start+config.window_chunk
        result = system.matcher.qrru(projected[0], projected[1], pa[start:end], pb[start:end], projected=True)
        # Use separate full-batch control/center normalization, retaining live losses.
        vc, vv = valid[start:end], center_valid[start:end]
        def average(error, mask):
            selected = error[mask]
            # Index before the penalty so invalid GT cannot form 0*NaN grads.
            return (torch.sqrt(selected.square()+1e-6)-.001).mean() if selected.numel() else selected.sum()
        for t in range(config.iterations):
            weight = config.gamma**(config.iterations-1-t)
            total = total+weight*(int(vc.sum())/control_count*average((result["flows"][t]-target[start:end]).permute(0, 2, 3, 1), vc)
                                 +int(vv.sum())/center_count*average(result["centers"][t]-center[start:end], vv))
        if bool(vv.any()):
            errors_before.append((pb[start:end]-center[start:end]).norm(dim=-1)[vv].detach())
            errors_after.append((result["centers"][-1]-center[start:end]).norm(dim=-1)[vv].detach())
    before = torch.cat(errors_before) if errors_before else d2.new_empty(0)
    after = torch.cat(errors_after) if errors_after else d2.new_empty(0)
    return total, dict(qrru_queries=len(pa), predicted_queries=len(pa_pred), noise_queries=len(pa_noise),
                       initial_epe_d2=float(before.mean()) if len(before) else None,
                       final_epe_d2=float(after.mean()) if len(after) else None,
                       improved_fraction=float((after < before).float().mean()) if len(after) else None,
                       damaged_fraction=float((after > before).float().mean()) if len(after) else None)


def batch_loss(system, batch, device, training_config):
    tensors = {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    shared = system(tensors["images"])
    if system.stage == "flow_head":
        loss, metrics = flow_head_loss(shared["stage1"], tensors["H_gt_norm"], tensors["mask_A_overlap"], tensors["mask_B_overlap"])
    else:
        pair_identity = batch["pair_id"][0]+"|"+batch.get("parent_group", [""])[0]
        loss, metrics = mixed_qrru_loss(system, shared, tensors, pair_identity,
            predicted_fraction=training_config.get("predicted_fraction", .5),
            correction_limit_d2=training_config.get("correction_limit_d2", 6.),
            cache_root=training_config.get("d2_cache"), source_identity=training_config.get("cache_source_identity", ""))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite v2 loss")
    return loss, metrics
