"""Validation-only source/target overlays in target pixel coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from .geometry import normalized_homography_to_pixel


def _project(h, corners):
    if not np.isfinite(h).all():
        return None
    try:
        if np.linalg.cond(h) > 1e12:
            return None
    except np.linalg.LinAlgError:
        return None
    q = np.concatenate((corners, np.ones((4, 1))), axis=1) @ h.T
    z = q[:, 2]
    if not np.isfinite(q).all() or np.any(np.abs(z) < 1e-8) or not (np.all(z > 0) or np.all(z < 0)):
        return None
    return q[:, :2] / z[:, None]


def write_iteration_overlays(images, h_gt, updates, schedule, output_dir, *, pair_id,
                             accepted=None, ghim_valid=True, h0=None):
    """Write one PNG per accepted-state H, including retained states on rejection.

    A is warped by the predicted A-to-B H onto a canvas in B's coordinates.
    The green polygon is H_gt(A corners), red is H_pred(A corners). No GT
    image warp is used to manufacture a visually better predicted overlay.
    """
    if len(updates) != len(schedule) or (accepted is not None and len(accepted) != len(schedule)):
        raise ValueError("Every scheduled iteration requires an H and acceptance flag")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rgb = (images.detach().cpu().float().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    height, width = rgb.shape[1:3]
    corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float64)

    def pixel_h(h):
        return normalized_homography_to_pixel(h.detach().cpu().double().reshape(1, 3, 3),
                                             (height, width), (height, width))[0].numpy()

    gt = pixel_h(h_gt)
    gt_points = _project(gt, corners)
    if gt_points is not None and np.abs(gt_points).max() >= 1e6:
        gt_points = None
    records = []
    scale_rounds = {}
    states = [(index, h, scale) for index, (h, scale) in enumerate(zip(updates, schedule))]
    if h0 is not None:
        states.insert(0, (-1, h0, None))
    for index, h, scale in states:
        predicted = pixel_h(h)
        points = _project(predicted, corners)
        safe = points is not None and np.abs(points).max() < 1e6
        polygons = [corners]
        if gt_points is not None and np.abs(gt_points).max() < 1e6:
            polygons.append(gt_points)
        if safe:
            polygons.append(points)
        all_points = np.concatenate(polygons)
        lo, hi = all_points.min(axis=0)-16, all_points.max(axis=0)+16
        factor = min(1., 1800. / float((hi-lo).max()))
        shift = np.array([[factor, 0, -lo[0]*factor], [0, factor, -lo[1]*factor], [0, 0, 1.]])
        canvas_size = tuple(np.maximum(1, np.ceil((hi-lo)*factor).astype(int)))
        base = cv2.warpPerspective(rgb[1], shift, canvas_size)
        mask_b = cv2.warpPerspective(np.ones((height, width), np.uint8), shift, canvas_size, flags=cv2.INTER_NEAREST) > 0
        if safe:
            warp = cv2.warpPerspective(rgb[0], shift @ predicted, canvas_size)
            mask_a = cv2.warpPerspective(np.ones((height, width), np.uint8), shift @ predicted, canvas_size, flags=cv2.INTER_NEAREST) > 0
            both = mask_a & mask_b
            base[mask_a & ~mask_b] = warp[mask_a & ~mask_b]
            base[both] = ((base[both].astype(np.float32) + warp[both]) * .5).astype(np.uint8)
        for polygon, color, thickness in [(gt_points, (0, 255, 0), 4), (points if safe else None, (255, 0, 0), 2)]:
            if polygon is not None:
                draw_points = np.rint((polygon-lo)*factor).astype(np.int32)
                cv2.polylines(base, [draw_points], True, color, thickness, cv2.LINE_AA)
        scale_rounds[scale] = scale_rounds.get(scale, 0) + 1
        accepted_state = bool(accepted[index]) if accepted is not None and index >= 0 else None
        label = "GHIM initialization / H0" if index < 0 else f"D{scale} round {scale_rounds[scale]} / H{index+1}"
        title = f"{label} | green=GT red=prediction"
        canvas = Image.new("RGB", (base.shape[1], base.shape[0]+44), "black")
        canvas.paste(Image.fromarray(base), (0, 44))
        painter = ImageDraw.Draw(canvas)
        painter.text((8, 5), title, fill="white")
        painter.text((8, 22), f"GHIM valid={ghim_valid} update accepted={accepted_state} warp valid={safe}", fill="white")
        filename = "H0_initialization.png" if index < 0 else f"D{scale}_round{scale_rounds[scale]}_H{index+1:02d}.png"
        path = directory / filename
        canvas.save(path)
        records.append({"path": str(path.resolve()), "scale": None if scale is None else int(scale), "iteration": index+1,
                        "update_accepted": accepted_state, "warp_valid": bool(safe),
                        "H_pred_pixel": predicted.tolist(), "H_gt_pixel": gt.tolist()})
    metadata = {"pair_id": pair_id, "ghim_valid": bool(ghim_valid),
                "coordinate_system": "target input pixels; A warped by prediction, GT green, prediction red",
                "images": records}
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
