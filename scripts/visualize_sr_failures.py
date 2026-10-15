#!/usr/bin/env python3
"""Show cached Full/A SR failures with many correct matches; never refit or infer."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Rectangle

REPO = Path(__file__).resolve().parents[1]
CASES = [
    ('JL1Flight', 1117, [(385, 349, 64, 64), (373, 157, 64, 64)]),
    ('JL1Flight', 80, [(371, 158, 64, 64), (32, 112, 64, 64)]),
    ('GoogleEarth', 43, [(147, 426, 64, 64), (178, 16, 64, 64)]),
    ('JL1Flight', 872, [(355, 32, 64, 64), (141, 394, 64, 64)]),
]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(points, H):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.c_[points, np.ones(len(points))] @ np.asarray(H).T
    valid = np.isfinite(q).all(1) & (np.abs(q[:, 2]) > 1e-10)
    xy = np.full_like(points, np.nan)
    xy[valid] = q[valid, :2] / q[valid, 2:]
    return xy, valid


def write_image(path, pixels):
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path)


def display_gray(image, limits):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    low, high = limits
    return np.rint(np.clip((gray-low) / max(1., high-low), 0, 1)*255).astype(np.uint8)


def red_cyan(target, registered, mask, target_limits, source_limits):
    # Display-only linear intensity windows. The same source/target windows are
    # used for both H and GT overlays; original and registered RGB stay intact.
    target_gray = display_gray(target, target_limits)
    registered_gray = display_gray(registered, source_limits)
    output = np.stack((target_gray, registered_gray, registered_gray), -1)
    output[~mask] = 42
    return output


def plot_case(folder, A, B, warped, overlay, gt_overlay, H_gt, rois, info, chinese):
    colors = ['#1765b0', '#d16016']
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.6), dpi=128)
    images = [A, B, warped, overlay]
    titles = ['原始 A（512输入）', '原始 B（512输入）', '预测 H 配准后的 A', '预测叠加（仅显示亮度归一化）'] if chinese else [
        'Original A (512 input)', 'Original B (512 input)', 'A registered by predicted H', 'Prediction overlay: B red / A cyan']
    for ax, image, title in zip(axes[0], images, titles):
        ax.imshow(image, interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
    for index, (x, y, width, height) in enumerate(rois):
        corners = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
        source_corners, valid = project(corners, np.linalg.inv(H_gt))
        assert valid.all()
        axes[0, 0].add_patch(Polygon(source_corners, fill=False, edgecolor=colors[index], linewidth=1.8))
        for ax in axes[0, 1:]:
            ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=colors[index], linewidth=1.8))
            ax.text(x + 2, y - 3, str(index + 1), color='white', fontsize=11,
                    bbox=dict(facecolor=colors[index], edgecolor='none', pad=2))
        roi = info['rois'][index]
        region = np.s_[y:y + height, x:x + width]
        zoom_images = [B[region], warped[region], overlay[region], gt_overlay[region]]
        names = ['B 原始局部', '预测 H 配准局部', '预测叠加：红B / 青A', 'GT 配准叠加（参考）'] if chinese else [
            'Original target B', 'A registered by predicted H', 'Predicted-H overlay', 'GT-H overlay (reference)']
        for col, (ax, image, name) in enumerate(zip(axes[index + 1], zoom_images, names)):
            ax.imshow(image, interpolation='nearest')
            suffix = f" | ROI {index + 1} | {width}x{height}px"
            ax.set_title(name + suffix, fontsize=10)
            ax.set_axis_off()
            if col == 0:
                ax.text(0, -0.08,
                        f"B: x={x}:{x+width}, y={y}:{y+height}\n"
                        + (f"区域内 H/GT 网格差异：{roi['pixel_grid_rmse']:.3f}px" if chinese
                           else f"Local H-vs-GT RMSE: {roi['pixel_grid_rmse']:.3f}px"),
                        transform=ax.transAxes, fontsize=10, va='top', color=colors[index])
    m = info['metrics']
    title = info['pair_id'] + '\n' + f"NCM@3={m['ncm']['3']:,} / NM={m['matches']:,} | P@3={100*m['precision']['3']:.2f}% | "
    title += (f"全域网格 RMSE={m['registration_rmse']:.3f}px > 3px：SR失败" if chinese
              else f"Grid RMSE={m['registration_rmse']:.3f}px > 3px: SR failure")
    fig.suptitle(title, fontsize=14, y=.992)
    if chinese:
        caption = '局部框均为目标 B 坐标；64×64像素放大显示。红=B，青=配准A；叠加图仅做显示亮度归一化，原图/配准RGB未改动。\n灰区表示A无覆盖。GT列仅作参考，时相差异也会改变颜色；原图A的框由GT反投影定位。'
    else:
        caption = 'Crops: target-B coordinates (64x64, enlarged). Red=B, cyan=registered A. Overlay brightness normalized for display only; RGB unchanged.\nGray=no A support. GT is a reference; temporal changes affect colors. A boxes are GT back-projections.'
    fig.text(.5, .008, caption, ha='center', va='bottom', fontsize=10)
    fig.subplots_adjust(left=.018, right=.99, bottom=.075, top=.92, wspace=.065, hspace=.28)
    fig.savefig(folder / 'comparison.png')
    plt.close(fig)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    chinese = False
    if args.font and args.font.is_file():
        font_manager.fontManager.addfont(args.font)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=args.font).get_name()
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.weight'] = 'medium'
        plt.rcParams['axes.titleweight'] = 'medium'
        plt.rcParams['figure.titleweight'] = 'medium'
        chinese = True
    cv2.setNumThreads(2)
    cache = {}
    selected = []
    for dataset, index, rois in CASES:
        if dataset not in cache:
            origin = args.source / ('A_' + dataset)
            registration = json.loads((origin / 'registration.json').read_text())
            manifest = Path(registration['manifest']['path'])
            assert sha(manifest) == registration['manifest']['sha256']
            cache[dataset] = (origin, read_rows(manifest), read_rows(origin / 'pairs.jsonl'),
                              json.loads((origin / 'match_index.json').read_text()))
        origin, rows, records, mappings = cache[dataset]
        row, original, mapping = rows[index], records[index], mappings[index]
        assert row['pair_id'] == original['pair_id'] == mapping['pair_id']
        m = original['metrics']
        assert m['ncm']['3'] >= 6000 and not m['sr']['3'] and m['H_estimated'] is not None
        a_path, b_path = Path(mapping['image_A']), Path(mapping['image_B'])
        assert sha(a_path) == original['image_sha256']['A']
        assert sha(b_path) == original['image_sha256']['B']
        A, B = [np.asarray(Image.open(path).convert('RGB')) for path in (a_path, b_path)]
        assert A.shape == B.shape == (512, 512, 3)
        H = np.asarray(m['H_estimated'], dtype=np.float64)
        H_gt = np.asarray(row['H_A_to_B'], dtype=np.float64)
        xx, yy = np.meshgrid(np.linspace(0, 511, 20), np.linspace(0, 511, 20))
        grid = np.c_[xx.ravel(), yy.ravel()]
        truth, valid = project(grid, H_gt)
        valid &= (truth >= 0).all(1) & (truth <= 511).all(1)
        predicted, pvalid = project(grid[valid], H)
        assert pvalid.all()
        error = np.linalg.norm(predicted - truth[valid], axis=1)
        assert abs(float(np.sqrt(np.mean(error**2))) - m['registration_rmse']) < 1e-9
        source_npz = origin / 'matches' / f'{index:05d}.npz'
        with np.load(source_npz, allow_pickle=False) as stored:
            pa, pb = stored['points_a'], stored['points_b']
            correct_truth, projectable = project(pa, H_gt)
            inside = projectable & (pa >= 0).all(1) & (pa <= 511).all(1)
            inside &= (pb >= 0).all(1) & (pb <= 511).all(1)
            inside &= (correct_truth >= 0).all(1) & (correct_truth <= 511).all(1)
            correct = inside & (np.linalg.norm(correct_truth - pb, axis=1) <= 3)
            assert int(correct.sum()) == m['ncm']['3']
        folder = args.output / f'{dataset}_{index:05d}'
        folder.mkdir(exist_ok=True)
        one = np.ones((512, 512), dtype=np.uint8)
        warped = cv2.warpPerspective(A, H, (512, 512), flags=cv2.INTER_LINEAR)
        warped_gt = cv2.warpPerspective(A, H_gt, (512, 512), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(one, H, (512, 512), flags=cv2.INTER_NEAREST).astype(bool)
        mask_gt = cv2.warpPerspective(one, H_gt, (512, 512), flags=cv2.INTER_NEAREST).astype(bool)
        source_limits = np.percentile(cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), [2, 98])
        target_limits = np.percentile(cv2.cvtColor(B, cv2.COLOR_RGB2GRAY), [2, 98])
        overlay = red_cyan(B, warped, mask, target_limits, source_limits)
        gt_overlay = red_cyan(B, warped_gt, mask_gt, target_limits, source_limits)
        shutil.copyfile(a_path, folder / 'original_A.png')
        shutil.copyfile(b_path, folder / 'original_B.png')
        for name, image in [('registered_A', warped), ('registered_A_GT', warped_gt),
                            ('overlay_pred', overlay), ('overlay_GT', gt_overlay)]:
            write_image(folder / (name + '.png'), image)
        roi_records = []
        for rid, (x, y, w, h) in enumerate(rois, 1):
            assert mask[y:y+h, x:x+w].all() and mask_gt[y:y+h, x:x+w].all()
            rx, ry = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))
            target_points = np.c_[rx.ravel(), ry.ravel()]
            source_points, valid = project(target_points, np.linalg.inv(H_gt))
            expected, _ = project(source_points, H_gt)
            actual, valid_pred = project(source_points, H)
            assert valid.all() and valid_pred.all()
            local_error = np.linalg.norm(actual-expected, axis=1)
            roi_records.append(dict(id=rid, target_xywh=[x, y, w, h],
                                    pixel_grid_rmse=float(np.sqrt(np.mean(local_error**2))),
                                    error_min=float(local_error.min()), error_max=float(local_error.max()),
                                    visibility='Both GT and predicted source-warp support fully cover this crop.'))
            for name, image in [('B', B), ('registered_A', warped), ('overlay_pred', overlay), ('overlay_GT', gt_overlay)]:
                crop = image[y:y+h, x:x+w]
                write_image(folder / f'roi{rid}_{name}_native.png', crop)
                write_image(folder / f'roi{rid}_{name}_4x.png', cv2.resize(crop, (w*4, h*4), interpolation=cv2.INTER_NEAREST))
        info = dict(dataset=dataset, index=index, pair_id=row['pair_id'], variant='Full / original A',
                    coordinate_unit='stored_512_target_pixels', metrics=m, rois=roi_records,
                    input_images=dict(A=str(a_path), B=str(b_path)),
                    input_sha256=dict(A=sha(a_path), B=sha(b_path)), source_npz=str(source_npz),
                    source_npz_sha256=sha(source_npz), H_A_to_B=H.tolist(), H_gt_A_to_B=H_gt.tolist(),
                    display_intensity_windows=dict(source_p2_p98=source_limits.tolist(), target_p2_p98=target_limits.tolist(),
                                                   scope='Overlays only; identical windows for prediction and GT. Original/registered RGB unchanged.'),
                    grid=dict(valid_points=int(len(error)), rmse=float(np.sqrt(np.mean(error**2))),
                              median=float(np.median(error)), p90=float(np.percentile(error, 90)),
                              maximum=float(error.max()), fraction_within3=float(np.mean(error <= 3))),
                    scope='Selected qualitative examples; no training, neural inference, RANSAC refitting, or metric change.',
                    folder=folder.name)
        # The full SR grid retains all GT-visible points, even if predicted
        # projections fall outside the target image. Crop support is display-only.
        plot_case(folder, A, B, warped, overlay, gt_overlay, H_gt, rois, info, chinese)
        (folder / 'case.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
        selected.append(info)
        print(json.dumps(dict(pair=row['pair_id'], comparison=str(folder/'comparison.png'),
                              ncm3=m['ncm']['3'], grid_rmse=m['registration_rmse']), ensure_ascii=False))
    (args.output / 'cases.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n')
    page = '<!doctype html><meta charset="utf-8"><title>SR失败但正确匹配较多的样本</title>'
    page += '<style>body{font-family:system-ui,sans-serif;margin:28px;line-height:1.6;color:#192536}img{max-width:100%;height:auto}section{margin:28px 0}a{margin-right:12px}h1,h2{line-height:1.3}</style>'
    page += '<h1>SR失败，但正确匹配点很多</h1><p>使用已保存的Full/A最终匹配和预测H。没有重新推理或重新拟合。原始图为测试时的512×512输入。</p>'
    page += '<p>每张对比图：第一行是原始A、原始B、预测H配准后的A、红青叠加；第二和第三行分别放大两个64×64局部，最后一列为GT参考。红=B灰度，青=配准A灰度。为便于比较，叠加图使用原始A/B各自2%–98%灰度窗口作显示归一化，预测和GT使用完全相同的窗口；原始与配准RGB不作亮度处理。时相差异仍可能改变颜色。</p>'
    for info in selected:
        f, m = info['folder'], info['metrics']
        page += '<section><h2>'+html.escape(info['pair_id'])+'</h2>'
        page += f"<p>NCM@3={m['ncm']['3']}，P@3={100*m['precision']['3']:.2f}%，网格RMSE={m['registration_rmse']:.3f}px &gt; 3px。</p>"
        page += f'<a href="{f}/comparison.png"><img src="{f}/comparison.png" alt="原始与配准结果及两处局部放大"></a><p>'
        for filename, label in [('original_A.png','原始A'),('original_B.png','原始B'),('registered_A.png','配准A'),('overlay_pred.png','预测叠加'),('overlay_GT.png','GT参考'),('case.json','详细坐标与矩阵')]:
            page += f'<a href="{f}/{filename}">{label}</a>'
        page += '</p></section>'
    (args.output / 'index.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO/'outputs/h0_position_ablation_512_20261013')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/sr_failure_high_ncm_cases_512_20261015')
    parser.add_argument('--font', type=Path, default=None)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.output == args.source or args.output.is_relative_to(args.source):
        parser.error('Output must be outside the immutable source experiment')
    run(args)


if __name__ == '__main__':
    main()
