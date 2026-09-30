"""Copy common weights into weight/ with checkpoint-derived Chinese sidecars.

Run with the project's PyTorch environment. Does not alter training or sources.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'weight'


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def atomic_text(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.description-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def copy_snapshot(source, target):
    """Read one open inode: latest.pt can be atomically replaced while copying."""
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix='.weight-')
    digest = hashlib.sha256()
    try:
        with source.open('rb') as src, os.fdopen(fd, 'wb') as dst:
            before = os.fstat(src.fileno())
            while block := src.read(8 * 1024 * 1024):
                digest.update(block)
                dst.write(block)
            after = os.fstat(src.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Source modified in place: {source}')
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return digest.hexdigest()


def training_description(target, source):
    payload = torch.load(target, map_location='cpu', weights_only=True, mmap=True)
    meta = payload['metadata']
    config = meta['config']
    runtime = meta.get('runtime') or json.loads(Path(config['runtime']).read_text())
    progress = payload.get('progress', {})
    step = progress.get('optimizer_step', meta.get('step'))
    total = meta.get('total_steps')
    semidense = meta['task'] == 'semidense_qrru_v1'
    world = meta.get('world_size', 1)
    lines = [
        '用途：' + ('半密集匹配：D8 粗匹配 + D2 细匹配 + QRRU 细化。' if semidense else
                   'GHIM 与共享描述子预训练导出；不含半密集匹配头，也不含冻结的 DINO 基础参数，使用时需配套基础权重。'),
        f'训练进度：{step}/{total} 优化步；' + ('已完成该阶段。' if step == total else '中间快照，非最终模型。'),
        f'输入尺寸：{meta.get("matcher", {}).get("input_size", 784)} × {meta.get("matcher", {}).get("input_size", 784)}',
        f'数据根目录：{runtime["data_root"]}',
        f'训练/验证清单：{runtime["data_root"]}/train/pairs.jsonl；{runtime["data_root"]}/val/pairs.jsonl',
        f'档位过滤：{config.get("tier")}（null/None 表示不按档位过滤）',
        f'样本量：训练 {meta.get("train_pairs")} 对，验证 {meta.get("val_pairs")} 对。',
        f'随机种子：{config.get("seed")}；world_size={world}；每卡 microbatch=1 对；梯度累计={config["accumulation"]}；有效批量={world * config["accumulation"]} 对。',
        f'初始化：{meta.get("source_checkpoint", config.get("initialize_checkpoint", "官方 LoRetta 与 LoMa 基础权重"))}',
        '阶段间只迁移模型参数，重新建立优化器和调度器；同一任务 resume 则恢复优化器、调度器与进度。',
        '优化器：AdamW；weight_decay=1e-4；梯度范数裁剪=1.0。',
    ]
    if semidense:
        lines += [
            f'训练轮数：{config["epochs"]}；初始 shared_lr={config["shared_lr"]}，head_lr={config["head_lr"]}。',
            f'调度：CosineAnnealingLR，T_max={total}，eta_min=0。',
            '损失：lambda_c*Lc + lambda_f*Lf + lambda_q*Lq。Lc 为 GT 粗对应的正样本 focal；Lf 为 GT 监督的局部细匹配；Lq 为多轮 2×2 控制流场及中心位移监督。',
            'H0 保留为细匹配窗口的几何先验并 detach；QRRU 使用独立 GT 中心与扰动监督，不输入 H0。',
            '冻结策略：' + meta.get('freeze_policy', '未记录'),
            '注意：H0 头即使 requires_grad=True，当前 Lc/Lf/Lq 也不直接更新该头；共享 MVT 解冻时可通过描述子路径更新。',
            '匹配器完整参数（旧权重未记录 input_size 时为 784；旧 inference_window_chunk 未记录时请在评测配置显式确认）：\n' + dump(meta['matcher']),
            '检查点时优化器参数组（lr 是保存时值，不是初始值）：\n' + dump([
                {k: v for k, v in group.items() if k != 'params'}
                for group in payload['optimizer']['param_groups']]),
            '执行设置：\n' + dump((payload.get('auxiliary_state') or {}).get('execution', {})),
        ]
    else:
        scale = config.get('learning_rate_scale', 1.)
        lines += [
            '训练方法：GHIM 几何监督 + D8/D4/D2 双向 GT 对应 InfoNCE 描述子监督。',
            '冻结 DINO、无 LoRA；训练 MVT、VGG、DeDoDe 与 GHIM 头；VGG BN 统计固定，D1 分支不训练。',
            f'基础学习率乘 learning_rate_scale={scale}；MVT={1e-6*scale:g}，VGG={5e-6*scale:g}，DeDoDe={1e-5*scale:g}，GHIM={1e-6*scale:g}。',
            '调度：前约 5% 步数 warmup，随后余弦衰减到基础学习率的 10%；该阶段遍历一轮选定数据。',
            f'描述子 queries={config.get("queries",1024)}，InfoNCE temperature=0.1；双向、多尺度监督。',
        ]
    dataset_summary = Path(runtime['data_root']) / 'dataset_summary.json'
    if dataset_summary.exists():
        lines += ['数据生成与来源说明（直接复制数据集记录）：\n' + dataset_summary.read_text()]
    # Include the historical lineage, rather than mislabelling only the last tier
    # as all data seen by the model. run.json is evidence, not a dependency rewrite.
    lineage, seen = [], set()
    previous = meta.get('source_checkpoint', config.get('initialize_checkpoint'))
    while previous and previous not in seen:
        seen.add(previous)
        run = Path(previous).parent / 'run.json'
        if not run.exists():
            lineage.append({'checkpoint': previous, 'note': '无同目录 run.json，历史细节未核实'})
            break
        ancestor = json.loads(run.read_text())
        lineage.append({'checkpoint': previous, 'record': ancestor})
        previous = ancestor.get('source_checkpoint', ancestor.get('config', {}).get('initialize_checkpoint'))
    lines += ['初始化训练链（祖先 run.json 原始记录）：\n' + dump(lineage),
              '本权重内嵌训练元数据（权威快照，保留原路径和哈希）：\n' + dump(meta),
              '本权重数据进度：\n' + dump(progress)]
    return '\n\n'.join(lines), step, total


def main():
    DEST.mkdir(exist_ok=True)
    runtime = json.loads((ROOT / 'configs/runtime_paths.mhinet_rgb512.server.json').read_text())
    entries = [
        (Path(runtime['dino_checkpoint']), 'loretta_base.pth', '官方 LoRetta 基础权重，提供项目使用的 DINO、MVT 和 GHIM 初始化。'),
        (Path(runtime['pyramid_checkpoint']), 'loma_B_base.pt', '官方 LoMa-B 基础权重，提供 VGG 与描述子解码器初始化。'),
    ]
    trained = sorted((ROOT / 'outputs').glob('shared_stable_v2_full_tier*_seed0/shared_descriptor.pt'))
    trained += sorted((ROOT / 'outputs').glob('semidense_stable_v2_tier*_seed0/latest.pt'))
    trained += [ROOT / 'outputs/semidense_rgb512_synth_seed42_5ep/latest.pt']
    entries += [(p, p.parent.name + '.pt', None) for p in trained]
    manifest = []
    for source, name, basic in entries:
        target = DEST / name
        sha = copy_snapshot(source, target)
        step = total = None
        if basic:
            body = basic + '\n属于上游预训练权重。本地没有完整上游训练数据和超参数记录，不推测或补写；不属于本项目训练结果。'
        else:
            body, step, total = training_description(target, source)
        header = f'权重：{name}\n来源：{source}\n来源真实路径：{source.resolve()}\nSHA256：{sha}\n整理时间 UTC：{datetime.now(timezone.utc).isoformat()}\n'
        header += '本文件为独立副本，不是软链接。原训练输出未移动；活动任务后续保存不会自动更新本副本，重新运行整理脚本可更新。\n'
        atomic_text(target.with_suffix('.txt'), header + '\n' + body + '\n')
        manifest.append(dict(file=name, description=target.with_suffix('.txt').name,
                             source=str(source), sha256=sha, bytes=target.stat().st_size,
                             step=step, total_steps=total))
        print(f'{name}: step={step}/{total}, SHA256={sha}', flush=True)
    atomic_text(DEST / 'manifest.json', dump(manifest) + '\n')
    atomic_text(DEST / 'README.txt',
        '常用权重目录\n\n每个权重均附同名 TXT：训练数据、训练方法、超参数、冻结策略、初始化链、进度和 SHA256。\n'
        '基础权重：loretta_base.pth 与 loma_B_base.pt。\n'
        '常用 E 组：semidense_stable_v2_tier3_lr_e_seed0.pt。\n'
        '当前 512 训练快照：semidense_rgb512_synth_seed42_5ep.pt；具体步数见同名 TXT，不能视为自动跟随 latest.pt 的最终结果。\n'
        '原 outputs/ 与运行配置不变，以保证正在运行的训练和历史来源校验可继续使用。新配置可显式引用这里的基础权重副本。\n'
        '二进制权重被 Git 忽略，仅存本机；说明与索引可提交 Git。\n\n'
        '更新副本与说明：\n/root/miniconda3/envs/loma-repro/bin/python /home/disk1/MHINet/scripts/collect_common_weights.py\n\n'
        + '\n'.join(f'{r["file"]} -> {r["description"]}' for r in manifest) + '\n')


if __name__ == '__main__':
    main()
