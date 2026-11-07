"""Live progress bar for the joint 3-GPU RGB512 training run.

Reads only the run directory (``status.json`` / ``history.jsonl`` / ``run.json``),
so it can be started and stopped at any time without touching the training job.

    python tools/mhi_v2/joint_progress.py --run outputs/mhi_v2_joint_ddp20_20261105
    python tools/mhi_v2/joint_progress.py --run outputs/mhi_v2_joint_ddp20_20261105 --watch 2
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def tail_jsonl(path, count=40, window=256 * 1024):
    """Read only the tail of an append-only JSONL file."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - window))
        chunk = stream.read().decode('utf-8', 'replace')
    rows = []
    for line in chunk.splitlines()[-count:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def bar(fraction, width, filled='█', empty='░'):
    fraction = min(1., max(0., fraction))
    count = int(round(fraction * width))
    return filled * count + empty * (width - count)


def duration(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return '--'
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f'{hours}h{minutes:02d}m'
    if minutes:
        return f'{minutes}m{secs:02d}s'
    return f'{secs}s'


def num(value, digits=4):
    if value is None:
        return '--'
    if isinstance(value, int):
        return f'{value:,}'
    return f'{value:.{digits}f}'


def pid_alive(run):
    for name in ('pid.txt', 'watchdog.pid'):
        path = Path(run) / name
        if not path.is_file():
            continue
        try:
            os.kill(int(path.read_text().strip()), 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            continue
    return False


def render(run, width=None):
    run = Path(run)
    width = width or max(40, min(64, shutil.get_terminal_size((100, 24)).columns - 40))
    status = read_json(run / 'status.json') or {}
    registration = read_json(run / 'run.json') or {}
    history = read_jsonl(run / 'history.jsonl')
    alive = pid_alive(run)
    lines = []

    epoch = status.get('epoch', 0)
    epochs = status.get('epochs') or registration.get('epochs') or 0
    step = status.get('step', 0)
    total_steps = status.get('total_steps') or registration.get('total_steps') or 0
    cursor = status.get('cursor', 0)
    epoch_pairs = status.get('epoch_pairs') or 0
    state = status.get('state', 'starting' if alive else 'not started')

    elapsed = None
    started = registration.get('started_at')
    if started:
        try:
            elapsed = (datetime.now().astimezone() - datetime.fromisoformat(started)).total_seconds()
        except ValueError:
            elapsed = None
    # Steady-state throughput from the recent steps, not from wall clock (which
    # includes model build and the initial full validation).
    recent = [row['seconds_per_step'] for row in tail_jsonl(run / 'train.jsonl', 40)
              if row.get('seconds_per_step')]
    steady = sorted(recent)[len(recent) // 2] if recent else None
    rate = (1 / steady) if steady else ((step / elapsed) if (elapsed and step) else None)
    eta = ((total_steps - step) / rate) if (rate and total_steps) else None

    alive = pid_alive(run)
    alive = pid_alive(run)
    lines.append(f'MHINet-v2 joint DDP · {run}')
    lines.append(f'状态 {state}   更新 {status.get("updated_at", "--")}   '
                 f'进程 {"alive" if alive else "exited"}')

    overall = (step / total_steps) if total_steps else 0.
    lines.append(f'总进度 {bar(overall, width)} {overall * 100:5.1f}%  {step:,} / {total_steps:,} step')
    epoch_fraction = (cursor / epoch_pairs) if epoch_pairs else 0.
    if epoch_pairs:
        lines.append(f'轮次   epoch {epoch}/{epochs}   机内 {bar(epoch_fraction, width)} '
                     f'{epoch_fraction * 100:5.1f}%  {cursor:,} / {epoch_pairs:,} 对')
    else:
        lines.append(f'轮次   epoch {epoch}/{epochs}   机内 {"░" * width}   --（首个优化步尚未写入）')
    lines.append(f'吞吐   {num(rate, 2)} step/s ({num(rate * 12 if rate else None, 1)} 对/s)   '
                 f'已用 {duration(elapsed)}   剩余 {duration(eta)}')

    metrics = status.get('metrics') or {}
    if metrics:
        lines.append(
            'loss   {loss} | d8g {d8g}  d8l {d8l}  d2 {d2}  refine {ref} (loc {loc} / harm {harm})'.format(
                loss=num(status.get('loss')), d8g=num(metrics.get('d8_global')),
                d8l=num(metrics.get('d8_local')), d2=num(metrics.get('d2')),
                ref=num(metrics.get('refine')), loc=num(metrics.get('localization')),
                harm=num(metrics.get('harm'))))

    validation = status.get('last_validation')
    if validation:
        pre, post = validation['pre_refine'], validation['post_refine']
        lines.append(
            '验证   ep{ep}  post p3 {post:.4f} / pre p3 {pre:.4f}  Δ{delta:+.4f}  改坏 {worse:.1%}  '
            'epe {epe:.3f}'.format(
                ep=len(history), post=post['p3_visible'], pre=pre['p3_visible'],
                delta=validation.get('delta_p3_visible', 0.),
                worse=validation.get('worse_fraction') or 0., epe=post['epe']))
    best = status.get('best')
    if best is not None and best >= 0:
        lines.append(f'最佳   post p3 {best:.4f} @ep{status.get("best_epoch", 0)}   '
                     f'（best.pt 自带 epoch，与训练轮数解耦）')

    if history:
        last = history[-1]
        per_epoch = last['seconds'] if last.get('seconds') else None
        lines.append(f'上轮   ep{last["epoch"]} 训练 {duration(per_epoch)}  '
                     f'显存 {num(last.get("peak_gib"), 2)} GiB  '
                     f'lr(vgg/dedode/d2) {last["lr"]["vgg"]:.1e}/{last["lr"]["dedode"]:.1e}/{last["lr"]["matcher"]:.1e}')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path,
                        default=Path('outputs/mhi_v2_joint_ddp20_20261105'))
    parser.add_argument('--watch', type=float, nargs='?', const=2., default=None,
                        help='Refresh every N seconds (default 2).')
    parser.add_argument('--width', type=int)
    args = parser.parse_args(argv)

    if args.watch is None:
        print(render(args.run, args.width))
        return 0
    try:
        while True:
            sys.stdout.write('\x1b[2J\x1b[H' + render(args.run, args.width) + '\n')
            sys.stdout.write('\x1b[2mCtrl-C 退出\x1b[0m\n')
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
