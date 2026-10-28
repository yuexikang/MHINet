"""Read-only live charts for full_rgb.py; no GPU or training-process access."""
import argparse
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = Path(__file__).resolve().parents[3]/'outputs/sr_failure_high_ncm_cases_512_20261015/fonts/NotoSansSC-Medium.ttf'
if FONT.exists():
    font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist if f.name != 'Noto Sans SC']
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = 'Noto Sans SC'
plt.rcParams['axes.unicode_minus'] = False


def records(path):
    if not path.exists(): return []
    result = []
    for line in path.read_text().splitlines():
        try: row = json.loads(line)
        except json.JSONDecodeError: continue  # A live append may end mid-line.
        if isinstance(row, dict): result.append(row)
    return result


def read_state(root):
    log = records(root/'run.log')
    config = json.loads((root/'config.json').read_text())
    data = json.loads((root/'data.json').read_text())
    train = {}
    for r in log:
        if r.get('event') == 'train': train[r['stage'], r['epoch'], r['pairs']] = r
    stages = {}
    for stage in ('flow', 'd2'):
        history = records(root/stage/'history.jsonl')
        initial = root/stage/'epoch_000.json'
        baseline = json.loads(initial.read_text())['summary'] if initial.exists() else None
        stages[stage] = dict(history=history, baseline=baseline,
            train=sorted([r for (s, _, _), r in train.items() if s == stage], key=lambda r:(r['epoch'],r['pairs'])))
    pid_file = root/'pid.txt'
    alive = False
    if pid_file.exists():
        try: os.kill(int(pid_file.read_text().strip()), 0); alive = True
        except (ProcessLookupError, PermissionError, ValueError): pass
    return dict(stages=stages, last=log[-1] if log else {}, config=config,
                train_pairs=data['train']['pairs'], alive=alive, complete=(root/'done.json').exists(),
                updated=datetime.fromtimestamp((root/'run.log').stat().st_mtime, ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M:%S'))


def chart(state, stage, fmt='svg'):
    s = state['stages'][stage]
    history, train = s['history'], s['train']
    key = 'flow' if stage == 'flow' else 'local_flow'
    metric = 'p3_visible' if stage == 'flow' else 'p3'
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), layout='constrained')
    color, accent = '#147d92', '#c57925'
    fig.suptitle(f"{'Flow 预训练' if stage == 'flow' else '冻结 Flow → D2 训练'} · 日志更新 {state['updated']}（北京时间）", fontsize=13)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.16)
        ax.set_xlabel('Epoch')
    ax = axes[0, 0]
    if train:
        ax.plot([r['epoch']-1+r['pairs']/state['train_pairs'] for r in train],
                [r['loss'] for r in train], color=color, lw=1, label='本轮累计均值，每100对记录')
    if history:
        ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history],
                'o', color=accent, ms=4, label='完整轮均值')
    ax.set_title('训练损失（每轮重新累计）'); ax.set_ylabel('Loss')
    if train: ax.legend(fontsize=8)
    for ax, field, title, unit, factor in (
        (axes[0, 1], metric, '完整验证 P@3', '准确率 (%)', 100),
        (axes[1, 0], 'epe', '完整验证 EPE', '512原图像素', 1)):
        ax.set_title(title); ax.set_ylabel(unit)
        if history:
            x = [r['epoch'] for r in history]
            y = [r['validation'][key][field]*factor for r in history]
            ax.plot(x, y, '-o', color=color, ms=4)
            ax.annotate(f'{y[-1]:.3f}', (x[-1], y[-1]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
            ax.margins(x=.15, y=.3)
        else: ax.text(.5,.5,'尚无完成轮次的验证结果',ha='center',transform=ax.transAxes)
        if s['baseline']:
            b = s['baseline'][key][field]*factor
            ax.text(.02,.97,f'训练前：{b:.3f}；曲线从第1轮开始',va='top',transform=ax.transAxes,fontsize=8)
    ax = axes[1, 1]; ax.set_title('每轮结束时的学习率'); ax.set_ylabel('Learning rate')
    if history:
        for name in history[-1]['lr']:
            ax.plot([r['epoch'] for r in history], [r['lr'][name] for r in history], '-o',ms=3,label=name)
        ax.set_yscale('log'); ax.legend(fontsize=8)
    else: ax.text(.5,.5,'等待第一轮完成',ha='center',transform=ax.transAxes)
    if not train: axes[0,0].text(.5,.5,'此阶段尚未开始',ha='center',transform=axes[0,0].transAxes)
    buffer = BytesIO(); fig.savefig(buffer, format=fmt, dpi=150); plt.close(fig)
    return buffer.getvalue()


def page(state, stage):
    s = state['stages'][stage]; history = s['history']; last = state['last']
    current = s['train'][-1] if s['train'] else None
    state_text = '全部完成' if state['complete'] else ('训练进程运行中' if state['alive'] else '训练进程未运行，请检查日志')
    progress = f"第 {current['epoch']} / {state['config']['epochs']} 轮 · {current['pairs']:,} / {state['train_pairs']:,} 对" if current else '此阶段尚未开始'
    if last.get('event') == 'evaluation' and last.get('stage') == stage:
        progress += f" · 正在验证 {last['pairs']:,}/{last['total']:,} 对"
    latest = history[-1] if history else {}
    patience = state['config'].get('early_stop_patience', 3)
    bad = latest.get('bad_epochs', 0)
    early_enabled = state['config'].get('early_stopping', True)
    early_status = f'连续未显著提升 {bad}/{patience} 轮' if early_enabled else '早停已关闭'
    policy = (f"早停：至少{state['config'].get('early_stop_min_epochs', 5)}轮、耐心{patience}轮、"
              f"显著提升超过{100*state['config'].get('early_stop_min_delta', .001):g}个百分点"
              if early_enabled else f"早停已关闭，两阶段各训练至第{state['config']['epochs']}轮")
    monitor = '有效可见D16点' if stage == 'flow' else '全部D2输出点'
    rows = ''
    key, metric = ('flow','p3_visible') if stage == 'flow' else ('local_flow','p3')
    for r in reversed(history):
        m = r['validation'][key]
        rows += f"<tr><td>{r['epoch']}</td><td>{r['train_loss']:.6f}</td><td>{100*m[metric]:.3f}%</td><td>{m['epe']:.3f}</td><td>{r.get('bad_epochs','—')}</td></tr>"
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="5">
<title>MHI 实时训练曲线</title><style>
body{{font:16px system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1200px;color:#20323d;background:#f7fafb}}
nav{{display:flex;gap:20px;flex-wrap:wrap}}a{{color:#147d92}}h1{{font-size:24px}}p{{line-height:1.6}}
img{{width:100%;height:auto;background:white}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
td,th{{padding:10px;text-align:right;border-bottom:1px solid #dce5e8}}small{{color:#52636e}}
</style><h1>MHI 实时训练曲线</h1><nav><a href="/?stage=flow">Flow 预训练</a><a href="/?stage=d2">冻结 Flow → D2</a><a href="/api/state">原始指标</a></nav>
<p><strong>{escape(state_text)} · {escape(progress)}</strong><br>
<small>每5秒刷新 · 日志最后更新：{state['updated']}（北京时间） · {early_status}</small></p>
<img src="/plot.svg?stage={stage}" alt="训练损失、验证P@3、验证EPE与学习率曲线">
<p><small>P@3 分母：{monitor}；阈值为512原图3像素。验证集共5,106对，每轮结束后更新；训练损失为本轮累计均值，不是单批损失。前两阶段损失尺度不同。{policy}；best按验证P@3的任何严格提升保存。</small></p>
<table><thead><tr><th>Epoch</th><th>训练均值Loss</th><th>验证P@3</th><th>EPE(px)</th><th>早停计数</th></tr></thead><tbody>{rows}</tbody></table></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/full_rgb_d2_20261027'))
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8766)
    p.add_argument('--snapshot', type=Path)
    args = p.parse_args()
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_bytes(chart(read_state(args.root), 'flow', 'png')); return
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path not in ('/', '/plot.svg', '/api/state'):
                self.send_error(404); return
            state = read_state(args.root)
            stage = parse_qs(url.query).get('stage', ['flow'])[0]
            if stage not in ('flow','d2'):
                self.send_error(400); return
            if url.path == '/plot.svg': body, mime = chart(state,stage), 'image/svg+xml'
            elif url.path == '/api/state': body, mime = json.dumps(state).encode(), 'application/json'
            else: body, mime = page(state,stage).encode(), 'text/html; charset=utf-8'
            self.send_response(200); self.send_header('Content-Type',mime)
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *unused): pass
    print(f'Live curves: http://{args.host}:{args.port}', flush=True)
    HTTPServer((args.host,args.port), Handler).serve_forever()


if __name__ == '__main__': main()
