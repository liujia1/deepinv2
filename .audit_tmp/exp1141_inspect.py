# -*- coding: utf-8 -*-
"""实验11.4-1 复核脚本1：checkpoint元数据 + loss曲线核对"""
import sys, io, os, json
import io
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EXP = r"D:\works\deepinv\第11章 扩散模型：变分视角\第十一章 配套实验\实验11.4-1"

for name in ['ddpm_epsilon_checkpoint.pth', 'ddpm_x0_checkpoint.pth']:
    p = os.path.join(EXP, name)
    ck = torch.load(p, map_location='cpu', weights_only=False)
    print('='*60)
    print(name)
    print('  keys:', list(ck.keys()))
    print('  epoch:', ck.get('epoch'))
    print('  pred_type:', ck.get('pred_type'))
    print('  is_final:', ck.get('is_final'))
    print('  loss(final):', ck.get('loss'))
    hist = ck.get('history', {})
    losses = hist.get('loss', [])
    print('  history len:', len(losses))
    if losses:
        print('  loss[0]=%.6f loss[1]=%.6f loss[-1]=%.6f' % (losses[0], losses[1], losses[-1]))
        print('  all losses[:5]:', ['%.6f' % v for v in losses[:5]])
        print('  all losses[-5:]:', ['%.6f' % v for v in losses[-5:]])
    print('  has optimizer_state_dict:', 'optimizer_state_dict' in ck)
    sd = ck['model_state_dict']
    print('  n params: %.4fM' % (sum(v.numel() for v in sd.values())/1e6))
    # 打印前几个shape用于架构比对
    shapes = {k: tuple(v.shape) for k, v in list(sd.items())[:8]}
    print('  sample shapes:', shapes)

# 架构兼容性：当前代码的SmallUNet能否加载
sys.path.insert(0, EXP)
import importlib.util
spec = importlib.util.spec_from_file_location('net1141', os.path.join(EXP, '11.4-1.py'))
# 不执行整个脚本（会训练），改为手动复制类定义太繁琐——用exec截断到类定义结束
src = open(os.path.join(EXP, '11.4-1.py'), encoding='utf-8').read()
cut = src.index('# ============================================================\n# DDPM采样算法')
ns = {'__name__': '__main__'}
# 需要 np, torch, nn, F 等，脚本头部已import；exec头部段
head = src[:cut]
# 去掉设备打印段以防副作用？head里只有调度定义和类定义，安全
exec(compile(head, '11.4-1_head', 'exec'), ns)
SmallUNet = ns['SmallUNet']
for name in ['ddpm_epsilon_checkpoint.pth', 'ddpm_x0_checkpoint.pth']:
    ck = torch.load(os.path.join(EXP, name), map_location='cpu', weights_only=False)
    m = SmallUNet(pred_type=ck.get('pred_type', 'epsilon'))
    try:
        missing, unexpected = m.load_state_dict(ck['model_state_dict']), None
        print(f"\n[OK] 当前SmallUNet 可加载 {name}")
    except RuntimeError as e:
        print(f"\n[FAIL] {name} 加载失败: {e}")
