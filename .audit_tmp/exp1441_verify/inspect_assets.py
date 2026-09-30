# -*- coding: utf-8 -*-
"""14.4-1 checkpoint / reflow_pairs 体检"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import torch

FOLDER = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.4-1"

for name in ['rf_final.pth', 'rf_checkpoint.pth', '2rf_final.pth', '2rf_checkpoint.pth']:
    p = os.path.join(FOLDER, name)
    ck = torch.load(p, map_location='cpu', weights_only=False)
    print(f"== {name} == keys: {list(ck.keys())}")
    if 'epoch' in ck:
        print(f"  epoch(最后一次保存): {ck['epoch']}")
    sd = ck['model_state_dict']
    print(f"  参数量: {sum(v.numel() for v in sd.values())}, 层数: {len(sd)}")

# 1rf 与 2rf 是否为不同权重（防止复制粘贴出的假2-RF）
ck1 = torch.load(os.path.join(FOLDER, 'rf_final.pth'), map_location='cpu', weights_only=False)
ck2 = torch.load(os.path.join(FOLDER, '2rf_final.pth'), map_location='cpu', weights_only=False)
same = all(torch.equal(ck1['model_state_dict'][k], ck2['model_state_dict'][k]) for k in ck1['model_state_dict'])
diff_l1 = sum((ck1['model_state_dict'][k] - ck2['model_state_dict'][k]).abs().sum().item() for k in ck1['model_state_dict'])
print(f"\n1-RF 与 2-RF final 权重完全相同: {same}  (L1差总和: {diff_l1:.4e})")

# Reflow 端点对
rp = torch.load(os.path.join(FOLDER, 'reflow_pairs.pth'), map_location='cpu', weights_only=False)
print(f"\n== reflow_pairs.pth == keys: {list(rp.keys())}")
z, x0 = rp['reflow_z'], rp['reflow_x0']
print(f"  reflow_z: {tuple(z.shape)} mean={z.mean():.4f} std={z.std():.4f} min={z.min():.3f} max={z.max():.3f}")
print(f"  reflow_x0: {tuple(x0.shape)} mean={x0.mean():.4f} std={x0.std():.4f} min={x0.min():.3f} max={x0.max():.3f}")
# x0 应是 clamp 后的图像样式：统计被截断到 0/1 的比例
frac_lo = (x0 <= 1e-6).float().mean().item()
frac_hi = (x0 >= 1 - 1e-6).float().mean().item()
print(f"  reflow_x0 贴0比例: {frac_lo:.3f}  贴1比例: {frac_hi:.3f}")
# 端点对是否已与真实数据做过重耦合？(纯RF生成, 无reflow后OT重排的标志: 检查 z-x0 相关性)
zf = z.view(z.shape[0], -1)[:, :50]; xf = x0.view(x0.shape[0], -1)[:, :50]
corr = torch.corrcoef(torch.stack([zf.mean(0), xf.mean(0)]))[0, 1].item()
print(f"  (z, x0) 像素级弱相关示例 corr={corr:.4f}（独立耦合，Reflow不做OT重排——符合14.4-1定义）")
