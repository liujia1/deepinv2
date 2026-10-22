# -*- coding: utf-8 -*-
"""交叉验证 f19: sigma-sweep 数值变化是否来自"训练集 -> 测试集"的评估口径改动。

方法: 固定同一份 DnCNN 权重 (dncnn_checkpoint.pth), 只改变评估图像集:
  A) test_loader 前 128 张 (新代码口径, shuffle=False)
  B) train_loader 的若干 batch (旧代码口径, 随机抽取; 取多 batch 平均以消除单批偶然性)
若 B 的衰减比显著高于 A, 说明旧 md 数值偏高是"用训练样本评估"造成的, 修复方向正确。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT = r'h:\deepinv\deepinv2\第15章 扩散模型的架构实践：UNet → DiT\第十五章 配套实验\实验15.2-1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DnCNN(nn.Module):
    def __init__(self, channels=32, n_blocks=6):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU()]
        for _ in range(n_blocks - 2):
            layers.extend([nn.Conv2d(channels, channels, 3, padding=1),
                           nn.BatchNorm2d(channels), nn.ReLU()])
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_ds = datasets.MNIST(os.path.join(ROOT, 'data'), train=True, download=False, transform=tf)
test_ds = datasets.MNIST(os.path.join(ROOT, 'data'), train=False, download=False, transform=tf)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

dncnn = DnCNN().to(device)
ck = torch.load(os.path.join(ROOT, 'dncnn_checkpoint.pth'), map_location=device)
dncnn.load_state_dict(ck['model_state_dict'])
dncnn.eval()
print(f"loaded ckpt: epoch={ck.get('epoch')}, is_final={ck.get('is_final')}, loss={ck.get('loss')}")


def sweep(x_imgs, tag):
    torch.manual_seed(7)
    out = {}
    for s_val in [0.1, 0.3, 0.5]:
        mn = md = 0.0
        with torch.no_grad():
            for _ in range(5):
                x_s = x_imgs + torch.randn_like(x_imgs) * s_val
                mn += F.mse_loss(x_s, x_imgs).item()
                md += F.mse_loss(dncnn(x_s), x_imgs).item()
        mn /= 5
        md /= 5
        out[s_val] = (mn, md, md / mn)
        print(f"  [{tag}] sigma={s_val}: noisy={mn:.4f} denoised={md:.4f} ratio={md/mn:.4f}")
    return out


test_imgs = next(iter(test_loader))[0][:6].to(device)
print("\n=== A) 测试集前128张 (新代码口径) ===")
a = sweep(test_imgs, 'test')

# 多个训练 batch 平均(模拟旧代码的随机 train batch)
train_batches = []
torch.manual_seed(0)
it = iter(train_loader)
for _ in range(5):
    train_batches.append(next(it)[0].to(device))
print("\n=== B) 训练集 batch x5 逐个统计 (旧代码口径) ===")
ratios = {0.1: [], 0.3: [], 0.5: []}
for i, tb in enumerate(train_batches):
    r = sweep(tb, f'train#{i}')
    for s in ratios:
        ratios[s].append(r[s][2])
print("\n=== 汇总 ===")
for s in [0.1, 0.3, 0.5]:
    print(f"sigma={s}: test={a[s][2]:.4f} | train 5批次均值={sum(ratios[s])/len(ratios[s]):.4f} "
          f"(范围 {min(ratios[s]):.4f}~{max(ratios[s]):.4f})")
print("\n旧 md 记录(旧 train batch 单次): 0.1->0.3381, 0.3->0.0796, 0.5->0.0852")
print("新 json(测试集):                0.1->0.2030, 0.3->0.0548, 0.5->0.0761")
