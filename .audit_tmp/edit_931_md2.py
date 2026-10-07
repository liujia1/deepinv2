# -*- coding: utf-8 -*-
"""9.3 讲义第二轮修订：初始化探针实测值 + 复现精度说明"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'D:\works\deepinv\第9章 VAE与重参数化\9.3 ELBO训练与KL正则化.md'
t = open(P, encoding='utf-8', newline='').read()

A1 = r'''KL 只有约 0.2 nats/样本（每维 0.004）；'''
B1 = r'''KL 只有 0.30 nats/样本（每维 0.006，脚本的 `[初始化探针]` 会直接打印这个数）；'''

A2 = r'''> **⚠修订说明（复核后）**：本次复核逐 epoch 复现了训练曲线（本地 CPU 重训 20 轮，与存档 checkpoint 的 history 一致到 1% 量级，活跃维度轨迹 50→50→40→…→12 同形），补上了缺失的 `results_summary.json` 归档；并修正了三处表述：① 目标函数写成 $-\text{ELBO}$（旧版写作 "ELBO = BCE + β·KL"，符号反了）；② 图(c) 纵轴改为从 0 起，避免截断轴把 4.2 的起点视觉上压成"从 0 爬起"；③ 训练初期信念的描述与初始化事实对齐（旧版"信念乱七八糟"与 9.3 自己的 KL≈0 论断矛盾）。下文数据均为实测值。'''

B2 = r'''> **⚠修订说明（复核后）**：本次复核在本地 CPU 从零重训 20 轮做逐 epoch 对照：总损失曲线与存档一致到 0.04%（epoch 10 之后），KL 曲线一致到 1% 以内，"50 维全活跃 → 逐轮压到约 12 维"的坍缩轨迹完全复现（终值 11/50 vs 存档 12/50，差异来自一维恰好卡在 0.01 阈值附近；权重本身不逐位相同，存档结果产自 Colab，随机流与设备不同）。同时补上了缺失的 `results_summary.json` 归档，并修正四处表述：① 目标函数写成 $-\text{ELBO}$（旧版写作 "ELBO = BCE + β·KL"，符号反了）；② 图(c) 纵轴改为从 0 起，避免截断轴把 4.2 的起点视觉上伪装成"从 0 爬起"；③ 训练初期信念的描述与初始化事实对齐（旧版"编码器的信念乱七八糟、各图天各一方"与同一节"初期 KL≈0"直接矛盾——初期所有编码几乎重合在先验上）；④ 把"坍缩是温和表现"改为按实测失活比例陈述。下文数据均为实测值。'''

REPL = [(A1, B1), (A2, B2)]
for i, (a, b) in enumerate(REPL, 1):
    c = t.count(a)
    print('[%d] count=%d' % (i, c))
    assert c == 1, a[:60]
    t = t.replace(a, b)
open(P, 'w', encoding='utf-8', newline='').write(t)
print('ok, double-backslash lines:', sum(1 for l in t.splitlines() if chr(92) * 2 in l))
