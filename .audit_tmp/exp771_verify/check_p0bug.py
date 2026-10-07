# -*- coding: utf-8 -*-
"""复核实验7.7-1：定位 p0_pdf 只累加2/4模态 的 bug 对 NLL 指标的影响。"""
import numpy as np

np.random.seed(42)
MU = 1.2 * np.array([[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]])
W = np.ones(4) / 4.0
SIG = 0.8
DIM = 2


def p0_pdf_buggy(x):
    y = np.zeros(x.shape[0])
    for i in range(2):
        d = x - MU[i]
        y += W[i] * np.exp(-0.5 * np.sum(d * d, axis=-1) / (SIG ** 2)) \
             / ((2 * np.pi * SIG ** 2) ** (DIM / 2.0))
    return y


def p0_pdf_fixed(x):
    y = np.zeros(x.shape[0])
    for i in range(len(W)):
        d = x - MU[i]
        y += W[i] * np.exp(-0.5 * np.sum(d * d, axis=-1) / (SIG ** 2)) \
             / ((2 * np.pi * SIG ** 2) ** (DIM / 2.0))
    return y


def sample_p0(n):
    k = np.random.choice(len(W), p=W, size=n)
    return np.random.randn(n, DIM) * SIG + MU[k]


x = sample_p0(20000)
nll_buggy = np.mean(-np.log(p0_pdf_buggy(x) + 1e-30))
nll_fixed = np.mean(-np.log(p0_pdf_fixed(x) + 1e-30))
print("真实分布(4模态)样本 NLL —— buggy(2模态p0): %.4f" % nll_buggy)
print("真实分布(4模态)样本 NLL —— fixed(4模态p0): %.4f" % nll_fixed)

# 模拟一个"塌缩到底部2模态"的生成器（CM一步的弧形输出）
mask = (x[:, 1] < 0)
x_collapse = x[mask]
print("\n若生成器只覆盖底部2模态(塌缩):")
print("  buggy NLL: %.4f" % np.mean(-np.log(p0_pdf_buggy(x_collapse) + 1e-30)))
print("  fixed NLL: %.4f" % np.mean(-np.log(p0_pdf_fixed(x_collapse) + 1e-30)))

# 理论熵
ent = np.mean(-np.log(p0_pdf_fixed(sample_p0(200000)) + 1e-30))
print("\n理论下界(真实分布交叉熵≈熵) fixed: %.4f" % ent)
