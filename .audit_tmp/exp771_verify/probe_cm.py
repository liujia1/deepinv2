# -*- coding: utf-8 -*-
"""探测：一致性模型步骤2的弧形欠训练是固有局限还是可通过更多训练/容量修复。
仅复用 7.7-1 的 CD 训练逻辑，扫描不同超参组合，报告 fixed-p0 下的 NLL。"""
import numpy as np
import sys, io, os
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, r"D:\works\deepinv\第7章 扩散模型：SDE视角\第七章 配套实验\实验7.7-1")
# 直接复用主脚本里的函数（导入但不执行 __main__）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "m771", r"D:\works\deepinv\.audit_tmp\exp771_verify\771_fixed.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)   # 运行到 __main__ 守卫前，只定义函数（step1/step2 不在导入时执行）

np.random.seed(42)


def train_cm(iters, h, lr, ema_decay, nint, n_pairs, seed=1):
    model = m.TinyMLP(h=h, seed=seed)
    model_ema = m.TinyMLP(h=h, seed=seed)

    def copy_to(src, dst):
        dst.W1[:], dst.b1[:] = src.W1.copy(), src.b1.copy()
        dst.W2[:], dst.b2[:] = src.W2.copy(), src.b2.copy()
    copy_to(model, model_ema)

    t_grid = m.schedule_times(nint, 1.0, 0.01)[::-1][1:]
    idx = np.random.randint(0, len(t_grid) - 1, size=n_pairs)
    t1 = t_grid[idx]; t2 = t_grid[idx + 1]
    x0 = m.sample_p0(n_pairs)
    ab1 = m.alpha_bar(t1); eps = np.random.randn(n_pairs, m.DIM)
    x_t1 = np.sqrt(ab1)[:, None] * x0 + np.sqrt(1 - ab1)[:, None] * eps
    score = m.forward_score(x_t1, t1)
    beta = m.B_MIN + t1 * (m.B_MAX - m.B_MIN)
    vel = -0.5 * beta[:, None] * (x_t1 + score)
    x_t2 = x_t1 + vel * (t2 - t1)[:, None]

    batch = 512
    params = [model.W1, model.b1, model.W2, model.b2]
    ms = [np.zeros_like(w) for w in params]
    vs = [np.zeros_like(w) for w in params]
    b1m, b2m, epsa = 0.9, 0.999, 1e-8
    for it in range(1, iters + 1):
        sel = np.random.choice(n_pairs, batch, replace=False)
        x_a, ta = x_t1[sel], t1[sel]
        x_b, tb = x_t2[sel], t2[sel]
        tgt = m.cm_forward(model_ema, x_a, ta)
        tb_enc = (tb - 0.5) * 2.0
        z = np.concatenate([x_b, tb_enc[:, None]], axis=-1)
        F, a1 = model.forward(z)
        cs_b, co_b = m.c_skip_out(tb)
        pred = cs_b[:, None] * x_b + co_b[:, None] * F
        diff = pred - tgt
        loss = np.mean(np.sum(diff * diff, axis=-1))
        grad = 2.0 * diff * co_b[:, None]
        gw2, gb2, gw1, gb1 = model.backward(z, a1, grad)
        grads = [gw1, gb1, gw2, gb2]
        for k, (pp, gg) in enumerate(zip(params, grads)):
            ms[k][:] = b1m * ms[k] + (1 - b1m) * gg
            vs[k][:] = b2m * vs[k] + (1 - b2m) * (gg * gg)
            pp -= lr * (ms[k] / (1 - b1m ** it)) / (np.sqrt(vs[k] / (1 - b2m ** it)) + epsa)
        model_ema.W1[:] = ema_decay * model_ema.W1 + (1 - ema_decay) * model.W1
        model_ema.b1[:] = ema_decay * model_ema.b1 + (1 - ema_decay) * model.b1
        model_ema.W2[:] = ema_decay * model_ema.W2 + (1 - ema_decay) * model.W2
        model_ema.b2[:] = ema_decay * model_ema.b2 + (1 - ema_decay) * model.b2

    n = 5000
    xT = np.random.randn(n, m.DIM)
    x_cm1 = m.cm_forward(model_ema, xT, np.full(n, 1.0))
    nll = np.mean(m.neg_loglik0(x_cm1))
    # 统计覆盖到几个模态（按最近角分类）
    corner = np.zeros(4)
    dx = x_cm1[:, 0][:, None] - m.MU[None, :, 0]
    dy = x_cm1[:, 1][:, None] - m.MU[None, :, 1]
    nearest = np.argmin(dx * dx + dy * dy, axis=1)
    for c in range(4):
        corner[c] = np.mean(nearest == c)
    return loss, nll, corner


print("配置: iters,h,lr,ema,nint -> CDloss, CM1步NLL(fixed), 四角覆盖率")
for cfg in [(2500, 64, 1e-2, 0.999, 16),
            (8000, 64, 1e-2, 0.999, 16),
            (15000, 128, 1e-2, 0.999, 16),
            (15000, 128, 5e-3, 0.99, 16),
            (20000, 128, 1e-2, 0.99, 32)]:
    it, h, lr, ema, nint = cfg
    loss, nll, corner = train_cm(it, h, lr, ema, nint, 20000)
    print("  (%5d,%3d,%.0e,%.3f,%2d) -> loss=%.4f NLL=%.3f cover=%s" %
          (it, h, lr, ema, nint, loss, nll, np.round(corner, 2)))
