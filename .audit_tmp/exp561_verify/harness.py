# -*- coding: utf-8 -*-
"""5.6-1 临时验证：合成 npz，验证上游闸门（新键/旧键/未平台）三种路径，不碰实验目录。"""
import os, shutil, subprocess, sys, json
import numpy as np

EXP = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-1\5.6-1.py"
FONT = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-1\.chinese"
ROOT = r"d:\works\deepinv\.audit_tmp\exp561_verify"
PY = r"D:\Programs\Python\Python310\python.exe"

rng = np.random.default_rng(0)
S = 64
xx, yy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
x_true = (0.3 + 0.4 * xx + 0.2 * (yy > 0.5)).astype(np.float64)
error_pattern = 0.02 + 0.06 * np.clip((xx - 0.5) ** 2 + (yy - 0.4) ** 2, 0, 1)
post_mean = np.clip(x_true + rng.normal(0, 1, (S, S)) * error_pattern, 0, 1)
true_err = np.abs(x_true - post_mean)
post_std = 0.01 + 1.2 * true_err + rng.normal(0, 0.002, (S, S))
post_std = np.clip(post_std, 1e-4, None)
post_var = post_std ** 2
y_obs = np.clip(x_true + rng.normal(0, 0.02, (S, S)), 0, 1)

M = 40
mc = np.empty((M, S, S))
e = rng.normal(size=(M, S, S))
for m in range(M):
    e[m] = (rng.normal(size=(S, S)) if m == 0 else 0.5 * e[m - 1]) + rng.normal(size=(S, S))
mc = post_mean[None] + e * post_std[None] / np.sqrt(1 - 0.5 ** 2)


def ar_trace(n, rho, scale=1.0):
    z = rng.normal(size=n)
    out = np.empty(n)
    for i in range(n):
        out[i] = z[i] if i == 0 else rho * out[i - 1] + scale * np.sqrt(1 - rho ** 2) * z[i]
    return out


def make_npz(path, kind):
    d = dict(post_mean=post_mean, post_var=post_var,
             post_mean_h1=post_mean, post_var_h1=post_var,
             post_mean_h2=post_mean, post_var_h2=post_var,
             mc_samples=mc.astype(np.float32), x_true=x_true, y_obs=y_obs)
    if kind in ("plateau", "no_plateau"):
        t = np.arange(9500)
        if kind == "plateau":
            tp = 30.0 + 3.0 * (1 - np.exp(-t / 300.0)) + rng.normal(0, 0.002, 9500)  # 末段近平
            fid = ar_trace(9500, 0.05)  # IAT≈1.1 << 100
        else:
            tp = 24.0 + 3.0 * t / 9500.0  # 末1/4仍+0.75dB
            fid = ar_trace(9500, 0.995)  # IAT≈400 > 100
        d.update(trace_psnr=tp, trace_fid=fid,
                 chain_config=np.array([10000, 500, 5250, 100], dtype=np.int64))
    np.savez(path, **d)


def run_case(name, kind):
    tdir = os.path.join(ROOT, "case_" + name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
    os.makedirs(tdir)
    shutil.copy2(EXP, os.path.join(tdir, "5.6-1.py"))
    shutil.copytree(FONT, os.path.join(tdir, ".chinese"))
    make_npz(os.path.join(tdir, "sampling_results.npz"), kind)
    r = subprocess.run([PY, "5.6-1.py"], cwd=tdir, capture_output=True, text=True, encoding="utf-8")
    print("=" * 30, name, "exit", r.returncode, "=" * 30)
    tail = "\n".join(r.stdout.splitlines()[:40])
    print(tail)
    if r.returncode != 0:
        print(r.stderr[-3000:])
    pngs = [f for f in os.listdir(tdir) if f.endswith(".png")]
    j = json.load(open(os.path.join(tdir, "results_summary.json"), encoding="utf-8"))
    print("PNGs:", pngs)
    print("上游JSON:", json.dumps(j["上游链状态_5.5-1"], ensure_ascii=False))
    return r.returncode


rc = 0
rc |= run_case("new_plateau", "plateau")
rc |= run_case("new_noplateau", "no_plateau")
rc |= run_case("old_keys", "old")
print("ALL EXIT", rc)
sys.exit(rc)
