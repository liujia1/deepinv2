# -*- coding: utf-8 -*-
"""5.6-2 临时验证：plateau / no_plateau / old_keys / missing 四路径。"""
import os, shutil, subprocess, sys, json
import numpy as np

EXP = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-2\5.6-2.py"
FONT = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.6-2\.chinese"
ROOT = r"d:\works\deepinv\.audit_tmp\exp562_verify"
PY = r"D:\Programs\Python\Python310\python.exe"

rng = np.random.default_rng(0)
S = 64
xx, yy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
x_true = (0.3 + 0.4 * xx + 0.2 * (yy > 0.5)).astype(np.float64)
err_pat = 0.02 + 0.06 * np.clip((xx - 0.5) ** 2 + (yy - 0.4) ** 2, 0, 1)
post_mean = np.clip(x_true + rng.normal(0, 1, (S, S)) * err_pat, 0, 1)
post_std = np.clip(0.01 + 1.2 * np.abs(x_true - post_mean) + rng.normal(0, 0.002, (S, S)), 1e-4, None)
post_var = post_std ** 2

M = 95
e = np.empty((M, S, S))
for m in range(M):
    e[m] = (rng.normal(size=(S, S)) if m == 0 else 0.5 * e[m - 1]) + rng.normal(size=(S, S))
mc = (post_mean[None] + e * post_std[None] / np.sqrt(1 - 0.25)).astype(np.float32)


def ar_trace(n, rho):
    z = rng.normal(size=n)
    out = np.empty(n)
    for i in range(n):
        out[i] = z[i] if i == 0 else rho * out[i - 1] + np.sqrt(1 - rho ** 2) * z[i]
    return out


def make_npz(path, kind):
    d = dict(post_mean=post_mean, post_var=post_var,
             post_mean_h1=post_mean, post_var_h1=post_var,
             post_mean_h2=post_mean, post_var_h2=post_var,
             mc_samples=mc, x_true=x_true,
             y_obs=np.clip(x_true + rng.normal(0, 0.02, (S, S)), 0, 1))
    if kind in ("plateau", "no_plateau"):
        t = np.arange(9500)
        if kind == "plateau":
            tp, fid = 30 + 3 * (1 - np.exp(-t / 300.0)), ar_trace(9500, 0.05)
        else:
            tp, fid = 24 + 3 * t / 9500.0, ar_trace(9500, 0.995)
        d.update(trace_psnr=tp, trace_fid=fid,
                 chain_config=np.array([10000, 500, 5250, 100], dtype=np.int64))
    np.savez(path, **d)


def run_case(name, kind, put_npz=True):
    tdir = os.path.join(ROOT, "case_" + name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
    os.makedirs(tdir)
    shutil.copy2(EXP, os.path.join(tdir, "5.6-2.py"))
    shutil.copytree(FONT, os.path.join(tdir, ".chinese"))
    if put_npz:
        make_npz(os.path.join(tdir, "sampling_results.npz"), kind)
    r = subprocess.run([PY, "5.6-2.py"], cwd=tdir, capture_output=True, text=True, encoding="utf-8")
    print("=" * 25, name, "exit", r.returncode, "=" * 25)
    if not put_npz:
        print("STDOUT tail:", r.stdout[-300:])
        print("STDERR head:", r.stderr.splitlines()[-1] if r.stderr else "")
        return r.returncode
    gate = [ln for ln in r.stdout.splitlines() if ("上游链" in ln or "[警告]" in ln or "覆盖率" in ln
            or "分位约对应" in ln or "全链后验σ" in ln)]
    print("\n".join(gate[:20]))
    if r.returncode != 0:
        print(r.stderr[-2000:])
    pngs = sorted(f for f in os.listdir(tdir) if f.endswith(".png"))
    print("PNGs:", pngs)
    j = json.load(open(os.path.join(tdir, "results_summary.json"), encoding="utf-8"))
    print("gate JSON warnings:", j["上游链状态_5.5-1"].get("警告"))
    return r.returncode


rc = 0
rc |= run_case("new_plateau", "plateau")
rc |= run_case("new_noplateau", "no_plateau")
rc |= run_case("old_keys", "old")
rc_missing = run_case("missing", None, put_npz=False)
print("CASES EXIT", rc, "| missing exit", rc_missing, "(期望非0)")
sys.exit(0 if rc == 0 and rc_missing != 0 else 1)
