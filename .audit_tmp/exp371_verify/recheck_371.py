# -*- coding: utf-8 -*-
"""实验3.7-1 修改后工作的独立复核（全新脚本，不复用上一轮 verify_371.py 的结论）。
A. 从中立 cwd 子进程重跑实验 -> 字体/异常检查
B. 重跑前后 JSON 逐位比对（确定性）
C. seed=42 展示流逐位复算
D. theoretical_exact_chi 与 scipy 精确值核对
E. 多种子区间逐位复算 + seed42 分位数（核对 md 声明）
F. md 数字与表述核对
G. 脚本遗留问题扫描
H. 产物文件存在性/尺寸/时间戳
"""
import json, math, os, re, struct, subprocess, sys, time
import numpy as np
from scipy import stats

EXP = r"d:\works\deepinv\第3章 从MAP到后验探索\第三章 配套实验\实验3.7-1"
PY = os.path.join(EXP, "3.7-1.py")
JS = os.path.join(EXP, "results_summary.json")
MD = r"d:\works\deepinv\第3章 从MAP到后验探索\3.7 从MAP到后验：点估计的局限与分叉.md"
DIMS = [2, 10, 100, 1000]
fails = []

def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        fails.append(name)

print("== A. 从中立 cwd 子进程重跑 ==")
snap = open(JS, encoding="utf-8").read()
r = subprocess.run([sys.executable, PY], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", cwd=r"d:\works\deepinv", timeout=280)
out = r.stdout + r.stderr
check("退出码为 0", r.returncode == 0, f"code={r.returncode}")
check("中文字体成功加载", ("已检测到中文字体" in out) or ("已加载缓存字体" in out) or ("已下载并注册" in out))
check("无 Traceback", "Traceback" not in out)
check("无字体缺失警告", "chinese_font 模块未找到" not in out and "方框" not in out)
check("核心结论 1-7 行齐备", all(f"{i}." in out for i in range(1, 8)))

print("== B. 重跑前后 JSON 逐位比对 ==")
after = open(JS, encoding="utf-8").read()
check("JSON 逐位一致（完全确定性）", after == snap)
J = json.loads(after)
check("output_files 指向的文件均存在", all(os.path.exists(p) for p in J["output_files"]))

print("== C. seed=42 展示流逐位复算 ==")
np.random.seed(42)
ok_c = True
for d in DIMS:
    x = np.random.randn(500, d)
    rr = np.linalg.norm(x, axis=1)
    ok_c &= math.isclose(np.median(rr), J["scene_a_typical_set"]["typical_norm_median"][str(d)], abs_tol=1e-12)
    ok_c &= math.isclose(np.mean(rr), J["scene_a_typical_set"]["mean_norm"][str(d)], abs_tol=1e-12)
check("scene_a 全部 median/mean 逐位复现", ok_c)

print("== D. 精确 chi(n) 矩核对 ==")
ok_d = True
for d in DIMS:
    c = stats.chi(d); e = J["theoretical_exact_chi"][str(d)]
    ok_d &= abs(e["median"] - c.ppf(0.5)) < 1e-12
    ok_d &= abs(e["mean"] - c.mean()) < 1e-12
    ok_d &= abs(e["std"] - c.std()) < 1e-12
check("theoretical_exact_chi 与 scipy 一致到 1e-12", ok_d)

print("== E. 多种子区间逐位复算 + 分位数核对 ==")
rng = np.random.default_rng(20261014)
ok_e = True
for d in DIMS:
    meds, means = [], []
    for _ in range(200):
        rr = np.linalg.norm(rng.standard_normal((500, d)), axis=1)
        meds.append(np.median(rr)); means.append(np.mean(rr))
    meds, means = np.array(meds), np.array(means)
    mlo, mhi = np.percentile(meds, [2.5, 97.5])
    alo, ahi = np.percentile(means, [2.5, 97.5])
    mb = J["multiseed_robustness"]["median_band"][str(d)]
    ab = J["multiseed_robustness"]["mean_band"][str(d)]
    ok_e &= math.isclose(mlo, mb[0], abs_tol=1e-12) and math.isclose(mhi, mb[1], abs_tol=1e-12)
    ok_e &= math.isclose(alo, ab[0], abs_tol=1e-12) and math.isclose(ahi, ab[1], abs_tol=1e-12)
    med42 = J["scene_a_typical_set"]["typical_norm_median"][str(d)]
    mean42 = J["scene_a_typical_set"]["mean_norm"][str(d)]
    p_m = float((meds < med42).mean() * 100); p_a = float((means < mean42).mean() * 100)
    in_m, in_a = mb[0] <= med42 <= mb[1], ab[0] <= mean42 <= ab[1]
    print(f"    n={d:<5} median分位={p_m:5.1f}%({'带内' if in_m else '带外'})  "
          f"mean分位={p_a:5.1f}%({'带内' if in_a else '带外'})")
check("median/mean 区间逐位复现", ok_e)
md = open(MD, encoding="utf-8").read()
# md 声明：n=2,100,1000 全部带内；n=10 仅 median 带外且约 0.5% 分位
p10m = None
rng2 = np.random.default_rng(20261014)
meds10 = np.array([np.median(np.linalg.norm(rng2.standard_normal((500, 10)), axis=1)) for _ in range(200)])
p10m = float((meds10 < J["scene_a_typical_set"]["typical_norm_median"]["10"]).mean() * 100)
check("md 'n=10 median 约0.5%分位' 属实", abs(p10m - 0.5) < 1.0, f"实测 {p10m}%")

print("== F. md 数字与表述核对 ==")
for name, cond in [
    ("四子图描述", "四子图" in md),
    ("'两张图（步骤1、步骤2）'", "两张图（步骤1、步骤2）" in md),
    ("多种子段落存在且仅一次", md.count("多种子稳健性与概念辨析") == 1),
    ("区间下缘 2.994", "2.994" in md),
    ("mean 区间 [3.034, 3.138]", "3.034" in md and "3.138" in md),
    ("精确均值 3.0843", "3.0843" in md),
    ("壳层众数 3.00", "3.00" in md),
    ("圆内质量 63.2%", "63.2" in md),
    ("JSON 键名 theoretical_exact_chi / multiseed_robustness 被引用",
     "theoretical_exact_chi" in md and "multiseed_robustness" in md),
    ("旧'三子图'已清除", "三子图" not in md),
    ("旧'一张图'已清除", "一张图" not in md),
    ("'RMS 半径圆'新表述", "RMS 半径圆" in md),
]:
    check(name, cond)

print("== G. 脚本遗留问题扫描 ==")
src = open(PY, encoding="utf-8").read()
for pat, name in [("图3-7", "无出处'图3-7'声明已删"),
                  ("端到端", "'端到端学习'错误归属已清"),
                  ("众数位置 = 0", "旧 r=0 图注已改"),
                  ("_chinese_path = '.chinese'", "字体相对路径已修"),
                  ("step='mid'", "误导性阶梯填充已移除")]:
    check(name, pat not in src)
check("四子图布局", "1, 4, 1" in src and "1, 3, 1" not in src)
check("误差棒绘制", "errorbar" in src)

print("== H. 产物文件 ==")
for p in [os.path.join(EXP, "步骤1_高维高斯众数不等于典型.png"),
          os.path.join(EXP, "步骤2_两条路径分叉.png")]:
    with open(p, "rb") as f:
        w, h = struct.unpack(">II", f.read(24)[16:24])
    age = time.time() - os.path.getmtime(p)
    check(f"{os.path.basename(p)} 尺寸/新鲜度", w >= 1900 and age < 3600, f"{w}x{h}, {age:.0f}s 前")

print()
print("总结:", "全部通过" if not fails else f"{len(fails)} 项失败: {fails}")
