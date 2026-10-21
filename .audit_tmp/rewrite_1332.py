# -*- coding: utf-8 -*-
"""重写 13.3 md 的实验13.3-2 节（旧文与脚本严重脱节）。按行号替换 763-807 行。"""
p = r"h:\deepinv\deepinv2\第13章 条件生成与逆问题求解\13.3 近似方法分类与DPS深度剖析.md"
lines = open(p, encoding="utf-8").read().split("\n")
# 1-based 763..807 => 0-based 762..806
assert lines[762].startswith("## 实验 13.3-2"), lines[762]
assert lines[806].startswith("**解读警示**"), lines[806]
assert lines[808].startswith("#### 核心知识点"), lines[808]

new = r"""## 实验 13.3-2：DPS vs DiffPIR对比——Grad vs Opt两类方法

实验 `13.3-2.py` 演示**使用 deepinv 框架对比 DPS（Grad类）与 DiffPIR（Opt类）在 64×64 图像修复（inpainting）任务上的重建效果**，直观展示13.3.6节Daras分类中两类方法的差异。

### 实验场景

实验考虑图像修复问题：`y = M ⊙ x`（中心 16×16 像素被遮挡，无测量噪声；`M` 为二值掩码）。使用 deepinv 库提供的预训练模型和算法实现：

- **DPS**（Grad类，13.3.2节）：逆向 SDE 采样中每步用梯度步融入数据一致性约束。`dinv.sampling.PosteriorDiffusion` + `DPSDataFidelity(weight=1024)` + NCSNpp（FFHQ-64 EDM 得分模型，VE-SDE，250步，`alpha=0.25`）
- **DiffPIR**（Opt类，13.3.4节）：交替执行去噪和数据一致性投影（HQS），`DRUNet` 去噪器（DDPM），`max_iter=100`、`ζ=0.7`、`λ=7.0`

> **为什么任务选 inpainting 而非去模糊**：这是 deepinv 官方 DPS demo（`demo_diffusion_sde.py`/`demo_dps.py`）验证过的场景。官方从未在去模糊任务上验证过 DPS；本实验实测 DRUNet + 256×256 去模糊会结构性失败（踩坑记录见脚本结论 4/6）。

> **为什么两方法不共享先验**：DPS 为得分模型设计，需要去噪器在 SDE 全程 `σ∈[0.001,80]` 上近似有效（NCSNpp 正如此训练）；DRUNet 只在 `σ≲0.5` 有效，塞进 DPS 的高 σ 区会失效。DiffPIR 的 HQS 框架才接受任意普通去噪器。故"各自原生先验"才是两种算法的典型用法；先验不同这一混杂因素在解读时需注明。

#### 修订历史（版本漂移记录）

旧版配置（DiffUNet/ADM + VE-SDE 手工组装 `PosteriorDiffusion`，`clip=(0,1)`、`weight=0.5`）在 7/6 版 deepinv 上可运行（DPS 24.63 dB / DiffPIR 22.55 dB）；但在 0.4.2 上重跑同一配置，DPS 崩至 9.35 dB——根因是 0.4.2 默认 `minus_one_one=True`（状态空间 [-1,1]）与观测 `[0,1]` 空间失配，`clip=(0,1)` 又截掉一半动态范围，`weight=0.5` 引导过弱，DPS 退化为近似无条件采样。这是**实现-空间错误，不是算法属性**。现脚本已改用官方 `PosteriorDiffusion` + `DPSDataFidelity` 装配（对齐官方 demo），并对齐 NCSNpp 的 `[0,1]` 约定。

### 实验目的

1. **对比Grad与Opt两类方法重建效果**：DPS 在线修正得分 vs DiffPIR 交替去噪+投影
2. **对照两类方法输出性质差异**：DPS 给近似后验样本（随机种子影响大）；DiffPIR 缺口外是硬数据锁存（跨种子稳定），缺口内由随机扩散轨迹+弱先验决定（跨种子摆动大）——不是干净的单因素对比
3. **展示deepinv框架统一接口**：不同方法共享同一前向算子和观测数据，仅更换算法

### 实验输入

| 参数 | 设置 |
|------|------|
| 逆问题 | 64×64 Inpainting，中心 16×16 缺口，无测量噪声 |
| DPS先验/算法 | NCSNpp（FFHQ-64 EDM）+ VE-SDE，`weight=1024`、`alpha=0.25`、250步 |
| DiffPIR先验/算法 | DRUNet + DDPM/HQS，`max_iter=100`、`ζ=0.7`、`λ=7.0` |
| 随机种子 | seed=42（显式 `torch.Generator` 传参，已实证生效） |
| 框架 | deepinv 0.4.2 |

> **运行说明**：本实验需要 GPU 与 deepinv 库；首次运行联网下载 NCSNpp 权重。CPU 上无法复现（DPS 的 VE-SDE 250 步过慢）。

**DPS `weight` 校准**：`DPSDataFidelity` 的 `weight` 经本图实测扫描（4→1024 单调改善：总 PSNR 19.8→34.5 dB，可见区 MSE 0.0101→0.0001；2048 发散至 16.6 dB，缺口呈蓝黑色块、可见区完好）。官方 demo 的 `weight=4.0` 只是其演示图取值。Inpainting 的 `A` 是对角掩码，高权重近似每步良性硬投影，故可远高于官演示值；换图/换 seed/换步数后应重扫。

**DiffPIR `ζ` 校准**：本地 CPU×5种子（`σ=0.02`、`λ=7`、100步）：`ζ=0.1` 涂抹伪影；`ζ=0.3` 均值 24.5 dB、21.0–30.2 摆动、3/5 坏签（缺口内发丝涂抹）；`ζ=0.7` 均值 28.4 dB、1/5 坏签（定稿）。`σ=0.1/0.4` 数据锁变松（缺口外 MSE 0.0016/0.009）不可用；`λ=25` 无改善。

### 预期结果

- DPS 和 DiffPIR 均能有效恢复图像，PSNR 显著高于含遮挡观测（15.89 dB）
- DPS 在缺口内生成合理内容；DiffPIR 缺口外硬锁存、缺口内由随机轨迹+DRUNet 弱先验决定
- 两者 PSNR 差异需注明抽签成分：DiffPIR 的缺口填充跨种子摆动大（`ζ=0.3` 时 21–30 dB）

### 实验结果

运行 `13.3-2.py` 后（需要 GPU），生成 DPS vs DiffPIR 对比图：

| 方法 | PSNR |
|------|------|
| 观测（中心遮挡） | 15.89 dB |
| DPS（NCSNpp + VE-SDE，`weight=1024`） | **34.85 dB** |
| DiffPIR（DRUNet + HQS，`ζ=0.7`） | 31.12 dB |

![实验13.3-2：DPS vs DiffPIR对比](./第十三章%20配套实验/实验13.3-2/DPSvsDiffPIR对比.png)

**解读警示**：

- 两方法使用**不同先验**（NCSNpp vs DRUNet）：PSNR 差异混杂模型差异，不能读作"Grad 类 vs Opt 类"的纯净算法对比；同先验对照因 DRUNet 在高 σ 区失效而不可行（DPS 需要 `σ∈[0.001,80]` 全程有效的得分模型）。
- DPS 的 `weight=1024` 经本图扫描校准、DiffPIR 的 `ζ=0.7` 经 5 种子校准——两者都是"调好后的最佳值"，但仍非同一超参预算下的公平竞赛。
- DiffPIR 缺口外为硬数据锁存（跨种子/超参稳定 ~0.0002），缺口内填充随随机轨迹大幅摆动（`ζ=0.3` 时 21–30 dB）；PSNR 对比需注明抽签成分。
"""

lines[762:807] = new.split("\n")
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("rewritten lines 763-807; new total lines:", len(lines))
