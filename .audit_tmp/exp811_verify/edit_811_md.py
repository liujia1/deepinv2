# -*- coding: utf-8 -*-
"""修正 8.1 讲义中实验8.1-1的失实数值、失效免责、图片链接，并补计时bug修复说明。
锚点唯一性断言 + 写回后双反斜杠核查。"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MD = r"d:\works\deepinv\第8章 变分推断与ELBO\8.1 为什么需要变分推断？.md"
text = open(MD, encoding='utf-8').read()
bs_before = sum(1 for line in text.splitlines() if '\\\\' in line)

reps = []

# 1) 解析后验数值修正
reps.append((
r'''其中 $w_1 \approx 0.36$, $\mu_1 \approx -0.24$, $\sigma_1 \approx 0.26$；$w_2 \approx 0.64$, $\mu_2 \approx 1.09$, $\sigma_2 \approx 0.26$。真实后验仍为双峰结构，但两峰位置相对于先验发生了偏移（似然的"拉力"）。''',
r'''其中 $w_1 = 0.40$, $\mu_1 \approx -0.53$, $\sigma_1 \approx 0.26$；$w_2 = 0.60$, $\mu_2 \approx +0.53$, $\sigma_2 \approx 0.26$。注意 $x=0$ 到两个先验分量（$\pm 2$）的距离严格相等，两分量的边际似然相同，因此后验权重保持先验权重 $(0.4,\,0.6)$ 不变。真实后验仍为双峰结构，但两峰位置从 $\pm 2$ 大幅向 $\pm 0.53$ 收缩——这是观测 $x=0$ 通过似然施加的"拉力"。'''
))

# 2) 结果段开头补 JSON 归档说明
reps.append((
r'''运行 `8.1-1.py` 后，生成可视化图。实验结果如下：''',
r'''运行 `8.1-1.py` 后，生成可视化图，完整数值归档于 `第八章 配套实验/实验8.1-1/results_summary.json`。实验结果如下（时间为本机实测，换机器数值会变、排序稳定）：'''
))

# 3) MCMC 结果块
reps.append((
r'''- 采样数：$10000$，预热期：$2000$
- 计算时间：约 $0.1-0.5$ 秒（取决于系统）
- 样本均值/标准差渐近收敛到真实后验的统计量
- KL 散度最小（渐近精确）''',
r'''- 采样数：$10000$，预热期：$2000$，接受率 $\approx 0.72$
- 计算时间：约 $0.4$ 秒（四策略中最慢）
- 样本均值 $0.110$、标准差 $0.578$，与真实后验统计量（均值 $0.106$、标准差 $0.579$）吻合
- KL 散度最小（$\approx 0.014$；注意这是直方图分箱估计的正偏置，MCMC 的真实 KL 随样本数趋于 $0$）'''
))

# 4) Laplace 结果块
reps.append((
r'''- 后验众数（MAP）：$z_{\text{MAP}} \approx 1.09$（位于右峰）
- Laplace 近似：$q(z) = \mathcal{N}(1.09, 0.26^2)$（单高斯）
- 计算时间：约 $0.01-0.05$ 秒（优化方法）
- KL 散度中等：只能捕获单峰，丢失左峰''',
r'''- 后验众数（MAP）：$z_{\text{MAP}} \approx 0.53$（位于右峰）
- Laplace 近似：$q(z) = \mathcal{N}(0.53, 0.26^2)$（单高斯）
- 计算时间：约 $0.003$ 秒（优化方法，最快之一）
- KL 散度最大（$\approx 0.48$）：只能捕获右峰，完全丢失左峰——四策略中近似质量最差'''
))

# 5) MAP 结果块
reps.append((
r'''- MAP 点：$z_{\text{MAP}} \approx 1.09$
- 计算时间：约 $0.01$ 秒（最快）''',
r'''- MAP 点：$z_{\text{MAP}} \approx 0.53$
- 计算时间：约 $0.004$ 秒（最快之一）'''
))

# 6) VI 结果块（删除失实的"取决于优化起点"免责）
reps.append((
r'''- 变分族：$q(z) = \mathcal{N}(\mu, \sigma^2)$
- 最优参数：$\mu^* \approx 0.5-1.0$（取决于优化起点），$\sigma^* \approx 0.3-0.5$
- 计算时间：约 $0.05-0.1$ 秒（优化方法）
- KL 散度中等：快速但有近似误差''',
r'''- 变分族：$q(z) = \mathcal{N}(\mu, \sigma^2)$，代码固定起点 $[\mu,\log\sigma]=[0,0]$ 与随机种子，结果确定
- 最优参数：$\mu^* \approx 0.10$，$\sigma^* \approx 0.53$；最优 ELBO $= -6.509$
- 一致性校验：$\log Z - \mathrm{ELBO} = 0.248$ 恰为真实 $KL(q\,\|\,p(z|x))$，与直方图估计 $0.271$（含分箱偏置）吻合
- 计算时间：约 $0.12$ 秒（优化方法）
- KL 散度中等（$\approx 0.27$）：单高斯无法表示双峰，但本场景中逆向 KL 找到的"覆盖两峰"折中解优于 Laplace 的锁峰解——这依赖两峰间距较近（$\pm 0.53$），并非普适结论。ELBO 关于 $(\mu,\sigma)$ 非凸，个别起点（如 $\mu_0=2$）会陷入更差的局部解，但全局解占绝对主导（多起点测试 6 中 5 收敛到同一解）'''
))

# 7) 图片链接路径修复
reps.append((
r'''![实验8.1-1 步骤3：计算代价vs近似质量](./第八章配套实现/8.1-1/步骤3_计算代价vs近似质量.png)''',
r'''![实验8.1-1 步骤3：计算代价vs近似质量](./第八章 配套实验/实验8.1-1/步骤3_计算代价vs近似质量.png)'''
))

# 8) 权衡表数值更新 + 计时bug复盘注记
reps.append((
r'''| MCMC | 慢（$\sim 0.3$s） | 小（渐近精确） | ✓ 完整 | 右下角（慢但准） |
| Laplace | 快（$\sim 0.02$s） | 中等 | ✓ 单峰 | 左中（快但欠准） |
| MAP | 最快（$\sim 0.01$s） | N/A | ✗ 无 | 无可比性（点估计） |
| 变分推断 | 快（$\sim 0.05$s） | 中等 | ✓ 近似 | 左中（快且较准） |''',
r'''| MCMC | 慢（$\sim 0.4$s） | 最小（$\approx 0.014$，渐近精确） | ✓ 完整 | 右下角（慢但准） |
| Laplace | 最快（$\sim 0.003$s） | 最大（$\approx 0.48$） | ✓ 单峰 | 左上角（快但最差） |
| MAP | 最快（$\sim 0.004$s） | N/A | ✗ 无 | 无可比性（点估计） |
| 变分推断 | 快（$\sim 0.12$s） | 中等（$\approx 0.27$） | ✓ 近似 | 左中（快且较准） |

> **复盘注记**：早期版本把 `from scipy.optimize import minimize` 写在 Laplace 计时段内部，scipy 首次导入的秒级开销被误计入 Laplace 耗时，旧配图因此显示"Laplace 最慢（0.45s）、位于右上角"。该测量 bug 已修复（import 移至文件顶部）并重跑，KL 数值逐位不变，耗时排序恢复为 MCMC > VI > MAP ≈ Laplace。"Laplace 慢"从来不是固有属性，纯属计时污染。'''
))

for i, (old, new) in enumerate(reps, 1):
    c = text.count(old)
    assert c == 1, f"锚点{i}出现{c}次，非唯一，中止"
    text = text.replace(old, new)

bs_after = sum(1 for line in text.splitlines() if '\\\\' in line)
assert bs_after == bs_before == 0, f"双反斜杠核查失败: before={bs_before}, after={bs_after}"

open(MD, 'w', encoding='utf-8').write(text)
print(f"OK: {len(reps)} 处替换完成；双反斜杠行数 before={bs_before} after={bs_after}")
