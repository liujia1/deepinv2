# -*- coding: utf-8 -*-
import io, sys
p = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
txt = io.open(p, encoding="utf-8").read()

# R1: 实验目的 item 3
r1_old = "3. **展示 TV 先验的阶梯效应局限**：在渐变区域出现阶梯状伪影"
r1_new = "3. **展示 TV 先验的阶梯效应局限**：借助合成斜坡信号无歧义地演示渐变区域的阶梯状伪影"
assert txt.count(r1_old) == 1, "R1 anchor"
txt = txt.replace(r1_old, r1_new)

# R2: 实验输入表新增合成信号行
r2_old = "| 求解方法 | Sobolev：FFT 频域闭式解；TV：Chambolle 原始-对偶算法 |"
r2_new = (r2_old + "\n" +
    r'| 合成信号（步骤2） | 一维"平台—线性渐变—平台—阶跃"信号，$m=128$、$\sigma=0.1$，Sobolev $\lambda=3.0$、TV $\alpha=0.15$ |')
assert txt.count(r2_old) == 1, "R2 anchor"
txt = txt.replace(r2_old, r2_new)

# R3: 重写步骤2正文（三面板），用切片替换规避引号/空白匹配问题
start = "左图展示了自动检测行"
endstr = "这就是阶梯效应（staircase effect）。"
i = txt.index(start)
j = txt.index(endstr, i) + len(endstr)
new_R3 = r'''图 2 由三个面板组成，从真实图像到合成信号逐层递进，无歧义地展示两种先验的行为差异。

**左图（真实剖面）**——自动检测行（第 50 行）：
- **Sobolev**：整体平滑，但边缘处过渡缓慢、尖锐峰值被明显压低（过度平滑磨边）
- **TV**：边缘跃变陡峭、峰值保留更好（保边）

**中图（合成斜坡＋阶跃）**——为排除真实纹理的干扰，用一段"平台—线性渐变—平台—阶跃"的合成信号（叠加 $\sigma=0.1$ 噪声）重复同一对比：
- 在**阶跃**处（绿色虚线），TV 几乎垂直下落、Sobolev 被磨成缓坡；阶跃区 MSE：TV $4.3\times10^{-4}$ vs Sobolev $4.1\times10^{-3}$，TV 优约一个数量级
- 在**渐变区**（金色底纹），TV 已呈阶梯状，Sobolev 平滑跟随

**右图（渐变区放大）**——把线性斜坡放大：真实信号是一条斜率恒定的直线，TV 却用"平坦段＋跳变"去逼近它，Sobolev 则连续平滑地跟踪。以"近零梯度占比"客观度量阶梯程度：TV 高达 61.7%（大量平台＝分段常数），Sobolev 仅 12.8%——这是阶梯效应的直接量化证据。

**阶梯效应的根源**：TV 先验假设梯度稀疏，等价于假设图像是**分段常数**的。对于真实图像中的渐变区域（如天空渐变、光影过渡），TV 会将连续变化近似为一系列阶梯——这就是阶梯效应（staircase effect）。'''
txt = txt[:i] + new_R3 + txt[j:]

io.open(p, "w", encoding="utf-8").write(txt)
print("R1/R2/R3 applied OK")

# integrity scan: no broken \nabla / \times etc.
bad = txt.count(chr(10)+"abla") + txt.count(chr(10)+"imes") + txt.count(chr(9))
print("broken-scan (should be 0):", bad)
print("intact \\nabla:", txt.count(r"\nabla"), " \\times:", txt.count(r"\times"), " \\sigma:", txt.count(r"\sigma"))
