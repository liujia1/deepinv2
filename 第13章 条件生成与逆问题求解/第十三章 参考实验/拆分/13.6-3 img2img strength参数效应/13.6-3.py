# -*- coding: utf-8 -*-
"""
实验13.6-3：img2img strength参数效应
对应章节：13.6节 闭环：回到逆问题

素材来源：实验13.5-步骤2

实验内容：
  - img2img的strength参数：控制加噪程度
  - strength小：保持原图（高保真度）
  - strength大：自由生成（高多样性）
  - 与DPS的ζ类比：图文一致性 vs 多样性的权衡

注意：本实验需要GPU和预训练模型。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====== 解决中文乱码的核心代码 ======
import logging
import warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
import platform
from matplotlib.font_manager import FontManager

def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'simhei']:
            if re.search(pat, f.name.lower()):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


print("=" * 60)
print("步骤1：strength参数效应（13.6节）")
print("=" * 60)

print("""
13.6节：img2img的strength参数控制加噪程度

strength = 0.2: 轻微加噪，保留原图结构（高保真）
strength = 0.5: 中等加噪，平衡保真与变化
strength = 0.8: 重加噪，自由生成（高多样性）

与DPS的ζ类比：
  strength ↔ ζ（条件生成强度）
  strength小 = 数据一致性强 = ζ大
  strength大 = 数据一致性弱 = ζ小
""")

print("""
img2img strength的工作机制（13.6节）：

设总步数T=50, 当前strength=s
  t_start = int(T * (1 - s))
  → 加噪到噪声水平t_start
  → 逆向SDE从t_start开始去噪到0

strength=0.0: t_start=T (纯加噪)→无条件生成
strength=0.2: t_start=40 (重加噪)→大幅自由生成
strength=0.5: t_start=25 (中等)→平衡
strength=0.8: t_start=10 (轻加噪)→保留大部分原图
strength=1.0: t_start=0 (无加噪)→原图
""")

import torch
try:
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        DiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")

if HAS_DEPS and device == "cuda":
    print("\n实际运行需要SD模型组件，参考13.4-4的自定义Pipeline构建方式")
    print("然后调用以下流程：")
    print("""
    for strength in [0.2, 0.5, 0.8]:
        result = pipeline(
            prompt="a beautiful landscape painting",
            init_image=init_tensor,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=torch.Generator(device=device).manual_seed(42),
        )
        PIL.Image.fromarray(result).save(f"img2img_s{strength}.png")
    """)
else:
    print("\n跳过实际运行（需要GPU + 预训练模型）")


# ============================================================
# 步骤2：strength权衡曲线（理论分析）
# ============================================================
print("\n" + "=" * 60)
print("步骤2：strength参数的理论分析（13.6节）")
print("=" * 60)

# 模拟strength权衡
strength_values = np.linspace(0.0, 1.0, 11)
fidelity_to_init = (1 - strength_values) ** 0.5  # 与原图相似度
alignment_to_prompt = 1 - np.exp(-3 * strength_values)  # 与prompt对齐度
diversity = strength_values  # 多样性

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) strength-保真度
axes[0].plot(strength_values, fidelity_to_init, 'b-o', lw=2, markersize=6, label='原图保真度')
axes[0].plot(strength_values, alignment_to_prompt, 'r-s', lw=2, markersize=6, label='Prompt对齐度')
axes[0].plot(strength_values, diversity, 'g-^', lw=2, markersize=6, label='多样性')
axes[0].set_xlabel('strength', fontsize=12)
axes[0].set_ylabel('指标值 [0,1]', fontsize=12)
axes[0].set_title('(a) strength 多目标权衡', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='strength=0.5')

# (b) Pareto前沿
axes[1].plot(fidelity_to_init, alignment_to_prompt, 'b-o', lw=2, markersize=8)
for i, s in enumerate(strength_values[::2]):
    axes[1].annotate(f's={s:.1f}', (fidelity_to_init[::2][i], alignment_to_prompt[::2][i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)
axes[1].set_xlabel('原图保真度', fontsize=12)
axes[1].set_ylabel('Prompt对齐度', fontsize=12)
axes[1].set_title('(b) Pareto前沿：保真度 vs 对齐度', fontsize=13)
axes[1].grid(alpha=0.3)

# (c) 与DPS ζ的类比
zeta_values = np.linspace(0.0, 3.0, 11)
axes[2].plot(zeta_values, fidelity_to_init, 'b-', lw=2, label='DPS: 数据一致性')
axes[2].plot(strength_values * 3, 1 - diversity, 'r--', lw=2, label='img2img: 原图保真度')
axes[2].set_xlabel('DPS的ζ / img2img的3·strength', fontsize=12)
axes[2].set_ylabel('与"约束条件"的一致性', fontsize=12)
axes[2].set_title('(c) DPS-ζ 与 img2img-strength 类比', fontsize=13)
axes[2].legend(fontsize=10)
axes[2].grid(alpha=0.3)
axes[2].annotate('DPS: ζ大→强数据一致\nimg2img: strength小→高保真',
                xy=(0.55, 0.4), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', alpha=0.8))

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, 'img2img strength参数效应.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {fig_path}")

print(f"\n{'='*60}")
print("实验13.6-3 完成!")
print("=" * 60)
print("""
关键结论:
1. strength的物理意义（13.6节）
   - 控制加噪程度t_start = int(T·(1-strength))
   - strength=0: 从完全噪声开始（无条件）
   - strength=1: 几乎不加噪（原图）

2. 与DPS-ζ的类比
   - strength ↔ 1/ζ（反向关系）
   - 两者都是"条件强度"的旋钮
   - 工程上根据任务调整：图像编辑用高保真（strength小），风格转换用高自由（strength大）

3. 闭环视角（13.6节）
   - img2img是扩散先验在"图像到图像"任务上的工程实现
   - 与DPS共享13.2.2节的后验得分分解理论
   - 区别仅在"条件来源"和"条件注入方式"
""")
