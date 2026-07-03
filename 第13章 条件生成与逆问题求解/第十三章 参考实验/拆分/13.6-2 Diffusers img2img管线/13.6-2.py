# -*- coding: utf-8 -*-
"""
实验13.6-2：Diffusers img2img管线（作为逆问题求解的工程范例）
对应章节：13.6节 闭环：回到逆问题

素材来源：实验13.4-步骤2（img2img管线）

实验内容：
  - Diffusers库中的img2img管线
  - 理解img2img作为"加噪逆问题"的特例
  - 从DPS后验采样角度重新解读img2img

注意：本实验需要GPU和预训练模型。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
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
print("步骤1：img2img作为条件扩散的工程实现（13.6节）")
print("=" * 60)

print("""
13.6节：img2img是"加噪逆问题"的特例

img2img 视角下的逆问题分解：
  - 正向算子 A: 把"清晰图像"加噪到t_start对应的噪声水平
    y = A(x_0) = √ᾱ_{t_start}·x_0 + √(1-ᾱ_{t_start})·ε
  - 观测 y: 噪声化后的潜空间表示
  - 条件 y: 文本prompt（通过CFG提供额外约束）
  - 求解: 从p(z|y, prompt)采样 = img2img生成

类比DPS：
  DPS: 在每步加似然梯度，保持数据一致性
  img2img: 直接以加噪后的y作为起点（无需显式似然梯度）
""")

import torch
try:
    from diffusers import StableDiffusionImg2ImgPipeline
    import PIL.Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("diffusers库未安装")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

if HAS_DEPS and device == "cuda":
    try:
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

        # 准备测试图像
        init_image_path = os.path.join(SAVE_DIR, "sketch.png")
        if os.path.isfile(init_image_path):
            init_image = PIL.Image.open(init_image_path).resize((512, 512))
            for strength in [0.3, 0.6, 0.9]:
                result = img2img(
                    prompt="a beautiful oil painting",
                    image=init_image,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=25
                ).images[0]
                result.save(os.path.join(SAVE_DIR, f"img2img_strength_{strength}.png"))
            print("img2img结果已保存")
        else:
            print(f"未找到{init_image_path}，请放置512x512图像后再运行")

    except Exception as e:
        print(f"执行出错: {e}")
else:
    print("跳过实际运行（需要GPU）")


print("\n" + "=" * 60)
print("步骤2：img2img与DPS的统一视角（13.6节）")
print("=" * 60)

print("""
img2img vs DPS 的统一视角（13.6节闭环）：

| 维度       | DPS（13.3.2节）              | img2img（13.4/13.6节）
|------------|----------------------------|-------------------------
| 条件来源   | 观测y（似然梯度）            | 加噪y + 文本prompt
| 条件注入   | 似然梯度修正SDE              | t_start + CFG
| 引导强度   | ζ（数据一致性权重）          | guidance_scale（CFG）
| 求解对象   | 物理逆问题（去噪/去模糊）    | 视觉风格转换
| 零样本迁移 | 是（不同A都能处理）          | 隐式（通过strength调节）

两者都是13.2.2节后验得分分解的工程实现：
  ∇log p(x_t|y) = ∇log p(x_t) + ζ·∇log p(y|x_t)
""")

print(f"\n{'='*60}")
print("实验13.6-2 完成!")
print("=" * 60)
print("""
关键结论:
1. img2img是条件扩散的特例（13.6节）
   - 正向算子A是"加噪"，观测y是噪声化后的图像
   - strength控制加噪程度（=逆向SDE的起点）
   - text prompt通过CFG提供额外约束

2. 闭环视角（13.6节）
   - 第1章：贝叶斯逆问题 y = Ax + n
   - 第7章：Score SDE（无条件）
   - 第13章：条件扩散（统一框架）
   - img2img是这一框架在视觉任务上的工程实现
""")
