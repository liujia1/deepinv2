# -*- coding: utf-8 -*-
"""
实验13.6-3：img2img strength参数效应
对应章节：13.6节 闭环：回到逆问题求解

实验内容：
  - 使用同一初始图像 + 不同strength参数
  - 演示图文一致性与多样性的权衡
  - strength与DPS的zeta、CFG的s的类比

注意：本实验需要GPU和预训练模型下载。
"""

import sys
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib相关警告
import logging
import warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验13.6-3')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")
# ========================================================

print("\n" + "=" * 60)
print("实验13.6-3: img2img strength参数效应")
print("=" * 60)
print("对应章节: 13.6节 闭环：回到逆问题求解")
print("知识点: strength参数, 质量-多样性权衡的工程实现")


import torch
try:
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("diffusers/PIL库未安装，请先安装")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")


print("=" * 60)
print("步骤1：strength参数与质量-多样性权衡（13.6节）")
print("=" * 60)

print("""
13.6节 strength参数含义:
  strength in [0, 1]: 控制加噪程度
  strength=0:   几乎不加噪，输出几乎等于原图（多样性极低）
  strength=0.5: 中等加噪，保留中等结构信息
  strength=1.0: 满噪声，相当于text-to-image（图文一致性强，多样性高）

与DPS和CFG的类比（13.4.3节 + 13.6节）:
  strength(大) <-> zeta(小) <-> guidance_scale(小): 弱一致性，高多样性
  strength(小) <-> zeta(大) <-> guidance_scale(大): 强一致性，低多样性
  （注意：DPS的zeta与strength的关系是反向的，因zeta控制的是修正项强度）
""")

if HAS_DEPS and device == "cuda":
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe = pipe.to(device)

        init_image_pil = None
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
            urllib.request.urlretrieve(url, os.path.join(SAVE_DIR, "init_image.jpg"))
            init_image_pil = Image.open(os.path.join(SAVE_DIR, "init_image.jpg")).convert("RGB")
            init_image_pil = init_image_pil.resize((512, 512))
        except Exception as e:
            print(f"无法下载初始图像: {e}")

        if init_image_pil is not None:
            strengths = [0.1, 0.3, 0.5, 0.75, 0.95]
            prompt = "a beautiful sunset over mountains, oil painting style"

            fig, axes = plt.subplots(1, len(strengths) + 1, figsize=(18, 4))
            axes[0].imshow(init_image_pil)
            axes[0].set_title('原图', fontsize=12)
            axes[0].axis('off')

            for idx, strength in enumerate(strengths):
                img = pipe(
                    prompt=prompt,
                    image=init_image_pil,
                    strength=strength,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=device).manual_seed(42),
                ).images[0]
                axes[idx + 1].imshow(img)
                axes[idx + 1].axis('off')
                axes[idx + 1].set_title(f'strength={strength}', fontsize=12)

            plt.suptitle(f'strength参数效应：图文一致性与多样性的权衡（13.6节）\nprompt: "{prompt}"',
                         fontsize=13, y=1.05)
            plt.tight_layout()
            strength_path = os.path.join(SAVE_DIR, "strength参数效应.png")
            plt.savefig(strength_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"strength参数对比图已保存: {strength_path}")
    except Exception as e:
        print(f"执行出错: {e}")
else:
    print("跳过实际运行（需要GPU + Diffusers）")
    print("以下是strength参数的理论说明：")
    print("""
strength参数效应（13.6节）:
  strength=0.1: 几乎保留原图，多样性低
  strength=0.5: 平衡点
  strength=0.95: 几乎重新生成，多样性高

  对应DPS的t_start:
    DPS的t_start = int(strength * T)
    t_start大 -> 起始噪声大 -> 多样性高、一致性低
""")

print("\n" + "=" * 60)
print("实验13.6-3 完成!")
print("=" * 60)
print("""
关键结论:
1. strength参数（13.6节）
   - 控制加噪程度，对应DPS的t_start
   - strength大 -> 多样性高、图文一致性低
   - strength小 -> 多样性低、保留原图

2. 统一的质量-多样性权衡
   - DPS: zeta控制似然修正强度
   - CFG: s控制条件/无条件混合
   - img2img: strength控制加噪程度
   - 三者本质上都是"先验强度 vs 条件强度"的tradeoff
""")
