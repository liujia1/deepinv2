# -*- coding: utf-8 -*-
"""
实验13.6-2：Diffusers img2img管线
对应章节：13.6节 闭环：回到逆问题求解

实验内容：
  - 使用Diffusers的StableDiffusionImg2ImgPipeline
  - 文本到图像 + 退化算子A的逆问题框架
  - img2img作为闭环案例：扩散模型是通用的逆问题求解器

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
    SAVE_DIR = os.path.join(_gdrive, '实验13.6-2')
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
print("实验13.6-2: Diffusers img2img管线")
print("=" * 60)
print("对应章节: 13.6节 闭环：回到逆问题求解")
print("知识点: img2img作为逆问题求解器, DPS到img2img的统一")


import torch
try:
    from diffusers import StableDiffusionImg2ImgPipeline
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    print("diffusers库未安装，请先安装: pip install diffusers transformers accelerate")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")


print("=" * 60)
print("步骤1：img2img作为逆问题求解器（13.6节 统一框架）")
print("=" * 60)

print("""
13.6节核心观点：img2img是逆问题求解的特殊形式
  逆问题: y = A(x) + n
  img2img的A: 加噪算子 A(z) = sqrt(alpha_bar_t) * z + sqrt(1-alpha_bar_t) * eps
            (VLB框架下的简化前向算子)
  观测 y: 加噪后的潜空间表示
  文本prompt通过CFG提供额外约束

  通用公式: eps_hat = eps_hat_uncond + s * (eps_hat_cond - eps_hat_uncond)
  对比DPS:   eps_hat = eps_hat_theta - zeta * sqrt(1-alpha_bar_t) * grad_x
  两者在数学上形式类似，但作用对象不同
""")

if HAS_DIFFUSERS and device == "cuda":
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe = pipe.to(device)

        init_image_pil = None
        try:
            from PIL import Image
            import urllib.request
            url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
            urllib.request.urlretrieve(url, os.path.join(SAVE_DIR, "init_image.jpg"))
            init_image_pil = Image.open(os.path.join(SAVE_DIR, "init_image.jpg")).convert("RGB")
            init_image_pil = init_image_pil.resize((512, 512))
        except Exception as e:
            print(f"无法下载初始图像: {e}")

        if init_image_pil is not None:
            prompts = [
                "a beautiful landscape painting of mountains, oil painting style",
                "a photorealistic photo of mountains, sunny day",
                "a sketch of mountains with pencil",
            ]
            fig, axes = plt.subplots(1, len(prompts), figsize=(15, 5))

            for idx, prompt in enumerate(prompts):
                img = pipe(
                    prompt=prompt,
                    image=init_image_pil,
                    strength=0.75,
                    guidance_scale=7.5,
                ).images[0]
                axes[idx].imshow(img)
                axes[idx].axis('off')
                axes[idx].set_title(prompt[:30] + "..." if len(prompt) > 30 else prompt,
                                    fontsize=10)

            plt.suptitle('img2img作为闭环逆问题求解器（13.6节）', fontsize=13, y=1.02)
            plt.tight_layout()
            img2img_path = os.path.join(SAVE_DIR, "img2img闭环对比.png")
            plt.savefig(img2img_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"img2img对比图已保存: {img2img_path}")
    except Exception as e:
        print(f"执行出错: {e}")
        print("可能原因：GPU内存不足、网络无法下载模型等")
else:
    print("跳过实际运行（需要GPU + Diffusers）")
    print("以下是img2img作为逆问题求解器的理论说明：")
    print("""
img2img与逆问题求解的统一视图（13.6节）：

通用逆问题形式: y = A(x) + n, 已知y和A, 求x
  - DPS: A=任意算子（去噪、去模糊等）
  - DiffPIR: A=任意算子，交替去噪+投影
  - img2img: A=加噪算子（VLB框架），y=加噪潜空间表示
  - CFG: y=文本嵌入，A=CLIP, 通过s参数控制sampling

13.6节统一框架的启示:
  1. 同一预训练扩散模型可作为多种逆问题求解器
  2. 改变A的数学形式即可适配新问题
  3. 引导强度(s/zeta)统一控制数据一致性的强度
""")

print("\n" + "=" * 60)
print("实验13.6-2 完成!")
print("=" * 60)
print("""
关键结论:
1. img2img是逆问题求解的特殊形式（13.6节）
   - A = 加噪算子 (VLB前向算子)
   - y = 加噪后的潜空间表示
   - 文本prompt提供额外约束

2. 统一框架的工程实现
   - StableDiffusionImg2ImgPipeline封装了CFG和加噪逻辑
   - strength参数控制加噪程度（与DPS的t_start等价）
   - guidance_scale对应CFG的s（与DPS的zeta类比）
""")
