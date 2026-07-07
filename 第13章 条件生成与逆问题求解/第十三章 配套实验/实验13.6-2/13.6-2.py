# -*- coding: utf-8 -*-
"""
实验13.6-2：Diffusers img2img管线
对应章节：13.6节 闭环：回到逆问题求解

实验内容：
  - 使用Diffusers的StableDiffusionImg2ImgPipeline
  - img2img与逆问题求解的本质区别：z_0已知完整 vs y有信息损失
  - 扩散模型引入外部信息的多种方式：初始化位置、似然梯度、CFG

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
print("知识点: img2img与逆问题求解的本质区别, 扩散模型统一框架")


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
print("步骤1：img2img与逆问题求解器的对比（13.6节）")
print("=" * 60)

print("""
img2img与真正逆问题求解器（DPS/DiffPIR）的对比（而非"img2img是逆问题的特例"）：
  - DPS/DiffPIR: 存在真实的退化y=A(x)+n，A有信息损失，需要用似然梯度∇log p(y|x_t)引导采样以恢复x
  - img2img: z_0已知且完整，无信息损失，"加噪"只是选择一个中间噪声水平作为采样起点，
    不涉及似然约束，本质是用strength参数在"忠于原图"与"服从文本"之间做插值
  - 两者共享的框架元素：都利用同一个预训练扩散模型的先验，只是"引入外部信息"的方式不同
    （img2img通过初始化位置，DPS/DiffPIR通过每步的似然梯度修正）

CFG与DPS对比的进一步说明：
  - CFG: 两次模型输出的线性插值/外推（同一UNet对同一x_t的不同条件预测）
  - DPS: 在score预测基础上加一个来自似然函数梯度的修正项（梯度来自观测约束）
  - 共同点：都是在原始预测基础上叠加一个加权修正项
  - 本质差异：修正项来源不同（CFG来自同一模型的条件差异，DPS来自外部似然约束）
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

            plt.suptitle('img2img与文本引导生成对比（13.6节）', fontsize=13, y=1.02)
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
    print("以下是img2img与逆问题求解器的对比理论说明：")
    print("""
img2img与逆问题求解器的对比（13.6节）：

通用逆问题形式: y = A(x) + n, 已知y和A, 求x
  - DPS/DiffPIR: A=真实退化算子（模糊、下采样等），y=观测数据，存在信息损失
  - img2img: z_0已知且完整，无真实退化，"加噪"是主动选择而非被动测量
  - 本质区别：逆问题求解是从残缺观测恢复真值，img2img是从已知起点重新生成

扩散模型框架的统一视角:
  1. 预训练扩散模型提供强大的图像先验（无条件采样能力）
  2. 引入外部信息的不同方式：
     - img2img: 通过初始化位置（strength参数）在"忠于原图"与"自由生成"间插值
     - DPS/DiffPIR: 通过每步似然梯度修正，将采样拉向满足观测约束的区域
     - CFG: 通过条件与无条件输出的差异放大文本引导
  3. 这些技术可组合使用（如img2img+CFG）
""")

print("\n" + "=" * 60)
print("实验13.6-2 完成!")
print("=" * 60)
print("""
关键结论:
1. img2img与逆问题求解的本质区别（13.6节）
   - img2img: z_0已知且完整，无真实退化，strength参数控制"忠于原图"程度
   - 逆问题求解: y=A(x)+n中y有信息损失，需似然梯度引导恢复x

2. 扩散模型统一框架的启示
   - 预训练扩散模型提供强大先验，可通过不同方式引入外部信息
   - StableDiffusionImg2ImgPipeline封装了初始化位置+CFG的组合
   - 这些技术可相互组合（如img2img+CFG），实现更灵活的生成控制
""")
