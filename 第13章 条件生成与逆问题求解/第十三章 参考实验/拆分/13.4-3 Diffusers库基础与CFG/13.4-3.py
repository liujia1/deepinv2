# -*- coding: utf-8 -*-
"""
实验13.4-3：Diffusers库基础与CFG
对应章节：13.4节 引导采样（CFG引导）

素材来源：实验13.4-步骤1+3

实验内容：
  - Diffusers库基础（Pipeline/UNet/Scheduler架构）
  - 文本到图像生成（Stable Diffusion v1.4）
  - CFG（Classifier-Free Guidance）引导对比

注意：本实验需要GPU和预训练模型下载（约4GB），CPU上运行极慢。
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
print("步骤1：Diffusers库基础（Pipeline/UNet/Scheduler）")
print("=" * 60)

print("""
Diffusers库架构（13.4节）：
  Pipeline (高层API)
    ↓
  Model (UNet2DConditionModel, VAE, CLIP) + Scheduler (DDIM, PNDM)

主要组件：
  - UNet2DConditionModel: 预测噪声残差 ε̂_θ(x_t, t, c)
  - DDIMScheduler/PNDMScheduler: 控制去噪步
  - VAE (AutoencoderKL): 潜空间编码/解码
  - CLIP Text Encoder: 文本 → 嵌入向量
""")

import torch
try:
    from diffusers import (
        StableDiffusionPipeline,
        DDIMScheduler,
        UNet2DConditionModel,
    )
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    print("Diffusers库未安装，请先安装: pip install diffusers transformers accelerate")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cpu":
    print("⚠️ 警告: CPU上运行极慢，建议使用GPU")

if HAS_DIFFUSERS and device == "cuda":
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe = pipe.to(device)

        prompt = "a cat wearing sunglasses, high quality"
        print(f"\n生成图像: '{prompt}'")
        image = pipe(prompt, num_inference_steps=25).images[0]
        image.save(os.path.join(SAVE_DIR, "文生图示例.png"))
        print("文本到图像结果已保存")

        # ---- CFG引导对比 ----
        print("\n" + "=" * 60)
        print("步骤2：CFG引导对比（13.4.2节 Classifier Guidance与CFG）")
        print("=" * 60)

        guidance_scales = [1, 3, 7.5, 15]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for idx, gs in enumerate(guidance_scales):
            img = pipe(prompt,
                       num_inference_steps=25,
                       guidance_scale=gs,
                       generator=torch.Generator(device=device).manual_seed(42)
                       ).images[0]
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'guidance_scale={gs}')

        plt.suptitle('CFG Guidance Scale 效果对比（13.4.2节）', fontsize=14, y=1.02)
        plt.tight_layout()
        cfg_path = os.path.join(SAVE_DIR, "CFG引导对比.png")
        plt.savefig(cfg_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"CFG对比图已保存: {cfg_path}")

    except Exception as e:
        print(f"执行出错: {e}")
        print("可能原因：GPU内存不足、网络无法下载模型等")
else:
    print("\n跳过实际运行（需要GPU + Diffusers + 网络）")
    print("以下为理论说明：")

    print("""
13.4.2节 CFG（Classifier-Free Guidance）核心：
  ε̂_cfg = ε̂_uncond + s · (ε̂_cond - ε̂_uncond)
  其中 s = guidance_scale

  s=1:  无条件生成（cond和uncond混合）
  s>1:  强引导（更接近文本描述，但多样性下降）
  s=7.5: 常用默认值
  s→∞: 完全忽略无条件分布，类似"硬约束"
""")

print(f"\n{'='*60}")
print("实验13.4-3 完成!")
print("=" * 60)
print("""
关键结论:
1. Diffusers库架构（13.4节）
   - Pipeline是高层API，组合Model + Scheduler
   - UNet2DConditionModel是核心得分网络
   - VAE/CLIP分别处理图像和文本
   - Scheduler控制去噪步（DDIM/PNDM等）

2. CFG（13.4.2节）
   - ε̂_cfg = ε̂_uncond + s·(ε̂_cond - ε̂_uncond)
   - s控制cond和uncond的混合比例
   - s=1：无引导；s=7.5：常用默认；s>10：强引导
   - 与13.4.3节的ζ类比：s对应条件生成的引导强度
""")
