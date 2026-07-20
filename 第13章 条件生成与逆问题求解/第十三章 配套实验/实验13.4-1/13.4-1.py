# -*- coding: utf-8 -*-
"""
实验13.4-3：Diffusers库基础与CFG

实验内容：
  - Diffusers库基础（Pipeline/UNet/Scheduler架构）
  - 文本到图像生成（Stable Diffusion v1.4）
  - CFG（Classifier-Free Guidance）引导对比

注意：本实验需要GPU和预训练模型下载（约4GB），CPU上运行极慢。
"""

import sys
import io
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib和diffusers相关警告
import logging
import warnings
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('diffusers').setLevel(logging.ERROR)  # 过滤Flax弃用警告
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)  # 过滤未认证请求警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*safety_checker.*")  # 过滤安全检查器禁用警告

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验13.4-3')
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
print("实验13.4-3: Diffusers库基础与CFG")
print("=" * 60)
print("知识点: Diffusers库架构, CFG核心公式, 引导强度guidance_scale")


print("""
Diffusers库架构：
  Pipeline (高层API)
    |
  Model (UNet2DConditionModel, VAE, CLIP) + Scheduler (PNDM)

主要组件：
  - UNet2DConditionModel: 预测噪声残差 eps_hat_theta(x_t, t, c)
  - Scheduler: 控制去噪步（默认PNDM）
  - VAE (AutoencoderKL): 潜空间编码/解码
  - CLIP Text Encoder: 文本 -> 嵌入向量
""")

import torch
try:
    from diffusers import (
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    print("Diffusers库未安装，请先安装: pip install diffusers transformers accelerate")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")
if device == "cpu":
    print("警告: CPU上运行极慢，建议使用GPU")

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
        image = pipe(prompt,
                     num_inference_steps=25,
                     generator=torch.Generator(device=device).manual_seed(42)
                     ).images[0]
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
  eps_hat_cfg = eps_hat_uncond + s * (eps_hat_cond - eps_hat_uncond)
  其中 s = guidance_scale

  s=0:   纯无条件生成（仅用eps_hat_uncond）
  s=1:   标准条件生成（相当于不使用CFG增强，直接用条件预测eps_hat_cond）
  s∈(1,~5):   温和引导（轻微增强条件性）
  s∈(~5,~15): 常用/较强引导（7.5为经典默认值）
  s过大(>15-20): 过度饱和、多样性显著下降
  s→INF: 无条件项系数(1-s)→-∞，通过外推放大cond与uncond的差异方向

  注：diffusers内部当guidance_scale ≤ 1.0时跳过uncond分支计算，等价于纯条件生成
""")

print("\n" + "=" * 60)
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
   - eps_hat_cfg = eps_hat_uncond + s * (eps_hat_cond - eps_hat_uncond)
   - s控制cond和uncond的混合比例：
     * s=0：纯无条件生成
     * s=1：标准条件生成（无CFG增强）
     * s∈(1,~5)：温和引导
     * s∈(~5,~15)：常用/较强引导（7.5为经典默认）
     * s>15：过度饱和
   - 与13.4.3节的zeta类比：s对应条件生成的引导强度
""")
# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    "guidance_scales": [1, 3, 7.5, 15],
    "note": "CFG引导对比实验，需要GPU+Diffusers运行",
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

