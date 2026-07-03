# -*- coding: utf-8 -*-
"""
实验13.4-4：自定义img2img Pipeline
对应章节：13.4节 引导采样（CFG嵌入分析）

素材来源：实验13.5-步骤1+3

实验内容：
  - 自定义Stable Diffusion img2img管线
  - 显式实现CFG：uncond和cond的拼接与权重混合
  - 分析img2img流程的每一步（CLIP/VAE/UNet/Scheduler）

注意：本实验需要GPU和预训练模型下载。
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


import torch
from tqdm.auto import tqdm

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
    print("diffusers/transformers库未安装，请先安装")
    print("pip install diffusers transformers accelerate")


class CustomImg2ImgPipeline(DiffusionPipeline):
    """自定义Stable Diffusion img2img管线"""

    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        init_image,
        strength=0.8,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=None,
    ):
        batch_size = 1

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        self.scheduler.set_timesteps(num_inference_steps)
        init_latents = self.vae.encode(
            init_image.to(self.device)
        ).latent_dist.sample()
        init_latents = init_latents * self.vae.config.scaling_factor

        t_start = max(num_inference_steps - int(num_inference_steps * strength), 0)
        timesteps = self.scheduler.timesteps[t_start:]
        start_timestep = timesteps[0:1]

        noise = torch.randn(
            init_latents.shape, generator=generator, device=self.device
        )
        latents = self.scheduler.add_noise(init_latents, noise, start_timestep)

        for t in tqdm(timesteps, desc="img2img sampling"):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return (image[0] * 255).astype(np.uint8)


print("=" * 60)
print("步骤1：构建自定义img2img Pipeline")
print("=" * 60)

if HAS_DEPS:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if device.type == "cuda":
        try:
            model_id = "CompVis/stable-diffusion-v1-4"
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
            tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(
                model_id, subfolder="text_encoder"
            ).to(device)
            unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet"
            ).to(device)
            scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

            pipeline = CustomImg2ImgPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
            )
            print("自定义Pipeline构建成功！")
            print(f"  VAE参数: {sum(p.numel() for p in vae.parameters()) / 1e6:.0f}M")
            print(f"  UNet参数: {sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M")
            print(f"  CLIP参数: {sum(p.numel() for p in text_encoder.parameters()) / 1e6:.0f}M")

        except Exception as e:
            print(f"加载失败: {e}")
    else:
        print("⚠️ CPU上无法实际运行Pipeline构建（模型太大）")
        print("Pipeline类已定义，可参考源代码理解img2img的完整流程")
else:
    print("diffusers库未安装")


print("\n" + "=" * 60)
print("步骤2：img2img Pipeline流程分析（13.4.2节）")
print("=" * 60)

print("""
img2img Pipeline 完整流程（13.4.2节 CFG的工程实现）：

1. CLIP Text Encoder: prompt → text embeddings
2. 构造 [uncond_emb, cond_emb] 拼接（用于CFG）
3. VAE Encoder: input_image → latent z_0
4. 确定起始步 t_start = int(strength * T)
5. 加噪: z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
6. 对每个去噪步 t:
   a. UNet前向: [uncond_pred, cond_pred] = UNet(z_t, t, [uncond, cond])
   b. CFG混合: ε̂ = uncond_pred + s·(cond_pred - uncond_pred)
   c. Scheduler step: z_{t-1} = step(ε̂, t, z_t)
7. VAE Decoder: z_0 → output_image

CFG的作用（13.4.2节）：
  - ε̂_cfg = ε̂_uncond + s·(ε̂_cond - ε̂_uncond)
  - s=1: 退化为无条件/条件混合
  - s=7.5: 常用默认，平衡质量和多样性
  - s→∞: 完全忽略无约束，生成结果更"硬"对齐prompt
""")

print(f"\n{'='*60}")
print("实验13.4-4 完成!")
print("=" * 60)
print("""
关键结论:
1. img2img是逆问题求解的特例（13.6节）
   - A = 加噪算子 (z_t = √ᾱ_t·z_0 + √(1-ᾱ_t)·ε)
   - y = 噪声化后的潜空间表示
   - 文本prompt通过CFG提供额外约束

2. CFG的工程实现（13.4.2节）
   - 显式构造uncond和cond的拼接
   - 在UNet前向后做一次线性混合
   - 与DPS的 ζ·似然梯度 在数学上形式类似，但作用对象不同
""")
