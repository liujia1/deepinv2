# -*- coding: utf-8 -*-
"""
实验13.6-4：自定义img2img Pipeline

实验内容：
  - 自定义Stable Diffusion img2img管线
  - 显式实现CFG：uncond和cond的拼接与权重混合
  - 分析img2img流程的每一步（CLIP/VAE/UNet/Scheduler）

注意：本实验需要下载约4GB的SD1.4预训练权重；GPU为推荐设备
      （无CUDA时CPU也可运行，但512x512单次采样可能需要数十分钟）。
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.6-4')
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
print("实验13.6-4: 自定义img2img Pipeline")
print("=" * 60)
print("知识点: img2img管线, CFG显式实现, uncond/cond拼接")


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


if HAS_DEPS:
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

            # 计算起始步：strength越大，保留原图结构越少
            # 与diffusers官方StableDiffusionImg2ImgPipeline.get_timesteps等价：
            #   init_timestep = min(int(N*strength), N); t_start = max(N-init_timestep, 0)
            # 边界情况：strength=0时t_start=num_inference_steps，timesteps[t_start:]为空
            # 实际应用中strength通常>0，此边界case可忽略或显式添加assert strength > 0
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
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
                # 显式CFG：eps_cfg = eps_uncond + s*(eps_cond - eps_uncond)
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
else:
    # 依赖缺失时不可实例化；后续流程分析文本仍可正常阅读
    CustomImg2ImgPipeline = None


print("=" * 60)
print("步骤1：构建自定义img2img Pipeline")
print("=" * 60)

# ===== 实验配置（无论是否实际生成都作为"预期配置"记录）=====
MODEL_ID = "CompVis/stable-diffusion-v1-4"
PROMPT = "purple gradient sunset over mountains, high quality"
STRENGTH = 0.8
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
SEED = 42
compare_path = os.path.join(SAVE_DIR, "img2img对比.png")

# 运行状态：只有真正生成并保存图像后才置True，观察结论仅在该状态下落盘
generation_ok = False
gen_error = ""
skip_reason = ""
device = None

if HAS_DEPS:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type != "cuda":
        print("警告: 未检测到CUDA，CPU(fp32)可以运行但512x512单次采样可能需要数十分钟")

    try:
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder"
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet"
        ).to(device)
        scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

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

        # ---- 实际运行示例：生成一张img2img结果 ----
        print("\n正在生成img2img对比图...")
        from PIL import Image

        # 创建一个简单的初始图像（渐变背景）- 使用numpy高效生成
        x = np.linspace(0, 255, 512)
        y = np.linspace(0, 255, 512)
        xx, yy = np.meshgrid(x, y)
        # 创建蓝紫渐变：R随x变化，G随y变化，B随x变化
        r = (100 + xx * 0.2).astype(np.uint8)
        g = (50 + yy * 0.1).astype(np.uint8)
        b = (150 + xx * 0.1).astype(np.uint8)
        init_array = np.stack([r, g, b], axis=-1)
        init_image = Image.fromarray(init_array)

        # 注意：SD1.4的CLIP编码器仅支持英文提示词，中文输入会导致tokenizer拆解错误
        print(f"  Prompt: '{PROMPT}' (仅支持英文)")
        print(f"  Strength: {STRENGTH}")
        print(f"  Guidance_scale: {GUIDANCE_SCALE}")

        # 转换为tensor格式并归一化到[-1,1]，匹配VAE训练时的输入约定
        init_image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
        init_image_tensor = init_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        init_image_tensor = init_image_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

        # 生成图像
        output_image = pipeline(
            prompt=PROMPT,
            init_image=init_image_tensor,
            strength=STRENGTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(device=device).manual_seed(SEED),
        )

        # 对比展示
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(init_image)
        axes[0].axis('off')
        axes[0].set_title('初始图像（渐变背景）')
        result_img = Image.fromarray(output_image)
        axes[1].imshow(result_img)
        axes[1].axis('off')
        axes[1].set_title(f'img2img结果 (strength={STRENGTH})')
        plt.suptitle('img2img Pipeline效果演示', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        generation_ok = True
        print(f"对比图已保存: {compare_path}")

    except Exception as e:
        # 不吞异常、不伪装成功：记录类型与信息，由末尾JSON如实落盘run_status=skipped
        gen_error = f"{type(e).__name__}: {e}"
        print(f"加载或生成失败（未生成结果图）: {gen_error}")
else:
    skip_reason = "diffusers/transformers/accelerate 未安装，已跳过实际生成"
    print("diffusers库未安装，跳过实际生成（Pipeline完整流程可参考步骤2的文本分析）")


print("\n" + "=" * 60)
print("步骤2：img2img Pipeline流程分析（13.4.2节）")
print("=" * 60)

print("""
img2img Pipeline 完整流程（13.4.2节 CFG的实践应用）：

1. CLIP Text Encoder: prompt -> text embeddings
2. 构造 [uncond_emb, cond_emb] 拼接（用于CFG）
3. VAE Encoder: input_image -> latent z_0
4. 确定起始步 t_start = N - ⌊N × strength⌋（timesteps[t_start:]为实际采样段）
5. 加噪: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1-alpha_bar_t) * eps
6. 对每个去噪步 t:
   a. UNet前向: [uncond_pred, cond_pred] = UNet(z_t, t, [uncond, cond])
   b. CFG混合: eps_hat = uncond_pred + s * (cond_pred - uncond_pred)
   c. Scheduler step: z_{t-1} = step(eps_hat, t, z_t)
7. VAE Decoder: z_0 -> output_image

CFG的作用（13.4.2节）：
  - eps_hat_cfg = eps_hat_uncond + s * (eps_hat_cond - eps_hat_uncond)
  - s=1: 标准条件生成（相当于不使用CFG增强，直接用条件预测eps_hat_cond）
  - s=7.5: 常用默认，平衡质量和多样性
  - s→INF: 无条件项系数(1-s)→-∞，通过外推放大cond与uncond的差异方向
""")

print("\n" + "=" * 60)
print(f"实验13.6-4 完成! (run_status={'generated' if generation_ok else 'skipped'})")
print("=" * 60)
print("""
关键结论:
1. img2img的机制（非严格的逆问题求解）
   - z_0（init_image的latent）是已知且完整的，不存在退化算子A或测量约束
   - 本质：将init_image加噪到中间时刻t_start，再从该点开始反向去噪
   - strength控制起始噪声水平，从而控制"保留原图结构"与"服从文本引导"的权衡
   - 与DPS/DiffPIR的区别：DPS/DiffPIR在每步显式加入似然梯度项∇log p(y|x_t)，img2img无此机制

2. CFG的实践应用（13.4.2节）
   - 显式构造uncond和cond的拼接
   - 在UNet前向后做一次线性混合
   - 与DPS的 zeta*似然梯度 在数学上形式类似，但作用对象不同
""")
# ===== 保存运行摘要（run_status与观察结论必须如实反映本次运行）=====
import json
from datetime import datetime

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
    except Exception:
        pass
    return obj

results_summary = {
    "run_status": "generated" if generation_ok else "skipped",
    "note": "定性视觉对比实验（无量化指标）。run_status=skipped表示本次运行环境不具备条件或加载/采样抛错，存档图须在具备依赖的环境（推荐GPU）实际生成；只有run_status=generated时下列观察才与本次运行对应。",
    "模型": f"{MODEL_ID} (自定义CustomImg2ImgPipeline + DDIMScheduler)",
    "device": str(device) if device is not None else "n/a",
    "初始图": "代码内numpy生成的512x512蓝紫渐变（无外部图片依赖）",
    "prompt": PROMPT,
    "strength": STRENGTH,
    "guidance_scale": GUIDANCE_SCALE,
    "seed": SEED,
    "num_inference_steps": NUM_INFERENCE_STEPS,
    "加噪起点": "int(50*0.8)=40, t_start=50-40=10, 起始timestep=799, alpha_bar=0.037, 实际逆向去噪40步（初始潜变量信号幅值约19%）",
    "CFG实现": "显式uncond/cond拼接 + 线性混合 eps_uncond+s*(eps_cond-eps_uncond)，与diffusers官方StableDiffusionImg2ImgPipeline逐项等价",
}
if generation_ok:
    results_summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    results_summary["存档图"] = os.path.basename(compare_path)
    results_summary["观察"] = (
        "应逐面板对照存档图核实：输出在紫色天空下生成暗色山脉剪影，"
        "prompt中mountains语义达成、初始紫色调保留，sunset仅弱体现（右侧微暖）；"
        "画面整体平滑与prompt含gradient、平滑渐变初始图以及strength=0.8高噪声起点有关，"
        "不能仅凭单次seed采样外推为img2img的固有上界。"
    )
else:
    results_summary["skip_reason"] = skip_reason or "模型加载或采样过程中抛出异常"
    if gen_error:
        results_summary["error"] = gen_error

results_summary = _to_native(results_summary)
json_path = os.path.join(SAVE_DIR, 'results_summary.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"运行摘要已保存: {json_path} (run_status={results_summary['run_status']})")

