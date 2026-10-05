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
            # 注意：sample(generator)必须传入同一个generator，否则z_0采样走全局随机流，
            # 相同seed的两次运行也会得到不同结果（diffusers官方retrieve_latents即如此传入）
            init_latents = self.vae.encode(
                init_image.to(self.device)
            ).latent_dist.sample(generator)
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
            # 记录本次实际使用的调度量（不同diffusers版本timestep取整/offset约定略有差异）
            self.last_start_timestep = int(start_timestep.item())
            self.last_denoise_steps = int(len(timesteps))
            self.last_alpha_bar = float(
                self.scheduler.alphas_cumprod[self.last_start_timestep].item()
            )

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
# prompt变更（2026-10-05，有证据支持的单因子调整，strength等其余配置一律不动）：
# 旧prompt "purple gradient sunset over mountains, high quality" 中的 "gradient" 是自描述词，
# 模型忠实地渲染"紫色渐变"，在无结构的平滑初始图上压制了mountains语义——
# diffusers 0.40.0下4/4种子与官方管线（MAE=0.50/255, corr=0.9999）均稳定输出平滑渐变。
# 新prompt去掉gradient、强化山的结构词，保留purple以与初始图紫色调的保留演示相洽。
PROMPT = ("a dramatic sunset over dark mountain silhouettes, "
          "purple and orange sky with glowing clouds, "
          "highly detailed landscape photo")
OLD_PROMPT = "purple gradient sunset over mountains, high quality"
OLD_STRENGTH = 0.8
# strength变更（2026-10-05，第二轮单因子调整，新prompt及其余配置全部冻结）：
# 新prompt在strength=0.8下4/4种子出现天空/云层结构，但只有2/4出现山脉（seed=123清晰剪影、
# seed=2024小山峰），未达预设的≥3/4稳健线。提高strength到0.9给文本引导更大去噪空间，
# 代价是初始图保留更少（初始图本身无结构，可接受）。验收仍以4种子扫描≥3/4出现可辨识山脉为准。
STRENGTH = 0.9
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
SEED = 42
# 多种子扫描：img2img结果对随机采样敏感（平滑渐变初始图下尤甚），
# 单一样本无法代表算法行为，故固定其余配置、只换种子做小扫描
SEED_SCAN = [42, 7, 123, 2024]
compare_path = os.path.join(SAVE_DIR, "img2img对比.png")
sweep_path = os.path.join(SAVE_DIR, "img2img_seed扫描.png")

# 运行状态：只有真正生成并保存图像后才置True，观察结论仅在该状态下落盘
generation_ok = False
gen_error = ""
skip_reason = ""
device = None
output_stats = None
actual_start_t = None
actual_denoise_steps = None
actual_alpha_bar = None
sweep_results = []
sweep_ok = False
sweep_error = ""
dep_versions = {}
# 官方管线对照（同一环境/权重/种子，判别是手写实现问题还是库版本行为）
official_ok = False
official_stats = None
official_mae = None
official_corr = None
official_error = ""
official_path = os.path.join(SAVE_DIR, "img2img_官方管线对照.png")

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

        # ---- 准备初始图像（渐变背景，所有种子共用）----
        from PIL import Image, ImageFilter
        from importlib.metadata import version as _pkg_version
        for _pkg in ("diffusers", "transformers", "torch", "accelerate"):
            try:
                dep_versions[_pkg] = _pkg_version(_pkg)
            except Exception:
                dep_versions[_pkg] = "unknown"
        print(f"  依赖版本: {dep_versions}")

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

        def compute_stats(out_img):
            """从输出图像自动计算定量特征（不写主观结论，语义需人工对照存档图判读）"""
            f = out_img.astype(np.float32)
            blur = np.asarray(
                Image.fromarray(out_img).filter(ImageFilter.GaussianBlur(4))
            ).astype(np.float32)
            h = f.shape[0]
            band = f[int(h*0.55):int(h*0.85)].mean(axis=(1, 2))
            return {
                "mean_RGB": [round(float(v), 2) for v in f.reshape(-1, 3).mean(0)],
                "grain_std": round(float((f - blur).std()), 3),
                "edge_y_mean": round(float(np.abs(np.diff(f, axis=0)).mean()), 3),
                # 55%-85%高度带内水平亮度骤降的最大幅度，仅作"可能存在横贯山脊"的线索
                "ridge_grad_min": round(float(np.diff(band).min()), 3),
                "dark_frac(<110)_top_mid_bottom": [
                    round(float((f[:h//3].mean(2) < 110).mean()), 3),
                    round(float((f[h//3:2*h//3].mean(2) < 110).mean()), 3),
                    round(float((f[2*h//3:].mean(2) < 110).mean()), 3),
                ],
            }

        def generate(seed):
            print(f"\n[seed={seed}] 采样中...")
            out_img = pipeline(
                prompt=PROMPT,
                init_image=init_image_tensor,
                strength=STRENGTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            stats = compute_stats(out_img)
            print(f"[seed={seed}] stats={stats}")
            return out_img, stats

        # ---- 主对比图：seed=42 ----
        print("\n正在生成img2img对比图(seed=42)...")
        output_image, output_stats = generate(SEED)
        actual_start_t = pipeline.last_start_timestep
        actual_denoise_steps = pipeline.last_denoise_steps
        actual_alpha_bar = pipeline.last_alpha_bar
        print(f"  实际起始timestep={actual_start_t}, alpha_bar={actual_alpha_bar:.4f}, "
              f"逆向去噪{actual_denoise_steps}步")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(init_image)
        axes[0].axis('off')
        axes[0].set_title('初始图像（渐变背景）')
        axes[1].imshow(Image.fromarray(output_image))
        axes[1].axis('off')
        axes[1].set_title(f'img2img结果 (strength={STRENGTH}, seed={SEED})')
        plt.suptitle('img2img Pipeline效果演示', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        generation_ok = True
        print(f"对比图已保存: {compare_path}")

        # ---- 多种子扫描：刻画采样敏感性（非挑选好看的样本）----
        print(f"\n开始多种子扫描: {SEED_SCAN}")
        try:
            fig2, axes2 = plt.subplots(1, len(SEED_SCAN) + 1,
                                       figsize=(4 * (len(SEED_SCAN) + 1), 4.2))
            axes2[0].imshow(init_image)
            axes2[0].axis('off')
            axes2[0].set_title('初始图像')
            for _i, _seed in enumerate(SEED_SCAN):
                try:
                    if _seed == SEED:
                        _out, _stats = output_image, output_stats  # 复用，不重复采样
                    else:
                        _out, _stats = generate(_seed)
                    sweep_results.append({"seed": _seed, "stats": _stats})
                    axes2[_i + 1].imshow(Image.fromarray(_out))
                    axes2[_i + 1].set_title(f'seed={_seed}')
                except Exception as e:
                    sweep_error = (sweep_error + " | " if sweep_error else "") \
                        + f"seed={_seed}: {type(e).__name__}: {e}"
                    print(f"扫描失败: {sweep_error}")
                    axes2[_i + 1].set_title(f'seed={_seed}\n(失败)')
                axes2[_i + 1].axis('off')
            plt.suptitle(f'img2img多种子扫描 (strength={STRENGTH}, gs={GUIDANCE_SCALE}, '
                         f'同一初始图与prompt，仅换种子)', fontsize=13)
            plt.tight_layout()
            plt.savefig(sweep_path, dpi=150, bbox_inches='tight')
            plt.close()
            sweep_ok = len(sweep_results) == len(SEED_SCAN)
            print(f"种子扫描图已保存: {sweep_path} (成功{len(sweep_results)}/{len(SEED_SCAN)})")
        except Exception as e:
            plt.close()
            sweep_error = f"扫描图绘制失败: {type(e).__name__}: {e}"
            print(f"种子扫描未完成（不影响主对比图）: {sweep_error}")

        # ---- 官方管线对照：同环境/权重/种子，验证手写实现与官方是否数值等价 ----
        print("\n开始官方StableDiffusionImg2ImgPipeline对照(seed=42)...")
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            # 复用同一批已加载权重，仅给官方一个全新scheduler实例（避免step_index状态共享）
            _fresh_sched = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
            _common = dict(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                           unet=unet, scheduler=_fresh_sched)
            try:
                # 不同diffusers版本对safety_checker参数签名要求不一，先显式禁用
                official_pipe = StableDiffusionImg2ImgPipeline(
                    safety_checker=None, feature_extractor=None,
                    requires_safety_checker=False, **_common
                ).to(device)
            except TypeError:
                official_pipe = StableDiffusionImg2ImgPipeline(**_common).to(device)
            official_out = np.array(official_pipe(
                prompt=PROMPT,
                image=init_image,  # PIL，官方内部自行预处理（图已是512x512）
                strength=STRENGTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator(device=device).manual_seed(SEED),
            ).images[0])
            official_stats = compute_stats(official_out)
            _a = output_image.astype(np.float32)
            _b = official_out.astype(np.float32)
            official_mae = round(float(np.abs(_a - _b).mean()), 3)
            official_corr = round(float(np.corrcoef(_a.ravel(), _b.ravel())[0, 1]), 4)
            print(f"  官方管线stats={official_stats}")
            print(f"  手写vs官方: MAE={official_mae}/255, corr={official_corr}")

            fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
            axes3[0].imshow(init_image)
            axes3[0].set_title('初始图像')
            axes3[1].imshow(output_image)
            axes3[1].set_title(f'手写管线 (seed={SEED})')
            axes3[2].imshow(official_out)
            axes3[2].set_title(f'官方diffusers管线 (seed={SEED})')
            for _ax in axes3:
                _ax.axis('off')
            plt.suptitle(f'手写vs官方管线对照 (MAE={official_mae}/255, corr={official_corr})',
                         fontsize=13)
            plt.tight_layout()
            plt.savefig(official_path, dpi=150, bbox_inches='tight')
            plt.close()
            official_ok = True
            print(f"官方对照图已保存: {official_path}")
        except Exception as e:
            official_error = f"{type(e).__name__}: {e}"
            print(f"官方管线对照失败（不影响其余产物）: {official_error}")

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
    "prompt变更记录": {
        "旧prompt": OLD_PROMPT,
        "新prompt": PROMPT,
        "变更日期": "2026-10-05",
        "变更原因": "旧prompt中的自描述词gradient引导模型渲染平滑渐变，在无结构渐变初始图上压制mountains语义；"
                    "diffusers 0.40.0下4/4种子及官方管线均无山脉（手写vs官方MAE=0.50/255, corr=0.9999，已排除实现问题）",
        "实验纪律": "仅改prompt单因子，strength/guidance_scale/步数/初始图/种子全部冻结；"
                    "是否恢复山脉语义以本次seed扫描4个面板的人工判读为准，不据此挑选单个种子",
    },
    "strength": STRENGTH,
    "strength变更记录": {
        "原值": OLD_STRENGTH,
        "新值": STRENGTH,
        "变更日期": "2026-10-05",
        "变更原因": "新prompt在0.8下4/4出天空结构但仅2/4出山脉，未达预设≥3/4稳健线；"
                    "提高strength给文本引导更大去噪空间（初始图本身无空间结构，保留更少可接受）",
        "实验纪律": "仅改strength单因子，prompt/guidance_scale/步数/初始图/种子全部冻结；"
                    "以4种子扫描≥3/4出现可辨识暗色山脊为验收线，不以单个种子下结论",
    },
    "guidance_scale": GUIDANCE_SCALE,
    "seed": SEED,
    "num_inference_steps": NUM_INFERENCE_STEPS,
    "加噪起点": (
        f"实际起始timestep={actual_start_t}, alpha_bar={actual_alpha_bar:.4f}, "
        f"逆向去噪{actual_denoise_steps}步（timestep取整/offset随diffusers版本略有差异，以本次实测为准）"
    ) if generation_ok else (
        f"配置值: num_inference_steps={NUM_INFERENCE_STEPS}, strength={STRENGTH} "
        f"→ 从t_start={NUM_INFERENCE_STEPS - int(NUM_INFERENCE_STEPS * STRENGTH)}起"
        f"去噪约{int(NUM_INFERENCE_STEPS * STRENGTH)}步, "
        "实际timestep以scheduler运行时为准"),
    "CFG实现": "显式uncond/cond拼接 + 线性混合 eps_uncond+s*(eps_cond-eps_uncond)，与diffusers官方StableDiffusionImg2ImgPipeline逐项等价",
    "随机源": "VAE latent采样与加噪均传入同一个generator，同环境同版本下结果可复现",
    "依赖版本": dep_versions,
}
if generation_ok:
    results_summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    results_summary["存档图"] = os.path.basename(compare_path)
    results_summary["output_stats"] = output_stats
    results_summary["seed扫描配置"] = {
        "seeds": SEED_SCAN,
        "strength": STRENGTH,
        "guidance_scale": GUIDANCE_SCALE,
        "其余配置": "与主对比图完全一致，仅换种子",
        "输出图": os.path.basename(sweep_path) if os.path.exists(sweep_path) else None,
    }
    results_summary["seed扫描"] = {
        "状态": "ok" if sweep_ok else "partial",
        "成功数": len(sweep_results),
        "总数": len(SEED_SCAN),
        "逐种子stats": sweep_results,
        "error": sweep_error,
    }
    results_summary["官方管线对照"] = {
        "目的": "同一diffusers环境、同一权重、同一seed下用官方StableDiffusionImg2ImgPipeline对照，"
                "判别山脉缺失是手写实现问题还是该库版本的模型行为",
        "状态": "ok" if official_ok else "failed",
        "输出图": os.path.basename(official_path) if os.path.exists(official_path) else None,
        "官方stats": official_stats,
        "手写vs官方_MAE(0-255)": official_mae,
        "手写vs官方_相关系数": official_corr,
        "判读": (
            "MAE很小（个位数/255）且corr接近1：手写实现在该版本下数值等价，可排除实现bug，"
            "山脉缺失应归因于库版本/模型行为（下一步可pin diffusers==0.27.2对照）；"
            "MAE明显大：需逐环节排查手写管线与官方在当前版本的API差异；"
            "无论MAE大小，若官方输出同样无山脉，则与手写实现无关。"
        ) if official_ok else None,
        "error": official_error,
    }
    results_summary["观察"] = (
        "output_stats与seed扫描stats均由图像自动计算，语义/结构达成度（如prompt中mountains是否可见）"
        "必须逐面板对照两张存档图人工判读，脚本不替代人工结论。"
        "判读要点：①主图与4个种子面板应一起看，若有的种子出山脉、有的只有平滑渐变，"
        "说明该配置下结果存在采样方差，单一样本不能外推为算法固有上界；"
        "②若所有种子都无山脉，再考虑diffusers版本（本文件记录版本号与实际timestep）、"
        "prompt措辞或初始图信息量，不能靠反复挑种子掩盖；"
        "③ridge_grad_min仅为横贯边缘的线索指标，grain_std低只说明画面平滑。"
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

