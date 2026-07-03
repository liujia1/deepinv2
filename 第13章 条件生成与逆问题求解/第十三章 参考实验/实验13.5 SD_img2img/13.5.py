"""
实验13.5：Stable Diffusion img2img管线构建（D8）
对应章节：13.4（引导采样）、13.6（闭环：回到逆问题）
参考素材：scripts/img2img.py (Diffusion_models_tutorial-main)

注意：本实验需要GPU（VRAM≥4GB）和预训练模型下载。
"""
import torch
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("WARNING: Running on CPU. This will be extremely slow.")
    print("The markdown file provides the code and expected results for reference.")


class CustomImg2ImgPipeline(DiffusionPipeline):
    """自定义Stable Diffusion img2img管线（参考scripts/img2img.py）"""

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


print("\n=== Step 1: Building Custom img2img Pipeline ===")
print("Loading SD v1.4 components...")

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
    print("Custom pipeline built successfully!")
    print(f"  VAE params: {sum(p.numel() for p in vae.parameters()) / 1e6:.0f}M")
    print(f"  UNet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.0f}M")
    print(f"  CLIP params: {sum(p.numel() for p in text_encoder.parameters()) / 1e6:.0f}M")

    print("\n=== Step 2: Strength Parameter Effects ===")
    print("To run img2img, place a 512x512 image named 'input.png' and run:")
    img2img_demo = '''
init_image = PIL.Image.open("input.png").resize((512, 512))
init_tensor = torch.from_numpy(
    np.array(init_image) / 255.0 * 2.0 - 1.0
).permute(2, 0, 1).unsqueeze(0).float()

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
'''
    print(img2img_demo)

    print("\n=== Step 3: CFG Guidance in img2img ===")
    cfg_demo = '''
for gs in [1, 3, 7.5, 15]:
    result = pipeline(
        prompt="a beautiful landscape painting",
        init_image=init_tensor,
        strength=0.6,
        guidance_scale=gs,
        num_inference_steps=25,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    PIL.Image.fromarray(result).save(f"img2img_cfg{gs}.png")
'''
    print(cfg_demo)

except Exception as e:
    print(f"Skipping SD loading: {e}")
    print("This is expected if running on CPU or without internet access.")
    print("The markdown file provides the code and expected results for reference.")

print("\n=== img2img Pipeline Flow ===")
print("1. CLIP Text Encoder: prompt -> text embeddings")
print("2. VAE Encoder: input image -> latent z_0")
print("3. Determine t_start = int(strength * T)")
print("4. Add noise: z_t = sqrt(alpha_bar) * z_0 + sqrt(1-alpha_bar) * eps")
print("5. CFG denoising: eps = uncond + guidance_scale * (cond - uncond)")
print("6. VAE Decoder: latent z_0 -> output image")

print("\nDone!")