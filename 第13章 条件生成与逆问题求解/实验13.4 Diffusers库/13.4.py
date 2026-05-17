"""
实验13.4：Diffusers库条件生成实践（D2）
对应章节：13.4（引导采样）、13.6（闭环：回到逆问题）
参考素材：Diffusers_library.ipynb (Diffusion_models_tutorial-main)

注意：本实验需要GPU和预训练模型下载（约4GB），CPU上运行极慢。
"""
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("WARNING: Running on CPU. This will be extremely slow.")
    print("Consider using GPU or reducing image size.")

print("\n=== Step 1: Diffusers Library Basics ===")
print("Loading Stable Diffusion v1.4...")

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    prompt = "a cat wearing sunglasses, high quality"
    print(f"Generating: '{prompt}'")
    image = pipe(prompt, num_inference_steps=25).images[0]
    image.save("步骤1_文生图示例.png")
    print("Text-to-image result saved.")

    print("\n=== Step 3: CFG Guidance Scale Effects ===")
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

    plt.suptitle('Step 3: CFG Guidance Scale Effects (SD)')
    plt.tight_layout()
    plt.savefig('步骤3_CFG引导对比.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("CFG guidance comparison saved.")

except Exception as e:
    print(f"Skipping Stable Diffusion steps: {e}")
    print("This is expected if running on CPU or without internet access.")
    print("The markdown file provides the code and expected results for reference.")

print("\n=== Step 2: img2img Pipeline ===")
print("For img2img, you need an initial image (sketch.png).")
print("Place a 512x512 image named 'sketch.png' in this directory and run:")

img2img_code = '''
from diffusers import StableDiffusionImg2ImgPipeline
import PIL.Image

img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to("cuda")

init_image = PIL.Image.open("sketch.png").resize((512, 512))

for strength in [0.3, 0.6, 0.9]:
    result = img2img(
        prompt="a beautiful oil painting",
        image=init_image,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=25
    ).images[0]
    result.save(f"img2img_strength_{strength}.png")
'''
print(img2img_code)

print("\n=== Diffusers Library Architecture ===")
print("Pipeline (high-level) -> Model (neural network) -> Scheduler (algorithm)")
print("Example components:")
print("  - UNet2DConditionModel: predicts noise residual")
print("  - DDIMScheduler / PNDMScheduler: controls denoising steps")
print("  - VAE (AutoencoderKL): latent space encoding/decoding")
print("  - CLIP Text Encoder: text -> embeddings")

print("\nDone!")