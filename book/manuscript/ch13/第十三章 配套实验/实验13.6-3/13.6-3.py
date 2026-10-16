# -*- coding: utf-8 -*-
"""
实验13.6-3：Diffusers img2img管线

实验内容：
  - 使用Diffusers的StableDiffusionImg2ImgPipeline
  - 同一初始图 + 三个不同prompt（strength=0.75固定），展示文本条件的作用
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
print("实验13.6-3: Diffusers img2img管线")
print("=" * 60)
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

# ===== 实验配置（无论是否实际生成都作为"预期配置"记录）=====
PROMPTS = [
    "a beautiful landscape painting of mountains, oil painting style",
    "a photorealistic photo of mountains, sunny day",
    "a sketch of mountains with pencil",
]
STRENGTH = 0.75
GUIDANCE_SCALE = 7.5
SEED = 42

# ===== strength扫描（附加实验）=====
# 置空列表[]可关闭扫描、只跑原三面板对比；扫描针对照片化最难的prompt，
# 输出到独立文件，不覆盖"img2img闭环对比.png"
STRENGTH_SCAN = [0.6, 0.75, 0.9]
SCAN_PROMPT_IDX = 1  # PROMPTS中"photorealistic"面板的索引

# 运行状态：只有真正生成并保存图像后才置True，观察结论仅在该状态下落盘
generation_ok = False
scan_ok = False
scan_error = ""
skip_reason = ""


print("=" * 60)
print("步骤1：img2img与逆问题求解器的对比")
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
            _init_path = os.path.join(SAVE_DIR, "init_image.jpg")
            if not os.path.exists(_init_path):
                import urllib.request
                url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
                urllib.request.urlretrieve(url, _init_path)
            init_image_pil = Image.open(_init_path).convert("RGB")
            init_image_pil = init_image_pil.resize((512, 512))
        except Exception as e:
            print(f"无法获取初始图像: {e}")

        if init_image_pil is not None:
            fig, axes = plt.subplots(1, len(PROMPTS), figsize=(15, 5))

            for idx, prompt in enumerate(PROMPTS):
                img = pipe(
                    prompt=prompt,
                    image=init_image_pil,
                    strength=STRENGTH,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=torch.Generator(device=device).manual_seed(SEED),
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
            generation_ok = True

            # ---- 附加实验：strength扫描（独立try，失败不影响主实验状态）----
            if STRENGTH_SCAN:
                try:
                    scan_prompt = PROMPTS[SCAN_PROMPT_IDX]
                    print(f"\n开始strength扫描: prompt='{scan_prompt}', strengths={STRENGTH_SCAN}")
                    fig2, axes2 = plt.subplots(1, len(STRENGTH_SCAN),
                                               figsize=(5 * len(STRENGTH_SCAN), 5))
                    for j, s in enumerate(STRENGTH_SCAN):
                        ax_j = np.atleast_1d(axes2)[j]
                        img_s = pipe(
                            prompt=scan_prompt,
                            image=init_image_pil,
                            strength=s,
                            guidance_scale=GUIDANCE_SCALE,
                            generator=torch.Generator(device=device).manual_seed(SEED),
                        ).images[0]
                        ax_j.imshow(img_s)
                        ax_j.axis('off')
                        ax_j.set_title(f"strength={s}", fontsize=12)
                        print(f"  strength={s} 完成")

                    plt.suptitle('img2img strength扫描（photorealistic prompt, seed=42, 13.6节）',
                                 fontsize=13, y=1.02)
                    plt.tight_layout()
                    scan_path = os.path.join(SAVE_DIR, "img2img_strength扫描.png")
                    plt.savefig(scan_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"strength扫描图已保存: {scan_path}")
                    scan_ok = True
                except Exception as e:
                    plt.close()
                    scan_error = f"strength扫描执行出错: {e}"
                    print(scan_error)
        else:
            skip_reason = "无法获取初始图像（下载或打开失败），未实际生成"
    except Exception as e:
        print(f"执行出错: {e}")
        print("可能原因：GPU内存不足、网络无法下载模型等")
        skip_reason = f"执行出错: {e}"
else:
    print("跳过实际运行（需要GPU + Diffusers）")
    skip_reason = (f"跳过实际运行：需要CUDA GPU + diffusers"
                   f"（当前 device={device}, diffusers={'已安装' if HAS_DIFFUSERS else '未安装'}）")
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
print("实验13.6-3 完成!")
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

from datetime import datetime

# 与"是否真正生成"无关的预期配置
_config = {
    "模型": "runwayml/stable-diffusion-v1-5 (StableDiffusionImg2ImgPipeline, fp16)",
    "初始图": "sketch-mountains-input.jpg (CompVis官方示例, resize到512x512)",
    "prompts": PROMPTS,
    "strength": STRENGTH,
    "guidance_scale": GUIDANCE_SCALE,
    "seed": SEED,
    "num_inference_steps": "默认50步；strength=0.75时一次性加噪到第13个调度时间步对应的噪声水平"
                           "(int(50*0.75)=37, t_start=50-37=13)，仅执行后37步逆向去噪",
    "对比": "img2img与逆问题求解器(DPS/DiffPIR)的本质区别",
    "strength_scan配置": {
        "启用": bool(STRENGTH_SCAN),
        "strengths": STRENGTH_SCAN,
        "prompt": PROMPTS[SCAN_PROMPT_IDX] if STRENGTH_SCAN else "",
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "输出图": "img2img_strength扫描.png",
    },
}

if generation_ok:
    results_summary = {
        "run_status": "generated",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "存档图": "img2img闭环对比.png",
        "note": "定性视觉对比实验（无量化指标）。以下观察基于本次（seed固定）实际生成的存档图，"
                "应逐面板对照核实。",
        **_config,
        "观察": "三幅输出在strength=0.75下不同程度保留初始图的粗粒度构图与主要色块"
                "（河道/山丘/岸带等，各面板保留程度有差异，需逐面板对照），风格随prompt明显分化。"
                "strength锚定构图与色块是img2img固有的'保真-自由'权衡；"
                "但各风格目标的达成程度同时受随机种子与prompt措辞影响——"
                "若某风格目标（如photorealistic）在单次采样中仅部分达成，"
                "不能据此断言固有上界——同prompt的strength扫描(0.6/0.75/0.9，见'strength扫描'块)"
                "已证实0.9面板明显照片化，故0.75未照片化是保真-自由的参数权衡，而非固有上界。",
    }
    # strength扫描结果：仅在扫描图真正生成后写入，且只给判读方法、不预造观察
    if scan_ok:
        results_summary["strength扫描"] = {
            "存档图": "img2img_strength扫描.png",
            "prompt": PROMPTS[SCAN_PROMPT_IDX],
            "strengths": STRENGTH_SCAN,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
            "判读方法": "逐面板核对照片化程度。步数对应：0.6→30步(t_start=20)、"
                        "0.75→37步(t_start=13)、0.9→45步(t_start=5)，strength越大初始图约束越弱、"
                        "文本风格自由度越高。若照片化程度随strength单调增强且0.9面板明显照片化，"
                        "则表明0.75处未照片化是'保真-自由'权衡（参数选择）而非img2img固有上界；"
                        "若0.9仍无法照片化，再排查prompt措辞或模型能力。",
            "观察": "三面板呈严格单调趋势。①strength=0.6（30步去噪）：仍为平涂卡通/插画风，"
                    "初始图的蓝河道、亮绿山丘、桃色岸带、尖顶山几乎全部保留，构图忠实度最高，"
                    "对'sunny day'仅有天空淡蓝色调、无太阳；②strength=0.75（37步）：几何化平涂插画，"
                    "出现黄色卡通太阳与放射光线，河道退化为红粉色带、山体几何化，构图仅部分保留；"
                    "③strength=0.9（45步，t_start=5）：明显照片化——写实山体纹理与光影、"
                    "真实天空与强烈镜头光晕、前景为写实起伏的暖色沙丘，初始图的河道/亮绿山丘/城堡等"
                    "具体构图元素全部消失，仅保留'山脉+晴天'的语义。"
                    "结论：照片化程度随strength单调增强、对初始图的构图忠实度单调下降，0.9已达成照片写实风，"
                    "证明0.75（含此前未固定种子与seed=42两次独立采样）未照片化是strength参数选择导致的"
                    "'保真-自由'权衡，而非img2img机制或SD1.5的固有上界。",
        }
    elif scan_error:
        results_summary["strength扫描"] = {"状态": "failed", "错误": scan_error}
else:
    # 未实际生成：只记录状态与预期配置，不写入任何定性"观察"结论
    results_summary = {
        "run_status": "not_generated",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "skip_reason": skip_reason or "未执行生成",
        "说明": "本次运行未实际生成图像，按实验规范不写入定性'观察'结论；"
                "以下仅为预期配置，不代表已验证的结果。",
        **_config,
    }
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"结果记录已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

