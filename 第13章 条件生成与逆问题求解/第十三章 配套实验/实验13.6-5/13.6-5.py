# -*- coding: utf-8 -*-
"""
实验13.6-5：img2img strength参数效应

实验内容：
  - 使用同一初始图像 + 不同strength参数
  - 演示图文一致性与多样性的权衡
  - strength与DPS的zeta、CFG的s的类比

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
    SAVE_DIR = os.path.join(_gdrive, '实验13.6-5')
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
print("实验13.6-5: img2img strength参数效应")
print("=" * 60)
print("知识点: strength参数, 质量-多样性权衡的实践验证")


import torch
try:
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("diffusers/PIL库未安装，请先安装")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")


print("=" * 60)
print("步骤1：strength参数与质量-多样性权衡")
print("=" * 60)

print("""
strength参数含义:
  strength in [0, 1]: 控制加噪程度
  strength=0:   几乎不加噪，输出几乎等于原图（多样性极低）
  strength=0.5: 中等加噪，保留中等结构信息
  strength=1.0: 满噪声，相当于text-to-image（与初始图的一致性最低，文本自由度与多样性最高）

与DPS的zeta参数类比（图像一致性维度）:
  strength(小) <-> zeta(大): 都倾向于强图像一致性、低多样性
  strength(大) <-> zeta(小): 都倾向于弱图像一致性、高多样性
  （注：这里的"一致性"指与初始/观测图像的接近程度）

与CFG的guidance_scale参数类比（独立的文本一致性维度）:
  guidance_scale(大): 更贴合文本prompt描述，但与"像不像原图"无直接关系
  guidance_scale(小): 更接近无条件生成的自然多样性

注意：strength和zeta控制图像层面的观测一致性，而guidance_scale控制文本语义一致性，
      两者是独立的维度，不应混为一谈。
""")

# ===== 实验常量与运行状态（无论是否实际生成都作为"预期配置"记录）=====
PROMPT = "a beautiful sunset over mountains, oil painting style"
STRENGTHS = [0.1, 0.3, 0.5, 0.75, 0.95]
GUIDANCE_SCALE = 7.5
SEED = 42

# 只有真正完成生成并保存图像后才置True；定性观察与量化指标仅在该状态下落盘
generation_ok = False
gen_error = ""
skip_reason = ""
strength_metrics = []

if HAS_DEPS and device == "cuda":
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe = pipe.to(device)

        init_image_pil = None
        try:
            _init_path = os.path.join(SAVE_DIR, "init_image.jpg")
            if not os.path.exists(_init_path):
                import urllib.request
                url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
                urllib.request.urlretrieve(url, _init_path)
            init_image_pil = Image.open(_init_path).convert("RGB")
            # 等比缩放+中心裁剪：官方示例原始尺寸为1024x512，直接resize到512x512会横向拉伸
            _w0, _h0 = init_image_pil.size
            _scale = 512 / min(_w0, _h0)
            _rw, _rh = max(512, round(_w0 * _scale)), max(512, round(_h0 * _scale))
            init_image_pil = init_image_pil.resize((_rw, _rh), Image.LANCZOS)
            _lx, _ty = (_rw - 512) // 2, (_rh - 512) // 2
            init_image_pil = init_image_pil.crop((_lx, _ty, _lx + 512, _ty + 512))
        except Exception as e:
            print(f"无法获取初始图像: {e}")

        if init_image_pil is not None:
            fig, axes = plt.subplots(1, len(STRENGTHS) + 1, figsize=(18, 4))
            axes[0].imshow(init_image_pil)
            axes[0].set_title('原图', fontsize=12)
            axes[0].axis('off')

            _init_arr = np.asarray(init_image_pil, dtype=np.float32)
            for idx, strength in enumerate(STRENGTHS):
                img = pipe(
                    prompt=PROMPT,
                    image=init_image_pil,
                    strength=strength,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=torch.Generator(device=device).manual_seed(SEED),
                ).images[0]
                # 零成本像素级量化：输出相对（预处理后）初始图的差异
                _d = np.asarray(img, dtype=np.float32) - _init_arr
                _mse = float((_d ** 2).mean())
                strength_metrics.append({
                    "strength": strength,
                    "mse": round(_mse, 3),
                    "rmse": round(_mse ** 0.5, 3),
                    "mae": round(float(np.abs(_d).mean()), 3),
                })
                axes[idx + 1].imshow(img)
                axes[idx + 1].axis('off')
                axes[idx + 1].set_title(f'strength={strength}', fontsize=12)

            plt.suptitle(f'strength参数效应：初始图一致性与多样性的权衡（13.6节）\nprompt: "{PROMPT}"',
                         fontsize=13, y=1.05)
            plt.tight_layout()
            strength_path = os.path.join(SAVE_DIR, "strength参数效应.png")
            plt.savefig(strength_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"strength参数对比图已保存: {strength_path}")
            generation_ok = True
        else:
            gen_error = "无法获取初始图像（下载或打开失败），未实际生成"
    except Exception as e:
        print(f"执行出错: {e}")
        gen_error = f"执行出错: {e}"
        try:
            plt.close('all')  # 生成中途失败时释放未保存的图窗
        except Exception:
            pass
else:
    skip_reason = (f"跳过实际运行：需要CUDA GPU + diffusers/PIL"
                   f"（当前 device={device}，依赖={'已安装' if HAS_DEPS else '未安装'}）")
    print(skip_reason)
    print("以下是strength参数的理论说明：")
    print("""
strength参数效应（13.6节）:
  strength=0.1: 几乎保留原图，多样性低
  strength=0.5: 平衡点
  strength=0.95: 几乎重新生成，多样性高

  加噪起点对应扩散时间步:
    t_start ≈ int(strength * T)（前向过程加噪到的噪声水平）
    注意这是img2img自身的机制，与DPS无关（DPS从纯噪声起步、靠似然梯度引导）
    t_start大 -> 起始噪声大 -> 多样性高、与初始图一致性低
""")

print("\n" + "=" * 60)
print("实验13.6-5 完成!")
print("=" * 60)
print("""
关键结论:
1. strength参数（13.6节）
   - 控制加噪起点（t_start≈strength·T）；这是img2img自身的机制，
     DPS并无此旋钮（DPS从纯噪声起步、靠似然梯度引导），
     图像一致性维度上正确的类比对象只有DPS的zeta
   - strength大 -> 加噪起点高 -> 多样性高、与初始图一致性低
   - strength小 -> 加噪起点低 -> 多样性低、保留原图

2. 统一的质量-多样性权衡框架
   - DPS: zeta控制似然修正强度（约束对象：观测图像y）
   - CFG: s控制条件/无条件混合（约束对象：文本prompt）
   - img2img: strength控制加噪程度（约束对象：初始图像）
     （strength大 -> 加噪到更高噪声水平t≈strength·T后再生成）
   - 三者本质上都是"先验强度 vs 条件强度"的tradeoff
   - 但需注意：zeta和strength约束的是图像层面的观测一致性，
     而CFG约束的是文本语义一致性，两者是独立的维度
""")
# ===== 保存结果记录：严格区分"本次实际生成"与"目录内既有存档" =====
import json
import importlib
from datetime import datetime


def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return _to_native(obj.detach().cpu().tolist())
    except Exception:
        pass
    return obj


def _module_version(name):
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except Exception:
        return None


_env = {
    "python": sys.version.split()[0],
    "torch": _module_version("torch"),
    "diffusers": _module_version("diffusers"),
    "transformers": _module_version("transformers"),
    "device": device,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}

_config = {
    "模型": "runwayml/stable-diffusion-v1-5 (StableDiffusionImg2ImgPipeline, fp16)",
    "初始图": "init_image.jpg（CompVis官方sketch-mountains示例，原始1024x512）",
    "预处理": "等比缩放(scale=512/min(w,h))后中心裁剪到512x512，避免长宽比拉伸",
    "prompt": PROMPT,
    "strength_values": STRENGTHS,
    "guidance_scale": GUIDANCE_SCALE,
    "seed": SEED,
    "num_inference_steps": "默认50步；strength=s时跳过前 50-⌊50s⌋ 步，"
                           "一次性加噪到噪声水平 t≈s·T 后仅执行后 ⌊50s⌋ 步逆向去噪",
    "样本数说明": "每个strength仅1个样本(seed=42)，图中展示的是重绘幅度/自由度趋势；"
                 "统计意义上的'多样性'需每档多种子采样后用方差/最近邻距离等度量",
}

# 目录内既有存档文件的客观信息：
# 仅当本次"未实际生成"时才作为历史存档附带；本次若已生成，PNG即本次产物，不另挂存档块。
# 旧存档的来源/观察说明只在文件mtime确为2026-07-07时附带，避免把旧结论张冠李戴到新文件上。
_archive_path = os.path.join(SAVE_DIR, "strength参数效应.png")
_archive_block = None
if (not generation_ok) and os.path.exists(_archive_path):
    _mtime = datetime.fromtimestamp(os.path.getmtime(_archive_path))
    _archive_block = {
        "文件": "strength参数效应.png",
        "存档生成时间": _mtime.isoformat(timespec="seconds"),
    }
    if _mtime.strftime("%Y-%m-%d") == "2026-07-07":
        _archive_block["来源说明"] = (
            "该存档图由2026-07-07运行的前身脚本生成；前身脚本未随本实验目录存档，"
            "且当时预处理为直接resize拉伸(1024x512→512x512)，与当前代码的等比中心裁剪不同，"
            "故不应期待用当前代码对存档图做像素级复现；同种子、同依赖版本可复现定性趋势。")
        _archive_block["观察（仅针对该存档文件的图面判读）"] = (
            "strength=0.1/0.3几乎保留原图（含铅笔线条）；0.5保留构图骨架但色块被抹匀；"
            "0.75大幅重绘但仍同源构图；0.95几乎变成另一幅text-to-image图。"
            "prompt中的'sunset'在该存档图中于strength≥0.75开始出现（单样本现象，阈值不具普适性），"
            "整体印证小strength下初始图约束压制文本约束。")
    else:
        _archive_block["说明"] = (
            "该图为其他日期的历史运行产物，非本次运行生成；请对照图面自行判读，"
            "其结论不属于本次运行记录。")

if generation_ok:
    results_summary = {
        "run_status": "generated",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "环境": _env,
        **_config,
        "输出图": "strength参数效应.png（本次运行新生成，已覆盖同名旧文件）",
        "量化指标_对初始图的像素距离": strength_metrics,
        "指标判读": "MSE/MAE度量输出与（预处理后）初始图的像素差异，strength越大总体应越高"
                  "（对初始图保真度越低）；单次采样允许局部波动。"
                  "更贴合感知的度量可用LPIPS，文本语义一致性可用CLIP-score。",
    }
else:
    results_summary = {
        "run_status": "not_generated",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "环境": _env,
        **_config,
        "skip_reason": gen_error or skip_reason or "未执行生成",
        "说明": "本次运行未实际生成图像，按规范不写入本次定性观察与量化指标；"
                "若目录内存在历史存档，见下方'存档图'块，其中结论仅属于该存档文件。",
    }

if _archive_block is not None:
    results_summary["存档图"] = _archive_block

results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"结果记录已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

