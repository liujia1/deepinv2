# -*- coding: utf-8 -*-
"""
实验13.4-5：ControlNet深度条件生成

实验内容：
  - ControlNet架构原理（13.4节条件控制）
  - 使用diffusers库加载预训练ControlNet模型
  - depth-to-image示例（深度图→生成图像）
  - ControlNet vs CFG对比（条件控制强度）
  - ControlNet的零样本迁移能力

注意：本实验需要GPU和预训练模型下载（约5GB），CPU上运行极慢。
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
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*safety_checker.*")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验13.4-5')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    # 将 HuggingFace 模型缓存指向 Google Drive，避免每次重启重新下载
    os.environ['HF_HOME'] = os.path.join(_gdrive, 'hf_cache')
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)
    print(f"HuggingFace 缓存目录: {os.environ['HF_HOME']}")
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

# matplotlib LaTeX格式用于数学符号显示（必须在中文配置之后设置，否则会被覆盖）
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

print("\n" + "=" * 60)
print("实验13.4-5: ControlNet深度条件生成")
print("=" * 60)
print("知识点: ControlNet架构(13.4节), 条件控制, 预训练模型应用")

print("""
ControlNet核心思想（13.4节）：
  传统条件生成：修改UNet内部参数 → 破坏原有能力
  ControlNet创新：保留原始SD + 新增条件控制分支

  ControlNet架构：
    原始Stable Diffusion UNet（冻结权重）
      +
    ControlNet分支（可训练）
      - 复制UNet的encoder和middle block结构
      - 通过zero convolution连接（初始输出为零）
      - 条件输入：深度图、边缘图、姿态等

  关键优势：
    - 保留原始SD生成能力（冻结权重）
    - 精确控制生成结构（深度、边缘等）
    - 零样本迁移：一个模型处理多种条件类型
""")

import torch
import numpy as np
try:
    from skimage.metrics import structural_similarity as compute_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    compute_ssim = None

try:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
    from PIL import Image
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    print("\nDiffusers库未安装，请先安装:")
    print("pip install diffusers transformers accelerate")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")
if device == "cpu":
    print("警告: CPU上运行极慢，建议使用GPU")

# ============================================================
# 步骤1：ControlNet架构原理
# ============================================================
print("\n" + "=" * 60)
print("步骤1：ControlNet架构原理（13.4节）")
print("=" * 60)

print("""
ControlNet架构设计（2023年论文）：

1. 原始Stable Diffusion架构：
   - UNet: encoder → middle block → decoder
   - 每个block包含ResNet + Attention层
   - 训练在数百万图像上，生成能力强大

2. ControlNet分支设计：
   - 复制UNet的encoder和middle block结构
   - 添加zero convolution层（初始权重为零）
   - 条件输入通过ControlNet处理后，注入UNet

3. Zero Convolution关键机制：
   - 初始化：输出为零（不干扰原始SD）
   - 训练：逐渐学习如何注入条件信息
   - 结果：精确控制，同时保留原有能力

4. 条件类型：
   - Depth: 深度图 → 生成对应深度结构的图像
   - Edge: 边缘图（Canny）→ 生成对应轮廓的图像
   - Pose: 姿态图 → 生成对应姿态的人物图像
   - Segmentation: 分割图 → 生成对应区域的图像

架构流程：
  条件输入 (Depth/Edge/Pose)
       ↓
  ControlNet分支 (可训练权重)
       ↓ 条件注入
  原始Stable Diffusion UNet (冻结权重)
       ↓
  生成图像

关键优势：保留原始SD生成能力 + 精确条件控制
""")

# ============================================================
# 步骤2：深度图条件生成原理可视化
# ============================================================
print("\n" + "=" * 60)
print("步骤2：深度图条件生成原理可视化")
print("=" * 60)

print("""
深度图条件生成原理：

1. 深度图编码：
   - 输入：深度图（灰度图，亮度表示距离）
   - 编码：通过ControlNet的Encoder提取空间结构特征
   - 输出：条件特征图（与UNet的encoder特征对齐）

2. 条件注入：
   - Zero Convolution：初始权重为零，逐渐学习如何注入条件
   - 注入位置：UNet的每个encoder block和middle block
   - 注入方式：相加（additive）

3. 生成过程：
   - 噪声图像 x_T → 逐步去噪 → 生成图像 x_0
   - 每步去噪都受到深度图条件的引导
   - 最终生成的图像符合深度图的空间结构

4. ControlNet vs CFG对比：
   - ControlNet：精确控制空间结构（深度、边缘等）
   - CFG：仅控制文本语义，无法精确控制空间结构
""")

# 可视化ControlNet与CFG的能力对比（基于受控模拟数据的定量演示）
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：ControlNet能力展示
# 子图1：条件输入（深度图）
np.random.seed(42)
depth_map = np.zeros((100, 100))
depth_map[20:80, 20:80] = 0.5
depth_map[30:70, 30:70] = 0.7
depth_map[40:60, 40:60] = 0.9
axes[0, 0].imshow(depth_map, cmap='gray_r')
axes[0, 0].set_title('条件输入：深度图')
axes[0, 0].axis('off')

# 子图2：ControlNet生成效果（模拟：保留深度结构）
np.random.seed(43)
ctrlnet_result = np.zeros((100, 100))
ctrlnet_result[20:80, 20:80] = 0.6 + np.random.randn(60, 60) * 0.05
ctrlnet_result[30:70, 30:70] = 0.75 + np.random.randn(40, 40) * 0.05
ctrlnet_result[40:60, 40:60] = 0.9 + np.random.randn(20, 20) * 0.05
axes[0, 1].imshow(ctrlnet_result, cmap='gray_r')

# 子图3：CFG生成效果（模拟：无法控制结构）
np.random.seed(44)
cfg_result = np.random.rand(100, 100) * 0.4 + 0.3
axes[0, 2].imshow(cfg_result, cmap='gray_r')

# 第二行：定量对比与原理
# 子图4：条件遵守度对比（基于模拟数据真实计算SSIM）
methods = ['ControlNet', 'CFG', '无控制']
colors = ['steelblue', 'coral', 'gray']
if HAS_SKIMAGE:
    np.random.seed(45)
    none_result = np.random.rand(100, 100)
    ssim_ctrlnet = compute_ssim(depth_map, ctrlnet_result, data_range=1.0)
    ssim_cfg = compute_ssim(depth_map, cfg_result, data_range=1.0)
    ssim_none = compute_ssim(depth_map, none_result, data_range=1.0)
    ssim_values = [ssim_ctrlnet, ssim_cfg, ssim_none]
    axes[0, 1].set_title(f'ControlNet生成\n（保留深度结构，SSIM={ssim_ctrlnet:.2f}）')
    axes[0, 2].set_title(f'CFG生成（仅文本条件）\n（无结构控制，SSIM={ssim_cfg:.2f}）')
else:
    ssim_values = [0.85, 0.12, 0.08]
    axes[0, 1].set_title('ControlNet生成\n（保留深度结构，SSIM≈0.85）')
    axes[0, 2].set_title('CFG生成（仅文本条件）\n（无结构控制，SSIM≈0.12）')
    print("警告：未安装scikit-image，SSIM使用占位值，仅供示意。")
axes[0, 1].axis('off')
axes[0, 2].axis('off')

bars = axes[1, 0].bar(methods, ssim_values, color=colors, alpha=0.8)
axes[1, 0].set_ylabel('深度图相似度 (SSIM)')
axes[1, 0].set_title('条件遵守度对比（基于模拟数据计算）')
axes[1, 0].set_ylim(0, 1.0)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, ssim_values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', fontweight='bold')

# 子图5：可控性维度对比（示意性，非实测数据）
axes[1, 1].axis('off')
categories = ['空间结构', '语义内容', '风格', '姿态', '布局']
ctrlnet_scores = [0.9, 0.7, 0.6, 0.9, 0.8]
cfg_scores = [0.1, 0.9, 0.7, 0.3, 0.4]
x = np.arange(len(categories))
width = 0.35
axes[1, 1].bar(x - width/2, ctrlnet_scores, width, label='ControlNet', color='steelblue', alpha=0.8)
axes[1, 1].bar(x + width/2, cfg_scores, width, label='CFG', color='coral', alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(categories, rotation=15, ha='right')
axes[1, 1].set_ylabel('控制能力 (0-1)')
axes[1, 1].set_title('可控性维度对比（示意性，非实测）')
axes[1, 1].set_ylim(0, 1.0)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 子图6：计算代价对比
# 数据来源：ControlNet论文(Zhang et al. 2023)，非本实验测算
trainable_params = [361, 0, 0]  # ControlNet可训练参数量(M)，CFG=0
colors2 = ['steelblue', 'coral', 'gray']
bars2 = axes[1, 2].bar(methods, trainable_params, color=colors2, alpha=0.8)
axes[1, 2].set_ylabel('可训练参数 (M)')
axes[1, 2].set_title('训练代价对比（数据来源：ControlNet论文）')
axes[1, 2].grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars2, trainable_params)):
    if val > 0:
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, val + 10,
                        f'{val}M', ha='center', fontweight='bold')
    else:
        # 对0值柱子添加"无需训练"标注（类似误差棒风格）
        x_center = bar.get_x() + bar.get_width()/2
        # 底部短横线
        axes[1, 2].plot([x_center-0.2, x_center+0.2], [5, 5],
                        color=colors2[i], linewidth=4, solid_capstyle='round')
        axes[1, 2].text(x_center, -25, '无需训练', ha='center',
                        fontsize=9, style='italic', color=colors2[i])

plt.suptitle('ControlNet vs CFG 能力对比：受控模拟数据演示', fontsize=14, y=1.02)
plt.tight_layout()
result_path = os.path.join(SAVE_DIR, "ControlNet_vs_CFG能力对比.png")
plt.savefig(result_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"ControlNet vs CFG能力对比图已保存: {result_path}")
if HAS_SKIMAGE:
    print(f"  SSIM计算结果: ControlNet={ssim_ctrlnet:.3f}, CFG={ssim_cfg:.3f}, 无控制={ssim_none:.3f}")

if HAS_DIFFUSERS and device == "cuda":
    try:
        print("\n加载预训练ControlNet模型...")
        # 加载ControlNet depth模型
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16
        )

        # 加载Stable Diffusion pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # 使用更快的scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # 启用内存优化（添加异常处理，兼容accelerate库问题）
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"  警告: enable_model_cpu_offload 失败 ({e})，回退到直接to(device)")
            pipe = pipe.to(device)

        print("模型加载完成！")
        print("  关键参数说明:")
        print("  - controlnet_scale: 条件控制强度（默认1.0，越大控制越强）")
        print("  - num_inference_steps: 推理步数（默认20，步数越多质量越好）")
        print("  - guidance_scale: CFG引导强度（默认7.5）")

        # 创建测试深度图（简化示例）
        print("\n生成测试深度图...")
        # 创建一个简单的深度图（模拟深度信息）
        depth_map = np.zeros((512, 512), dtype=np.uint8)
        # 模拟一个房间的深度结构
        depth_map[100:400, 100:400] = 128  # 中间区域
        depth_map[150:350, 150:350] = 200  # 更深的区域
        depth_map[200:300, 200:300] = 255  # 最深区域

        depth_image = Image.fromarray(depth_map)

        prompt = "a beautiful room with furniture, high quality"
        print(f"\n生成图像: '{prompt}'")
        print("条件输入：深度图（模拟房间深度结构）")

        # ControlNet生成（使用controlnet_scale参数控制条件强度）
        controlnet_scale = 0.8  # 0.5-1.0之间，0.8为推荐值
        image_controlnet = pipe(
            prompt,
            image=depth_image,
            num_inference_steps=20,
            controlnet_conditioning_scale=controlnet_scale,  # 条件控制强度
            generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]
        print(f"  controlnet_scale = {controlnet_scale}")

        # 保存结果
        image_controlnet.save(os.path.join(SAVE_DIR, "ControlNet_depth示例.png"))
        depth_image.save(os.path.join(SAVE_DIR, "测试深度图.png"))
        print("ControlNet depth-to-image结果已保存")

        # ============================================================
        # 步骤3：ControlNet vs CFG对比
        # ============================================================
        print("\n" + "=" * 60)
        print("步骤3：ControlNet vs CFG对比")
        print("=" * 60)

        # CFG生成（无ControlNet）
        print("\n对比CFG生成（无条件控制）...")
        from diffusers import StableDiffusionPipeline
        pipe_cfg = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe_cfg.scheduler = UniPCMultistepScheduler.from_config(pipe_cfg.scheduler.config)
        # 启用内存优化（与ControlNet pipeline风格保持一致）
        try:
            pipe_cfg.enable_model_cpu_offload()
        except Exception as e:
            print(f"  警告: enable_model_cpu_offload 失败 ({e})，回退到直接to(device)")
            pipe_cfg = pipe_cfg.to(device)

        # CFG不同引导强度
        cfg_scales = [1.0, 7.5, 15.0]
        images_cfg = []

        for gs in cfg_scales:
            img_cfg = pipe_cfg(
                prompt,
                num_inference_steps=20,
                guidance_scale=gs,
                generator=torch.Generator(device=device).manual_seed(42)
            ).images[0]
            images_cfg.append(img_cfg)

        # 对比ControlNet与CFG
        # 打印ControlNet与CFG的能力对比说明
        print("\n" + "=" * 60)
        print("ControlNet vs CFG 能力对比")
        print("=" * 60)
        print("ControlNet优势：")
        print("  - 保留深度结构")
        print("  - 精确空间控制")
        print("CFG局限：")
        print("  - 无法控制结构")
        print("  - 仅依赖文本提示")
        print("CFG引导强度说明：")
        print("  - 引导强度越高，条件约束越强")
        print("  - 但无法精确控制空间结构")
        print("=" * 60)

        # 创建对比图：第一行深度图和ControlNet结果，第二行CFG不同引导强度结果
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # 第一行：ControlNet结果
        axes[0, 0].imshow(depth_image, cmap='gray')
        axes[0, 0].set_title('条件输入：深度图')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(image_controlnet)
        axes[0, 1].set_title('ControlNet生成')
        axes[0, 1].axis('off')

        # ControlNet特征可视化（使用深度图的梯度强度作为结构保持指标）
        depth_array = np.array(depth_image)
        grad_x = np.abs(np.diff(depth_array, axis=1))
        grad_y = np.abs(np.diff(depth_array, axis=0))
        # 使用边缘检测结果表示结构信息
        edge_strength = np.zeros_like(depth_array, dtype=float)
        edge_strength[:, 1:] += grad_x
        edge_strength[1:, :] += grad_y
        axes[0, 2].imshow(edge_strength, cmap='hot')
        axes[0, 2].set_title('深度结构特征')
        axes[0, 2].axis('off')

        # ControlNet生成图像的边缘强度
        controlnet_array = np.array(image_controlnet.convert('L'))
        grad_x_c = np.abs(np.diff(controlnet_array, axis=1))
        grad_y_c = np.abs(np.diff(controlnet_array, axis=0))
        edge_c = np.zeros_like(controlnet_array, dtype=float)
        edge_c[:, 1:] += grad_x_c
        edge_c[1:, :] += grad_y_c
        axes[0, 3].imshow(edge_c, cmap='hot')
        axes[0, 3].set_title('生成图像结构')
        axes[0, 3].axis('off')

        # 第二行：CFG不同引导强度
        for idx, (gs, img_cfg) in enumerate(zip(cfg_scales, images_cfg)):
            axes[1, idx].imshow(img_cfg)
            axes[1, idx].set_title(f'CFG (scale={gs})')
            axes[1, idx].axis('off')

        # CFG生成图像的边缘强度对比
        cfg_array = np.array(images_cfg[-1].convert('L'))  # 使用最高引导强度的结果
        grad_x_cfg = np.abs(np.diff(cfg_array, axis=1))
        grad_y_cfg = np.abs(np.diff(cfg_array, axis=0))
        edge_cfg = np.zeros_like(cfg_array, dtype=float)
        edge_cfg[:, 1:] += grad_x_cfg
        edge_cfg[1:, :] += grad_y_cfg
        axes[1, 3].imshow(edge_cfg, cmap='hot')
        axes[1, 3].set_title('CFG生成结构')
        axes[1, 3].axis('off')

        plt.suptitle('ControlNet vs CFG对比：条件控制强度对比', fontsize=14, y=0.98)
        plt.tight_layout()
        compare_path = os.path.join(SAVE_DIR, "ControlNet vs CFG对比.png")
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比结果已保存: {compare_path}")
        print("  注：GPU真实生成图为RGB图像，与合成深度图属于不同模态，本对比为定性视觉比较；")
        print("      定量的SSIM条件遵守度已在上方受控模拟数据图中基于真实计算给出。")

    except Exception as e:
        print(f"\n执行出错: {e}")
        print("可能原因：GPU内存不足、网络无法下载模型等")
        print("\n以下为理论说明：")
        print("""
ControlNet depth-to-image流程：
  1. 输入深度图（深度信息）
  2. ControlNet编码深度图
  3. 注入到原始SD UNet
  4. 生成符合深度结构的图像

关键优势：
  - 精确控制图像的空间结构
  - 保留原始SD的生成质量
  - 无需重新训练整个模型
""")
else:
    print("\n跳过实际运行（需要GPU + Diffusers + 网络）")
    print("以下为理论说明：")

    print("""
ControlNet depth-to-image原理（13.4节）：

1. ControlNet架构
   原始SD UNet（冻结） + ControlNet分支（可训练）
   通过zero convolution连接，初始输出为零

2. Depth条件生成流程
   输入：深度图 + 文本提示
   ControlNet：编码深度图，注入UNet
   输出：符合深度结构的图像

3. ControlNet vs CFG对比
   ControlNet：
     - 精确空间控制（深度、边缘等）
     - 保留原始SD能力
     - 结构条件优先

   CFG：
     - 仅文本条件控制
     - 无法精确控制结构
     - 引导强度调整文本约束

4. 关键优势
   - 零样本迁移：一个模型处理多种条件
   - 无需重新训练整个模型
   - 精确条件控制 + 高质量生成
""")

# ============================================================
# 步骤4：ControlNet零样本迁移能力
# ============================================================
print("\n" + "=" * 60)
print("步骤4：ControlNet零样本迁移能力")
print("=" * 60)

print("""
ControlNet零样本迁移（13.4节）：

1. 同一个ControlNet架构可处理：
   - Depth：深度图条件生成
   - Edge：边缘图条件生成（Canny）
   - Pose：姿态图条件生成
   - Segmentation：分割图条件生成

2. 零样本迁移原理：
   - 不同条件类型→不同ControlNet分支
   - 原始SD核心权重冻结
   - 仅需训练ControlNet分支（少量数据）

3. 对比传统方法：
   传统方法：每种条件→重新训练整个模型
   ControlNet：保留核心权重 + 新增分支

4. 教学意义：
   - 展示条件生成的灵活架构
   - 理解零样本迁移的价值
   - 对比不同条件控制方法的适用场景
""")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("实验13.4-5 完成!")
print("=" * 60)
print("""
关键结论:
1. ControlNet架构原理（13.4节）
   - 原始SD冻结 + ControlNet可训练分支
   - Zero convolution连接，初始输出为零
   - 精确条件控制 + 保留原有能力

2. ControlNet vs CFG对比
   - ControlNet：精确空间控制（深度、边缘等）
   - CFG：仅文本条件控制，无法精确控制结构
   - 两者各有优势，适用不同场景

3. ControlNet零样本迁移
   - 一个架构处理多种条件类型
   - 无需重新训练整个模型
   - 教学价值：展示条件生成的灵活架构

4. 实践意义
   - ControlNet是当前主流方法（2023年论文）
   - depth-to-image、edge-to-image等直接可用
   - 对学生理解条件生成有重要实践意义
""")