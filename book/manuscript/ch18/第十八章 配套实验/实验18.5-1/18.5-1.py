# -*- coding: utf-8 -*-
"""
实验18.5-1 不确定性量化与后验采样
对应知识点：18.5节（不确定性量化：从点估计到分布推断）

实验内容：
Step 0: 采样器正确性自检 —— 在解析可解问题（线性高斯 inpainting，后验有
        闭式解）上验证同一套 ULA 代码收敛到真实后验
Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
Step 2: 后验采样实现 —— 退火 PnP-ULA（DRUNet 先验，S 条链 batch 同时演化）
Step 3: 不确定性地图计算 —— 像素级标准差、经验分位数区间、偏差场结构诊断
Step 4: 校准检验 —— 覆盖率与可靠性分析（观测/缺失半区分开统计）
Step 5: 样本数对不确定性估计的影响

===========================================================================
本版（v3）关键设计 —— 每一项都有解析可解问题上的定量依据或失败样本对照
===========================================================================
1. lambda_data = 1.0：数据保真项的贝叶斯权重必须是 1。取 0.05 会让链收敛到
   p_σ(x)·p(y|x)^0.05 而非 p(x|y)，观测像素 std 被撑到 σ_d/√λ（解析可算）。
2. σ 调度 = geomspace(0.196, 2/255, 14)：上界 50/255 是 DRUNet 的训练噪声
   上限（deepinv DRUNet 直接把 σ 作噪声图，无 /255 缩放），外推无依据。
3. 400 步/级 = 2 个松弛时间（收缩率 c=0.005 → 1/c=200 步；解析问题上
   把覆盖率从 89.1% 修到 94.5%）。
4. DPIR-MAP 热启动 + MYULA 近端数据项（v3 核心修复）：零填充初始化的链
   到退火结束仍停留在错误盆域（缺失半区只有 8~13 dB），因为"50% 黑洞"
   这种粗尺度填充误差只有高 σ 级能搬运。先用 12 级×8 次硬投影去噪把链
   放进正确盆域（开销 <2%），再做退火 Langevin；数据项改用闭式 prox
   （观测像素无条件稳定）。修复后：缺失 RMS 比(均值/MAP) 12.8→1.06，
   PIT 秩 18.8→15.2，覆盖率对 S=30 零模型由 -5% 偏差转为 +4.8%。
5. 分区统计 + 有限样本零模型评级：池化数字被未约束的一半主导；S=30 时
   名义 93.3% 区间的零模型期望覆盖率只有 87.3%，评级必须对零模型。
6. 验收测试：不达标时明确声明"本次 UQ 不可信"，不把失败包装成"后验偏保守"。

运行前提：GPU 优先。Colab T4 实测 batch-30 采样 14×400 步 ≈ 34 分钟（2059 s）。
本目录产物（uq_results.json / 4 张 PNG / experiment_cache_S30.pkl）即由该次
运行产出（缓存指纹记录 device='cuda'）。注意：缓存指纹含 device 字段——
Colab 生成的缓存在纯 CPU 环境不会命中、会触发完整重采样；把本目录上传
Colab GPU 重跑则命中缓存、只重算统计与图。纯 CPU 下 Step 2 慢 1~2 个
数量级（本机 MX110 2GB 无法安装 cu126 版 torch）。DRUNet 权重需能访问
HuggingFace（下载失败会走"伪逆+噪声"兜底并明确标记 UQ 不可信）。
"""

import os
import sys
import math
import json
import time
import hashlib
import traceback

import numpy as np
import torch

# ★ CUDA 显存优化：减少碎片，提升长时间运行的稳定性
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 非交互式后端（必须在 import matplotlib.pyplot 之前）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # 无 tqdm 时的空实现：接口与真实进度条一致但不产生输出
        def __init__(self, iterable=None, **k):
            self._it = iterable

        def __iter__(self):
            return iter(self._it) if self._it is not None else iter(())

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def update(self, n=1):
            pass

        def set_postfix_str(self, *a, **k):
            pass

        @staticmethod
        def write(s, *a, **k):
            print(s)

        def close(self):
            pass


# ========================================================================
# torchvision 兼容层
# ========================================================================
# 有些环境里 torchvision 与 torch 的 ABI 不匹配（典型报错：
#   RuntimeError: operator torchvision::nms does not exist ），
# 会让 `import deepinv` 直接失败。本实验的代码路径完全不需要 torchvision
# （图像用 PIL 读，物理算子/去噪器都来自 deepinv 的纯 torch 模块），
# 所以这里在导入失败时放一个最小占位模块，保证实验能够运行。
# 若你的环境 torchvision 正常，这个函数什么都不做。
def _ensure_torchvision_importable():
    try:
        import torchvision  # noqa: F401
        import torchvision.transforms  # noqa: F401
        return False
    except Exception as exc:
        import types

        def _dummy(*a, **k):
            return None

        class _AnyMod(types.ModuleType):
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return _dummy

        subs = ["transforms", "transforms.functional", "utils", "ops", "io", "models",
                "datasets", "datasets.utils", "datasets.folder"]
        root = _AnyMod("torchvision")
        root.__version__ = "0.0.0-stub"
        sys.modules["torchvision"] = root
        for s in subs:
            m = _AnyMod("torchvision." + s)
            sys.modules["torchvision." + s] = m
            setattr(root, s.split(".")[0], sys.modules["torchvision." + s.split(".")[0]])
        print(f"[环境] torchvision 不可用({type(exc).__name__})，已启用占位模块；"
              f"本实验不需要 torchvision 的真实功能。")
        return True


_TV_STUBBED = _ensure_torchvision_importable()


# ========================================================================
# deepinv 自动安装引导（Colab 等干净环境里缺包时自动 pip 安装）
# ========================================================================
# Python 脚本本身不会安装依赖；首次在 Colab 运行时 deepinv 不是预装包。
# 这里在 import 失败时用当前解释器执行 pip install deepinv==固定版本，
# 装完在同一进程内重新导入，无需用户手动开单元、也不用重启运行时。
# 若 pip 顺带升级了 numpy/torch 等已加载的包导致导入仍失败，会明确提示
# 重启运行时（Colab 菜单：代码执行程序 → 重新启动）后重跑本脚本。
_DEEPINV_PIN = "0.4.0"   # 与本实验验证过的版本保持一致


def _ensure_deepinv_installed():
    try:
        import deepinv  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    print(f"[环境] 未检测到 deepinv，正在自动安装 deepinv=={_DEEPINV_PIN} …")
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           f"deepinv=={_DEEPINV_PIN}"]
    ret = subprocess.call(cmd)
    if ret != 0:
        raise SystemExit(
            "deepinv 自动安装失败（pip 返回码 %d）。请在单独的单元里执行\n"
            "    !pip install deepinv==%s\n"
            "安装成功后重新运行本脚本。" % (ret, _DEEPINV_PIN))
    import importlib
    importlib.invalidate_caches()
    try:
        import deepinv  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "deepinv 已安装但仍无法导入（%s: %s）。\n"
            "这通常是 pip 顺带升级了 numpy/torch 等已加载的包。请在 Colab 菜单\n"
            "『代码执行程序 → 重新启动』后重新运行本脚本（包已装好，不会再装）。"
            % (type(exc).__name__, exc))
    print(f"[环境] deepinv 安装完成。")


_ensure_deepinv_installed()

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验18.5-1')
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
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

if _TV_STUBBED:
    pass  # 占位模块只用于让 deepinv 可导入；本实验不使用 torchvision 的任何功能

import deepinv as dinv
from deepinv.physics import Inpainting, GaussianNoise
from deepinv.models import DRUNet

print(f"deepinv {dinv.__version__} | torch {torch.__version__} | 保存目录: {SAVE_DIR}")


# ========================================================================
# 设备（GPU 优先）
# ========================================================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    _prop = torch.cuda.get_device_properties(0)
    print(f"使用设备: {device}  ({_prop.name}, {_prop.total_memory / 1e9:.1f} GB, "
          f"sm_{_prop.major}{_prop.minor})")
else:
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    print("[警告] 未检测到 CUDA，Step 2 的扩散采样会非常慢；"
          "如可用请安装 CUDA 版 PyTorch（GPU 上快 1~2 个数量级）。")

# 固定随机种子（注意：物理掩码另用独立种子，见下）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ========================================================================
# 统一配置
# ========================================================================
CFG = {
    "image_size": 128,

    # ---- 观测模型 ----
    "sigma_data": 0.01,          # 观测噪声标准差（对应 PSNR 40 dB）
    "mask_ratio": 0.5,           # inpainting 保留比例
    "mask_seed": 20240518,       # ★ 掩码专用种子（与全局种子解耦，跨设备可复现）

    # ---- 退火 PnP-ULA ----
    # σ 上界 = DRUNet 训练上限 50/255；下界 = 2/255（PnP-ULA 惯例）。
    "sigma_max": 50.0 / 255.0,   # ≈ 0.1961
    "sigma_min": 2.0 / 255.0,    # ≈ 0.00784
    "n_sigma_levels": 14,
    "steps_each_sigma": 400,     # 松弛时间 1/c = 200 步，取 2 倍
    "lambda_data": 1.0,          # ★ 贝叶斯权重，必须为 1
    "step_size_coeff_annealed": 0.005,
    "ula_noise_scale": 1.0,
    "ula_projection": (-0.25, 1.25),   # 数值保护；边界远宽于 [0,1]，但缓存样本最小值约 -0.021，说明偶尔生效
    "ula_init_jitter": 0.01,
    "max_grad_norm": None,       # 可选梯度裁剪（None = 关闭）

    # ---- v3：热启动与数据项形式 ----
    # "map_warmstart": 先 DPIR-MAP 把链放进正确后验盆域（v3 默认，修复输送失败）
    # "zerofill"   : v2 行为（零填充 A†y 直接启动，仅用于对照复现失败模式）
    "init_mode": "map_warmstart",
    # "prox": MYULA 近端数据项（inpainting 闭式 prox，无条件稳定）
    # "grad": 显式数据梯度（v2 行为，需 ελ/σ_d²<2；Step 0 解析自检走这条路径）
    "data_fidelity": "prox",
    "map_sigma_max": 49.0 / 255.0,   # DPIR 惯例起点 ≈ 0.1922
    "map_levels": 12,
    "map_steps_per_level": 8,        # 共 96 次去噪（相对 5600 步采样 <2% 开销）

    # ---- 采样规模 ----
    "n_samples": 30,             # 后验样本数 S
    "sample_chunk": 30,          # 一次前向同时演化的链数（显存不足时调小）

    # ---- 校准 ----
    "calib_good": 0.04,          # |覆盖率-名义值| < 4%  判为"校准良好"
    "calib_ok": 0.08,            # < 8% 判为"存在偏差"；否则"严重失配"

    # ---- 缓存 ----
    "use_cache": True,
    "cache_version": 3,          # v3：init_mode/data_fidelity/MAP 预热改变样本，旧缓存作废

    # ---- 调试 ----
    "quick_test": bool(int(os.environ.get("EXP1851_QUICK", "0"))),
}

CFG["sigma_schedule"] = list(np.geomspace(CFG["sigma_max"], CFG["sigma_min"],
                                          CFG["n_sigma_levels"]))
if CFG["quick_test"]:
    CFG["n_samples"] = 4
    CFG["steps_each_sigma"] = 40
    CFG["n_sigma_levels"] = 4
    CFG["sigma_schedule"] = list(np.geomspace(CFG["sigma_max"], CFG["sigma_min"], 4))
    print("[调试] quick_test 模式：S=4, 40 步/级, 4 级")

# 可选的环境变量覆盖（不设置则用默认/quick 配置；指纹含完整 σ 调度，不会撞缓存）
# 用途：CPU 上先跑中等预算验证（如 EXP1851_NSAMPLES=4 EXP1851_LEVELS=14 EXP1851_STEPS=100）
_env_ns = os.environ.get("EXP1851_NSAMPLES")
_env_lv = os.environ.get("EXP1851_LEVELS")
_env_st = os.environ.get("EXP1851_STEPS")
if _env_ns:
    CFG["n_samples"] = int(_env_ns)
if _env_lv:
    CFG["n_sigma_levels"] = int(_env_lv)
if _env_st:
    CFG["steps_each_sigma"] = int(_env_st)
if _env_lv:
    CFG["sigma_schedule"] = list(np.geomspace(CFG["sigma_max"], CFG["sigma_min"],
                                              CFG["n_sigma_levels"]))
if _env_ns or _env_lv or _env_st:
    print(f"[调试] 环境变量覆盖：S={CFG['n_samples']}, "
          f"{CFG['n_sigma_levels']} 级 × {CFG['steps_each_sigma']} 步/级")

IMG_SIZE = CFG["image_size"]
N_SAMPLES = CFG["n_samples"]
SIGMA_DATA = CFG["sigma_data"]

# 稳定性自检：观测半区的更新是 x ← x(1 - ε·λ/σ_d²) + ...，需要 ε·λ/σ_d² < 2
_stab = CFG["step_size_coeff_annealed"] * CFG["sigma_max"] ** 2 * CFG["lambda_data"] / SIGMA_DATA ** 2
print(f"[配置] λ_data={CFG['lambda_data']}, σ∈[{CFG['sigma_min']:.4f}, {CFG['sigma_max']:.4f}] "
      f"({CFG['n_sigma_levels']}级 x {CFG['steps_each_sigma']}步 = "
      f"{CFG['n_sigma_levels'] * CFG['steps_each_sigma']} 步/样本)")
if CFG["data_fidelity"] == "prox":
    print(f"[配置] 数据项=MYULA 近端步：观测像素无条件稳定（ελ/σ_d²={_stab:.2f}"
          f" 仅在 data_fidelity='grad' 时需 <2）")
else:
    print(f"[配置] 观测半区稳定性指标 ε·λ/σ_d² = {_stab:.2f} (需 < 2.0) "
          f"{'✓' if _stab < 2.0 else '✗ 不稳定，请减小 step_size_coeff_annealed'}")


# ========================================================================
# 通用工具
# ========================================================================
def psnr(a, b):
    """两幅图像的 PSNR（数据范围 [0,1]）"""
    mse = float(((a - b) ** 2).mean())
    return float('inf') if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def _sha1_tensor(t):
    return hashlib.sha1(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def release_model(name):
    """释放模型并清理显存"""
    import gc
    if name in globals():
        globals()[name] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _derive_confidence_levels(n_samples):
    """根据样本数 S 动态推导校准曲线使用的经验分位数覆盖率。

    候选集 = {2k/S : k=1..S//2} ∩ (0,1)（排除退化的 100% 区间）；
    候选 ≤4 个时全部保留，否则按索引均匀取 4 个。
    典型输出：S=8→[.25,.5,.75]  S=16→[.125,.375,.625,.875]
              S=30→[.067,.333,.667,.933]  S=64→[.031,.344,.656,.969]
    """
    lv = [2 * k / n_samples for k in range(1, n_samples // 2 + 1)
          if 0 < 2 * k / n_samples < 1]
    if not lv:
        return [0.5]
    if len(lv) <= 4:
        return lv
    n = len(lv)
    return sorted({lv[int(round(i * (n - 1) / 3))] for i in range(4)})


def _calib_grade(dev, good=None, ok=None):
    good = CFG["calib_good"] if good is None else good
    ok = CFG["calib_ok"] if ok is None else ok
    if dev < good:
        return "校准良好", 0
    if dev < ok:
        return "存在偏差", 1
    return "严重失配", 2


def _null_empirical_coverage(n_samples, levels, n_trials=120000, seed=0):
    """S 个 iid 连续样本的线性插值经验分位数区间，在【完美校准】下对独立真值
    的实际期望覆盖率（分布无关，用标准正态蒙特卡洛即可）。

    必要性：S=30 时名义 93.3% 的区间实际取在第 1~30 个次序统计量之间，
    完美校准的期望覆盖率只有 ~87.3%（解析值 1−2(1+0.967)/31=0.8731），
    不是 93.3%。直接拿实测覆盖率对名义值
    打分（v2 的做法）会把有限样本离散性误判成"校准存在偏差"。
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_trials, n_samples))
    t = rng.standard_normal(n_trials)
    out = {}
    for cl in levels:
        a = (1.0 - cl) / 2.0
        ql = np.quantile(z, a, axis=1)
        qh = np.quantile(z, 1.0 - a, axis=1)
        out[float(cl)] = float(((t >= ql) & (t <= qh)).mean())
    return out


def _pit_rank(samples, x_true):
    """概率积分变换：真值在 S 个样本中的秩（0..S，结做随机化）。

    样本来自正确后验且真值独立时，每个像素的秩在 0..S 上均匀，期望 S/2=15。
    这是对"区间宽但中心偏"也敏感的校准检验——v2 缺失半区秩均值 18.8，
    说明样本整体偏暗（系统性偏差），尽管覆盖率看起来接近名义值。

    实现细节：无结时秩 = #below + U(-0.5, 0.5)，即 #below 的中点化随机化，
    均值恰为 E[#below] = S/2（与理论值对齐）。若用 U(0,1) 抖动会系统性 +0.5，
    在 S 小时（如 S=2）相对偏差达 50%，不可忽略。
    """
    lt = (samples < x_true).sum(0, keepdim=True).float()
    eq = (samples == x_true).sum(0, keepdim=True).float()
    g = torch.Generator(device=samples.device.type).manual_seed(20261003)
    jitter = torch.rand(lt.shape, generator=g, device=samples.device) * (eq + 1.0) - 0.5
    return (lt + jitter)


# ========================================================================
# ★ 核心：批量退火 PnP-ULA 采样器
# ========================================================================
def sample_annealed_ula(y_obs, A, At, denoiser, cfg, n_samples, device, seed=0,
                        desc="ULA", verbose=True, init_x=None, prox_op=None,
                        progress=False):
    """退火 PnP-ULA：从 p(x|y) ∝ p(y|x)p(x) 采样。

    显式梯度版（Laumont et al. 2022, Algorithm 2 的退火变体）：
        x_{k+1} = Π_[a,b]( x_k + ε_k ∇log p_{σ_k}(x_k)
                                 + ε_k λ A^T(y - A x_k)/σ_d²
                                 + noise_scale·sqrt(2ε_k)·z_k )
    近端版（MYULA, Durmus et al. 2018；cfg["data_fidelity"]="prox" 时使用）：
        x_{k+1} = Π_[a,b]( prox_{ε_k g}( x_k + ε_k ∇log p_{σ_k}(x_k)
                                          + noise_scale·sqrt(2ε_k)·z_k ) )
        其中 g(x)=½||Ax-y||²/σ_d²。对逐像素 inpainting，prox 在观测像素上为
        (x + (ε/σ_d²)·y)/(1 + ε/σ_d²)，缺失像素恒等。平稳分布不变（O(ε)
        近似），且观测像素更新无条件稳定（不存在 ε/σ_d²<2 的步长限制）。
    其中先验 score 用 Tweedie 公式：∇log p_σ(x) = (D(x,σ) - x)/σ²，
    D 是 σ 级去噪器（这里是 DRUNet），ε_k = c·σ_k²。

    ★ init_x：热启动初值（v3 默认传 DPIR-MAP 解）；None 时退化为 A†y 零填充。
    ★ prox_op(x, a)：数据项近端算子回调，a = ε·λ/σ_d²；None 时用显式梯度。
    ★ 全部 n_samples 条链作为一个 batch 同时推进（GPU 上快约 S 倍）。
    ★ A / At 以可调用对象传入，这样 Step 0 的解析自检可以复用完全相同的代码路径。

    返回 (n_samples, C, H, W) 的张量（在 device 上）。
    """
    gen = torch.Generator(device=device.type)
    gen.manual_seed(int(seed))

    with torch.no_grad():
        x0 = At(y_obs).detach() if init_x is None else init_x.detach()
        x = x0.repeat(n_samples, *([1] * (x0.dim() - 1)))
        x = x + cfg["ula_init_jitter"] * torch.randn(x.shape, generator=gen, device=device)

    lo, hi = cfg["ula_projection"]
    sched = cfg["sigma_schedule"]
    steps = cfg["steps_each_sigma"]
    lam = cfg["lambda_data"]
    coeff = cfg["step_size_coeff_annealed"]
    ns = cfg["ula_noise_scale"]
    inv_sd2 = 1.0 / (SIGMA_DATA ** 2)
    use_prox = prox_op is not None

    # 单条总进度条：覆盖全部 σ 级×每级步数，显示已用/剩余时间与当前 σ。
    # Colab（ipywidgets）与命令行（!python）都会由 tqdm.auto 自动选样式。
    total_steps = len(sched) * steps
    bar = tqdm(total=total_steps, desc=f"[{desc}] S={n_samples} 退火采样",
               disable=not progress, unit="步", smoothing=0.1)

    for si, sigma in enumerate(sched):
        eps = coeff * (sigma ** 2)
        noise_std = ns * math.sqrt(2.0 * eps)
        prox_a = eps * lam * inv_sd2          # prox_{εg} 的步长参数
        sigma_t = torch.full((n_samples, 1, 1, 1), float(sigma), device=device)
        t_lv = time.time()
        bar.set_postfix_str(f"σ={sigma:.4f} 级{si + 1}/{len(sched)}")
        for _ in range(steps):
            with torch.no_grad():
                x0_hat = denoiser(x, sigma_t)
                if isinstance(x0_hat, (tuple, list)):
                    x0_hat = x0_hat[0]
                score = (x0_hat - x) / (sigma ** 2)
                if use_prox:
                    # MYULA：先验走显式 score，数据项走近端
                    x = x + eps * score
                    x = x + noise_std * torch.randn(x.shape, generator=gen, device=device)
                    x = prox_op(x, prox_a)
                else:
                    data_grad = At(A(x) - y_obs) * inv_sd2
                    grad = score - lam * data_grad
                    if cfg.get("max_grad_norm"):
                        gn = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
                        scale = (cfg["max_grad_norm"] / gn).clamp_max(1.0)
                        grad = grad * scale.view(-1, *([1] * (grad.dim() - 1)))
                    x = x + eps * grad
                    x = x + noise_std * torch.randn(x.shape, generator=gen, device=device)
                x = x.clamp(lo, hi)
                if not torch.isfinite(x).all():
                    x = torch.nan_to_num(x, nan=0.5, posinf=hi, neginf=lo)
            bar.update(1)
        if verbose:
            tqdm.write(f"    [{desc}] σ={sigma:.4f} ({si + 1}/{len(sched)}) {steps}步 "
                       f"完成 {time.time() - t_lv:.1f}s")
    bar.close()
    return x.detach()


# ========================================================================
# ★ Step 0：采样器正确性自检（解析可解问题）
# ========================================================================
class _AnalyticGaussian:
    """线性高斯 inpainting：后验有闭式解，用来验证采样器。

        x ~ N(mu, Σ),  Σ = a²·exp(-d²/2l²) + b²I   （环面上的光滑 GP）
        y = M x + N(0, σ_d²I)
        Tweedie 去噪器 D(x,σ) = mu + Σ(Σ+σ²I)^{-1}(x-mu)   （对高斯先验是精确的）
        后验 p(x|y) = N(μ_p, C_p),  C_p = (Σ^{-1} + MᵀM/σ_d²)^{-1}
    """

    def __init__(self, side=10, a=0.13, b=0.06, ell=2.0, mean=0.5):
        self.side, self.N, self.mean = side, side * side, mean
        idx = torch.arange(side)
        coords = torch.stack(torch.meshgrid(idx, idx, indexing="ij"), -1).reshape(-1, 2)
        d = coords[:, None, :] - coords[None, :, :]
        d = torch.minimum(d.abs(), side - d.abs())
        Sigma = a ** 2 * torch.exp(-(d ** 2).sum(-1) / (2 * ell ** 2)) + b ** 2 * torch.eye(self.N)
        lam, U = torch.linalg.eigh(Sigma.double())
        self.lam, self.U = lam.clamp_min(1e-9), U

    def denoiser(self, x, sigma_t):
        """批量 Tweedie 去噪器，签名与 DRUNet 一致（x: (B,...,N), sigma_t: (B,1,1,1)）"""
        s = sigma_t.reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)      # (B,1,...,1)
        lam = self.lam.reshape(1, *([1] * (x.dim() - 2)), -1)             # (1,...,1,N)
        v = lam / (lam + s ** 2)                                          # (B,1,...,N)
        z = x - self.mean
        return ((z @ self.U) * v) @ self.U.T + self.mean

    def posterior(self, mask, y, s=0.0):
        P = (self.U * (1.0 / (self.lam + s ** 2))) @ self.U.T
        Ap = P + torch.diag(mask) / SIGMA_DATA ** 2
        C = torch.linalg.inv(Ap)
        mu = C @ (P @ torch.full((self.N,), float(self.mean), dtype=P.dtype)
                  + mask * y / SIGMA_DATA ** 2)
        return mu, C

    def sample_posterior(self, mu, C, n, gen):
        L = torch.linalg.cholesky(C + 1e-15 * torch.eye(self.N))
        return mu + torch.randn(n, self.N, generator=gen) @ L.T


def _field_lag1(field, weight=None):
    """单个 (C,H,W) 场的一阶空间自相关（逐通道平均）。

    weight: 可选的 (C,H,W) 0/1 权重，权重为 0 的像素被置零后不参与结构统计
    （用于把结构信号限制在缺失半区 —— 那里才是"不同合理猜测"出现的地方）。
    """
    vals = []
    for c in range(field.shape[0]):
        a = field[c].numpy()
        if weight is not None:
            a = np.where(weight[c].numpy() > 0.5, a, 0.0)
        a = a - a.mean()
        v = float((a * a).mean())
        if v > 1e-12:
            vals.append(0.5 * (float((a[:-1, :] * a[1:, :]).mean() / v)
                               + float((a[:, :-1] * a[:, 1:]).mean() / v)))
    return float(np.mean(vals)) if vals else float('nan')


def _deviation_lag1(samples_cpu, weight_cpu=None):
    """样本相对后验均值的偏差场的一阶空间自相关（逐样本逐通道平均）。

    这是判定"样本到底是不是后验样本"的关键指标：
      * 真正的后验样本，其在缺失区的差异应该是**图像状**的（边缘位置、纹理、亮度
        等不同但都合理的猜测），偏差场空间相关 → lag1 明显 > 0；
      * 如果样本只是"后验均值 + 逐像素白噪声"，偏差场是白的 → lag1 ≈ 0。
    weight_cpu: 可选 (C,H,W) 权重，把结构统计限制在缺失半区（观测半区的偏差
    被数据钉在 ≈0，会把全图 lag1 稀释到接近白噪声水平）。
    """
    dev = samples_cpu - samples_cpu.mean(0, keepdim=True)
    vals = []
    for s in range(dev.shape[0]):
        v = _field_lag1(dev[s], weight=weight_cpu)
        if not math.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


def run_analytic_selftest(cfg, verbose=True):
    """在同一份 sample_annealed_ula 代码上跑解析可解问题。

    这是"实验代码是否正确"的硬证据：如果采样器实现有误，这里立刻暴露。
    返回 (metrics_dict, figure_saved)
    """
    print("\n" + "=" * 72)
    print("Step 0: 采样器正确性自检（解析可解问题，后验有闭式解）")
    print("=" * 72)

    # 解析问题在 float64 下做（闭式解需要；200 维以内 CPU 上很快）
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return _run_analytic_selftest_impl(cfg, verbose)
    finally:
        torch.set_default_dtype(_prev_dtype)


def _run_analytic_selftest_impl(cfg, verbose=True):
    prob = _AnalyticGaussian(side=10)
    N = prob.N
    gen = torch.Generator().manual_seed(7)
    mask = (torch.rand(N, generator=gen) < 0.5).double()
    x_true = (torch.randn(1, N, generator=gen) * prob.lam.sqrt()) @ prob.U.T + prob.mean
    x_true = x_true[0]
    y = mask * x_true + SIGMA_DATA * torch.randn(N, generator=gen)
    ym = mask * y

    A = lambda t: t * mask
    At = lambda t: t * mask

    mu0, C0 = prob.posterior(mask, ym, 0.0)
    sd_ref = C0.diag().clamp_min(0).sqrt()
    obs, mis = mask > 0.5, mask < 0.5

    n_chains = 300
    variants = []

    # (a) 初版配置
    old = dict(cfg)
    old.update(dict(sigma_schedule=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
                    steps_each_sigma=50, lambda_data=0.05,
                    ula_projection=(-0.1, 1.1)))
    variants.append(("初版配置 (λ=0.05, 50步, σ→1.0)", old))
    # (b) 本版配置
    NEW_KEY = (f"本版配置 (λ={cfg['lambda_data']}, {cfg['steps_each_sigma']}步/级, "
               f"{len(cfg['sigma_schedule'])}级, σ≤{cfg['sigma_max']:.3f})")
    variants.append((NEW_KEY, dict(cfg)))
    # (c) 只改 λ
    mid = dict(cfg)
    mid.update(dict(sigma_schedule=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01],
                    steps_each_sigma=50, lambda_data=1.0,
                    ula_projection=(-0.1, 1.1)))
    variants.append(("只改 λ=1.0 (50步, σ→1.0)", mid))

    # 参考：直接从闭式后验采样
    ref = prob.sample_posterior(mu0, C0, n_chains, gen)
    handles = [("精确后验样本（参考）", ref, None)]
    for name, vcfg in variants:
        smp = sample_annealed_ula(ym.reshape(1, 1, -1), A, At, prob.denoiser, vcfg,
                                  n_chains, torch.device('cpu'), seed=11,
                                  desc=f"自检{name[:10]}", verbose=False)
        handles.append((name, smp.reshape(n_chains, N), None))

    print(f"\n{'配置':<40s} {'std观测':>8s} {'std缺失':>8s} {'覆盖率93.3%':>11s}")
    print("-" * 74)
    print(f"{'解析后验（真值）':<40s} {sd_ref[obs].mean():8.5f} {sd_ref[mis].mean():8.5f} "
          f"{'93.3%(名义)':>11s}")
    fig_data = {}
    for name, smp, _ in handles:
        emp_sd = smp.var(0, unbiased=True).clamp_min(0).sqrt()
        ql = torch.quantile(smp, (1 - 0.9333333) / 2, dim=0)
        qh = torch.quantile(smp, 1 - (1 - 0.9333333) / 2, dim=0)
        inside = (x_true >= ql) & (x_true <= qh)
        # emp_sd 与 sd_ref 都已经是标准差，比值即“实测 std / 真实后验 std”，
        # 不能再开平方根（否则报告的只是标准差比的平方根，验收区间也会被放宽）。
        r_o = float(emp_sd[obs].mean() / sd_ref[obs].mean())
        r_m = float(emp_sd[mis].mean() / sd_ref[mis].mean())
        cov = float(inside.float().mean())
        print(f"{name:<40s} {r_o:8.2f} {r_m:8.2f} {cov * 100:10.1f}%")
        fig_data[name] = dict(emp_sd=emp_sd.numpy(), ratio_obs=r_o, ratio_mis=r_m, cov=cov)

    # 作图
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        ax = axes[0]
        ref_sd = sd_ref.numpy()
        ax.scatter(ref_sd[obs.numpy()], fig_data[NEW_KEY]["emp_sd"][obs.numpy()],
                   s=14, alpha=0.7, label="本版配置", color="#2b7bba")
        ax.scatter(ref_sd[mis.numpy()], fig_data[NEW_KEY]["emp_sd"][mis.numpy()],
                   s=14, alpha=0.7, color="#2b7bba")
        ax.scatter(ref_sd, fig_data["初版配置 (λ=0.05, 50步, σ→1.0)"]["emp_sd"], s=14, alpha=0.6,
                   label="初版配置", color="#c0392b", marker="^")
        lim = max(ref_sd.max(), max(v["emp_sd"].max() for v in fig_data.values())) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=1, label="完美校准 (y=x)")
        ax.set_xlabel("解析后验 std"); ax.set_ylabel("采样得到的 std")
        ax.set_title("Step 0-a 采样器是否收敛到真实后验\n（点应落在 y=x 上）")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = axes[1]
        names = list(fig_data.keys())
        xpos = np.arange(len(names))
        ax.bar(xpos - 0.2, [fig_data[n]["ratio_obs"] for n in names], width=0.4,
               label="观测半区", color="#2b7bba")
        ax.bar(xpos + 0.2, [fig_data[n]["ratio_mis"] for n in names], width=0.4,
               label="缺失半区", color="#e67e22")
        ax.axhline(1.0, color='k', ls='--', lw=1)
        ax.axhspan(0.8, 1.25, color='green', alpha=0.10)
        ax.set_xticks(xpos)
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_ylabel("实测 std / 真实后验 std")
        ax.set_title("Step 0-b 标准差标定\n（理想值 1.00，绿带 = 可接受范围）")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        out = os.path.join(SAVE_DIR, "step0_sampler_selftest.png")
        plt.savefig(out, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f"\n[图] 已保存: {out}")
    except Exception as e:
        print(f"[警告] Step 0 作图失败: {e}")

    best = fig_data[NEW_KEY]
    ok = (0.8 <= best["ratio_obs"] <= 1.25) and (0.7 <= best["ratio_mis"] <= 1.4) \
        and (best["cov"] >= 0.85)
    print(f"\n[自检结论] 本版配置 std 比值 观测 {best['ratio_obs']:.2f} / 缺失 {best['ratio_mis']:.2f}，"
          f"名义覆盖率 93.3% 实测 {best['cov'] * 100:.1f}% —— "
          f"{'✓ 采样器通过解析验证' if ok else '✗ 采样器未通过，请检查配置'}")
    print("[自检范围] 该结果只证明：① 迭代公式/λ/σ 调度/步数在【共轭高斯、精确线性 score、"
          "单盆域强相关】后验上正确；")
    print("           不能外推到真实 DRUNet：自然图像后验非凸且去噪器只近似 score，"
          "粗尺度填充误差模态的输送时间远长于本玩具。")
    print("           真实实验的收敛性必须另看 MAP 基线对比 / PIT / std 比（Step 3 验收），"
          "这正是 v3 增加这些检查的原因。")
    return dict(rows=fig_data, passed=bool(ok), new_key=NEW_KEY,
                ratio_obs=best["ratio_obs"], ratio_mis=best["ratio_mis"], cov=best["cov"])


# ========================================================================
# 数据与物理算子
# ========================================================================
def load_celeba_image(img_size, device):
    """读取示例图（与 deepinv.load_example(..., resize_mode='resize') 等价）。

    直接使用 PIL，避免依赖 torchvision 的 ABI 版本。
    """
    from PIL import Image
    import urllib.request

    url = ("https://huggingface.co/datasets/deepinv/images/resolve/main/celeba_example.jpg")
    candidates = [
        os.path.join(SAVE_DIR, "celeba_example.jpg"),
        os.path.join(os.path.expanduser("~"), ".cache", "deepinv", "url_cache",
                     "huggingface.co", "datasets", "deepinv", "images", "resolve", "main",
                     "celeba_example.jpg"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        path = candidates[0]
        print(f"[数据] 正在下载示例图 -> {path}")
        urllib.request.urlretrieve(url, path)
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def build_inpainting_physics(x_true, cfg, device):
    """显式、可复现地构造 inpainting 掩码。

    ★ 初版把 mask=0.5 交给 deepinv 内部生成，而 deepinv 用的是
      `torch.Generator(device=device)`（见 deepinv/physics/generator/base.py），
      这个生成器**不受 torch.manual_seed 控制**，而且状态依赖设备；
      结果是掩码既没被记录、也无法跨设备复现，缓存里也没存它。
      这里改成：固定 CPU 种子生成 → 存进缓存 → 指纹里放 mask hash。
    """
    C, H, W = x_true.shape[1:]
    g = torch.Generator(device="cpu").manual_seed(int(cfg["mask_seed"]))
    aux = torch.rand((C, H, W), generator=g)
    mask = torch.ones((C, H, W))
    mask[:, aux[0] > (1.0 - cfg["mask_ratio"])] = 0.0   # 保留 mask_ratio 比例的像素
    mask = mask.unsqueeze(0).to(device)
    physics = Inpainting(mask=mask, img_size=(C, H, W), device=device)
    physics.set_noise_model(GaussianNoise(sigma=cfg["sigma_data"]))
    return physics, mask


# ========================================================================
# ★ v3：DPIR-MAP 热启动
# ========================================================================
def map_warmstart_dpir(y_obs, mask, denoiser, cfg, device, verbose=True):
    """DPIR/HQS 式 MAP 近似：逐级去噪 + 观测像素硬投影（不注入采样噪声）。

    注意：这不是完整 DPIR——它只借“去噪 + 数据投影”的结构做热启动与 MAP 基线，
    缺少 DPIR 的逐级噪声调度与数据保真近端步（步长/权重按 σ² 缩放）。

        x ← D_σ(x)；x[观测] ← y[观测]
    这是退火 ULA 的 σ→0 确定性极限（DPIR, Zhang et al. 2021），给出后验众数
    附近的重建。用它热启动的理由（见文件头 v3 说明）：零填充初始化时，
    "50% 黑洞"这种粗尺度填充误差只有高 σ 级能搬运，而 400 步/级在真实
    DRUNet 上远不足以完成输送；先用 96 次硬投影去噪把链放进正确盆域，
    后续 Langevin 只需在该盆域内做局部平衡。
    返回 (MAP 图像, σ 调度表)。
    """
    levels = list(np.geomspace(cfg["map_sigma_max"], cfg["sigma_min"],
                               cfg["map_levels"]))
    x = y_obs.clone()                       # 从零填充 A†y 开始
    t0 = time.time()
    n_map = len(levels) * cfg["map_steps_per_level"]
    mbar = tqdm(total=n_map, desc="[MAP] DPIR预热", unit="次",
                disable=not verbose)
    with torch.no_grad():
        for si, sigma in enumerate(levels):
            sigma_t = torch.full((1, 1, 1, 1), float(sigma), device=device)
            mbar.set_postfix_str(f"σ={sigma:.4f} 级{si + 1}/{len(levels)}")
            for _ in range(cfg["map_steps_per_level"]):
                x = denoiser(x, sigma_t)
                if isinstance(x, (tuple, list)):
                    x = x[0]
                x = mask * y_obs + (1.0 - mask) * x   # inpainting 精确数据投影
                mbar.update(1)
    mbar.close()
    if verbose:
        print(f"[MAP] DPIR 预热完成：{len(levels)} 级 × "
              f"{cfg['map_steps_per_level']} 次 = "
              f"{len(levels) * cfg['map_steps_per_level']} 次去噪，"
              f"{time.time() - t0:.1f}s")
    return x.detach(), levels


# ========================================================================
# 缓存
# ========================================================================
# 缓存文件名包含样本数，避免 quick_test（S=4）的运行覆盖全量（S=30）的缓存
CACHE_FILE = os.path.join(SAVE_DIR, f"experiment_cache_S{N_SAMPLES}.pkl")
# 旧版全量缓存仍在 experiment_cache.pkl（v1 格式，指纹不同会自动重采）


def compute_fingerprint(x_true, mask, cfg, device):
    return {
        "cache_version": cfg["cache_version"],
        "sampler": "annealed_pnp_ula_batched_v2",
        "x_true_hash": _sha1_tensor(x_true),
        "x_true_mean": round(float(x_true.mean()), 6),
        "x_true_std": round(float(x_true.std()), 6),
        "x_true_shape": list(x_true.shape),
        "mask_hash": _sha1_tensor(mask),          # ★ 初版缺失
        "mask_keep_ratio": round(float(mask.mean()), 6),
        "mask_seed": cfg["mask_seed"],
        "sigma_data": cfg["sigma_data"],
        "sigma_schedule": [round(float(s), 8) for s in cfg["sigma_schedule"]],
        "steps_each_sigma": cfg["steps_each_sigma"],
        "lambda_data": cfg["lambda_data"],
        "step_size_coeff_annealed": cfg["step_size_coeff_annealed"],
        "ula_noise_scale": cfg["ula_noise_scale"],
        "ula_projection": list(cfg["ula_projection"]),
        "n_samples": cfg["n_samples"],
        "img_size": cfg["image_size"],
        # v3：热启动与数据项形式直接决定样本来自哪个分布，必须进指纹
        "init_mode": cfg["init_mode"],
        "data_fidelity": cfg["data_fidelity"],
        "map_sigma_max": cfg["map_sigma_max"],
        "map_levels": cfg["map_levels"],
        "map_steps_per_level": cfg["map_steps_per_level"],
        "device": str(device),
    }


def load_cache(fingerprint):
    if not CFG["use_cache"] or not os.path.isfile(CACHE_FILE):
        return None
    try:
        obj = torch.load(CACHE_FILE, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[缓存] 读取失败({type(e).__name__}), 忽略: {e}")
        return None
    if not isinstance(obj, dict) or "fingerprint" not in obj:
        print("[缓存] 格式不符, 忽略")
        return None
    # ★ 必须先校验指纹：样本数够不代表配置一致（改 λ/σ 调度后若不校验，
    #   会静默复用旧超参下采出的样本，与"跨设备可复现"的声明矛盾）
    diff = [k for k in set(list(obj["fingerprint"].keys()) + list(fingerprint.keys()))
            if obj["fingerprint"].get(k) != fingerprint.get(k)]
    if diff:
        print(f"[缓存] 指纹不匹配({diff[:6]}), 重新采样")
        return None
    n = len(obj.get("samples", []))
    if n >= N_SAMPLES:
        print(f"[缓存] 命中（{n} 个样本）")
        return obj
    print(f"[缓存] 指纹匹配但样本不足（{n} < {N_SAMPLES}），重新采样")
    return None


def save_cache(fingerprint, samples, methods, times, extra=None):
    if not CFG["use_cache"]:
        return
    payload = {"fingerprint": fingerprint, "samples": samples,
               "methods": methods, "times": times}
    if extra:
        payload.update(extra)
    tmp = CACHE_FILE + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, CACHE_FILE)


# ========================================================================
# Step 0
# ========================================================================
selftest = run_analytic_selftest(CFG)


# ========================================================================
# Step 1: 数据准备
# ========================================================================
print("\n" + "=" * 72)
print("Step 1: 从点估计到分布 —— 数据与观测")
print("=" * 72)

x_true = load_celeba_image(IMG_SIZE, device)
physics_inp, MASK = build_inpainting_physics(x_true, CFG, device)

with torch.no_grad():
    y_inp = physics_inp(x_true)
    x_zf = physics_inp.A_dagger(y_inp)          # 零填充 = 伪逆

keep_ratio = float(MASK.mean())
psnr_zf = psnr(x_true, x_zf)
psnr_obs_noise = 10.0 * math.log10(1.0 / SIGMA_DATA ** 2)
print(f"图像: {tuple(x_true.shape)}  (像素均值 {float(x_true.mean()):.3f})")
print(f"观测模型: 逐像素 inpainting, 保留比例 {keep_ratio:.1%}, 加性高斯噪声 σ={SIGMA_DATA}")
print(f"观测噪声本身对应 SNR = {psnr_obs_noise:.1f} dB")
print(f"零填充(伪逆 A†y)基线 PSNR = {psnr_zf:.2f} dB   ← 初版把它误标为『观测PSNR』")
print(f"   注: 该值低是因为一半像素被置零，与噪声水平无关；它是重建的合理下界。")

# 偏差场结构参考在 Step 3 计算（诊断用白噪声对照）


# ========================================================================
# Step 2: 后验采样
# ========================================================================
print("\n" + "=" * 72)
print("Step 2: 后验采样（退火 PnP-ULA, DRUNet 先验）")
print("=" * 72)

FINGERPRINT = compute_fingerprint(x_true, MASK, CFG, device)
cached = load_cache(FINGERPRINT)

all_samples, sample_methods, sample_times = [], [], []
MAP_X = cached.get("map_x") if cached is not None else None

if cached is not None and len(cached["samples"]) >= N_SAMPLES:
    all_samples = [s for s in cached["samples"][:N_SAMPLES]]
    sample_methods = list(cached["methods"][:N_SAMPLES])
    sample_times = list(cached["times"][:N_SAMPLES])
    print(f"[缓存] 复用 {len(all_samples)} 个样本")
    if MAP_X is not None:
        print(f"[缓存] 复用 MAP 热启动图（{psnr(x_true, MAP_X.to(device)):.2f} dB）")

if len(all_samples) < N_SAMPLES:
    t_start_all = time.time()
    try:
        print("[ULA] 加载 DRUNet 去噪器…")
        denoiser = DRUNet(in_channels=x_true.shape[1], out_channels=x_true.shape[1],
                          pretrained="download", device=device).to(device).eval()
        for p in denoiser.parameters():
            p.requires_grad_(False)

        # ---- v3：MAP 热启动（修复零填充初始化的粗尺度输送失败）----
        init_x = None
        if CFG["init_mode"] == "map_warmstart":
            if MAP_X is None:
                print("[ULA] 先做 DPIR-MAP 热启动…")
                MAP_X, _ = map_warmstart_dpir(y_inp, MASK, denoiser, CFG, device)
            else:
                MAP_X = MAP_X.to(device)
            init_x = MAP_X
            print(f"[MAP] 热启动图 PSNR: 整体 {psnr(x_true, MAP_X):.2f} dB / "
                  f"观测 {psnr(x_true[0][MASK[0] > 0.5], MAP_X[0][MASK[0] > 0.5]):.2f} / "
                  f"缺失 {psnr(x_true[0][MASK[0] < 0.5], MAP_X[0][MASK[0] < 0.5]):.2f}")
        else:
            print("[ULA] init_mode=zerofill（v2 行为，仅用于对照：零填充初始化）")

        # ---- v3：MYULA 近端数据项（inpainting 闭式 prox）----
        if CFG["data_fidelity"] == "prox":
            def _inpaint_prox(x, a):
                # prox_{a·g}(x)，g=½||Mx-y||²/σ_d²；观测像素凸组合，缺失像素恒等
                return (x + a * y_inp) / (1.0 + a * MASK)
            prox_op = _inpaint_prox
            print("[ULA] 数据项：MYULA 近端步（无条件稳定）")
        else:
            prox_op = None
            print("[ULA] 数据项：显式梯度（需 ελ/σ_d²<2）")

        chunk = max(1, min(CFG["sample_chunk"], N_SAMPLES - len(all_samples)))
        while len(all_samples) < N_SAMPLES:
            need = min(chunk, N_SAMPLES - len(all_samples))
            print(f"[ULA] 采样第 {len(all_samples) + 1}~{len(all_samples) + need} 个样本"
                  f"（batch={need}）…")
            t0 = time.time()
            with torch.no_grad():
                xb = sample_annealed_ula(y_inp, physics_inp.A, physics_inp.A_adjoint,
                                         denoiser, CFG, need, device,
                                         seed=1000 + len(all_samples),
                                         desc="ULA", verbose=False,
                                         init_x=init_x, prox_op=prox_op,
                                         progress=True)
            dt = (time.time() - t0) / need
            for i in range(need):
                all_samples.append(xb[i:i + 1].detach().cpu())
                sample_methods.append("退火ULA")
                sample_times.append(dt)
            del xb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[ULA] 本批完成: {dt:.1f} s/样本")
            save_cache(FINGERPRINT, all_samples, sample_methods, sample_times,
                       extra={"mask": MASK.cpu(), "y_inp": y_inp.cpu(),
                              "x_true": x_true.cpu(),
                              "map_x": (MAP_X.cpu() if MAP_X is not None else None)})
        print(f"[ULA] 完成: {len(all_samples)} 个样本, 总耗时 {time.time() - t_start_all:.1f}s")
    except Exception as e:
        print(f"[ULA] 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        release_model("denoiser")

    if len(all_samples) < N_SAMPLES:
        # ---- 兜底：伪逆 + 噪声（不是后验采样！）----
        print("\n" + "!" * 72)
        print("!! 退化兜底：伪逆 + 噪声")
        print("!! 这【不是】后验采样，只是后验均值附近的随机扰动，")
        print("!! 其 std 完全由扰动幅度决定。下面所有 UQ 结论都会被标记为不可信。")
        print("!" * 72)
        g = torch.Generator(device="cpu").manual_seed(SEED)
        while len(all_samples) < N_SAMPLES:
            xa = x_zf + 0.02 * torch.randn(x_true.shape, generator=g)
            all_samples.append(xa.cpu())
            sample_methods.append("伪逆+噪声(非后验)")
            sample_times.append(0.0)

if all(m == "退火ULA" for m in sample_methods):
    ula_method, IS_POSTERIOR = "退火ULA", True
elif all("非后验" in m for m in sample_methods):
    ula_method, IS_POSTERIOR = "伪逆+噪声(非后验)", False
else:
    ula_method, IS_POSTERIOR = "混合采样", False
print(f"\n最终采样方法: {ula_method}, 样本数: {len(all_samples)}")
if not IS_POSTERIOR:
    print("⚠️ 采样器未成功运行 —— 本次实验的『不确定度』不能被解释为后验不确定度。")

SAMPLES = torch.cat(all_samples, 0).to(device)          # (S,C,H,W)
S = SAMPLES.shape[0]
# ★ 口径说明：下文所有统计量（PSNR/std/覆盖率）都在采样器原始输出（已投影到 [-0.25,1.25]）
#   上计算；clamp(0,1) 仅用于绘图。实测两口径的 PSNR 差 ≤0.013 dB（2 位小数下个别值差 0.01），
#   覆盖率 93.33% 均为 0.9217。缓存样本最小值约 -0.021，说明投影边界偶尔生效。
sample_psnrs = [psnr(x_true, SAMPLES[i:i + 1]) for i in range(S)]


# ========================================================================
# Step 3: 不确定性地图 + 结构诊断
# ========================================================================
print("\n" + "=" * 72)
print("Step 3: 不确定性地图与结构诊断")
print("=" * 72)

posterior_mean = SAMPLES.mean(0, keepdim=True)
posterior_std = SAMPLES.std(0, keepdim=True)

mask_el = MASK.bool().expand_as(posterior_std)     # 逐元素：True = 有观测
obs_pix = MASK[0].sum(0) > 0
mis_pix = ~obs_pix
xt = x_true[0]
pm = posterior_mean[0]
ps = posterior_std[0]

# ---- 分区误差分解 ----
def _rms(t):
    return float(t.pow(2).mean().sqrt())

err_mean = pm - xt
err_samp = SAMPLES - x_true
rms_mean_obs = _rms(err_mean[mask_el[0]])
rms_mean_mis = _rms(err_mean[~mask_el[0]])
rms_samp_obs = _rms(err_samp[mask_el.expand_as(SAMPLES)])
rms_samp_mis = _rms(err_samp[~mask_el.expand_as(SAMPLES)])
std_obs = float(ps[mask_el[0]].pow(2).mean().sqrt())
std_mis = float(ps[~mask_el[0]].pow(2).mean().sqrt())
tot = float(err_mean.pow(2).sum())
frac_mis = float(err_mean[~mask_el[0]].pow(2).sum()) / max(tot, 1e-12)

print(f"\n后验均值 PSNR: 整体 {psnr(x_true, posterior_mean):.2f} dB")
print(f"  观测半区 {psnr(xt[mask_el[0]], pm[mask_el[0]]):.2f} dB   "
      f"缺失半区 {psnr(xt[~mask_el[0]], pm[~mask_el[0]]):.2f} dB")
print(f"  单样本 PSNR: {min(sample_psnrs):.2f} ~ {max(sample_psnrs):.2f} dB")
print(f"  后验均值平方误差有 {frac_mis:.1%} 落在缺失半区 —— "
      f"所以整体 PSNR 必须分区看，池化数字被未约束那一半主导。")
print(f"\n标准差: 观测半区 {std_obs:.4f} (= {std_obs / SIGMA_DATA:.2f}×σ_d)   "
      f"缺失半区 {std_mis:.4f}   整体 {float(posterior_std.mean()):.4f}")
print(f"  ★ 判据: 观测半区的后验 std 应当 ≈ σ_d = {SIGMA_DATA}（观测已经把该像素定住了），"
      f"实测 {std_obs / SIGMA_DATA:.2f}×；缺失/观测 std 比 = {std_mis / std_obs:.2f}")

# ---- ★ v3：与 MAP 基线对比（抓"全体链一致待在错误盆域"）----
map_psnr = map_psnr_obs = map_psnr_mis = None
map_rms_mis = mean_over_map_rms = None
if MAP_X is not None:
    MAP_X = MAP_X.to(device)
    map_psnr = psnr(x_true, MAP_X)
    map_psnr_obs = psnr(xt[mask_el[0]], MAP_X[0][mask_el[0]])
    map_psnr_mis = psnr(xt[~mask_el[0]], MAP_X[0][~mask_el[0]])
    map_rms_mis = _rms(MAP_X[0][~mask_el[0]] - xt[~mask_el[0]])
    mean_over_map_rms = rms_mean_mis / max(map_rms_mis, 1e-12)
    print(f"\n★ vs MAP 基线（同一个 DRUNet 的 DPIR 式热启动近似：去噪+硬投影，非完整 DPIR）:")
    print(f"  MAP      : 整体 {map_psnr:.2f} / 观测 {map_psnr_obs:.2f} / "
          f"缺失 {map_psnr_mis:.2f} dB（缺失 RMS {map_rms_mis:.4f}）")
    print(f"  后验均值 : 整体 {psnr(x_true, posterior_mean):.2f} / "
          f"观测 {psnr(xt[mask_el[0]], pm[mask_el[0]]):.2f} / "
          f"缺失 {psnr(xt[~mask_el[0]], pm[~mask_el[0]]):.2f} dB"
          f"（缺失 RMS {rms_mean_mis:.4f}）")
    print(f"  缺失 RMS 比值（后验均值/MAP）= {mean_over_map_rms:.2f}"
          f"  ← 应在 1~2 之间；v2 失败样本为 12.8（链没到后验盆域）")
    print(f"  注：后验均值的期望误差 = 偏差²+后验方差，略高于 MAP 是正常的，"
          f"但不应差出数量级。")

# ---- 偏差场结构诊断（区分"后验样本"与"均值+白噪声"）----
# 观测半区的偏差被数据钉在 ≈0，全图 lag1 会被稀释；
# 因此把结构统计限制在缺失半区（weight = 1-观测掩码），同时保留全图值作对照。
_w_mis = 1.0 - MASK[0].float()                       # (C,H,W): 缺失半区=1
dev_lag1 = _deviation_lag1(SAMPLES.cpu(), _w_mis.cpu())          # 缺失半区（主判据）
dev_lag1_full = _deviation_lag1(SAMPLES.cpu())                   # 全图（对照）
_g = torch.Generator(device=device.type).manual_seed(0)
_wn = posterior_mean + posterior_std * torch.randn(SAMPLES.shape, generator=_g, device=device)
wn_lag1 = _deviation_lag1(_wn.cpu(), _w_mis.cpu())
wn_lag1_full = _deviation_lag1(_wn.cpu())
bias_lag1 = _field_lag1((posterior_mean - x_true)[0].cpu(), _w_mis[0].cpu())
print(f"\n偏差场一阶空间自相关 lag1（结构诊断；缺失半区为主判据）:")
print(f"  真实样本 (x_s - x̄) 缺失半区: {dev_lag1:+.4f}   (全图 {dev_lag1_full:+.4f})")
print(f"  白噪声对照(同像素 std) 缺失半区: {wn_lag1:+.4f}   (全图 {wn_lag1_full:+.4f})")
print(f"  重建偏差 (x̄ - x_true) 缺失半区: {bias_lag1:+.4f}"
      f"   ← 图像状结构应有的量级")
print(f"  ★ 判据（必要非充分）: x_s-x̄ 在缺失半区空间相关(lag1 明显大于白噪声对照)，"
      f"只能排除『后验均值+逐像素白噪声』式赝品；\n     它不能证明样本来自正确后验——"
      f"v2 的失败样本 lag1 也显著（残留噪声被去噪器着了色），但全体链都停在错误"
      f"盆域。后验正确性还要看上面的 MAP 对比、PIT 秩与零模型校准。")
structure_ratio = dev_lag1 / max(abs(wn_lag1), 1e-6)

# ---- ★ v3：PIT 真值秩检验（对"区间宽但中心偏"敏感）----
pit_rank = _pit_rank(SAMPLES, x_true)
pit_obs = float(pit_rank[mask_el].mean())
pit_mis = float(pit_rank[~mask_el].mean())
pit_theo = S / 2.0
print(f"\nPIT 真值秩（理论均值 S/2={pit_theo:.0f}，校准良好时各像素秩在 0..{S} 均匀）:")
print(f"  观测半区 {pit_obs:.2f}   缺失半区 {pit_mis:.2f}   "
      f"偏差: 观测 {pit_obs-pit_theo:+.2f} / 缺失 {pit_mis-pit_theo:+.2f}")
print(f"  ★ 缺失半区 |秩均值-{pit_theo:.0f}| > 3 表示样本整体偏亮/偏暗（系统性偏差），"
      f"区间再宽也救不回来；v2 失败样本为 18.8。")

# ---- 经验分位数区间 ----
confidence_levels = _derive_confidence_levels(S)
cl_max = max(confidence_levels)
print(f"\n[校准] 使用置信水平: {[round(c, 3) for c in confidence_levels]} (S={S})")

ci_low, ci_high, coverages = {}, {}, {}
for cl in confidence_levels:
    ql = torch.quantile(SAMPLES, (1 - cl) / 2, dim=0)
    qh = torch.quantile(SAMPLES, 1 - (1 - cl) / 2, dim=0)
    ci_low[cl], ci_high[cl] = ql, qh
    inside = (x_true[0] >= ql) & (x_true[0] <= qh)
    coverages[cl] = dict(pool=float(inside.float().mean()),
                         obs=float(inside[mask_el[0]].float().mean()),
                         mis=float(inside[~mask_el[0]].float().mean()))
ci_width = ci_high[cl_max] - ci_low[cl_max]          # (C,H,W)

# ---- 链间相关性 ----
_dev = (SAMPLES - posterior_mean).reshape(S, -1)
_dev = _dev - _dev.mean(0, keepdim=True)
_dn = _dev / _dev.norm(dim=1, keepdim=True).clamp_min(1e-12)
_C = (_dn @ _dn.T).cpu().numpy()
_off = _C[np.triu_indices(S, 1)] if S > 1 else np.array([0.0])
rho = float(_off.mean()) if len(_off) else 0.0
rho_centered = -1.0 / max(S - 1, 1)       # 中心化偏差场的代数基线，不是“独立链理论值”
# 中心化后每条链的向量和为零，任意两条链的相关系数必然 ≈ -1/(S-1)；
# 因此 rho 与基线一致不能证明链独立，只能排除“全体塌缩到同一点”（rho≈0）
# 或“塌到 k 个模态”（rho≈1/k）。收敛性判定以上面的 MAP 基线对比、PIT、lag1 为准。
if abs(rho - rho_centered) < 0.05 * max(abs(rho_centered), 0.05):
    ess = float(S)   # 由中心化 rho 反推的粗估，仅作参照
else:
    ess = float(min(S, max(1.0, S / max(1.0 + (S - 1) * rho, 1e-6))))
print(f"[样本相关性] 中心化偏差场的链间相关均值 {rho:+.4f} "
      f"(代数基线 {rho_centered:+.4f}), ESS 粗估 ≈ {ess:.1f}（仅作参照）")
if abs(rho - rho_centered) < 0.05 * max(abs(rho_centered), 0.05):
    print("  与中心化基线一致：未检测到全塌缩/多模态塌缩；这不构成链独立的证据。")
    print("  该统计量由中心化强行固定基线，不能区分独立链与部分相关的链；")
    print("  真正判定『样本是不是后验样本』的是上面的 MAP 基线对比、PIT 与偏差场 lag1。")
else:
    print("  ⚠️ 中心化链间相关偏离代数基线，提示链间可能存在塌缩/强相关，UQ 需谨慎。")


# ========================================================================
# Step 4: 校准检验
# ========================================================================
print("\n" + "=" * 72)
print("Step 4: ★ 校准检验（覆盖率）")
print("=" * 72)
print("注意：① 覆盖率把所有像素池化，像素高度相关，统计不确定性远大于二项估计；")
print("     ② S 个样本的经验分位数区间在完美校准下也达不到名义覆盖率（S=30 时")
print("        93.3% 名义区间的零模型期望只有 ~87.3%），评级必须对零模型、不对名义值；")
print("     ③ 『是否能覆盖』在观测半区与缺失半区是两件不同的事，必须分开看。\n")
null_covs = _null_empirical_coverage(S, confidence_levels)
print(f"{'名义覆盖率':>9s} {'零模型期望':>10s} {'池化':>8s} {'观测半区':>9s} {'缺失半区':>9s} "
      f"{'对名义':>8s} {'对零模型':>9s} {'状态':>9s}")
print("-" * 78)
grades = {}
for cl in confidence_levels:
    c = coverages[cl]
    null_c = null_covs[float(cl)]
    dev_nom = abs(c['pool'] - cl)
    dev_null = abs(c['pool'] - null_c)
    g, sev = _calib_grade(dev_null)
    grades[cl] = sev
    print(f"{cl:>9.4f} {null_c:>10.4f} {c['pool']:>8.4f} {c['obs']:>9.4f} {c['mis']:>9.4f} "
          f"{c['pool']-cl:>+8.4f} {c['pool']-null_c:>+9.4f} {g:>9s}")

cov_main = coverages[cl_max]
null_main = null_covs[float(cl_max)]
dev_main = abs(cov_main['pool'] - null_main)
grade_main, sev_main = _calib_grade(dev_main)
print(f"\n主置信水平 {cl_max:.2%}（零模型期望 {null_main:.2%}）: "
      f"池化覆盖率 {cov_main['pool']:.2%} "
      f"(观测 {cov_main['obs']:.2%} / 缺失 {cov_main['mis']:.2%})，"
      f"对零模型偏差 {cov_main['pool']-null_main:+.1%}，{grade_main}")


# ========================================================================
# Step 5: 样本数影响
# ========================================================================
print("\n" + "=" * 72)
print("Step 5: ★ 样本数 S 对不确定性估计的影响")
print("=" * 72)

s4_comparison = {}
psnr_by_s = None
if S >= 4:
    n_list = sorted({2, 4, 8, 16, S})
    n_list = [n for n in n_list if n <= S]
    psnr_by_s, std_err_by_s, std_by_s = [], [], []
    for n in n_list:
        sub = SAMPLES[:n]
        m = sub.mean(0, keepdim=True)
        # 后验均值这一估计量的标准误：链间标准差 / sqrt(n)。
        # 注意：它与“像素 std 估计量自身的误差”不是同一个量（后者见附录 18B）。
        se = sub.std(0, keepdim=True) / math.sqrt(n)
        psnr_by_s.append(psnr(x_true, m))
        std_err_by_s.append(float(se.mean()))
        std_by_s.append(float(sub.std(0, keepdim=True).mean()))
    print(f"{'S':>4s} {'后验均值PSNR':>13s} {'像素std均值':>12s} {'后验均值标准误':>16s}")
    print("-" * 50)
    for i, n in enumerate(n_list):
        print(f"{n:>4d} {psnr_by_s[i]:>13.2f} {std_by_s[i]:>12.4f} {std_err_by_s[i]:>16.4f}")
    j4 = n_list.index(4) if 4 in n_list else 0
    jS = len(n_list) - 1
    s4_comparison = {
        "S4_PSNR": round(psnr_by_s[j4], 2),
        f"S{S}_PSNR": round(psnr_by_s[jS], 2),
        "S4_vs_full_PSNR差": round(psnr_by_s[jS] - psnr_by_s[j4], 2),
        "S4_std": round(std_by_s[j4], 4),
        f"S{S}_std": round(std_by_s[jS], 4),
    }
    print(f"\n★ 关键: 『像素 std』几乎不随 S 变化（它是后验的固有量，S 只影响估计精度）；"
          f"\n   随 S 按 1/√S 收缩的是『后验均值的标准误』({std_err_by_s[j4]:.4f} → "
          f"{std_err_by_s[-1]:.4f})，以及后验均值的 PSNR。")
    print(f"   像素 std 估计量自身的相对误差约 sqrt(1/(2(S-1)))≈{math.sqrt(1/(2*(S-1))):.1%}（附录 18B），"
          f"与上面的均值标准误不是同一个量。")
    print(f"   所以『S 的影响』应当用估计量的方差来衡量，而不是用 PSNR 一个数字。")


# ========================================================================
# ★ 验收测试
# ========================================================================
print("\n" + "=" * 72)
print("★ 验收测试：本次『不确定度』能不能信？")
print("=" * 72)


def _verdict(value, lo=None, hi=None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "失败"
    ok = True
    if lo is not None and value < lo:
        ok = False
    if hi is not None and value > hi:
        ok = False
    return "OK" if ok else "失败"


checks = []
checks.append(("观测半区 std / σ_d", std_obs / SIGMA_DATA, 0.4, 2.0,
               "≈1 说明数据保真项权重正确(λ=1)"))
checks.append(("后验均值·观测半区 RMS / σ_d", rms_mean_obs / SIGMA_DATA, None, 2.5,
               "均值应当几乎复现观测"))
checks.append(("单样本·观测半区 RMS / σ_d", rms_samp_obs / SIGMA_DATA, None, 3.0,
               "单个样本也应满足观测（>3 说明链没被数据钉住）"))
# v3：分区覆盖率对 S 有限样本零模型打分（阈值随零模型自适应，不能用固定 0.80：
# S=30 主水平零模型≈0.875，而 S=4 时只有 0.323；容差 0.15 覆盖空间相关带来的宽 CI）
checks.append((f"{cl_max:.0%}覆盖率·观测|对零模型|", abs(cov_main['obs'] - null_main), None, 0.15,
               f"实测 {cov_main['obs']:.3f} vs 零模型 {null_main:.3f}（名义 {cl_max:.1%} "
               f"在 S={S} 下达不到，按偏差判定）"))
checks.append((f"{cl_max:.0%}覆盖率·缺失|对零模型|", abs(cov_main['mis'] - null_main), None, 0.15,
               f"实测 {cov_main['mis']:.3f} vs 零模型 {null_main:.3f}；真正考验先验 UQ 的地方"))
checks.append(("覆盖率池化|对零模型偏差|", abs(cov_main['pool'] - null_main), None, 0.06,
               f"零模型 {null_main:.3f}，实测 {cov_main['pool']:.3f}；"
               f"对名义差 {cov_main['pool']-cl_max:+.3f} 主要是有限样本离散"))
checks.append(("偏差场结构 lag1（缺失半区）", dev_lag1, 0.02, None,
               f"白噪声对照 {wn_lag1:+.4f}（结构比 {structure_ratio:.1f}×）；必要非充分条件，"
               f"只排除『均值+白噪声』赝品"))
# ---- v3 新增：专门抓"全体链一致待在错误盆域" ----
checks.append(("缺失/观测 std 比", std_mis / std_obs, None, 4.0,
               "正确后验约1.5~2.5（本次 v3 实测 2.34，先验把缺失像素约束得很紧）；v2 失败样本 14.8"))
checks.append((f"PIT 秩均值·缺失（理论{S/2:.0f}）", abs(pit_mis - S / 2), None, 3.0,
               f"实测 {pit_mis:.2f}；>3 表示样本整体偏亮/偏暗，v2 失败样本 18.8"))
if mean_over_map_rms is not None:
    checks.append(("缺失RMS比(均值/MAP)", mean_over_map_rms, None, 2.0,
                   f"后验均值不能比 MAP 差出数量级；v2 失败样本 12.8（{rms_mean_mis:.3f}"
                   f" vs MAP {map_rms_mis:.3f}）"))

print(f"{'指标':<30s} {'实测':>10s} {'可接受范围':>16s} {'结论':>6s}  说明")
print("-" * 100)
n_fail = 0
for name, val, lo, hi, note in checks:
    v = _verdict(val, lo, hi)
    if v == "失败":
        n_fail += 1
    rng = "[%s, %s]" % ("−∞" if lo is None else lo, "+∞" if hi is None else hi)
    print(f"{name:<30s} {val:>10.4f} {rng:>16s} {v:>6s}  {note}")

if not IS_POSTERIOR:
    verdict_txt = ("不可信 —— 采样器没有跑起来（使用了『伪逆+噪声』兜底），"
                   "下面的 std/区间都只是扰动幅度的反映。")
elif n_fail == 0:
    # checks 只验收主置信水平；校准表还可能把其他档判为『严重失配』。
    # 结论必须显式披露这种局部失配，不能笼统写『区间校准全部通过』。
    n_severe = sum(1 for sev in grades.values() if sev == 2)
    n_dev = sum(1 for sev in grades.values() if sev == 1)
    if n_severe > 0:
        verdict_txt = (
            "主水平通过 —— 数据保真、主置信水平 "
            f"{cl_max:.2%} 的区间校准（对 S 有限样本零模型）、结构相关性、"
            "MAP 点估计对比与 PIT 秩检验全部通过；"
            f"但校准表显示 {n_severe} 个置信水平档判『严重失配』、{n_dev} 档『存在偏差』，"
            f"该局部失配未纳入本 {len(checks)} 项 checks，解读 UQ 结论时需注意。"
        )
    else:
        verdict_txt = ("通过 —— 数据保真、区间校准（对 S 有限样本零模型）、结构相关性、"
                       "MAP 点估计对比与 PIT 秩检验全部通过，可用于 UQ 结论。")
else:
    verdict_txt = (f"存疑 —— 有 {n_fail} 项未通过，按以下方向定位："
                   f"『缺失RMS比(均值/MAP)』>2 或『缺失/观测 std 比』>4 或 PIT 偏移，"
                   f"说明链没有到达正确后验盆域（v2 的零填充初始化就是这种失败），"
                   f"应确认 init_mode='map_warmstart' 并加大 map_steps_per_level；"
                   f"若仅 lag1 接近白噪声对照值，说明链内混合不足，"
                   f"可增加 steps_each_sigma（当前 {CFG['steps_each_sigma']} 步/级，可试 2 倍）；"
                   f"若『观测半区 std/σ_d』偏大，检查 lambda_data 是否为 1。")

print(f"\n【总体结论】{verdict_txt}")


# ========================================================================
# 保存结果
# ========================================================================
ula_params = {k: CFG[k] for k in ("sigma_schedule", "steps_each_sigma", "lambda_data",
                                  "step_size_coeff_annealed", "ula_noise_scale",
                                  "ula_projection")}
ula_params["sigma_max"] = CFG["sigma_max"]
ula_params["sigma_min"] = CFG["sigma_min"]
ula_params["total_steps_per_sample"] = CFG["n_sigma_levels"] * CFG["steps_each_sigma"]
ula_params["stability_ratio_eps_lambda_over_sigma_data2"] = round(_stab, 3)
ula_params["init_mode"] = CFG["init_mode"]
ula_params["data_fidelity"] = CFG["data_fidelity"]
ula_params["map_levels"] = CFG["map_levels"]
ula_params["map_steps_per_level"] = CFG["map_steps_per_level"]

uq_results = {
    "采样方法": ula_method,
    "是否为后验采样": bool(IS_POSTERIOR),
    "样本数": S,
    "图像尺寸": IMG_SIZE,
    "观测噪声sigma_data": SIGMA_DATA,
    "掩码保留比例": round(keep_ratio, 4),
    "掩码hash": FINGERPRINT["mask_hash"],
    "ULA参数配置": ula_params,

    "零填充基线PSNR": round(psnr_zf, 2),
    "观测噪声SNR_dB": round(psnr_obs_noise, 2),

    "后验均值PSNR_整体": round(psnr(x_true, posterior_mean), 2),
    "后验均值PSNR_观测半区": round(psnr(xt[mask_el[0]], pm[mask_el[0]]), 2),
    "后验均值PSNR_缺失半区": round(psnr(xt[~mask_el[0]], pm[~mask_el[0]]), 2),
    "单样本PSNR范围": [round(min(sample_psnrs), 2), round(max(sample_psnrs), 2)],
    "单样本PSNR标准差": round(float(np.std(sample_psnrs)), 4),
    "缺失半区占后验均值平方误差比例": round(frac_mis, 4),

    "std_观测半区": round(std_obs, 4),
    "std_缺失半区": round(std_mis, 4),
    "std_整体": round(float(posterior_std.mean()), 4),
    "std观测比sigma_data": round(std_obs / SIGMA_DATA, 3),
    "std缺失比观测": round(std_mis / std_obs, 3),

    "偏差场lag1_缺失半区": round(dev_lag1, 4),
    "偏差场lag1_全图": round(dev_lag1_full, 4),
    "白噪声对照lag1_缺失半区": round(wn_lag1, 4),
    "偏差场结构比": round(structure_ratio, 3),
    "ESS粗估_仅参照": round(ess, 2),
    "链间相关_中心化均值": round(rho, 4),
    "链间相关_中心化基线": round(rho_centered, 4),
    "链间相关_说明": "中心化偏差场的相关系数被代数地固定在 -1/(S-1)，只能排除塌缩，不能证明链独立",

    # v3：MAP 基线（DPIR 式热启动近似，非完整 DPIR）与后验均值的点估计对比
    "MAP基线PSNR_整体": round(map_psnr, 2) if map_psnr is not None else None,
    "MAP基线PSNR_观测半区": round(map_psnr_obs, 2) if map_psnr_obs is not None else None,
    "MAP基线PSNR_缺失半区": round(map_psnr_mis, 2) if map_psnr_mis is not None else None,
    "后验均值缺失RMS": round(rms_mean_mis, 4),
    "MAP缺失RMS": round(map_rms_mis, 4) if map_rms_mis is not None else None,
    "缺失RMS比_后验均值除MAP": round(mean_over_map_rms, 3) if mean_over_map_rms is not None else None,

    # v3：PIT 真值秩（正确后验 → 均值 S/2）
    "PIT秩均值_观测半区": round(pit_obs, 3),
    "PIT秩均值_缺失半区": round(pit_mis, 3),
    "PIT秩理论均值": round(pit_theo, 2),

    "主置信水平": round(cl_max, 4),
    "全部置信水平": [round(c, 4) for c in confidence_levels],
    f"主置信水平覆盖率_池化": round(cov_main['pool'], 4),
    f"主置信水平覆盖率_观测半区": round(cov_main['obs'], 4),
    f"主置信水平覆盖率_缺失半区": round(cov_main['mis'], 4),
    # v3：校准同时报告有限样本零模型期望与两种偏差
    "主置信水平零模型期望覆盖率": round(null_main, 4),
    "主水平覆盖率偏差_对名义": round(cov_main['pool'] - cl_max, 4),
    "主水平覆盖率偏差_对零模型": round(cov_main['pool'] - null_main, 4),
    "校准状态": grade_main,
    "各置信水平覆盖率": {str(round(cl, 4)): {k: round(v, 4) for k, v in coverages[cl].items()}
                          for cl in confidence_levels},
    "各置信水平零模型期望": {str(round(cl, 4)): round(null_covs[float(cl)], 4)
                             for cl in confidence_levels},

    "验收测试未通过项数": n_fail,
    "验收结论": verdict_txt,
    "Step0自检通过": bool(selftest["passed"]),
    "Step0_std比_观测": round(selftest["ratio_obs"], 3),
    "Step0_std比_缺失": round(selftest["ratio_mis"], 3),
    "Step0_覆盖率": round(selftest["cov"], 4),

    "总采样耗时_秒": round(float(np.sum(sample_times)), 1),
    "平均每样本耗时_秒": round(float(np.mean(sample_times)), 2),
    **s4_comparison,
}
with open(os.path.join(SAVE_DIR, 'uq_results.json'), 'w', encoding='utf-8') as f:
    json.dump(uq_results, f, ensure_ascii=False, indent=2)
print("\n数值结果已保存: uq_results.json")


# ========================================================================
# 作图
# ========================================================================
print("\n[绘图] 生成图表…")
try:
    # ---- 图 1: 不确定性地图（v3：6 列，第一行加入 MAP 基线面板）----
    fig = plt.figure(figsize=(18, 10.4))
    gs = fig.add_gridspec(3, 6, hspace=0.32, wspace=0.12)
    if MAP_X is not None:
        map_panel = MAP_X[0].clamp(0, 1)
        map_title = f"MAP 基线（DPIR，同 DRUNet）\n缺失 {map_psnr_mis:.2f} dB"
    else:
        map_panel = x_zf[0]
        map_title = "MAP 基线\n（init_mode=zerofill，未做预热）"
    panels = [
        (xt, "真值 x", None),
        (x_zf[0], f"观测（零填充，保留{keep_ratio:.0%}像素）\n零填充基线 {psnr_zf:.2f} dB", None),
        (map_panel, map_title, None),
        (pm, f"后验均值\n整体 {psnr(x_true, posterior_mean):.2f} dB"
             f" / 缺失 {psnr(xt[~mask_el[0]], pm[~mask_el[0]]):.2f} dB", None),
        (ps.mean(0), f"后验标准差\n观测 {std_obs:.4f} / 缺失 {std_mis:.4f}", None),
        (ci_width.mean(0), f"{cl_max:.0%} 经验分位数区间宽度\n均值 {float(ci_width.mean()):.4f}", None),
    ]
    for i, (img, title, _) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
            ax.imshow(np.clip(arr, 0, 1))
        else:
            im = ax.imshow(arr, cmap='magma')
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # 行2（★ 承载核心教学目标"从点估计到分布"）：并排多个后验样本。
    # 观测半区各样本几乎一致（被数据钉住），缺失半区在正确盆域内给出不同填法，
    # 直观呈现"后验是一组不同而合理的猜测，而非单点估计"。
    n_show = min(5, S)
    # 从全部 S 个样本中等间隔取 n_show 个，代表整个样本集合（而非"前几个"）
    _show_idx = np.linspace(0, S - 1, n_show).round().astype(int)
    for _c, _k in enumerate(_show_idx):
        ax = fig.add_subplot(gs[1, _c])
        _smp = SAMPLES[int(_k)].clamp(0, 1).detach().cpu().numpy()
        ax.imshow(np.transpose(_smp, (1, 2, 0)))
        ax.set_title(f"后验样本 #{int(_k) + 1}\nPSNR {sample_psnrs[int(_k)]:.2f} dB", fontsize=9)
        ax.axis('off')
    ax = fig.add_subplot(gs[1, 5]); ax.axis('off')

    # 第三行：分区误差 + 结构诊断
    ax = fig.add_subplot(gs[2, 0])
    err_map = (pm - xt).abs().mean(0).detach().cpu().numpy()
    im = ax.imshow(err_map, cmap='inferno'); plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"|后验均值 - 真值|\n缺失半区占误差 {frac_mis:.1%}", fontsize=9); ax.axis('off')

    ax = fig.add_subplot(gs[2, 1])
    s0 = (SAMPLES[0] - posterior_mean)[0].mean(0).detach().cpu().numpy()
    s0 = np.where(_w_mis[0].detach().cpu().numpy() > 0.5, s0, 0.0)
    im = ax.imshow(s0, cmap='coolwarm', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"单样本偏差（样本1-后验均值，仅缺失半区）\nlag1(缺失)={dev_lag1:+.3f}", fontsize=9); ax.axis('off')

    ax = fig.add_subplot(gs[2, 2])
    _bmap = (posterior_mean - x_true)[0].mean(0).detach().cpu().numpy()
    _bmap = np.where(_w_mis[0].detach().cpu().numpy() > 0.5, _bmap, 0.0)
    im = ax.imshow(_bmap, cmap='coolwarm', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("重建偏差（后验均值-真值，仅缺失半区）", fontsize=9); ax.axis('off')

    ax = fig.add_subplot(gs[2, 3:])
    for lbl, val, col in [("样本偏差(缺失)", dev_lag1, "#2b7bba"),
                          ("白噪声对照(缺失)", wn_lag1, "#c0392b"),
                          ("重建偏差(缺失)", bias_lag1, "#27ae60")]:
        ax.bar(lbl, val, color=col)
    ax.axhline(0, color='k', lw=0.8)
    _thr = max(0.02, 3 * max(wn_lag1, 0))
    ax.axhline(_thr, color='gray', ls='--', lw=1,
               label=f"通过阈值（max(0.02, 3×白噪声对照)；必要非充分）")
    ax.set_ylabel("lag-1 空间自相关"); ax.grid(alpha=0.3, axis='y')
    # 重建偏差柱（~0.35）远高于另外两根，会压住右下角图例；
    # 因此留出上方空间并把图例放到左上空白区。
    ax.set_ylim(min(-0.05, dev_lag1 - 0.05), max(_thr, bias_lag1, dev_lag1) * 1.45)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title("★ 结构诊断：缺失半区偏差是图像状(lag1>0)还是白噪声？", fontsize=10)
    plt.suptitle(f"实验18.5-1 Step 1–3: MAP 基线、后验均值、样本与不确定性地图   "
                 f"[{ula_method}, {CFG['init_mode']}/{CFG['data_fidelity']}, S={S}, "
                 f"{CFG['n_sigma_levels']}×{CFG['steps_each_sigma']}步]", fontsize=12)
    out1 = os.path.join(SAVE_DIR, "step1_3_uncertainty_maps.png")
    plt.savefig(out1, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f"  {out1}")

    # ---- 图 2: 校准 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    cls = list(confidence_levels)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label="完美校准（名义值对角线）")
    ax.plot(cls, [null_covs[float(c)] for c in cls], 's--', color='#8e44ad',
            lw=1.2, ms=6, label=f"S={S} 有限样本零模型（评级基准）")
    for key, lbl, col in [('pool', '池化', '#7f8c8d'), ('obs', '观测半区', '#2b7bba'),
                          ('mis', '缺失半区', '#e67e22')]:
        ax.plot(cls, [coverages[c][key] for c in cls], 'o-', label=lbl, color=col)
    ax.set_xlabel("名义覆盖率"); ax.set_ylabel("实测覆盖率")
    ax.set_title("Step 4-a 校准曲线（实测应贴近紫色零模型，而非对角线）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    xpos = np.arange(len(cls))
    ax.bar(xpos - 0.3, [coverages[c]['pool'] for c in cls], width=0.2, label='池化', color='#7f8c8d')
    ax.bar(xpos - 0.1, [coverages[c]['obs'] for c in cls], width=0.2, label='观测半区', color='#2b7bba')
    ax.bar(xpos + 0.1, [coverages[c]['mis'] for c in cls], width=0.2, label='缺失半区', color='#e67e22')
    ax.plot(xpos, cls, 'k*--', ms=9, label='名义值')
    ax.plot(xpos, [null_covs[float(c)] for c in cls], 'D', color='#8e44ad', ms=7,
            label=f'零模型期望 S={S}')
    ax.set_xticks(xpos); ax.set_xticklabels([f"{c:.3f}" for c in cls])
    ax.set_xlabel("名义覆盖率"); ax.set_ylabel("实测覆盖率")
    ax.set_title("Step 4-b 分区覆盖率"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out2 = os.path.join(SAVE_DIR, "step4_calibration.png")
    plt.savefig(out2, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f"  {out2}")

    # ---- 图 3: 样本数影响 ----
    if psnr_by_s is not None and len(n_list) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        ax = axes[0]
        ax.plot(n_list, psnr_by_s, 'o-', color='#2b7bba')
        ax.set_xscale('log', base=2); ax.set_xlabel("样本数 S"); ax.set_ylabel("后验均值 PSNR (dB)")
        ax.set_title("Step 5-a 均值估计随 S 收敛"); ax.grid(alpha=0.3)
        ax = axes[1]
        ax.loglog(n_list, std_err_by_s, 'o-', color='#e67e22', label="后验均值的标准误")
        ax.loglog(n_list, [std_err_by_s[0] * math.sqrt(n_list[0] / n) for n in n_list], 'k--',
                  lw=1, label=r"$\propto 1/\sqrt{S}$")
        ax.set_xscale('log', base=2); ax.set_xlabel("样本数 S"); ax.set_ylabel("后验均值的标准误")
        ax.set_title("Step 5-b 不确定度估计精度 ∝ 1/√S\n(像素 std 本身几乎不随 S 变)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
        plt.tight_layout()
        out3 = os.path.join(SAVE_DIR, "step5_sample_size_effect.png")
        plt.savefig(out3, dpi=110, bbox_inches='tight'); plt.close(fig)
        print(f"  {out3}")
except Exception as e:
    print(f"[警告] 绘图失败: {type(e).__name__}: {e}")
    traceback.print_exc()


# ========================================================================
# 总结
# ========================================================================
_s5_txt = (f"{std_err_by_s[0]:.4f} → {std_err_by_s[-1]:.4f}（S={n_list[0]}→{n_list[-1]}）"
           if (psnr_by_s is not None and len(n_list) > 1) else "（样本数不足，未评估）")

print("\n" + "=" * 72)
print("实验18.5-1 总结")
print("=" * 72)
_map_line = (f"   MAP 基线缺失 {map_psnr_mis:.2f} dB → 后验均值缺失 "
             f"{psnr(xt[~mask_el[0]], pm[~mask_el[0]]):.2f} dB（RMS 比 {mean_over_map_rms:.2f}，应≈1）") \
    if map_psnr_mis is not None else "   （未做 MAP 预热：init_mode=zerofill）"
print(f"""
1. 从点估计到分布 ✓
   零填充基线 {psnr_zf:.2f} dB → 后验均值 整体 {psnr(x_true, posterior_mean):.2f} dB
   （观测半区 {psnr(xt[mask_el[0]], pm[mask_el[0]]):.2f} dB / 缺失半区 {psnr(xt[~mask_el[0]], pm[~mask_el[0]]):.2f} dB）
{_map_line}
   单样本 {min(sample_psnrs):.2f}~{max(sample_psnrs):.2f} dB —— 后验是分布而不是一个点。

2. 后验采样 ✓
   退火 PnP-ULA: ∇log p_σ(x) = (D(x,σ)-x)/σ²,  ε = {CFG['step_size_coeff_annealed']}·σ²
   λ_data = {CFG['lambda_data']}（贝叶斯权重，必须为 1）
   σ ∈ [{CFG['sigma_min']:.4f}, {CFG['sigma_max']:.4f}]（DRUNet 训练范围）
   {CFG['n_sigma_levels']} 级 × {CFG['steps_each_sigma']} 步 = {CFG['n_sigma_levels'] * CFG['steps_each_sigma']} 步/样本
   v3 修复：初始化={CFG['init_mode']}（DPIR 预热 {CFG['map_levels']}×{CFG['map_steps_per_level']} 次去噪），
            数据项={CFG['data_fidelity']}（MYULA 近端步）——修复零填充初始化的粗尺度输送失败。

3. 不确定性量化 ✓
   后验均值 / 像素级标准差 / {cl_max:.0%} 经验分位数区间
   观测半区 std = {std_obs:.4f} ({std_obs / SIGMA_DATA:.2f}×σ_d)   缺失半区 std = {std_mis:.4f}
   缺失/观测 std 比 = {std_mis / std_obs:.2f}；PIT 秩均值 观测 {pit_obs:.1f} / 缺失 {pit_mis:.1f}（理论 {pit_theo:.0f}）

4. 校准检验 ✓
   {cl_max:.0%} 名义区间: 池化 {cov_main['pool']:.1%} / 观测 {cov_main['obs']:.1%} / 缺失 {cov_main['mis']:.1%}
   S={S} 零模型期望 {null_main:.1%}（不是名义 {cl_max:.0%}！）→ 对零模型 {grade_main}
   像素相关使不确定性远大于二项估计；评级必须对有限样本零模型、分区看。

5. 样本数影响 ✓
   像素 std 基本不随 S 变化（它是后验固有量）；
   随 S 按 1/√S 收缩的是后验均值的标准误 {_s5_txt}。

★ 验收：{verdict_txt}

所有文件已保存至: {SAVE_DIR}
""")
print("=" * 72)
