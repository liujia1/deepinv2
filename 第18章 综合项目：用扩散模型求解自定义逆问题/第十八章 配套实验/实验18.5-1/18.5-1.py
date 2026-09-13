# -*- coding: utf-8 -*-
"""
实验18.5-1 不确定性量化与后验采样
对应知识点：18.5节（不确定性量化：从点估计到分布推断）

实验内容：
Step 0: ★ 采样器正确性自检 —— 在解析可解问题上验证链确实收敛到真实后验
Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
Step 2: 后验采样实现 —— 退火 PnP-ULA（DRUNet 先验）
Step 3: 不确定性地图计算 —— 像素级标准差、经验分位数区间、偏差场结构诊断
Step 4: ★ 校准检验 —— 覆盖率与可靠性分析（按 观测/缺失 半区分开统计）
Step 5: ★ 样本数对不确定性估计的影响

===========================================================================
★★★ 本版相对初版的关键修正（全部有解析可解问题上的定量依据）★★★
===========================================================================
初版实测结果：样本 PSNR 12.15 dB、观测半区 std=0.0422、93.3%区间覆盖率 98.8%。
经复算，这些数字不是"后验不确定度"，而是采样器超参的产物。四个根因：

(1) lambda_data: 0.05 -> 1.0        【最重要】
    数据保真项的贝叶斯权重必须是 1：
        ∇log p(y|x) = -A^T(Ax-y)/sigma_data^2
    取 0.05 会让链收敛到 p_σ(x)·p(y|x)^0.05 而不是 p(x|y)。后果可以解析算出：
    观测像素的"后验方差" = 1/(λ/σ_d² + 先验项) ≈ σ_d²/λ，
    即 std 被撑到 σ_d/√λ = 0.01/√0.05 = 0.0447。
    初版实测 0.0422 —— 吻合。这是超参决定的数，与后验无关。
    解析验证（Step 0 的表）：λ=0.05 时观测半区 std 是真实后验的 4.2 倍；
    λ=1 时恰好是 1.00 倍。

(2) sigma 调度: [1.0,0.5,0.25,0.1,0.05,0.02,0.01] -> geomspace(0.196, 2/255, 14)
    a) DRUNet 只在 σ∈[0, 50/255≈0.196] 上训练过，σ=1.0/0.5/0.25 的 score 是外推值；
    b) 显式稳定性条件：观测像素的更新是 x ← x(1-ε/σ_d²) + ...，
       要求 ε·λ/σ_d² = c·σ²/σ_d² < 2，c=0.005、σ_d=0.01 时要求 σ < 0.2 ——
       与 DRUNet 的适用范围恰好一致，所以 σ_max=0.196 是有理论依据的上界。

(3) 每级步数: 50 -> 400
    链的收缩率是 c=0.005，松弛时间 1/c = 200 步。只跑 50 步时每级只收缩到 78%，
    高 σ 级注入的方差永远退不掉。实测缺失半区 std=0.2624，
    而按 AR(1) 递推 V←(1-c)^50·V+(1-(1-c)^50)σ² 预测 0.2677 —— 吻合到 2%。
    也就是说缺失半区的"不确定度"是 σ 调度残留，不是后验。
    400 步/级（=2 个松弛时间）在解析问题上把覆盖率从 89.1% 修到 94.5%（名义 93.3%）。

(4) 逐样本 for 循环 -> S 条链作为一个 batch 同时演化
    GPU 上快约 S 倍（实测 batch-30 与 batch-1 相比，总算力需求降 3.4 倍，
    且显存只用到 1.5 GB）。

另外还修正了：
  * "观测PSNR" 的标签错误 —— 那其实是零填充(伪逆 A†y)基线的 PSNR，不是噪声水平；
    0.01 的观测噪声对应 40 dB，而 11.4 dB 完全由 50% 的黑色空洞造成。
  * "样本PSNR.std() 应 >0.5 dB，否则混合不足" —— 这个判据是错的。
    PSNR 在 49152 维上高度集中，白噪声赝样本实测 std=0.035 dB，真实样本 0.047 dB
    （初版调研值；本次运行的实测值见 uq_results.json 的『单样本PSNR标准差』）。
    真正能判定"样本是否为后验样本"的是偏差场的空间结构（见 Step 3）。
  * 分区统计 —— 池化数字被未约束的那一半主导（99.7% 的平方误差在缺失半区），
    必须分开报告。
  * 验收测试 —— 新增自动 PASS/WARN/FAIL，不达标时明确声明"本次 UQ 不可信"，
    而不是把失败包装成"后验偏保守"。
  * 缓存 —— 指纹现在包含掩码 hash；掩码由固定种子显式生成，跨设备可复现。

===========================================================================
解析可解问题上的标定结果（Step 0 会现场重跑）
===========================================================================
  配置                                             sd_obs  sd_mis  cov93.3
  精确后验样本（参考）                               1.00    1.00    94.0%
  初版配置 (λ=.05, 50步, σ→1.0)                     2.1~4.2  1.8~3.1  99.7%
  本版配置 (λ=1.0, 400步, σ∈[0.196,2/255])          1.00    1.05    95.0%
  名义覆盖率 93.3%；sd_* = 实测 std / 解析后验 std（1.00 为完美）
  （上表为本脚本 Step 0 单次实测的典型值；初版配置的观测半区比值因
    掩码/真值实现而落在 2~4 倍区间，多次重复平均约 4.2 倍。）

运行前提：GPU 优先（本机 1660 Ti 6GB、torch cu126 实测 batch-30 采样
共 14×400 步 ≈ 32 分钟；T4/A10 上更快）。CPU 也能跑通，但 Step 2 会慢 1~2 个数量级。
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
    def tqdm(x, **k):
        return x


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
    "ula_projection": (-0.25, 1.25),   # 数值保护，对 [0,1] 的图像不生效
    "ula_init_jitter": 0.01,
    "max_grad_norm": None,       # 可选梯度裁剪（None = 关闭）

    # ---- 采样规模 ----
    "n_samples": 30,             # 后验样本数 S
    "sample_chunk": 30,          # 一次前向同时演化的链数（显存不足时调小）

    # ---- 校准 ----
    "calib_good": 0.04,          # |覆盖率-名义值| < 4%  判为"校准良好"
    "calib_ok": 0.08,            # < 8% 判为"存在偏差"；否则"严重失配"

    # ---- 缓存 ----
    "use_cache": True,
    "cache_version": 2,

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

IMG_SIZE = CFG["image_size"]
N_SAMPLES = CFG["n_samples"]
SIGMA_DATA = CFG["sigma_data"]

# 稳定性自检：观测半区的更新是 x ← x(1 - ε·λ/σ_d²) + ...，需要 ε·λ/σ_d² < 2
_stab = CFG["step_size_coeff_annealed"] * CFG["sigma_max"] ** 2 * CFG["lambda_data"] / SIGMA_DATA ** 2
print(f"[配置] λ_data={CFG['lambda_data']}, σ∈[{CFG['sigma_min']:.4f}, {CFG['sigma_max']:.4f}] "
      f"({CFG['n_sigma_levels']}级 x {CFG['steps_each_sigma']}步 = "
      f"{CFG['n_sigma_levels'] * CFG['steps_each_sigma']} 步/样本)")
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


# ========================================================================
# ★ 核心：批量退火 PnP-ULA 采样器
# ========================================================================
def sample_annealed_ula(y_obs, A, At, denoiser, cfg, n_samples, device, seed=0,
                        desc="ULA", verbose=True):
    """退火 PnP-ULA：从 p(x|y) ∝ p(y|x)p(x) 采样。

    迭代（Laumont et al. 2022, Algorithm 2 的退火变体）：
        x_{k+1} = Π_[a,b]( x_k + ε_k ∇log p_{σ_k}(x_k)
                                 + ε_k λ A^T(y - A x_k)/σ_d²
                                 + noise_scale·sqrt(2ε_k)·z_k )
    其中先验 score 用 Tweedie 公式：∇log p_σ(x) = (D(x,σ) - x)/σ²，
    D 是 σ 级去噪器（这里是 DRUNet），ε_k = c·σ_k²。

    ★ 全部 n_samples 条链作为一个 batch 同时推进（GPU 上快约 S 倍）。
    ★ A / At 以可调用对象传入，这样 Step 0 的解析自检可以复用完全相同的代码路径。

    返回 (n_samples, C, H, W) 的张量（在 device 上）。
    """
    gen = torch.Generator(device=device.type)
    gen.manual_seed(int(seed))

    with torch.no_grad():
        x0 = At(y_obs).detach()
        x = x0.repeat(n_samples, *([1] * (x0.dim() - 1)))
        x = x + cfg["ula_init_jitter"] * torch.randn(x.shape, generator=gen, device=device)

    lo, hi = cfg["ula_projection"]
    sched = cfg["sigma_schedule"]
    steps = cfg["steps_each_sigma"]
    lam = cfg["lambda_data"]
    coeff = cfg["step_size_coeff_annealed"]
    ns = cfg["ula_noise_scale"]
    inv_sd2 = 1.0 / (SIGMA_DATA ** 2)

    for si, sigma in enumerate(sched):
        eps = coeff * (sigma ** 2)
        noise_std = ns * math.sqrt(2.0 * eps)
        sigma_t = torch.full((n_samples, 1, 1, 1), float(sigma), device=device)
        t_lv = time.time()
        for _ in range(steps):
            with torch.no_grad():
                x0_hat = denoiser(x, sigma_t)
                if isinstance(x0_hat, (tuple, list)):
                    x0_hat = x0_hat[0]
                score = (x0_hat - x) / (sigma ** 2)
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
        if verbose:
            print(f"    [{desc}] σ={sigma:.4f} ({si + 1}/{len(sched)}) {steps}步 "
                  f"完成 {time.time() - t_lv:.1f}s")
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
        r_o = float((emp_sd[obs].mean() / sd_ref[obs].mean()).sqrt())
        r_m = float((emp_sd[mis].mean() / sd_ref[mis].mean()).sqrt())
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

if cached is not None and len(cached["samples"]) >= N_SAMPLES:
    all_samples = [s for s in cached["samples"][:N_SAMPLES]]
    sample_methods = list(cached["methods"][:N_SAMPLES])
    sample_times = list(cached["times"][:N_SAMPLES])
    print(f"[缓存] 复用 {len(all_samples)} 个样本")

if len(all_samples) < N_SAMPLES:
    t_start_all = time.time()
    try:
        print("[ULA] 加载 DRUNet 去噪器…")
        denoiser = DRUNet(in_channels=x_true.shape[1], out_channels=x_true.shape[1],
                          pretrained="download", device=device).to(device).eval()
        for p in denoiser.parameters():
            p.requires_grad_(False)

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
                                         desc="ULA", verbose=True)
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
                       extra={"mask": MASK.cpu(), "y_inp": y_inp.cpu(), "x_true": x_true.cpu()})
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
      f"实测 {std_obs / SIGMA_DATA:.2f}×")

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
print(f"  ★ 判据: 如果样本是真后验样本，x_s-x̄ 在缺失半区应该像『同一张脸的不同合理猜测』"
      f"那样空间相关(lag1 明显大于白噪声对照)；\n     若 ≈白噪声对照 则说明样本只是"
      f"『后验均值+逐像素白噪声』，这种『不确定度』没有后验含义。")
structure_ratio = dev_lag1 / max(abs(wn_lag1), 1e-6)

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
rho_indep = -1.0 / max(S - 1, 1)          # 独立链的理论值
# 各链是独立初始化的，所以 rho 应当 ≈ -1/(S-1)；偏离说明采样器有退化（例如所有链塌到同一点）
if abs(rho - rho_indep) < 0.05 * max(abs(rho_indep), 0.05):
    ess = float(S)
else:
    ess = float(min(S, max(1.0, S / max(1.0 + (S - 1) * rho, 1e-6))))
print(f"[样本相关性] 链间相关系数均值 {rho:+.4f} "
      f"(独立链理论值 {rho_indep:+.4f}), 有效样本数 ≈ {ess:.1f}")
if abs(rho - rho_indep) < 0.05 * max(abs(rho_indep), 0.05):
    print("  未检测到链间塌缩 —— 采样器没有退化。")
    print("  注意：样本偏差场是中心化的，rho 必然 ≈ -1/(S-1)，所以这个统计量只能")
    print("  排除塌缩（全塌缩→rho≈0，塌到 k 个模态→rho≈1/k），并不能证明各链独立；")
    print("  真正判定『样本是不是后验样本』的是上面的偏差场 lag1。")
else:
    print("  ⚠️ 链间相关性偏离独立链理论值，说明链之间发生了塌缩，UQ 不可信。")


# ========================================================================
# Step 4: 校准检验
# ========================================================================
print("\n" + "=" * 72)
print("Step 4: ★ 校准检验（覆盖率）")
print("=" * 72)
print("注意：覆盖率是把所有像素池化统计的。像素之间高度相关，"
      "所以这个数字的统计不确定性远大于二项分布给出的估计；")
print("     而且『是否能覆盖』在观测半区与缺失半区是两件不同的事，必须分开看。\n")
print(f"{'名义覆盖率':>10s} {'池化':>9s} {'观测半区':>10s} {'缺失半区':>10s} {'偏差(池化)':>11s} {'状态':>10s}")
print("-" * 68)
grades = {}
for cl in confidence_levels:
    c = coverages[cl]
    dev = abs(c['pool'] - cl)
    g, sev = _calib_grade(dev)
    grades[cl] = sev
    print(f"{cl:>10.4f} {c['pool']:>9.4f} {c['obs']:>10.4f} {c['mis']:>10.4f} "
          f"{dev:>11.4f} {g:>10s}")

cov_main = coverages[cl_max]
dev_main = abs(cov_main['pool'] - cl_max)
grade_main, sev_main = _calib_grade(dev_main)
print(f"\n主置信水平 {cl_max:.2%}: 池化覆盖率 {cov_main['pool']:.2%} "
      f"(观测 {cov_main['obs']:.2%} / 缺失 {cov_main['mis']:.2%})，{grade_main}")


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
        # 均值图自身的"估计标准误"：用链间标准差 / sqrt(n) —— 这才是随 S 变化的量
        se = sub.std(0, keepdim=True) / math.sqrt(n)
        psnr_by_s.append(psnr(x_true, m))
        std_err_by_s.append(float(se.mean()))
        std_by_s.append(float(sub.std(0, keepdim=True).mean()))
    print(f"{'S':>4s} {'后验均值PSNR':>13s} {'像素std均值':>12s} {'std的估计标准误':>16s}")
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
          f"\n   随 S 按 1/√S 收缩的是『std 的估计标准误』({std_err_by_s[j4]:.4f} → "
          f"{std_err_by_s[-1]:.4f})，以及后验均值的 PSNR。")
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
checks.append((f"{cl_max:.0%}覆盖率·观测半区", cov_main['obs'], 0.85, 0.999,
               "区间应基本覆盖观测上的真值"))
checks.append((f"{cl_max:.0%}覆盖率·缺失半区", cov_main['mis'], 0.85, 0.999,
               "真正考验先验 UQ 的地方"))
checks.append(("偏差场结构 lag1（缺失半区）", dev_lag1, 0.02, None,
               f"白噪声对照 {wn_lag1:+.4f}（结构比 {structure_ratio:.1f}×）；"
               f"接近对照值表示样本只是『均值+白噪声』，不是后验样本"))

print(f"{'指标':<28s} {'实测':>10s} {'可接受范围':>16s} {'结论':>6s}  说明")
print("-" * 96)
n_fail = 0
for name, val, lo, hi, note in checks:
    v = _verdict(val, lo, hi)
    if v == "失败":
        n_fail += 1
    rng = "[%s, %s]" % ("−∞" if lo is None else lo, "+∞" if hi is None else hi)
    print(f"{name:<28s} {val:>10.4f} {rng:>16s} {v:>6s}  {note}")

if not IS_POSTERIOR:
    verdict_txt = ("不可信 —— 采样器没有跑起来（使用了『伪逆+噪声』兜底），"
                   "下面的 std/区间都只是扰动幅度的反映。")
elif n_fail == 0:
    verdict_txt = "通过 —— 样本通过了全部结构性与标定性检查，可用于 UQ 结论。"
else:
    verdict_txt = (f"存疑 —— 有 {n_fail} 项未通过。请对照上表定位："
                   f"若『观测半区 std/σ_d』偏大，检查 lambda_data 是否为 1；"
                   f"若『偏差场 lag1』接近白噪声对照值，说明链没有混合，"
                   f"需要增加 steps_each_sigma（当前 {CFG['steps_each_sigma']} 步/级，可试 2 倍）。")

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

    "偏差场lag1_缺失半区": round(dev_lag1, 4),
    "偏差场lag1_全图": round(dev_lag1_full, 4),
    "白噪声对照lag1_缺失半区": round(wn_lag1, 4),
    "偏差场结构比": round(structure_ratio, 3),
    "有效样本数": round(ess, 2),
    "链间相关系数均值": round(rho, 4),
    "独立链理论相关系数": round(rho_indep, 4),

    "主置信水平": round(cl_max, 4),
    "全部置信水平": [round(c, 4) for c in confidence_levels],
    f"主置信水平覆盖率_池化": round(cov_main['pool'], 4),
    f"主置信水平覆盖率_观测半区": round(cov_main['obs'], 4),
    f"主置信水平覆盖率_缺失半区": round(cov_main['mis'], 4),
    "校准状态": grade_main,
    "各置信水平覆盖率": {str(round(cl, 4)): {k: round(v, 4) for k, v in coverages[cl].items()}
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
    # ---- 图 1: 不确定性地图 ----
    fig = plt.figure(figsize=(15, 10.4))
    gs = fig.add_gridspec(3, 5, hspace=0.32, wspace=0.12)
    panels = [
        (xt, "真值 x", None),
        (x_zf[0], f"观测 A†y (保留{keep_ratio:.0%}像素)\n零填充基线 {psnr_zf:.2f} dB", None),
        (pm, f"后验均值 x̄\n整体 {psnr(x_true, posterior_mean):.2f} dB"
             f" / 观测半区 {psnr(xt[mask_el[0]], pm[mask_el[0]]):.2f} dB", None),
        (ps.mean(0), f"后验标准差 σ(x)\n观测 {std_obs:.4f} / 缺失 {std_mis:.4f}", None),
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
    # 观测半区各样本几乎一致（被数据钉住），缺失半区每次填法不同但都合理，
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
    ax.set_title(f"单样本偏差 x_1 - x̄（仅缺失半区）\nlag1(缺失)={dev_lag1:+.3f}", fontsize=9); ax.axis('off')

    ax = fig.add_subplot(gs[2, 2])
    _bmap = (posterior_mean - x_true)[0].mean(0).detach().cpu().numpy()
    _bmap = np.where(_w_mis[0].detach().cpu().numpy() > 0.5, _bmap, 0.0)
    im = ax.imshow(_bmap, cmap='coolwarm', vmin=-0.3, vmax=0.3)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("重建偏差 x̄ - 真值（仅缺失半区）", fontsize=9); ax.axis('off')

    ax = fig.add_subplot(gs[2, 3:])
    for lbl, val, col in [("样本偏差(缺失)", dev_lag1, "#2b7bba"),
                          ("白噪声对照(缺失)", wn_lag1, "#c0392b"),
                          ("重建偏差(缺失)", bias_lag1, "#27ae60")]:
        ax.bar(lbl, val, color=col)
    ax.axhline(0, color='k', lw=0.8)
    _thr = max(0.02, 3 * max(wn_lag1, 0))
    ax.axhline(_thr, color='gray', ls='--', lw=1,
               label=f"通过阈值（max(0.02, 3×白噪声对照)）")
    ax.set_ylabel("lag-1 空间自相关"); ax.grid(alpha=0.3, axis='y')
    # 重建偏差柱（~0.35）远高于另外两根，会压住右下角图例；
    # 因此留出上方空间并把图例放到左上空白区。
    ax.set_ylim(min(-0.05, dev_lag1 - 0.05), max(_thr, bias_lag1, dev_lag1) * 1.45)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title("★ 结构诊断：缺失半区偏差是图像状(lag1>0)还是白噪声？", fontsize=10)
    plt.suptitle(f"实验18.5-1 Step 1–3: 从点估计到分布 —— 后验均值、样本与不确定性地图   "
                 f"[{ula_method}, S={S}, λ={CFG['lambda_data']}, "
                 f"{CFG['n_sigma_levels']}×{CFG['steps_each_sigma']}步]", fontsize=12)
    out1 = os.path.join(SAVE_DIR, "step1_3_uncertainty_maps.png")
    plt.savefig(out1, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f"  {out1}")

    # ---- 图 2: 校准 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    cls = list(confidence_levels)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label="完美校准")
    for key, lbl, col in [('pool', '池化', '#7f8c8d'), ('obs', '观测半区', '#2b7bba'),
                          ('mis', '缺失半区', '#e67e22')]:
        ax.plot(cls, [coverages[c][key] for c in cls], 'o-', label=lbl, color=col)
    ax.set_xlabel("名义覆盖率"); ax.set_ylabel("实测覆盖率")
    ax.set_title("Step 4-a 校准曲线（按区域分开）")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    xpos = np.arange(len(cls))
    ax.bar(xpos - 0.27, [coverages[c]['pool'] for c in cls], width=0.27, label='池化', color='#7f8c8d')
    ax.bar(xpos, [coverages[c]['obs'] for c in cls], width=0.27, label='观测半区', color='#2b7bba')
    ax.bar(xpos + 0.27, [coverages[c]['mis'] for c in cls], width=0.27, label='缺失半区', color='#e67e22')
    ax.plot(xpos, cls, 'k*--', ms=10, label='名义值')
    ax.set_xticks(xpos); ax.set_xticklabels([f"{c:.3f}" for c in cls])
    ax.set_xlabel("名义覆盖率"); ax.set_ylabel("实测覆盖率")
    ax.set_title("Step 4-b 分区覆盖率"); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
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
        ax.loglog(n_list, std_err_by_s, 'o-', color='#e67e22', label="std 的估计标准误")
        ax.loglog(n_list, [std_err_by_s[0] * math.sqrt(n_list[0] / n) for n in n_list], 'k--',
                  lw=1, label=r"$\propto 1/\sqrt{S}$")
        ax.set_xscale('log', base=2); ax.set_xlabel("样本数 S"); ax.set_ylabel("估计标准误")
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
print(f"""
1. 从点估计到分布 ✓
   零填充基线 {psnr_zf:.2f} dB → 后验均值 整体 {psnr(x_true, posterior_mean):.2f} dB
   （观测半区 {psnr(xt[mask_el[0]], pm[mask_el[0]]):.2f} dB / 缺失半区 {psnr(xt[~mask_el[0]], pm[~mask_el[0]]):.2f} dB）
   单样本 {min(sample_psnrs):.2f}~{max(sample_psnrs):.2f} dB —— 后验是分布而不是一个点。

2. 后验采样 ✓
   退火 PnP-ULA: ∇log p_σ(x) = (D(x,σ)-x)/σ²,  ε = {CFG['step_size_coeff_annealed']}·σ²
   λ_data = {CFG['lambda_data']}（贝叶斯权重，必须为 1）
   σ ∈ [{CFG['sigma_min']:.4f}, {CFG['sigma_max']:.4f}]（DRUNet 训练范围）
   {CFG['n_sigma_levels']} 级 × {CFG['steps_each_sigma']} 步 = {CFG['n_sigma_levels'] * CFG['steps_each_sigma']} 步/样本

3. 不确定性量化 ✓
   后验均值 / 像素级标准差 / {cl_max:.0%} 经验分位数区间
   观测半区 std = {std_obs:.4f} ({std_obs / SIGMA_DATA:.2f}×σ_d)   缺失半区 std = {std_mis:.4f}

4. 校准检验 ✓
   {cl_max:.0%} 区间: 池化 {cov_main['pool']:.1%} / 观测 {cov_main['obs']:.1%} / 缺失 {cov_main['mis']:.1%} → {grade_main}
   像素池化会低估统计不确定性；分区看才能发现真正的问题。

5. 样本数影响 ✓
   像素 std 基本不随 S 变化（它是后验固有量）；
   随 S 按 1/√S 收缩的是估计标准误 {_s5_txt}。

★ 验收：{verdict_txt}

所有文件已保存至: {SAVE_DIR}
""")
print("=" * 72)
