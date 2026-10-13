"""
实验4.5-1 半二次最小化 vs GLM（Gibbs采样）：MAP vs MMSE
对应章节：4.5 Gibbs采样——利用条件结构的采样
知识点：Gibbs交替采样；半二次最小化→GLM；高斯尺度混合表示；
        GIG分布采样；Gibbs=带噪声的交替最小化；MAP vs MMSE

修改说明：
  从原参考实验4.5.py迁移，适配配套实验风格（matplotlib.use('Agg')、
  中文字体模块、Colab支持、LaTeX格式符号）。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
# 注：scipy.stats.geninvgauss 是精确的 GIG 采样器，但对 2·N·N 个分量抽样耗时极高
#     (单帧 ~18s)，无法支撑教学所需的上千次 Gibbs 迭代；本实验改用等价的
#     精确目标采样器 sample_gig_half（对数空间 MH）。复核中已用 5e4 样本/参数点
#     独立验证其 E[z]、E[1/z] 与理论矩及 scipy 精确采样一致（接受率 0.71~0.84）。
import os
import sys
import warnings
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验4.5-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════
# 0. 精度/设备与缓存配置
# ══════════════════════════════════════════════════════════
# ★ 必须用 float64：TV-GLM 链中近零梯度处 1/z 重尾（99.9分位~6e6），条件高斯矩阵
#   条件数约 1e9。float32 机器精度(~6e-8)下内层 CG 在极端帧会数值崩溃
#   （实测终止残差可达 1e11、采样步移误差达真实位移的~70%并强欠冲，静默污染样本）；
#   float64 下 CG 稳定收敛。配合 float64，本实验规模(1e4维稠密小核)在 CPU 上反而
#   更快更稳——消费级 GPU(Colab T4 等)的 FP64 吞吐极低，故本实验固定 CPU。
DTYPE = torch.float64
device = torch.device('cpu')
if torch.cuda.is_available():
    print("Device: 检测到 GPU，但本实验固定使用 CPU + float64（GPU 的 FP64 极慢且本问题为小核 CG）")
else:
    print("Device: cpu (float64)")

# v3：v2 仍用 float32（CG 在极端帧数值崩溃）；v3 改 float64 后数值不同代，
# 改名使旧缓存失效、强制重算。
CACHE_FILE = os.path.join(SAVE_DIR, 'experiment_4.5-1_cache_v3.pth')
USE_CACHE = os.path.exists(CACHE_FILE)

if USE_CACHE:
    print(f"\n发现缓存文件: {CACHE_FILE}")
    print("加载缓存结果（跳过计算）...")
    cache = torch.load(CACHE_FILE, map_location=device, weights_only=False)
    u = cache['u']
    f = cache['f']
    f_ = cache['f_'].to(device)
    u_hq = cache['u_hq'].to(device)
    u_avg_ = cache['u_avg_'].to(device)
    u_var_ = cache['u_var_'].to(device)
    sigma = cache['sigma']
    N = cache['N']
    chain_diag = cache.get('chain_diag', {})
    print("缓存加载完成")
else:
    print("\n未找到缓存文件，开始计算...")

# ══════════════════════════════════════════════════════════
# 1. 辅助函数
# ══════════════════════════════════════════════════════════

def CG(x, Ax, sigma, lamb, z, b, maxit=1000, tol=1e-9, verbose=0, warn=True):
    """共轭梯度求解 Ax=b（须在 float64 下调用，float32 对病态矩阵不稳定）。

    返回 (x, n_iter, converged, mean_r2)：
      n_iter    —— 实际迭代次数；
      converged —— 是否在 maxit 内达到 mean(r^2) < tol；
      mean_r2   —— 终止时的 mean(r^2)。
    warn=False 时不发警告（用于 Gibbs 主循环：调用方自行统计触顶率并在循环末
    汇总报告，避免逐帧刷屏；信息不丢弃，计数与残差写入 JSON）。
    若迭代中出现非有限值（NaN/Inf，通常意味着精度不足），立即中止并标记，
    绝不静默返回垃圾解。
    """
    r = b - Ax(sigma, lamb, z, x)
    p = r
    rsold = torch.sum(r**2)
    R = torch.mean(r**2)
    n_iter = 0
    converged = bool(R.item() < tol)
    diverged = False
    for it in range(maxit):
        n_iter = it + 1
        Ap = Ax(sigma, lamb, z, p)
        alpha = rsold / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r**2)
        if not torch.isfinite(rsnew):
            diverged = True
            break
        beta = rsnew / rsold
        rsold = rsnew.clone()
        p = r + beta * p
        R = torch.mean(r**2)
        if R < tol:
            converged = True
            break
    if verbose > 0:
        print("CG: it = ", n_iter, ", mse = ", float(R))
    if diverged:
        msg = (f"CG 出现 NaN/Inf 后中止（第 {n_iter} 步）：当前精度不足以处理该条件数，"
               f"必须使用 float64；请勿在此状态下使用返回值。")
        if warn:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return x, n_iter, False, float('nan')
    if not converged and warn:
        warnings.warn(
            f"CG 触顶 maxit={maxit} 仍未收敛: mean(r^2)={float(R):.3e} >= tol={tol:.0e}；"
            f"内层求解不精确会压缩后验方差，应增大 maxit 或引入预条件。",
            RuntimeWarning, stacklevel=2)
    return x, n_iter, converged, float(R)

def ess_from_chain(seq):
    """标量链的简单 ESS 估计（正滞后截断）：tau=1+2*sum_{k: rho(k)>0} rho(k)，ESS=n/tau。

    只用于诊断混合速度（对图像均值/能量等标量统计量）；
    逐像素 ESS 计算量过大，不在教学实验中给出。
    """
    x = np.asarray(seq, dtype=np.float64)
    n = x.size
    x = x - x.mean()
    v = np.dot(x, x)
    if v == 0.0 or n < 3:
        return float(n), 1.0
    acf = np.correlate(x, x, mode='full')[n - 1:] / v
    s = 0.0
    for k in range(1, min(n - 1, n // 2)):
        if acf[k] <= 0:
            break
        s += acf[k]
    tau = 1.0 + 2.0 * s
    return float(n / tau), float(tau)

def Du(u):
    M, N = u.shape
    Du_out = torch.zeros((2, M, N), dtype=u.dtype, device=u.device)
    Du_out[0, :, :-1] += u[:, 1:] - u[:, :-1]
    Du_out[1, :-1, :] += u[1:, :] - u[:-1, :]
    return Du_out

def DTp(p):
    C, M, N = p.shape
    DTp_out = torch.zeros((M, N), dtype=p.dtype, device=p.device)
    DTp_out[:, 1:] += p[0, :, :-1]
    DTp_out[:, :-1] -= p[0, :, :-1]
    DTp_out[1:, :] += p[1, :-1, :]
    DTp_out[:-1, :] -= p[1, :-1, :]
    return DTp_out

def sample_gig_half(t, lam, steps=150):
    """从 GIG(1/2, chi=t^2, psi=lam^2) 精确目标采样（对数空间随机游走 MH）。
    目标密度 p(z) ∝ z^{-1/2} exp(-(t^2/z + lam^2 z)/2)，即 TV 高斯尺度混合
    表示中辅助变量 z 的条件分布（讲义 4.5 节 L62/L65）。
    用 np.random 全局流，保持 np.random.seed(42) 下的可复现性。"""
    chi = np.maximum(t * t, 1e-30)
    psi = lam * lam
    logz = np.log(np.maximum(chi / psi, 1e-12))
    step = 0.5
    def lpz(lz):
        return 0.5 * lz - 0.5 * (chi * np.exp(-lz) + psi * np.exp(lz))
    cur = lpz(logz)
    for _ in range(steps):
        prop = logz + np.random.normal(0, step, size=logz.shape)
        pc = lpz(prop)
        acc = np.log(np.random.random(size=logz.shape)) < (pc - cur)
        logz = np.where(acc, prop, logz)
        cur = np.where(acc, pc, cur)
    return np.exp(logz)

# ★ 自检：验证Du/DTp伴随算子关系与边界处理(避免discrete boundary条件改错)
# 伴随关系: 对任意u, p, 应有 <Du(u), p> = <u, DTp(p)>  (内积守恒)
# 这是GLM中交替采样的数学基础(也是18.4-1标准化TV散度修复的依据)
# 检验规则: |<Du(u), p> - <u, DTp(p)>| / <u,u> < 1e-10 (浮点精度内)
print("[算子自检] Du/DTp伴随关系验证(实际运行, 非手算):")
np.random.seed(42)
_torch_device = device  # 始终CPU
_M, _N = 20, 30
# 用 float64：伴随误差阈值 1e-10，float32 求和舍入达不到
_u_test = torch.randn(_M, _N, device=_torch_device, dtype=torch.float64)
_p_test = torch.randn(2, _M, _N, device=_torch_device, dtype=torch.float64)
_Du_u = Du(_u_test)
_DTp_p = DTp(_p_test)
# 内积: <Du(u), p> = sum(Du_u * p), <u, DTp(p)> = sum(u * DTp_p)
_ip_left = (_Du_u * _p_test).sum().item()
_ip_right = (_u_test * _DTp_p).sum().item()
_diff = abs(_ip_left - _ip_right)
_norm = (_u_test * _u_test).sum().item()
print(f"  <Du(u), p>={_ip_left:.4f}, <u, DTp(p)>={_ip_right:.4f}, 差={_diff:.2e}, 相对={_diff/max(_norm,1e-10):.2e}")
print(f"  伴随关系成立: {'OK' if _diff / max(_norm, 1e-10) < 1e-10 else 'FAIL'}")
# 额外验证: DTp(Du(u)) = -离散Laplace算子·u (内部点), 边界处因fill=0有数值差异
_Lap = DTp(Du(_u_test))
_interior_err = (_Lap[1:-1, 1:-1] - ((_u_test[:-2, 1:-1] + _u_test[2:, 1:-1] + _u_test[1:-1, :-2] + _u_test[1:-1, 2:]) - 4*_u_test[1:-1, 1:-1])).abs().max().item()
print(f"  DTp(Du(u))与5点Laplace差(内部点): {_interior_err:.2e} (应≈0, 边界fill方式决定边缘值)")

# ══════════════════════════════════════════════════════════
# 2. 合成图像与加噪
# ══════════════════════════════════════════════════════════
if not USE_CACHE:
    N = 100
    x_grid = np.linspace(-1, 1, N)
    y_grid = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x_grid, y_grid)

    u1 = 0 * X + 0.5
    u2 = X + Y
    u2 -= u2.min()
    u2 /= u2.max()

    u3 = np.clip(1 - ((1 - X)**2 + (1 - Y)**2), a_min=0, a_max=None)
    u3 -= u3.min()
    u3 /= u3.max()

    m1 = 0 * X
    m1[N//8:-N//8, N//8:-N//8] = 1
    u = u1.copy()
    # u2 已归一化到[0,1]，不再乘1.25：真值各分量都在[0,1]内，
    # 与可视化 vmin/vmax=1 及 PSNR peak=1.0 的假设严格一致（旧版乘1.25
    # 会使方块内真值超过1，被imshow静默裁切）。
    u[m1 == 1] = u2[m1 == 1]

    m2 = 0 * X
    m2[(X - 1)**2 + (Y - 1)**2 <= 1] = 1
    u[m2 == 1] = u3[m2 == 1]

    sigma = 0.1
    f = u + np.random.randn(N, N) * sigma
    f_ = torch.from_numpy(f).to(DTYPE).to(device)  # float64，CPU

# ══════════════════════════════════════════════════════════
# 3. 步骤1：MAP——半二次最小化
# ══════════════════════════════════════════════════════════
# |t| = min_{z>0} { t^2/(2z) + z/2 }
# 交替更新：z = clip(|Du|, eps)，然后用CG求解u

print("=" * 60)
print("实验4.5-1 半二次最小化 vs GLM（Gibbs采样）")
print("=" * 60)

maxiter_hq = 50
lamb_hq = 10.0
# HQ路径z=clip(|Du|,1e-6)在平坦区同样病态（λ/z~1e7）；float64 下达到 mean(r^2)<1e-9
# 实测中位约4000次、最大约5000次，故上限设6000（float32 下因精度不足无法达到该阈值）。
CG_MAXIT_HQ = 6000

Au_hq = lambda sigma, lamb, z, u: lamb_hq * DTp(Du(u) / z) + u / sigma**2
rhs_hq = f_ / sigma**2

print(f"\n[步骤1] 半二次最小化 MAP (lambda={lamb_hq}, {maxiter_hq} iters)")
if not USE_CACHE:
    hq_cg_iters, hq_cg_capped, hq_cg_resid_max = [], 0, 0.0
    u_ = torch.from_numpy(f).to(DTYPE).to(device)  # float64，CPU
    for it in range(maxiter_hq):
        diff_u = torch.abs(Du(u_))
        z_ = np.clip(diff_u.cpu().numpy(), a_min=1e-6, a_max=None)  # 移回CPU进行clip
        z_ = torch.from_numpy(z_).to(DTYPE).to(device)  # float64，CPU
        u_, _it, _ok, _R = CG(u_, Au_hq, sigma, lamb_hq, z_, rhs_hq,
                              maxit=CG_MAXIT_HQ, verbose=0)
        hq_cg_iters.append(_it)
        hq_cg_capped += int(not _ok)
        hq_cg_resid_max = max(hq_cg_resid_max, _R)
        # 使用更新后的u_计算TV和Dat，确保一致性
        diff_u = torch.abs(Du(u_))
        TV = lamb_hq * diff_u.sum()
        Dat = torch.sum((u_ - f_)**2 / (2 * sigma**2))
        if (it + 1) % 10 == 0:
            print(f"  iter {it+1}: TV + Dat = {(TV + Dat).item():.6f}")

    u_hq = u_
    print(f"  半二次最小化完成 (CG触顶 {hq_cg_capped}/{maxiter_hq} 次, "
          f"中位迭代 {int(np.median(hq_cg_iters))}, 最大残差 {hq_cg_resid_max:.2e})")
else:
    print("  从缓存加载 MAP 结果")

# ══════════════════════════════════════════════════════════
# 4. 步骤2：MMSE——GLM Gibbs采样
# ══════════════════════════════════════════════════════════
# Gibbs交替采样：
#   z_j ~ GIG(lambda^2, |Du_j|^2, 1/2)  (辅助变量)
#   x ~ N(mu(z), Sigma(z))              (条件高斯，CG+扰动)
#
# 注意（复核修订）：样本数影响MMSE与后验方差的MC精度。复核中实测这条Gibbs链
# 混合其实很快（标量积分自相关时间 tau≈1.2~2.1，逐像素ESS中位约1800/2100），
# "10^4维=混合慢"并不成立；但后验std图的*空间结构*需要约800以上样本才稳定，
# 故采用 1000 次迭代（含 60 次 burn-in，940 个保留样本），并报告标量链ESS供核查。
# 两个关键点（float64 复核修订）：
# ①近零梯度处 1/z 重尾使条件高斯矩阵条件数高达~1e9，内层CG必须 float64——float32
#   下极端帧CG数值崩溃（残差可达1e11、采样步移欠冲~70%，静默污染样本）；
# ②达到 mean(r^2)<1e-9 中位需约5000次CG，旧版 maxit=1000 每次触顶、静默压缩后验
#   方差，故上限提至8000；float64 下仍有约3%~6%的极端重尾帧触顶，但这些帧残差已
#   很小（<1e-7量级）、解质量可靠，触顶率统计写入 JSON，不逐帧报警。

maxiter_glm = 1000
lamb_glm = 10.0  # 与lamb_hq保持一致，确保MAP与MMSE对比的公平性
# 实测达到 mean(r^2)<1e-9 中位需约5000次CG、尾部可达约7000次；8000覆盖绝大多数情形。
CG_MAXIT_GLM = 8000

# Au_glm: GLM条件高斯的系统矩阵
# 注意：lamb参数未在函数体中使用，因为λ已通过GIG采样隐含在z_中
# 保留lamb参数是为了与Au_hq保持一致的函数签名
Au_glm = lambda sigma, lamb, z, u: DTp(Du(u) / z) + u / sigma**2

u_sum_ = 0
u_sqr_ = 0
n_burn_in = 60  # burn-in期，丢弃前60次迭代的样本（复核实测60次足够）

print(f"\n[步骤2] GLM Gibbs采样 MMSE (lambda={lamb_glm}, {maxiter_glm} iters, burn-in={n_burn_in})")
if not USE_CACHE:
    # 内层CG收敛统计 + 标量诊断链（图像均值/后验能量），用于ESS与混合速度报告
    cg_iters, cg_capped, cg_resid_max = [], 0, 0.0
    chain_mean, chain_energy = [], []
    u_ = torch.from_numpy(f).to(DTYPE).to(device)  # float64，CPU
    for it in range(maxiter_glm):
        diff_u = torch.abs(Du(u_))
        tmp = diff_u.cpu().numpy().ravel()  # 移回CPU进行GIG采样
        # 注：Du的最后一列/最后一行是零Neumann边界的"幻影边"（恒为0），此处置
        # chi=1e-30 仅为数值占位；这些z分量随后被 DTp 的 [:-1] 索引丢弃，不进入
        # 右端项与后验，故其GIG不混合不影响结果（但改动边界索引时需重新审视）。
        tmp[tmp == 0] = 1e-30

        # z_j ~ GIG(1/2, chi=|Du_j|^2, psi=lambda^2)（精确目标采样，见 sample_gig_half；
        # 其 E[z]、E[1/z] 已在复核中与 scipy.stats.geninvgauss 精确采样核对一致）
        z_ = torch.from_numpy(sample_gig_half(tmp, lamb_glm)).to(DTYPE).reshape(2, N, N).to(device)
        eta = torch.normal(0, 1, size=(3, N, N), dtype=DTYPE, device=device)

        rhs = DTp(eta[:2] / torch.sqrt(z_)) + eta[2] / sigma + f_ / sigma**2
        # warn=False：极端1/z帧（约3%~6%）可能在8000次内未到1e-9，逐帧警告会刷屏；
        # 这里自行统计，循环末汇总。若出现NaN/Inf（float64下不应发生）则立即报错。
        u_, _cg_it, _cg_ok, _cg_R = CG(u_, Au_glm, sigma, lamb_glm, z_, rhs,
                                       maxit=CG_MAXIT_GLM, verbose=0, warn=False)
        if not np.isfinite(_cg_R) or not torch.isfinite(u_).all():
            raise RuntimeError(
                f"Gibbs 第{it+1}帧内层CG数值发散(mean_r2={_cg_R})；请确认全程使用float64。")
        cg_iters.append(_cg_it)
        cg_capped += int(not _cg_ok)
        cg_resid_max = max(cg_resid_max, _cg_R)

        # 使用更新后的u_计算TV和Dat，确保一致性
        diff_u = torch.abs(Du(u_))
        TV = lamb_glm * diff_u.sum()
        Dat = torch.sum((u_ - f_)**2 / (2 * sigma**2))

        # 仅在burn-in期后累加样本与诊断标量
        if it >= n_burn_in:
            u_sum_ += u_
            u_sqr_ += u_**2
            chain_mean.append(float(u_.mean()))
            chain_energy.append(float(TV + Dat))

        if (it + 1) % 50 == 0:
            print(f"  iter {it+1}: TV + Dat = {(TV + Dat).item():.6f}")

    n_samples = maxiter_glm - n_burn_in
    u_avg_ = u_sum_ / n_samples
    u_var_ = u_sqr_ / n_samples - u_avg_**2
    ess_mean, iat_mean = ess_from_chain(chain_mean)
    ess_energy, iat_energy = ess_from_chain(chain_energy)
    chain_diag = {
        'n_samples_after_burnin': int(n_samples),
        'ess_scalar_mean': ess_mean, 'iat_scalar_mean': iat_mean,
        'ess_scalar_energy': ess_energy, 'iat_scalar_energy': iat_energy,
        'cg_maxit': int(CG_MAXIT_GLM),
        'cg_capped_count': int(cg_capped),
        'cg_capped_fraction': float(cg_capped / maxiter_glm),
        'cg_iter_median': float(np.median(cg_iters)),
        'cg_iter_max': int(np.max(cg_iters)),
        'cg_resid_max': float(cg_resid_max),
        'hq_cg_capped_count': int(hq_cg_capped),
        'hq_cg_resid_max': float(hq_cg_resid_max),
    }
    print(f"  GLM Gibbs采样完成 (保留样本: {n_samples}, 总迭代={maxiter_glm}, burn-in={n_burn_in})")
    print(f"  标量ESS: 图像均值 {ess_mean:.0f} (tau={iat_mean:.2f}), "
          f"后验能量 {ess_energy:.0f} (tau={iat_energy:.2f})")
    print(f"  内层CG: 触顶 {cg_capped}/{maxiter_glm} ({cg_capped/maxiter_glm:.1%}), "
          f"中位迭代 {int(np.median(cg_iters))}, 最大 {int(np.max(cg_iters))}, "
          f"最大残余 mean(r^2) {cg_resid_max:.2e}")
else:
    print("  从缓存加载 MMSE 结果")
    # chain_diag 已随缓存加载（旧缓存缺该键时为 {}，JSON 中相应指标显示为 null）

# ══════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0][0].imshow(u, cmap='gray', vmin=0, vmax=1)
axes[0][0].set_title(r'Ground truth $u$')
axes[0][0].axis('off')

axes[0][1].imshow(f, cmap='gray', vmin=0, vmax=1)
axes[0][1].set_title(rf'Noisy image ($\sigma={sigma}$)')
axes[0][1].axis('off')

axes[0][2].axis('off')

axes[1][0].imshow(u_hq.cpu().numpy(), cmap='gray', vmin=0, vmax=1)  # 移回CPU可视化
axes[1][0].set_title(r'MAP (Half-quadratic minimization)')
axes[1][0].axis('off')

axes[1][1].imshow(u_avg_.cpu().numpy(), cmap='gray', vmin=0, vmax=1)  # 移回CPU可视化
axes[1][1].set_title(r'MMSE (GLM Gibbs posterior mean)')
axes[1][1].axis('off')

std_map = torch.sqrt(torch.clamp(u_var_, min=0)).cpu().numpy()  # 移回CPU可视化
# 固定色标 0~σ：避免 auto-scale 使整图饱和为红色而掩盖边缘/平坦的差异
im_std = axes[1][2].imshow(std_map, cmap='hot', vmin=0, vmax=sigma)
axes[1][2].set_title(r'Posterior std (uncertainty)')
axes[1][2].axis('off')
cbar = fig.colorbar(im_std, ax=axes[1][2], fraction=0.046, pad=0.04)
cbar.set_label(r'$\sigma_{post}$ (0 to $\sigma$)', fontsize=9)

fig.suptitle(r'Experiment 4.5-1: Half-quadratic minimization vs GLM (Gibbs sampling)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'hq_vs_glm_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 6. 保存缓存（用于下次快速加载）
# ══════════════════════════════════════════════════════════
if not USE_CACHE:
    cache = {
        'u': u,
        'f': f,
        'f_': f_.cpu(),  # 保存时移回CPU，确保跨设备兼容
        'u_hq': u_hq.cpu(),
        'u_avg_': u_avg_.cpu(),
        'u_var_': u_var_.cpu(),
        'sigma': sigma,
        'N': N,
        'chain_diag': chain_diag,  # ESS/积分自相关时间/CG收敛统计
    }
    torch.save(cache, CACHE_FILE)
    print(f"\n缓存已保存至: {CACHE_FILE}")

# ══════════════════════════════════════════════════════════
# 7. 输出结论
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print(r"1. 半二次最小化: |t| = min_{z>0} {t^2/(2z) + z/2}")
print(r"   确定性: z = clip(|Du|, eps), 然后CG求解u -> MAP")
print(r"2. GLM Gibbs: z ~ GIG(nu=1/2, chi=|Du|^2, psi=lambda^2), 然后CG+扰动采样u -> MMSE")
print(r"3. Gibbs = 半二次最小化 + 噪声")
print(r"   确定版: z = clip(|Du|, eps)  vs  Gibbs版: z ~ GIG(1/2; |Du|^2, lambda^2)")
print(r"   确定版: CG精确右端          vs  Gibbs版: CG求解带高斯扰动的右端")
print(r"4. MAP收敛到众数(分段常数), MMSE收敛到分布(后验均值+不确定性)")
print(r"5. 统一转化律: 优化算法 + 噪声 = 采样算法")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")

# ══════════════════════════════════════════════════════════
# 保存数值结果到JSON文件
# ══════════════════════════════════════════════════════════
import json

def _to_native(obj):
    """递归将numpy/torch类型转换为Python原生类型，便于JSON序列化"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, RuntimeError, TypeError):
            return obj
    return obj

# ====== 结果量化指标（用于 JSON 归档与图文一致性核查）======
def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

def _psnr(x, ref, peak=1.0):
    mse = np.mean((_np(x).astype(np.float64) - ref.astype(np.float64))**2)
    return float(10 * np.log10(peak**2 / mse))

_gt = np.asarray(u, dtype=np.float64)
_std = np.sqrt(np.clip(_np(u_var_), 0, None))
_gx = np.abs(np.diff(_gt, axis=1)); _gy = np.abs(np.diff(_gt, axis=0))
_edge = np.zeros_like(_gt); _edge[:, :-1] += _gx; _edge[:-1, :] += _gy
_edge = _edge / max(_edge.max(), 1e-12)
_flat = _edge < 0.05; _strong = _edge > 0.3
_corr_std_edge = float(np.corrcoef(_std.ravel(), _edge.ravel())[0, 1])

metrics = {
    'psnr_noisy_vs_gt': _psnr(f, _gt),
    'psnr_map_hq_vs_gt': _psnr(u_hq, _gt),
    'psnr_mmse_glm_vs_gt': _psnr(u_avg_, _gt),
    'posterior_std_mean': float(_std.mean()),
    'posterior_std_flat_mean': float(_std[_flat].mean()),
    'posterior_std_edge_mean': float(_std[_strong].mean()),
    'corr_std_vs_edge': _corr_std_edge,
}
# 链诊断：名义保留样本数 + 标量链实测ESS/积分自相关时间 + 内层CG收敛统计。
# 注意区分：n_samples_after_burnin 是名义样本数，ess_* 才是去相关后的有效样本量。
metrics.update(chain_diag)
print("\n[量化指标]")
for k, v in metrics.items():
    print(f"  {k}: {v if v is None else (f'{v:.4f}' if isinstance(v, float) else v)}")

results_summary = {
    'experiment': '4.5-1',
    'title': '半二次最小化 vs GLM（Gibbs采样）',
    'setup': {
        'image_size': [int(N), int(N)],
        'sigma': float(sigma),
        'lamb_hq': float(lamb_hq),
        'lamb_glm': float(lamb_glm),
        'maxiter_hq': maxiter_hq,
        'maxiter_glm': maxiter_glm,
        'n_burn_in': n_burn_in,
        'cg_maxit_hq': int(CG_MAXIT_HQ),
        'cg_maxit_glm': int(CG_MAXIT_GLM),
        'cg_tol_mean_r2': 1e-9,
    },
    'metrics': metrics,
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as _fj:
    json.dump(_to_native(results_summary), _fj, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
