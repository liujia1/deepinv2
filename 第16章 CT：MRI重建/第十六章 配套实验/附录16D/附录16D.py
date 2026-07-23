# -*- coding: utf-8 -*-
"""
附录16D  PET 发射断层重建——射线追踪正演 + Poisson 统计 + OSEM / OSL
对应章节：16.1 节（Radon / 线积分正演）的 PET 推广；本书附录：发射断层

实验目的：
    PET 与 CT 共享同一套线积分正演模型 A（投影 = 沿射线的线积分），
    但 PET 的探测计数服从泊松统计，且需从投影数据里“反演”活度分布。
    本实验动手实现：
      (1) 用 Radon 变换作为 2D 射线追踪正演（与 CT 的 16.1-1 一致）；
      (2) 在投影域叠加泊松噪声（计数越高的 bin 方差越大）；
      (3) OSEM：有序子集期望最大化，即最大似然估计（MLE）；
      (4) OSL：一步延迟 MAP，叠加二次先验抑制噪声。
    对比 OSEM 与 OSL 的重建质量与噪声特性。

素材来源（算法参考，非运行依赖）：
    sirf Python examples/PET/osem_reconstruction.py
        —— OSMAPOSL / OSEM 主流程；
    sirf Python examples/PET/user_osmaposl.py
        —— OSMAPOSL 迭代更新公式
           x <- x * (A_s^T (y_s / (A_s x + b))) / (A_s^T 1)；
    sirf Python examples/PET/osl_reconstruction.py
        —— QuadraticPrior / OSL 先验梯度；
    sirf Python examples/PET/acquisition_model.py
        —— 射线追踪正演 A（这里以 Radon 替代 3D 环形几何，原理一致）。
    本文件为完全自包含的 numpy / skimage 复刻，无需安装 SIRF / STIR。

运行前提：纯 CPU 即可运行，无需 GPU，无需下载数据。
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端（静默模式）
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import sys
import io
import json
import warnings
import logging

# ----- Windows 控制台 UTF-8 输出 -----
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

# ----- 静默 matplotlib 告警 -----
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams['axes.unicode_minus'] = False

# ----- 中文 / Colab 支持 -----
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '附录16D')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

# ----- 随机种子 -----
np.random.seed(42)

print(f"\n{'='*64}")
print(f"附录16D : PET 发射断层重建 —— OSEM / OSL")
print(f"{'='*64}")


# ========== 工具函数 ==========

def _to_native(obj):
    """把 numpy 标量与数组转成 JSON 可序列化对象。"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.complexfloating,)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def metrics(ref, x):
    """计算 ref 与 x 之间的 PSNR / SSIM / NMSE。

    采用动态 data_range = max(ref.max(), x.max())：体模已归一化到 [0,1]，
    但迭代重建可能因噪声/过冲使结果超过 1；固定 data_range=1.0 会低估真实
    动态范围、导致 PSNR 偏保守，故取两者峰值以保证度量口径严格一致。
    """
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dr = float(max(ref.max(), x.max()))
    p = float(psnr(ref, x, data_range=dr))
    s = float(ssim(ref, x, data_range=dr))
    nmse = float(np.mean((ref - x) ** 2) / (np.mean(ref ** 2) + 1e-12))
    return p, s, nmse


# ========== 步骤0 : 发射活度体模 ==========
print("\n>>> 步骤0 : 准备 PET 发射活度体模")

N = 111                                   # 图像尺寸（与 SIRF PET 示例一致）
theta = np.linspace(0, 180, 180, endpoint=False)   # 投影角度

base = resize(shepp_logan_phantom(), (N, N), order=0,
              preserve_range=True, anti_aliasing=False)
x_true = base / base.max()

# 叠加若干“热区”模拟病灶/高摄取（PET 典型对比度更高）
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
for (cy, cx, r, v) in [(40, 40, 8, 1.8), (72, 70, 6, 1.5),
                       (55, 55, 5, 1.3), (30, 80, 5, 1.4)]:
    x_true += v * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
x_true = np.clip(x_true, 0, None)
x_true = x_true / x_true.max()
print(f"    活度体模尺寸 {N}x{N}，含 4 个高摄取热区，已归一化到 [0,1]")


# ========== 步骤1 : 射线追踪正演 + 泊松噪声 ==========
print("\n>>> 步骤1 : 射线追踪正演（Radon）+ 投影域泊松噪声")

# 正演 A 与伴随 A^T（Radon / 无滤波反投影）
def A(x):
    return radon(x, theta=theta)


def AT(sino):
    return iradon(sino, theta=theta, filter_name=None, output_size=N)


proj_true = A(x_true)                     # 真值线积分（活度沿射线求和）
total_counts = 3.0e5                      # 期望总计数级别（PET 典型量级）
scale = total_counts / proj_true.sum()    # 把线积分换算成计数水平
# 理想化假设：重建时用的 background 与仿真真值完全一致（完美背景校正）。
# 真实 PET 中 randoms/scatter 的背景是独立测量估计的，与真值通常有误差；
# 见步骤6 的拓展实验，故意把重建用 background 乘以 0.5 / 2.0（约 -50% / +100%）观察敏感度。
# 关键：背景必须放在“计数”量纲里取 5%，否则占比会随 total_counts 漂移。
#   scale*proj_true 才是计数（与 y 同量纲），因此先换算再取均值；
#   若用 0.05*proj_true.mean()（未乘 scale），则 background/y.mean() 只在
#   scale≈1 时凑巧≈5%，一旦学生改 total_counts 量级，占比会剧烈失实。
background = 0.05 * (scale * proj_true).mean()   # 均匀本底（randoms / scatter 近似），占计数均值 5%
y_rate = scale * proj_true + background
y_counts = np.random.poisson(y_rate).astype(np.float64)   # 含泊松噪声的实测计数

print(f"    总期望计数 ≈ {total_counts:.0f}，叠加 Poisson 噪声")
print(f"    实测计数均值 ≈ {y_counts.mean():.2f}，本底 b ≈ {background:.2f}")

# 显示：体模 + 真值 sinogram + 含噪 sinogram
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(x_true, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('发射活度体模 $x$', fontsize=12)
axes[0].axis('off')
axes[1].imshow(proj_true, cmap='gray')
axes[1].set_title('无噪投影 $Ax$ (Radon)', fontsize=12)
axes[1].axis('off')
axes[2].imshow(y_counts, cmap='gray')
axes[2].set_title('含 Poisson 噪声投影 $y$', fontsize=12)
axes[2].axis('off')
fig.suptitle('PET 正演：射线追踪 + 泊松统计', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤1_正演与泊松噪声.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤1_正演与泊松噪声.png")


# ========== 步骤2 : OSEM 重建（MLE）==========
print("\n>>> 步骤2 : OSEM 重建（参考 SIRF user_osmaposl.py 的 OSMAPOSL 更新公式）")
print("    教学提醒：本实验受限于 skimage.radon 只能整幅计算、无法只算部分角度，")
print("        子集按‘探测器 bin’（第 0 轴，跨全部 180 角度）而非‘投影角度’划分。")
print("        真实 OSEM 的子集是按投影角度分组的——每个角度子集本身已是对全物体")
print("        的一次有代表性观测，可用子集梯度近似全梯度、实现类似 SGD 的加速收敛；")
print("        而按 bin 切分不具备这种‘子集≈全局’的统计代表性，只能演示 EM 迭代结构，")
print("        并不能反映真实 OSEM 相对普通 EM 的加速收敛优势，请勿把它当作标准做法。")


def osem_recon(y_counts, scale, background, num_subsets=10, n_subiter=20,
               prior_beta=0.0, denom_floor_frac=0.5, ratio_clip=3.0):
    """有序子集期望最大化（OSEM），可选 OSL 二次先验。

    每次子集更新（参考 user_osmaposl.py）：
        grad = A_s^T ( y_s / (A_s x + b) )
        sens = A_s^T 1
        denom = sens + prior_grad / num_subsets   （prior_beta>0 时）
        x <- x * grad / denom

    关于 scale：真实正演为 A_full = scale * A，标准更新是
        x <- x * A_full^T(y/(A_full x+b)) / A_full^T(1)
    把 A_full = scale*A 代入后，分子分母的 scale 代数上完全约掉，
        = x * A^T(y/(scale*Ax+b)) / A^T(1)
    因此代码里 grad 用 AT(...)（不含 scale）、denom 用 sens_s（不含 scale）
    是自洽且正确的，并非漏乘 scale。

    关于 OSL 稳定性（实测排查记录，非纯理论提醒）：
    OSL 是 one-step-late 近似，不是严格的凸优化，denom 中的离散拉普拉斯 lap
    在 FOV 圆外（np.roll 造成的循环边界人为不连续处）可能出现很大的负值。
    早期版本用绝对下限 denom=max(denom,1e-6) 兜底——但 sens_s 的正常量级在
    0.1~0.5 左右，1e-6 比这小 5 个数量级，一旦 denom 被夹到该下限、而 grad
    在同一像素并不同样小，grad/denom 就会产生天文数字级的单步增益；由于更新
    是乘法形式（x <- x*step），会在后续子迭代里指数级级联放大。实测 beta=0.5
    时到第 4 次子迭代 x.max() 已达数千，20 次子迭代后发散到 ~1e6，此时打印的
    PSNR/SSIM 会被动态 data_range 一起“冲高”，看起来反而比不发散的结果更好，
    极具误导性。
    现改为两层稳定化（仅在 prior_beta>0 时生效，纯 OSEM(MLE) 不受影响——它的
    denom 恒为 sens_s>0，从未表现出发散，不需要也不应该被限幅），已用真实数据
    验证 beta 扫到 2.0 仍收敛到有限解、不再失控发散；但 std/max 并不随 beta
    单调：OSL 为 one-step-late，热点区 lap<0 会压低 denom、配合单步限幅反而
    放大峰值（实测 β=0.5 时 x.max()≈1.93 高于 β=0 的≈1.83，PSNR 也更高），只有
    beta 足够大（如 2.0）才转为过度平滑；稳定化保证的是收敛到有限解、而非单调平滑：
      (1) denom 下限改为与该像素自身 sens_s 成比例（denom_floor_frac*sens_s），
          而不是脱离物理尺度的绝对常数，避免上述量级失配（此式对 beta=0 恒等，
          max(sens_s,0.5*sens_s)=sens_s，不改变纯 OSEM 行为）；
      (2) 仅当 prior_beta>0 时，单步更新比例 grad/denom 才显式限幅到
          [1/ratio_clip, ratio_clip]。早期实现曾对 beta=0 也无差别限幅，
          实测会让纯 OSEM 结果失真（PSNR 25.44→25.09 dB，约 6.8 万个
          像素-子迭代被限幅命中），现已收窄为仅 OSL 分支生效。
    """
    n_bins = y_counts.shape[0]                    # 探测器 bin 数（≈157），注意不是角度数（180）
    x = np.ones((N, N), dtype=np.float64)         # 初值：均匀活度
    rows_list = [np.arange(s, n_bins, num_subsets) for s in range(num_subsets)]

    for it in tqdm(range(1, n_subiter + 1), desc='  OSEM/OSL', leave=False):
        subset = (it - 1) % num_subsets
        rows = rows_list[subset]
        # 正演（仅取当前子集所用的探测器 bin 行）。
        # 注意：本 sinogram 第 0 轴是探测器 bin（≈157），第 1 轴才是角度（180），
        # 这里的子集是按 bin 索引 rows 划分的（每个 bin 在整轮 10 个子集里恰好用一次）。
        # skimage 的 radon 始终返回“全部 bin × 全部角度”的整幅投影，无法只算子集角度，
        # 因此这里对整幅正演 A(x) 做 bin 行切片；本例 N=111 计算量很小，无需进一步优化。
        full = np.zeros_like(y_counts)
        full[rows] = (scale * A(x))[rows]
        ratio = y_counts[rows] / (full[rows] + background)
        # 伴随：把子集投影填回整幅 sinogram 再反投影（零行不贡献）
        tmp = np.zeros_like(y_counts)
        tmp[rows] = ratio
        grad = AT(tmp)
        # 子集灵敏度图 A_s^T 1（OSEM 每子集用各自灵敏度，保证无偏）
        ones_s = np.zeros_like(y_counts)
        ones_s[rows] = 1.0
        sens_s = np.maximum(AT(ones_s), 1e-6)
        if prior_beta > 0:
            lap = (4 * x - np.roll(x, 1, 0) - np.roll(x, -1, 0)
                   - np.roll(x, 1, 1) - np.roll(x, -1, 1))
            denom = sens_s + (prior_beta * lap) / num_subsets
        else:
            denom = sens_s
        # 相对下限：与 sens_s 自身量级成比例（而非固定绝对小数），
        # 避免 denom 在 FOV 边界附近被夹到远小于 sens_s 正常量级的值。
        # 注：prior_beta=0 时 denom=sens_s，此式恒等于原值（max(sens_s,0.5*sens_s)=sens_s），
        # 不影响纯 OSEM(MLE) 的行为。
        denom = np.maximum(denom, denom_floor_frac * sens_s)
        raw_step = grad / denom
        if prior_beta > 0:
            # 单步更新幅度限幅：仅在叠加先验（OSL）时才需要，因为发散的根源是
            # 先验项 lap 把 denom 拉负、被下限夹住后 grad/denom 产生的失控增益。
            # 纯 OSEM(MLE) 从未表现出这种发散（denom 恒为正的 sens_s），不应对它
            # 施加这层限幅——早期实现曾无差别限幅，实测会让纯 OSEM 结果失真
            # （PSNR 25.44→25.09 dB，约 6.8 万个像素-子迭代被限幅命中），故收窄范围。
            step = np.clip(raw_step, 1.0 / ratio_clip, ratio_clip)
        else:
            step = raw_step
        x = x * step
        x = np.clip(x, 0, None)
    return x


x_osem = osem_recon(y_counts, scale, background,
                    num_subsets=10, n_subiter=20, prior_beta=0.0)
print("    OSEM 完成（num_subsets=10, subiterations=20）")


# ========== 步骤3 : OSL 重建（带二次先验的 MAP）==========
print("\n>>> 步骤3 : OSL 重建（参考 SIRF osl_reconstruction.py 的 QuadraticPrior）")

x_osl = osem_recon(y_counts, scale, background,
                   num_subsets=10, n_subiter=20, prior_beta=0.05)
print("    OSL 完成（叠加二次先验 beta=0.05，抑制噪声但轻微平滑）")
print("    注：OSL 为 one-step-late 近似，不保证目标函数单调下降；")
print("        denom 已改用相对下限 + 单步幅度限幅两层稳定化（见函数注释），")
print("        beta 过大时表现为过度平滑而非失控发散，见步骤5 稳定性实验。")


# ========== 步骤4 : 指标对比与可视化 ==========
print("\n>>> 步骤4 : 计算指标并对比两种重建结果")

p_o, s_o, nmse_o = metrics(x_true, x_osem)
p_l, s_l, nmse_l = metrics(x_true, x_osl)

print(f"    OSEM : PSNR={p_o:6.2f} dB, SSIM={s_o:.4f}, NMSE={nmse_o:.4f}")
print(f"    OSL  : PSNR={p_l:6.2f} dB, SSIM={s_l:.4f}, NMSE={nmse_l:.4f}")
print("    说明：OSEM 为最大似然估计，计数充足时分辨率好但偏 noisy；")
print("          OSL 在分母加入二次先验梯度，抑制斑点噪声、分辨率略降，")
print("          是典型的‘保真度 ↔ 正则’折中，与 16.2 节的正则化思想一致。")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
imgs = [x_true, x_osem, x_osl]
titles = ['真值活度 $x$', 'OSEM (MLE)', 'OSL (MAP, 二次先验)']
for ax, im, ti in zip(axes[0], imgs, titles):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax.set_title(ti, fontsize=12)
    ax.axis('off')
errs = [np.zeros_like(x_true),
        np.abs(x_true - x_osem),
        np.abs(x_true - x_osl)]
etitles = ['', 'OSEM 误差', 'OSL 误差']
for ax, er, ti in zip(axes[1], errs, etitles):
    if ti == '':
        ax.axis('off')
        continue
    ax.imshow(er, cmap='hot')
    ax.set_title(ti, fontsize=12)
    ax.axis('off')
fig.suptitle('PET 重建对比：OSEM (MLE) vs OSL (带先验 MAP)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤4_OSEM与OSL对比.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤4_OSEM与OSL对比.png")


# ========== 步骤5 : OSL 对先验强度 beta 的稳定性（拓展实验）==========
print("\n>>> 步骤5 : OSL 对先验强度 beta 的稳定性（拓展实验）")
print("    说明：OSL 是 one-step-late 近似，不是严格凸优化；")
print("          denom = sens_s + beta*lap/num_subsets 中的离散拉普拉斯 lap")
print("          在 FOV 边界附近可能为负且量级较大。早期版本用绝对下限")
print("          denom=max(denom,1e-6) 兜底，因 1e-6 远小于 sens_s 正常量级")
print("          （约0.1~0.5），beta 稍大就会在个别像素触发指数级级联发散，")
print("          且发散后算出的 PSNR/SSIM 反而因动态 data_range 被推高、极具")
print("          误导性。现改为 denom 相对下限（∝sens_s）+ 单步幅度限幅两层")
print("          稳定化，保证 beta 扫到 2.0 仍收敛到有限解、不再失控发散；")
print("          但需注意：OSL 为 one-step-late，中等 beta 在热点区（lap<0）反而可能放大峰值")
print("          （实测 β=0.5 峰值≈1.93 > β=0 的≈1.83、PSNR 也更高），并非'beta 越大越平滑'，")
print("          只有 beta 足够大（如 2.0）才表现为过度平滑（细节丢失、对比度下降）。")

beta_sweep = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
psnr_beta, ssim_beta = [], []
for beta in beta_sweep:
    xb = osem_recon(y_counts, scale, background,
                   num_subsets=10, n_subiter=20, prior_beta=beta)
    pb, sb, nmb = metrics(x_true, xb)
    psnr_beta.append(pb)
    ssim_beta.append(sb)
    print(f"    beta={beta:4.2f} : PSNR={pb:6.2f} dB, SSIM={sb:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(beta_sweep, psnr_beta, 'o-', label='PSNR')
ax.set_xlabel(r'先验强度 $\beta$')
ax.set_ylabel('PSNR (dB)')
ax.set_title('OSL：稳定化后 beta 扫到 2.0 不发散，但非单调平滑（甜点处峰值反升）')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤5_OSL_beta稳定性.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤5_OSL_beta稳定性.png")


# ========== 步骤6 : 背景（randoms/scatter）估计误差敏感度（拓展实验）==========
print("\n>>> 步骤6 : 背景（randoms/scatter）估计误差敏感度（拓展实验）")
print("    说明：真实 PET 中 randoms/scatter 背景是独立测量估计的，与真值有误差；")
print("          步骤1 的仿真里重建用 background 与真值一致（完美背景校正）。")
print("          此处故意把重建用 background 乘以 0.5 / 2.0（约 -50% / +100%），观察对重建的影响。")

bg_factors = [0.5, 1.0, 2.0]
psnr_bg_osem, psnr_bg_osl = [], []
ssim_bg_osem, ssim_bg_osl = [], []
for f in bg_factors:
    xbo = osem_recon(y_counts, scale, background * f,
                     num_subsets=10, n_subiter=20, prior_beta=0.0)
    xbl = osem_recon(y_counts, scale, background * f,
                     num_subsets=10, n_subiter=20, prior_beta=0.05)
    pbo, sbo, nbo = metrics(x_true, xbo)
    pbl, sbl, nbl = metrics(x_true, xbl)
    psnr_bg_osem.append(pbo); ssim_bg_osem.append(sbo)
    psnr_bg_osl.append(pbl);  ssim_bg_osl.append(sbl)
    print(f"    bg×{f:4.2f} : OSEM PSNR={pbo:6.2f} dB, OSL PSNR={pbl:6.2f} dB")
print("    注：本仿真 background 仅约为投影均值的 5%，故 ±20% 量级的误差")
print("         只相当于约 1% 信号误差、对重建几乎无影响；这里把误差放大到")
print("         ±50%/×2 以让灵敏度可见。真实 PET 的 randoms/scatter 占比常达")
print("         10%~30%，此时背景估计误差对定量结果影响显著，正是 PET 校正的经典难点。")

xpos = np.arange(len(bg_factors))
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(xpos - 0.2, psnr_bg_osem, width=0.4, label='OSEM')
ax.bar(xpos + 0.2, psnr_bg_osl, width=0.4, label='OSL')
ax.set_xticks(xpos)
ax.set_xticklabels([f'bg×{f:g}' for f in bg_factors])
ax.set_ylabel('PSNR (dB)')
ax.set_title('背景估计误差敏感度')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤6_背景误差敏感度.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤6_背景误差敏感度.png")


# ========== 步骤7 : 保存 JSON 结果 ==========
print("\n>>> 步骤7 : 保存运行结果到 JSON")

results = {
    "experiment": "附录16D",
    "title": "PET发射断层重建——OSEM / OSL",
    "image_size": N,
    "n_angles": int(theta.size),
    "total_expected_counts": float(total_counts),
    "background": float(background),
    "num_subsets": 10,
    "n_subiterations": 20,
    "osl_prior_beta": 0.05,
    "reference": "sirf Python examples/PET/osem_reconstruction.py, user_osmaposl.py, osl_reconstruction.py, acquisition_model.py",
    "metrics": {
        "OSEM": {"PSNR_dB": p_o, "SSIM": s_o, "NMSE": nmse_o},
        "OSL": {"PSNR_dB": p_l, "SSIM": s_l, "NMSE": nmse_l},
    },
    "extension_experiments": {
        "osl_beta_sweep": {
            "betas": beta_sweep,
            "psnr_db": psnr_beta,
            "ssim": ssim_beta,
        },
        "background_error_sensitivity": {
            "factors": bg_factors,
            "osem_psnr_db": psnr_bg_osem,
            "osem_ssim": ssim_bg_osem,
            "osl_psnr_db": psnr_bg_osl,
            "osl_ssim": ssim_bg_osl,
        },
    },
}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)
print("    已保存：results_summary.json")
print(f"\n{'='*64}")
print("附录16D 完成。结果见 results_summary.json 与各步骤 PNG。")
print(f"{'='*64}")