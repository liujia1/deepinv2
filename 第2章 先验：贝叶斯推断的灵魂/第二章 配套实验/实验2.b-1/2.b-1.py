"""
实验2.b-1 TGV：TV的改进
对应章节：附录2B TGV：TV的改进
知识点：阶梯效应根源；TGV定义；TGV vs TV对比；分段仿射 vs 分段常数

素材来源：
  - 2.5.py: TV去噪与阶梯效应展示
  - 附录2B: TGV理论
"""

import numpy as np
import os
import sys
import warnings

# ====== 静默模式配置 ======
SILENT_MODE = True  # True: 不弹窗、不显示警告；False: 正常交互模式

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.b-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.b-1')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')  # 用脚本目录绝对路径，避免依赖运行时CWD
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    cn_font = setup_chinese_font(save_dir=_chinese_path)
    if cn_font:
        plt.rcParams['font.sans-serif'] = [cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

np.random.seed(42)

def gradient_forward(x):
    """
    前向差分梯度算子（零边界条件）
    
    (Dx)_h[i,j] = x[i,j+1] - x[i,j]  (水平方向)
    (Dx)_v[i,j] = x[i+1,j] - x[i,j]  (垂直方向)
    
    边界处理: 最后一列/行的梯度为0（零边界）
    """
    h, w = x.shape
    dx_h = np.zeros_like(x)
    dx_v = np.zeros_like(x)
    dx_h[:, :-1] = x[:, 1:] - x[:, :-1]
    dx_v[:-1, :] = x[1:, :] - x[:-1, :]
    return dx_h, dx_v

def divergence_backward(p_h, p_v):
    """
    后向差分散度算子（零边界条件）
    
    这是 gradient_forward 的伴随算子（负梯度）:
    div(p) = -D^* p，满足 <Dx, p> = <x, -div(p)>
    
    实现方式（严格为 gradient_forward 的负伴随，满足 <Dx,p> = <x,-div p>）:
    div[i,j] = (p_h[i,j] - p_h[i,j-1]) + (p_v[i,j] - p_v[i-1,j])
    超出边界的 p 值视为 0（零边界），故末列/末行自然落到 -p_h[i,-2]、-p_v[-2,j]。
    
    注意: 必须与 gradient_forward 的前向差分严格配套。若误写成
    p[i,j-1]-p[i,j] 得到的是 +D^T 而非 -D^T，Chambolle-Pock 会发散。
    """
    h, w = p_h.shape
    div = np.zeros((h, w))
    # 水平: -D_h^T p_h
    div[:, :-1] += p_h[:, :-1]
    div[:, 1:] -= p_h[:, :-1]
    # 垂直: -D_v^T p_v
    div[:-1, :] += p_v[:-1, :]
    div[1:, :] -= p_v[:-1, :]
    return div

def symmetrized_gradient_forward(w_h, w_v):
    """对称梯度算子 E(w) = [∂h w_h, ∂v w_v; (∂h w_v + ∂v w_h)/2]"""
    h, w = w_h.shape
    Ew_1 = np.zeros_like(w_h)
    Ew_2 = np.zeros_like(w_v)
    Ew_3 = np.zeros_like(w_h)
    
    Ew_1[:, :-1] = w_h[:, 1:] - w_h[:, :-1]
    Ew_2[:-1, :] = w_v[1:, :] - w_v[:-1, :]
    
    d_wv_h = np.zeros_like(w_v)
    d_wh_v = np.zeros_like(w_h)
    d_wv_h[:, :-1] = w_v[:, 1:] - w_v[:, :-1]
    d_wh_v[:-1, :] = w_h[1:, :] - w_h[:-1, :]
    Ew_3 = 0.5 * (d_wv_h + d_wh_v)
    
    return Ew_1, Ew_2, Ew_3

def symmetrized_divergence_backward(q_1, q_2, q_3):
    """
    对称梯度算子的伴随（负散度）: 返回 -E^T q
    
    E(w) 的三个分量: E1=∂h w_h, E2=∂v w_v, E3=0.5(∂h w_v + ∂v w_h)。
    其负伴随按分量组装（每个差分都用与 symmetrized_gradient_forward
    严格配套的 -D^T，否则 Chambolle-Pock 发散）:
    - w_h 分量: -D_h^T q_1 + 0.5*(-D_v^T q_3)   （q_3 经 ∂v w_h 交叉贡献 w_h）
    - w_v 分量: -D_v^T q_2 + 0.5*(-D_h^T q_3)   （q_3 经 ∂h w_v 交叉贡献 w_v）
    """
    def minus_DhT(p):
        o = np.zeros_like(p)
        o[:, :-1] += p[:, :-1]
        o[:, 1:] -= p[:, :-1]
        return o
    def minus_DvT(p):
        o = np.zeros_like(p)
        o[:-1, :] += p[:-1, :]
        o[1:, :] -= p[:-1, :]
        return o
    div_w_h = minus_DhT(q_1) + 0.5 * minus_DvT(q_3)
    div_w_v = minus_DvT(q_2) + 0.5 * minus_DhT(q_3)
    return div_w_h, div_w_v

def tgv_objective(x, w_h, w_v, y, alpha0, alpha1):
    """
    教科书 TGV^2 目标值:
        J = 0.5||x-y||^2 + alpha1*||Dx-w||_2(逐点) + alpha0*||Ew||_F(逐点)

    注意 Frobenius 内积约定: E = [E1, E3; E3, E2] 时
        ||E||_F^2 = E1^2 + E2^2 + 2*E3^2
    （E3 是"半"剪切分量 (∂h wv + ∂v wh)/2）。
    对偶投影球必须与之配套使用 sqrt(q1^2+q2^2+0.5*q3^2)，
    二者是 Fenchel 共轭对；若投影用 2*q3^2 则解的是剪切项被折半的
    各向异性变体，而非教科书 TGV^2。
    """
    Dhx, Dvx = gradient_forward(x)
    Ew_1, Ew_2, Ew_3 = symmetrized_gradient_forward(w_h, w_v)
    J = 0.5 * np.sum((x - y) ** 2)
    J += alpha1 * np.sum(np.sqrt((Dhx - w_h) ** 2 + (Dvx - w_v) ** 2))
    J += alpha0 * np.sum(np.sqrt(Ew_1 ** 2 + Ew_2 ** 2 + 2 * Ew_3 ** 2))
    return float(J)

def estimate_K_norm_sq(shape, n_power=200, seed=0):
    """
    幂迭代估计联合线性算子 K = [D, -I; 0, E] 的谱范数平方。
    （散度/对称散度实现即 -D^T / -E^T，故 K* 直接复用。）
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((h, w))
    wh = rng.standard_normal((h, w))
    wv = rng.standard_normal((h, w))
    lam = 0.0
    for _ in range(n_power):
        dx_h, dx_v = gradient_forward(x)
        e1, e2, e3 = symmetrized_gradient_forward(wh, wv)
        # K (x, w) = (Dx - w, Ew)
        z_p_h, z_p_v = dx_h - wh, dx_v - wv
        # K* (p, q) = (-D^T p, -p - E^T q)；本代码散度即 -D^T/-E^T
        x_new = -divergence_backward(z_p_h, z_p_v)
        dqh, dqv = symmetrized_divergence_backward(e1, e2, e3)
        wh_new = -z_p_h - dqh
        wv_new = -z_p_v - dqv
        lam = float(np.sum(x * x_new + wh * wh_new + wv * wv_new)
                    / np.sum(x ** 2 + wh ** 2 + wv ** 2))
        nrm = np.sqrt(np.sum(x_new ** 2 + wh_new ** 2 + wv_new ** 2))
        x, wh, wv = x_new / nrm, wh_new / nrm, wv_new / nrm
    return lam

def tgv_denoise_primal_dual(y, alpha0, alpha1, max_iter=5000, min_iter=500,
                            tol=5e-5, tau=0.25, sigma=0.25, check_every=100,
                            return_history=False):
    """
    TGV去噪的原始-对偶（Chambolle-Pock）实现
    
    TGV模型: min_{x,w} 0.5||x-y||^2 + alpha1*||Dx-w||_1 + alpha0*||Ew||_F
    
    其中:
    - x: 图像
    - w: 辅助向量场（梯度的仿射分量逼近）
    - D: 梯度算子
    - E: 对称梯度算子（Frobenius 内积约定）
    
    迭代停止:
        每 check_every 次迭代计算一次原始目标值 J，当 J 的相对变化
        小于 tol 且已迭代不少于 min_iter 时停止；最多 max_iter 次。
        大 alpha0（趋近 TV 极限）时对偶变量达到约束界很慢，
        固定 500 次会把"尚未收敛"误报为"参数不敏感平台"，故必须早停或加大迭代。
    
    参数:
        y: 含噪图像
        alpha0: 二阶正则化参数
        alpha1: 一阶正则化参数
        max_iter, min_iter: 最大 / 最小迭代次数
        tol: 原始目标值相对变化停止阈值
        tau, sigma: 步长参数，需满足 tau*sigma*||K||^2 <= 1
    
    返回:
        x: 去噪后图像
        w_h, w_v: 辅助向量场
        info: {'n_iter': 实际迭代次数, 'history': [(iter, J), ...]}
    """
    h, w = y.shape
    x = y.copy()
    w_h = np.zeros((h, w))
    w_v = np.zeros((h, w))
    p_h = np.zeros((h, w))
    p_v = np.zeros((h, w))
    q_1 = np.zeros((h, w))
    q_2 = np.zeros((h, w))
    q_3 = np.zeros((h, w))
    
    theta = 1.0
    x_bar = x.copy()
    w_h_bar = w_h.copy()
    w_v_bar = w_v.copy()
    
    history = []
    prev_J = None
    n_iter_used = max_iter
    for k in range(max_iter):
        Dx_h, Dx_v = gradient_forward(x_bar)
        
        p_h = p_h + sigma * (Dx_h - w_h_bar)
        p_v = p_v + sigma * (Dx_v - w_v_bar)
        
        norm_p = np.sqrt(p_h**2 + p_v**2)
        p_h = p_h / np.maximum(1, norm_p / alpha1)
        p_v = p_v / np.maximum(1, norm_p / alpha1)
        
        Ew_1, Ew_2, Ew_3 = symmetrized_gradient_forward(w_h_bar, w_v_bar)
        
        q_1 = q_1 + sigma * Ew_1
        q_2 = q_2 + sigma * Ew_2
        q_3 = q_3 + sigma * Ew_3
        
        # 投影到 ||q||_F <= alpha0 的对偶球：
        # 与 E 的 Frobenius 内积（E3 为半剪切分量）严格配套，系数必须是 0.5
        norm_q = np.sqrt(q_1**2 + q_2**2 + 0.5*q_3**2)
        q_1 = q_1 / np.maximum(1, norm_q / alpha0)
        q_2 = q_2 / np.maximum(1, norm_q / alpha0)
        q_3 = q_3 / np.maximum(1, norm_q / alpha0)
        
        div_p = divergence_backward(p_h, p_v)
        div_q_h, div_q_v = symmetrized_divergence_backward(q_1, q_2, q_3)
        
        x_new = (x + tau * div_p + tau * y) / (1 + tau)
        # ∇_w L = -p + E^T q，故下降步 w - tau*∇_w = w + tau*(p - E^T q) = w + tau*(p + div_q)
        w_h_new = w_h + tau * (div_q_h + p_h)
        w_v_new = w_v + tau * (div_q_v + p_v)
        
        x_bar = x_new + theta * (x_new - x)
        w_h_bar = w_h_new + theta * (w_h_new - w_h)
        w_v_bar = w_v_new + theta * (w_v_new - w_v)
        
        x = x_new
        w_h = w_h_new
        w_v = w_v_new
        
        if (k + 1) % check_every == 0:
            J = tgv_objective(x, w_h, w_v, y, alpha0, alpha1)
            history.append((k + 1, J))
            if prev_J is not None and (k + 1) >= min_iter:
                if abs(prev_J - J) / max(abs(prev_J), 1e-12) < tol:
                    n_iter_used = k + 1
                    break
            prev_J = J
    
    info = {'n_iter': n_iter_used}
    if return_history:
        info['history'] = history
    return x, w_h, w_v, info

def find_smooth_gradient_region(image, win=32, min_slope=0.0015, max_slope=0.02,
                                max_curvature=0.003, align_frac=0.8):
    """
    在"干净参考图"上自动检测真正的平滑渐变区域。
    返回 (orientation, fixed_index, range_slice):
      orientation='h' 表示沿水平方向（fixed_index=行号，slice给列范围）；
      orientation='v' 表示沿垂直方向（fixed_index=列号，slice给行范围）。

    旧实现只在单行上按"一阶梯度落在 20%-70% 分位"筛选，会把草地/衣物等
    中等幅度纹理误判成渐变区。新检测器在水平、垂直两个方向上对每个
    滑动窗口同时要求:
      1) 斜率显著且不过陡（|线性拟合斜率| 在 [min_slope, max_slope]，
         排除平坦常数区与强边缘）；
      2) 曲率低（二阶差分 RMS 小于 max_curvature，排除纹理）；
      3) 方向一致（一阶梯度同号比例 >= align_frac，排除震荡段）。
    在所有候选中取"曲率/斜率"最小者；找不到则回退到天空经验区域（垂直）。
    """
    H, W = image.shape
    best = None  # (score, orientation, idx, j0, j1, slope, curv)

    def scan(orientation, idx, prof):
        nonlocal best
        for j0 in range(2, len(prof) - win - 1, 2):
            seg = prof[j0:j0 + win]
            t = np.arange(win)
            slope = float(np.polyfit(t, seg, 1)[0])
            if not (min_slope <= abs(slope) <= max_slope):
                continue
            g = np.diff(seg)
            d2 = seg[2:] - 2 * seg[1:-1] + seg[:-2]
            curv = float(np.sqrt(np.mean(d2 ** 2)))
            if curv > max_curvature:
                continue
            aligned = np.mean(np.sign(g) == np.sign(slope))
            if aligned < align_frac:
                continue
            score = curv / (abs(slope) + 1e-9)
            if best is None or score < best[0]:
                best = (score, orientation, idx, j0, j0 + win, slope, curv)

    for r in range(4, H - 3, 2):       # 水平剖面
        scan('h', r, image[r].astype(np.float64))
    for c in range(4, W - 3, 2):       # 垂直剖面（cameraman天空主要是垂直渐变）
        scan('v', c, image[:, c].astype(np.float64))

    if best is not None:
        return best[1], best[2], slice(best[3], best[4])
    # 回退：cameraman 天空区（列88，行2-34，垂直方向经验位置）
    return 'v', int(W * 0.69), slice(2, 2 + win)

n = 128
camera = resize(data.camera(), (n, n))

sigma_noise = 0.1
try:  # skimage>=0.22 用 rng；显式传种子保证跨版本可复现
    camera_noisy = random_noise(camera, mode='gaussian', var=sigma_noise**2, rng=42)
except TypeError:
    camera_noisy = random_noise(camera, mode='gaussian', var=sigma_noise**2, seed=42)

alpha0 = 0.3
alpha1 = 0.15

# ====== TV 基线（公平口径） ======
# 同正则预算：TGV 在 w≡0 时第一项退化为 α1·TV(x)，故匹配基线取 weight=α1；
# 另做权重扫描给出"各自最优调参"口径，避免用欠/过平滑的 TV 人为制造差距。
import time as _time
alpha_tv = alpha1
_t0 = _time.time()
camera_tv = denoise_tv_chambolle(camera_noisy, weight=alpha_tv)
time_tv = _time.time() - _t0

tv_weight_grid = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
tv_sweep = []
camera_tv_best, best_tv_rec = None, None
for _wt in tv_weight_grid:
    _im = denoise_tv_chambolle(camera_noisy, weight=_wt)
    _p = peak_signal_noise_ratio(camera, _im)
    _s = structural_similarity(camera, _im, data_range=1.0)
    rec = {'weight': float(_wt), 'psnr_dB': float(round(_p, 3)),
           'ssim': float(round(_s, 5))}
    tv_sweep.append(rec)
    if best_tv_rec is None or _p > best_tv_rec['psnr_dB']:
        best_tv_rec, camera_tv_best = rec, _im

print("===== TGV：TV的改进 =====")
print(f"\n[步长选择说明]")
print(f"  Chambolle-Pock收敛条件: τ·σ·||K||² ≤ 1")
print(f"  K 是联合算子 K=[D,-I; 0,E]（注意 -I 耦合，不能只算 D 的范数）")
K_norm_sq = estimate_K_norm_sq((n, n))
print(f"  幂迭代实测 ||K||² = {K_norm_sq:.3f} (||K||≈{np.sqrt(K_norm_sq):.3f})，"
      f"纯梯度界 ||D||²≤8")
print(f"  本实验取 τ=σ=0.25，τ·σ·||K||² = {0.25*0.25*K_norm_sq:.3f} < 1 ✓")
print(f"  迭代策略: 原始目标相对变化<5e-5 时早停（最少500，最多5000）")

_t0 = _time.time()
camera_tgv, w_h, w_v, tgv_info = tgv_denoise_primal_dual(
    camera_noisy, alpha0, alpha1, return_history=True)
time_tgv = _time.time() - _t0
n_iter_main = tgv_info['n_iter']

psnr_noisy = peak_signal_noise_ratio(camera, camera_noisy)
psnr_tv = peak_signal_noise_ratio(camera, camera_tv)
psnr_tgv = peak_signal_noise_ratio(camera, camera_tgv)
psnr_tv_best = peak_signal_noise_ratio(camera, camera_tv_best)

ssim_noisy = structural_similarity(camera, camera_noisy, data_range=1.0)
ssim_tv = structural_similarity(camera, camera_tv, data_range=1.0)
ssim_tgv = structural_similarity(camera, camera_tgv, data_range=1.0)
ssim_tv_best = structural_similarity(camera, camera_tv_best, data_range=1.0)

print(f"\n噪声水平 σ = {sigma_noise}")
print(f"\n去噪结果对比:")
print(f"  含噪图像: PSNR = {psnr_noisy:.2f} dB, SSIM = {ssim_noisy:.4f}")
print(f"  TV(weight={alpha_tv:g}, 同预算): PSNR = {psnr_tv:.2f} dB, SSIM = {ssim_tv:.4f}")
print(f"  TV(weight={best_tv_rec['weight']:g}, 最优调参): PSNR = {psnr_tv_best:.2f} dB, SSIM = {ssim_tv_best:.4f}")
print(f"  TGV ({n_iter_main}次迭代): PSNR = {psnr_tgv:.2f} dB, SSIM = {ssim_tgv:.4f}")
print(f"  耗时: TV {time_tv:.3f}s, TGV {time_tgv:.1f}s")
print(f"\nTGV参数:")
print(f"  α₀ (二阶正则化) = {alpha0}")
print(f"  α₁ (一阶正则化) = {alpha1}")
print(f"  α₀/α₁ = {alpha0/alpha1:.2f} (Bredies等建议的起点值，需按数据重新标定)")
print(f"  口径提醒: 全局PSNR/SSIM上TGV并不必然胜过调好参的TV；")
print(f"  TGV的核心收益是分段仿射区域的阶梯伪影，见步骤6金标准实验。")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(camera_noisy, cmap='gray')
axes[0].set_title(f'含噪图像\nPSNR={psnr_noisy:.2f}dB')
axes[0].axis('off')

axes[1].imshow(camera, cmap='gray')
axes[1].set_title('原始图像')
axes[1].axis('off')

axes[2].imshow(camera_tv, cmap='gray')
axes[2].set_title(f'TV去噪 (weight={alpha_tv:g}，与$\\alpha_1$同预算)\n边缘锐利，渐变区阶梯化\nPSNR={psnr_tv:.2f}dB')
axes[2].axis('off')

axes[3].imshow(camera_tgv, cmap='gray')
axes[3].set_title(f'TGV去噪 ({n_iter_main}次迭代)\nSSIM与TV持平、PSNR略低，渐变区更自然\nPSNR={psnr_tgv:.2f}dB')
axes[3].axis('off')

plt.suptitle('TV vs TGV：阶梯效应的改善（公平基线：TV权重=$\\alpha_1$）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV_vs_TGV对比.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

row = n // 2
grad_ori, grad_fix, grad_region = find_smooth_gradient_region(camera)
if grad_ori == 'h':
    prof_true = camera[grad_fix, grad_region]
    prof_tv = camera_tv[grad_fix, grad_region]
    prof_tgv = camera_tgv[grad_fix, grad_region]
    region_desc = f'第{grad_fix}行，列{grad_region.start}-{grad_region.stop}（水平渐变）'
    x_axis_label = '窗口内列序号'
else:
    prof_true = camera[grad_region, grad_fix]
    prof_tv = camera_tv[grad_region, grad_fix]
    prof_tgv = camera_tgv[grad_region, grad_fix]
    region_desc = f'第{grad_fix}列，行{grad_region.start}-{grad_region.stop}（垂直渐变）'
    x_axis_label = '窗口内行序号'
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(camera[row, :], 'k-', linewidth=2, label='真实信号')
axes[0].plot(camera_tv[row, :], 'r-', linewidth=1.5, label='TV（阶梯效应）')
axes[0].plot(camera_tgv[row, :], 'g-', linewidth=1.5, label='TGV（更平滑）')
axes[0].set_title(f'中心行剖面对比（第{row}行）')
axes[0].legend()
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

axes[1].plot(prof_true, 'k-', linewidth=2,
             marker='o', markersize=3, label='真实（平滑渐变区）')
axes[1].plot(prof_tv, 'r-s', linewidth=1.5,
             markersize=3, label='TV（阶梯效应）')
axes[1].plot(prof_tgv, 'g-', linewidth=1.5,
             label='TGV（渐变自然）')
axes[1].set_title(f'真正的渐变区域放大（{region_desc}）\n'
                  f'检测准则：显著斜率+低二阶曲率+方向一致')
axes[1].legend()
axes[1].set_xlabel(x_axis_label)
axes[1].set_ylabel('灰度值')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_阶梯效应改善分析.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

w_mag = np.sqrt(w_h**2 + w_v**2)
im = axes[0].imshow(w_mag, cmap='hot')
axes[0].set_title('辅助向量场 |w|（梯度仿射分量）')
axes[0].axis('off')
plt.colorbar(im, ax=axes[0], fraction=0.046)

h, w = camera_tgv.shape
skip = 8
Y, X = np.mgrid[0:h:skip, 0:w:skip]
# 旧写法 scale = p90(|w|)*skip*10 使箭长约为网格间距的千分之一，箭头退化成点；
# 且 -w_v 方向取反。这里按数据坐标显式放大：典型箭长≈0.55倍网格间距。
p90 = float(np.percentile(w_mag, 90)) + 1e-12
arrow_gain = 0.55 * skip / p90
U = w_h[::skip, ::skip] * arrow_gain
V = w_v[::skip, ::skip] * arrow_gain  # 行号向下增大，与 imshow 的 y 方向一致
axes[1].imshow(camera_tgv, cmap='gray', alpha=0.3)
q = axes[1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1.0,
                   width=0.0028, alpha=0.85, color='blue')
axes[1].quiverkey(q, 0.92, 0.06, 0.55 * skip,
                  f'|w|={p90:.4f}', labelpos='E', coordinates='axes',
                  fontproperties={'size': 9})
axes[1].set_title('向量场w叠加在去噪图像上\n(箭头长度≈典型幅值放大显示)')
axes[1].axis('off')

axes[2].hist(w_mag.flatten(), bins=50, density=True, alpha=0.7, color='blue')
axes[2].set_xlabel('|w|')
axes[2].set_ylabel('概率密度')
axes[2].set_title('辅助向量场幅值分布')
axes[2].grid(True, alpha=0.3)

plt.suptitle('TGV核心机制：辅助向量场w可视化', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_辅助向量场可视化.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

alpha_ratios = [0.5, 1.0, 2.0, 4.0, 8.0]
alpha1_fixed = 0.15
tgv_results = []
psnr_results = []
ssim_results = []
niter_results = []

assert len(alpha_ratios) <= 5, "alpha_ratios最多5个元素，否则步骤4子图布局会溢出"

print("\n===== 参数敏感性实验（均求解至收敛：500-5000次自适应） =====")
print(f"固定 α₁ = {alpha1_fixed}，变化 α₀/α₁ 比值:")
print("  提醒: 比值越大（二阶项越受压制）TGV行为越接近TV，收敛也越慢；")
print("        旧版固定500次会把大比值的'未收敛段'误读成'参数不敏感平台'。")

for i, ratio in enumerate(alpha_ratios):
    print(f"  [{i+1}/{len(alpha_ratios)}] 正在计算 α₀/α₁ = {ratio:.1f} ... ", end='', flush=True)
    a0 = ratio * alpha1_fixed
    x_tgv, _, _, info_r = tgv_denoise_primal_dual(camera_noisy, a0, alpha1_fixed)
    tgv_results.append(x_tgv)
    niter_results.append(info_r['n_iter'])
    psnr_r = peak_signal_noise_ratio(camera, x_tgv)
    ssim_r = structural_similarity(camera, x_tgv, data_range=1.0)
    psnr_results.append(psnr_r)
    ssim_results.append(ssim_r)
    print(f"完成 ({info_r['n_iter']}次, PSNR={psnr_r:.2f}dB, SSIM={ssim_r:.4f})")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (ratio, img) in enumerate(zip(alpha_ratios[:5], tgv_results[:5])):
    ax = axes[i // 3, i % 3]
    ax.imshow(img, cmap='gray')
    ax.set_title(f'$\\alpha_0/\\alpha_1$ = {ratio:.1f} ({niter_results[i]}次)\n'
                 f'PSNR={psnr_results[i]:.2f}dB, SSIM={ssim_results[i]:.3f}')
    ax.axis('off')

axes[1, 2].plot(alpha_ratios, psnr_results, 'bo-', linewidth=2, markersize=8, label='PSNR')
for r_, p_, ni_ in zip(alpha_ratios, psnr_results, niter_results):
    axes[1, 2].annotate(f'{ni_}', (r_, p_), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color='blue')
axes[1, 2].set_xlabel('$\\alpha_0/\\alpha_1$（数字为实际迭代次数）')
axes[1, 2].set_ylabel('PSNR (dB)', color='blue')
axes[1, 2].tick_params(axis='y', labelcolor='blue')
axes[1, 2].grid(True, alpha=0.3)
ax2 = axes[1, 2].twinx()
ax2.plot(alpha_ratios, ssim_results, 'rs-', linewidth=2, markersize=8, label='SSIM')
ax2.set_ylabel('SSIM', color='red')
ax2.tick_params(axis='y', labelcolor='red')
axes[1, 2].set_title('参数敏感性分析（收敛解；比值↑→趋近TV）')

plt.suptitle('TGV参数敏感性：$\\alpha_0/\\alpha_1$对去噪效果的影响（全部收敛后）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_参数敏感性分析.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods = ['含噪', f'TV\n(weight={alpha_tv:g})', f'TGV\n($\\alpha_0/\\alpha_1={alpha0/alpha1:g}$)']
psnrs = [psnr_noisy, psnr_tv, psnr_tgv]
ssims = [ssim_noisy, ssim_tv, ssim_tgv]
colors = ['gray', 'red', 'green']

x_pos = np.arange(len(methods))
bars1 = axes[0].bar(x_pos, psnrs, color=colors, alpha=0.7)
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(methods)
axes[0].set_title('PSNR对比')
axes[0].set_ylim([0, max(psnrs) + 5])
for i, p in enumerate(psnrs):
    axes[0].text(i, p + 0.3, f'{p:.2f}', ha='center', fontsize=10)

bars2 = axes[1].bar(x_pos, ssims, color=colors, alpha=0.7)
axes[1].set_ylabel('SSIM')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(methods)
axes[1].set_title('SSIM对比')
axes[1].set_ylim([0, 1.05])
for i, s in enumerate(ssims):
    axes[1].text(i, s + 0.02, f'{s:.3f}', ha='center', fontsize=10)

table_data = [
    ['惩罚项', r'$\|\nabla x\|_1$', r'$\alpha_1\|\nabla x-w\|_1+\alpha_0\|Ew\|_1$'],
    ['偏好解', '分段常数', '分段仿射'],
    ['渐变区域', '阶梯效应', '自然渐变'],
    ['边缘保持', '能', '能'],
    ['参数数量', r'1个 ($\alpha$)', r'2个 ($\alpha_0$, $\alpha_1$)'],
    ['计算复杂度', '低', '较高'],
]
axes[2].axis('off')
table = axes[2].table(cellText=table_data,
                      colLabels=['特性', 'TV', 'TGV'],
                      loc='center',
                      cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
axes[2].set_title('TV vs TGV 特性对比', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤5_性能与特性对比.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

# ====== 步骤6：合成分段仿射幻影（理论金标准） ======
print("\n===== 步骤6：合成分段仿射幻影（常数-线性斜坡-常数） =====")
N = 128
_cols = np.arange(N)
phantom = np.zeros((N, N))
_ramp = 0.3 + 0.4 * np.clip((_cols - 20) / 80.0, 0.0, 1.0)
_ramp[_cols >= 100] = 0.7
phantom[:, :] = _ramp[None, :]
sigma_phantom = 0.05
rng_ph = np.random.default_rng(20240520)
phantom_noisy = phantom + rng_ph.normal(0.0, sigma_phantom, phantom.shape)

def ramp_curvature(u, lo=28, hi=94):
    """斜坡内部二阶差分总变差（逐行平均）：分段仿射真值≈0，TV阶梯解显著偏大"""
    d2 = u[:, lo + 2:hi] - 2 * u[:, lo + 1:hi - 1] + u[:, lo:hi - 2]
    return float(np.mean(np.sum(np.abs(d2), axis=1)))

def ramp_mse(u, lo=28, hi=94):
    return float(np.mean((u[:, lo:hi] - phantom[:, lo:hi]) ** 2))

ph_a1 = 0.08
ph_a0 = 2.0 * ph_a1
_t0 = _time.time()
ph_tv = denoise_tv_chambolle(phantom_noisy, weight=ph_a1)
ph_tgv, ph_wh, ph_wv, ph_info = tgv_denoise_primal_dual(phantom_noisy, ph_a0, ph_a1)
time_ph_tgv = _time.time() - _t0

# 双方同一系数网格"各自最优调参"，避免"没调好TV"的质疑
ph_coef_grid = [0.04, 0.06, 0.08, 0.10, 0.12]
ph_tv_sweep = []
ph_best_tv_img, ph_best_tv_p = None, -1.0
for _c in ph_coef_grid:
    _im = denoise_tv_chambolle(phantom_noisy, weight=_c)
    _p = peak_signal_noise_ratio(phantom, _im)
    ph_tv_sweep.append({'weight': float(_c), 'psnr_dB': float(round(_p, 3)),
                        'ssim': float(round(structural_similarity(phantom, _im, data_range=1.0), 5))})
    if _p > ph_best_tv_p:
        ph_best_tv_p, ph_best_tv_img = _p, _im

def _ph_psnr(u): return peak_signal_noise_ratio(phantom, u)
def _ph_ssim(u): return structural_similarity(phantom, u, data_range=1.0)
phantom_metrics = {
    'noise_sigma': sigma_phantom,
    'matched_budget_alpha1': ph_a1,
    'n_iter_tgv': ph_info['n_iter'],
    'psnr_noisy_dB': float(round(_ph_psnr(phantom_noisy), 2)),
    'psnr_tv_matched_dB': float(round(_ph_psnr(ph_tv), 2)),
    'psnr_tv_best_dB': float(round(_ph_psnr(ph_best_tv_img), 2)),
    'psnr_tgv_dB': float(round(_ph_psnr(ph_tgv), 2)),
    'ssim_tv_matched': float(round(_ph_ssim(ph_tv), 4)),
    'ssim_tv_best': float(round(_ph_ssim(ph_best_tv_img), 4)),
    'ssim_tgv': float(round(_ph_ssim(ph_tgv), 4)),
    'ramp_curvature': {'noisy': round(ramp_curvature(phantom_noisy), 4),
                       'tv_matched': round(ramp_curvature(ph_tv), 4),
                       'tv_best': round(ramp_curvature(ph_best_tv_img), 4),
                       'tgv': round(ramp_curvature(ph_tgv), 4)},
    'ramp_mse': {'tv_matched': round(ramp_mse(ph_tv), 6),
                 'tgv': round(ramp_mse(ph_tgv), 6)},
}
print(f"  含噪 {phantom_metrics['psnr_noisy_dB']:.2f}dB | "
      f"TV(同预算{ph_a1}) {phantom_metrics['psnr_tv_matched_dB']:.2f}dB | "
      f"TV(最优调参) {phantom_metrics['psnr_tv_best_dB']:.2f}dB | "
      f"TGV {phantom_metrics['psnr_tgv_dB']:.2f}dB ({ph_info['n_iter']}次, {time_ph_tgv:.1f}s)")
print(f"  斜坡曲率TV（越小越接近仿射）: TV {phantom_metrics['ramp_curvature']['tv_matched']:.4f} "
      f"-> TGV {phantom_metrics['ramp_curvature']['tgv']:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes[0, 0].imshow(phantom, cmap='gray'); axes[0, 0].set_title('真值：分段仿射幻影\n常数-线性斜坡-常数'); axes[0, 0].axis('off')
axes[0, 1].imshow(ph_tv, cmap='gray')
axes[0, 1].set_title(f"TV去噪 (weight={ph_a1}，与$\\alpha_1$同预算)\nPSNR={_ph_psnr(ph_tv):.2f}dB，斜坡曲率={ramp_curvature(ph_tv):.3f}")
axes[0, 1].axis('off')
axes[0, 2].imshow(ph_tgv, cmap='gray')
axes[0, 2].set_title(f"TGV去噪 ($\\alpha_0/\\alpha_1=2$, {ph_info['n_iter']}次)\nPSNR={_ph_psnr(ph_tgv):.2f}dB，斜坡曲率={ramp_curvature(ph_tgv):.4f}")
axes[0, 2].axis('off')

pr = N // 2
cc = np.arange(N)
axes[1, 0].plot(cc, phantom[pr], 'k-', lw=2, label='真值')
axes[1, 0].plot(cc, phantom_noisy[pr], color='gray', lw=0.6, alpha=0.6, label='含噪')
axes[1, 0].plot(cc, ph_tv[pr], 'r-', lw=1.4, label='TV（阶梯）')
axes[1, 0].plot(cc, ph_tgv[pr], 'g-', lw=1.4, label='TGV（仿射）')
axes[1, 0].set_title('中心行剖面'); axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=.3)
axes[1, 0].set_xlabel('像素索引'); axes[1, 0].set_ylabel('灰度值')

lo, hi = 28, 94
axes[1, 1].plot(cc[lo:hi], phantom[pr, lo:hi], 'k-', lw=2, label='真值')
axes[1, 1].plot(cc[lo:hi], ph_tv[pr, lo:hi], 'r-s', ms=3, lw=1.2, label='TV（阶梯）')
axes[1, 1].plot(cc[lo:hi], ph_tgv[pr, lo:hi], 'g-', lw=1.5, label='TGV（与真值重合）')
axes[1, 1].set_title('线性斜坡内部放大'); axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=.3)
axes[1, 1].set_xlabel('像素索引'); axes[1, 1].set_ylabel('灰度值')

axes[1, 2].axis('off')
_ph_table = [
    ['PSNR (dB)', f"{_ph_psnr(ph_tv):.2f}", f"{_ph_psnr(ph_best_tv_img):.2f}", f"{_ph_psnr(ph_tgv):.2f}"],
    ['SSIM', f"{_ph_ssim(ph_tv):.4f}", f"{_ph_ssim(ph_best_tv_img):.4f}", f"{_ph_ssim(ph_tgv):.4f}"],
    ['斜坡曲率TV', f"{ramp_curvature(ph_tv):.4f}", f"{ramp_curvature(ph_best_tv_img):.4f}", f"{ramp_curvature(ph_tgv):.4f}"],
    ['斜坡MSE', f"{ramp_mse(ph_tv):.2e}", '—', f"{ramp_mse(ph_tgv):.2e}"],
]
_tbl = axes[1, 2].table(cellText=_ph_table,
                        colLabels=['指标', f'TV\n同预算({ph_a1})', 'TV\n各自最优调参', 'TGV\n($\\alpha_0/\\alpha_1=2$)'],
                        loc='center', cellLoc='center')
_tbl.auto_set_font_size(False); _tbl.set_fontsize(10); _tbl.scale(1.15, 1.8)
axes[1, 2].set_title('金标准定量对比\n（TGV对分段仿射信号的优势与调参无关）', fontsize=11, pad=15)

plt.suptitle('步骤6：合成分段仿射幻影——TV阶梯效应 vs TGV自然仿射（理论金标准）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤6_分段仿射幻影验证.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()
plt.close(fig)

print("\n===== 附录2B 核心结论 =====")
print("\n1. 阶梯效应根源:")
print("   TV只使用一阶导数，偏好分段常数函数")
print("   渐变区域被近似为一系列阶梯（步骤6幻影给出决定性证据）")
print("\n2. TGV改进思路:")
print("   引入二阶导数信息，允许分段仿射函数")
print("   TGV = min_w [α₁||∇x-w||₁ + α₀||Ew||₁]")
print("   辅助向量场w逼近梯度的仿射分量")
print("\n3. TV与TGV的关系:")
print("   α₀→∞ 迫使 Ew=0（w为无穷小刚体运动场）；再附加 w≡0 限制时")
print("   TGV 严格退化为 α₁·TV——即 TV 是 TGV 的特例（比值越大TGV行为越接近TV）")
print("\n4. 参数选择:")
print("   α₀/α₁≈2 只是 Bredies 等建议的起点，必须按数据/噪声重新标定")
print("   本例 camera 全局指标峰不在2；且全局指标本就不是TGV的卖点")
print("   α₀控制二阶平滑，α₁控制一阶平滑")
print("\n5. 应用场景:")
print("   PET/MRI重建、图像修复等渐变区域重要的场景")
print("\n6. 局限性:")
print("   TGV仍是显式先验，参数需手工调节")
print("   更根本的改进需要走向隐式先验（2.4节）")

# ===== 保存数值结果 =====
import json
results_summary = {
    'schema_version': 2,
    'revision_note': 'v2: 修正q3对偶投影权重(2.0->0.5, Frobenius内积配套); '
                     '全部TGV求解至收敛(自适应500-5000次); TV基线改为同预算weight=alpha1并附调参扫描; '
                     '新增步骤6分段仿射幻影金标准; psnr_tv_dB 含义同v1但取值已改为同预算口径',
    'image_size': n,
    'noise_sigma': float(sigma_noise),
    'model_convention': '二阶TGV^2; E(w)对称梯度取半剪切分量E3=(dh wv+dv wh)/2, '
                        '||Ew||_F^2=E1^2+E2^2+2E3^2; 对偶球投影 sqrt(q1^2+q2^2+0.5q3^2)',
    'solver': {'method': 'Chambolle-Pock primal-dual',
               'tau': 0.25, 'sigma': 0.25,
               'K_norm_sq_measured': float(round(K_norm_sq, 4)),
               'tau_sigma_K2': float(round(0.25 * 0.25 * K_norm_sq, 4)),
               'min_iter': 500, 'max_iter': 5000, 'tol': 5e-5},
    'psnr_noisy_dB': float(round(psnr_noisy, 2)),
    'psnr_tv_dB': float(round(psnr_tv, 2)),          # 同预算 TV(weight=alpha1)
    'psnr_tgv_dB': float(round(psnr_tgv, 2)),
    'ssim_noisy': float(round(ssim_noisy, 4)),
    'ssim_tv': float(round(ssim_tv, 4)),
    'ssim_tgv': float(round(ssim_tgv, 4)),
    'tv_baseline': {
        'policy': 'matched_budget_weight=alpha1; plus best-tuned weight sweep',
        'matched_weight': float(alpha1),
        'matched_psnr_dB': float(round(psnr_tv, 2)),
        'matched_ssim': float(round(ssim_tv, 4)),
        'best_weight': float(best_tv_rec['weight']),
        'best_psnr_dB': float(round(psnr_tv_best, 2)),
        'best_ssim': float(round(ssim_tv_best, 4)),
        'weight_sweep': tv_sweep,
        'time_seconds': float(round(time_tv, 4)),
    },
    'tgv_main': {'n_iter': int(n_iter_main),
                 'time_seconds': float(round(time_tgv, 2))},
    'alpha0': float(alpha0),
    'alpha1': float(alpha1),
    'alpha0_alpha1_ratio': float(round(alpha0/alpha1, 2)),
    'sensitivity_alpha_ratio': {
        f'ratio_{r:.1f}': {'psnr_dB': float(round(p, 2)),
                           'ssim': float(round(s, 4)),
                           'n_iter': int(ni)}
        for r, p, s, ni in zip(alpha_ratios, psnr_results, ssim_results, niter_results)},
    'phantom_piecewise_affine': phantom_metrics,
    'gradient_region_detected': {'orientation': grad_ori,
                                 'fixed_index': int(grad_fix),
                                 'range_start': int(grad_region.start),
                                 'range_stop': int(grad_region.stop)},
}

def _to_native(obj):
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

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
