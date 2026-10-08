# -*- coding: utf-8 -*-
"""
实验5.4-2 近端算子 vs 学习去噪器：结构对偶性验证
对应章节：5.4 MAP与MMSE的结构对偶性
  - 结构对偶：Moreau包络 vs 软下卷积（第5小节）
  - 温度参数：从MMSE到MAP的连续过渡（第6小节，本实验通过温度扫描真实演示）

知识点：
  - TV-ULA（显式先验）：后验 exp(-f-β·TV)，β为先验权重；用 Moreau-Yoshida
    平滑参数 γ 把 TV 光滑化，∇(βTV)_γ = (x-prox_{γβ·TV}(x))/γ
  - PnP-ULA（隐式先验）：使用学习去噪器+Tweedie等式，ε 为先验平滑参数
  - 结构对偶：两者形式相同 y - c·∇(·)，但目标不同（众数 vs 均值）
  - 温度参数：ULA 噪声幅度 √(2Tδ)。T=1 采样后验→后验均值(MMSE)；
    T=0 退化为确定性（近端）梯度下降→光滑后验的众数(MAP)
  - 注意：TV-ULA 与 PnP-ULA 两条链 T 都等于 1，输出都是【后验均值(MMSE)】；
    MAP 由 T=0 的确定性链给出，不能把带噪声的 TV-ULA 链标成"MAP/T=0"。

运行前提：
  推荐 GPU + 预训练 RealSN-DnCNN 模型（自动下载到 Pretrained_models/）。
  - 作为 .py 直接运行：资源（cman.png、sampling_tools/ 等）从脚本所在目录读取。
  - Google Colab：把整个"实验5.4-2"文件夹上传到云盘根目录
    （即 /content/drive/MyDrive/实验5.4-2/），直接粘贴到单元格运行即可，
    会自动挂载 Drive 并从该目录读取资源、写回结果。
  - 本地 Jupyter：无 __file__，默认以当前工作目录为资源目录；
    若不在实验目录，先设置 os.environ['EXP542_DIR'] 为实验5.4-2的完整路径。
  - EXP542_OUTDIR：仅指定图片/JSON 的输出目录（默认资源目录）。
  - 断点续跑：每条链的状态自动保存到输出目录的 checkpoints/ 子目录，
    中断（断网/关页面）后重新运行会自动从最后检查点继续；已完成的链直接跳过。
    想强制从头跑：删除 checkpoints/ 目录，或设置 EXP542_FRESH=1。
    注意：checkpoint 按实验参数（尺寸/β/γ/ε/σ/迭代数/温度）自动区分，改参数不冲突。
  - EXP542_ALLOW_CPU=1：无 GPU 时按正式预算在 CPU 上运行（较慢）；
    EXP542_QUICK=1：快速冒烟模式（128×128 裁剪、少量迭代，仅验证流程）。

本实验对应5.4节第5小节"结构对偶"和第6小节"温度参数"。
"""

import math
import json
import hashlib
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys

torch.set_num_threads(os.cpu_count())

# ====== 路径解析（兼容 .py 脚本 / Colab / 本地 Jupyter）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules


def _resolve_dirs():
    """
    返回 (SCRIPT_DIR, SAVE_DIR)：
      - SCRIPT_DIR：资源目录（cman.png、sampling_tools/、.chinese/、Pretrained_models/）
      - SAVE_DIR ：图片/JSON 输出目录（可用 EXP542_OUTDIR 覆盖）
    Colab 单元格里没有 __file__，统一从 Google Drive 的实验目录取资源；
    本地 Jupyter 同样没有 __file__，回退到当前工作目录（可用 EXP542_DIR 指定）。
    """
    if _IN_COLAB:
        from google.colab import drive
        if not os.path.isdir(_gdrive):
            print("正在挂载 Google Drive...")
            drive.mount('/content/drive')
        exp_dir = os.path.join(_gdrive, '实验5.4-2')
        if not os.path.isdir(exp_dir):
            print(f"[警告] 未找到云盘目录 {exp_dir}，请确认实验文件（cman.png、"
                  f"sampling_tools/、Pretrained_models/ 等）已上传到该目录。")
        return exp_dir, exp_dir

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 本地 Jupyter/IPython 单元格：无 __file__
        script_dir = os.environ.get('EXP542_DIR', os.getcwd())
        print(f"[路径] 检测到 Notebook 环境（无 __file__），资源目录取: {script_dir}")
        if not os.path.exists(os.path.join(script_dir, 'cman.png')):
            print("[警告] 该目录下未找到 cman.png。请在单元格中设置 "
                  "os.environ['EXP542_DIR'] 为实验5.4-2 的实际目录后重跑，"
                  "或把工作目录切换到该目录。")
    save_dir = os.environ.get('EXP542_OUTDIR', script_dir)
    return script_dir, save_dir


SCRIPT_DIR, SAVE_DIR = _resolve_dirs()
os.makedirs(SAVE_DIR, exist_ok=True)

_chinese_path = os.path.join(SCRIPT_DIR, '.chinese')
os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
torch.manual_seed(42)

# 导入sampling_tools模块
if os.path.isdir(os.path.join(SCRIPT_DIR, 'sampling_tools')):
    sys.path.insert(0, SCRIPT_DIR)
    try:
        from sampling_tools import chambolle_prox_TV, blur_operators, PSNR
        from sampling_tools.load_model import DnCNN
        _has_sampling_tools = True
    except ImportError as e:
        print(f"警告: sampling_tools 导入失败: {e}")
        _has_sampling_tools = False
else:
    _has_sampling_tools = False
    print("警告: sampling_tools 模块未找到")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ====== 运行模式：正式 / CPU 正式 / QUICK 冒烟 ======
_QUICK = os.environ.get('EXP542_QUICK', '0') == '1'
_ALLOW_CPU = os.environ.get('EXP542_ALLOW_CPU', '0') == '1'
_has_gpu = device.type == 'cuda'

if _QUICK:
    # 快速冒烟：128中心裁剪、短迭代，只用于验证代码流程，结果不代表算法性能
    CROP, TV_NITER, PNP_NITER = 128, 400, 400
    TV_BURNIN, PNP_BURNIN = 200, 200
    CKPT_EVERY = 50
    TEMP_CONF = {0.0: (300, 150), 0.3: (400, 200)}
    WARM_INNER = 10  # QUICK 仅验证流程；prox 精度不达标但不影响流程检查
    print("[模式] QUICK 冒烟模式：小裁剪+短迭代，数值仅用于流程验证")
else:
    CROP = None  # 使用完整 256×256
    # 迭代预算对齐官方教学实验 lab1(TV,2万)/lab2(PnP,1万) 的量级；
    # TV 链采用热启动 Chambolle，预算 12000（烧入40%）足以越过混合时间
    TV_NITER, PNP_NITER = 12000, 10000
    TV_BURNIN, PNP_BURNIN = 4800, 1000
    CKPT_EVERY = 200
    # 温度扫描：T=0 确定性链取末段（=众数）；T=0.3 中间温度
    TEMP_CONF = {0.0: (3000, 1500), 0.3: (6000, 2000)}
    # 热启动内迭代数：实测在链轨迹上 prox 稳态 rms 误差 2.6e-4，
    # 比 MYULA 固有平滑尺度 γ=2.5e-3 小近 10 倍（内求解误差可忽略）
    WARM_INNER = 50

# 模型自动下载功能
_model_dir = os.path.join(SCRIPT_DIR, 'Pretrained_models')
_model_path = os.path.join(_model_dir, 'RealSN_DnCNN_noise5.pth')

def download_model_if_needed(model_path, model_dir):
    """自动下载预训练模型（如果不存在），返回模型是否可用"""
    if os.path.exists(model_path):
        print(f"[Model] 检测到已缓存的模型权重: {model_path}")
        return True
    print(f"[Model] 未找到预训练模型: {model_path}，正在尝试自动下载...")
    os.makedirs(model_dir, exist_ok=True)
    model_url = "https://github.com/uclaopt/Provable_Plug_and_Play/raw/master/Pretrained_models/RealSN_DnCNN_noise5.pth"
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[Model] 模型下载成功！已保存到: {model_path}")
        return True
    except Exception as e:
        print(f"[Model] 模型下载失败: {e}")
        print("[Model] 请手动下载 Pretrained_models/RealSN_DnCNN_noise5.pth")
        return False

_has_model = download_model_if_needed(_model_path, _model_dir)

_run_ok = _has_sampling_tools and _has_model and (_has_gpu or _ALLOW_CPU or _QUICK)
if not _run_ok:
    print("\n" + "=" * 60)
    print("运行环境检查")
    print("=" * 60)
    print(f"  GPU可用: {_has_gpu}")
    print(f"  sampling_tools可用: {_has_sampling_tools}")
    print(f"  预训练模型可用: {_has_model}")
    print("  CPU 正式运行请设环境变量 EXP542_ALLOW_CPU=1（较慢）；")
    print("  快速流程验证请设 EXP542_QUICK=1。")
    print("=" * 60)


# ============================================================
# 热启动 Chambolle：与 sampling_tools/chambolle_prox_TV.py 完全相同的
# 对偶迭代与边界约定（2026-10复核：两者均已修复末行/列负伴随索引
# -p[-1]→-p[-2]），但对偶变量在链迭代间持续复用（warm start）。
# 链每步 X 变化极小（δ~1e-5），热启动后仅需少量内迭代即可达到 prox 精度，
# 使 TV 链在 CPU 上也可承担上万次 prox 求值。
# ============================================================
def _gradient_im(u):
    """前向差分，末行/末列补0（与库 GradientIm 完全一致）"""
    sx = list(u.shape); sx[0] = 1
    dux = torch.vstack([u[1:, :] - u[:-1, :], torch.zeros(sx, device=u.device)])
    sy = list(u.shape); sy[1] = 1
    duy = torch.hstack([u[:, 1:] - u[:, :-1], torch.zeros(sy, device=u.device)])
    return dux, duy

def _divergence_im(p1, p2):
    """与库 DivergenceIm 完全一致（严格负伴随，末行/末列取 -p[-2]）"""
    # 2026-10复核：与库同步修复，末行/末列原为恒为0的 -p[-1]
    z = p2[:, 1:-1] - p2[:, :-2]
    s2 = list(p2.shape); s2[1] = 1
    v = torch.hstack([p2[:, 0].reshape(s2), z, -p2[:, -2].reshape(s2)])
    z = p1[1:-1, :] - p1[:-2, :]
    s1 = list(p1.shape); s1[0] = 1
    u = torch.vstack([p1[0, :].reshape(s1), z, -p1[-2, :].reshape(s1)])
    return v + u

class WarmStartTV:
    """带持久对偶变量的 Chambolle TV 近端算子"""
    def __init__(self, lam, n_inner=10, tau=0.249):
        self.lam = float(lam)
        self.K = int(n_inner)
        self.tau = tau
        self.px = None
        self.py = None

    def reset(self, shape, device):
        self.px = torch.zeros(shape, device=device)
        self.py = torch.zeros(shape, device=device)

    def prox(self, g):
        if self.px is None or self.px.shape != g.shape:
            self.reset(g.shape, g.device)
        lam, tau, K = self.lam, self.tau, self.K
        px, py = self.px, self.py
        for _ in range(K):
            u = _divergence_im(px, py) - g / lam
            upx, upy = _gradient_im(u)
            tmp = torch.sqrt(upx * upx + upy * upy)
            px = (px + tau * upx) / (1.0 + tau * tmp)
            py = (py + tau * upy) / (1.0 + tau * tmp)
        self.px, self.py = px, py
        return g - lam * _divergence_im(px, py)


def _selfcheck_warm_tv(y_ref):
    """
    自校验1：热启动实现(K=300,从零初始化) vs 库 chambolle_prox_TV(300次)。
    两者是同一不动点迭代，输出应高度一致。
    """
    ref = chambolle_prox_TV(y_ref, device, {'lambda': 0.05, 'maxiter': 300, 'tol': 1e-14})
    w = WarmStartTV(0.05, n_inner=300)
    out = w.prox(y_ref)
    err = float((ref - out).abs().max())
    print(f"[SelfCheck] 热启动Chambolle vs 库实现 最大偏差 = {err:.2e}（应 < 2e-4）")
    return err < 2e-4


def load_denoiser_cpu_safe(path, dev):
    """兼容 CPU/GPU 的 RealSN-DnCNN 加载（库的 load_model 内部强制 .cuda）"""
    net = DnCNN(channels=1, num_of_layers=17)
    sd = torch.load(path, map_location=dev, weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval().to(dev)
    return net


# ============================================================
# 主体实验
# ============================================================
if _run_ok:
    print("\n" + "=" * 60)
    print("步骤1：观测模型与先验准备")
    print("=" * 60)

    # ---- 图像（QUICK 模式做中心裁剪）----
    im = np.array(Image.open(os.path.join(SCRIPT_DIR, "cman.png")))
    if CROP is not None:
        h0 = (im.shape[0] - CROP) // 2
        w0 = (im.shape[1] - CROP) // 2
        im = im[h0:h0 + CROP, w0:w0 + CROP]
    x = torch.Tensor(im / 255.).to(device)

    # ---- 模糊算子与含噪观测（与官方 lab2 一致）----
    kernel_len = [5, 5]
    size = [im.shape[0], im.shape[1]]
    A, AT, AAT_norm = blur_operators(kernel_len, size, "uniform", device)

    torch.manual_seed(42)
    y0 = A(x)
    BSNRdb = 40
    sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') \
        / math.sqrt(torch.numel(x) * 10 ** (BSNRdb / 10))
    y = y0 + sigma * torch.randn_like(x)

    gradf = lambda X: AT(A(X) - y) / sigma ** 2
    L_y = float(AAT_norm / sigma ** 2)

    # ---- 显式 TV 先验参数 ----
    # 目标后验：π(x) ∝ exp(-f(x) - β·TV(x))
    # γ：Moreau-Yoshida 平滑参数；prox 的正则参数 = γβ；
    # ∇(βTV)_γ = (x - prox_{γβ·TV}(x))/γ，其 Lipschitz 常数 ≤ 1/γ。
    beta_tv = 20.0
    gamma_tv = 0.0025
    prox_lam = beta_tv * gamma_tv              # = 0.05，chambolle 的 lambda
    delta_tv = 0.99 / (1.0 / gamma_tv + L_y)   # 教材4.4节性质4的标准稳定界（非"简化"）

    # ---- 隐式学习先验参数 ----
    L_net = 1.0
    model = load_denoiser_cpu_safe(_model_path, device)
    denoise = lambda z: (z - model(z[None][None])[0][0]).detach()
    eps = (5 / 255) ** 2                       # PnP 先验平滑参数（Tweedie 扰动尺度）
    delta_pnp = 0.99 / (L_net / eps + L_y)

    projbox = lambda z: torch.clamp(z, min=0.0, max=1.0)

    print(f"图像尺寸: {tuple(x.shape)}")
    print(f"噪声 σ={float(sigma):.6f}（BSNR={BSNRdb}dB），似然 Lipschitz L_y={L_y:.1f}")
    print(f"TV 先验: β={beta_tv}, γ={gamma_tv}, prox参数 γβ={prox_lam}, δ={delta_tv:.3e}")
    print(f"PnP 先验: ε={eps:.6e}, δ={delta_pnp:.3e}")
    print(f"等效移动尺度对照: 每步 TV={delta_tv/gamma_tv:.2e} vs PnP={delta_pnp/eps:.2e}（每步先验作用量）")

    assert _selfcheck_warm_tv(y), "热启动 Chambolle 自校验失败，请检查实现"
    _warm_impl_err = float((chambolle_prox_TV(y, device, {'lambda': prox_lam, 'maxiter': 300, 'tol': 1e-14})
                           - WarmStartTV(prox_lam, n_inner=300).prox(y)).abs().max())

    def _warm_traj_check():
        """沿真实链轨迹跑50步后，比较热启动prox与库冷启动300次参考解"""
        w = WarmStartTV(prox_lam, n_inner=WARM_INNER)
        X = y.clone(); sd = math.sqrt(2.0 * delta_tv)
        with torch.no_grad():
            for _ in range(50):
                X = projbox(X - delta_tv * (gradf(X) + (X - w.prox(X)) / gamma_tv) + sd * torch.randn_like(X))
            pw = w.prox(X)
            pr = chambolle_prox_TV(X, device, {'lambda': prox_lam, 'maxiter': 300, 'tol': 1e-14})
        return float(torch.sqrt(((pw - pr) ** 2).mean()))

    _warm_traj_rms = _warm_traj_check()
    print(f"[SelfCheck] 链轨迹热启动 prox rms = {_warm_traj_rms:.2e}（阈值5e-4"
          + ("，QUICK模式不强制）" if _QUICK else "）"))
    if not _QUICK:
        assert _warm_traj_rms < 5e-4, "链轨迹上热启动 prox 精度不达标，请增大 WARM_INNER"

    # ============================================================
    # 断点续跑：每条链的状态（X/均值/二阶矩/TV对偶变量/RNG）落盘。
    # 中断后重运行会从最后一个检查点继续；已完成的链直接加载，不重跑。
    # 想强制全新运行：删除 checkpoints/ 目录，或设 EXP542_FRESH=1。
    # ============================================================
    ckpt_dir = os.path.join(SAVE_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    _FRESH = os.environ.get('EXP542_FRESH', '0') == '1'

    def _ckpt_path(tag, T, niter, burnin, delta):
        key = "|".join(map(str, [tag, T, niter, burnin, f"{delta:.10g}",
                                 tuple(x.shape), beta_tv, gamma_tv, prox_lam,
                                 f"{eps:.6e}", float(sigma), 'Q' if _QUICK else 'F']))
        h = hashlib.md5(key.encode('utf-8')).hexdigest()[:10]
        return os.path.join(ckpt_dir, f"{tag}_T{T}_n{niter}_{h}.pt")

    # ============================================================
    # 通用 ULA 链（带温度 T）
    #   X_{m+1} = X_m - δ·漂移(X_m) + √(2Tδ) Z
    #   T=1：标准后验采样（MMSE）；T=0：确定性（近端）梯度下降（MAP）
    # ============================================================
    def run_chain(name, drift_fn, niter, delta, T, burnin, ckpt_every,
                  warm=None, tag='chain'):
        noise_scale = math.sqrt(2.0 * delta * T)
        ckpt_file = _ckpt_path(tag, T, niter, burnin, delta)

        X, sm, s2, cnt, traj, start = None, None, None, 0, [], 0
        resumed = False

        # RNG 状态恢复仅为与"一次连续运行"逐位等价；噪声 i.i.d.，恢复失败也不影响
        # 链的统计正确性。不同 torch 版本（如 Colab 的 Python3.13 新版）反序列化
        # 出的 RNG 张量可能不被 set_rng_state 接受，此处严格校验并容错。
        def _restore_rng(ckd, step):
            r_cpu = ckd.get('rng_cpu')
            ok_cpu = (isinstance(r_cpu, torch.Tensor) and r_cpu.device.type == 'cpu'
                      and r_cpu.dtype == torch.uint8 and r_cpu.numel() > 0)
            if ok_cpu:
                try:
                    torch.set_rng_state(r_cpu.contiguous())
                except TypeError:
                    torch.manual_seed(42 + step)
            else:
                torch.manual_seed(42 + step)  # 新 i.i.d. 噪声流，统计上等价
            r_dev = ckd.get('rng_dev')
            if device.type == 'cuda' and isinstance(r_dev, torch.Tensor) \
                    and r_dev.dtype == torch.uint8:
                try:
                    torch.cuda.set_rng_state(r_dev.contiguous().cuda(device))
                except (TypeError, RuntimeError):
                    pass

        if not _FRESH and os.path.exists(ckpt_file):
            try:
                ck = torch.load(ckpt_file, map_location=device, weights_only=True)
            except Exception as e:
                ck = None
                print(f"  [警告] 检查点读取失败（{e}），该链从头开始")
            # 参数一致性校验：参数变了就当作新链从头跑
            if ck is not None:
                same = (ck.get('niter') == niter and ck.get('burnin') == burnin
                        and ck.get('T') == T
                        and abs(ck.get('delta', -1.0) - delta) < 1e-15)
                if same:
                    X, sm, s2, cnt = ck['X'], ck['sm'], ck['s2'], ck['cnt']
                    traj, start = ck.get('traj', []), int(ck['step'])
                    if warm is not None and ck.get('px') is not None:
                        warm.px = ck['px'].to(device)
                        warm.py = ck['py'].to(device)
                    _restore_rng(ck, start)
                    resumed = True

        def _build_result():
            mean = sm / max(cnt, 1)
            var = torch.clamp(s2 / max(cnt, 1) - mean ** 2, min=0.0)
            return {'final': X, 'mean': mean, 'std': torch.sqrt(var), 'traj': traj,
                    'niter': niter, 'burnin': burnin, 'T': T}

        if start >= niter:
            print(f"[跳过] {name} T={T} 已完成（{niter}步），直接加载既有结果")
            return _build_result()

        if not resumed:
            torch.manual_seed(42)  # 两条链同一噪声序列，保证对照公平
            X = y.clone()
            sm = torch.zeros_like(X)
            s2 = torch.zeros_like(X)
            print(f"[新链] {name} T={T}：{niter}步（检查点 {os.path.basename(ckpt_file)}）")
        else:
            print(f"[续跑] {name} T={T}：从第 {start}/{niter} 步继续")

        # mininterval=2.0：每2秒才推一帧。Colab 前端对高频进度流（默认约10帧/秒）
        # 会节流丢弃，末帧（100%）可能不被渲染而冻在半路；降低帧率可显著减少丢弃。
        # 仅影响显示，不影响采样与结果。完成状态另有下方 [完成] 永久打印行兜底。
        log_every = max(niter // 10, 1)  # 永久里程碑行间隔（每条链约9行）
        pbar = tqdm(range(start, niter), desc=f"{name} T={T}", initial=start, total=niter,
                    mininterval=2.0)

        def _save(step):
            ckpt = {
                'step': step, 'niter': niter, 'burnin': burnin, 'T': T, 'delta': delta,
                'X': X, 'sm': sm, 's2': s2, 'cnt': cnt, 'traj': traj,
                'rng_cpu': torch.get_rng_state(),
                'rng_dev': torch.cuda.get_rng_state() if device.type == 'cuda' else None,
                'px': warm.px if warm is not None else None,
                'py': warm.py if warm is not None else None,
            }
            tmp = ckpt_file + '.tmp'
            torch.save(ckpt, tmp)
            os.replace(tmp, ckpt_file)  # 原子替换，防止中断时写坏检查点

        for k in pbar:
            Z = noise_scale * torch.randn_like(X) if T > 0 else 0.0
            X = projbox(X - delta * drift_fn(X) + Z)
            if k >= burnin:
                cnt += 1
                sm += X
                s2 += X * X
            if (k + 1) % ckpt_every == 0:
                if cnt > 0:
                    ps = float(PSNR(x, sm / cnt))
                    traj.append((k + 1, round(ps, 3)))
                    pbar.set_postfix(meanPSNR=f"{ps:.2f}")
                _save(k + 1)  # 烧入期也存，任何阶段中断都能续
            # 每10%一条带换行的永久进度行：Colab 的 widget 进度条可能随机冻在
            # 半路（前端丢帧），而 print 永久行必被渲染，保证进度可见且可信
            if (k + 1) % log_every == 0 and (k + 1) < niter:
                pct = 100 * (k + 1) // niter
                if cnt > 0:
                    print(f"    · {name} T={T}：{k + 1}/{niter}（{pct}%），"
                          f"均值PSNR={float(PSNR(x, sm / cnt)):.2f} dB", flush=True)
                else:
                    print(f"    · {name} T={T}：{k + 1}/{niter}（{pct}%），烧入中…",
                          flush=True)
        _save(niter)
        # 带换行的永久完成行：防止背靠背的下一条进度条在 Colab 里
        # 通过 \r 把本条的 100% 末帧覆盖掉（进度条本身不影响，仅显示问题）
        print(f"  [完成] {name} T={T}：{niter}/{niter} 步，"
              f"后验均值 PSNR={float(PSNR(x, sm / max(cnt, 1))):.2f} dB", flush=True)
        return _build_result()

    # TV 先验漂移：∇f + ∇(βTV)_γ
    warm_tv = WarmStartTV(prox_lam, n_inner=WARM_INNER)
    tv_drift = lambda X: gradf(X) + (X - warm_tv.prox(X)) / gamma_tv
    # PnP 漂移：∇f - s_ε，s_ε=(D_ε-X)/ε
    pnp_drift = lambda X: gradf(X) - (denoise(X) - X) / eps

    # ============================================================
    # 步骤2：T=1 正式采样（两条链都是后验均值/MMSE）
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤2：TV-ULA / PnP-ULA 后验采样（T=1，MMSE）")
    print("=" * 60)
    warm_tv.reset(x.shape, device)
    tv_res = run_chain("TV-ULA", tv_drift, TV_NITER, delta_tv, 1.0, TV_BURNIN, CKPT_EVERY,
                       warm=warm_tv, tag='tv')
    pnp_res = run_chain("PnP-ULA", pnp_drift, PNP_NITER, delta_pnp, 1.0, PNP_BURNIN, CKPT_EVERY,
                        tag='pnp')

    def convergence_gain(traj):
        """后1/4段的后验均值PSNR提升：用于判断链是否已进入平台期"""
        if len(traj) < 4:
            return float('nan')
        return round(traj[-1][1] - traj[int(len(traj) * 0.75)][1], 3)

    psnr_y = float(PSNR(x, y))
    psnr_tv = float(PSNR(tv_res['mean'], x))
    psnr_pnp = float(PSNR(pnp_res['mean'], x))
    print(f"\n含噪观测 PSNR = {psnr_y:.2f} dB")
    print(f"TV-ULA  后验均值 PSNR = {psnr_tv:.2f} dB（后1/4段提升 {convergence_gain(tv_res['traj'])} dB）")
    print(f"PnP-ULA 后验均值 PSNR = {psnr_pnp:.2f} dB（后1/4段提升 {convergence_gain(pnp_res['traj'])} dB）")

    # ============================================================
    # 步骤3：显式先验(TV) vs 隐式先验(学习去噪器) —— 两种 MMSE 的对比
    # ============================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    axes[0][0].imshow(im, cmap='gray')
    axes[0][0].set_title('原始图像 $x$')
    axes[0][0].axis('off')

    axes[0][1].imshow(y.cpu().numpy(), cmap='gray')
    axes[0][1].set_title('含噪+模糊观测 $y$\n' + f'PSNR: {psnr_y:.2f} dB')
    axes[0][1].axis('off')

    axes[0][2].imshow(tv_res['mean'].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][2].set_title(f'TV-ULA 后验均值 (MMSE, T=1)\nPSNR: {psnr_tv:.2f} dB')
    axes[0][2].axis('off')

    axes[0][3].imshow(pnp_res['mean'].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][3].set_title(f'PnP-ULA 后验均值 (MMSE, T=1)\nPSNR: {psnr_pnp:.2f} dB')
    axes[0][3].axis('off')

    axes[1][0].axis('off')
    axes[1][1].axis('off')
    axes[1][2].imshow(tv_res['final'].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1][2].set_title('TV-ULA 最终样本（带噪，非MAP）')
    axes[1][2].axis('off')
    axes[1][3].imshow(pnp_res['final'].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1][3].set_title('PnP-ULA 最终样本（带噪，非MAP）')
    axes[1][3].axis('off')

    vmax_std = max(float(tv_res['std'].max()), float(pnp_res['std'].max()))
    axes[2][0].axis('off')
    axes[2][1].axis('off')
    im2 = axes[2][2].imshow(tv_res['std'].cpu().numpy(), cmap='hot', vmin=0, vmax=vmax_std)
    axes[2][2].set_title('TV-ULA 后验标准差')
    axes[2][2].axis('off')
    axes[2][3].imshow(pnp_res['std'].cpu().numpy(), cmap='hot', vmin=0, vmax=vmax_std)
    axes[2][3].set_title('PnP-ULA 后验标准差')
    axes[2][3].axis('off')

    fig.suptitle('实验5.4-2 显式先验(TV) vs 隐式先验(学习去噪器)：两条链均为 T=1 的后验均值(MMSE)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤3_显式vs隐式先验.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # 步骤3b：收敛诊断（后验均值 PSNR 随迭代轨迹）
    # ============================================================
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, tag in [(axs[0], tv_res, 'TV-ULA'), (axs[1], pnp_res, 'PnP-ULA')]:
        it = [t[0] for t in res['traj']]
        ps = [t[1] for t in res['traj']]
        ax.plot(it, ps, '-o', ms=3, label='后验均值 PSNR')
        ax.axhline(psnr_y, color='gray', ls='--', lw=1, label=f'观测 $y$ = {psnr_y:.2f} dB')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'{tag} 收敛轨迹（末值 {ps[-1]:.2f} dB，后1/4段提升 {convergence_gain(res["traj"])} dB）')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤3b_收敛诊断.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # 步骤4：近端算子 vs 去噪器的结构对偶性（算子层面的 MAP/MMSE 对偶）
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤4：近端算子 vs 去噪器的结构对偶性")
    print("=" * 60)

    prox_result = chambolle_prox_TV(y, device, {'lambda': prox_lam, 'maxiter': 200})
    with torch.no_grad():
        denoise_result = denoise(y)

    residual_prox = y - prox_result          # = γ·∇(βTV)_γ(y)
    residual_denoise = y - denoise_result    # = ε·∇R̄_ε(y) = -ε·∇log p_ε(y)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0][0].imshow(y.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][0].set_title('含噪观测 $y$')
    axes[0][0].axis('off')
    axes[0][1].imshow(prox_result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][1].set_title(r'TV近端算子 $\mathrm{prox}_{\gamma\beta\mathrm{TV}}(y)$' + '\n硬选择(min)→众数(MAP)方向')
    axes[0][1].axis('off')
    axes[0][2].imshow(denoise_result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][2].set_title(r'去噪器 $D_\varepsilon(y)$' + '\n软选择(softmin)→均值(MMSE)方向')
    axes[0][2].axis('off')

    axes[1][0].axis('off')
    # 两种残差幅度不同（γβ∼0.05 vs ε∼3.8e-4），各自独立定色标并标注 RMS，不做幅度直读比较
    rp = residual_prox.cpu().numpy()
    rd = residual_denoise.cpu().numpy()
    im_a = axes[1][1].imshow(rp, cmap='RdBu_r', vmin=-np.abs(rp).max(), vmax=np.abs(rp).max())
    axes[1][1].set_title(r'$y-\mathrm{prox}(y)=\gamma\nabla(\beta\mathrm{TV})_\gamma(y)$'
                         + f'\nRMS={np.sqrt(np.mean(rp**2)):.4f}')
    axes[1][1].axis('off')
    fig.colorbar(im_a, ax=axes[1][1], fraction=0.046, pad=0.04)
    im_b = axes[1][2].imshow(rd, cmap='RdBu_r', vmin=-np.abs(rd).max(), vmax=np.abs(rd).max())
    axes[1][2].set_title(r'$y-D_\varepsilon(y)=\varepsilon\nabla\bar R_\varepsilon(y)$'
                         + f'\nRMS={np.sqrt(np.mean(rd**2)):.4f}')
    axes[1][2].axis('off')
    fig.colorbar(im_b, ax=axes[1][2], fraction=0.046, pad=0.04)

    fig.suptitle('近端算子 vs 去噪器：结构对偶性（残差均刻画边缘，幅度尺度不同不可直读）',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_结构对偶性.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # 步骤5：温度参数实验 T ∈ {0, 0.3, 1}
    #   T=1 的结果直接复用步骤2；T=0/0.3 重新跑链。
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤5：温度参数扫描（T=0 众数MAP → T=1 均值MMSE）")
    print("=" * 60)

    temp_tv = {1.0: tv_res}
    temp_pnp = {1.0: pnp_res}
    for T, (nit, bur) in TEMP_CONF.items():
        print(f"\n-- 温度 T={T} --")
        warm_tv.reset(x.shape, device)
        temp_tv[T] = run_chain("TV", tv_drift, nit, delta_tv, T, bur, CKPT_EVERY,
                               warm=warm_tv, tag='tv')
        temp_pnp[T] = run_chain("PnP", pnp_drift, nit, delta_pnp, T, bur, CKPT_EVERY,
                                tag='pnp')

    temps = sorted(temp_tv.keys())
    fig, axes = plt.subplots(2, len(temps), figsize=(6 * len(temps), 11))
    label_map = {0.0: 'T=0：确定性链→众数(MAP)',
                 0.3: 'T=0.3：中间温度',
                 1.0: 'T=1：后验采样→均值(MMSE)'}
    for row, (store, tag) in enumerate([(temp_tv, 'TV 先验'), (temp_pnp, '学习去噪器先验')]):
        for j, T in enumerate(temps):
            r = store[T]
            img = r['mean'] if T > 0 else r['final']
            axes[row][j].imshow(img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            tag_psnr = float(PSNR(x, img))
            extra = f"\n平均std={float(r['std'].mean()):.4f}" if T > 0 else \
                    f"\n末段漂移std={float(r['std'].mean()):.2e}（应≈0）"
            axes[row][j].set_title(f'{tag}  {label_map[T]}\nPSNR: {tag_psnr:.2f} dB{extra}', fontsize=11)
            axes[row][j].axis('off')
    fig.suptitle('温度参数：噪声幅度 √(2Tδ) 控制从众数(MAP)到均值(MMSE)的过渡', fontsize=15, y=1.02)
    fig.text(0.5, 0.01,
             '注：T=0 链无随机噪声，收敛到 Moreau/ε-光滑化后验的众数（教学意义上的MAP；'
             '严格TV-MAP可由Chambolle-Pock/ADMM求得）；T=1 即 TV-ULA/PnP-ULA 的后验均值。',
             ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(SAVE_DIR, '步骤5_温度参数_MAP到MMSE.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # 控制台总结
    # ============================================================
    print("\n" + "=" * 60)
    print("实验5.4-2 总结")
    print("=" * 60)
    print("1. 结构对偶（算子层面，步骤4）：prox（min硬选择→众数/MAP方向）")
    print("   与 D_ε（softmin软选择→均值/MMSE方向）同为 y-c·∇(·) 的一步，残差都刻画边缘。")
    print("2. 采样层面（步骤3）：TV-ULA 与 PnP-ULA 都注入 √(2δ) 噪声，T=1，")
    print("   输出都是后验均值(MMSE)；区别在先验表达能力，不在温度。")
    print("3. 温度对偶（步骤5）：把噪声幅度改为 √(2Tδ)，T=0 确定性收敛到众数(MAP)，")
    print("   T=1 采样得到均值(MMSE)——MAP 是零温度的 MMSE。")
    print(f"4. 性能：观测 {psnr_y:.2f} dB → TV-ULA {psnr_tv:.2f} dB → PnP-ULA {psnr_pnp:.2f} dB。")
    if psnr_pnp > psnr_tv:
        print(f"   本例学习先验({psnr_pnp:.2f})强于手工TV先验({psnr_tv:.2f})，")
        print("   与TV先验的分片常数(卡通化)表达上限一致；β为人为标定，换成其他图像需重调。")
    else:
        print("   本例 TV 先验不弱于学习先验；可检查去噪器适用域与迭代预算。")
    print("5. 说明：ε 是 PnP 先验的平滑参数（教材5.5），不要求等于观测噪声 σ，")
    print("   不存在'固定噪声水平与低噪观测错配'的问题。")

    # ============================================================
    # 保存数值结果
    # ============================================================
    def _temp_entry(store):
        out = {}
        for T in sorted(store.keys()):
            r = store[T]
            img = r['mean'] if T > 0 else r['final']
            out[f"T={T}"] = {
                "迭代次数": r['niter'], "烧入": r['burnin'],
                "报告量": "后验均值" if T > 0 else "最终样本(众数)",
                "PSNR_dB": round(float(PSNR(x, img)), 4),
                "平均标准差": round(float(r['std'].mean()), 6),
            }
        return out

    results_summary = {
        "实验名称": "实验5.4-2 近端算子 vs 学习去噪器：结构对偶性验证",
        "运行模式": "QUICK冒烟（数值不代表性能）" if _QUICK else "正式",
        "环境信息": {
            "GPU可用": bool(_has_gpu), "device": str(device),
            "sampling_tools可用": bool(_has_sampling_tools),
            "预训练模型可用": bool(_has_model),
            "图像尺寸": list(im.shape),
        },
        "实验参数": {
            "BSNR_db": float(BSNRdb),
            "噪声标准差_sigma": round(float(sigma), 8),
            "似然Lipschitz_Ly": round(L_y, 2),
            "TV先验权重_beta": beta_tv,
            "TV平滑参数_gamma": gamma_tv,
            "TV近端参数_gammaBeta": prox_lam,
            "TV_ULA步长_delta": float(f"{delta_tv:.8g}"),
            "TV_ULA迭代次数": TV_NITER,
            "TV_ULA烧入": TV_BURNIN,
            "PnP平滑参数_eps": round(eps, 8),
            "PnP_ULA步长_delta": float(f"{delta_pnp:.8g}"),
            "PnP_ULA迭代次数": PNP_NITER,
            "PnP_ULA烧入": PNP_BURNIN,
            "Chambolle热启动内迭代": WARM_INNER,
            "热启动实现一致性_max偏差": round(_warm_impl_err, 8),
            "链轨迹prox_rms误差": round(_warm_traj_rms, 8),
        },
        "重建质量": {
            "含噪观测_PSNR_dB": round(psnr_y, 4),
            "TV_ULA后验均值_PSNR_dB": round(psnr_tv, 4),
            "PnP_ULA后验均值_PSNR_dB": round(psnr_pnp, 4),
            "TV_ULA平均标准差": round(float(tv_res['std'].mean()), 6),
            "PnP_ULA平均标准差": round(float(pnp_res['std'].mean()), 6),
        },
        "收敛诊断": {
            "TV_ULA_PSNR轨迹": tv_res['traj'],
            "PnP_ULA_PSNR轨迹": pnp_res['traj'],
            "TV_ULA后1/4段提升_dB": convergence_gain(tv_res['traj']),
            "PnP_ULA后1/4段提升_dB": convergence_gain(pnp_res['traj']),
            "判读": "后1/4段提升接近0表示链已进入平台期",
        },
        "温度扫描_TV先验": _temp_entry(temp_tv),
        "温度扫描_PnP先验": _temp_entry(temp_pnp),
    }

    with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

else:
    # 资源不足时仍写一个状态 JSON，避免留下与旧代码相关的误导性结果文件
    results_summary = {
        "实验名称": "实验5.4-2 近端算子 vs 学习去噪器：结构对偶性验证",
        "环境信息": {
            "GPU可用": bool(_has_gpu), "device": str(device),
            "sampling_tools可用": bool(_has_sampling_tools),
            "预训练模型可用": bool(_has_model),
        },
        "实验结果": "未执行（设置 EXP542_ALLOW_CPU=1 可在CPU按正式预算运行，"
                   "或 EXP542_QUICK=1 进行快速流程验证）",
    }
    with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"状态已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
