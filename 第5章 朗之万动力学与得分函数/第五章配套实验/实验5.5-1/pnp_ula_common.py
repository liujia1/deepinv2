# -*- coding: utf-8 -*-
"""
PnP-ULA 共享模块（唯一事实源）
================================
实验5.5-1 与实验5.6-1 / 5.6-2 / 5.6-3 共同引用本模块，禁止在各实验脚本中
复制去模糊问题构建、PnP-ULA 采样、npz 加载、上游收敛闸门等代码——
5.5-1 的算法一旦调整，只改本文件，所有下游实验自动跟随。

对外提供：
  - build_problem()       构建 cman 去模糊问题（图像/前向算子/噪声/去噪器/步长）
  - run_pnp_ula()         运行一条 PnP-ULA 链（主链或参数扫描链），返回全部统计量
  - iat_geyer()           Geyer 初始正序列 IAT/ESS
  - npz_candidate_paths() / load_sampling_result()
                          统一查找并加载 5.5-1 产出的 sampling_results.npz
  - upstream_chain_gate() 从 npz 诊断上游链收敛性，返回状态与警告

数据契约（sampling_results.npz 必需键）：
  post_mean, post_var, mc_samples, x_true, y_obs
  劈半/诊断键（post_mean_h1/h2、post_var_h1/h2、trace_psnr、trace_fid、
  chain_config）缺失时不报错，但闸门会如实给出"无法核查"警告。
"""

import os
import sys
import math
import time
import types

import numpy as np
import torch

# 本模块所在目录即“实验5.5-1”目录（sampling_tools/、cman.png、权重均在此或其兄弟目录）
HERE = os.path.dirname(os.path.abspath(__file__))
_GDRIVE = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules
_EXP551_NAME = '实验5.5-1'
_NPZ_NAME = 'sampling_results.npz'
_MODEL_REL = os.path.join('Pretrained_models', 'RealSN_DnCNN_noise5.pth')
_MODEL_URL = ("https://github.com/uclaopt/Provable_Plug_and_Play/"
              "raw/master/Pretrained_models/RealSN_DnCNN_noise5.pth")
_REQUIRED_KEYS = ('post_mean', 'post_var', 'mc_samples', 'x_true', 'y_obs')

# 保证本目录在 sys.path 上，sampling_tools 可直接导入
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from tqdm.auto import tqdm
except ImportError:  # 无 tqdm 时退化为普通迭代
    def tqdm(iterable, **kwargs):
        return iterable


class Missing551DataError(FileNotFoundError):
    """找不到实验5.5-1 的数据/共享资源时抛出，消息中明确提示先运行 5.5-1。"""


def _missing_data_msg(searched, extra=''):
    msg = (
        "未找到实验5.5-1 的采样结果 sampling_results.npz。\n"
        "请先【运行实验5.5-1/5.5-1.py】生成该数据（GPU/Colab 约十分钟量级），再运行本实验。\n"
        "本实验不内嵌采样代码、也不使用合成数据兜底，以保证分析结果与5.5-1严格同源。\n"
        f"已查找路径: {searched}"
    )
    if extra:
        msg += '\n' + extra
    return msg


# ============================================================
# 资源定位：sampling_tools / 模型权重 / cman
# ============================================================
def _import_sampling_tools():
    """导入 sampling_tools 中的必要组件；缺失时给出指向 5.5-1 的明确报错。"""
    try:
        from sampling_tools.blur_operators import blur_operators
        from sampling_tools.load_model import DnCNN, load_model
        from sampling_tools.welford import welford
        from sampling_tools.measures import NRMSE, PSNR, SSIM
    except ImportError as e:
        raise Missing551DataError(
            f"无法从实验5.5-1 目录导入 sampling_tools（{e}）。\n"
            f"请确认完整的“{_EXP551_NAME}”目录（含 sampling_tools/）与本实验同在“第五章配套实验”下，"
            "或在 Colab 上已挂载云盘。") from e
    return blur_operators, DnCNN, load_model, welford, NRMSE, PSNR, SSIM


def find_model_path(auto_download=True):
    """按 5.5-1 本目录 → 兄弟实验目录 → 自动下载的顺序定位去噪器权重。"""
    sibling_root = os.path.dirname(HERE)
    candidates = [
        os.path.join(HERE, _MODEL_REL),
        os.path.join(sibling_root, '实验5.4-2', _MODEL_REL),
        os.path.join(sibling_root, '实验5.4-1', _MODEL_REL),
        os.path.join(sibling_root, '实验5.3-1', _MODEL_REL),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    if not auto_download:
        raise Missing551DataError(
            f"未找到去噪器权重 RealSN_DnCNN_noise5.pth（已查找 {candidates}）。\n"
            f"请先运行实验5.5-1（首次运行会自动下载权重），或手动放置到 {_EXP551_NAME}/Pretrained_models/。")
    target = os.path.join(HERE, _MODEL_REL)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f"[Model] 未找到预训练权重，正在自动下载到 {target} ...")
    import urllib.request
    try:
        urllib.request.urlretrieve(_MODEL_URL, target)
    except Exception as e:
        raise Missing551DataError(
            f"去噪器权重自动下载失败（{e}）。请手动下载 {_MODEL_URL}\n"
            f"并保存为 {target}，或先运行实验5.5-1。") from e
    return target


def find_cman_path():
    """定位 cman.png（仅 5.5-1 目录）。"""
    p = os.path.join(HERE, 'cman.png')
    if os.path.exists(p):
        return p
    raise Missing551DataError(
        f"未找到 {_EXP551_NAME}/cman.png。该图像是全链统一输入，请确认完整的“{_EXP551_NAME}”目录存在。")


# ============================================================
# IAT / ESS（Geyer 初始正序列，与实验5.2-1同实现）
# ============================================================
def iat_geyer(vals):
    """标量序列的 IAT 与 ESS：IAT = 1 + 2*sum rho(k)，ESS = n / IAT。"""
    x = np.asarray(vals, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 0 or n < 4:
        return 1.0, float(n)
    max_lag = min(1000, n // 3)
    acf = np.array([np.dot(x[:-k], x[k:]) / ((n - k) * var)
                    for k in range(1, max_lag + 1)])
    iat = 1.0
    k = 0
    while k + 1 < len(acf):
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        iat += 2.0 * pair
        k += 2
    return float(iat), float(n / iat)


# ============================================================
# 问题构建：cman + uniform 5×5 模糊 + BSNR=40dB 高斯噪声 + RealSN DnCNN
# 注意 RNG 顺序与旧版 5.5-1 逐行一致（先 y 加噪、后构造网络），
# 同种子下生成的 y、链随机数流与旧版完全相同，数值结果可复现。
# ============================================================
def build_problem(device=None, model_path=None, seed=42, bsnr_db=40.0,
                  kernel=(5, 5), blur_type='uniform'):
    blur_operators, DnCNN, load_model, welford, NRMSE, PSNR, SSIM = _import_sampling_tools()
    from PIL import Image

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    torch.manual_seed(seed)

    cman_path = find_cman_path()
    im = np.array(Image.open(cman_path))
    x = torch.Tensor(im / 255.).to(device)

    A, AT, AAT_norm = blur_operators(list(kernel), [im.shape[0], im.shape[1]], blur_type, device)
    y0 = A(x)
    sigma = (torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro')
             / math.sqrt(torch.numel(x) * 10 ** (bsnr_db / 10.0)))
    y = y0 + sigma * torch.randn_like(x)

    def gradf(z):
        return AT(A(z) - y) / sigma ** 2

    def f_fid(z):
        return (torch.linalg.matrix_norm(y - A(z), ord='fro') ** 2.0) / (2.0 * sigma ** 2)

    L_y = AAT_norm / (sigma ** 2)
    eps0 = (5.0 / 255.0) ** 2
    L_net = 1.0
    alpha = 1.0

    C_hi = torch.tensor(1.0, device=device)
    C_lo = torch.tensor(0.0, device=device)

    def projbox(z):
        return torch.clamp(z, min=C_lo, max=C_hi)

    if model_path is None:
        model_path = find_model_path()
    net = load_model(model_path, device)

    def denoise(z):
        return (z - net(z[None][None].to(device))[0][0]).detach()

    return types.SimpleNamespace(
        device=device, x=x, y=y, A=A, AT=AT, sigma=sigma, L_y=L_y,
        eps0=eps0, L_net=L_net, alpha=alpha, bsnr_db=bsnr_db,
        gradf=gradf, f_fid=f_fid, projbox=projbox, denoise=denoise,
        cman_path=cman_path, model_path=model_path, image_shape=im.shape,
        NRMSE=NRMSE, PSNR=PSNR, SSIM=SSIM, welford=welford,
    )


# ============================================================
# PnP-ULA 采样（5.5-1 主链与 5.6-3 参数扫描链共用同一份实现）
# ============================================================
def run_pnp_ula(prob, maxit=10000, burnin=None, n_samples=100, seed=None,
                delta_frac=0.99, eps=None, desc='PnP-ULA采样',
                trace_every=1, want_fourier=True, want_metric_curves=True,
                progress=True, trace_fn=None):
    """
    运行一条 projected PnP-ULA 链。

    trace_fn：可选的标量轨迹钩子 X(torch)->float，烧入后逐步记录到
    trace_extra（如下游实验需要链样本的像素均值轨迹）；不给则为空数组。

    返回 dict（torch 张量与 numpy 数组并存，供绘图/落盘直接使用）：
      maxit/burnin/half_mark/thinning_step, x, y,
      post_meanvar/post_meanvar_h1/post_meanvar_h2（welford 对象）,
      absfouriercoeff（want_fourier=True 时）,
      post_mean/post_var（及 h1/h2）numpy、mc_samples、
      trace_fid、trace_psnr（按 trace_every 抽点）、trace_extra、
      nrmse_values/psnr_values/ssim_values（want_metric_curves=True，逐点）、
      iat_fid/ess_fid/last_quarter_gain/half_sigma_corr/elapsed
    """
    if burnin is None:
        burnin = int(maxit * 0.05)
    if eps is None:
        eps = prob.eps0
    delta_max = 1.0 / (prob.L_net / eps + prob.L_y)
    delta = float(delta_max) * delta_frac
    half_mark = burnin + (maxit - burnin) // 2
    thinning_step = max(1, int(maxit / n_samples))

    if seed is not None:
        torch.manual_seed(seed)

    def markov_kernel(X):
        return prob.projbox(
            X - delta * prob.gradf(X)
            + prob.alpha * delta / eps * (prob.denoise(X) - X)
            + math.sqrt(2.0 * delta) * torch.randn_like(X))

    X = prob.y.clone()
    MC_X = []
    trace_fid = []
    trace_psnr = []
    trace_extra = []
    nrmse_values, psnr_values, ssim_values = [], [], []
    post_meanvar = post_meanvar_h1 = post_meanvar_h2 = None
    absfouriercoeff = None
    count = 0
    start = time.time()

    iterator = tqdm(range(maxit), desc=desc, mininterval=2.0) if progress else range(maxit)
    for i in iterator:
        X = markov_kernel(X)
        if i == burnin:
            post_meanvar = prob.welford(X)
            post_meanvar_h1 = prob.welford(X)
            if want_fourier:
                absfouriercoeff = prob.welford(torch.fft.fft2(X).abs())
            count = 0
        elif i > burnin:
            post_meanvar.update(X)
            if want_fourier:
                absfouriercoeff.update(torch.fft.fft2(X).abs())
            if i < half_mark:
                post_meanvar_h1.update(X)
            elif i == half_mark:
                post_meanvar_h2 = prob.welford(X)
            else:
                post_meanvar_h2.update(X)

            trace_fid.append(float(prob.f_fid(X).detach().cpu()))
            if trace_fn is not None:
                trace_extra.append(float(trace_fn(X)))
            cur_mean = post_meanvar.get_mean()
            if want_metric_curves:
                nrmse_values.append(prob.NRMSE(prob.x, cur_mean))
                psnr_values.append(prob.PSNR(prob.x, cur_mean))
                ssim_values.append(prob.SSIM(prob.x, cur_mean))
            elif (i - burnin) % trace_every == 0:
                trace_psnr.append([i, float(prob.PSNR(prob.x, cur_mean))])

            if count == thinning_step - 1:
                MC_X.append(X.detach().cpu().numpy())
                count = 0
            else:
                count += 1

    elapsed = time.time() - start
    n_post = len(trace_fid)

    # 收敛诊断（与 5.5-1 旧版同口径）
    iat_fid, ess_fid = iat_geyer(trace_fid)
    if want_metric_curves:
        last_quarter_gain = float(psnr_values[-1] - psnr_values[int(0.75 * len(psnr_values))])
        psnr_last = float(psnr_values[-1])
    else:
        if len(trace_psnr) >= 4:
            _qi = int(0.75 * len(trace_psnr))
            last_quarter_gain = float(trace_psnr[-1][1] - trace_psnr[_qi][1])
            psnr_last = float(trace_psnr[-1][1])
        else:
            last_quarter_gain = float('nan')
            psnr_last = float('nan')
    half_std1 = np.sqrt(post_meanvar_h1.get_var().detach().cpu().numpy())
    half_std2 = np.sqrt(post_meanvar_h2.get_var().detach().cpu().numpy())
    half_sigma_corr = float(np.corrcoef(half_std1.flatten(), half_std2.flatten())[0, 1])

    return {
        'maxit': int(maxit), 'burnin': int(burnin), 'half_mark': int(half_mark),
        'thinning_step': int(thinning_step), 'delta': delta, 'delta_max': float(delta_max),
        'eps': float(eps), 'x': prob.x, 'y': prob.y,
        'post_meanvar': post_meanvar, 'post_meanvar_h1': post_meanvar_h1,
        'post_meanvar_h2': post_meanvar_h2, 'absfouriercoeff': absfouriercoeff,
        'post_mean': post_meanvar.get_mean().detach().cpu().numpy(),
        'post_var': post_meanvar.get_var().detach().cpu().numpy(),
        'post_mean_h1': post_meanvar_h1.get_mean().detach().cpu().numpy(),
        'post_var_h1': post_meanvar_h1.get_var().detach().cpu().numpy(),
        'post_mean_h2': post_meanvar_h2.get_mean().detach().cpu().numpy(),
        'post_var_h2': post_meanvar_h2.get_var().detach().cpu().numpy(),
        'mc_samples': np.array(MC_X),
        'trace_fid': np.array(trace_fid, dtype=np.float64),
        'trace_psnr': (np.array(psnr_values, dtype=np.float64)
                       if want_metric_curves else np.array([p for _, p in trace_psnr])),
        'trace_psnr_points': (None if want_metric_curves else list(trace_psnr)),
        'trace_extra': np.array(trace_extra, dtype=np.float64),
        'nrmse_values': nrmse_values, 'psnr_values': psnr_values, 'ssim_values': ssim_values,
        'n_post': int(n_post), 'iat_fid': iat_fid, 'ess_fid': ess_fid,
        'last_quarter_gain': last_quarter_gain, 'psnr_last': psnr_last,
        'half_sigma_corr': half_sigma_corr, 'elapsed': elapsed,
    }


# ============================================================
# sampling_results.npz 的统一查找/加载
# ============================================================
def npz_candidate_paths(consumer_save_dir):
    """下游实验查找 npz 的路径顺序：本目录 → 兄弟实验5.5-1 → Colab 云盘。"""
    paths = [
        os.path.join(consumer_save_dir, _NPZ_NAME),
        os.path.join(os.path.dirname(consumer_save_dir), _EXP551_NAME, _NPZ_NAME),
    ]
    if _IN_COLAB:
        paths.insert(0, os.path.join(_GDRIVE, _EXP551_NAME, _NPZ_NAME))
    return paths


def load_sampling_result(consumer_save_dir):
    """
    查找并加载 5.5-1 的 sampling_results.npz；找不到或缺必需键即硬失败，
    返回 (np.lib.npyio.NpzFile, 命中的绝对路径)。
    """
    searched = npz_candidate_paths(consumer_save_dir)
    data_path = next((p for p in searched if os.path.exists(p)), None)
    if data_path is None:
        raise Missing551DataError(_missing_data_msg(searched))
    data = np.load(data_path)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise Missing551DataError(
            f"{data_path} 缺少必需字段 {missing}，不是实验5.5-1 当前版本的完整产物。\n"
            "请重新运行实验5.5-1生成 sampling_results.npz 后再运行本实验。")
    return data, os.path.abspath(data_path)


# ============================================================
# 上游链（5.5-1）收敛闸门：5.6-1/5.6-2/5.6-3 共用同一份判定
# ============================================================
def upstream_chain_gate(data, data_path, purpose='下游分析'):
    """
    返回 (status:dict, warnings:list, extras:dict)。
    purpose 用于定制警告中“哪类结论可能随链长变化”的措辞。
    """
    warnings = []
    status = {'数据文件': os.path.abspath(data_path)}
    extras = {'n_post_full': None, 'gain_db': None, 'iat': None, 'ess': None,
              'thinning': None, 'maxit': None, 'burnin': None, 'half_mark': None,
              'psnr_last': None}

    if 'chain_config' in data:
        cfg = [int(v) for v in data['chain_config'].tolist()]
        extras['maxit'], extras['burnin'], extras['half_mark'], extras['thinning'] = cfg
        status['maxit'] = cfg[0]
        status['burnin'] = cfg[1]
        status['劈半点'] = cfg[2]
        status['抽稀间隔'] = cfg[3]

    if 'trace_psnr' in data:
        tp = data['trace_psnr'].astype(np.float64)
        n_post = int(len(tp))
        gain = float(tp[-1] - tp[int(0.75 * n_post)])
        extras.update(n_post_full=n_post, gain_db=gain, psnr_last=round(float(tp[-1]), 4))
        status['post_burnin步数'] = n_post
        status['末段PSNR'] = round(float(tp[-1]), 4)
        status['末1/4段PSNR提升_dB'] = round(gain, 4)
        if gain >= 0.2:
            warnings.append(
                f"上游链末1/4段运行均值PSNR仍提升 {gain:.3f} dB(≥0.2)，未确认到平台；"
                f"本实验的{purpose}结论可能随链长变化，请延长5.5-1链长后重跑")
    else:
        warnings.append("npz 中无 trace_psnr（旧版5.5-1产物），无法核查上游链是否到平台")

    if 'trace_fid' in data:
        iat, ess = iat_geyer(data['trace_fid'].astype(np.float64))
        extras.update(iat=iat, ess=ess)
        status['全链IAT_步'] = round(iat, 2)
        status['全链ESS'] = round(ess, 1)
        thin = extras['thinning'] or 0
        status['抽稀间隔_ge_IAT'] = bool(thin >= iat) if thin else None
        if thin and thin < iat:
            warnings.append(
                f"抽稀间隔 {thin} 步 < 全链IAT {iat:.1f}：mc_samples 仍有自相关，"
                "凡使用样本独立性的下游统计需谨慎；全链Welford量不受影响")
    else:
        warnings.append("npz 中无 trace_fid（旧版5.5-1产物），无法复核全链IAT/ESS与抽稀充分性")

    return status, warnings, extras


def print_gate(status, warnings, ok_tail):
    """统一打印闸门核查块；ok_tail 为无警告时的结论尾句。"""
    print("\n[上游链状态] 实验5.5-1 链的收敛核查：")
    for k, v in status.items():
        print(f"  {k}: {v}")
    if warnings:
        for w in warnings:
            print(f"  [警告] {w}")
    else:
        print(f"  上游链末1/4段已到平台（<0.2 dB），{ok_tail}")
