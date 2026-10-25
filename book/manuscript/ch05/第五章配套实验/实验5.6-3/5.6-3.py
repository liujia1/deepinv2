# -*- coding: utf-8 -*-
"""
实验5.6-3 MAP vs MMSE综合对比
对应章节：5.6 近似理论与收敛保证 -> 5.6.5 MAP vs MMSE + 不确定性量化的实践指导意义
知识点：
  - MAP估计：后验众数，"最可能"的重建（本实验用确定性PnP迭代真实求解）
  - MMSE估计：后验均值，"平均意义下最好"的重建
  - MMSE在MSE/PSNR意义下最优但更平滑；MAP保留更多高频细节（可能带振铃）
  - 不确定性量化的4个应用场景

设计说明（相对旧版的修订）：
  - 旧版用"含噪观测 y"冒充 MAP 基准，实际只演示了"重建优于退化观测"，
    并未真正对比 MAP vs MMSE。本版用确定性 PnP（去掉朗之万噪声项的不动点迭代）
    真实求解平滑后验的 MAP，兑现"两种估计哲学"的教学对比。
  - 旧版步骤6为闭式公式画的"理论示意"曲线。本版改为真实跑多条 ULA 链，
    分别扫描 ε、δ、M 三个旋钮，用实测数据作图（CPU 上较慢，建议 GPU/Colab）。

前置条件 / 依赖（引用实验5.5-1，不内嵌其采样代码）：
  - MMSE(后验均值)、后验样本、真值/观测一律读取实验5.5-1 产出的
    sampling_results.npz；找不到该文件或字段缺失会硬失败并提示先运行5.5-1，
    不使用合成数据兜底——与5.6-1/5.6-2严格同源，可直接横向比较。
  - MAP 求解与步骤6 的扫描链通过实验5.5-1/pnp_ula_common.py 共享的
    build_problem()/run_pnp_ula() 完成（同一正向算子、同一去噪器、同一RNG口径），
    本文件不复制问题构建与采样循环代码；去噪器权重缺失时由共享模块自动定位/下载。
  - 运行顺序：先跑 5.5-1（maxit=10000 主链，GPU/Colab 约十分钟量级）生成 npz，
    再跑本实验；步骤6 共 10 条 2000 步扫描链，建议 GPU/Colab，CPU 很慢。
    MAP 解与每条扫描链均按“参数+采样内核版本”缓存到本实验目录（首次运行建立，
    之后重跑自动跳过；改图/改文字无需再跑链，改采样递推并 bump 内核版本才失效重算，
    仅 IAT/闸门等诊断改动不使缓存失效）。

素材来源：
  - 实验5.5-1/pnp_ula_common.py（正向算子/去噪器/PnP-ULA 设定的唯一事实源）
  - Terris et al. (2022); Laumont et al. (2021)
"""

import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验5.6-3')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)

    # 在Colab中自动创建chinese_font.py
    _chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
    if not os.path.exists(_chinese_font_path):
        print("正在创建中文字体配置模块...")
        _chinese_font_code = '''# -*- coding: utf-8 -*-
import os, sys, platform, warnings, logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available: return font
    import re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']:
            if re.search(pat, f.name.lower()): return f.name
    return None

def setup_chinese_font(save_dir=None):
    if save_dir is None: save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    _cn_font = _find_chinese_font()
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"[Font] 已检测到中文字体: {_cn_font}")
        return _cn_font
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(save_dir, 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            return 'Noto Sans SC'
        try:
            import urllib.request
            urllib.request.urlretrieve(_font_url, _font_file)
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            return 'Noto Sans SC'
        except: pass
    return None

__all__ = ['setup_chinese_font']
'''
        with open(_chinese_font_path, 'w', encoding='utf-8') as f:
            f.write(_chinese_font_code)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    os.makedirs(SAVE_DIR, exist_ok=True)
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到")
# ========================================================

# ============================================================
# 引用实验5.5-1 的唯一共享模块（问题构建/采样/npz加载/上游闸门的唯一事实源
# 在 实验5.5-1/pnp_ula_common.py）。本文件不复制 5.5-1 的问题构建与采样代码。
# ============================================================
_exp551_dir = (os.path.join(_gdrive, '实验5.5-1') if _IN_COLAB
               else os.path.join(os.path.dirname(SAVE_DIR), '实验5.5-1'))
if os.path.isdir(_exp551_dir) and _exp551_dir not in sys.path:
    sys.path.insert(0, _exp551_dir)
# Colab Drive FUSE：目录在 pnp_ula_common.py 上传前被首次 import 探测时，
# FileFinder 会缓存“该目录无此模块”，文件之后到位也不刷新 → 导入前强制失效缓存。
import importlib
importlib.invalidate_caches()
sys.path_importer_cache.pop(_exp551_dir, None)
try:
    import pnp_ula_common   # 需绑定模块名以读取 __file__（缓存版本指纹用）
    from pnp_ula_common import (build_problem, run_pnp_ula,
                                load_sampling_result, upstream_chain_gate,
                                print_gate)
except ImportError as _e:
    raise FileNotFoundError(
        "未找到实验5.5-1 的共享代码 pnp_ula_common.py。\n"
        "请确认完整的“实验5.5-1”目录（含 pnp_ula_common.py、sampling_tools/、cman.png）"
        "与本实验同在“第五章配套实验”下（Colab 为 我的云端硬盘/实验5.5-1/）。\n"
        "并请先【运行实验5.5-1/5.5-1.py】生成 sampling_results.npz 后，再运行本实验。"
    ) from _e

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 结果缓存（checkpoint）
# MAP 解与步骤6 的每条扫描链都很昂贵（3000 步确定性迭代 / 每条 2000 步 ULA）。
# 缓存指纹 = 参数指纹 + 采样内核版本（SAMPLING_KERNEL_VERSION）：
#   - 重跑时指纹命中 → 直接加载、跳过计算（改绘图/解读/文字无需再跑链）；
#   - 修改采样递推（步长/马氏核/噪声/RNG 口径）并在共享模块 bump 内核版本
#     → 指纹变化，自动重新计算；
#   - 仅修改 IAT/闸门等诊断代码 → 内核版本不变，旧缓存继续复用（采样数组与诊断无关）；
#   - 修改扫描参数（SCAN_MAXIT/ε/δ/seed 等）→ 参数指纹变化，仅重算受影响的链；
#   - 手动删除缓存文件或“步骤6_扫描链缓存”目录即可强制重跑。
# ============================================================
import hashlib
import json as _json

REUSE_CACHE = True          # 置 False 可临时关闭缓存、强制全部重算
# 共享模块源码文件（诊断指纹仅用于展示/追溯，不参与缓存命中）
_COMMON_FILE = getattr(pnp_ula_common, '__file__', None) or os.path.join(
    _exp551_dir, 'pnp_ula_common.py')
_COMMON_DIAG_VER = hashlib.md5(
    open(_COMMON_FILE, 'rb').read()).hexdigest()[:10]
# 缓存版本键：优先使用共享模块显式声明的采样内核版本；
# 共享模块过旧、无该常量时回退到全文指纹（退化为旧行为，不会错误复用）
KERNEL_VER = getattr(pnp_ula_common, 'SAMPLING_KERNEL_VERSION', None)
if KERNEL_VER is None:
    KERNEL_VER = _COMMON_DIAG_VER
    print("[Warn] 共享模块未声明 SAMPLING_KERNEL_VERSION（版本过旧），"
          f"缓存回退为全文指纹 {KERNEL_VER}；请上传最新 pnp_ula_common.py")
# 历史缓存别名：采样内核与当前相同、仅诊断层不同的旧版共享模块全文指纹。
# 命中别名缓存时自动迁移（重命名）为当前内核指纹文件名，无需重跑链。
#   47bc93b584 = IAT 截断诊断升级前 Colab 使用的 pnp_ula_common.py
#   0bdaff85d6 = IAT 截断诊断升级版（iat_geyer 三元组/闸门警告），
#                但尚未声明 SAMPLING_KERNEL_VERSION、5.6-3.py 回退全文MD5时产出
_LEGACY_CACHE_VERSIONS = ('0bdaff85d6', '47bc93b584')
SCAN_CACHE_DIR = os.path.join(SAVE_DIR, '步骤6_扫描链缓存')
_legacy_cache_hits = []   # 记录由旧别名迁移复用的缓存文件（写入 JSON，透明可查）


def _cache_fingerprint(kind, params, ver):
    src = _json.dumps(params, sort_keys=True, ensure_ascii=False)
    h = hashlib.md5((kind + '|' + src + '|' + str(ver)).encode('utf-8')).hexdigest()[:16]
    return h, src


def _cache_path(kind, h):
    return (os.path.join(SCAN_CACHE_DIR, h + '.npz') if kind == 'scan'
            else os.path.join(SAVE_DIR, f'cache_{kind}_{h}.npz'))


def cache_load(kind, params):
    """命中返回 {数组名: 数组} 字典（句柄已关闭），否则 None。
    先查当前内核版本，未命中再按历史别名查找并原子迁移（重命名）到新文件名。"""
    if not REUSE_CACHE:
        return None
    h_new, _ = _cache_fingerprint(kind, params, KERNEL_VER)
    path_new = _cache_path(kind, h_new)
    if os.path.exists(path_new):
        hit_path = path_new
    else:
        hit_path = None
        for _old_ver in _LEGACY_CACHE_VERSIONS:
            h_old, _ = _cache_fingerprint(kind, params, _old_ver)
            path_old = _cache_path(kind, h_old)
            if os.path.exists(path_old):
                hit_path = path_old
                break
        if hit_path is None:
            return None
        # 迁移为当前内核文件名（同目录原子重命名；Drive FUSE 失败则原地复用）
        if hit_path != path_new:
            try:
                os.replace(hit_path, path_new)
                _legacy_cache_hits.append(
                    os.path.basename(hit_path) + ' -> ' + os.path.basename(path_new))
                hit_path = path_new
            except OSError:
                _legacy_cache_hits.append(os.path.basename(hit_path) + '（原地复用）')
    try:
        with np.load(hit_path) as z:
            return {k: z[k] for k in z.files}
    except Exception as _ce:
        print(f"[缓存] {os.path.basename(hit_path)} 读取失败({_ce})，改为重新计算")
        return None


def cache_save(kind, params, **arrays):
    h, src = _cache_fingerprint(kind, params, KERNEL_VER)
    path = _cache_path(kind, h)
    if kind == 'scan':
        os.makedirs(SCAN_CACHE_DIR, exist_ok=True)
    arrays['_meta'] = np.array(src)   # 记录参数原文，便于人工核对缓存身份
    np.savez(path, **arrays)
    return path


# ============================================================
# 辅助函数：图像质量指标
# 注意：这是简化版实现，用于相对比较
# NRMSE 与 5.5-1 的 sampling_tools/measures.py 严格同一口径
# （Frobenius 归一化，上游 MI2G 论文官方定义）；PSNR 在 [0,1] 图像上与
# skimage 版等价；仅“全局相似度”是简化的全局 SSIM，非标准局部窗口 SSIM。
# ============================================================
def compute_psnr(x, y):
    """计算PSNR（以 max(x.max(),1.0) 为峰值，[0,1]图像即标准PSNR）"""
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(x.max(), 1.0)
    return 10 * np.log10(max_val ** 2 / mse)

def compute_global_similarity(x, y):
    """
    全局相似度指标（简化版，用于相对比较）
    注意：这不是标准的局部窗口SSIM，仅用于对比不同方法的相对性能。
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = np.mean(x); mu_y = np.mean(y)
    sigma_x = np.var(x); sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim

def compute_nrmse(x, y):
    """NRMSE（Frobenius 归一化）：||x-y||_F / ||x||_F。

    与 5.5-1 sampling_tools/measures.py 的 NRMSE 严格同一公式（上游 MI2G
    官方口径），分母是整幅图信号能量 RMS=sqrt(mean(x^2))（含均值能量），
    而非去均值标准差 std(x)；故 MMSE 行结果与 5.5-1 JSON 的 NRMSE 完全一致、
    可直接互证。输入为 numpy 数组（measures.NRMSE 收 torch tensor，故此处
    保留 numpy 自包含实现，公式逐字对齐，勿改成 RMSE/std 口径）。
    """
    xn = np.asarray(x, dtype=np.float64)
    yn = np.asarray(y, dtype=np.float64)
    denom = np.linalg.norm(xn, 'fro')
    if denom == 0:
        return float('inf')
    return float(np.linalg.norm(xn - yn, 'fro') / denom)


# ============================================================
# 步骤1：引用 5.5-1 的后验样本，并共享其问题构建（不内嵌采样代码）
# ============================================================
print("=" * 60)
print("步骤1：加载5.5-1后验样本并构建同一去模糊问题")
print("=" * 60)

# 1) MMSE/后验样本/真值/观测一律读 5.5-1 的 sampling_results.npz：
#    找不到或必需字段缺失即硬失败（Missing551DataError），消息中明确要求
#    先运行实验5.5-1；不用合成数据兜底，保证与5.6-1/5.6-2严格同源。
data, _data_path = load_sampling_result(SAVE_DIR)
print(f"[Data] 从 {_data_path} 加载5.5-1采样结果（MMSE/样本与5.6-1/5.6-2严格同源）")
x_true = np.asarray(data['x_true'])
y_obs = np.asarray(data['y_obs'])
post_mean = np.asarray(data['post_mean'])                    # 全链 Welford 后验均值(MMSE)
post_var = np.maximum(np.asarray(data['post_var']), 0.0)     # 全链 Welford 后验方差
mc_samples = np.asarray(data['mc_samples'])

# 上游链收敛闸门（判定实现与5.6-1/5.6-2共用，改判据只改 pnp_ula_common）
src_status, src_warnings, src_extras = upstream_chain_gate(
    data, _data_path, purpose='MAP-vs-MMSE/不确定性')
print_gate(src_status, src_warnings, 'MAP-vs-MMSE/不确定性对比结论有效')

maxit = src_extras['maxit']
burnin = src_extras['burnin']
thinning_step = src_extras['thinning']
n_samples = len(mc_samples)
psnr_last = src_extras['psnr_last']
last_quarter_gain = src_extras['gain_db']
# 平台判据：末1/4段运行均值 PSNR 提升 <0.2 dB（与5.5-1/5.4-2一致）
_plateau_ok = (last_quarter_gain is not None
               and not math.isnan(last_quarter_gain) and last_quarter_gain < 0.2)

# 2) MAP 求解与步骤6 扫描链需要正向算子/去噪器：由共享模块按 5.5-1 同配置
#    构建（同种子、同参数，prob.y 即 npz 中的 y_obs；权重缺失自动定位/下载）。
prob = build_problem(device=device)
eps0 = prob.eps0
print(f"[Problem] device={prob.device}, sigma={float(prob.sigma):.5f}, "
      f"L_y={float(prob.L_y):.1f}, eps0={eps0:.2e}")
print(f"后验均值形状: {post_mean.shape}, 抽稀样本数量: {n_samples}")
if _plateau_ok:
    print(f"[Sampling] 上游主链末段运行均值PSNR={psnr_last:.2f} dB，"
          f"末1/4段提升 {last_quarter_gain:+.3f} dB -> 已到平台")
else:
    print(f"[Sampling] 上游主链末段运行均值PSNR={psnr_last} dB，末1/4段提升={last_quarter_gain} -> 未确认到平台")
    print("[Warn] 上游5.5-1主链未确认到平台：以下 MMSE/MAP 对比与步骤5不确定性"
          "均可能随链长变化，请勿当作最终结论；请延长5.5-1链长后重跑。")


# ============================================================
# 步骤2：MAP vs MMSE 对比（真实MAP）
# ============================================================
print("\n" + "=" * 60)
print("步骤2：MAP vs MMSE 对比")
print("=" * 60)

# MMSE估计 = 后验均值
x_mmse = post_mean

def pnp_map(maxit=3000, delta_frac=0.99, momentum=0.9, tol=1e-6):
    """确定性PnP不动点迭代（去掉朗之万噪声项）求平滑后验的MAP。
    收敛点满足 ∇f(x) = (α/ε)(D_ε(x)-x) = -α∇g_ε(x)，即 f+α·g_ε 的驻点——
    与PnP-ULA采样的是同一个后验（正向算子/去噪器均取自共享模块构建的 prob，
    即5.5-1同一问题），因此这是与MMSE(后验均值)严格对应的后验众数。
    该去模糊后验非常刚性(L_y~1e5)，纯不动点步长δ~6e-6需上万步，
    故用重球动量(heavy-ball)加速并用相对变化早停。"""
    delta_max_map = 1.0 / (prob.L_net / prob.eps0 + prob.L_y)
    delta = float(delta_max_map) * delta_frac
    X = prob.y.clone()
    vel = torch.zeros_like(X)
    used = maxit
    for k in range(maxit):
        drift = (-delta * prob.gradf(X)
                 + prob.alpha * delta / prob.eps0 * (prob.denoise(X) - X))
        vel = momentum * vel + drift
        X_new = prob.projbox(X + vel)
        if (k + 1) % 25 == 0:
            rel = float((X_new - X).norm() / (X.norm() + 1e-12))
            if rel < tol:
                X = X_new
                used = k + 1
                break
        X = X_new
    if used == maxit:
        print(f"  [Warn] MAP 迭代 {maxit} 步未达 tol={tol:g}（相对变化仍大于容差），"
              f"该点可能不是严格不动点，MAP 指标仅供参考")
    print(f"  [MAP] 确定性PnP收敛：iter={used}, delta={delta:.3e}")
    info = {'iterations': int(used), 'converged': bool(used < maxit),
            'tol': float(tol), 'delta': float(delta)}
    return X.detach().cpu().numpy(), info

MAP_PARAMS = dict(maxit=3000, delta_frac=0.99, momentum=0.9, tol=1e-6)
_map_z = cache_load('map', MAP_PARAMS)
if _map_z is not None:
    x_map = _map_z['x_map']
    # 旧缓存未存收敛元数据时取 None（JSON 中显式写 null 并加注记，不伪装成"已收敛"）
    map_info = {k: (np.asarray(_map_z[k]).item() if k in _map_z else None)
                for k in ('map_used', 'map_converged', 'map_tol', 'map_delta')}
    print("[MAP] 命中磁盘缓存，跳过 3000 步确定性PnP迭代")
    if map_info['map_used'] is None:
        print("[MAP] 旧缓存缺少收敛元数据（新增于本版），JSON 中记为 null；"
              "删除 cache_map_*.npz 重跑可补齐")
else:
    x_map, map_info_raw = pnp_map(**MAP_PARAMS)
    map_info = {'map_used': map_info_raw['iterations'],
                'map_converged': map_info_raw['converged'],
                'map_tol': map_info_raw['tol'],
                'map_delta': map_info_raw['delta']}
    cache_save('map', MAP_PARAMS, x_map=np.asarray(x_map),
               **{k: np.asarray(v) for k, v in map_info.items()})
    print("[MAP] 确定性PnP解已写入缓存，下次重跑将直接加载")
map_label = 'MAP (确定性PnP)'
print("[Info] 使用确定性PnP迭代真实求解MAP（后验众数）")

# 计算质量指标
psnr_mmse = compute_psnr(x_true, x_mmse)
psnr_map = compute_psnr(x_true, x_map)
sim_mmse = compute_global_similarity(x_true, x_mmse)
sim_map = compute_global_similarity(x_true, x_map)
nrmse_mmse = compute_nrmse(x_true, x_mmse)
nrmse_map = compute_nrmse(x_true, x_map)

print("\nMMSE估计（后验均值）:")
print(f"  PSNR: {psnr_mmse:.2f} dB")
print(f"  全局相似度: {sim_mmse:.4f} (全局近似，非标准SSIM)")
print(f"  NRMSE: {nrmse_mmse:.4f}  (Frobenius口径 ||e||_F/||x||_F，与5.5-1同一口径)")
print(f"\n{map_label}:")
print(f"  PSNR: {psnr_map:.2f} dB")
print(f"  全局相似度: {sim_map:.4f} (全局近似，非标准SSIM)")
print(f"  NRMSE: {nrmse_map:.4f}  (Frobenius口径，同上)")

print("\n[教学点·一般原理] MMSE是MSE最优估计（Bayes风险意义下PSNR上界），"
      "\n             且自带后验方差=不确定性；MAP是后验众数，通常边缘更锐利。")
print("[本例实测] 该去模糊后验较集中，MAP(众数)与收敛链的后验均值PSNR可能很接近，"
      "\n             MAP 在单例上甚至可能略高——这不违反MMSE最优性(那是Bayes期望意义)；")
print("             MMSE的额外价值主要在于它同时给出不确定性(见步骤5)，而非单纯更平滑。")
print("             若主链未到平台（见上方[Warn]），二者差距会随链长继续变化。")

# 高频能量对比（量化"MAP更锐利/MMSE更平滑"，用梯度幅值均值）
def _grad_energy(img):
    gy, gx = np.gradient(img)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
hf_mmse = _grad_energy(x_mmse)
hf_map = _grad_energy(x_map)
print(f"[高频能量] MMSE={hf_mmse:.5f}, {map_label}={hf_map:.5f}")
# 本例实测方向可能与“MAP通常更锐利”的一般趋势相反：梯度能量同时包含真实细节
# 与噪声/振铃，高不等于更真实，必须如实点明而不是只给数字。
if hf_map > hf_mmse * 1.02:
    print("  本例 MAP 梯度能量更高，与“MAP 通常边缘更锐利”的一般趋势一致")
elif hf_mmse > hf_map * 1.02:
    print("  本例 MMSE 梯度能量反而略高（差异约"
          f"{(hf_mmse / hf_map - 1) * 100:.0f}%，二者接近）：“MAP更锐利”是一般趋势而非定律；")
    print("  梯度能量同时含噪声/振铃成分，高不等于细节更真实，请结合步骤3局部放大图判断")
else:
    print("  两者梯度能量几乎相同（差异<2%），与“该后验较集中、MAP≈MMSE”一致")


# ============================================================
# 步骤3：可视化对比
# ============================================================
print("\n" + "=" * 60)
print("步骤3：可视化对比")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(x_true, cmap='gray')
axes[0, 0].set_title('原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_obs, cmap='gray')
axes[0, 1].set_title('含噪+模糊观测 $y$')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_mmse, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f'MMSE估计 (PSNR: {psnr_mmse:.2f}dB)')
axes[0, 2].axis('off')

axes[0, 3].imshow(x_map, cmap='gray', vmin=0, vmax=1)
axes[0, 3].set_title(f'{map_label} (PSNR: {psnr_map:.2f}dB)')
axes[0, 3].axis('off')

error_mmse = np.abs(x_true - x_mmse)
error_map = np.abs(x_true - x_map)
diff = np.abs(x_mmse - x_map)

axes[1, 0].imshow(error_mmse, cmap='hot')
axes[1, 0].set_title('MMSE误差')
axes[1, 0].axis('off')

axes[1, 1].imshow(error_map, cmap='hot')
axes[1, 1].set_title(f'{map_label}误差')
axes[1, 1].axis('off')

axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title('MMSE与MAP的差异')
axes[1, 2].axis('off')

# 局部放大对比（MMSE vs MAP）
h, w = x_true.shape
zoom_size = min(32, h // 2, w // 2)
zoom_region = (h // 2 - zoom_size // 2, h // 2 + zoom_size // 2,
               w // 2 - zoom_size // 2, w // 2 + zoom_size // 2)
axes[1, 3].axis('off')
ax_mmse_zoom = axes[1, 3].inset_axes([0, 0, 0.48, 1])
ax_map_zoom = axes[1, 3].inset_axes([0.52, 0, 0.48, 1])
ax_mmse_zoom.imshow(x_mmse[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]], cmap='gray')
ax_mmse_zoom.set_title('MMSE', fontsize=10)
ax_mmse_zoom.axis('off')
ax_map_zoom.imshow(x_map[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]], cmap='gray')
ax_map_zoom.set_title('MAP', fontsize=10)
ax_map_zoom.axis('off')
axes[1, 3].set_title('局部放大对比', fontsize=11)

fig.suptitle('实验5.6-3 MMSE vs MAP 对比',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_MAP_vs_MMSE对比.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 步骤4：两种估计哲学的对比
# ============================================================
print("\n" + "=" * 60)
print("步骤4：两种估计哲学的对比")
print("=" * 60)

print("\nMAP估计（第3-5章优化路径）:")
print("  x̂_MAP = argmax_x log p(x|y) = argmin_x {f(x) + g(x)}")
print("  - 给出'最可能的'重建——后验分布的众数")
print("  - 本实验用确定性PnP（去噪声项的不动点迭代）真实求解")
print("  - 保留更多高频细节，但可能放大噪声/振铃，PSNR通常不占优")

print("\nMMSE估计（第5章采样路径）:")
print("  x̂_MMSE = E[x|y] = ∫ x·p(x|y) dx")
print("  - 给出'平均意义下最好'的重建——后验分布的均值")
print("  - 是估计+不确定性：均值是点估计，方差是置信度")
print("  - MSE最优 => PSNR通常≥MAP，视觉上更平滑")

print("\n关键洞察:")
print("  后验均值天然平滑多峰不确定性；MAP取峰值、MMSE平均所有峰。")
print("  一般地（后验宽/多峰时）MMSE更平滑、MAP边缘更锐利；但二者高低关系")
print("  依赖具体后验与采样是否收敛——本例的实测PSNR/高频能量见步骤2输出。")


# ============================================================
# 步骤5：不确定性量化的实践指导意义
# ============================================================
print("\n" + "=" * 60)
print("步骤5：不确定性量化的实践指导意义")
print("=" * 60)

post_std = np.sqrt(post_var)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(x_mmse, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('MMSE重建结果')
axes[0].axis('off')

im1 = axes[1].imshow(post_std, cmap='hot')
axes[1].set_title('不确定性（后验标准差）')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

uncertainty_norm = (post_std - post_std.min()) / (post_std.max() - post_std.min() + 1e-10)
overlay = np.zeros((*x_mmse.shape, 3))
overlay[:, :, 0] = x_mmse
overlay[:, :, 1] = x_mmse
overlay[:, :, 2] = x_mmse
threshold = np.percentile(uncertainty_norm, 90)
high_uncertainty_mask = uncertainty_norm > threshold
overlay[high_uncertainty_mask, 0] = 1.0
overlay[high_uncertainty_mask, 1] = 0.3
overlay[high_uncertainty_mask, 2] = 0.3

axes[2].imshow(overlay)
axes[2].set_title(f'不确定性叠加（红色=高不确定区域，占{np.mean(high_uncertainty_mask)*100:.1f}%）')
axes[2].axis('off')

fig.suptitle('高不确定区域自动标记（应用示意：医学影像中可提示人工复核）', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤5_实践指导意义.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n应用场景说明:")
print("  医学影像：高不确定性区域（红色）应标记为'需医生复核'")
print("  遥感图像：不确定性图指导采样策略——高不确定区域增加采样")
print("  科学计算：不确定性量化是实验结果可信度的核心指标")
print("  主动学习：选择高不确定性样本进行主动标注或重新测量")


# ============================================================
# 步骤6：参数敏感性分析（真实ULA扫描）
# ============================================================
print("\n" + "=" * 60)
print("步骤6：参数敏感性分析")
print("=" * 60)

SCAN_MAXIT = 2000        # 每条扫描链预算；CPU 上请勿随意调小——过短的扫描链未收敛，结果不可信
SCAN_BURN_FRAC = 0.25    # 扫描链统一烧入比例（δ 小时链混合慢，烧入比例取大一些）
_eps_factors = [0.5, 1.0, 2.0, 4.0, 8.0]
eps_test = np.array([eps0 * f for f in _eps_factors])
M_test = np.array([10, 20, 50, 100])
delta_test = np.array([0.1, 0.3, 0.5, 0.7, 0.99])

print(f"[Info] 真实扫描模式：{len(eps_test) + len(delta_test)} 条 ULA 链，每条 {SCAN_MAXIT} 步，"
        "测量 ε/δ/M 的影响（GPU 建议，CPU 很慢）...")
print(f"[Info] 扫描链缓存：{SCAN_CACHE_DIR}")
print(f"       缓存开启={REUSE_CACHE}，采样内核版本={KERNEL_VER}"
        f"（共享模块诊断指纹={_COMMON_DIAG_VER}，仅诊断升级不使缓存失效）；"
        "命中自动跳过，删目录可强制重跑")

def run_scan(eps, delta_frac, maxit, n_samples, seed=0, burn_frac=SCAN_BURN_FRAC, tag=''):
    """通过共享模块运行一条扫描PnP-ULA链（采样循环唯一实现见
    pnp_ula_common.run_pnp_ula，本文件不复制）。
    返回 (粗化样本数组, 像素均值轨迹, 末1/4段运行均值PSNR提升dB)。
    末1/4段提升用于判断扫描链是否已到平台：未收敛链上的"方差/稳定性"对比会
    混入瞬态成分，必须显式警告，而不是直接画进图里当作稳态结论。
    结果按参数指纹+采样内核版本缓存：命中则直接加载，不重跑链。
    旧版全文指纹缓存（采样内核相同、仅诊断不同）由 cache_load 自动迁移复用。"""
    _sparams = dict(eps=float(eps), delta_frac=float(delta_frac),
                    maxit=int(maxit), burnin=int(maxit * burn_frac),
                    n_samples=int(n_samples), seed=int(seed),
                    burn_frac=float(burn_frac), trace_every=25)
    _z = cache_load('scan', _sparams)
    if _z is not None:
        print(f"  [扫描缓存] 命中 {tag}（{maxit} 步链），直接加载")
        return _z['samples'], np.asarray(_z['trace'], dtype=np.float64), float(_z['gain'])
    res = run_pnp_ula(prob, maxit=maxit, burnin=int(maxit * burn_frac),
                      n_samples=n_samples, seed=seed, eps=float(eps),
                      delta_frac=float(delta_frac), desc=f'扫描链 {tag}',
                      want_fourier=False, want_metric_curves=False,
                      trace_every=25, progress=True,
                      trace_fn=lambda X: float(X.mean()))
    _s = res['mc_samples']
    _tr = np.asarray(res['trace_extra'], dtype=np.float64)
    _gain = float(res['last_quarter_gain'])
    cache_save('scan', _sparams, samples=_s, trace=_tr, gain=np.array(_gain))
    return _s, _tr, _gain

scan_warnings = []

# --- 实验1：ε 对后验方差的影响 ---
var_vs_epsilon = []
eps_gains = []
for j, e in enumerate(eps_test):
    s, _, gain = run_scan(float(e), 0.99, SCAN_MAXIT, 60,
                          seed=100 + j, tag=f'eps_x{_eps_factors[j]:g}')
    var_vs_epsilon.append(float(np.mean(np.var(s, axis=0, ddof=1))))
    eps_gains.append(gain)
    print(f"  ε={e:.2e} -> 平均后验方差={var_vs_epsilon[-1]:.3e}, 末1/4段PSNR提升={gain:+.3f} dB")
    if not math.isnan(gain) and gain >= 0.2:
        scan_warnings.append(f'ε={e:.2e} 扫描链未到平台（末1/4段提升{gain:.2f} dB≥0.2），该点方差含瞬态成分')
var_vs_epsilon = np.array(var_vs_epsilon)

# --- 实验2：M 对方差估计稳定性的影响（复用主链抽稀样本做无放回bootstrap）---
# 直接复用 5.5-1 npz 中的抽稀样本，不另跑短链造假样本。样本是否近似独立
# 必须用 npz 里的实测全链 IAT 判定，不能口头假设（见 _m_independence 注记）。
_m_iat = src_extras['iat']
_m_thin = src_extras['thinning']
if _m_iat is None:
    _m_independent = None
    _m_indep_note = ("抽稀样本独立性无法核查（npz 无 trace_fid，旧版5.5-1产物）；"
                     "1/√M 仅作为独立样本的理想参照，实测衰减仅供定性参考")
elif _m_thin is not None and _m_thin >= _m_iat:
    _m_independent = True
    _m_indep_note = (f"抽稀间隔 {_m_thin} 步 ≥ 全链IAT {_m_iat:.0f} 步，"
                     "样本可近似视为独立，实测SE衰减可与1/√M 参考线定量比较")
else:
    _m_independent = False
    _m_indep_note = (f"抽稀间隔 {_m_thin} 步 < 全链IAT {_m_iat:.0f} 步：{len(mc_samples)}个样本仍强自相关"
                     f"（全链ESS仅{src_extras['ess']:.0f}），bootstrap 子样本不是独立样本组；"
                     "1/√M 只是独立样本的理想衰减率，实测曲线比参考线平正是自相关使"
                     "有效样本量增长慢于 M 的直接体现，不能宣称吻合1/√M规律")
print(f"[M实验] {_m_indep_note}")

long_samples = mc_samples
n_long = len(long_samples)
rng = np.random.default_rng(0)
std_error_vs_M = []
m_used = []
m_warnings = []
for M in M_test:
    M_use = int(min(M, n_long))
    m_used.append(M_use)
    if M_use >= n_long:
        # 无放回抽取只能取全部 n_long 个样本：60 次重采样的子样本集合唯一，
        # bootstrap 变异恒为 0（有限总体效应），不是“M 增大使估计更稳定”。
        # 该点记为缺失，不连线、不参与 1/√M 趋势比较，只在图上以空心点示意。
        std_error_vs_M.append(np.nan)
        m_warnings.append(
            f"M={M} 达到/超过可用抽稀样本数 {n_long}：无放回子样本只能取全部 {n_long} 个，"
            "bootstrap 标准误差人为趋零（有限总体效应），该点按缺失处理，不参与M趋势比较")
        print(f"  M={M}(实际{M_use}) -> 无放回取完全部 {n_long} 个样本，bootstrap 变异为 0"
              "（有限总体效应），按缺失处理，不参与趋势")
        continue
    est = []
    for _ in range(60):
        idx = rng.choice(n_long, size=M_use, replace=False)
        est.append(float(np.mean(np.var(long_samples[idx], axis=0, ddof=1))))
    std_error_vs_M.append(float(np.std(est, ddof=1)))
    print(f"  M={M_use} -> 方差估计标准误差={std_error_vs_M[-1]:.3e}")
std_error_vs_M = np.array(std_error_vs_M)
# 1/√M 参考线（以首个 M 点标定）：仅在样本独立时才是应遵循的理论衰减率
se_ref_slope = std_error_vs_M[0] * math.sqrt(float(M_test[0]))

# --- 实验3：δ 对估计稳定性的影响（用链自相关估计有效样本数）---
var_stability = []
delta_ess = []
delta_gains = []
for j, df in enumerate(delta_test):
    _, tr, gain = run_scan(prob.eps0, float(df), SCAN_MAXIT, 120,
                           seed=200 + j, tag=f'delta_x{df:g}')
    delta_gains.append(gain)
    tr = tr - tr.mean()
    if len(tr) > 2:
        rho = float(np.corrcoef(tr[:-1], tr[1:])[0, 1])
        rho = min(max(rho, 0.0), 0.999)
        ess = len(tr) * (1 - rho) / (1 + rho)  # 有效样本数（lag-1 AR(1)近似）
    else:
        ess = len(tr)
    delta_ess.append(ess)
    se = float(np.std(tr, ddof=1) / (math.sqrt(max(ess, 1.0)) + 1e-12))
    var_stability.append(se)
    print(f"  δ_frac={df:g} -> ESS≈{ess:.1f}(lag-1近似), 均值标准误差={se:.3e}, "
          f"末1/4段PSNR提升={gain:+.3f} dB")
    if not math.isnan(gain) and gain >= 0.2:
        scan_warnings.append(f'δ_frac={df:g} 扫描链未到平台（末1/4段提升{gain:.2f} dB≥0.2），该点SE含瞬态成分')
var_stability = np.array(var_stability)

base_var = float(np.mean(post_var))
if scan_warnings:
    print("[Warn] 部分扫描链未到平台（详见 JSON 步骤6.扫描警告）：")
    for _w in scan_warnings:
        print("   - " + _w)
    print("       对应曲线只能定性看趋势；如要定量结论请增大 SCAN_MAXIT。")

# --- 绘图 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(eps_test, var_vs_epsilon, 'o-', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('去噪器噪声水平 ε')
axes[0].set_ylabel('平均后验方差')
axes[0].set_title(f'ε对后验方差的影响（实测，每链{SCAN_MAXIT}步）')
axes[0].grid(True, alpha=0.3)
axes[0].annotate('ε↑ → 先验约束弱 → 方差↑', xy=(0.7, 0.7), xycoords='axes fraction',
                 fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

_m_label = ('近似独立' if _m_independent is True
            else ('独立性无法核查' if _m_independent is None else '仍强自相关'))
_m_valid = ~np.isnan(std_error_vs_M)
axes[1].plot(M_test[_m_valid], std_error_vs_M[_m_valid], 's-', color='#A23B72',
             linewidth=2, markersize=8, label='主链抽稀样本bootstrap')
# 越界点（M≥可用样本数）：bootstrap 变异人为为 0，画在最小有效值下方作空心示意，
# 不与有效点连线，避免被误读为“M 再增大稳定性还能垂直提升”
if (~_m_valid).any():
    _floor = float(np.nanmin(std_error_vs_M)) * 0.35
    axes[1].plot(M_test[~_m_valid], np.full(int((~_m_valid).sum()), _floor),
                 marker='o', linestyle='None', markersize=11,
                 markerfacecolor='none', markeredgecolor='#A23B72',
                 markeredgewidth=1.8, label='M≥可用样本数(人为为0,不参与趋势)')
axes[1].plot(M_test, se_ref_slope / np.sqrt(M_test), 'k--', linewidth=1.2, alpha=0.6, label='1/√M 参考(独立样本理想率)')
axes[1].set_xlabel(f'样本数 M（取自10000步主链的抽稀样本；{_m_label}）')
axes[1].set_ylabel('方差估计标准误差')
axes[1].set_title('M对估计稳定性的影响（实测）')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8, loc='upper right')
_m_annotate = ('M↑ → 估计更稳定' if _m_independent is not False
               else '样本强自相关：\n实测衰减慢于1/√M')
axes[1].annotate(_m_annotate, xy=(0.62, 0.72), xycoords='axes fraction',
                 fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[2].plot(delta_test, var_stability, '^-', color='#F18F01', linewidth=2, markersize=8)
# Colab 的 Noto Sans SC 缺 δ(U+03B4) 字形（ε 可正常显示），用 mathtext 渲染避免丢字
axes[2].set_xlabel(r'步长比例 $\delta/\delta_{\max}$')
axes[2].set_ylabel('均值估计标准误差')
axes[2].set_title(f'$\\delta$对估计稳定性的影响（实测，每链{SCAN_MAXIT}步，ESS为lag-1近似）')
axes[2].grid(True, alpha=0.3)
axes[2].annotate(r'$\delta$过小 → 有效样本少 → 不稳定', xy=(0.25, 0.8), xycoords='axes fraction',
                 fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('关键参数对不确定性估计的影响（真实ULA扫描实测）', fontsize=14, y=1.02)
_foot_notes = []
if scan_warnings:
    _foot_notes.append(f'{len(scan_warnings)} 个扫描点未到平台（见控制台/JSON），相应趋势含瞬态成分，仅作定性参考')
if _m_independent is False:
    _foot_notes.append('M实验样本强自相关（抽稀间隔<全链IAT），1/√M仅为理想参照，不构成定量吻合')
elif _m_independent is None:
    _foot_notes.append('M实验样本独立性无法核查（旧版npz），1/√M对比仅作定性参考')
if m_warnings:
    _foot_notes.append('空心点：M≥可用抽稀样本数，无放回bootstrap变异人为为0（有限总体效应），不参与趋势')
if _foot_notes:
    fig.text(0.5, 0.01, '注：' + '；'.join(_foot_notes),
             ha='center', fontsize=9, color='firebrick')
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(SAVE_DIR, '步骤6_参数敏感性.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n参数敏感性实测结论（以上方每次扫描打印的实测值为准）:")
print("  1. ε: 理论预期 ε越大→先验越弱→后验方差越大（实测见上）")
if _m_independent is True:
    print("  2. M: 实测标准误差衰减可与1/√M 参考线定量比较（抽稀间隔≥全链IAT，样本近似独立）")
elif _m_independent is None:
    print("  2. M: 1/√M 仅为独立样本理想参照；样本独立性无法核查（旧版npz），实测只作定性参考")
else:
    print("  2. M: 实测衰减慢于1/√M 理想率——抽稀间隔<全链IAT、样本强自相关，"
          "增大M带来的有效样本量增长慢于名义M，不能宣称吻合1/√M")
for _mw in m_warnings:
    print(f"     [注意] {_mw}")
print("  3. δ: δ越小→链相关越强、ESS越小→均值估计越不稳定（ESS为lag-1 AR(1)近似）")
print("  [注意] DnCNN为单噪声水平去噪器，此处ε作为先验温度旋钮使用；")
print("         实测曲线可能与理论方向不完全一致，这本身也是有价值的观察。")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.6-3 总结")
print("=" * 60)
print("1. MAP估计: 后验众数（确定性PnP+动量求解），与ULA采样同一后验，通常边缘更锐利")
print("2. MMSE估计: 后验均值，Bayes风险意义下MSE最优，且额外提供后验方差=不确定性")
print("3. 二者PSNR/高频的具体高低关系依赖后验形状与采样是否收敛（见步骤2实测与步骤3局部放大）")
_psnr_last_txt = f"{psnr_last:.2f}" if psnr_last is not None else "不可核查（旧版npz）"
_gain_txt = (f"{last_quarter_gain:+.3f}"
             if last_quarter_gain is not None and not math.isnan(last_quarter_gain)
             else "不可核查（旧版npz）")
print(f"4. 上游5.5-1主链 {maxit} 步：末段运行均值PSNR={_psnr_last_txt} dB，"
      f"末1/4段提升 {_gain_txt} dB（{'已到平台' if _plateau_ok else '未到平台'}）")
print("5. 不确定性量化在医学、遥感、科学计算、主动学习中有重要应用")
_s6_tail = (f"{len(scan_warnings)} 个扫描点未到平台" if scan_warnings else "全部扫描点已到平台")
print(f"6. 步骤6为真实ULA扫描：{len(eps_test) + len(delta_test)} 条各 {SCAN_MAXIT} 步链，{_s6_tail}")
print("\n来源: 实验5.5-1/pnp_ula_common.py; Terris et al. (2022); Laumont et al. (2021)")


# ===== 保存数值结果 =====
import json

def _to_native(obj):
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

def _round_or_none(v, nd=4):
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except TypeError:
        pass
    return round(float(v), nd)


results_summary = {
    "实验名称": "实验5.6-3 MAP vs MMSE综合对比",
    "数据来源": "实验5.5-1 的 sampling_results.npz（真实 PnP-ULA 采样；MMSE/样本与5.6-1/5.6-2严格同源）",
    "上游链状态_5.5-1": {**src_status, "警告": src_warnings},
    "数据信息": {
        "数据文件": _data_path,
        "后验均值形状": list(post_mean.shape),
        "抽稀样本数量": int(len(mc_samples)),
        "device": str(device),
        "主链配置（5.5-1，npz读取）": {
            "maxit": (int(maxit) if maxit is not None else None),
            "burnin": (int(burnin) if burnin is not None else None),
            "抽稀样本数": int(n_samples),
            "抽稀间隔": (int(thinning_step) if thinning_step is not None else None),
            "末段运行均值PSNR_dB": _round_or_none(psnr_last),
            "末1/4段提升_dB": _round_or_none(last_quarter_gain),
            "平台判据_末1/4段提升小于0.2dB": bool(_plateau_ok),
        },
        "说明": "MMSE/后验样本读自5.5-1的npz；MAP与步骤6扫描链由pnp_ula_common共享构建/采样",
    },
    "步骤2_MAP_vs_MMSE对比": {
        "MMSE估计_后验均值": {
            "PSNR_dB": round(float(psnr_mmse), 4),
            "全局相似度": round(float(sim_mmse), 6),
            "NRMSE": round(float(nrmse_mmse), 6),
            "高频能量": round(hf_mmse, 6),
        },
        "MAP估计_确定性PnP": {
            "PSNR_dB": round(float(psnr_map), 4),
            "全局相似度": round(float(sim_map), 6),
            "NRMSE": round(float(nrmse_map), 6),
            "高频能量": round(hf_map, 6),
            "MAP收敛信息": {
                "最大迭代": int(MAP_PARAMS['maxit']),
                "实际迭代": (None if map_info['map_used'] is None
                             else int(map_info['map_used'])),
                "是否达tol收敛": (None if map_info['map_converged'] is None
                                  else bool(map_info['map_converged'])),
                "tol": (None if map_info['map_tol'] is None
                        else float(map_info['map_tol'])),
                "步长delta": (None if map_info['map_delta'] is None
                              else float(map_info['map_delta'])),
                "说明": ("本字段由本版新增；若命中旧缓存则记为 null（删 cache_map_*.npz "
                         "重跑可补齐）。是否收敛不改变已算出的 MAP 图像，只用于判断"
                         "该点是否为严格不动点"),
            },
        },
        "指标口径说明": (
            "NRMSE = ||x_true-x_hat||_F / ||x_true||_F（Frobenius 归一化），"
            "与5.5-1 sampling_tools/measures.py 及5.5-1 JSON 同一口径，"
            "故MMSE行NRMSE与5.5-1的0.03144一致、可直接互证；"
            "全局相似度为简化的全局SSIM（非skimage标准局部窗口SSIM），勿与5.5-1的0.9191直接比"),
    },
    "步骤5_不确定性量化": {
        "后验标准差_平均值": round(float(np.mean(post_std)), 6),
        "后验标准差_最大值": round(float(np.max(post_std)), 6),
        "高不确定性区域占比_百分比": round(float(np.mean(high_uncertainty_mask) * 100), 4),
    },
    "步骤6_参数敏感性": {
        "模式": "真实ULA扫描实测",
        "每条扫描链预算": int(SCAN_MAXIT),
        "扫描链烧入比例": SCAN_BURN_FRAC,
        "结果缓存": {
            "开启": bool(REUSE_CACHE),
            "扫描链缓存目录": SCAN_CACHE_DIR,
            "采样内核版本": KERNEL_VER,
            "共享模块诊断指纹": _COMMON_DIAG_VER,
            "旧别名缓存迁移记录": _legacy_cache_hits,
            "说明": "按参数指纹+采样内核版本缓存；改参数或采样递推（bump SAMPLING_KERNEL_VERSION）自动重算；"
                    "仅诊断/IAT/闸门改动不使缓存失效；删除缓存目录强制全部重跑"
        },
        "M实验说明": "复用5.5-1 npz抽稀样本无放回bootstrap，60次重复；" + _m_indep_note,
        "基准平均方差": round(base_var, 10),
        "epsilon测试值": [float(v) for v in eps_test],
        "epsilon对应平均后验方差": [float(v) for v in var_vs_epsilon],
        "epsilon扫描链末1/4段PSNR提升_dB": [None if math.isnan(g) else round(float(g), 4) for g in eps_gains],
        "样本数测试值": [int(v) for v in M_test],
        "样本数测试值_实际使用样本数": m_used,
        "样本数对应方差估计标准误差": [None if (isinstance(v, float) and math.isnan(v)) else float(v)
                                  for v in std_error_vs_M],
        "M实验有限总体警告": m_warnings,
        "步长比例测试值": [float(v) for v in delta_test],
        "步长对应均值估计标准误差": [float(v) for v in var_stability],
        "步长对应ESS_lag1近似": [float(v) for v in delta_ess],
        "步长扫描链末1/4段PSNR提升_dB": [None if math.isnan(g) else round(float(g), 4) for g in delta_gains],
        "扫描警告": scan_warnings,
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
