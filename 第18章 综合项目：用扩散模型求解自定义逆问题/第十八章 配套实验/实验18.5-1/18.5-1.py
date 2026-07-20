# -*- coding: utf-8 -*-
"""
实验18.5-1 不确定性量化与后验采样
对应知识点：18.5节（不确定性量化：从点估计到分布推断）

实验内容：
Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
Step 2: 后验采样实现 —— PnP-ULA与扩散采样
Step 3: 不确定性地图计算 —— 像素级标准差、经验分位数区间
Step 4: ★ 校准检验 —— 覆盖率与可靠性分析
Step 5: ★ 样本数对不确定性估计的影响

★原创设计：
- Step 4的校准检验：验证经验分位数区间覆盖率（⚠️ 样本数较少时结果仅供参考）
- Step 5的样本数对比：量化样本数对不确定性估计的影响
- 不确定性地图按问题类型(去模糊/超分/修复)分类解读
- 后验样本数 S 在脚本运行时打印, 校准阈值与总结文字均基于实际S动态计算

素材来源：18.5节后验采样代码、deepinv sampling API
运行前提：需GPU（Colab T4即可），需下载预训练模型(DRUNet/DiffUNet)
"""

import os, sys, time, pickle, hashlib
import numpy as np
import torch
# 设置非交互式后端（必须在 import matplotlib.pyplot 之前）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import tqdm as _tqdm_module   # ★ 补上: patch 逻辑需要的是 tqdm 模块对象, 而非 tqdm 类

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验18.5-1')
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
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

# ★ 缓存配置
use_cache = True
cache_file = os.path.join(SAVE_DIR, 'experiment_cache.pkl')
print(f"缓存配置: use_cache={use_cache}")

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cpu':
    print("[警告] 扩散采样在CPU上会非常慢，强烈建议使用GPU")

# ★ 全局噪声标准差（观测噪声水平 sigma_data）
# 含义：物理算子 GaussianNoise、DataFid_L2/ULA step_size、DPS_L2 三处统一引用。
# 与去噪器噪声 sigma_denoiser=2/255 物理含义不同：sigma_data 是 y 中加性高斯噪声的标准差，
# sigma_denoiser 是 ScorePrior 内部 Tweedie 公式所假设的去噪器训练噪声水平。
sigma_data = 0.01
sigma_denoiser = 2.0 / 255.0
print(f"噪声标准差: sigma_data={sigma_data} (观测), sigma_denoiser={sigma_denoiser:.5f} (去噪器)")

# 安装deepinv
try:
    import deepinv as dinv  # ★ 统一使用 as dinv 别名
except ImportError:
    import subprocess
    print("正在安装 deepinv ...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/deepinv/deepinv.git#egg=deepinv'
        ])
        # ★ 安装成功后重新导入
        import deepinv as dinv
        print(f"deepinv 安装成功，版本: {dinv.__version__}")
    except subprocess.CalledProcessError as e:
        print(f"[错误] deepinv 安装失败，请手动安装: pip install git+https://github.com/deepinv/deepinv.git")
        sys.exit(1)
else:
    print(f"deepinv 版本: {dinv.__version__}")

from deepinv.physics import Blur, Inpainting, GaussianNoise
from deepinv.utils import load_example


# ========================================================================
# 辅助函数
# ========================================================================
def compute_psnr(img1, img2):
    """计算PSNR (dB)。

    注意: 本函数假设图像动态范围为[0,1]，但不执行clamp操作。
    理由:
    1. PSNR应反映真实重建误差，而非人为限制范围后的误差
    2. 后验采样结果可能超出[0,1]（如迭代去噪过程中），clamp会低估真实误差
    3. 可视化时统一clamp(0,1)是为了显示，但PSNR计算应保持原始数值精度

    参数:
        img1, img2: 4D张量 (B, C, H, W)，假设值域[0,1]但允许超出
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)



# ========================================================================
# Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
# 对应18.5节知识点：点估计的局限性与后验分布的意义
# ========================================================================
print("\n" + "="*70)
print("Step 1: 从点估计到分布")
print("="*70)

print("""
点估计的局限性:
  单次重建 x^ = f(y) 只给出一个解，无法量化可靠性
  对于欠定逆问题（如50%像素缺失的修复——本实验实际配置），不同先验
  可能给出完全不同的解

后验分布的价值:
  p(x|y) 描述了所有与观测一致的解的概率分布
  多次采样 {x^(s)} ~ p(x|y) 可量化:
  - 均值 E[x|y]: 最优点估计
  - 标准差 std[x|y]: 像素级不确定性
  - 分位数: 置信区间
""")

# 加载测试图像
x_true = load_example("celeba_example.jpg", img_size=(256, 256), resize_mode='resize')
# 确保是4D张量 (B, C, H, W)
if x_true.ndim == 3:
    x_true = x_true.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
elif x_true.ndim == 5:
    x_true = x_true.squeeze(0) if x_true.shape[0] == 1 else x_true[0]
x_true = x_true.to(device)

# 创建退化模型（修复场景，50%缺失——欠定程度适中）
torch.manual_seed(42)
physics_inp = Inpainting(img_size=(3, 256, 256), mask=0.5, device=device)
physics_inp.set_noise_model(GaussianNoise(sigma=sigma_data))
y_inp = physics_inp(x_true)

print(f"真值 shape: {x_true.shape}")
print(f"观测 shape: {y_inp.shape}")
print(f"观测 PSNR: {compute_psnr(x_true, y_inp):.2f} dB")


# ========================================================================
# Step 2: 后验采样实现
# 对应18.5节知识点：PnP-ULA与扩散采样
# ========================================================================
print("\n" + "="*70)
print("Step 2: 后验采样实现")
print("="*70)

print("""
★ 自动降级策略说明:
  本脚本采用"尝试-失败-降级"策略：
  1. 优先尝试 ULA 采样（最准确的后验采样方法）
  2. 若 ULA 失败，降级到 DPS 扩散采样
  3. 若 DPS 失败，降级到 PnP 近似采样
  4. 若 PnP 失败，降级到伪逆+噪声近似（最粗糙）

  ⚠️ 注意: 这种降级策略是工程容错设计，但会产生"混合样本"：
     - 混合样本的统计同质性较差（不同样本来自不同的后验近似）
     - 混合采样时，统计结果（均值、标准差）可能受方法切换影响
     - 代码会在混合采样时打印警告，并改用按方法分组的可视化

  如果需要严格统计同质性，建议：
  - 固定使用单一方法（删除其他分支）
  - 或增大样本数 S 以降低单方法失败的影响
""")

ula_method = None

S = 8
print(f"后验采样数量: S={S}")

all_samples = []
sample_times = []
# ★ 新增：记录每个样本的来源方法，避免混合采样标签污染
sample_methods = []
# ★ 新增：用于动态生成总结文字的PSNR-vs-S曲线（None表示未计算）
psnr_by_s = None

import traceback

# ★ 缓存超参数指纹（与第4-6章 checkpoint 约定一致）
# 目的：检测"修改 sigma_data/换图/改 S/换设备"后误用旧缓存的隐患。
# 物理算子 mask=0.5 与 Inpainting 构造的 torch.manual_seed(42) 固定，
# 物理算子类型本身由代码写死（Inpainting+50%缺失），因此指纹只需覆盖
# 真正可能变化的维度：观测噪声、图像内容、目标样本数、运行设备。
# 图像内容用整图 MD5(16位) + 统计量(mean/std) 双重把关，避免仅哈希
# 前若干字节的碰撞风险（与第4-6章约定一致：hash + 统计量兜底）。
def _compute_fingerprint():
    """基于当前 (sigma_data, x_true, S, device) 计算缓存指纹字典。"""
    arr = x_true.detach().cpu().contiguous().numpy()
    img_hash = hashlib.md5(arr.tobytes()).hexdigest()[:16]  # 整图 MD5 前 16 位
    return {
        'sigma_data': float(sigma_data),
        'x_true_shape': tuple(arr.shape),
        'x_true_hash': img_hash,
        'x_true_mean': float(arr.mean()),  # 统计量兜底
        'x_true_std': float(arr.std()),    # 统计量兜底
        'S': int(S),
        'device': str(device),
    }

def _compare_fingerprint(stored_fp, current_fp):
    """逐字段对比指纹，返回 (matched: bool, diff_message: str)。"""
    if stored_fp is None:
        return False, "缓存无 fingerprint 字段（疑似旧版缓存）"
    diffs = []
    for k, v_now in current_fp.items():
        v_old = stored_fp.get(k, '<missing>')
        if v_old != v_now:
            diffs.append(f"  - {k}: 缓存={v_old!r}, 当前={v_now!r}")
    if diffs:
        return False, "指纹不匹配:\n" + "\n".join(diffs)
    return True, "指纹一致"

_current_fp = _compute_fingerprint()

cached_samples = []
cached_methods = []  # ★ 修改：支持样本级方法记录
cached_times = []    # ★ 新增：支持样本级时间记录
if use_cache and os.path.exists(cache_file):
    try:
        if os.path.getsize(cache_file) > 0:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # ★ 指纹校验：与第4-6章 checkpoint 约定一致
            stored_fp = cached_data.get('fingerprint')
            fp_matched, fp_msg = _compare_fingerprint(stored_fp, _current_fp)
            if not fp_matched:
                print(f"⚠️ [缓存] {fp_msg}")
                print(f"⚠️ [缓存] 丢弃旧缓存，将基于当前 (sigma_data={sigma_data}, "
                      f"image_hash={_current_fp['x_true_hash']}, S={S}) 重新采样")
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
            elif 'samples' in cached_data and len(cached_data['samples']) > 0:
                cached_samples = cached_data['samples']
                # ★ 兼容旧缓存：如果没有methods字段，使用method字段
                cached_methods = cached_data.get('methods', [])
                if not cached_methods and 'method' in cached_data:
                    cached_methods = [cached_data['method']] * len(cached_samples)
                # ★ 兼容旧缓存：如果没有times字段，补充为空列表
                cached_times = cached_data.get('times', [])
                if cached_samples and not cached_times:
                    # ★ 旧版缓存不含耗时字段，后续平均耗时统计仅包含本次新增样本
                    print(f"⚠️ 缓存为旧版格式（无times字段），平均耗时统计将仅包含本次新增样本")
                print(f"[缓存] 指纹一致 (sigma_data={sigma_data}, image_hash={_current_fp['x_true_hash']}, S={S})")
                print(f"[缓存] 加载了 {len(cached_samples)} 个样本")
                # ★ 打印方法分布
                method_counts = {}
                for m in cached_methods:
                    method_counts[m] = method_counts.get(m, 0) + 1
                print(f"[缓存] 方法分布: {method_counts}")
                # ★ 降级结果警告：缓存样本并非全部来自 ULA(最优方法)
                # 即便 fingerprint 匹配, 也不应静默复用降级结果, 避免学生因
                # 首次偶发失败而错过本可获得的更优样本。
                # 兼容旧缓存: 没有 best_achieved 字段时按方法分布推断
                best_achieved = cached_data.get('best_achieved', None)
                if best_achieved is None:
                    # 旧版缓存: 按 method_counts 推断
                    best_achieved = (list(method_counts.keys()) == ["ULA"])
                if not best_achieved:
                    print(f"⚠️⚠️⚠️ [缓存] 缓存中的样本并非全部来自 ULA (最优方法)")
                    print(f"⚠️⚠️⚠️ [缓存] 方法分布: {method_counts}")
                    print(f"⚠️⚠️⚠️ [缓存] 如本次运行 ULA/DPS 已可正常采样, 建议删除缓存以重新尝试更优方法")
                    print(f"⚠️⚠️⚠️ [缓存]   rm \"{cache_file}\"")
                if len(cached_samples) >= S:
                    print(f"[缓存] 样本数已满足需求，跳过采样")
                    all_samples = cached_samples[:S]
                    sample_methods = cached_methods[:S]
                    sample_times = cached_times[:S] if cached_times else []  # ★ 加载时间记录
                    # ★ 计算最终方法（检查是否混合采样）
                    unique_methods = list(set(sample_methods))
                    if len(unique_methods) == 1:
                        ula_method = unique_methods[0]
                    else:
                        ula_method = "混合采样"
        else:
            print(f"[缓存] 缓存文件为空，将重新采样")
            os.remove(cache_file)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"[缓存] 缓存文件损坏，将重新采样")
        try:
            os.remove(cache_file)
        except:
            pass
    except Exception as e:
        print(f"[缓存] 加载失败: {e}")

def save_samples_to_cache(samples, methods, times=None):
    """保存样本到缓存，包含样本级方法标签、超参数指纹与"是否达到最优方法"标记。

    字段说明:
    - fingerprint: 输入维度指纹（sigma_data/x_true/S/device），与加载侧 _compare_fingerprint 配套
    - best_achieved: bool, 是否所有样本均来自 ULA（最优方法）。
      加载时若为 False, 说明缓存中包含降级结果（DPS/PnP/伪逆）, 即便 fingerprint
      匹配也会打醒目警告，避免学生因首次偶发失败而静默复用质量较差的样本。
    """
    if not use_cache:
        return
    try:
        # ★ best_achieved: 仅当所有样本方法标签都是 "ULA" 时为 True
        # "混合采样" 或任何降级方法都会让 best_achieved = False
        best_achieved = bool(methods) and all(m == "ULA" for m in methods)
        cached_data = {
            'samples': samples,
            'methods': methods,
            'times': times or [],
            'fingerprint': _compute_fingerprint(),  # ★ 与加载侧 _compare_fingerprint 配套
            'best_achieved': best_achieved,
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        quality_tag = "全部ULA(最优)" if best_achieved else "含降级样本"
        print(f"[缓存] 已保存 {len(samples)} 个样本 (含 fingerprint, {quality_tag})")
    except Exception as e:
        print(f"[缓存] 保存失败: {e}")

def release_model(model_var_name):
    """释放全局作用域中的模型变量并清理显存。

    注意：dir()在函数内部只返回局部变量，需用globals()检查全局变量。
    """
    # ★ 修复：使用globals()而非dir()检查全局变量
    if model_var_name in globals():
        del globals()[model_var_name]
        print(f"[显存] 已删除全局变量 {model_var_name}")
    else:
        print(f"[显存] 全局变量 {model_var_name} 不存在，可能已被释放")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[显存] 当前显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

if len(all_samples) < S:
    start_idx = len(all_samples)
    print(f"需要采样 {S - start_idx} 个样本")

    # ★ 修改：使用 cached_methods 和 cached_times 而非 cached_method
    if cached_samples and cached_methods:
        all_samples = cached_samples.copy()
        sample_methods = cached_methods.copy()
        sample_times = cached_times.copy() if cached_times else []  # ★ 加载已有时间记录

    # ========== 2a. ULA 采样 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2a. ULA 后验采样 ---")
        print("="*50)

        denoiser_ula = None
        try:
            from deepinv.models import DRUNet
            print("[ULA] 加载 DRUNet...")
            denoiser_ula = DRUNet(pretrained='download').to(device)
            print(f"[ULA] DRUNet 加载成功，显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        except Exception as e:
            print(f"[ULA] DRUNet 加载失败!")
            print(f"[ULA] 错误类型: {type(e).__name__}")
            print(f"[ULA] 错误信息: {e}")
            print("[ULA] 详细堆栈:")
            traceback.print_exc()

        if denoiser_ula is not None:
            try:
                from deepinv.sampling import ULA
                from deepinv.optim import L2 as DataFid_L2
                from deepinv.optim import ScorePrior

                print("[ULA] 创建 ScorePrior...")
                # ★ sigma_denoiser：去噪器噪声水平（ScorePrior 内部 Tweedie 公式使用）
                #   对应 deepinv 官方示例中的 sigma_denoiser = 2/255，与图像像素值域 [0,1] 匹配
                # ★ sigma_data：观测噪声标准差（已在脚本顶部定义为全局变量），同时用于
                #   DataFid_L2 与 ULA step_size。依据 deepinv 官方 PnP-ULA 教程
                #   （demo_sampling.html）：step_size = 0.01 * sigma_data**2。
                #   官方公式中的 sigma 指数据噪声水平（与 L2 数据保真度同一 sigma），
                #   而非 sigma_denoiser，二者物理含义不同（数据噪声 vs 先验/去噪器噪声），
                #   切勿混用。
                prior_score = ScorePrior(denoiser=denoiser_ula)
                data_fidelity = DataFid_L2(sigma=sigma_data)

                print("[ULA] 创建 ULA 采样器...")
                # ★ 根据deepinv官方示例参数：
                # step_size = 0.01 * sigma_data^2, alpha = 0.9
                # ★ max_iter 区分GPU/CPU:
                #   - GPU: 5000 (接近deepinv官方教程推荐值, 链混合更充分)
                #   - CPU: 1000 (CPU运行ULA极慢, 5000会导致单样本耗时数十分钟)
                # ★ 教学权衡说明:
                #   即便使用5000, 在S=8的有限样本量下, ULA链的自相关仍可能偏高;
                #   本实验核心结论是"用S=8演示UQ可视化+提示S≥30才可靠",
                #   链的绝对混合质量受限于教学时长, 需在更大规模实验中验证。
                # ★★ 步长风险提示 (与第8章 Nelder-Mead 伪收敛同源):
                #   当前 sigma_data=0.01, 代入得 step_size = 0.01 * 1e-4 = 1e-6。
                #   虽符合 deepinv 官方公式形式, 但其官方教程示例的 sigma 通常 ≥0.1,
                #   0.01 这一量级下步长偏小, Langevin 链在 256x256x3 维空间中可能
                #   "看起来收敛但实际未充分探索" (各样本高度相似、不确定性 std 偏小)。
                #   诊断信号: 脚本末尾 sample_psnrs.std() 若远小于 0.5 dB, 提示链
                #   混合不足, 可考虑调大 step_size 系数(0.01→0.1)或 max_iter。
                ula_max_iter = 5000 if device.type == 'cuda' else 1000
                ula = ULA(prior=prior_score, data_fidelity=data_fidelity,
                          max_iter=ula_max_iter, burnin_ratio=0.5, thinning=1,
                          step_size=0.01 * (sigma_data**2), alpha=0.9, sigma=sigma_denoiser,
                          verbose=True)
                print(f"[ULA] max_iter={ula_max_iter} (设备: {device.type})")

                print(f"[ULA] 开始采样...")
                start_idx = len(all_samples)
                pbar = tqdm(range(start_idx, S), desc="[ULA] 采样", unit="样本")
                # ★ 保存原始 tqdm 并 patch 模块级 tqdm（防止 deepinv 内部 ULA 创建嵌套进度条）
                _orig_tqdm_global = _tqdm_module.tqdm
                _tqdm_module.tqdm = lambda *a, **kw: _orig_tqdm_global(*a, **{**kw, 'disable': True})

                for s in pbar:
                    # ★ 每次采样使用不同的随机种子，确保样本多样性
                    torch.manual_seed(s * 1000 + 42)
                    t_start = time.time()
                    ula_result = ula(y_inp, physics_inp)
                    x_sample = ula_result[0] if isinstance(ula_result, tuple) else ula_result
                    t_sample = time.time() - t_start
                    all_samples.append(x_sample.detach().cpu())
                    sample_times.append(t_sample)  # ★ 记录实际耗时
                    sample_methods.append("ULA")  # ★ 记录样本来源方法
                    psnr_val = compute_psnr(x_true, x_sample)
                    pbar.set_postfix({"耗时": f"{t_sample:.1f}s", "PSNR": f"{psnr_val:.1f}dB"})
                    del x_sample, ula_result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    save_samples_to_cache(all_samples, sample_methods, sample_times)

                if sample_times:
                    print(f"[ULA] 采样完成! 平均耗时: {np.mean(sample_times[-(S-start_idx):]):.2f}s")

                # ★ 恢复 tqdm 模块
                _tqdm_module.tqdm = _orig_tqdm_global

            except ImportError as e:
                print(f"[ULA] 导入失败: {e}")
                print("[ULA] 详细堆栈:")
                traceback.print_exc()
            except Exception as e:
                print(f"[ULA] 采样失败!")
                print(f"[ULA] 错误类型: {type(e).__name__}")
                print(f"[ULA] 错误信息: {e}")
                print("[ULA] 详细堆栈:")
                traceback.print_exc()
            finally:
                release_model('denoiser_ula')

    # ========== 2b. DPS 采样 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2b. DPS 扩散采样 ---")
        print("="*50)

        denoiser_dps = None
        try:
            from deepinv.models import DiffUNet
            print("[DPS] 加载 DiffUNet...")
            denoiser_dps = DiffUNet(pretrained='download').to(device)
            print(f"[DPS] DiffUNet 加载成功，显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        except Exception as e:
            print(f"[DPS] DiffUNet 加载失败!")
            print(f"[DPS] 错误类型: {type(e).__name__}")
            print(f"[DPS] 错误信息: {e}")
            print("[DPS] 详细堆栈:")
            traceback.print_exc()

        if denoiser_dps is not None:
            try:
                from deepinv.sampling import DPS

                print("[DPS] 创建 DPS 采样器...")
                n_dps_iter = 100
                # ★ 修复：与 ULA 保持一致, 显式传入 sigma=sigma_data 匹配观测噪声水平
                #   （L2 默认 sigma=1.0 会与 physics_inp 的 GaussianNoise(sigma=sigma_data) 严重失配）
                # ★★ 与 ULA 中 sigma_data 语义的细微差异说明 (供学生参考, 避免误类比):
                #   ULA:  Langevin 步长公式  step_size = 0.01 * sigma_data**2 中的 sigma_data
                #         严格对应"数据保真项 L2 的噪声水平", 也即 y_obs = A(x)+n, n~N(0, sigma_data^2 I),
                #         deepinv 官方 PnP-ULA 教程 (demo_sampling.html) 明确该 sigma 即为物理观测噪声。
                #   DPS:  Diffusion Posterior Sampling (Chung et al., 2022) 原文 Eq.(9) 给出的
                #         guidance scale 形式为  ∇_x ||y - A(x_hat_0)||^2 / (2 * rho^2),
                #         其中 rho 才是"y 中的噪声水平" (与 ULA 中 sigma_data 同物理含义)。
                #         但 deepinv 的 DPS 实现 (deepinv/sampling/dps.py) 实际并未直接调用该
                #         原论文公式, 而是将 L2(sigma=...) 整体作为 data_fidelity 传入, 由 DPS
                #         类在每个扩散时间步 t 计算  ||y - A(x_hat_0(x_t, t))||^2, 并按 1/sigma^2
                #         缩放梯度。 因此 deepinv 的 DPS(sigma=sigma_data) 中:
                #         - 物理含义上: sigma 仍表示"观测噪声水平" (与 ULA 的 sigma_data 同义)
                #         - 数值效果上: 与原论文 rho 略有不同 —— deepinv 默认不引入 DPS 原论文
                #           建议的 1/||y - A(x_hat_0)|| 归一化缩放 (DPS 论文推荐按噪声大小做
                #           启发式缩放以适应不同噪声量级), 而是固定使用 1/sigma_data^2 缩放。
                #         简化结论: 在 deepinv 框架下, ULA 与 DPS 传入相同的 sigma=sigma_data
                #         物理含义一致, 但 DPS 内部并未做论文推荐的启发式缩放, 严格意义上
                #         "guidance scale" 不严格等于 1/sigma_data^2。学生在跨方法对比时应注意
                #         这一差异 (尤其是 sigma_data 较大时, deepinv 的 DPS 可能会比论文版本
                #         给出更保守的 likelihood weight)。本实验选择 deepinv 默认实现以保证
                #         API 一致性, 论文原版实现请参考 Chung et al. (2022) 公开代码。
                # ★ 新版 deepinv (>=0.4) DPS API：
                #   - 必填首参是 denoiser（旧版为 model）
                #   - 内部自动构造 DPSDataFidelity，禁止再传 data_fidelity
                #     （会引发 super().__init__ 重复关键字报错）
                #   - 采样步数用 num_steps（旧版为 max_iter）
                #   - 噪声水平由 physics_inp 的 GaussianNoise(sigma=sigma_data) 提供
                #     DPS 在 forward 时按 1/sigma² 缩放似然梯度，由 weight 控制步长 λ
                dps = DPS(
                    denoiser=denoiser_dps,
                    schedule="vp",
                    num_steps=n_dps_iter,
                    weight=1.0,
                    alpha=1.0,
                    verbose=False,
                    device=device,
                )

                print(f"[DPS] 开始采样 (每样本 {n_dps_iter} 步)...")
                start_idx = len(all_samples)
                pbar = tqdm(range(start_idx, S), desc="[DPS] 采样", unit="样本")
                # ★ 保存原始 tqdm 并 patch 模块级 tqdm（防止 deepinv 内部 DPS 创建嵌套进度条）
                _orig_tqdm_global = _tqdm_module.tqdm
                _tqdm_module.tqdm = lambda *a, **kw: _orig_tqdm_global(*a, **{**kw, 'disable': True})
                for s in pbar:
                    # ★ 使用差异更大的种子确保样本多样性
                    torch.manual_seed(s * 1000 + 42)
                    t_start = time.time()
                    dps_result = dps(y_inp, physics_inp, seed=s * 1000 + 42)
                    x_sample = dps_result[0] if isinstance(dps_result, tuple) else dps_result
                    t_sample = time.time() - t_start
                    all_samples.append(x_sample.detach().cpu())
                    sample_times.append(t_sample)  # ★ 记录实际耗时
                    sample_methods.append("DPS")  # ★ 记录样本来源方法
                    psnr_val = compute_psnr(x_true, x_sample)
                    pbar.set_postfix({"耗时": f"{t_sample:.1f}s", "PSNR": f"{psnr_val:.1f}dB"})
                    del x_sample, dps_result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    save_samples_to_cache(all_samples, sample_methods, sample_times)

                if sample_times:
                    print(f"[DPS] 采样完成! 平均耗时: {np.mean(sample_times[-(S-start_idx):]):.2f}s")

                # ★ 恢复 tqdm 模块
                _tqdm_module.tqdm = _orig_tqdm_global

            except ImportError as e:
                print(f"[DPS] 导入失败: {e}")
                print("[DPS] 详细堆栈:")
                traceback.print_exc()
            except Exception as e:
                print(f"[DPS] 采样失败!")
                print(f"[DPS] 错误类型: {type(e).__name__}")
                print(f"[DPS] 错误信息: {e}")
                print("[DPS] 详细堆栈:")
                traceback.print_exc()
            finally:
                release_model('denoiser_dps')

    # ========== 2c. PnP 近似采样 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2c. PnP 近似后验采样 ---")
        print("="*50)

        denoiser_pnp = None
        try:
            from deepinv.models import DRUNet
            print("[PnP] 加载 DRUNet...")
            denoiser_pnp = DRUNet(pretrained='download').to(device)
            print(f"[PnP] DRUNet 加载成功，显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        except Exception as e:
            print(f"[PnP] DRUNet 加载失败!")
            print(f"[PnP] 错误类型: {type(e).__name__}")
            print(f"[PnP] 错误信息: {e}")
            print("[PnP] 详细堆栈:")
            traceback.print_exc()

        if denoiser_pnp is not None:
            try:
                print("[PnP] 开始采样...")
                start_idx = len(all_samples)
                pbar_outer = tqdm(range(start_idx, S), desc="[PnP] 采样", unit="样本")
                for s in pbar_outer:
                    t_start = time.time()
                    # ★ 与 ULA/DPS 一致, 使用 s*1000+42 公式
                    torch.manual_seed(s * 1000 + 42)
                    x_pnp = physics_inp.A_adjoint(y_inp) + 0.05 * torch.randn_like(x_true)

                    n_iter = 20
                    pbar_inner = tqdm(range(n_iter), desc=f"  [PnP] 样本 {s+1} 迭代", unit="步",
                                      leave=False)
                    for it in pbar_inner:
                        with torch.no_grad():
                            grad = physics_inp.A_adjoint(physics_inp.A(x_pnp) - y_inp)
                            x_pnp = x_pnp - 0.5 * grad
                            sigma_cur = max(0.1 * (1 - it / n_iter), 0.01)
                            noise_level = torch.tensor([sigma_cur] * x_pnp.shape[0]).to(device)
                            x_pnp = denoiser_pnp(x_pnp, noise_level)
                            del grad, noise_level
                        if (it + 1) % 5 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    pbar_inner.close()

                    t_sample = time.time() - t_start
                    all_samples.append(x_pnp.detach().cpu())
                    sample_times.append(t_sample)  # ★ 记录实际耗时
                    sample_methods.append("PnP近似")  # ★ 记录样本来源方法
                    del x_pnp
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    save_samples_to_cache(all_samples, sample_methods, sample_times)
                    pbar_outer.set_postfix({"耗时": f"{t_sample:.1f}s"})

                if sample_times:
                    print(f"[PnP] 采样完成! 平均耗时: {np.mean(sample_times[-(S-start_idx):]):.2f}s")

            except Exception as e:
                print(f"[PnP] 采样失败!")
                print(f"[PnP] 错误类型: {type(e).__name__}")
                print(f"[PnP] 错误信息: {e}")
                print("[PnP] 详细堆栈:")
                traceback.print_exc()
            finally:
                release_model('denoiser_pnp')

    # ========== 2d. 伪逆近似 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2d. 伪逆+噪声近似 ---")
        print("="*50)

        start_idx = len(all_samples)
        pbar = tqdm(range(start_idx, S), desc="[伪逆] 采样", unit="样本",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for s in pbar:
            t_start = time.time()
            # ★ 与 ULA/DPS/PnP 一致, 使用 s*1000+42 公式
            torch.manual_seed(s * 1000 + 42)
            x_approx = physics_inp.A_adjoint(y_inp) + 0.02 * torch.randn_like(x_true)
            t_sample = time.time() - t_start
            all_samples.append(x_approx.cpu())
            sample_times.append(t_sample)  # ★ 记录实际耗时(伪逆操作极快)
            sample_methods.append("伪逆+噪声")  # ★ 记录样本来源方法
            # ★ 修复：与 ULA/DPS/PnP 三个分支保持一致, 改为逐样本保存
            #   理由: 伪逆分支虽然极快(无 GPU 显存压力), 但如果未来在该循环
            #   中加入耗时操作(如伪逆后跑一轮快速去噪), 仍需保证中断后能
            #   从缓存中恢复; 同时保持四个分支的代码风格统一, 便于阅读。
            del x_approx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_samples_to_cache(all_samples, sample_methods, sample_times)
            pbar.set_postfix({"耗时": f"{t_sample:.1f}s"})

# ★ 计算最终采样方法（检查是否混合采样）
if len(sample_methods) > 0:
    unique_methods = list(set(sample_methods))
    if len(unique_methods) == 1:
        ula_method = unique_methods[0]
    else:
        ula_method = "混合采样"
        # 计算各方法占比
        method_counts = {}
        for m in sample_methods:
            method_counts[m] = method_counts.get(m, 0) + 1
        print(f"\n⚠️ 警告: 采样为混合方法，统计结果可能受不同后验近似方法的影响")
        print(f"   方法分布: {method_counts}")

print(f"\n最终采样方法: {ula_method}, 样本数: {len(all_samples)}")


# ========================================================================
# ★ 根据S动态推导置信水平（供Step3/4/5/总结统一使用）
# ========================================================================
# 经验分位数在样本中对应整数位置 k=1,2,...,S-1
# 双侧coverage level = 2k/S，候选分位点为 2*k/S (k=1..S//2)
# 过滤掉边界值1.0（对应[0%,100%]退化情况）和0（无意义）
# 候选数≤4时全部保留；候选数>4时按索引均匀采样4个，让校准曲线在可用水平上均匀分布
# 典型S的预期输出（运行下方自检可看到实际结果）:
#   S=8  -> [0.250, 0.500, 0.750]   (候选数=3, 全部保留)
#   S=16 -> [0.125, 0.375, 0.625, 0.875]   (候选数=7, 索引[0,2,4,6])
#   S=30 -> [0.067, 0.333, 0.667, 0.933]   (候选数=14, 索引[0,4,9,13])
#   S=64 -> [0.031, 0.344, 0.656, 0.969]   (候选数=31, 索引[0,10,20,30])
def _derive_confidence_levels(n_samples):
    """根据样本数S动态推导用于校准曲线绘制的经验分位数覆盖率列表。

    推导规则:
    1. 候选集 = {2k/S : k=1..S//2} ∩ (0, 1)   排除退化覆盖率
    2. 候选数 ≤ 4: 全部保留
    3. 候选数 > 4: 在候选索引[0, n-1]上均匀取4个点 (i*(n-1)/3, i=0..3)
    4. 候选数 = 0: 退化为单点中位数0.5 (S=1或S=2)

    关键点: 索引均匀 ≠ 值均匀 (因为候选本身在[2/S, (S-1)/S]上等距,
    所以索引均匀也意味着值近似均匀, 但精度受S的离散性影响)。
    """
    _all_levels = [2 * k / n_samples for k in range(1, n_samples // 2 + 1)
                   if 0 < 2 * k / n_samples < 1]
    if len(_all_levels) == 0:
        return [0.5]  # S=1或S=2的退化情况
    if len(_all_levels) <= 4:
        return _all_levels
    # 候选数>4: 按索引均匀采样4个, 覆盖[0, n-1]的完整范围
    n = len(_all_levels)
    indices = [int(round(i * (n - 1) / 3)) for i in range(4)]
    return sorted({_all_levels[i] for i in indices})

confidence_levels = _derive_confidence_levels(S)
# ★ cl_max: S可靠范围内最高置信水平, 贯穿Step3/4/5/JSON/总结
cl_max = max(confidence_levels)
print(f"[校准] 使用置信水平: {[round(cl, 3) for cl in confidence_levels]}, "
      f"主置信水平 cl_max={cl_max:.1%} (基于S={S}动态推导, 候选数={len([2*k/S for k in range(1, S//2 + 1) if 0 < 2*k/S < 1])})")

# ★ 自检: 打印几个典型S值, 方便读者直接验证推导公式的正确性
# (避免注释里写"算例"但实际跑出来对不上的问题)
print(f"[校准自检] 几个典型S值的confidence_levels推导结果(实际运行, 非手算):")
for _S_test in [4, 8, 16, 30, 64]:
    _cls = _derive_confidence_levels(_S_test)
    print(f"  S={_S_test:3d} -> {[round(c, 3) for c in _cls]}")


# ========================================================================
# Step 3: 不确定性地图计算
# 对应18.5节知识点：像素级标准差与置信区间
# ========================================================================
print("\n" + "="*70)
print("Step 3: 不确定性地图计算")
print("="*70)

# 将样本堆叠为张量
samples_tensor = torch.stack(all_samples, dim=0)  # (S, 1, 3, 256, 256)
print(f"样本张量 shape: {samples_tensor.shape}")

# 计算后验统计量
posterior_mean = samples_tensor.mean(dim=0)  # (1, 3, 256, 256)
posterior_std = samples_tensor.std(dim=0)    # (1, 3, 256, 256)
posterior_var = samples_tensor.var(dim=0)

# ★ 主置信区间（基于动态推导的 cl_max，而非硬编码95%）
# S=8 时 cl_max=0.75（即75%区间）；S≥16 时 cl_max 可能更大
# 校准检验、覆盖率统计、可视化均统一使用此区间，避免硬编码95%在S=8时几乎外推的误导
# ★★ 实现细节: torch.quantile 默认走线性插值（type=7, R 默认），
#   与 _derive_confidence_levels 中"分位数对应整数位置 k"的严格顺序统计量
#   推导在数值上接近但不等价；下文 Step4 覆盖率统计的"理论上应≈cl_max"是
#   渐近成立, S=8 时会有几个百分点的偏差, 不视为算法错误。
q_low_cl = samples_tensor.quantile((1 - cl_max) / 2, dim=0)
q_high_cl = samples_tensor.quantile(1 - (1 - cl_max) / 2, dim=0)
ci_width_cl = q_high_cl - q_low_cl

# 统计量
psnr_mean = compute_psnr(x_true, posterior_mean.to(device))
mean_std = posterior_std.mean().item()
max_std = posterior_std.max().item()
mean_ci_width_cl = ci_width_cl.mean().item()

print(f"后验均值 PSNR: {psnr_mean:.2f} dB")
print(f"平均像素标准差: {mean_std:.4f}")
print(f"最大像素标准差: {max_std:.4f}")
print(f"平均经验分位数区间宽度(cl={cl_max:.0%}, S={S}):  {mean_ci_width_cl:.4f}")
print(f"⚠️ 注意: S={S} 时区间估计不可靠，建议 S≥30")

# 各样本的PSNR分布
sample_psnrs = [compute_psnr(x_true, s.to(device)) for s in all_samples]
print(f"样本PSNR范围: {min(sample_psnrs):.2f} - {max(sample_psnrs):.2f} dB")
print(f"样本PSNR标准差: {np.std(sample_psnrs):.2f} dB")

# 可视化: 后验样本 + 均值 + 不确定性
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 4, figure=fig)

# 第一行: 真值 + 4个后验样本
ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax00.set_title('真值 x', fontsize=12)
ax00.axis('off')

for i in range(min(3, len(all_samples))):
    ax = fig.add_subplot(gs[0, i+1])
    ax.imshow(all_samples[i][0].cpu().permute(1, 2, 0).clamp(0, 1))
    ax.set_title(f'样本 {i+1}\nPSNR={sample_psnrs[i]:.1f}dB', fontsize=10)
    ax.axis('off')

# 第二行: 均值 + 观测 + 标准差地图 + CI宽度
ax10 = fig.add_subplot(gs[1, 0])
ax10.imshow(posterior_mean[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax10.set_title(f'后验均值\nPSNR={psnr_mean:.1f}dB', fontsize=11)
ax10.axis('off')

ax11 = fig.add_subplot(gs[1, 1])
ax11.imshow(y_inp[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax11.set_title('观测 y (50%修复)', fontsize=11)
ax11.axis('off')

ax12 = fig.add_subplot(gs[1, 2])
# 不确定性地图（灰度，越亮越不确定）
std_map = posterior_std[0].cpu().mean(dim=0).numpy()
im12 = ax12.imshow(std_map, cmap='hot', vmin=0, vmax=std_map.max())
ax12.set_title('★ 不确定性地图\n(像素级std)', fontsize=11)
ax12.axis('off')
plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

ax13 = fig.add_subplot(gs[1, 3])
# ★ 修改：主图展示主置信区间（动态cl_max，S=8时为75%）
ci_map_cl = ci_width_cl[0].cpu().mean(dim=0).numpy()
im13 = ax13.imshow(ci_map_cl, cmap='hot', vmin=0, vmax=ci_map_cl.max())
ax13.set_title(f'{cl_max:.0%}经验分位数区间宽度\n(S={S}, 可靠范围)', fontsize=10)
ax13.axis('off')
plt.colorbar(im13, ax=ax13, fraction=0.046, pad=0.04)

# 第三行: 真值vs均值误差 + 覆盖图 + PSNR直方图 + 误差分布
ax20 = fig.add_subplot(gs[2, 0])
error_map = (x_true - posterior_mean.to(device)).abs()[0].cpu().mean(dim=0).numpy()
im20 = ax20.imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
ax20.set_title('重建误差地图', fontsize=11)
ax20.axis('off')
plt.colorbar(im20, ax=ax20, fraction=0.046, pad=0.04)

# ★ 修改：覆盖率基于主置信区间（动态cl_max）
ax21 = fig.add_subplot(gs[2, 1])
in_ci_cl = ((x_true >= q_low_cl.to(device)) & (x_true <= q_high_cl.to(device))).float()
coverage_map = in_ci_cl[0].cpu().mean(dim=0).numpy()
im21 = ax21.imshow(coverage_map, cmap='RdYlGn', vmin=0, vmax=1)
overall_coverage = in_ci_cl.mean().item()
ax21.set_title(f'★ {cl_max:.0%}经验分位数覆盖图\n(S={S}, 可靠范围, 覆盖率={overall_coverage:.1%})', fontsize=10)
ax21.axis('off')
plt.colorbar(im21, ax=ax21, fraction=0.046, pad=0.04)

# PSNR直方图
ax22 = fig.add_subplot(gs[2, 2])
ax22.hist(sample_psnrs, bins=max(5, S//2), color='steelblue', edgecolor='white', alpha=0.8)
ax22.axvline(psnr_mean, color='red', linestyle='--', label=f'均值={psnr_mean:.1f}dB')
ax22.set_xlabel('PSNR (dB)', fontsize=10)
ax22.set_ylabel('频次', fontsize=10)
ax22.set_title('样本PSNR分布', fontsize=11)
ax22.legend(fontsize=9)

# 误差分布
ax23 = fig.add_subplot(gs[2, 3])
errors = (posterior_mean.to(device) - x_true).cpu().numpy().flatten()
ax23.hist(errors, bins=100, density=True, color='steelblue', alpha=0.7)
ax23.axvline(0, color='red', linestyle='--')
ax23.set_xlabel('误差值', fontsize=10)
ax23.set_ylabel('概率密度', fontsize=10)
ax23.set_title('重建误差分布', fontsize=11)

fig.suptitle('Step 1-3: 后验采样与不确定性量化 (50%修复场景)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_3_uncertainty_maps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: step1_3_uncertainty_maps.png")


# ========================================================================
# Step 4: ★ 校准检验
# 对应18.5节知识点：校准检查与覆盖率分析
# ========================================================================
print("\n" + "="*70)
print("Step 4: ★ 校准检验")
print("="*70)

# ★ 共享的校准偏差等级阈值 (Step 4 的逐cl判断与最终总结使用同一套)
#   偏差 < 0.05  → 良好 (✓)
#   偏差 < 0.15  → 存在偏差 (△)
#   偏差 ≥ 0.15  → 严重失配 (✗)
_CALIB_GOOD_THRESH = 0.05
_CALIB_FAIR_THRESH = 0.15

def _calib_grade(dev):
    """根据覆盖率偏差 |coverage - cl| 返回 (等级文字, 标记符号)。"""
    if dev < _CALIB_GOOD_THRESH:
        return "良好", "✓"
    if dev < _CALIB_FAIR_THRESH:
        return "存在偏差", "△"
    return "严重失配", "✗"

print("""
★ 校准检验原理:
  如果后验分布p(x|y)是"正确校准"的，那么:
  - 经验分位数区间应该覆盖对应比例的真值像素
  - 覆盖率 ≈ 名义覆盖率 → 校准良好
  - 覆盖率 > 名义覆盖率 → 过于保守（区间太宽）
  - 覆盖率 < 名义覆盖率 → 过于自信（区间太窄）

  ⚠️ 注意: 本检验是"单图像×像素空间池化"(详见下方说明)
     要获得统计意义严格的95%置信区间校准，需要 S≥30 且多次独立实验
""")

print("""
★ 单图像×像素空间池化 vs 多次独立实验校准:
  本实验的"校准"做法: 在单张真值图像x上, 对每个像素构造经验分位数
  区间, 再把整图256×256×3个像素的覆盖结果pool起来, 统计"被覆盖比例"。
  这与统计教材里"频率学派置信区间覆盖率"(多个独立(x_i, y_i)各自采样、
  各自检查真值是否落入置信区间、统计覆盖频率)有重要区别:
    - 多次独立校准: 各实验独立, 可估计覆盖率方差和置信区间
    - 像素空间池化: 像素高度相关(邻居像素通常共享后验结构),
      pooled coverage的方差估计会失真, 但点估计仍有描述性参考价值
  这是图像UQ文献中的常见做法(逐像素校准曲线),
  但不宜与统计推断意义下的"覆盖率"直接等同。
""")

# ★ confidence_levels 已在 Step3 之前动态推导，此处直接使用
coverages = []

for cl in confidence_levels:
    # ★ 复用: cl_max 在 Step 3 算 overall_coverage 时已做过相同的
    # quantile + 区间内判断, 与本循环体内重新计算结果应逐位一致。
    # 跳过该分支避免重复计算, 防止以后改一处忘改另一处导致的不一致。
    if cl == cl_max:
        coverage = overall_coverage
    else:
        q_low = samples_tensor.quantile((1 - cl) / 2, dim=0).to(device)
        q_high = samples_tensor.quantile(1 - (1 - cl) / 2, dim=0).to(device)
        in_interval = ((x_true >= q_low) & (x_true <= q_high)).float()
        coverage = in_interval.mean().item()
    coverages.append(coverage)
    # ★ 使用共享的三档校准阈值, 与最终总结保持一致
    dev = abs(coverage - cl)
    grade, mark = _calib_grade(dev)
    print(f"  名义覆盖率 {cl:.0%}: 实际覆盖率 = {coverage:.1%}  {mark} 偏差={dev:.1%} ({grade})")

# 校准曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: 校准曲线
axes[0].plot([0, 1], [0, 1], 'k--', label='完美校准', alpha=0.5)
axes[0].plot(confidence_levels, coverages, 'ro-', label='实际校准', markersize=8)
axes[0].set_xlabel('名义覆盖率', fontsize=12)
axes[0].set_ylabel('实际覆盖率', fontsize=12)
axes[0].set_title('★ 校准曲线', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# 右: 按像素强度的覆盖率分析
x_flat = x_true[0].cpu().numpy().flatten()
# ★ 修复：使用 in_ci_cl（主置信区间覆盖率）而非 in_ci_75
in_ci_flat = in_ci_cl[0].cpu().numpy().flatten()
# 按像素强度分组
# ★ 修复：用 np.digitize 替代手写 (>= ) & (<) 掩码，避免最后一个 bin 漏掉
#   强度恰好等于 1.0 的像素（CelebA 人像过曝高光区域常见）的问题。
#   digitize 默认右开 bins[i-1] <= x < bins[i]，恰好 1.0 落到索引 len(bins)
#   的越界位置，需要 clip 到 [0, n_bins-1]。
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
n_bins = len(bins) - 1
bin_idx = np.digitize(x_flat, bins) - 1  # 范围 [-1, n_bins]
bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # 钳制到 [0, n_bins-1]
coverage_by_intensity = []
for i in range(n_bins):
    mask = bin_idx == i
    if mask.sum() > 0:
        coverage_by_intensity.append(in_ci_flat[mask].mean())
    else:
        coverage_by_intensity.append(np.nan)

axes[1].bar(bin_centers, coverage_by_intensity, width=0.08, color='steelblue', alpha=0.8)
axes[1].axhline(cl_max, color='red', linestyle='--', label=f'{cl_max:.0%}名义覆盖率')
axes[1].set_xlabel('像素强度', fontsize=12)
axes[1].set_ylabel('实际覆盖率', fontsize=12)
axes[1].set_title('★ 按像素强度的覆盖率', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

fig.suptitle('Step 4: ★ 校准检验', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_calibration.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: step4_calibration.png")


# ========================================================================
# Step 5: ★ 样本数对不确定性估计的影响
# 对应18.5节知识点：采样数量与不确定性估计精度
# ========================================================================
print("\n" + "="*70)
print("Step 5: ★ 样本数对不确定性估计的影响")
print("="*70)

print("""
★ 样本数对不确定性估计的影响:
  样本数 S 直接决定后验统计量的估计精度:
  - S过少: 标准差估计偏差大，置信区间不可靠
  - S适中(≥8): 均值和标准差趋于稳定
  - S较多(≥30): 高置信水平分位数也可可靠估计

本步骤对比: S=4 vs S={S} 样本数对不确定性的影响
""")

# 减少样本数的不确定性对比
if len(all_samples) >= 4:
    # 用前4个样本
    samples_s4 = torch.stack(all_samples[:4], dim=0)
    mean_s4 = samples_s4.mean(dim=0)
    std_s4 = samples_s4.std(dim=0)
    psnr_s4 = compute_psnr(x_true, mean_s4.to(device))

    # 用全部样本
    psnr_full = compute_psnr(x_true, posterior_mean.to(device))

    print(f"\nS=4 采样:  PSNR={psnr_s4:.2f} dB, 平均std={std_s4.mean():.4f}")
    print(f"S={S} 采样: PSNR={psnr_full:.2f} dB, 平均std={posterior_std.mean():.4f}")

    # 样本数对不确定性的影响
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 上行: S=4
    axes[0, 0].imshow(mean_s4[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title(f'S=4 后验均值\nPSNR={psnr_s4:.1f}dB', fontsize=11)
    axes[0, 0].axis('off')

    std_s4_map = std_s4[0].cpu().mean(dim=0).numpy()
    im01 = axes[0, 1].imshow(std_s4_map, cmap='hot', vmin=0, vmax=np.percentile(std_s4_map, 98))
    axes[0, 1].set_title(f'S=4 不确定性\n平均std={std_s4.mean():.4f}', fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046)

    # S=4 vs S=full 差异
    diff_std = (std_s4 - posterior_std).abs()[0].cpu().mean(dim=0).numpy()
    im02 = axes[0, 2].imshow(diff_std, cmap='hot', vmin=0, vmax=np.percentile(diff_std, 98))
    axes[0, 2].set_title(f'|std(S=4) - std(S={S})|', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046)

    # 下行: S=full
    axes[1, 0].imshow(posterior_mean[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, 0].set_title(f'S={S} 后验均值\nPSNR={psnr_full:.1f}dB', fontsize=11)
    axes[1, 0].axis('off')

    im11 = axes[1, 1].imshow(std_map, cmap='hot', vmin=0, vmax=np.percentile(std_map, 98))
    axes[1, 1].set_title(f'S={S} 不确定性\n平均std={posterior_std.mean():.4f}', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046)

    # 样本数 vs PSNR收敛
    # ★ 注意: 如果为混合采样（如ULA+DPS），曲线跳变反映方法切换而非收敛
    if ula_method != "混合采样":
        # ★ 非混合采样：绘制PSNR收敛曲线
        psnr_by_s = []
        for n in range(1, len(all_samples) + 1):
            mean_n = torch.stack(all_samples[:n], dim=0).mean(dim=0)
            psnr_by_s.append(compute_psnr(x_true, mean_n.to(device)))

        axes[1, 2].plot(range(1, len(all_samples) + 1), psnr_by_s, 'bo-', markersize=6)
        axes[1, 2].set_xlabel('样本数 S', fontsize=11)
        axes[1, 2].set_ylabel('PSNR (dB)', fontsize=11)
        axes[1, 2].set_title(f'★ 后验均值PSNR vs 样本数', fontsize=11)
        axes[1, 2].grid(alpha=0.3)
    else:
        # ★ 混合采样：改为按方法分组的柱状图，避免混淆"样本数"和"方法质量"
        method_psnrs = {}
        for s_idx, (sample, method) in enumerate(zip(all_samples, sample_methods)):
            if method not in method_psnrs:
                method_psnrs[method] = []
            method_psnrs[method].append(compute_psnr(x_true, sample.to(device)))

        methods_list = list(method_psnrs.keys())
        mean_psnrs = [np.mean(method_psnrs[m]) for m in methods_list]
        std_psnrs = [np.std(method_psnrs[m]) for m in methods_list]

        bars = axes[1, 2].bar(range(len(methods_list)), mean_psnrs,
                              yerr=std_psnrs, capsize=5, color='steelblue', alpha=0.8)
        axes[1, 2].set_xticks(range(len(methods_list)))
        axes[1, 2].set_xticklabels(methods_list, rotation=15, ha='right', fontsize=9)
        axes[1, 2].set_ylabel('平均PSNR (dB)', fontsize=11)
        axes[1, 2].set_title('★ 按方法分组的PSNR\n(混合采样)', fontsize=11)
        axes[1, 2].grid(alpha=0.3, axis='y')

        # 在柱状图上标注样本数
        for i, method in enumerate(methods_list):
            count = len(method_psnrs[method])
            axes[1, 2].text(i, mean_psnrs[i] + std_psnrs[i] + 0.5, f'n={count}',
                           ha='center', va='bottom', fontsize=9)

    fig.suptitle('Step 5: ★ 样本数对不确定性估计的影响', fontsize=14)
    # ★ 混合采样警告：S=4 子集不一定具有方法同质性
    # 在"尝试-失败-降级"策略下，前 4 个样本可能全部来自 ULA、后 4 个降级
    # 为 DPS/PnP，导致 S=4 与 S=8 对比实际测量的是"方法差异"而非
    # "样本数效应"，与 PSNR 曲线分支采用同样的同质性提示。
    if ula_method == "混合采样":
        sample_methods_s4 = sample_methods[:4]
        if len(set(sample_methods_s4)) > 1:
            subset_mix_note = f"S=4 子集已混合: {dict((m, sample_methods_s4.count(m)) for m in set(sample_methods_s4))}"
        else:
            subset_mix_note = f"S=4 子集单方法 ({sample_methods_s4[0]}), 全量 S={S} 中混入其他方法"
        fig.text(0.5, -0.02,
                 f"⚠️ {subset_mix_note} — S=4 vs S=full 对比同时混合了'样本数效应'与'方法质量差异'，"
                 f"结论应保守",
                 ha='center', va='top', fontsize=9, color='darkorange', style='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='darkorange'))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'step5_sample_size_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: step5_sample_size_effect.png")
else:
    print("样本数不足4个，跳过样本数对比")

# ★ 不确定性地图按问题类型解读
print("""
★ 不确定性地图解读指南（对应18.5节）:

去模糊:
  - 均匀分布的不确定性 → 噪声主导
  - 边缘区域不确定性高 → 模糊核导致的结构模糊

超分辨率:
  - 高频细节区域不确定性高 → 细节丢失无法恢复
  - 平滑区域不确定性低 → 低频信息保留完好

修复:
  - 缺失区域不确定性高 → 无观测约束
  - 观测区域不确定性低 → 有直接约束
  - 缺失区域边缘 → 不确定性过渡带

通用规律:
  - 不确定性高 ↔ 信息丢失严重
  - 不确定性低 ↔ 有充分观测约束
  - 决策时: 高不确定性区域应谨慎对待
""")


# ========================================================================
# 保存数值结果
# ========================================================================
import json

# ★ 计算方法分布（用于混合采样情况）
method_distribution = {}
for m in sample_methods:
    method_distribution[m] = method_distribution.get(m, 0) + 1

uq_results = {
    '采样方法': ula_method,
    '样本数': S,
    '主置信水平': round(cl_max, 4),
    '后验均值PSNR': round(psnr_mean, 2),
    '平均像素std': round(mean_std, 4),
    '最大像素std': round(max_std, 4),
    f'平均{cl_max:.0%}经验分位数区间宽度': round(ci_width_cl.mean().item(), 4),
    f'{cl_max:.0%}经验分位数区间覆盖率': round(overall_coverage, 4),
    '样本PSNR范围': [round(min(sample_psnrs), 2), round(max(sample_psnrs), 2)],
    '校准数据': {str(cl): round(cov, 4) for cl, cov in zip(confidence_levels, coverages)},
    '方法分布': method_distribution  # ★ 新增：记录各方法样本数
}
with open(os.path.join(SAVE_DIR, 'uq_results.json'), 'w', encoding='utf-8') as f:
    json.dump(uq_results, f, ensure_ascii=False, indent=2)
print("数值结果已保存: uq_results.json")


# ========================================================================
# ★ 动态生成总结结论（避免硬编码文字与实际计算结果脱节）
# ========================================================================

# 1. 校准状态（基于主置信区间覆盖率与名义值的偏差）
# ★ 使用与Step 4共享的 _calib_grade 函数, 保证两处判断一致
_coverage_dev = abs(overall_coverage - cl_max)
_calib_grade_text, _ = _calib_grade(_coverage_dev)
_calib_msg = (f"{cl_max:.0%}区间校准{_calib_grade_text}"
              f"（覆盖率={overall_coverage:.1%}，与名义值偏差{_coverage_dev:.1%}）")

# 2. PSNR收敛信息（仅在非混合采样且psnr_by_s已计算时使用）
if ula_method != "混合采样" and psnr_by_s is not None and len(psnr_by_s) > 1:
    _psnr_min = min(psnr_by_s)
    _psnr_max = max(psnr_by_s)
    _psnr_range = _psnr_max - _psnr_min
    if _psnr_range < 0.5:
        _convergence_msg = f"后验均值PSNR随S变化范围较小（{_psnr_min:.2f}~{_psnr_max:.2f} dB），已趋于稳定"
    elif psnr_by_s[-1] >= psnr_by_s[0]:
        _convergence_msg = f"后验均值PSNR随S大致递增（{_psnr_min:.2f}~{_psnr_max:.2f} dB）"
    else:
        _convergence_msg = f"后验均值PSNR随S非单调（{_psnr_min:.2f}~{_psnr_max:.2f} dB）"
elif ula_method == "混合采样":
    _convergence_msg = "混合采样下PSNR-vs-S曲线受方法切换影响，未单独分析S的影响"
else:
    _convergence_msg = f"样本数S={S}过少，未绘制PSNR-vs-S曲线"

# 3. 样本数建议（根据实际校准偏差和S大小动态调整）
if S >= 30 and _coverage_dev < 0.10:
    _sample_advice = f"S={S}已满足推荐阈值，校准检验较为可靠"
elif S >= 30:
    _sample_advice = f"S={S}已满足推荐阈值，但校准偏差较大（{_coverage_dev:.1%}），需检查采样方法"
elif S >= 8:
    _sample_advice = f"S={S}可获得基本不确定性估计，但S≥30才能可靠估计95%置信区间"
else:
    _sample_advice = f"S={S}远低于推荐阈值（建议S≥30以获得可靠的95%置信区间估计）"

# 4. ★ 不确定性数值可比性提示 (覆盖"纯降级方法"场景, 与"混合采样"警告互补)
#   原 Step 3/4/5 的可比性警告主要针对"混合采样" (ULA+DPS+PnP+伪逆 混合),
#   但学生也常使用纯 PnP 近似 或 纯伪逆+噪声 跑完整流程, 此时 posterior_std
#   同样不能与 ULA 的不确定性直接做绝对数值对比:
#     - PnP 近似: 20 步迭代是确定性梯度下降+去噪, 仅在初始化注入 0.05*randn,
#       缺少 Langevin 每步加噪, 不同"样本"可能收敛到彼此接近的点, std 系统性偏小;
#     - 伪逆+噪声: 各样本之间共享同一个伪逆解, 仅以 0.02*randn 扰动, std 几乎
#       完全由初始化噪声决定, 与"后验"语义相去甚远。
#   下面这段提示在最终总结中根据 ula_method 动态生成, 与上面的"混合采样"警告互补。
if ula_method in ("PnP近似", "伪逆+噪声"):
    _std_comparable_msg = (
        f"⚠️ 采样为纯降级方法 ({ula_method}), 该方法不构成严格后验采样 "
        f"(PnP 缺 Langevin 步加噪, 伪逆仅在初始化注入固定幅值噪声), "
        f"posterior_std 的绝对数值不能与 ULA 的不确定性直接比较大小, "
        f"本结果仅作'方法可运行'演示"
    )
elif ula_method == "混合采样":
    _std_comparable_msg = (
        f"⚠️ 采样为混合方法 (方法分布: {method_distribution}), "
        f"posterior_std 同时受'样本数'与'方法质量'影响, "
        f"不宜与单一 ULA 跑出的 std 直接比较"
    )
elif ula_method == "DPS":
    _std_comparable_msg = (
        f"DPS 采样的 posterior_std 与 ULA 的 std 在物理含义上一致 "
        f"(均为后验样本统计量), 但 deepinv 的 DPS 默认未做论文推荐的归一化缩放, "
        f"在 sigma_data 较大时 DPS 可能会比 ULA 给出更保守的不确定性"
    )
else:  # ULA 或 None
    _std_comparable_msg = "后验采样来自单一方法 (ULA), posterior_std 数值可作为该方法的不确定性估计"


# ========================================================================
# 实验总结
# ========================================================================
print("\n" + "="*70)
print("实验18.5-1 总结")
print("="*70)
print(f"""
本实验对应18.5节核心知识点：

1. 从点估计到分布 ✓
   - 点估计只给一个解，无法量化可靠性
   - 后验分布p(x|y)描述所有与观测一致的解

2. 后验采样方法 ✓
   - PnP-ULA: Langevin动力学 + DRUNet去噪器
   - DPS: 扩散后验采样
   - 加噪PnP: 近似后备方案

3. 不确定性量化 ✓
   - 后验均值: 最优点估计
   - 像素级标准差: 不确定性地图
   - 经验分位数区间: 覆盖真值的概率范围（⚠️ S<30时仅供参考）

4. ★ 校准检验 ✓
   - 覆盖率 vs 名义覆盖率（⚠️ S<30时结果不准确）
   - 按像素强度的覆盖率分析
   - {_calib_msg}

5. ★ 样本数影响 ✓
   - 样本数S对不确定性估计的影响
   - {_convergence_msg}
   - {_sample_advice}
   - {_std_comparable_msg}

关键发现:
- 采样方法: {ula_method}
- 后验均值PSNR: {psnr_mean:.2f} dB
- {cl_max:.0%}CI覆盖率: {overall_coverage:.1%}

所有图像已保存至: {SAVE_DIR}
""")