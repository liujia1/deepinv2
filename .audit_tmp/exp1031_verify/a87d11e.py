# -*- coding: utf-8 -*-
"""
瀹為獙10.3-1 浠庡彉鍒嗕笅鐣屽埌鍘诲櫔鐩爣
瀵瑰簲绔犺妭: 10.3 浠庡彉鍒嗕笅鐣屽埌鍘诲櫔鐩爣

鐭ヨ瘑鐐?
  - 涓夌鍙傛暟鍖栫殑绛変环鎬? x鈧€-prediction, 蔚-prediction, score prediction
  - 鍧囧€煎尮閰嶇洰鏍?vs L_simple绠€鍖栫洰鏍?  - VLB鏉冮噸闅忔椂闂寸殑鍙樺寲鐗规€?
瀹為獙鍐呭:
  姝ラ1: 涓夌鍙傛暟鍖栫瓑浠锋€ч獙璇?(x鈧€/蔚/score prediction)
  姝ラ2: 鍧囧€煎尮閰嶆崯澶?vs L_simple鏉冮噸鍒嗘瀽
  姝ラ3: 鎹熷け鏉冮噸鍙鍖?
鏈疄楠屼笉闇€瑕丟PU锛岄€氳繃鏁板€煎疄楠岄獙璇?0.3鑺傜殑鏍稿績鏁板缁撹銆?"""

import sys
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 闈炰氦浜掑紡鍚庣
import matplotlib.pyplot as plt
import logging
import warnings

# 璁剧疆鎺у埗鍙拌緭鍑轰负 UTF-8 (Windows涓嬮伩鍏嶄腑鏂囦贡鐮?
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 闈欓粯matplotlib鐩稿叧璀﹀憡
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 涓枃瀛椾綋閰嶇疆(鍏煎鏈湴鍜孏oogle Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("姝ｅ湪鎸傝浇 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '瀹為獙10.3-1')
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
    print("璀﹀憡: chinese_font妯″潡鏈壘鍒帮紝涓枃瀛椾綋鍙兘鏃犳硶姝ｅ父鏄剧ず")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

print("\n" + "="*60)
print("瀹為獙10.3-1: 浠庡彉鍒嗕笅鐣屽埌鍘诲櫔鐩爣")
print("="*60)
print("瀵瑰簲绔犺妭: 10.3 浠庡彉鍒嗕笅鐣屽埌鍘诲櫔鐩爣")
print("浣跨敤璁惧: CPU (鏈疄楠屼笉闇€瑕丟PU)")


# ============================================================
# 鍣０璋冨害瀹氫箟
# ============================================================
T = 1000  # 鎵╂暎姝ユ暟
beta_min, beta_max = 1e-4, 0.02

# 绾挎€у櫔澹拌皟搴?(DDPM)
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

print("\n鍣０璋冨害: beta浠? + str(beta_min) + "绾挎€у鑷? + str(beta_max) + ", T=" + str(T))


# ============================================================
# 姝ラ1锛氫笁绉嶅弬鏁板寲鐨勭瓑浠锋€ч獙璇?鈥斺€?10.3鑺傛牳蹇?# ============================================================
print("\n" + "="*60)
print("姝ラ1锛氫笁绉嶅弬鏁板寲鐨勭瓑浠锋€ч獙璇?)
print("="*60)
print("\n[鏍稿績鎬濇兂]")
print("  涓夌鍙傛暟鍖栨柟寮忓湪瀹岀編棰勬祴涓嬬粰鍑哄畬鍏ㄧ浉鍚岀殑鍚庨獙鍧囧€?")
print("  (1) x_0-prediction: 棰勬祴hat_x_0 鈫?璁＄畻mu_theta")
print("  (2) epsilon-prediction: 棰勬祴hat_epsilon 鈫?璁＄畻mu_theta")
print("  (3) score prediction: 棰勬祴hat_s 鈫?璁＄畻mu_theta")

# 缁欏畾x_0, t, 蔚锛岄獙璇佷笁绉嶅弬鏁板寲鐨勪竴鑷存€?x0_test = torch.randn(3, 2)
eps_test = torch.randn_like(x0_test)

print("\n楠岃瘉涓夌鍙傛暟鍖栧鍚庨獙鍧囧€?tilde_mu_t 鐨勭瓑浠锋€?")
print("    t      鐪熷疄mu_t      x0-pred      eps-pred       score")
print("-" * 70)

for t_idx in [10, 100, 500, 900]:
    # 娉ㄦ剰锛歅ython绱㈠紩浠?寮€濮嬶紝鏃堕棿姝浠?寮€濮嬶紝鎵€浠ョ储寮曚负t-1
    idx = t_idx - 1
    ab = alpha_bars[idx]
    a = alphas[idx]
    ab_prev = alpha_bars_prev[idx]
    b = betas[idx]

    # 鏋勯€爔_t
    x_t = torch.sqrt(ab) * x0_test + torch.sqrt(1 - ab) * eps_test

    # 鐪熷疄鍚庨獙鍧囧€?mu_t(x_t, x_0)
    mu_true = (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x_t + \
              (torch.sqrt(ab_prev) * (1 - a) / (1 - ab)) * x0_test

    # (1) x鈧€-prediction: 鑻ョ煡閬搙_0, 鐩存帴璁＄畻
    mu_x0 = mu_true  # 瀹岀編棰勬祴鏃剁浉鍚?
    # (2) 蔚-prediction: eps 鈫?mu = (1/sqrt(a))(x_t - (1-a)/sqrt(1-ab)路eps)
    mu_eps = (1.0 / torch.sqrt(a)) * (x_t - (1 - a) / torch.sqrt(1 - ab) * eps_test)

    # (3) score prediction: s 鈫?eps = -sqrt(1-ab)路s 鈫?浠ｅ叆eps-prediction
    score_hat = -eps_test / torch.sqrt(1 - ab)  # 瀹岀編棰勬祴
    eps_from_score = -torch.sqrt(1 - ab) * score_hat
    mu_score = (1.0 / torch.sqrt(a)) * (x_t - (1 - a) / torch.sqrt(1 - ab) * eps_from_score)

    err_x0 = (mu_true - mu_x0).abs().max().item()
    err_eps = (mu_true - mu_eps).abs().max().item()
    err_score = (mu_true - mu_score).abs().max().item()

    print(str(t_idx).rjust(5) + "  " + str(round(mu_true[0,0].item(), 8)).rjust(15) + "  " + str(round(mu_x0[0,0].item(), 8)).rjust(15) + "  " + str(round(mu_eps[0,0].item(), 8)).rjust(15) + "  " + str(round(mu_score[0,0].item(), 8)).rjust(15))
    print("       璇樊: x0-pred=" + str(round(err_x0, 10)) + ", eps-pred=" + str(round(err_eps, 10)) + ", score=" + str(round(err_score, 10)))

print("\n鈫?涓夌鍙傛暟鍖栧湪瀹岀編棰勬祴涓嬬粰鍑哄畬鍏ㄧ浉鍚岀殑鍚庨獙鍧囧€?)
print("  杩欐槸10.3鑺傜殑鏍稿績缁撹: x0-prediction 鈮?eps-prediction 鈮?score prediction")
print("\n涓夌鍙傛暟鍖栫殑杞崲鍏崇郴:")
print("  - x0 鈫?epsilon: x0 = (x_t - sqrt(1-ab_t)路eps) / sqrt(ab_t)")
print("  - epsilon 鈫?score: s = -eps / sqrt(1-ab_t)")
print("  - score 鈫?x0: x0 = (x_t + (1-ab_t)路s) / sqrt(ab_t)")


# ============================================================
# 姝ラ2锛氬潎鍊煎尮閰嶆崯澶?vs L_simple鏉冮噸鍒嗘瀽 鈥斺€?10.3鑺?# ============================================================
print("\n" + "="*60)
print("姝ラ2锛氬潎鍊煎尮閰嶆崯澶?vs L_simple鏉冮噸鍒嗘瀽")
print("="*60)
print("\n[鏍稿績鎬濇兂]")
print("  VLB鐩爣涓殑鍧囧€煎尮閰嶉」鏈夋潈閲?")
print("  L_{t-1} 鈭?(1-alpha_t)^2 / (alpha_t(1-ab_t)) * ||eps - hat_eps||^2")
print("  L_simple瀵规墍鏈夋椂闂存璧嬬瓑鏉冮噸1锛岃繖鏄疕o et al. 2020鐨勫叧閿畝鍖?)

print("\n涓嶅悓鏃堕棿姝ョ殑VLB鏉冮噸:")
print("    t       ab_t       VLB鏉冮噸     L_simple鏉冮噸   鐩稿t=1")
print("-" * 55)

# 鍏堣绠梩=1鐨勬潈閲嶄綔涓哄熀鍑?weight_t1 = betas[0] / (2 * alphas[0] * (1 - alpha_bars[0]))

for t_idx in [1, 10, 50, 100, 300, 500, 700, 900, 999]:
    # 娉ㄦ剰锛歅ython绱㈠紩浠?寮€濮嬶紝鏃堕棿姝浠?寮€濮嬶紝鎵€浠ョ储寮曚负t-1
    idx = t_idx - 1
    ab = alpha_bars[idx]
    a = alphas[idx]
    b = betas[idx]

    # VLB鏉冮噸绯绘暟: 鏍规嵁Ho et al. 2020 eq.14锛屼娇鐢ㄏ僟t虏=尾_t褰掍竴鍖?    # L_{t-1} 鈭?1/(2蟽_t虏) 路 尾_t虏/(伪_t(1-峋盻t)) 路 ||蔚-蔚_胃||虏
    # 鍙栂僟t虏=尾_t鏃讹紝鏉冮噸绯绘暟 = 尾_t/(2伪_t(1-峋盻t))
    weight_vlb = b / (2 * a * (1 - ab))

    # 璁＄畻鐩稿浜巘=1鐨勬潈閲嶆瘮鍊?    ratio_to_t1 = weight_vlb / weight_t1

    print(str(t_idx).rjust(5) + "  " + str(round(ab.item(), 6)).rjust(10) + "  " + str(round(weight_vlb.item(), 6)).rjust(12) + "       1.000000      " + str(round(ratio_to_t1.item(), 4)).rjust(8))

print("\n鍏抽敭瑙傚療:")
print("  - t灏?1-100): 鏉冮噸闅弔澧炲ぇ鑰屾€ュ墽涓嬮檷(绾?00鍊?锛屾棭鏈熸椂闂存涓诲VLB鐩爣")
print("  - t涓(t鈮?00闄勮繎): 鏉冮噸杈惧埌璋峰€硷紝瀵筕LB璐＄尞鏈€灏?)
print("  - t澶?500+): 鏉冮噸缂撴參鍥炲崌锛屼絾鏁翠綋浠嶈繙灏忎簬t=1鏃剁殑姘村钩")
print("  - L_simple瀵规墍鏈塼璧嬬瓑鏉冮噸1锛岃繖鏄疕o et al. 2020鐨勫叧閿畝鍖?)
print("  - 瀹為檯鏁堟灉: L_simple蹇界暐鏉冮噸宸紓锛屼絾璁粌鏇寸ǔ瀹氾紙10.3鑺傦級")


# ============================================================
# 姝ラ3锛毼?prediction鐨勪紭鍔垮垎鏋?# ============================================================
print("\n" + "="*60)
print("姝ラ3锛歟ps-prediction鐨勪紭鍔垮垎鏋?)
print("="*60)
print("\n[涓轰粈涔圖DPM閫夋嫨eps-prediction]")
print("  1. 鐩爣鍑芥暟绠€鍗? L_simple = ||eps - hat_eps_theta(x_t, t)||^2")
print("  2. 缃戠粶杈撳嚭鑼冨洿: 鍣０eps鏈嶄粠N(0,I)锛岃緭鍑鸿寖鍥存湁鐣?)
print("  3. 璁粌绋冲畾鎬? 鏃犻渶澶勭悊x_0鐨勫鏉備緷璧?)
print("  4. 璁＄畻鏁堢巼: 鐩存帴棰勬祴鍣０锛屼竴姝ュ埌浣?)

# 鏁板€奸獙璇? 涓嶅悓t涓嬬殑鐩爣鍑芥暟鑼冨洿
print("\n涓嶅悓鍙傛暟鍖栦笅鐨勭洰鏍囧嚱鏁版暟鍊艰寖鍥?")
print("    t     ||x0-hat_x0||鑼冨洿   ||eps-hat_eps||鑼冨洿")
print("-" * 45)

for t_idx in [10, 100, 500, 900]:
    # 娉ㄦ剰锛歅ython绱㈠紩浠?寮€濮嬶紝鏃堕棿姝浠?寮€濮嬶紝鎵€浠ョ储寮曚负t-1
    idx = t_idx - 1
    ab = alpha_bars[idx]
    # 鍋囪棰勬祴璇樊涓?
    err_eps = 1.0
    # 杞崲鍒皒0绌洪棿鐨勮宸?    err_x0 = err_eps / torch.sqrt(ab)

    print(str(t_idx).rjust(5) + "           " + str(round(err_x0.item(), 4)).rjust(15) + "           " + str(round(err_eps, 4)).rjust(15))

print("\n鍏抽敭鍙戠幇:")
print("  - t灏?ab_t澶?: x0-prediction璇樊灏忥紝eps-prediction璇樊鏀惧ぇ")
print("  - t澶?ab_t灏?: x0-prediction璇樊鏋佸ぇ锛宔ps-prediction淇濇寔绋冲畾")
print("  鈫?eps-prediction鍦ㄦ墍鏈夋椂闂存閮芥湁绋冲畾鐨勬暟鍊艰寖鍥?)


# ============================================================
# 鍙鍖?# ============================================================
print("\n" + "="*60)
print("鐢熸垚鍙鍖栧浘琛?..")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) VLB鏉冮噸 vs L_simple鏉冮噸
ax = axes[0]
# 淇鍚庣殑VLB鏉冮噸鍏紡锛氫娇鐢ㄏ僟t虏=尾_t褰掍竴鍖?weights = betas / (2 * alphas * (1 - alpha_bars))
ax.semilogy(range(1, T + 1), weights.numpy(), 'b-', linewidth=2, label='VLB鏉冮噸')
ax.axhline(1, color='r', linestyle='--', linewidth=1.5, label='$L_{simple}$鏉冮噸 (=1)')
ax.set_xlabel('鏃堕棿姝?$t$', fontsize=12)
ax.set_ylabel('鎹熷け鏉冮噸 (瀵规暟灏哄害)', fontsize=12)
ax.set_title('(a) VLB鎹熷け鏉冮噸 vs $L_{simple}$', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.annotate('VLB鏉冮噸: 灏弔鍗犱富瀵? t鈮?00鍚庡皬骞呭洖鍗?, xy=(0.3, 0.8), xycoords='axes fraction',
            fontsize=10, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

# (b) 涓夌鍙傛暟鍖栫殑杞崲绯绘暟
ax = axes[1]
t_range = range(1, T + 1)

# eps鍒皒0鐨勮浆鎹㈢郴鏁? 1/sqrt(ab_t)
coeff_eps_to_x0 = 1.0 / torch.sqrt(alpha_bars)
# score鍒癳ps鐨勮浆鎹㈢郴鏁? sqrt(1-ab_t)
coeff_score_to_eps = torch.sqrt(1 - alpha_bars)

ax.plot(t_range, coeff_eps_to_x0.numpy(), 'b-', linewidth=2, label=r'$\epsilon \to x_0$: $1/\sqrt{\bar{\alpha}_t}$')
ax.plot(t_range, coeff_score_to_eps.numpy(), 'g-', linewidth=2, label=r'$s \to \epsilon$: $\sqrt{1-\bar{\alpha}_t}$')
ax.set_xlabel('鏃堕棿姝?$t$', fontsize=12)
ax.set_ylabel('杞崲绯绘暟', fontsize=12)
ax.set_title('(b) 涓夌鍙傛暟鍖栫殑杞崲绯绘暟', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 20)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '姝ラ1_鍙傛暟鍖栫瓑浠锋€т笌鎹熷け鏉冮噸.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print("鍥捐〃宸蹭繚瀛? 姝ラ1_鍙傛暟鍖栫瓑浠锋€т笌鎹熷け鏉冮噸.png")


# ============================================================
# 瀹為獙鎬荤粨
# ============================================================
print("\n" + "="*60)
print("瀹為獙10.3-1 鎬荤粨")
print("="*60)
print("\n鍏抽敭缁撹:")
print("\n1. 涓夌鍙傛暟鍖栫瓑浠锋€?(姝ラ1)")
print("   - x0-prediction, eps-prediction, score prediction鍦ㄥ畬缇庨娴嬩笅缁欏嚭鐩稿悓鍚庨獙鍧囧€?)
print("   - 杞崲鍏崇郴:")
print("     x0 = (x_t - sqrt(1-ab_t)路eps) / sqrt(ab_t)")
print("     s = -eps / sqrt(1-ab_t)")
print("\n2. VLB鏉冮噸鍒嗘瀽 (姝ラ2)")
print("   - VLB鏉冮噸 w_t = 尾_t / (2伪_t(1-峋盻t))  (浣跨敤蟽_t虏=尾_t褰掍竴鍖?")
print("   - 鏉冮噸鍛圲褰? t灏忔€ュ墽涓嬮檷锛宼鈮?00璋峰€硷紝t澶х紦鎱㈠洖鍗?)
print("   - L_simple瀵规墍鏈塼璧嬬瓑鏉冮噸1")
print("\n3. eps-prediction鐨勪紭鍔?(姝ラ3)")
print("   - 鐩爣鍑芥暟绠€鍗? ||eps - hat_eps||^2")
print("   - 鏁板€艰寖鍥寸ǔ瀹氾紝鎵€鏈夋椂闂存璇樊灏哄害涓€鑷?)
print("   - DDPM閫夋嫨eps-prediction浣滀负榛樿鍙傛暟鍖?)

print("\n" + "="*60)
print("瀹為獙10.3-1 瀹屾垚!")

# ===== 淇濆瓨鏁板€肩粨鏋?=====
import json

def _to_native(obj):
    """閫掑綊杞崲numpy/torch绫诲瀷涓篜ython鍘熺敓绫诲瀷"""
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

# VLB鏉冮噸 (浣跨敤蟽_t虏=尾_t褰掍竴鍖?
weights_vlb = (betas / (2 * alphas * (1 - alpha_bars))).numpy()
results_summary = {
    'parameterization_equivalence': {
        'note': '涓夌鍙傛暟鍖栧湪瀹岀編棰勬祴涓嬬粰鍑哄畬鍏ㄧ浉鍚岀殑鍚庨獙鍧囧€?,
    },
    'VLB_weights': {
        f't_{t}': weights_vlb[t-1]
        for t in [1, 10, 50, 100, 300, 500, 700, 900, 999]
    },
    'conversion_coefficients': {
        'note': 'eps_to_x0: 1/sqrt(ab_t), score_to_eps: sqrt(1-ab_t)',
        f't_{t}': {
            'eps_to_x0': (1.0 / torch.sqrt(alpha_bars[t-1])).item(),
            'score_to_eps': torch.sqrt(1 - alpha_bars[t-1]).item(),
        }
        for t in [10, 100, 500, 900]
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"鏁板€肩粨鏋滃凡淇濆瓨: {os.path.join(SAVE_DIR, 'results_summary.json')}")
