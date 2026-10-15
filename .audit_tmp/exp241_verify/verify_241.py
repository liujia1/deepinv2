# -*- coding: utf-8 -*-
"""
实验2.4-1 复核脚本：
1) 按 main() 的精确 RNG 时序复现图2/图3的 PSNR，与 results_summary.json 及讲义表格对比
2) 复现图3能量曲线（Start/End、单调性），与讲义"约593→约107"对比
3) 检验 unrolled 更新是否等于 -∇E（代码能量），并对比代码能量与讲义公式能量（patch-norm Student-t vs 逐像素）
4) 加载 checkpoint 验证架构兼容 + 记录 config/alphas
5) 滤波器方向选择性定量分析（与水平/垂直/对角/反对角方向导数核及相关纹理核的相关系数）
6) 图3硬编码"PSNR: 20.01dB"核查：理论值 10log10(1/0.12^2)=18.49，实测该 RNG 噪声实现的 PSNR
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
EXP_DIR = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.4-1"
sys.path.insert(0, EXP_DIR)
os.chdir(EXP_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location("exp241", os.path.join(EXP_DIR, "2.4-1.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # 只执行顶层（定义），main 在 __main__ 守卫下不会跑

torch.set_num_threads(1)
device = torch.device('cpu')

print("=" * 66)
print("[1] checkpoint 加载与 config 核验")
ck = torch.load(os.path.join(EXP_DIR, 'checkpoints', 'foe_final.pth'), map_location=device, weights_only=False)
print("  keys:", list(ck.keys()))
print("  epoch:", ck.get('epoch'), " config:", ck.get('config'))
model_after = mod.build_model()
sd = model_after.state_dict()
loaded = model_after.load_state_dict(ck['model_state_dict'])
print("  load_state_dict:", loaded)
model_after.eval()
alphas = model_after.alphas.detach().cpu().numpy()
print("  alphas(softplus+1e-4):", np.round(alphas, 3))
filters = model_after.get_normalized_filters().detach().cpu()

print("=" * 66)
print("[2] 复现 main() 的 RNG 时序")
x_gt_test = mod.create_rich_test_image(64).to(device)
torch.manual_seed(42); np.random.seed(42)
train_set = mod.SyntheticRichDataset(num_samples=128, size=64, seed_offset=0)
_ = torch.utils.data.DataLoader(train_set, batch_size=mod.BATCH_SIZE, shuffle=True)
# 训练分支被 FINAL_CKPT 跳过（文件存在），RNG 不消耗；load 不消耗 RNG
torch.manual_seed(42); np.random.seed(42)
model_before = mod.build_model()
model_before.eval()
# 图2 时序: x_gt 先, y 后
x_gt = x_gt_test
y2 = mod.degrade(x_gt, mod.NOISE_LEVEL)
xb = torch.clamp(model_before.forward_denoise_unrolled(y2, mod.TEST_UNROLL_STEPS, mod.TEST_STEP_SIZE), 0, 1)
xa = torch.clamp(model_after.forward_denoise_unrolled(y2, mod.TEST_UNROLL_STEPS, mod.TEST_STEP_SIZE), 0, 1)
psnr_y2 = mod.compute_psnr(y2, x_gt)
psnr_b = mod.compute_psnr(xb, x_gt)
psnr_a = mod.compute_psnr(xa, x_gt)
print(f"  图2: psnr_y={psnr_y2:.4f}  psnr_before={psnr_b:.4f}  psnr_after={psnr_a:.4f}")
# 图3 时序: x_gt 又先, y3 后
y3 = mod.degrade(x_gt, mod.NOISE_LEVEL)
psnr_y3 = mod.compute_psnr(y3, x_gt)
print(f"  图3含噪输入实测 PSNR={psnr_y3:.4f} dB  (图3标题硬编码 20.01)")
print(f"  理论值(σ=0.12, MSE=σ²): {10*np.log10(1/0.12**2):.4f} dB")

print("=" * 66)
print("[3] 复现图3能量轨迹 (10步, step=0.15)")
x_curr = y3.clone()
E_series = []
with torch.no_grad():
    for t in range(11):
        E_scalar, _ = model_after.compute_energy(x_curr, y3)
        E_total = float(E_scalar) * x_curr.numel()
        E_series.append(E_total)
        if t < 10:
            x_curr = model_after.forward_denoise_unrolled(x_curr, num_steps=1, step_size=mod.TEST_STEP_SIZE)
psnr_t10 = mod.compute_psnr(torch.clamp(x_curr, 0, 1), x_gt)
print("  E(t) =", np.round(E_series, 1).tolist())
mono = all(E_series[i+1] <= E_series[i] + 1e-6 for i in range(10))
print(f"  Start={E_series[0]:.1f} End={E_series[-1]:.1f}  单调下降: {mono}")
print(f"  10步末端 PSNR={psnr_t10:.4f} dB")

print("=" * 66)
print("[4] JSON 与讲义表格数值对比")
js = json.load(open(os.path.join(EXP_DIR, 'results_summary.json'), encoding='utf-8'))
print("  JSON:", {k: js[k] for k in js if k.startswith('psnr')})
print("  讲义表: y=18.50, before=20.58, after(50步)=23.42, after(10步)=23.81")
print(f"  复现值: y={psnr_y2:.2f}, before={psnr_b:.2f}, after={psnr_a:.2f}, 10步={psnr_t10:.2f}")

print("=" * 66)
print("[5] 更新式 = -∇E 一致性检验（代码能量）")
m = model_after
x = y3.clone().requires_grad_(True)
with torch.no_grad():
    # 解析梯度（forward_denoise_unrolled 单步）
    fl = m.get_normalized_filters(); al = m.alphas.view(1, -1, 1, 1); pad = fl.shape[-1] // 2
    resp = F.conv2d(x.detach(), fl, padding=pad)
    d_pot = (2.0 * m.gamma * al * resp) / (1.0 + m.gamma * resp ** 2)
    grad_analytic = m.lambda_reg * F.conv_transpose2d(d_pot, fl, padding=pad) + (x.detach() - y3)
# autograd 的代码能量 E_scalar = mean(α log(1+γr²)) + 0.5 mean((x-y)²)
E_scalar, _ = m.compute_energy(x, y3)
grad_auto = torch.autograd.grad(E_scalar, x)[0]
HW = 64 * 64
# 用归一化内积判断方向一致性 + 尺度因子: <g_analytic, g_auto>/||g_auto||² 应≈HW
scale = (grad_analytic * grad_auto).sum().item() / (grad_auto ** 2).sum().item()
cos = (grad_analytic * grad_auto).sum().item() / (grad_analytic.norm() * grad_auto.norm()).item()
rel_err = (grad_analytic - scale * grad_auto).norm().item() / grad_analytic.norm().item()
max_rel = ((grad_analytic - HW * grad_auto).abs().max() / grad_analytic.abs().max()).item()
print(f"  尺度因子<g_a,g_b>/||g_b||²={scale:.2f} (期望≈HW={HW}), cos={cos:.6f}, 去尺度后相对残差={rel_err:.2e}, max逐像素相对误差={max_rel:.2e}")
ok = cos > 0.999999 and max_rel < 1e-5
print("  → 更新方向 = -∇[0.5‖x-y² + λΣ像素 α log(1+r²)] 成立，与讲义总能量公式(λ=0.01)一致 ✓" if ok else "  → 梯度与能量不一致！")
# 判别：更新括号内 g_upd = λ·reg_g + data_g（reg_g=∇Σα log(1+r²) 逐像素未除HW, data_g=x-y）
#   讲义声称能量 E_md = 0.5‖x-y‖² + 0.01·Σ像素 α log(1+r²) → ∇E_md = data_g + 0.01·reg_g，应与 g_upd 共线
#   但代码 compute_energy 用 mean → ∇(HW·E_code) = reg_g + data_g，与 g_upd 不共线 → 内层真实最小化的是 λ_eff=λ/HW≈2.44e-6 的能量
with torch.no_grad():
    fl = m.get_normalized_filters(); al = m.alphas.view(1, -1, 1, 1); pad = 3
    x_mid = y3.clone()
    for _ in range(5):   # 走到轨迹中段，避开 x=y 退化点
        r = F.conv2d(x_mid, fl, padding=pad)
        dp = (2 * m.gamma * al * r) / (1 + m.gamma * r ** 2)
        x_mid = x_mid - 0.15 * (m.lambda_reg * F.conv_transpose2d(dp, fl, padding=pad) + (x_mid - y3))
    r2 = F.conv2d(x_mid, fl, padding=pad)
    dp2 = (2 * m.gamma * al * r2) / (1 + m.gamma * r2 ** 2)
    reg_g = F.conv_transpose2d(dp2, fl, padding=pad)
    data_g = x_mid - y3
    g_upd = m.lambda_reg * reg_g + data_g
    g_md = 0.01 * reg_g + data_g
    cos_md = float((g_upd * g_md).sum() / (g_upd.norm() * g_md.norm()))
    step = 0.15
    # 能量下降检验：沿 -g_upd 走一步(与真实轨迹同步 step=0.15)，看两个能量各下降多少
    def E_code_at(x):   # = HW·compute_energy = Σ像素α log(1+r²) + 0.5‖x-y‖²（先验:数据权重比 1:1）
        rr = F.conv2d(x, fl, padding=pad)
        return float((al * torch.log(1 + rr ** 2)).sum() + 0.5 * ((x - y3) ** 2).sum())
    def E_md_at(x):     # 讲义声称：0.5‖x-y‖² + 0.01·Σ像素 α log(1+r²)（先验:数据比 0.01:1）
        rr = F.conv2d(x, fl, padding=pad)
        return float(0.5 * ((x - y3) ** 2).sum() + 0.01 * (al * torch.log(1 + rr ** 2)).sum())
    x_next = x_mid - step * g_upd
    dE_code = E_code_at(x_next) - E_code_at(x_mid)
    dE_md = E_md_at(x_next) - E_md_at(x_mid)
    print(f"  先验:数据梯度权重比 —— 更新式: {m.lambda_reg:.2f};  讲义E(×HW后): {0.01:.2f} → 共线，cos={cos_md:.8f}")
    print(f"  而图3绘制/监控用的 E_code(=HW·compute_energy) 比值: 1.00 → 与更新式不共线！")
    print(f"  单步(x=第5步处, step=0.15): ΔE_plot(图3曲线用的能量)={dE_code:+.2f},  ΔE_md(讲义声称能量)={dE_md:+.2f}")
    # 全轨迹上检查两个能量是否都单调
    xs = [y3.clone()]
    with torch.no_grad():
        cur = y3.clone()
        for _ in range(10):
            rr = F.conv2d(cur, fl, padding=pad)
            dd = (2 * m.gamma * al * rr) / (1 + m.gamma * rr ** 2)
            cur = cur - 0.15 * (m.lambda_reg * F.conv_transpose2d(dd, fl, padding=pad) + (cur - y3))
            xs.append(cur.clone())
    Ec = [E_code_at(v) for v in xs]; Em = [E_md_at(v) for v in xs]
    mono_c = all(Ec[i+1] <= Ec[i] + 1e-6 for i in range(10))
    mono_m = all(Em[i+1] <= Em[i] + 1e-6 for i in range(10))
    print(f"  全轨迹单调性: E_code(图3曲线){'单调✓' if mono_c else '非单调✗'} (端点 {Ec[0]:.1f}→{Ec[-1]:.1f});  E_md(讲义){'单调✓' if mono_m else '非单调✗'} (端点 {Em[0]:.1f}→{Em[-1]:.1f})")

print("=" * 66)
print("[6] 讲义公式能量(整体 patch-norm Student-t) vs 代码逐像素能量")
with torch.no_grad():
    fl = m.get_normalized_filters(); al = m.alphas.view(1, -1, 1, 1)
    resp = F.conv2d(y3, fl, padding=3)
    E_code_md = (al * torch.log(1 + resp ** 2)).sum().item()  # Σ_k α_k Σ_i log(1+r²)
    E_patch = sum(float(al[0, k, 0, 0]) * np.log(1 + (resp[0, k] ** 2).sum().item()) for k in range(6))
    print(f"  逐像素求和(代码,×λ=0.01): {E_code_md:.1f} → λE={0.01*E_code_md:.1f}; 全图数据项 0.5‖y3-y3‖²=0")
    print(f"  讲义字面 patch-norm 形式: {E_patch:.3f} → λE={0.01*E_patch:.4f}  (两者差异巨大，公式表述与实现不同)")

print("=" * 66)
print("[7] 滤波器方向选择性定量分析")
def norm(a):
    a = a - a.mean(); return a / (np.linalg.norm(a) + 1e-12)
# 构造方向基（3x3 pad 到 7x7 中心）
basis = {}
g = np.array([-1, 0, 1], dtype=float)
basis['∂x(垂直梯度/水平边缘)'] = norm(np.kron(np.array([[1], [2], [1]]), g.reshape(1, 3)))
basis['∂y(水平梯度/垂直边缘)'] = norm(np.kron(g.reshape(3, 1), np.array([[1, 2, 1]])))
basis['∂xx(水平条纹)'] = norm(np.outer(np.array([1, 2, 1]), np.array([1, -2, 1])))
basis['∂yy(垂直条纹)'] = norm(np.outer(np.array([1, -2, 1]), np.array([1, 2, 1])))
basis['∂xy(对角)'] = norm(np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 1.0)
basis['∂x∂y\'(反对角)'] = norm(np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) + 2 * np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]) * 0)
# 3x3 pad 到 7x7 中心
def pad7(a):
    z = np.zeros((7, 7)); z[2:5, 2:5] = a; return z
F_np = filters.numpy()[:, 0]
print("  (相关系数为与中心 3x3 展开的各向基的 |corr|，>0.7 视为该方向主导)")
for k in range(6):
    f = F_np[k]
    cors = {name: abs(float(np.dot(pad7(b).ravel(), norm(f).ravel()))) for name, b in basis.items()}
    best = max(cors, key=cors.get)
    print(f"  f{k+1}: " + "  ".join(f"{n}={v:.2f}" for n, v in cors.items()) + f"   → 最强: {best}({cors[best]:.2f})")

print("=" * 66)
print("[8] 训练是否真正改变了滤波器？(初始化 vs final 的归一化距离, 及与初始方向的|cos|)")
torch.manual_seed(42); np.random.seed(42)
m_init = mod.FoEModule(num_experts=6, kernel_size=7, gamma=1.0, lambda_reg=0.01)
f_init = m_init.get_normalized_filters().detach().numpy()[:, 0]
f_fin = F_np
for k in range(6):
    a = norm(f_init[k]); b2 = norm(f_fin[k])
    print(f"  f{k+1}: ||Δf||={float(np.linalg.norm(b2 - a)):.3f}, cos(f_init,f_final)={float(np.dot(a.ravel(), b2.ravel())):.3f}, α_init={float(m_init.alphas[k]):.3f}")
# 低频/平滑成分占比：与均值模糊核(去掉常数后为0)无关，改用 DC增益+频谱质心
def freq_centroid(f):
    Ff = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    u = np.arange(7) - 3; U, V = np.meshgrid(u, u)
    return float((np.sqrt(U**2 + V**2) * Ff).sum() / (Ff.sum() + 1e-12))
print("  频谱质心(小=低频/平滑主导, 大=高频细节): final vs init")
for k in range(6):
    print(f"  f{k+1}: final={freq_centroid(f_fin[k]):.2f}  init={freq_centroid(norm(f_init[k])):.2f}")
