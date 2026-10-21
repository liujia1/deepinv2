"""验证 15.4-2.py 采样方差修复: sigma^2=beta_tilde 时重参数化系数应精确等于后验均值"""
import torch

T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

err_tilde, err_beta = [], []
for t in range(1, T):
    ab_t, ab_prev = alpha_bars[t], alpha_bars[t - 1]
    # 正确后验: x_{t-1} 中 eps_hat 的系数 = sqrt(alpha_t)*(1-ab_prev)/sqrt(1-ab_t)
    coef_true = torch.sqrt(alphas[t]) * (1 - ab_prev) / torch.sqrt(1 - ab_t)

    beta_tilde = betas[t] * (1 - ab_prev) / (1 - ab_t)
    coef_tilde = torch.sqrt(torch.clamp(1 - ab_prev - beta_tilde, min=0))
    coef_beta = torch.sqrt(torch.clamp(1 - ab_prev - betas[t], min=0))

    err_tilde.append(abs(coef_tilde - coef_true).item() / coef_true.item())
    err_beta.append(abs(coef_beta - coef_true).item() / coef_true.item())

print(f"beta_tilde 口径: 最大相对误差 = {max(err_tilde):.3e}")
print(f"beta_t 口径    : 最大相对误差 = {max(err_beta):.3e}")
print(f"beta_tilde 是否全为负: {(betas[1:] * (1 - alpha_bars[:-1]) / (1 - alpha_bars[1:]) < 0).any().item()}")
print("结论: ", "修复后公式自洽" if max(err_tilde) < 1e-5 else "仍有偏差")
