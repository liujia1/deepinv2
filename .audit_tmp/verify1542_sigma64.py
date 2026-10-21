"""float64 复核: β̃ 口径的重参数化系数是否精确等于后验均值系数"""
import torch

T = 200
betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
alphas = 1 - betas
ab = torch.cumprod(alphas, 0)

e_tilde, e_beta = [], []
for t in range(1, T):
    coef_true = torch.sqrt(alphas[t]) * (1 - ab[t - 1]) / torch.sqrt(1 - ab[t])
    beta_tilde = betas[t] * (1 - ab[t - 1]) / (1 - ab[t])
    c1 = torch.sqrt(torch.clamp(1 - ab[t - 1] - beta_tilde, min=0))
    c2 = torch.sqrt(torch.clamp(1 - ab[t - 1] - betas[t], min=0))
    e_tilde.append(abs((c1 - coef_true) / coef_true).item())
    e_beta.append(abs((c2 - coef_true) / coef_true).item())

print(f"float64 beta_tilde 口径 最大相对误差 = {max(e_tilde):.2e}")
print(f"float64 beta_t 口径     最大相对误差 = {max(e_beta):.2e}")
