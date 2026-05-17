"""
实验6.6：SMLD完整实现——从零训练到图像生成（DT3）
对应章节：6.5（多尺度得分匹配与NCSN）、6.7（用学习到的得分驱动采样）
参考素材：03-smld.ipynb (diffusion-tutorials-master)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

L = 10
sigma_min = 0.01
sigma_max = 1.0
sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** (i / (L - 1))
                       for i in range(L)])
print(f"Noise schedule: {sigmas.numpy()}")


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_sigmas):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_sigmas, num_features)
        self.beta = nn.Embedding(num_sigmas, num_features)
        self.gamma.weight.data.fill_(1.0)
        self.beta.weight.data.fill_(0.0)

    def forward(self, x, sigma_idx):
        out = self.bn(x)
        gamma = self.gamma(sigma_idx).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(sigma_idx).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


class NCSN_MNIST(nn.Module):
    def __init__(self, num_sigmas=10, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            ConditionalBatchNorm2d(base_ch, num_sigmas),
            nn.SiLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            nn.SiLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            nn.SiLU())
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            nn.SiLU())
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            ConditionalBatchNorm2d(base_ch, num_sigmas),
            nn.SiLU())
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            ConditionalBatchNorm2d(base_ch, num_sigmas),
            nn.SiLU())
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, sigma_idx):
        e1 = self.enc1[0](x)
        e1 = self.enc1[1](e1, sigma_idx)
        e1 = self.enc1[2](e1)
        e2 = self.enc2[0](e1)
        e2 = self.enc2[1](e2, sigma_idx)
        e2 = self.enc2[2](e2)
        e3 = self.enc3[0](e2)
        e3 = self.enc3[1](e3, sigma_idx)
        e3 = self.enc3[2](e3)
        b = self.bottleneck[0](e3)
        b = self.bottleneck[1](b, sigma_idx)
        b = self.bottleneck[2](b)
        d1 = self.dec1[0](b)
        d1 = self.dec1[1](d1, sigma_idx)
        d1 = self.dec1[2](d1)
        d2 = self.dec2[0](d1 + e1)
        d2 = self.dec2[1](d2, sigma_idx)
        d2 = self.dec2[2](d2)
        return self.out_conv(d2)


transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

model = NCSN_MNIST(num_sigmas=L).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
sigmas_dev = sigmas.to(device)

print("\n=== Step 2: DSM Training ===")
loss_history = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(device)
        b = x_batch.size(0)

        sigma_idx = torch.randint(0, L, (b,)).to(device)
        sigma_i = sigmas_dev[sigma_idx].view(b, 1, 1, 1)

        z = torch.randn_like(x_batch)
        x_noisy = x_batch + sigma_i * z

        pred = model(x_noisy, sigma_idx)
        target = -z / sigma_i

        loss = torch.mean((pred - target) ** 2 * sigma_i ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('DSM Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('损失曲线.png', dpi=150, bbox_inches='tight')
plt.close()
print("Loss curve saved.")


def annealed_langevin_sample(model, sigmas, n_samples=16, T=100, eps=2e-5):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device) * sigmas[0]
    L_local = len(sigmas)

    with torch.no_grad():
        for i in range(L_local - 1, -1, -1):
            alpha = eps * (sigmas[i] / sigmas[-1]) ** 2
            sigma_idx = torch.full((n_samples,), i, dtype=torch.long).to(device)
            for t in range(T):
                score = model(x, sigma_idx)
                z = torch.randn_like(x)
                x = x + alpha / 2 * score + torch.sqrt(alpha) * z

    return torch.clamp(x, 0, 1)


print("\n=== Step 3: Annealed Langevin Sampling ===")
n_samples = 16
samples = annealed_langevin_sample(model, sigmas_dev, n_samples=n_samples, T=100, eps=2e-5)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i in range(n_samples):
    row, col = i // 4, i % 4
    axes[row, col].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
    axes[row, col].axis('off')
plt.suptitle('Step 3: Generated Samples')
plt.tight_layout()
plt.savefig('步骤3_生成样本.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated samples saved.")


print("\n=== Step 5: Sampling Trajectory Visualization ===")
model.eval()
x_traj = torch.randn(1, 1, 28, 28).to(device) * sigmas_dev[0]
trajectory = [x_traj[0, 0].cpu().numpy()]

checkpoints = [9, 7, 5, 3, 1, 0]
cp_idx = 0
with torch.no_grad():
    for i in range(L - 1, -1, -1):
        alpha = 2e-5 * (sigmas_dev[i] / sigmas_dev[-1]) ** 2
        sigma_idx = torch.full((1,), i, dtype=torch.long).to(device)
        for t in range(100):
            score = model(x_traj, sigma_idx)
            z = torch.randn_like(x_traj)
            x_traj = x_traj + alpha / 2 * score + torch.sqrt(alpha) * z
        if cp_idx < len(checkpoints) and i == checkpoints[cp_idx]:
            trajectory.append(torch.clamp(x_traj, 0, 1)[0, 0].cpu().numpy())
            cp_idx += 1

fig, axes = plt.subplots(1, len(trajectory), figsize=(12, 3))
for j, (ax, img) in enumerate(zip(axes, trajectory)):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    sigma_labels = ['init', '0.60', '0.36', '0.13', '0.05', '0.01']
    ax.set_title(f'σ≈{sigma_labels[j]}')
plt.suptitle('Step 5: Sampling Trajectory')
plt.tight_layout()
plt.savefig('步骤5_采样轨迹.png', dpi=150, bbox_inches='tight')
plt.close()
print("Sampling trajectory saved.")

print("\nDone!")