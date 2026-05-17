"""
实验13.3：Classifier-Free Guidance (CFG) 实现（DT5）
对应章节：13.4.2（Classifier-Free Guidance）
参考素材：07-classifier-free-guidance.ipynb (diffusion-tutorials-master)
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

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, base_ch=64):
        super().__init__()
        emb_dim = base_ch * 4
        self.label_emb = nn.Embedding(num_classes, emb_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(base_ch),
            nn.Linear(base_ch, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))

        self.down1 = ResBlock(1, base_ch, emb_dim)
        self.down2 = ResBlock(base_ch, base_ch * 2, emb_dim)
        self.down3 = ResBlock(base_ch * 2, base_ch * 2, emb_dim)
        self.bottleneck = ResBlock(base_ch * 2, base_ch * 2, emb_dim)
        self.up1 = ResBlock(base_ch * 4, base_ch, emb_dim)
        self.up2 = ResBlock(base_ch * 2, base_ch, emb_dim)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, t, y_emb):
        t_emb = self.time_emb(t)
        emb = t_emb + y_emb

        h1 = self.down1(x, emb)
        h2 = self.down2(F.avg_pool2d(h1, 2), emb)
        h3 = self.down3(F.avg_pool2d(h2, 2), emb)
        hb = self.bottleneck(F.avg_pool2d(h3, 2), emb)

        u1 = self.up1(torch.cat([F.interpolate(hb, scale_factor=2), h3], 1), emb)
        u2 = self.up2(torch.cat([F.interpolate(u1, scale_factor=2), h2], 1), emb)
        return self.out_conv(F.interpolate(u2, scale_factor=2))


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

model = ConditionalUNet(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

p_uncond = 0.2
num_epochs = 50
sqrt_alpha_bars_dev = sqrt_alpha_bars.to(device)
sqrt_one_minus_alpha_bars_dev = sqrt_one_minus_alpha_bars.to(device)

print("\n=== Step 2: CFG Joint Training ===")
loss_history = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        b = x_batch.size(0)

        t = torch.randint(0, T, (b,)).to(device)
        eps = torch.randn_like(x_batch)
        x_t = (sqrt_alpha_bars_dev[t].view(b, 1, 1, 1) * x_batch +
               sqrt_one_minus_alpha_bars_dev[t].view(b, 1, 1, 1) * eps)

        y_emb = model.label_emb(y_batch)
        mask = (torch.rand(b, device=device) > p_uncond).float().view(-1, 1)
        y_emb = y_emb * mask

        pred = model(x_t, t, y_emb)
        loss = F.mse_loss(pred, eps)

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
plt.ylabel('Loss')
plt.title('CFG Training Loss')
plt.grid(True)
plt.savefig('CFG训练损失.png', dpi=150, bbox_inches='tight')
plt.close()


def ddpm_step(x, eps_pred, t):
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]
    alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)
    sigma_t = torch.sqrt((1 - alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t))

    x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
    x0_pred = torch.clamp(x0_pred, -1, 1)

    mean = (torch.sqrt(alpha_bar_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * x0_pred +
            torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x)
    if t > 0:
        z = torch.randn_like(x)
        return mean + sigma_t * z
    return mean


def cfg_sampling(model, class_id, s=3.0, n_samples=16):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)

    y_cond = model.label_emb(torch.full((n_samples,), class_id, device=device))
    y_uncond = torch.zeros_like(y_cond)

    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=device)
            eps_cond = model(x, t_batch, y_cond)
            eps_uncond = model(x, t_batch, y_uncond)
            eps = (1 + s) * eps_cond - s * eps_uncond
            x = ddpm_step(x, eps, t)

    return torch.clamp(x, 0, 1)


print("\n=== Step 3: CFG Sampling with Different Guidance Scales ===")
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
target_class = 0

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for s_idx, s in enumerate([0, 1, 3, 5]):
    samples = cfg_sampling(model, target_class, s=s, n_samples=4)
    for j in range(4):
        axes[s_idx, j].imshow(samples[j, 0].cpu().numpy(), cmap='gray')
        axes[s_idx, j].axis('off')
        if j == 0:
            axes[s_idx, j].set_ylabel(f's={s}', fontsize=12)
plt.suptitle(f'Step 3: CFG Samples (class={class_names[target_class]})')
plt.tight_layout()
plt.savefig('步骤3_CFG引导尺度对比.png', dpi=150, bbox_inches='tight')
plt.close()
print("CFG guidance scale comparison saved.")


print("\n=== Step 4: CFG vs Standard Conditional ===")
fig, axes = plt.subplots(2, 8, figsize=(12, 4))
for j in range(8):
    samples_s0 = cfg_sampling(model, target_class, s=0, n_samples=1)
    samples_s3 = cfg_sampling(model, target_class, s=3, n_samples=1)
    axes[0, j].imshow(samples_s0[0, 0].cpu().numpy(), cmap='gray')
    axes[0, j].axis('off')
    axes[1, j].imshow(samples_s3[0, 0].cpu().numpy(), cmap='gray')
    axes[1, j].axis('off')
axes[0, 0].set_ylabel('s=0 (standard)', fontsize=10)
axes[1, 0].set_ylabel('s=3 (CFG)', fontsize=10)
plt.suptitle('Step 4: CFG vs Standard Conditional Generation')
plt.tight_layout()
plt.savefig('步骤4_CFGvs标准条件.png', dpi=150, bbox_inches='tight')
plt.close()
print("CFG vs standard comparison saved.")

print("\nDone!")