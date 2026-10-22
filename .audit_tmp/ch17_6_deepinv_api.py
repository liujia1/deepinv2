"""验证 17.6.4 修正后的 deepinv 0.4.1 片段能否真正跑通"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from deepinv import Trainer
from deepinv.loss import SureGaussianLoss, SurePGLoss, R2RLoss, SplittingLoss, Neighbor2Neighbor, EILoss
from deepinv.physics import Inpainting
from deepinv.models import UNet
from deepinv.transform import Shift

# --- 损失构造（修正后）---
loss_sure = SureGaussianLoss(sigma=0.1)
loss_pg = SurePGLoss(sigma=0.1, gain=1.0)          # sigma, gain 必填
loss_r2r = R2RLoss()
loss_split = SplittingLoss()
loss_n2n = Neighbor2Neighbor()
loss_ei = EILoss(transform=Shift(shift_max=0.125))  # 参数名 transform，且必须是 Transform 对象
loss_unsure = SureGaussianLoss(sigma=0.1, unsure=True)

# --- 一个最小可跑的训练 ---
torch.manual_seed(0)
x = torch.rand(32, 1, 32, 32)
loader = DataLoader(TensorDataset(x, x), batch_size=8)

physics = Inpainting(img_size=(1, 32, 32), mask=0.5, noise_model=None)
model = UNet(in_channels=1, out_channels=1, scales=2, channels_per_scale=[4, 8])
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = Trainer(
    model=model,
    physics=physics,
    optimizer=opt,
    train_dataloader=loader,     # 必需构造参数
    epochs=1,                    # epochs 在构造时给
    losses=[loss_unsure, loss_ei],
    device="cpu",
    verbose=False,
    show_progress_bar=False,
)
trainer.train()                  # train() 不接受参数
print("OK: Trainer 跑通")
