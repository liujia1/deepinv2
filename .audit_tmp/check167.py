import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon

N = 128
ph = shepp_logan_phantom()
x_true = resize(ph, (N, N), anti_aliasing=True).astype(np.float64)
x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min())


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse)


for name, theta in {
    'full (180)': np.linspace(0, 180, 180, endpoint=False),
    'sparse (30)': np.linspace(0, 180, 30, endpoint=False),
    'very sparse (12)': np.linspace(0, 180, 12, endpoint=False),
}.items():
    sino = radon(x_true, theta=theta, circle=False)
    rec_fbp = iradon(sino, theta=theta, circle=False, filter_name='ramp')
    rec_bp = iradon(sino, theta=theta, circle=False, filter_name=None)
    print(f'[{name}] sino shape={sino.shape} sino_max={sino.max():.2f}')
    print(f'   bp: min={rec_bp.min():.2f} max={rec_bp.max():.2f} mean={rec_bp.mean():.2f} '
          f'frac>1={np.mean(rec_bp > 1.0):.4f}')
    print(f'   FBP clip PSNR={psnr(x_true, np.clip(rec_fbp, 0, 1)):.2f} dB | '
          f'BP clip PSNR={psnr(x_true, np.clip(rec_bp, 0, 1)):.2f} dB')
    # 尺度校正后的反投影（最优点乘）
    s = float((rec_bp * x_true).sum() / (rec_bp ** 2).sum())
    print(f'   scale-corrected BP PSNR={psnr(x_true, np.clip(s * rec_bp, 0, 1)):.2f} dB (s={s:.4f})')
