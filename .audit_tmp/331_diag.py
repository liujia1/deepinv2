import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

np.random.seed(42)
n = 128
x = resize(data.camera(), (n, n))
x_std = np.std(x)
DATA_RANGE = x.max() - x.min()
sigma_noise = 0.1
y = x + sigma_noise * np.random.randn(n, n)

print("mean(x) =", np.mean(x))
print("var(x)  =", np.var(x), " std =", x_std)
print("E[x^2] (second moment about ORIGIN) =", np.mean(x**2))
print("lam_bayes theory sigma^2/var(x) =", sigma_noise**2/np.var(x))
print("lam if use second moment sigma^2/E[x^2] =", sigma_noise**2/np.mean(x**2))

# brute force scan of true MSE (not PSNR) to find empirical argmin
lams = np.logspace(-4, 1, 20000)
mse = [np.mean((y/(1+L) - x)**2) for L in lams]
L_emp = lams[int(np.argmin(mse))]
print("empirical argmin-MSE lam =", round(L_emp,4))
# PSNR argmin
psnr = [peak_signal_noise_ratio(x, y/(1+L), data_range=DATA_RANGE) for L in lams]
L_psnr = lams[int(np.argmax(psnr))]
print("empirical argmax-PSNR lam =", round(L_psnr,4))

# analytic: MSE(lam)=(sigma^2 - 2 lam mean(eps x) + lam^2 E[x^2])/(1+lam)^2
eps = y - x
c = np.mean(eps*x); m2 = np.mean(x**2); s2 = sigma_noise**2
# solve derivative: minimize (s2 -2 lam c + lam^2 m2)/(1+lam)^2
import numpy.polynomial as P
# derivative numerator: ( -2c + 2 lam m2)(1+lam)^2 - (s2 -2lam c + lam^2 m2)*2(1+lam)=0
# divide 2(1+lam): (-2c+2 lam m2)(1+lam)/... let's just solve numerically via quadratic after expand
# Expand f= (s2 -2 c lam + m2 lam^2); g=(1+lam)^2 ; d(f/g)=0 -> f' g - f g'=0
# f' = -2c + 2 m2 lam ; g' = 2(1+lam)
# (-2c+2m2 lam)(1+lam)^2 - (s2-2c lam+m2 lam^2)*2(1+lam)=0 ; divide 2(1+lam):
# (-c+m2 lam)(1+lam) - (s2-2c lam+m2 lam^2)=0
# -c -c lam + m2 lam + m2 lam^2 - s2 + 2c lam - m2 lam^2 =0
# -c + c lam + m2 lam - s2 =0 -> lam(c+m2)= c + s2 -> lam=(c+s2)/(c+m2)
lam_analytic = (c + s2)/(c + m2)
print("analytic empirical argmin lam (misspecified toward 0) =", round(lam_analytic,4))
print("c=mean(eps*x)=", round(c,5))
