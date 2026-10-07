import numpy as np

def gaussian_mixture_score(x, y):
    mu1 = np.array([-2., -2.]); mu2 = np.array([2., 2.])
    S1 = np.array([[1., 0.3], [0.3, 1.]]); S2 = np.array([[1., -0.3], [-0.3, 1.]])
    inv_S1 = np.linalg.inv(S1); inv_S2 = np.linalg.inv(S2)
    det_S1 = np.linalg.det(S1); det_S2 = np.linalg.det(S2)
    xv = np.array([x, y])
    def gd(xv, mu, inv_S, det_S):
        d = xv - mu
        return np.exp(-0.5 * d @ inv_S @ d) / (2 * np.pi * np.sqrt(det_S))
    p1 = gd(xv, mu1, inv_S1, det_S1); p2 = gd(xv, mu2, inv_S2, det_S2)
    p = 0.5 * p1 + 0.5 * p2
    grad = (0.5*p1*(-inv_S1 @ (xv-mu1)) + 0.5*p2*(-inv_S2 @ (xv-mu2))) / p
    return grad

def logp(x, y):
    mu1 = np.array([-2., -2.]); mu2 = np.array([2., 2.])
    S1 = np.array([[1., 0.3], [0.3, 1.]]); S2 = np.array([[1., -0.3], [-0.3, 1.]])
    xv = np.array([x, y])
    p1 = np.exp(-0.5*(xv-mu1).T@np.linalg.inv(S1)@(xv-mu1))/(2*np.pi*np.sqrt(np.linalg.det(S1)))
    p2 = np.exp(-0.5*(xv-mu2).T@np.linalg.inv(S2)@(xv-mu2))/(2*np.pi*np.sqrt(np.linalg.det(S2)))
    return np.log(0.5*p1+0.5*p2)

# 1) 解析梯度 vs 数值梯度
print("=== 1) 解析梯度 vs 数值梯度 (应≈1e-6) ===")
rng = np.random.default_rng(0)
errs = []
for _ in range(2000):
    pt = rng.uniform(-5,5,2)
    g = gaussian_mixture_score(*pt)
    h = 1e-5
    gx = (logp(pt[0]+h,pt[1])-logp(pt[0]-h,pt[1]))/(2*h)
    gy = (logp(pt[0],pt[1]+h)-logp(pt[0],pt[1]-h))/(2*h)
    errs.append(np.linalg.norm(g-np.array([gx,gy])))
print("max abs err:", max(errs), " mean:", np.mean(errs))

# 2) 得分零点：在原点应为鞍点(≈0)，众数附近应≈0
print("\n=== 2) 得分场零点检查 ===")
print("score(0,0)      =", gaussian_mixture_score(0,0), "(期望≈0, 鞍点)")
print("score(-2,-2)    =", gaussian_mixture_score(-2,-2))
print("score(2,2)      =", gaussian_mixture_score(2,2))
# 用梯度下降找真实众数（局部）
from scipy.optimize import minimize
def neg(f): return lambda v: -f(v[0],v[1])
r1 = minimize(neg(lambda a,b: np.exp(logp(a,b))), [-1.5,-1.5])
r2 = minimize(neg(lambda a,b: np.exp(logp(a,b))), [1.5,1.5])
print("真实众数1:", r1.x, " 真实众数2:", r2.x)
print("score(真实众数1) =", gaussian_mixture_score(*r1.x))

# 3) ULA采样分布是否匹配目标GMM
print("\n=== 3) ULA 采样统计 ===")
np.random.seed(42)
def ULA(niter, delta, x0):
    s = np.zeros((niter,2)); X = x0.copy()
    for i in range(niter):
        sc = gaussian_mixture_score(X[0],X[1]); Z = np.random.randn(2)
        X = X + delta*sc + np.sqrt(2*delta)*Z; s[i]=X
    return s
samples = ULA(100000, 0.1, np.array([0.,0.]))[20000:]
print("样本数:", len(samples))
# 分量归属：分到两峰的比例（按更靠近哪个众数）
d1 = np.linalg.norm(samples-np.array([-2,-2]),axis=1)
d2 = np.linalg.norm(samples-np.array([2,2]),axis=1)
frac1 = np.mean(d1<d2)
print("归属峰1(-2,-2)比例:", round(frac1,3), " 峰2比例:", round(1-frac1,3), "(期望≈0.5/0.5)")
# 经验均值/协方差 vs 理论
mean_emp = samples.mean(0)
cov_emp = np.cov(samples.T)
# 理论: 0.5 N(mu1,S1)+0.5 N(mu2,S2): mean=0.5mu1+0.5mu2=0; cov=0.5(S1+S2)+0.5(mu1 mu2 spread)
mu1=np.array([-2.,-2.]); mu2=np.array([2.,2.])
S1=np.array([[1.,0.3],[0.3,1.]]); S2=np.array([[1.,-0.3],[-0.3,1.]])
mean_th = 0.5*mu1+0.5*mu2
cov_th = 0.5*(S1+S2) + 0.5*np.outer(mu1-mean_th,mu1-mean_th)+0.5*np.outer(mu2-mean_th,mu2-mean_th)
print("经验均值:", np.round(mean_emp,3), " 理论均值:", mean_th)
print("经验协方差:\n", np.round(cov_emp,3))
print("理论协方差:\n", np.round(cov_th,3))
