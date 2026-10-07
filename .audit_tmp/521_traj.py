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
    return (0.5*p1*(-inv_S1 @ (xv-mu1)) + 0.5*p2*(-inv_S2 @ (xv-mu2))) / p

# 完全复刻代码的随机流： np.random.seed(42)，step2无随机，step3循环调用 randn(2)
np.random.seed(42)
niter=10000; delta=0.1; x0=np.array([0.,0.]); burn=2000
samples=np.zeros((niter,2)); X=x0.copy()
for i in range(niter):
    sc=gaussian_mixture_score(X[0],X[1]); Z=np.random.randn(2)
    X=X+delta*sc+np.sqrt(2*delta)*Z; samples[i]=X

def which(s):
    d1=np.linalg.norm(s-np.array([-2,-2]),axis=1); d2=np.linalg.norm(s-np.array([2,2]),axis=1)
    return d1<d2  # True=峰1, False=峰2

# 轨迹段 [0:2000]
w_traj = which(samples[:2000])
print("轨迹[0:2000] 峰1点数:",w_traj.sum(),"峰2点数:",(~w_traj).sum())
print("轨迹[0:2000] 是否两峰都访问:", w_traj.any() and (~w_traj).any())
# 散点段 [burn:]
w_scat = which(samples[burn:])
print("散点[2000:10000] 峰1比例:",round(w_scat.mean(),3)," 峰2比例:",round(1-w_scat.mean(),3))
print("散点段两峰都访问:", w_scat.any() and (~w_scat.any()))
# 统计模式切换次数
switch = np.sum(w_traj[1:]!=w_traj[:-1])
switch_all = np.sum(which(samples)[1:]!=which(samples)[:-1])
print("前2000步模式切换次数:",switch," 全10000步切换次数:",switch_all)
# 终点
print("终点 samples[-1]:", np.round(samples[-1],2))
print("起点附近 samples[0]:", np.round(samples[0],2))
