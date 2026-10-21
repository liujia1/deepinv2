import numpy as np
from scipy.optimize import linear_sum_assignment
def ss(n):
    m=np.random.rand(n)<0.5
    return np.stack([np.where(m,-2,2)+0.3*np.random.randn(n), np.where(m,2,-2)+0.3*np.random.randn(n)],1)
def st(n):
    m=np.random.rand(n)<0.5
    return np.stack([np.where(m,2,-2)+0.3*np.random.randn(n), np.where(m,2,-2)+0.3*np.random.randn(n)],1)
fracs=[]
for s in range(20):
    np.random.seed(s); a=ss(50); b=st(50)
    _,idx=linear_sum_assignment(((a[:,None]-b[None])**2).sum(-1))
    d=b[idx]-a
    frac=(np.abs(d[:,0])>np.abs(d[:,1])).mean()
    fracs.append(frac)
fracs=np.array(fracs)
print("20个种子的OT解中'水平方向'占比:", np.round(fracs,2))
print("完全一致(0或1)的种子数:", int(((fracs<0.02)|(fracs>0.98)).sum()), "/20")
print("混合(0.2<f<0.8)的种子数:", int(((fracs>0.2)&(fracs<0.8)).sum()), "/20")
