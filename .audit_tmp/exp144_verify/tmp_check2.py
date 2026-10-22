import numpy as np
from scipy.optimize import linear_sum_assignment
def ss(n):
    m=np.random.rand(n)<0.5
    return np.stack([np.where(m,-2,2)+0.3*np.random.randn(n), np.where(m,2,-2)+0.3*np.random.randn(n)],1)
def st(n):
    m=np.random.rand(n)<0.5
    return np.stack([np.where(m,2,-2)+0.3*np.random.randn(n), np.where(m,2,-2)+0.3*np.random.randn(n)],1)
np.random.seed(42); a=ss(50); b=st(50)
_,idx=linear_sum_assignment(((a[:,None]-b[None])**2).sum(-1))
d=b[idx]-a
horiz=np.abs(d[:,0])>np.abs(d[:,1])
print("seed=42 实际图: 水平配对", int(horiz.sum()), "/50 ; 垂直配对", int((~horiz).sum()), "/50")
print("平均平方代价", float(np.mean((d**2).sum(1))))
# 交叉计数（线段相交）
from itertools import combinations
b_=b[idx]
def inter(p1,p2,p3,p4):
    def cw(A,B,C): return np.cross(B-A,C-A)
    d1=cw(p3,p4,p1); d2=cw(p3,p4,p2); d3=cw(p1,p2,p3); d4=cw(p1,p2,p4)
    return (d1*d2<0) and (d3*d4<0)
cross=0; tot=0
for i,j in combinations(range(50),2):
    tot+=1
    if inter(a[i],b_[i],a[j],b_[j]): cross+=1
print(f"OT耦合中真正几何相交的路径对: {cross}/{tot} = {100*cross/tot:.1f}%")
