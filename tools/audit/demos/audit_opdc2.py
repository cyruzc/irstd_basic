import sys, numpy as np
sys.path.insert(0, 'PyIRSTDMetrics')
from scipy import optimize
from skimage import measure

rng = np.random.default_rng(7)
BIG = 1e6

def lp(m):
    lm = measure.label(m, connectivity=2); return lm, measure.regionprops(lm)

def info(g,p,H=32,W=32):
    mm,mp = lp(g); pm,pp = lp(p); M,N=len(mp),len(pp)
    D=np.full((max(M,1),max(N,1)),H*W,float); I=np.zeros_like(D)
    for i,x in enumerate(mp):
        for j,y in enumerate(pp):
            D[i,j]=np.linalg.norm(np.subtract(x.centroid,y.centroid))
            inter=np.count_nonzero((mm==x.label)&(pm==y.label))
            if inter: I[i,j]=inter/(x.area+y.area-inter)
    return D[:M,:N],I[:M,:N],M,N

def opdc(D,I,ot=0.5,dt=3):
    M,N=D.shape
    if M==0 or N==0: return 0
    vi=I>=ot; vd=D<dt; ms=np.zeros((M,N),bool)
    if vi.any():
        r,c=optimize.linear_sum_assignment(D); ms[r,c]=True; ms&=vi
    if vd.any():
        ur,uc=np.where(ms.sum(1)==0)[0],np.where(ms.sum(0)==0)[0]
        if len(ur) and len(uc):
            r,c=optimize.linear_sum_assignment(D[np.ix_(ur,uc)])
            for a,b in zip(r,c):
                if vd[ur[a],uc[b]]: ms[ur[a],uc[b]]=True
    return int(ms.sum())

def opt(D,I,ot=0.5,dt=3):
    M,N=D.shape
    if M==0 or N==0: return 0
    v=(I>=ot)|(D<dt)
    r,c=optimize.linear_sum_assignment(np.where(v,D,BIG))
    return int(sum(v[i,j] for i,j in zip(r,c)))

# 战场化场景：少量真目标 + 若干杂波块
for n_clutter in (3,6,10):
    diff=tot=0
    for _ in range(400):
        H=W=32; g=np.zeros((H,W),bool)
        for _ in range(rng.integers(1,4)):
            h,w=rng.integers(1,5,2); r,c=rng.integers(0,H-h),rng.integers(0,W-w)
            g[r:r+h,c:c+w]=True
        p=g.copy()
        for _ in range(n_clutter):
            h,w=rng.integers(1,4,2); r,c=rng.integers(0,H-h),rng.integers(0,W-w)
            p[r:r+h,c:c+w]=True
        D,I,M,N=info(g,p); a,b=opdc(D,I),opt(D,I); tot+=1; diff+= a!=b
    print(f"杂波块 {n_clutter:>2} 个：{tot} 例中 OPDC 少算 {diff} 例 ({diff/tot*100:.1f}%)")

print()
print("=== 直接验证修法：把非法对代价设为 BIG ===")
D=np.array([[6.32,24.04,27.66],[8.56,16.19,20.30],[2.69,16.01,19.83]])
I=np.zeros_like(D)
print("OPDC 原始 :", opdc(D,I))
print("惩罚化单阶段:", opt(D,I))
