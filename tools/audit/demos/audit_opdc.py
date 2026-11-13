"""审计 OPDC 实现：两阶段"先分配后过滤"是否会漏匹配？
对照：把非法对代价设为 BIG 后单次匈牙利（等价于在可接受对上求最大基数+最小距离）。
"""
import sys, numpy as np, itertools
sys.path.insert(0, 'PyIRSTDMetrics')
from scipy import optimize
from skimage import measure
from py_irstd_metrics.targetwise_metrics import OPDCMatching

rng = np.random.default_rng(0)
BIG = 1e6

def label_props(m):
    lm = measure.label(m, connectivity=2)
    return lm, measure.regionprops(lm)

def pair_info(mask, pred, H, W):
    mm, mp = label_props(mask); pm, pp = label_props(pred)
    M, N = len(mp), len(pp)
    D = np.full((max(M,1), max(N,1)), H*W, float); I = np.zeros_like(D)
    for i,g in enumerate(mp):
        for j,p in enumerate(pp):
            D[i,j] = np.linalg.norm(np.subtract(g.centroid, p.centroid))
            inter = np.count_nonzero((mm==g.label)&(pm==p.label))
            if inter: I[i,j] = inter/(g.area+p.area-inter)
    return D, I, M, N

def opdc_tp(D, I, M, N, ot=0.5, dt=3):
    D, I = D[:M,:N], I[:M,:N]
    if M==0 or N==0: return 0
    vi = I >= ot; vd = D < dt
    ms = np.zeros((M,N), bool)
    if vi.any():
        r,c = optimize.linear_sum_assignment(D); ms[r,c] = True; ms &= vi
    if vd.any():
        ur = np.where(ms.sum(1)==0)[0]; uc = np.where(ms.sum(0)==0)[0]
        if len(ur) and len(uc):
            sub = D[np.ix_(ur,uc)]
            r,c = optimize.linear_sum_assignment(sub)
            for ri,ci in zip(r,c):
                if vd[ur[ri], uc[ci]]: ms[ur[ri], uc[ci]] = True
    return int(ms.sum()), ms

def optimal_tp(D, I, M, N, ot=0.5, dt=3):
    D, I = D[:M,:N], I[:M,:N]
    if M==0 or N==0: return 0
    valid = (I >= ot) | (D < dt)
    cost = np.where(valid, D, BIG)
    r,c = optimize.linear_sum_assignment(cost)
    return int(sum(valid[i,j] for i,j in zip(r,c)))

def make_case():
    H=W=32
    g = np.zeros((H,W), bool)
    nt = rng.integers(1,4)
    for _ in range(nt):
        h,w = rng.integers(1,6,2); r,c = rng.integers(0,H-h), rng.integers(0,W-w)
        g[r:r+h, c:c+w] = True
    p = np.zeros((H,W), bool)
    mode = rng.integers(0,4)
    if mode==0:   # 目标整体平移/缩放
        for (r0,c0,h,w) in [(0,0,0,0)]: pass
        lm,_ = label_props(g)
        for k in range(1, lm.max()+1):
            ys,xs = np.where(lm==k)
            dr,dc = rng.integers(-4,5,2)
            p[np.clip(ys+dr,0,H-1), np.clip(xs+dc,0,W-1)] = True
    elif mode==1: # 随机块
        for _ in range(rng.integers(1,4)):
            h,w = rng.integers(1,6,2); r,c = rng.integers(0,H-h), rng.integers(0,W-w)
            p[r:r+h, c:c+w] = True
    elif mode==2: # 目标 + 随机虚警
        p = g.copy()
        for _ in range(rng.integers(1,3)):
            h,w = rng.integers(1,4,2); r,c = rng.integers(0,H-h), rng.integers(0,W-w)
            p[r:r+h, c:c+w] = True
    else:         # 目标膨胀
        lm,_ = label_props(g)
        for k in range(1, lm.max()+1):
            ys,xs = np.where(lm==k)
            p[ys,xs]=True; p[np.clip(ys+1,0,H-1),xs]=True; p[ys,np.clip(xs+1,0,W-1)]=True
    return g,p,H,W

diff = 0; tot = 0; examples=[]
for _ in range(400):
    g,p,H,W = make_case()
    if g.sum()==0 and p.sum()==0: continue
    D,I,M,N = pair_info(g,p,H,W)
    a,_ = opdc_tp(D,I,M,N); b = optimal_tp(D,I,M,N)
    tot += 1
    if a != b:
        diff += 1
        if len(examples)<3: examples.append((g,p,M,N,a,b,D,I))
print(f"随机用例 {tot} 个：OPDC 的 TP 与「合法对约束下最优分配」不一致 {diff} 个 ({diff/tot*100:.1f}%)")
for g,p,M,N,a,b,D,I in examples:
    print(f"\n  例: GT={M} 个, pred={N} 个, OPDC TP={a}, 最优 TP={b}")
    print("  距离:\n", np.round(D[:M,:N],2))
    print("  IoU:\n", np.round(I[:M,:N],2))
