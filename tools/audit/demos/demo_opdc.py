import sys, numpy as np
sys.path.insert(0, 'PyIRSTDMetrics')
from skimage import measure
from py_irstd_metrics.targetwise_metrics import OPDCMatching, DistanceOnlyMatching

def blobs(boxes, H=64, W=64):
    m = np.zeros((H, W), dtype=bool)
    for (r0, c0, h, w) in boxes: m[r0:r0+h, c0:c0+w] = True
    return m

def props(m): return measure.regionprops(measure.label(m.astype(int), connectivity=2))

def iget(a,b):
    """pixel IoU between two boolean masks"""
    return (a&b).sum()/max((a|b).sum(),1)

def tp_of(mask, pred, matcher, **kw):
    mm = measure.label(mask.astype(int), connectivity=2); mp = props(mask)
    pm = measure.label(pred.astype(int), connectivity=2); pp = props(pred)
    st, iou = matcher(**kw)(pm, mm, pp, mp)
    return int(np.count_nonzero(st)), pp, mp, st, iou

def cdist(mask, pred):
    return np.linalg.norm(np.subtract(props(mask)[0].centroid, props(pred)[0].centroid))

print("="*80)
print("【A】大目标 (20x20) + 整体偏移 3 px   | 质心距 = 3.0")
print("="*80)
mask, pred = blobs([(20,20,20,20)]), blobs([(20,23,20,20)])
print(f"  像素 IoU = {iget(mask,pred):.3f},  质心距 = {cdist(mask,pred):.2f}")
for name, cls, kw in [("距离法(旧) ", DistanceOnlyMatching, dict(distance_threshold=3)),
                      ("OPDC(新)  ", OPDCMatching, dict(overlap_threshold=0.5, distance_threshold=3))]:
    tp,*_ = tp_of(mask, pred, cls, **kw)
    print(f"  {name} TP = {tp}/1   {'✅' if tp else '❌ 判丢'}")

print()
print("="*80)
print("【B】极小目标 (2x2) + 偏移 1 px   | 像素 IoU = 0.33 < 0.5")
print("="*80)
mask, pred = blobs([(20,20,2,2)]), blobs([(20,21,2,2)])
print(f"  像素 IoU = {iget(mask,pred):.3f},  质心距 = {cdist(mask,pred):.2f}")
for name, cls, kw in [("仅重叠准则", OPDCMatching, dict(overlap_threshold=0.5, distance_threshold=1)),
                      ("OPDC 两阶段", OPDCMatching, dict(overlap_threshold=0.5, distance_threshold=3)),
                      ("距离法(旧) ", DistanceOnlyMatching, dict(distance_threshold=3))]:
    try:
        tp,*_ = tp_of(mask, pred, cls, **kw)
        print(f"  {name} TP = {tp}/1   {'✅' if tp else '❌ 判丢'}")
    except Exception as e:
        print(f"  {name} -> {type(e).__name__}: {e}")

print()
print("="*80)
print("【C】贪心 vs 匈牙利：旧 MatchingBasedMetrics 内部是贪心、依赖遍历顺序")
print("="*80)
# GT1 与 GT2 相距 3px；pred1 同时落在两者 3px 内；pred2 只落在 GT1 附近
mask = blobs([(10,10,1,1), (10,13,1,1)])
pred = blobs([(10,11,1,1), (12,9,1,1)])
mp, pp = props(mask), props(pred)
print("  GT  质心:", [tuple(np.round(p.centroid,1)) for p in mp])
print("  pred质心:", [tuple(np.round(p.centroid,1)) for p in pp])
D = np.array([[np.linalg.norm(np.subtract(g.centroid,p.centroid)) for p in pp] for g in mp])
print("  距离矩阵:\n", np.round(D,2))
tp,*_ = tp_of(mask, pred, DistanceOnlyMatching, distance_threshold=3)
print(f"  旧贪心匹配 TP = {tp} / 2  (GT 顺序: 先到先得，pred1 被 GT1 抢走 -> GT2 落空)")
tp,*_ = tp_of(mask, pred, OPDCMatching, overlap_threshold=0.5, distance_threshold=3)
print(f"  OPDC      TP = {tp} / 2")
