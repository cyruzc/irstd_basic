import sys, numpy as np
sys.path.insert(0, 'PyIRSTDMetrics')
from py_irstd_metrics.targetwise_metrics import (MatchingBasedMetrics, DistanceOnlyMatching,
    OPDCMatching, ProbabilityDetectionAndFalseAlarmRate)

def blobs(boxes, H=64, W=64):
    m = np.zeros((H, W), dtype=bool)
    for (r0,c0,h,w) in boxes: m[r0:r0+h, c0:c0+w] = True
    return m

def evaluate(prob, mask, matcher=None):
    mac = MatchingBasedMetrics(matching_method=matcher or OPDCMatching())
    mac.update(prob, mask)
    d = mac.get()
    tp, fp, fn = mac.tp_objs.item(), mac.fp_objs.item(), mac.fn_objs.item()
    seg = mac.tp_ious.item()/tp if tp else 0.0
    loc = tp/(tp+fp+fn) if (tp+fp+fn) else 0.0
    closed = mac.tp_ious.item()/(tp+fp+fn) if (tp+fp+fn) else 0.0
    pd = ProbabilityDetectionAndFalseAlarmRate(); pd.update(prob, mask); d2 = pd.get()
    return dict(tp=tp,fp=fp,fn=fn,sum_iou=mac.tp_ious.item(),seg=seg,loc=loc,hiou=d['hiou'].item(),
                closed=closed, Pd=d2['probability_detection'].item(), Fa=d2['false_alarm'].item())

print("="*84)
print("【D】matched-but-blobby：4x4 GT vs 6x6 同心预测（像素 IoU = 0.444，质心距 = 0）")
print("="*84)
mask = blobs([(20,20,4,4)]); pred = blobs([(19,19,6,6)])
for tag, mt in [("OPDC", None), ("距离法", DistanceOnlyMatching(distance_threshold=3))]:
    r = evaluate(pred.astype(float), mask, mt)
    print(f"  [{tag}] Pd={r['Pd']:.3f}  Fa={r['Fa']:.2e}  seg={r['seg']:.3f}  loc={r['loc']:.3f}  hIoU={r['hiou']:.3f}")
print("  -> 旧 Pd/Fa 判定「命中且零虚警」(Pd=1, Fa=0)；hIoU 只给 0.444。")

print()
print("="*84)
print("【E】hIoU 闭式：hIoU = Σ(匹配对像素IoU) / (TP+FP+FN)")
print("="*84)
cases = {
  "完美命中":      (blobs([(10,10,4,4)]),                     blobs([(10,10,4,4)])),
  "画大了":        (blobs([(10,10,4,4)]),                     blobs([(9,9,6,6)])),
  "多一个虚警":    (blobs([(10,10,4,4)]),                     blobs([(10,10,4,4),(40,40,3,3)])),
  "漏一个目标":    (blobs([(10,10,4,4),(40,40,4,4)]),         blobs([(10,10,4,4)])),
  "两目标粘连":    (blobs([(10,10,4,4),(10,15,4,4)]),         blobs([(10,10,4,9)])),
  "整体偏移2px":   (blobs([(10,10,10,10)]),                   blobs([(10,12,10,10)])),
}
print(f"  {'场景':<12}{'TP/FP/FN':>10}{'ΣIoU':>8}{'seg':>7}{'loc':>7}{'hIoU':>8}{'闭式':>8}{'Pd':>7}{'Fa':>11}")
for name,(mk,pr) in cases.items():
    r = evaluate(pr.astype(float), mk)
    print(f"  {name:<12}{r['tp']}/{r['fp']}/{r['fn']:>6}{r['sum_iou']:>8.3f}{r['seg']:>7.3f}{r['loc']:>7.3f}"
          f"{r['hiou']:>8.4f}{r['closed']:>8.4f}{r['Pd']:>7.3f}{r['Fa']:>11.1e}")

print()
print("="*84)
print("【F】乘法 vs 加法：同一对 (loc, seg) 下两种组合的差异")
print("="*84)
print(f"  {'loc':>5}{'seg':>6}{'乘法 hIoU':>11}{'加法 aIoU':>11}")
for loc,seg in [(1.0,1.0),(1.0,0.5),(0.5,1.0),(0.9,0.6),(0.5,0.5),(1.0,0.3)]:
    print(f"  {loc:>5.1f}{seg:>6.1f}{loc*seg:>11.3f}{0.5*(loc+seg):>11.3f}")
