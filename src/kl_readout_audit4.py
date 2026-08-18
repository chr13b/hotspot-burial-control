#!/usr/bin/env python3
"""KL audit part 4 — rules out the last alternative explanation: geometry NONLINEARITY.

Parts 1-3 stratify/control on geometry with a LINEAR score. If the burial+nbr+dSASA -> hotspot
relationship were nonlinear, a linear control would underfit and KL (a nonlinear function of the
same structure) could look like it "adds" while only recapitulating geometry curvature. Here the
geometry baseline is a gradient-boosted tree (cross-fitted by complex) and every test is repeated.

  python3 src/kl_readout_audit4.py --out results/kl_readout_audit4.csv
"""

import numpy as np, pandas as pd, sys
sys.path.insert(0,'src')
from kl_readout_audit import sauc, auc, z, cv_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
SEED=20260803; NB=2000
j=pd.read_csv("results/kl_detector_joined.csv"); j=j[j.is_interface==1].copy(); j["icode"]=j.icode.fillna("").astype(str)
pos=pd.read_csv("results/p0_positions.csv",usecols=["complex_id","chain","resnum","icode","drsasa"])
pos["icode"]=pos.icode.fillna("").astype(str)
j=j.merge(pos,on=["complex_id","chain","resnum","icode"],how="left").rename(
    columns={"drsasa":"dsasa","is_hot":"y"}).dropna(subset=["dsasa","kl","burial","nbr","y"]).reset_index(drop=True)
y=j.y.to_numpy(float); g=j.complex_id.to_numpy(); ids=np.unique(g)
idx_by={c:np.where(g==c)[0] for c in ids}
Xg=np.column_stack([j.burial,j.nbr,j.dsasa]).astype(float)
kl=j.kl.to_numpy(float)
def cvnl(X):
    o=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        m=HistGradientBoostingClassifier(max_iter=200,max_depth=3,random_state=SEED).fit(X[tr],y[tr])
        o[te]=m.predict_proba(X[te])[:,1]
    return o
sg_lin=cv_score(np.column_stack([z(j.burial),z(j.nbr),z(j.dsasa)]),y,g)
sg_nl=cvnl(Xg)
sf_nl=cvnl(np.column_stack([Xg,kl]))
print(f"geometry AUROC  linear={auc(sg_lin,y):.4f}   NONLINEAR(GBM)={auc(sg_nl,y):.4f}")
print(f"geometry+KL AUROC NONLINEAR={auc(sf_nl,y):.4f}   dAUROC={auc(sf_nl,y)-auc(sg_nl,y):+.4f}")
bd=[]
for _ in range(NB):
    rng=np.random.default_rng(SEED+_)
    t=np.concatenate([idx_by[c] for c in rng.choice(ids,len(ids),True)]); yy=y[t]
    if 0<yy.sum()<len(yy): bd.append(auc(sf_nl[t],yy)-auc(sg_nl[t],yy))
bd=np.array(bd); print(f"   nonlinear nested dAUROC CI [{np.percentile(bd,2.5):+.4f},{np.percentile(bd,97.5):+.4f}] P(>0)={np.mean(bd>0):.3f}")
rng=np.random.default_rng(SEED)
for nb in (20,80):
    k=pd.qcut(pd.Series(sg_nl).rank(method="first"),nb,labels=False).to_numpy().astype(np.int64)
    a_kl,a_geo=sauc(kl,y,k),sauc(sg_nl,y,k)
    bk,bdd=[],[]
    for _ in range(NB):
        t=np.concatenate([idx_by[c] for c in rng.choice(ids,len(ids),True)])
        v1,v2=sauc(kl[t],y[t],k[t]),sauc(sg_nl[t],y[t],k[t])
        if np.isfinite(v1) and np.isfinite(v2): bk.append(v1); bdd.append(v1-v2)
    lk,hk=np.percentile(bk,[2.5,97.5]); ld,hd=np.percentile(bdd,[2.5,97.5])
    print(f"  NONLINEAR-geometry strata {nb:3d} bins: KL sAUROC={a_kl:.4f} [{lk:.4f},{hk:.4f}]  "
          f"leak={a_geo:.4f}  KL-leak={a_kl-a_geo:+.4f} [{ld:+.4f},{hd:+.4f}]")

rows=[dict(test="geometry_baseline_AUROC", linear=round(auc(sg_lin,y),4), nonlinear=round(auc(sg_nl,y),4)),
      dict(test="nested_dAUROC_KL_over_NONLINEAR_geometry", value=round(auc(sf_nl,y)-auc(sg_nl,y),4),
           lo=round(float(np.percentile(bd,2.5)),4), hi=round(float(np.percentile(bd,97.5)),4),
           p_gt0=round(float(np.mean(bd>0)),4))]
rng=np.random.default_rng(SEED)
for nb in (20,80):
    k=pd.qcut(pd.Series(sg_nl).rank(method="first"),nb,labels=False).to_numpy().astype(np.int64)
    a_kl,a_geo=sauc(kl,y,k),sauc(sg_nl,y,k); bk,bdd=[],[]
    for _ in range(NB):
        t=np.concatenate([idx_by[c] for c in rng.choice(ids,len(ids),True)])
        v1,v2=sauc(kl[t],y[t],k[t]),sauc(sg_nl[t],y[t],k[t])
        if np.isfinite(v1) and np.isfinite(v2): bk.append(v1); bdd.append(v1-v2)
    rows.append(dict(test=f"KL_within_NONLINEAR_geometry_strata_{nb}bins", value=round(a_kl,4),
        lo=round(float(np.percentile(bk,2.5)),4), hi=round(float(np.percentile(bk,97.5)),4),
        leakage_benchmark=round(a_geo,4), minus_leakage=round(a_kl-a_geo,4),
        ml_lo=round(float(np.percentile(bdd,2.5)),4), ml_hi=round(float(np.percentile(bdd,97.5)),4),
        p_gt_leak=round(float(np.mean(np.array(bdd)>0)),4)))
o=pd.DataFrame(rows); o["seed"]=SEED; o["n_boot"]=NB; o["n_pos"]=len(y); o["n_hot"]=int(y.sum()); o["n_complexes"]=len(ids)
o["note"]="nonlinear (HistGradientBoosting) geometry baseline; rules out geometry curvature as the explanation for KL's within-stratum signal"
o["command"]="python3 src/kl_readout_audit4.py"
o.to_csv("results/kl_readout_audit4.csv", index=False)
print("wrote results/kl_readout_audit4.csv")
