#!/usr/bin/env python3
"""Design-regime validation of the sequence-free KL hotspot detector (Bennett et al. 2023).

The load-bearing positive of this project is a sequence-free detector: KL between ProteinMPNN's
backbone-only distribution with vs without the partner. Every prior test used SKEMPI crystal / predicted
/ partial-diffusion backbones. This tests it on GENUINELY DE-NOVO binder backbones (RFdiffusion +
ProteinMPNN, the exact staged pipeline we critique) against EXPERIMENTAL per-residue binding labels from
site-saturation mutagenesis (Bennett et al. 2023, Nat Commun; files.ipd.uw.edu/pub/improving_dl_binders_2023).

Label (PROXY, honest): from the pre-computed affinity tables (ngs_data_analysis/affinities/{lib}.sc), per
binder position we measure BINDING RESTRICTIVENESS = fraction of the up-to-20 single substitutions that
lose binding (Kd >= highest tested concentration). A position where almost every mutation kills binding is
a binding hotspot. Caveat: restrictiveness convolves binding with fold/display stability; we restrict to
partner-contacting INTERFACE positions to focus it, and report it as a proxy.

Question: among interface positions, does sequence-free KL (and KL+burial) rank the restrictive
(experimental-hotspot) positions above the tolerant ones -- i.e. does the detector transfer to the design
regime? Reports AUROC(KL), AUROC(burial), AUROC(KL+burial) and the project's key statistic
dAUROC(KL+burial - burial), bootstrapped over DESIGNS. Recovery deficit is NOT tested (these sequences ARE
ProteinMPNN's output, so recovery is trivially high) -- only the detector, per denovo-datasets memory.

POSITIVE CONTROLS (gate the zero, CLAUDE.md 6): (1) PDB-native residues bind (validates pos->resnum map);
(2) burial itself predicts restrictiveness; (3) KL is non-degenerate and elevated at the interface.

  python3 src/bennett_kl_detector.py --out results/bennett_kl_detector.csv
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

SEED = 20260803
AAs = set("ACDEFGHIKLMNPQRSTVWY")
BEN = os.path.expanduser("~/ftax/bennett/x/supplemental_files")
LIBS = ["ALK_SSM1", "ALK_SSM2", "IL2Ra_SSM1", "IL2Ra_SSM2", "LTK_SSM1", "LTK_SSM2", "IL10Ra_SSM"]


def parse_desc(desc):
    t = desc.split("_")
    if len(t) >= 3 and t[-2].isdigit() and t[-1] in AAs:
        return "_".join(t[:-2]), int(t[-2]), t[-1]
    return None, None, None


def kl_complex_vs_binder(model, pdb, parent):
    """KL(p(.|complex bb) || p(.|binder-alone bb)) per binder(chain A) position. Backbone-only."""
    cx = fc.load_complex(pdb, parent, "A", "B", require_both=False)
    if cx is None or not (cx.group == 1).any():
        return None
    def dists(lp):
        z = lp[:, :20]; z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)
    P = dists(fc.mpnn_unconditional_logprobs(model, cx))
    mono = fc.load_complex(pdb, parent, "A", "", require_both=False)
    if mono is None:
        return None
    Qm = dists(fc.mpnn_unconditional_logprobs(model, mono))
    idx = {(c, int(r), i): k for k, (c, r, i) in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
    eps = 1e-12
    rows = []
    for j in range(cx.n):
        if cx.group[j] != 1:
            continue
        k = idx.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
        if k is None:
            continue
        kl = float((P[j] * (np.log(P[j] + eps) - np.log(Qm[k] + eps))).sum())
        rows.append((int(cx.resnums[j]), cx.seq[j], kl))
    return cx, rows


def burial_interface(pdb, parent):
    """rSASA in complex (burial) and dSASA-on-binding (interface) per chain-A residue."""
    sc = fc.residue_sasa(pdb, parent, "AB")   # complex context
    sm = fc.residue_sasa(pdb, parent, "A")    # binder alone
    out = {}
    for (ch, rn, ic), a_complex in sc.items():
        if ch != "A":
            continue
        out[int(rn)] = dict(sasa_complex=a_complex, sasa_mono=sm.get((ch, rn, ic), a_complex))
    return out


def restrictiveness(aff):
    """Per (parent,pos): fraction of tested substitutions that LOSE binding, + continuous log-Kd."""
    pr = aff["description"].map(parse_desc)
    aff = aff.assign(parent=[p[0] for p in pr], pos=[p[1] for p in pr], mut=[p[2] for p in pr])
    aff = aff[aff.parent.notna()].copy()
    aff["kd"] = pd.to_numeric(aff["kd_lb"], errors="coerce")
    cap = pd.to_numeric(aff["highest_conc"], errors="coerce").median()
    if not np.isfinite(cap):
        cap = 1000.0
    aff["binds"] = np.isfinite(aff["kd"]) & (aff["kd"] < cap)
    aff["logkd"] = np.log10(np.clip(aff["kd"].fillna(10 * cap), 1e-3, 10 * cap))
    # dedupe replicate rows to one value per (parent,pos,aa)
    per_sub = aff.groupby(["parent", "pos", "mut"]).agg(binds=("binds", "max"), logkd=("logkd", "median")).reset_index()
    lab = {}
    for (parent, pos), g in per_sub.groupby(["parent", "pos"]):
        n = len(g)
        if n < 8:                      # need enough substitutions to be a reliable column
            continue
        missing = AAs - set(g.mut)     # SSM tests the 19 NON-native aa; native = the excluded one
        native_aa = missing.pop() if len(missing) == 1 else None
        lab[(parent, int(pos))] = dict(restr=1.0 - g.binds.mean(), mean_logkd=float(g.logkd.mean()),
                                       n_sub=n, native_aa=native_aa)
    return lab, cap


def boot_auc(df, score, label="is_hot", nboot=2000, unit="parent"):
    rng = np.random.default_rng(SEED)
    ids = df[unit].unique()
    by = {u: df.loc[df[unit] == u, [score, label]].values for u in ids}

    def auc(a):
        s, y = a[:, 0], a[:, 1]; m = np.isfinite(s); s, y = s[m], y[m]
        if y.sum() == 0 or y.sum() == len(y):
            return np.nan
        r = stats.rankdata(s); n1 = y.sum(); n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pt = auc(np.concatenate([by[u] for u in ids]))
    b = np.array([auc(np.concatenate([by[ids[i]] for i in rng.choice(len(ids), len(ids), True)])) for _ in range(nboot)])
    return float(pt), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5)), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bennett_kl_detector.csv")
    ap.add_argument("--pos-out", default="results/bennett_kl_positions.csv")
    ap.add_argument("--hot-thresh", type=float, default=0.75)
    a = ap.parse_args()

    pdb_index = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{BEN}/design_models_ssm_natives/*/*.pdb")}
    print(f"[bennett] native design PDBs indexed: {len(pdb_index)}")
    model, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))

    recs = []
    for lib in LIBS:
        af_path = f"{BEN}/ngs_data_analysis/affinities/{lib}.sc"
        if not os.path.exists(af_path):
            print(f"  [{lib}] no affinity file, skip"); continue
        aff = pd.read_csv(af_path, sep=r"\s+", engine="python")
        lab, cap = restrictiveness(aff)
        parents = sorted({p for (p, _) in lab})
        used = 0
        for parent in parents:
            pdb = pdb_index.get(parent)
            if pdb is None:
                continue
            try:
                res = kl_complex_vs_binder(model, pdb, parent)
                if res is None:
                    continue
                cx, klrows = res
                bi = burial_interface(pdb, parent)
                nbr = fc.neighbour_counts(cx)
                nbr_by_rn = {int(cx.resnums[j]): int(nbr[j]) for j in range(cx.n) if cx.group[j] == 1}
            except Exception as e:
                print(f"    skip {parent[:30]}: {type(e).__name__}: {e}"); continue
            for rn, restype, kl in klrows:
                key = (parent, rn)
                if key not in lab or rn not in bi:
                    continue
                sc, sm = bi[rn]["sasa_complex"], bi[rn]["sasa_mono"]
                rsasa = fc.relative_sasa(sc, restype)
                dsasa = sm - sc
                L = lab[key]
                recs.append(dict(lib=lib, parent=parent, resnum=rn, restype=restype,
                                 kl=kl, burial=-(rsasa if np.isfinite(rsasa) else 0.0),
                                 rsasa=rsasa, dsasa=dsasa, nbr=nbr_by_rn.get(rn, np.nan),
                                 is_interface=int(dsasa > 1.0), restr=L["restr"],
                                 mean_logkd=L["mean_logkd"], n_sub=L["n_sub"],
                                 native_aa=L["native_aa"],
                                 native_match=int(L["native_aa"] == restype) if L["native_aa"] else 0))
            used += 1
        print(f"  [{lib}] parents used {used}/{len(parents)}  (cap={cap:.0f}nM)")

    df = pd.DataFrame(recs)
    df.to_csv(a.pos_out, index=False)

    print("\n=== POSITIVE CONTROLS (gate the zero) ===")
    have_nat = df[df.native_aa.notna()]
    match = have_nat.native_match.mean()
    print(f"  (1) MAPPING: SSM-excluded aa == PDB-native residue: {match:.3f} "
          f"({int(have_nat.native_match.sum())}/{len(have_nat)}) -- pos->resnum alignment OK if ~1.0")
    df = df[df.native_match == 1].copy()      # keep only mapping-confirmed positions
    iface = df[df.is_interface == 1].copy()
    iface["is_hot"] = (iface.restr >= a.hot_thresh).astype(int)
    print(f"  (2) after mapping filter: {len(iface)} interface positions across {iface.parent.nunique()} designs "
          f"{iface.groupby('lib').parent.nunique().to_dict()}")
    print(f"  (3) label non-degenerate: hotspots (restr>={a.hot_thresh}) = {int(iface.is_hot.sum())} "
          f"({iface.is_hot.mean():.2f}); KL mean iface {iface.kl.mean():.3f} vs non-iface "
          f"{df[df.is_interface==0].kl.mean():.3f}")

    print("\n=== DETECTOR (interface positions; bootstrap over designs) ===")
    rows = []
    for name, col in [("burial (-rSASA) BASELINE", "burial"), ("KL(complex||binder)", "kl"),
                      ("nbr", "nbr"), ("dSASA(interface size)", "dsasa")]:
        pt, lo, hi, _ = boot_auc(iface, col)
        print(f"  AUROC {name:26s} = {pt:.3f} [{lo:.3f},{hi:.3f}]")
        rows.append(dict(metric=f"auroc_{col}", value=pt, lo=lo, hi=hi, n=len(iface),
                         n_designs=iface.parent.nunique()))
    # KL+burial rank-average, and the key delta over burial
    for name, cols in [("burial+KL", ["burial", "kl"])]:
        iface["_combo"] = np.mean([stats.rankdata(iface[c]) / len(iface) for c in cols], axis=0)
        pt, lo, hi, bcombo = boot_auc(iface, "_combo")
        _, _, _, bbur = boot_auc(iface, "burial")
        d = bcombo - bbur
        dpt = float(np.nanmean(bcombo) - np.nanmean(bbur))
        print(f"  AUROC {name:26s} = {pt:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"  dAUROC(burial+KL - burial)  = {dpt:+.3f} [{np.nanpercentile(d,2.5):+.3f},{np.nanpercentile(d,97.5):+.3f}]  "
              f"P(>0)={np.mean(d>0):.3f}")
        rows.append(dict(metric="auroc_burial+KL", value=pt, lo=lo, hi=hi, n=len(iface), n_designs=iface.parent.nunique()))
        rows.append(dict(metric="dAUROC_KL_over_burial", value=dpt, lo=float(np.nanpercentile(d, 2.5)),
                         hi=float(np.nanpercentile(d, 97.5)), n=len(iface), n_designs=iface.parent.nunique()))
    # continuous: Spearman KL vs restrictiveness (interface)
    sr = stats.spearmanr(iface.kl, iface.restr)
    print(f"  Spearman(KL, restrictiveness) at interface = {sr.correlation:+.3f} p={sr.pvalue:.4f}")
    rows.append(dict(metric="spearman_kl_restr", value=float(sr.correlation), lo="", hi="",
                     n=len(iface), n_designs=iface.parent.nunique()))

    # --- Does KL ADD over burial? The naive rank-avg dAUROC dilutes the stronger predictor;
    #     the fair tests are a partial correlation and a properly-weighted (logistic) model. ---
    def _resid(yv, xv):
        X = np.c_[np.ones(len(xv)), xv]
        return yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
    rk = _resid(stats.rankdata(iface.kl), stats.rankdata(iface.burial.values).reshape(-1, 1))
    rr = _resid(stats.rankdata(iface.restr), stats.rankdata(iface.burial.values).reshape(-1, 1))
    pk = stats.spearmanr(rk, rr)
    cb = stats.spearmanr(iface.kl, iface.burial).correlation
    print(f"  corr(KL,burial)={cb:+.3f}; partial Spearman(KL,restr|burial)={pk.correlation:+.3f} p={pk.pvalue:.2g}")
    rows.append(dict(metric="corr_kl_burial", value=float(cb), lo="", hi="", n=len(iface), n_designs=iface.parent.nunique()))
    rows.append(dict(metric="partial_spearman_kl_given_burial", value=float(pk.correlation), lo="", hi="",
                     n=len(iface), n_designs=iface.parent.nunique()))
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        Zc = lambda c: ((iface[c] - iface[c].mean()) / iface[c].std()).values
        y = iface.is_hot.values; g = iface.parent.values
        X1 = np.c_[Zc("burial")]; X2 = np.c_[Zc("burial"), Zc("kl")]
        o1 = np.zeros(len(y)); o2 = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X1, y, g):
            o1[te] = LogisticRegression(max_iter=2000).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
            o2[te] = LogisticRegression(max_iter=2000).fit(X2[tr], y[tr]).predict_proba(X2[te])[:, 1]
        _auc = lambda s, yy: (stats.rankdata(s)[yy == 1].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (len(yy) - yy.sum()))
        d = _auc(o2, y) - _auc(o1, y)
        rng = np.random.default_rng(SEED); ids = np.unique(g); bl = []
        for _ in range(3000):
            idx = np.concatenate([np.where(g == u)[0] for u in rng.choice(ids, len(ids), True)])
            yy = y[idx]
            if 0 < yy.sum() < len(yy):
                bl.append(_auc(o2[idx], yy) - _auc(o1[idx], yy))
        bl = np.array(bl)
        print(f"  logistic-CV dAUROC(burial+KL - burial) = {d:+.3f} [{np.percentile(bl,2.5):+.3f},{np.percentile(bl,97.5):+.3f}] P(>0)={np.mean(bl>0):.3f}")
        rows.append(dict(metric="logistic_cv_dAUROC_KL_over_burial", value=float(d),
                         lo=float(np.percentile(bl, 2.5)), hi=float(np.percentile(bl, 97.5)),
                         n=len(iface), n_designs=iface.parent.nunique()))
    except ImportError:
        print("  (sklearn absent; partial correlation is the fair 'does KL add' test)")

    out = pd.DataFrame(rows)
    out["hot_thresh"] = a.hot_thresh; out["mapping_match_rate"] = match; out["seed"] = SEED
    out["label"] = "PROXY: SSM binding-restrictiveness; detector-only; de-novo backbones"
    out["command"] = f"python3 src/bennett_kl_detector.py --out {a.out}"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} and {a.pos_out}")


if __name__ == "__main__":
    main()
