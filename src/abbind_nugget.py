#!/usr/bin/env python3
"""AB-Bind second fixture — does the nugget ("confidence is not competence") replicate on antibody-antigen
ΔΔG, killing the single-fixture objection for the core claim?

AB-Bind (1101 mutants / 32 complexes; Sirin et al. 2016) gives per-mutation ΔΔG(kcal/mol) on antibody-antigen
and related interfaces — a SKEMPI-class fixture with different biophysics. We replicate the SKEMPI nugget:
per interface position, hotspot = has a single mutation with ΔΔG ≥ 1 (loose) / ≥ 2 (strict) [destabilising
binding]; then ask whether the model's per-residue CONFIDENCE (unconditional log p(native|backbone), the
sequence-free confidence used in the Bennett analyses) ranks hotspots — vs free geometry (burial, ΔSASA, nbr).
Nugget replicates if
confidence ≈ chance and geometry predicts, and confidence adds ~0 over full geometry.

Reuses the p0 feature machinery (SASA/rSASA/ΔSASA/nbr) and ProteinMPNN scoring for an identical pipeline.
Complex-clustered bootstrap, seed 20260803.
  python3 src/abbind_nugget.py --out results/abbind_nugget.csv
"""
import argparse, csv, os, re, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc
import p0_burial_matched as p0

SEED = 20260803
ABDIR = os.path.expanduser("~/ftax/data/ab-bind")
MUT_RE = re.compile(r"^([A-Za-z0-9]+):([A-Z])(-?\d+)([A-Z])$")


def load_labels():
    d = pd.read_csv(f"{ABDIR}/AB-Bind_experimental_data.csv", encoding="latin-1")
    d = d.rename(columns={"#PDB": "pdb", "Partners(A_B)": "partners", "ddG(kcal/mol)": "ddg"})
    d["ddg"] = pd.to_numeric(d["ddg"], errors="coerce")
    single = d[~d["Mutation"].astype(str).str.contains(",")].copy()
    recs = []
    for _, r in single.iterrows():
        m = MUT_RE.match(str(r["Mutation"]).strip())
        if m and np.isfinite(r["ddg"]):
            recs.append(dict(pdb=r["pdb"], chain=m.group(1), wt=m.group(2), resnum=int(m.group(3)),
                             mut=m.group(4), ddg=float(r["ddg"])))
    mut = pd.DataFrame(recs)
    partners = d.dropna(subset=["partners"]).groupby("pdb").partners.first().to_dict()
    # per-position label: most destabilising single mutation
    g = mut.groupby(["pdb", "chain", "resnum"]).agg(ddg_max=("ddg", "max"),
                                                    ddg_absmax=("ddg", lambda s: s.abs().max()),
                                                    wt=("wt", "first")).reset_index()
    g["hot_loose"] = (g.ddg_max >= 1.0).astype(int)
    g["hot_strict"] = (g.ddg_max >= 2.0).astype(int)
    g["is_null"] = (g.ddg_absmax < 0.25).astype(int)
    return g, partners


def build_features(partners, out_csv):
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    done = set()
    if os.path.exists(out_csv):
        try:
            done = set(pd.read_csv(out_csv).complex_id)
        except Exception:
            done = set()
    fh = open(out_csv, "a" if done else "w", newline=""); writer = None; t0 = time.time()
    for pi, (pdb, part) in enumerate(sorted(partners.items())):
        g1, g2 = part.split("_")
        cid = f"{pdb}_{g1}_{g2}"
        if cid in done:
            continue
        path = f"{ABDIR}/{pdb}.pdb"
        if not os.path.exists(path):
            print(f"  no pdb {pdb}"); continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
            if cx is None or cx.n > 3000:
                print(f"  skip {pdb} (empty/too big)"); continue
            all_ch = g1 + g2
            asa_b_atom = p0.atom_sasa(path, pdb, all_ch)
            asa_b = {}
            for (c, rn, ic, _an), v in asa_b_atom.items():
                asa_b[(c, rn, ic)] = asa_b.get((c, rn, ic), 0.0) + v
            asa_f1 = fc.residue_sasa(path, pdb, g1); asa_f2 = fc.residue_sasa(path, pdb, g2)
            nbr = fc.neighbour_counts(cx)
            lp = fc.mpnn_unconditional_logprobs(model, cx)   # sequence-free confidence (memory-light; robust)
            nat = np.array([fc.MPNN_ALPHABET.index(a) for a in cx.seq])
            for i in range(cx.n):
                key = (cx.chains[i], int(cx.resnums[i]), cx.icodes[i]); aa = cx.seq[i]
                sb = asa_b.get(key, np.nan); sf = (asa_f1 if cx.group[i] == 1 else asa_f2).get(key, np.nan)
                rb, rf = fc.relative_sasa(sb, aa), fc.relative_sasa(sf, aa)
                row = dict(complex_id=cid, pdb=pdb, chain=key[0], resnum=key[1], icode=key[2], aa=aa,
                           rsasa_complex=rb, drsasa=(rf - rb), burial=-rb, nbr=int(nbr[i]),
                           logp_native=float(lp[i, nat[i]]),
                           is_interface=int((rf - rb) > p0.INTERFACE_DRSASA))
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if not done:
                        writer.writeheader()
                writer.writerow(row)
            fh.flush()
            print(f"  [{pi+1}/{len(partners)}] {cid}: {cx.n} res, {time.time()-t0:.0f}s", flush=True)
            del cx, lp, asa_b_atom, asa_b, asa_f1, asa_f2, nbr
            import gc as _gc; _gc.collect()
        except Exception as e:
            print(f"  skip {pdb}: {type(e).__name__}: {e}", flush=True)
    fh.close()


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y); m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    from scipy import stats
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/abbind_nugget.csv")
    ap.add_argument("--positions", default="results/abbind_positions.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    lab, partners = load_labels()
    if not os.path.exists(a.positions):
        print(f"building features for {len(partners)} complexes ...")
        build_features(partners, a.positions)
    pos = pd.read_csv(a.positions)
    pos["icode"] = pos.icode.fillna("").astype(str)

    m = pos.merge(lab, on=["pdb", "chain", "resnum"], how="inner")
    match = float((m.aa == m.wt).mean())                       # positive control: mutation WT == structure aa
    m = m[m.aa == m.wt]                                        # keep only faithfully-mapped positions
    mi = m[m.is_interface == 1].copy()
    mi["is_hot"] = mi.hot_loose
    y = mi.is_hot.to_numpy(); g = mi.complex_id.to_numpy()
    print(f"\npositive control: mutation-WT == structure aa = {match:.3f}  (mapped {len(m)} positions)")
    print(f"interface measured positions: {len(mi)}  complexes {mi.complex_id.nunique()}  "
          f"hot(ΔΔG≥1) {int(y.sum())}  strict(≥2) {int(mi.hot_strict.sum())}")

    def boot(score):
        cids = np.unique(g); idx = {c: np.where(g == c)[0] for c in cids}; out = []
        for _ in range(5000):
            t = np.concatenate([idx[c] for c in rng.choice(cids, len(cids), True)])
            v = auc(score[t], y[t])
            if np.isfinite(v):
                out.append(v)
        return np.percentile(out, [2.5, 97.5])

    rows = []
    for name, s in [("confidence_logp", mi.logp_native.values), ("burial", mi.burial.values),
                    ("dSASA", mi.drsasa.values), ("nbr", mi.nbr.values)]:
        au = auc(s, y); lo, hi = boot(s)
        verdict = "~CHANCE" if lo <= 0.5 <= hi else ("predicts" if lo > 0.5 else "anti")
        print(f"  AUROC {name:15s} = {au:.3f} [{lo:.3f},{hi:.3f}]  {verdict}")
        rows.append(dict(feature=name, auroc=round(au, 4), lo=round(lo, 4), hi=round(hi, 4), verdict=verdict))

    # does confidence add over full geometry? (designer-table / combiner-light)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    Z = lambda c: ((mi[c] - mi[c].mean()) / (mi[c].std() + 1e-9)).to_numpy()
    geo = np.column_stack([Z("burial"), Z("nbr"), Z("drsasa")])
    geoC = np.column_stack([geo, Z("logp_native")])
    og = np.zeros(len(y)); ogc = np.zeros(len(y))
    for tr, te in GroupKFold(min(5, mi.complex_id.nunique())).split(geo, y, g):
        og[te] = LogisticRegression(max_iter=1000).fit(geo[tr], y[tr]).predict_proba(geo[te])[:, 1]
        ogc[te] = LogisticRegression(max_iter=1000).fit(geoC[tr], y[tr]).predict_proba(geoC[te])[:, 1]
    ag, agc = auc(og, y), auc(ogc, y)
    cids = np.unique(g); idx = {c: np.where(g == c)[0] for c in cids}
    dd = []
    for _ in range(5000):
        t = np.concatenate([idx[c] for c in rng.choice(cids, len(cids), True)]); yy = y[t]
        if 0 < yy.sum() < len(yy):
            dd.append(auc(ogc[t], yy) - auc(og[t], yy))
    dd = np.array(dd); lo, hi, p = np.percentile(dd, 2.5), np.percentile(dd, 97.5), float(np.mean(dd > 0))
    print(f"\n  full geometry (burial+nbr+ΔSASA) AUROC {ag:.3f}  |  +confidence {agc:.3f}")
    print(f"  ΔAUROC(confidence over geometry) = {agc-ag:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}")
    replicates = (rows[0]["verdict"] == "~CHANCE") and (rows[1]["verdict"] == "predicts") and (lo <= 0 or agc - ag < 0.01)
    print(f"\n  NUGGET {'REPLICATES on AB-Bind' if replicates else 'does NOT cleanly replicate — inspect'}")
    rows += [dict(feature="geometry_full", auroc=round(ag, 4)),
             dict(feature="geometry+confidence", auroc=round(agc, 4)),
             dict(feature="dAUROC_confidence_over_geometry", auroc=round(agc - ag, 4), lo=round(lo, 4),
                  hi=round(hi, 4), p_gt0=round(p, 3)),
             dict(feature="positive_control_wt_match", auroc=round(match, 4))]
    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_interface"] = len(mi)
    out["n_hot"] = int(y.sum()); out["n_complexes"] = int(mi.complex_id.nunique())
    out["note"] = "AB-Bind antibody-antigen 2nd fixture; nugget replication (confidence~chance, geometry predicts)"
    out["command"] = "python3 src/abbind_nugget.py"
    out.to_csv(a.out, index=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
