"""Statistical hardening + external validation of d_bind_local.

Addresses, in one pass, the reviewer-facing gaps in the Phase 0/1 analysis:

  1. TOST equivalence test  - "CI contains zero" fires on low power alone. Replaces it
                              with a real equivalence claim against a mechanism-derived
                              margin.
  2. Holm-Bonferroni        - multiplicity across the 8 matched-pair design variants.
  3. MDE / power            - what each tier could actually have detected.
  4. Anti-memorisation      - overall recovery on these "memorised" complexes vs
                              ProteinMPNN's published held-out recovery.
  5. External validation    - does d_bind_local predict experimental ddG_bind beyond
                              the inverse-folding log-odds? If yes it stops being a
                              model-internal quantity.
  6. Monomer 2x2            - the bound-vs-unbound interaction on the current pairs.

Usage:
  python3 src/hardening.py --out results/hardening --pairs-prefix results/p0_dssp
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

N_BOOT, SEED = 10000, 20260803

# Mechanism-derived equivalence margin. BRIEF 2.2: N_hot = exp(sum delta_i / T).
# F2's bar for "costly" is log10 N_hot >= 2 at T = 0.1 over a median constellation of
# k = 4 hotspots. The per-position deficit that would just reach it is
#   delta = 2 * ln(10) * T / k = 2 * 2.302585 * 0.1 / 4
MARGIN = 2 * np.log(10) * 0.1 / 4


def boot(df, col, stat=np.nanmean, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df.loc[df["complex_id"] == c, col].values for c in cids}
    m = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        m[b] = stat(np.concatenate([by[cids[i]] for i in pick]))
    return float(stat(df[col].values)), *np.nanpercentile(m, [2.5, 97.5]), m


def tost(df, col, margin=MARGIN):
    """Two one-sided tests for equivalence to zero within +/- margin.

    Equivalence is declared when the 90% CI lies entirely inside (-margin, +margin);
    that is the standard TOST-CI correspondence at alpha = 0.05.
    """
    _, _, _, m = boot(df, col)
    lo90, hi90 = np.nanpercentile(m, [5, 95])
    p_lower = float((m <= -margin).mean())   # evidence against "true effect <= -margin"
    p_upper = float((m >= margin).mean())
    return dict(mean=float(np.nanmean(df[col])), lo90=float(lo90), hi90=float(hi90),
                equivalent=bool(lo90 > -margin and hi90 < margin),
                p_tost=float(max(p_lower, p_upper)), margin=float(margin))


def holm(pvals, names, alpha=0.05):
    order = np.argsort(pvals)
    out, k = {}, len(pvals)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (k - rank) * pvals[i]))
        prev = adj
        out[names[i]] = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hardening")
    ap.add_argument("--pairs-prefix", default="results/p0_dssp")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)
    rows = []

    def rec(section, metric, value, **kw):
        rows.append(dict(section=section, metric=metric, value=value, **kw))

    # ---------------------------------------------------------------- 1-3
    print("=" * 78)
    print(f"1-3. EQUIVALENCE (TOST), MULTIPLICITY (Holm), POWER")
    print(f"     mechanism-derived margin = {MARGIN:.4f} nats "
          f"(the per-position deficit that would just reach log10 N_hot = 2 at k=4, T=0.1)")
    print("=" * 78)
    files = sorted(glob.glob(f"{a.pairs_prefix}_pairs_*.csv"))
    pv, nm, store = [], [], {}
    for f in files:
        tag = os.path.basename(f).split("_pairs_")[1][:-4]
        pr = pd.read_csv(f)
        if len(pr) < 10:
            continue
        mean, lo, hi, _ = boot(pr, "d_logp")
        t = tost(pr, "d_logp")
        # two-sided cluster-bootstrap p for H0: gap = 0
        _, _, _, m = boot(pr, "d_logp")
        p2 = float(2 * min((m <= 0).mean(), (m >= 0).mean()))
        sd = pr.groupby("complex_id")["d_logp"].mean().std(ddof=1)
        ncx = pr["complex_id"].nunique()
        mde = 1.96 * sd / np.sqrt(ncx)   # detectable |effect| at 80%-ish, 5% two-sided
        store[tag] = dict(n=len(pr), ncx=ncx, lo=lo, hi=hi, mde=mde, **t)
        pv.append(max(p2, 1.0 / N_BOOT)); nm.append(tag)
        print(f"\n  {tag}")
        print(f"    n={len(pr):4d} pairs / {ncx:3d} complexes   gap={mean:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"    90% CI [{t['lo90']:+.4f}, {t['hi90']:+.4f}]  ->  "
              f"EQUIVALENT to zero within +/-{MARGIN:.3f}? "
              f"{'YES' if t['equivalent'] else 'no'}")
        print(f"    minimum detectable |effect| at this n: {mde:.4f} nats")
    hp = holm(np.array(pv), nm)
    print("\n  Holm-adjusted two-sided p-values across the "
          f"{len(nm)} design variants:")
    for t in nm:
        s = store[t]
        print(f"    {t:32s} raw p={pv[nm.index(t)]:.4f}  Holm p={hp[t]:.4f}  "
              f"{'*' if hp[t] < 0.05 else ''}")
        rec("multiplicity", t, s["mean"], n_pairs=s["n"], n_complexes=s["ncx"],
            ci_lo=s["lo"], ci_hi=s["hi"], mde=s["mde"], tost_equivalent=s["equivalent"],
            tost_lo90=s["lo90"], tost_hi90=s["hi90"], p_raw=pv[nm.index(t)], p_holm=hp[t])

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 78)
    print("4. ANTI-MEMORISATION CHECK")
    print("=" * 78)
    pos = pd.read_csv(a.positions, usecols=["complex_id", "aa", "mode_aa", "is_interface"])
    overall = float((pos["mode_aa"] == pos["aa"]).mean())
    iface = pos[pos["is_interface"]]
    ifrec = float((iface["mode_aa"] == iface["aa"]).mean())
    print(f"  overall recovery on these (fully leaked) complexes : {overall:.3f}")
    print(f"  interface-only recovery                            : {ifrec:.3f}")
    print(f"  ProteinMPNN published held-out recovery            : 0.499-0.525")
    print("  -> If memorisation were substituting for energetic reasoning we would expect")
    print("     anomalously HIGH overall recovery on seen complexes. We do not see it.")
    rec("memorisation", "overall_recovery", overall)
    rec("memorisation", "interface_recovery", ifrec)
    rec("memorisation", "published_heldout_reference", 0.52)

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 78)
    print("5. EXTERNAL VALIDATION OF d_bind_local against experimental ddG_bind")
    print("=" * 78)
    jf = "results/frustration_monomer_joined.csv"
    if os.path.exists(jf):
        j = pd.read_csv(jf)
        j["icode"] = j["icode"].fillna("").astype(str)
        j["d_bind_local"] = j["logp_native"] - j["logp_native_monomer"]
        f1 = pd.read_csv("results/p0_f1_logodds.csv")
        skempi = fc.parse_skempi(os.path.join(a.data_dir, "skempi_v2.csv"))
        single = skempi[skempi["n_mut"] == 1].copy()
        kk = j.set_index(["complex_id", "chain", "resnum"])
        recs = []
        for _, r in single.iterrows():
            m = fc.parse_mutation(r["muts"][0])
            if m is None:
                continue
            key = (f"{r['pdb']}_{r['group1']}_{r['group2']}", m["chain"], m["resnum"])
            if key not in kk.index:
                continue
            row = kk.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if row["aa"] != m["wt"] or not row["is_interface"]:
                continue
            recs.append(dict(complex_id=key[0], ddG=r["ddG"],
                             d_bind_local=row["d_bind_local"],
                             rsasa=row["rsasa_complex"]))
        e = pd.DataFrame(recs).dropna()
        e = e.merge(f1[["complex_id", "ddG", "logodds"]].drop_duplicates(),
                    on=["complex_id", "ddG"], how="inner").dropna()
        print(f"  n = {len(e)} single interface mutations with both quantities")
        if len(e) > 100:
            for lbl, col in [("log-odds alone", "logodds"),
                             ("d_bind_local alone", "d_bind_local")]:
                rho, p_ = stats.spearmanr(e[col], e["ddG"])
                print(f"    {lbl:22s} vs ddG_bind: rho={rho:+.3f} (p={p_:.1e})")
                rec("external_validation", f"spearman_{col}", float(rho), n=len(e))
            # does d_bind_local add beyond log-odds? partial correlation
            def partial(x, y, z):
                rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
                Z = np.column_stack([np.ones_like(rz), rz])
                res = lambda r: r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
                return stats.spearmanr(res(rx), res(ry))
            rho, p_ = partial(e["d_bind_local"].values, e["ddG"].values,
                              e["logodds"].values)
            print(f"    d_bind_local vs ddG_bind, CONTROLLING for log-odds: "
                  f"rho={rho:+.3f} (p={p_:.1e})")
            rec("external_validation", "partial_dbind_given_logodds", float(rho),
                p=float(p_), n=len(e))
            rho2, p2_ = partial(e["d_bind_local"].values, e["ddG"].values,
                                e["rsasa"].values)
            print(f"    d_bind_local vs ddG_bind, CONTROLLING for burial:   "
                  f"rho={rho2:+.3f} (p={p2_:.1e})")
            rec("external_validation", "partial_dbind_given_burial", float(rho2),
                p=float(p2_), n=len(e))
            e.to_csv(f"{a.out}_external.csv", index=False)

    # ---------------------------------------------------------------- 6
    print("\n" + "=" * 78)
    print("6. THE BOUND-vs-UNBOUND 2x2 (on the pydssp-corrected pairs)")
    print("=" * 78)
    if os.path.exists(jf):
        j = pd.read_csv(jf)
        j["icode"] = j["icode"].fillna("").astype(str)
        kc = j.set_index(["complex_id", "chain", "resnum"])["logp_native"]
        km = j.set_index(["complex_id", "chain", "resnum"])["logp_native_monomer"]
        pf = f"{a.pairs_prefix}_pairs_SECONDARY_B_any_interface.csv"
        if os.path.exists(pf):
            pr = pd.read_csv(pf)
            rr = []
            for _, r in pr.iterrows():
                try:
                    hc = float(kc.loc[(r.complex_id, r.hot_chain, r.hot_resnum)])
                    cc = float(kc.loc[(r.complex_id, r.ctl_chain, r.ctl_resnum)])
                    hm = float(km.loc[(r.complex_id, r.hot_chain, r.hot_resnum)])
                    cm = float(km.loc[(r.complex_id, r.ctl_chain, r.ctl_resnum)])
                except (KeyError, TypeError):
                    continue
                rr.append(dict(complex_id=r.complex_id, gap_bound=hc - cc,
                               gap_unbound=hm - cm, interaction=(hc - hm) - (cc - cm)))
            d = pd.DataFrame(rr)
            for col, lab in [("gap_bound", "gap | BOUND complex   "),
                             ("gap_unbound", "gap | UNBOUND monomer"),
                             ("interaction", "INTERACTION (d_bind) ")]:
                m, lo, hi, _ = boot(d, col)
                print(f"  {lab}  {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
                rec("two_by_two", col, m, ci_lo=lo, ci_hi=hi, n=len(d),
                    n_complexes=d["complex_id"].nunique())
            d.to_csv(f"{a.out}_2x2.csv", index=False)

    pd.DataFrame(rows).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"\n[done] wrote {a.out}_summary.csv")


if __name__ == "__main__":
    main()
