#!/usr/bin/env python3
"""T3 — constraint-vs-leverage theory fork: does SCALAR confidence predict interface hotspots BEST in the
de-novo (binding-dominated) regime?

Pre-registered in results/PREREG_bennett_hardening.md. On SKEMPI, per-residue confidence (log p_native)
ranks interface hotspots at ~chance (0.538) — floored, so a "decay across selection regimes" cannot be seen
there. The theory (confidence estimates positional CONSTRAINT H(a_i|X); hotspot-ness is LEVERAGE
dDGbind/da_i; they coincide only under binding-dominated selection) predicts confidence should track
hotspots BEST in the maximally binding-dominated regime = de-novo designs (Bennett), where the monomer fold
imposes least evolutionary constraint.

Per interface position on the Bennett binder we compute scalar confidence (log p_native and negentropy of
the complex-conditioned distribution) and a position hotspot label (fraction of the position's measured
substitutions that abolish binding >= across-interface median). AUROC vs SKEMPI's 0.538. Design-clustered
bootstrap, seed 20260803.
  python3 src/p_bennett_conf_fork.py --out results/bennett_conf_fork.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc
import bennett_knows_where as bkw
import bennett_kl_detector as bkd

SEED = 20260803
SKEMPI_CONF_AUROC = 0.538   # from confidence_antipredicts.csv / xmodel_confidence.csv (ProteinMPNN interface)


def boot_auc(score, y, g, rng, nboot=3000):
    ids = np.unique(g); pos = {u: np.where(g == u)[0] for u in ids}
    out = []
    for _ in range(nboot):
        idx = np.concatenate([pos[u] for u in rng.choice(ids, len(ids), True)])
        a = bkw.auc(score[idx], y[idx])
        if np.isfinite(a):
            out.append(a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bennett_conf_fork.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    pdb_index = {os.path.basename(p)[:-4]: p for p in
                 __import__("glob").glob(f"{bkw.BEN}/design_models_ssm_natives/*/*.pdb")}
    model, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))

    recs = []
    for lib_name in bkw.LIBS:
        af = f"{bkw.BEN}/ngs_data_analysis/affinities/{lib_name}.sc"
        if not os.path.exists(af):
            continue
        lab = bkw.per_sub_labels(pd.read_csv(af, sep=r"\s+", engine="python"))
        for parent in sorted({p for (p, _) in lab}):
            pdb = pdb_index.get(parent)
            if pdb is None:
                continue
            try:
                pq = bkw.full_PQ(model, pdb, parent)
                bi = bkd.burial_interface(pdb, parent)
            except Exception as e:
                print(f"  skip {parent[:22]}: {type(e).__name__}"); continue
            if not pq:
                continue
            for (par, pos), (subs, native) in lab.items():
                if par != parent or pos not in pq or pos not in bi or native is None or native not in bkw.IDX:
                    continue
                Pv, Qv, restype = pq[pos]
                if native != restype:
                    continue
                dsasa = bi[pos]["sasa_mono"] - bi[pos]["sasa_complex"]
                if dsasa <= 5:                                  # interface only
                    continue
                tested = [(s, b) for s, b in subs.items() if s in bkw.IDX and s != native]
                if len(tested) < 6:
                    continue
                frac_abolish = 1.0 - np.mean([b for _, b in tested])
                p = np.clip(Pv[:20], 1e-9, 1)
                recs.append(dict(design=parent, resnum=int(pos), native=native,
                                 logp_native=float(np.log(Pv[bkw.IDX[native]] + 1e-9)),
                                 negentropy=float(np.sum(p * np.log(p))),   # = -H, higher = more confident
                                 frac_abolish=float(frac_abolish), n_sub=len(tested)))
    d = pd.DataFrame(recs)
    thr = d.frac_abolish.median()
    d["is_hot"] = (d.frac_abolish >= thr).astype(int)
    y = d.is_hot.to_numpy(); g = d.design.to_numpy()
    print(f"interface positions: {len(d)}  designs: {d.design.nunique()}  "
          f"hot(frac_abolish>={thr:.2f}): {int(y.sum())}/{len(d)}")

    rows = []
    for feat in ["logp_native", "negentropy"]:
        s = d[feat].to_numpy()
        au = bkw.auc(s, y); lo, hi = boot_auc(s, y, g, rng)
        fires = (lo > 0.5) and (au > SKEMPI_CONF_AUROC)
        verdict = ("theory FIRES (confidence>chance AND>SKEMPI)" if fires
                   else ("above chance but not > SKEMPI" if lo > 0.5 else "~chance -> SELF-REFUTES; fall back to P-Q"))
        print(f"  confidence[{feat:11s}] AUROC={au:.3f} [{lo:.3f},{hi:.3f}]  vs SKEMPI {SKEMPI_CONF_AUROC}  -> {verdict}")
        rows.append(dict(feature=feat, auroc=round(au, 4), lo=round(lo, 4), hi=round(hi, 4),
                         skempi_ref=SKEMPI_CONF_AUROC, n=len(d), n_hot=int(y.sum()), verdict=verdict))
    out = pd.DataFrame(rows); out["seed"] = SEED
    out["note"] = "position hot = frac of subs abolishing binding >= interface median; de-novo = binding-dominated regime"
    out["command"] = "python3 src/p_bennett_conf_fork.py"
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
