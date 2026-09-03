#!/usr/bin/env python3
"""Per-complex Cα-RMSD(predicted, crystal) for the OpenFold3 / AF2-multimer backbones used in the R2
result, so the R2 points (+0.039 OF3 / +0.032 AF2) can be placed at their TRUE effective σ on the
dose-law x-axis (Fig 3). Two definitions per predictor:
  * ca_rmsd_global : Kabsch superposition on ALL matched Cα, RMSD over all matched.
  * ca_rmsd_iface  : Kabsch superposition on INTERFACE Cα only, RMSD over interface Cα -- the local
                     interface deformation after removing rigid-body motion, the correct analogue of the
                     dose-law jitter (leverage_noise_ladder adds pure local per-atom noise, no rigid body).

Backbones are the EXACT renumbered PDBs the R2 leverage was read from (not the raw predictor output), so
the RMSD is the deformation of the very structure whose leverage we measured. Independently recomputed
here and CROSS-VALIDATED against the committed manifests (expA_confidence.csv OF3,
expD_backbone_manifest.csv AF2) as the rule-6 positive control.

  python3 src/predicted_backbone_rmsd.py --out results/predicted_backbone_rmsd.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc

SEED = 20260803
CRYS = os.path.expanduser("~/ftax/data/PDBs")
OF3 = os.environ["SCRATCH"] + "/ftax/predicted/PDBs"
AF2 = os.environ["SCRATCH"] + "/ftax/expD/PDBs"


def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid superposition of P onto Q (both [n,3])."""
    if len(P) < 3:
        return np.nan
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum(1).mean()))


def ca_map(cx):
    """(chain,resnum,icode) -> Cα coord for a loaded complex."""
    return {(cx.chains[j], int(cx.resnums[j]), cx.icodes[j]): cx.CA[j] for j in range(cx.n)}


def rmsds(cid, pdbdir, iface_keys):
    """Return (global_rmsd, iface_rmsd, n_matched, n_iface_matched) predicted vs crystal, or None."""
    pdb, g1, g2 = cid.split("_")
    cp = f"{CRYS}/{pdb}.pdb"
    pp = f"{pdbdir}/{pdb}.pdb"
    if not (os.path.exists(cp) and os.path.exists(pp)):
        return None
    cx_c = fc.load_complex(cp, pdb, g1, g2)
    cx_p = fc.load_complex(pp, pdb, g1, g2)
    if cx_c is None or cx_p is None:
        return None
    mc, mp = ca_map(cx_c), ca_map(cx_p)
    keys = [k for k in mc if k in mp]                       # residues present in BOTH structures
    if len(keys) < 3:
        return None
    P = np.array([mp[k] for k in keys]); Q = np.array([mc[k] for k in keys])
    ik = [k for k in keys if k in iface_keys]
    Pi = np.array([mp[k] for k in ik]); Qi = np.array([mc[k] for k in ik])
    g = kabsch_rmsd(P, Q)
    i = kabsch_rmsd(Pi, Qi) if len(ik) >= 3 else np.nan
    return g, i, len(keys), len(ik)


def positive_control(iface_by_cid):
    """crystal-onto-itself must be ~0; a genuine predicted-vs-crystal must be nonzero."""
    cid = "1A22_A_B"
    ik = iface_by_cid.get(cid, set())
    pdb, g1, g2 = cid.split("_")
    cx = fc.load_complex(f"{CRYS}/{pdb}.pdb", pdb, g1, g2)
    m = ca_map(cx); keys = list(m)
    P = np.array([m[k] for k in keys])
    self_rmsd = kabsch_rmsd(P, P.copy())
    self_perm = kabsch_rmsd(P, P[np.roll(np.arange(len(P)), 1)])   # mis-matched pairing -> nonzero
    of3 = rmsds(cid, OF3, ik)
    print(f"  [+control] {cid}: crystal-vs-self Cα-RMSD = {self_rmsd:.3e} Å (must be ~0); "
          f"mis-paired = {self_perm:.3f} Å (must be >0); OF3-vs-crystal global = {of3[0]:.3f} Å", flush=True)
    assert self_rmsd < 1e-6, "self-superposition not ~0 — Kabsch is broken"
    assert self_perm > 0.5, "mis-paired control not nonzero — control is inert"
    return self_rmsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/predicted_backbone_rmsd.csv")
    a = ap.parse_args()

    pos = pd.read_csv("results/leverage_skempi_positions.csv", low_memory=False)
    pos["icode"] = pos.icode.fillna("").astype(str)
    pos = pos[pos.is_interface == True]                                              # noqa: E712
    iface = {}
    for r in pos.itertuples():
        iface.setdefault(r.complex_id, set()).add((r.chain, int(r.resnum), r.icode))

    cids = sorted(c for c in iface
                  if os.path.exists(f"{OF3}/{c.split('_')[0]}.pdb")
                  and os.path.exists(f"{AF2}/{c.split('_')[0]}.pdb")
                  and os.path.exists(f"{CRYS}/{c.split('_')[0]}.pdb"))
    print(f"[rmsd] {len(cids)} complexes with crystal+OF3+AF2 PDBs and an interface set", flush=True)
    positive_control(iface)

    rows = []
    for cid in cids:
        ik = iface[cid]
        ro = rmsds(cid, OF3, ik)
        ra = rmsds(cid, AF2, ik)
        if ro is None or ra is None:
            print(f"  skip {cid}: could not load/match", flush=True)
            continue
        rows.append(dict(complex_id=cid,
                         of3_ca_rmsd_all=round(ro[0], 4), of3_ca_rmsd_iface=round(ro[1], 4),
                         af2_ca_rmsd_all=round(ra[0], 4), af2_ca_rmsd_iface=round(ra[1], 4),
                         n_res=ro[2], n_iface=ro[3]))
    df = pd.DataFrame(rows)
    df["seed"] = SEED
    df.to_csv(a.out, index=False)

    # cross-validation against the committed manifests (positive control on the numbers themselves)
    def xval(mycol, manifest, mcol, name):
        if not os.path.exists(manifest):
            print(f"  [xval] {name}: manifest {manifest} absent"); return
        m = pd.read_csv(manifest)
        j = df.merge(m[["complex_id", mcol]], on="complex_id", how="inner").dropna(subset=[mycol, mcol])
        if len(j) < 5:
            print(f"  [xval] {name}: only {len(j)} overlap"); return
        rho = j[mycol].corr(j[mcol], method="spearman")
        md = float((j[mycol] - j[mcol]).abs().median())
        print(f"  [xval] {name}: mine vs committed {mcol}  Spearman={rho:.3f}  median|Δ|={md:.3f} Å  (n={len(j)})")

    print("\n=== cross-validation vs committed manifests ===")
    xval("of3_ca_rmsd_all", "results/expA_confidence.csv", "rmsd_ca_global", "OF3 global")
    xval("of3_ca_rmsd_iface", "results/expA_confidence.csv", "rmsd_ca_interface", "OF3 interface")
    xval("af2_ca_rmsd_all", "results/expD_backbone_manifest.csv", "rmsd_ca_global", "AF2 global")
    xval("af2_ca_rmsd_iface", "results/expD_backbone_manifest.csv", "rmsd_ca_interface", "AF2 interface")

    print("\n=== median [IQR] Cα-RMSD (Å) ===")
    for c, lab in [("of3_ca_rmsd_iface", "OF3 interface"), ("af2_ca_rmsd_iface", "AF2 interface"),
                   ("of3_ca_rmsd_all", "OF3 global"), ("af2_ca_rmsd_all", "AF2 global")]:
        v = df[c].dropna()
        print(f"  {lab:16s} median={v.median():.3f}  IQR=[{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]  "
              f"frac<=1.0Å={float((v<=1.0).mean()):.2f}  frac<=1.5Å={float((v<=1.5).mean()):.2f}")

    # DOSE-LAW BRIDGE: does per-complex leverage RETENTION fall with interface RMSD, as the dose law
    # predicts? (needs deficit_vs_leverage_percomplex.csv from #7). A strong negative = the dose law
    # operates per complex on real predicted backbones, not just under synthetic jitter.
    dv = "results/deficit_vs_leverage_percomplex.csv"
    if os.path.exists(dv):
        from scipy import stats
        rng = np.random.default_rng(SEED)
        j = df.merge(pd.read_csv(dv)[["complex_id", "leverage_metric", "retention_af2"]],
                     on="complex_id", how="inner")
        print("\n=== dose-law bridge: Spearman(interface Cα-RMSD, leverage retention) ===")
        for rc, yc, lab in [("of3_ca_rmsd_iface", "leverage_metric", "OF3"),
                            ("af2_ca_rmsd_iface", "retention_af2", "AF2")]:
            s = j[[rc, yc]].dropna()
            x, y = s[rc].to_numpy(), s[yc].to_numpy()
            rho = stats.spearmanr(x, y).correlation
            b = [stats.spearmanr(x[i], y[i]).correlation
                 for i in (rng.integers(0, len(x), len(x)) for _ in range(5000))]
            b = np.array([v for v in b if np.isfinite(v)])
            print(f"  {lab}: rho={rho:+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}] "
                  f"n={len(s)}  (negative => higher RMSD -> lower leverage retention, dose law per complex)")
    print(f"\n[wrote] {a.out} ({len(df)} complexes)")


if __name__ == "__main__":
    main()
