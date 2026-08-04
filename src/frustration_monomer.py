"""The third frustration proxy: monomer-versus-complex local energy.

BRIEF.md §4 and STEP 2 name three frustration proxies. `p0_burial_matched.py` implements
buried-polar fraction and rotamer strain; this script adds the third.

Definition. For every interface residue, score log p(native) twice under the same model:
once conditioned on the BOUND complex backbone, and once conditioned on that residue's own
SKEMPI chain group ALONE (the unbound partner). Their difference

    d_bind_local(i) = log p(native_i | complex) - log p(native_i | own group alone)

is the local energetic benefit the binding partner confers, in the model's own units.
A *frustrated* residue is one the isolated partner disprefers but the complex requires, so
frustration predicts d_bind_local to be LARGER at hotspots than at burial-matched controls.

Circularity, stated up front. log p(native | complex) is also the Phase 0 outcome variable,
so d_bind_local is not independent of it, and regressing the pair gap on d_bind_local would
be partly circular. The clean, non-circular use - and the one reported as primary here - is
the PAIRED hotspot-minus-control contrast in d_bind_local itself, which asks whether hotspots
gain more from complexation than matched controls do. Both are computed; the correlation is
reported only with the circularity flagged.

Usage:
  python3 src/frustration_monomer.py --out results/frustration_monomer
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

N_ORDERS = 8
N_BOOT = 10000
SEED = 20260803


def boot_complex(df, col, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    cids = df["complex_id"].unique()
    by = {c: df.loc[df["complex_id"] == c, col].values for c in cids}
    m = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        m[b] = np.nanmean(np.concatenate([by[cids[i]] for i in pick]))
    lo, hi = np.nanpercentile(m, [2.5, 97.5])
    return float(np.nanmean(df[col])), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/frustration_monomer")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--pairs", default="results/p0_pairs_SECONDARY_B_any_interface.csv")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--mpnn-weights",
                    default=os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    ap.add_argument("--complexes", default="results/pair_complexes.txt")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cmd = "python3 " + " ".join(sys.argv)

    want = [l.strip() for l in open(a.complexes) if l.strip()]
    model, _ = fc.load_mpnn(a.mpnn_weights)

    import csv
    import gc
    out_csv = f"{a.out}_positions.csv"
    fh, writer, t0, n = open(out_csv, "w", newline=""), None, time.time(), 0

    for ci, cid in enumerate(want):
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            continue
        try:
            for grp, chains in ((1, g1), (2, g2)):
                mono = fc.load_complex(path, pdb, chains, "", require_both=False)
                if mono is None or mono.n < 5:
                    continue
                lp = fc.mpnn_conditional_logprobs(model, mono, seeds=range(N_ORDERS))
                lpm = lp.mean(axis=0)
                for i in range(mono.n):
                    aa = mono.seq[i]
                    writer_row = dict(
                        complex_id=cid, chain=mono.chains[i], resnum=int(mono.resnums[i]),
                        icode=mono.icodes[i], aa=aa, group=grp,
                        logp_native_monomer=float(lpm[i, fc.MPNN_ALPHABET.index(aa)]))
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(writer_row.keys()))
                        writer.writeheader()
                    writer.writerow(writer_row)
                    n += 1
                del lp, lpm
            fh.flush()
            gc.collect()
        except Exception as e:
            print(f"  skip {cid}: {type(e).__name__}: {e}")
            continue
        if (ci + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[monomer] {ci+1}/{len(want)} complexes, {n} residues, "
                  f"{el:.0f}s ({el/(ci+1):.1f}s/complex)", flush=True)
    fh.close()
    print(f"[monomer] wrote {out_csv}: {n} residues")

    # ---------------------------------------------------------------- analysis
    mono = pd.read_csv(out_csv)
    mono["icode"] = mono["icode"].fillna("").astype(str)
    pos = pd.read_csv(a.positions,
                      usecols=["complex_id", "chain", "resnum", "icode", "aa", "label",
                               "is_interface", "logp_native", "rsasa_complex"])
    pos["icode"] = pos["icode"].fillna("").astype(str)
    pos["label"] = pos["label"].fillna("null")

    m = pos.merge(mono[["complex_id", "chain", "resnum", "icode", "logp_native_monomer"]],
                  on=["complex_id", "chain", "resnum", "icode"], how="inner")
    m["d_bind_local"] = m["logp_native"] - m["logp_native_monomer"]
    m.to_csv(f"{a.out}_joined.csv", index=False)
    print(f"[monomer] joined {len(m)} positions")

    iface = m[m["is_interface"]]
    print("\n=== d_bind_local by label (interface positions) ===")
    print(iface.groupby("label")["d_bind_local"].agg(["count", "mean", "median"]).round(4).to_string())

    # PRIMARY, non-circular: paired hotspot-minus-control contrast in d_bind_local
    key = m.set_index(["complex_id", "chain", "resnum"])["d_bind_local"]
    pr = pd.read_csv(a.pairs)
    recs = []
    for _, r in pr.iterrows():
        try:
            h = float(key.loc[(r["complex_id"], r["hot_chain"], r["hot_resnum"])])
            c = float(key.loc[(r["complex_id"], r["ctl_chain"], r["ctl_resnum"])])
        except (KeyError, TypeError):
            continue
        recs.append(dict(complex_id=r["complex_id"], hot=h, ctl=c, d=h - c,
                         pair_gap=r["d_logp"]))
    d = pd.DataFrame(recs)
    d.to_csv(f"{a.out}_pairs.csv", index=False)

    rows = []
    if len(d) > 20:
        mean, lo, hi = boot_complex(d, "d")
        print(f"\n=== PRIMARY (non-circular): paired d_bind_local, hotspot - control ===")
        print(f"  n_pairs={len(d)}  n_complexes={d['complex_id'].nunique()}")
        print(f"  hotspot mean d_bind_local = {d['hot'].mean():+.4f}")
        print(f"  control mean d_bind_local = {d['ctl'].mean():+.4f}")
        print(f"  paired difference         = {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  frustration predicts this to be POSITIVE (hotspots gain more from binding)")
        rows += [dict(metric="n_pairs", value=len(d)),
                 dict(metric="n_complexes", value=d["complex_id"].nunique()),
                 dict(metric="hot_mean_d_bind_local", value=float(d["hot"].mean())),
                 dict(metric="ctl_mean_d_bind_local", value=float(d["ctl"].mean())),
                 dict(metric="paired_diff", value=mean),
                 dict(metric="ci_lo", value=lo), dict(metric="ci_hi", value=hi)]

        rho, p_ = stats.spearmanr(d["d"], d["pair_gap"])
        print(f"\n=== SECONDARY (PARTLY CIRCULAR - log p(native|complex) enters both sides) ===")
        print(f"  Spearman(delta d_bind_local, pair log-prob gap) = {rho:+.3f}  p={p_:.2e}")
        rows += [dict(metric="spearman_vs_pair_gap_CIRCULAR", value=float(rho)),
                 dict(metric="spearman_p_CIRCULAR", value=float(p_))]

    pd.DataFrame(rows).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    print(f"\n[done] wrote {a.out}_summary.csv")


if __name__ == "__main__":
    main()
