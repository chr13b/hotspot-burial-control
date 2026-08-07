"""KL-guided decoding experiment (BRIEF §2.3 on ProteinMPNN, fixed backbone).

Pre-registration: notes/SHERLOCK_HANDOFF.md context + the Fable design. Metric is
burial-matched hotspot-restricted recovery, difference-in-differences vs the released
default sampler:

  G(c,arm) = mean_{hotspot pos, K samples} 1[sampled==native]
           - mean_{matched-control pos, K samples} 1[sampled==native]
  Δ(arm)   = mean_c [ G(c,arm) - G(c,default) ]     complex-level bootstrap

Arms use TIER-RANDOMISATION: a fresh permutation WITHIN each tier per sample (a single
fixed order throws away the sampler's order diversity and loses best-of-K — PE-E).

  default        released sampler (fully random order)     baseline
  oracle_first   true SKEMPI hotspots in tier 1            the ceiling  (Stage 1)
  kl_first       top-q interface positions by KL in tier 1  the method   (Stage 2)
  kl_shuffled    KL permuted among interface positions      negative control (Stage 2)
  burial_first   most-buried interface positions tier 1     heuristic baseline (Stage 2)
  purity_first   lowest sequence-free entropy tier 1        MultiFlow's heuristic (Stage 2)

STAGING: run Stage 1 (default + oracle_first) first. If Δ(oracle_first) CI contains zero,
STOP — no detector-driven order can beat the oracle (kill K1).

Usage:
  python3 src/decoding/p2_ordering.py --stage 1 --Lmax 400 --K 100 --out results/p2_ordering
"""

import argparse
import csv
import gc
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import ftax_common as fc
import mpnn_steer as steer

SEED, N_BOOT = 20260807, 10000
ALPHA = fc.MPNN_ALPHABET


def tiered_orders(promote_idx, L, n, rng):
    """n orders [n,L]: tier1 = promote_idx permuted, tier2 = the rest permuted (fresh each)."""
    rest = np.setdiff1d(np.arange(L), promote_idx)
    out = np.empty((n, L), dtype=np.int64)
    for i in range(n):
        out[i] = np.concatenate([rng.permutation(promote_idx), rng.permutation(rest)])
    return out


def sample_arm(model, cx, promote_idx, K, batch, seed):
    """K samples with a fresh tier-randomised order per sample. promote_idx=None => default."""
    rng = np.random.default_rng(seed)
    out, done = [], 0
    import torch
    while done < K:
        b = min(batch, K - done)
        if promote_idx is None:
            S, _ = steer.draw(model, cx, b, b, order=None, seed=seed + done,
                              featurize=fc.featurize)
        else:
            orders = tiered_orders(promote_idx, cx.n, b, rng)
            S, got = steer.draw(model, cx, b, b, order=orders, seed=seed + done,
                                featurize=fc.featurize)
            assert (got == orders).all(), "injected order not honoured"
        out.append(S)
        done += b
    return np.concatenate(out, 0)


def recovery_at(S, idx, nat):
    if len(idx) == 0:
        return np.nan
    return float((S[:, idx] == nat[idx][None, :]).mean())


def boot(df, col):
    rng = np.random.default_rng(SEED)
    cids = df["complex_id"].unique()
    by = {c: df.loc[df["complex_id"] == c, col].values for c in cids}
    m = np.array([np.nanmean(np.concatenate([by[cids[i]] for i in rng.choice(len(cids), len(cids), True)]))
                  for _ in range(N_BOOT)])
    return float(np.nanmean(df[col])), *np.nanpercentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--Lmax", type=int, default=400)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--out", default="results/p2_ordering")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--pairs", default="results/p0_dssp_pairs_SECONDARY_B_any_interface.csv")
    ap.add_argument("--kl", default="results/kl_detector_joined.csv")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--mpnn-weights",
                    default=os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    import torch
    torch.set_num_threads(a.threads)

    pairs = pd.read_csv(a.pairs)
    pos = pd.read_csv(a.positions, usecols=["complex_id", "chain", "resnum", "icode", "idx",
                                            "aa", "label", "is_interface", "rsasa_complex"])
    pos["label"] = pos["label"].fillna("null")
    kl = pd.read_csv(a.kl); kl["icode"] = kl["icode"].fillna("").astype(str)
    klkey = kl.set_index(["complex_id", "chain", "resnum"])["kl"]

    arms = ["default", "oracle_first"] if a.stage == 1 else \
           ["default", "oracle_first", "kl_first", "kl_shuffled", "burial_first", "purity_first"]

    out_csv = f"{a.out}_stage{a.stage}_percomplex.csv"
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv)["complex_id"])
    fh = open(out_csv, "a" if done else "w", newline="")
    writer = None

    model, _ = fc.load_mpnn(a.mpnn_weights)
    cx_ids = sorted(pairs["complex_id"].unique())
    print(f"[p2] stage {a.stage}, {len(arms)} arms, K={a.K}, L<={a.Lmax}")

    for cid in cx_ids:
        if cid in done:
            continue
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb")
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None or cx.n > a.Lmax:
            continue
        sub = pos[pos.complex_id == cid]
        nat = np.array([ALPHA.index(x) for x in cx.seq])
        pr = pairs[pairs.complex_id == cid]
        # hotspot + matched-control indices for THIS complex, mapped to structure order
        idxmap = {(r.chain, int(r.resnum)): int(r.idx) for r in sub.itertuples()}
        hot = [idxmap.get((r.hot_chain, int(r.hot_resnum))) for r in pr.itertuples()]
        ctl = [idxmap.get((r.ctl_chain, int(r.ctl_resnum))) for r in pr.itertuples()]
        hot = np.array([h for h in hot if h is not None and h < cx.n])
        ctl = np.array([c for c in ctl if c is not None and c < cx.n])
        if len(hot) == 0 or len(ctl) == 0:
            continue

        iface = sub[sub.is_interface]
        iface_idx = iface["idx"].values.astype(int)
        iface_idx = iface_idx[iface_idx < cx.n]
        rng0 = np.random.default_rng(SEED + hash(cid) % 1000)

        seed_c = SEED + (hash(cid) % 1000000)
        batch = int(max(2, min(20, 8000 // max(cx.n, 1))))
        row = dict(complex_id=cid, L=cx.n, n_hot=len(hot), n_ctl=len(ctl))
        for arm in arms:
            if arm == "default":
                promote = None
            elif arm == "oracle_first":
                promote = hot
            elif arm == "kl_first":
                kv = np.array([klkey.get((cid, cx.chains[i], int(cx.resnums[i])), -1e9)
                               for i in iface_idx])
                q = max(1, int(0.10 * len(iface_idx)))
                promote = iface_idx[np.argsort(-kv)[:q]]
            elif arm == "kl_shuffled":
                q = max(1, int(0.10 * len(iface_idx)))
                promote = rng0.choice(iface_idx, q, replace=False)
            elif arm == "burial_first":
                bv = np.array([-float(sub[sub.idx == i]["rsasa_complex"].iloc[0]) for i in iface_idx])
                q = max(1, int(0.10 * len(iface_idx)))
                promote = iface_idx[np.argsort(-bv)[:q]]
            else:  # purity_first: lowest sequence-free entropy
                unc = fc.mpnn_unconditional_logprobs(model, cx)[:, :20]
                p = np.exp(unc - unc.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                ent = -(p * np.log(p + 1e-12)).sum(1)
                q = max(1, int(0.10 * len(iface_idx)))
                promote = iface_idx[np.argsort(ent[iface_idx])[:q]]
            S = sample_arm(model, cx, promote, a.K, batch, seed_c)
            row[f"Rhot_{arm}"] = recovery_at(S, hot, nat)
            row[f"Rctl_{arm}"] = recovery_at(S, ctl, nat)
            row[f"gap_{arm}"] = row[f"Rhot_{arm}"] - row[f"Rctl_{arm}"]
            row[f"Rall_{arm}"] = float((S == nat[None, :]).mean())
            del S; gc.collect()
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not done:
                writer.writeheader()
        writer.writerow(row); fh.flush()
        print(f"[p2] {cid} L={cx.n} k={len(hot)}  "
              + "  ".join(f"{arm}:gap={row[f'gap_{arm}']:+.3f}" for arm in arms), flush=True)
    fh.close()

    df = pd.read_csv(out_csv)
    print(f"\n=== DiD vs default ({len(df)} complexes) ===")
    rows = []
    for arm in arms:
        if arm == "default":
            m, lo, hi = boot(df, "gap_default")
            print(f"  {arm:14s} gap={m:+.4f} [{lo:+.4f},{hi:+.4f}]  (absolute, not DiD)")
            rows.append(dict(arm=arm, did=m, lo=lo, hi=hi, kind="absolute_gap"))
            continue
        df[f"did_{arm}"] = df[f"gap_{arm}"] - df["gap_default"]
        m, lo, hi = boot(df, f"did_{arm}")
        dq = float((df[f"Rall_{arm}"] - df["Rall_default"]).mean())
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {arm:14s} DiD={m:+.4f} [{lo:+.4f},{hi:+.4f}]{star}  Δoverall_recovery={dq:+.4f}")
        rows.append(dict(arm=arm, did=m, lo=lo, hi=hi, d_overall=dq, kind="DiD"))
    pd.DataFrame(rows).assign(command=cmd, K=a.K, Lmax=a.Lmax).to_csv(
        f"{a.out}_stage{a.stage}_summary.csv", index=False)
    if a.stage == 1:
        o = [r for r in rows if r["arm"] == "oracle_first"][0]
        fired = o["lo"] <= 0 <= o["hi"]
        print(f"\nKILL K1 (oracle ceiling): Δ(oracle)={o['did']:+.4f} [{o['lo']:+.4f},{o['hi']:+.4f}] "
              f"-> {'CI contains 0: STOP, ordering inert' if fired else 'excludes 0: proceed to Stage 2'}")
    print(f"[done] wrote {a.out}_stage{a.stage}_summary.csv")


if __name__ == "__main__":
    main()
