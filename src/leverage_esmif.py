#!/usr/bin/env python3
"""Model-generality check for the Confidence-Leverage Decomposition: re-run the WHOLE
decomposition under a second, architecturally-distinct inverse-folding model, ESM-IF1
(esm_if1_gvp4_t16_142M_UR50 -- GVP-transformer, 142M params, causal L->R), and ask whether
the two signature signs replicate:

    Spearman(L, DDG_bind) < 0          and          CPI(L | geometry) > 0
    while confidence stays conditionally independent of the hotspot label.

If they do, the "one model (ProteinMPNN)" limitation is answered and the decomposition is a
property of the inverse-folding *class*, not a ProteinMPNN quirk.

The leverage operator is the SAME double difference as in leverage_decomposition.py:
    L_i(a) = [logP_i(a) - logP_i(wt)] - [logQ_i(a) - logQ_i(wt)],  P=p(.|complex), Q=p(.|monomer)
Two honest conditioning differences from the MPNN run, stated not hidden:
  * ProteinMPNN leverage uses the SEQUENCE-FREE unconditional marginal; ESM-IF1 is autoregressive
    and has no such marginal, so we use its native-teacher-forced conditional p(a_i | backbone,
    native_{<i}). The native context is identical in P and Q, so the double difference still
    isolates the partner. Replicating under BOTH readouts strengthens generality.
  * ESM-IF1 conditions on the complex BACKBONE but not the partner SEQUENCE (official multichain
    protocol). The leverage is therefore a clean backbone-only partner ablation.

ALL statistics (CPI, within-stratum AUROC, Spearman, the blindness-theorem demo, the algebraic
identity) are imported verbatim from leverage_decomposition.py -- this is a pure scorer swap, so
the test is a genuine apples-to-apples model substitution.

  # smoke test first (rule 6: positive control before trusting output)
  python3 src/leverage_esmif.py --stage score --limit 4
  # then the full score (background) and analyse
  python3 src/leverage_esmif.py --stage score
  python3 src/leverage_esmif.py --stage analyse --out results/leverage_esmif.csv
"""
import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models"))
import ftax_common as fc
import leverage_decomposition as LD
import ftax_esmif as fe

SEED = LD.SEED
DATA = LD.DATA
AA20 = LD.AA20
IDX = LD.IDX
PQ_ESMIF = "results/leverage_pq_skempi_esmif.csv"


def esmif_lp(model, alphabet, cx, device="cpu"):
    """[cx.n, 21] ESM-IF1 log-probs (MPNN alphabet), averaged over the non-target-chain orders.

    `device` must match the device `model` was loaded on: the batch converter builds the coord /
    token tensors itself, so a model moved to CUDA with device left at "cpu" fails on a device
    mismatch rather than silently falling back.
    """
    lp = fe.esmif_conditional_logprobs(model, alphabet, cx, seeds=(0,), device=device)  # [1,L,21]
    return lp.mean(axis=0)


def _score_one(model, alphabet, path, pdb, g1, g2, keep, max_residues=0, device="cpu"):
    """Mirror of LD._score_one but with the ESM-IF1 scorer. -> list of dict rows.

    Returns ([], ("TOOLARGE", n)) for complexes above max_residues so the caller can drop-and-log
    them instead of the whole process OOM-SIGKILLing (ESM-IF1's GVP graph scales badly in residues;
    a ~2000-residue complex kills the process on a 4 GB box, uncatchable in-process).
    """
    cx = fc.load_complex(path, pdb, g1, g2)
    if cx is None:
        return [], None
    if max_residues and cx.n > max_residues:
        return [], ("TOOLARGE", int(cx.n))
    lP = LD.logdists(esmif_lp(model, alphabet, cx, device))         # [L,20] renorm over 20
    # native top-1 recovery on the complex (a positive control on the scorer + alphabet map)
    rec = float((lP.argmax(1) == np.array([IDX.get(s, -1) for s in cx.seq])).mean())
    lQ = np.full_like(lP, np.nan)
    for chains in (g1, g2):
        if not chains:
            continue
        mono = fc.load_complex(path, pdb, chains, "", require_both=False)
        if mono is None or mono.n < 5:
            continue
        lQm = LD.logdists(esmif_lp(model, alphabet, mono, device))
        im = {(c, int(r), i): k for k, (c, r, i)
              in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
        for j in range(cx.n):
            k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
            if k is not None:
                lQ[j] = lQm[k]
        del lQm, mono
    rows = []
    for j in range(cx.n):
        key = (cx.chains[j], int(cx.resnums[j]), cx.icodes[j])
        if keep is not None and key not in keep:
            continue
        if not np.isfinite(lQ[j]).all():
            continue
        r = dict(chain=key[0], resnum=key[1], icode=key[2], aa=cx.seq[j])
        for a in AA20:
            r[f"lP_{a}"] = float(lP[j, IDX[a]])
        for a in AA20:
            r[f"lQ_{a}"] = float(lQ[j, IDX[a]])
        rows.append(r)
    del lP, lQ, cx
    gc.collect()
    return rows, rec


def _keepset():
    """Exactly LD.stage_score_skempi's keep-set: interface positions + measured single-mut positions."""
    sk = fc.parse_skempi(f"{DATA}/skempi_v2.csv")
    sk = sk[sk.n_mut == 1].copy()
    sk["complex_id"] = sk.pdb + "_" + sk.group1 + "_" + sk.group2
    p0 = pd.read_csv("results/p0_positions.csv", low_memory=False,
                     usecols=["complex_id", "chain", "resnum", "icode", "is_interface"])
    p0["icode"] = p0.icode.fillna("").astype(str)
    keep = {}
    for cid, sub in p0[p0.is_interface == True].groupby("complex_id"):      # noqa: E712
        keep[cid] = set(zip(sub.chain, sub.resnum.astype(int), sub.icode))
    for _, r in sk.iterrows():
        m = fc.parse_mutation(r["muts"][0])
        if m is None:
            continue
        keep.setdefault(r["complex_id"], set()).add((m["chain"], m["resnum"], m["icode"]))
    return keep


def stage_score(a):
    import torch
    torch.set_num_threads(a.threads)
    keep = _keepset()
    cx_ids = sorted(keep)
    if a.limit:
        cx_ids = cx_ids[:a.limit]
    print(f"[score] {len(cx_ids)} complexes to score with ESM-IF1 "
          f"({sum(len(keep[c]) for c in cx_ids)} target positions)", flush=True)

    done = set()
    cache = a.cache
    if os.path.exists(cache) and not a.limit:
        try:
            done = set(pd.read_csv(cache, usecols=["complex_id"]).complex_id)
            print(f"[score] resuming, {len(done)} complexes already scored", flush=True)
        except Exception:
            done = set()
    skip = set()
    if a.skip_file and os.path.exists(a.skip_file):
        skip = {ln.strip() for ln in open(a.skip_file) if ln.strip()}
        print(f"[score] skip-list ({len(skip)} OOM/oversized): {sorted(skip)}", flush=True)
    model, alphabet = fe.load_esmif(device=a.device)
    # alphabet positive control -- an ESM->MPNN column slip silently drops recovery to ~0.05
    amap = fe.build_alphabet_map(alphabet)
    back = "".join(alphabet.get_tok(int(i)) for i in amap)
    assert back == fc.MPNN_ALPHABET, f"ALPHABET SLIP: {back!r}"
    print(f"[+control] ESM-IF1 alphabet map round-trips to {back!r}", flush=True)

    fh = open(cache, "a" if done else "w", newline="")
    writer, n, t0, recs, skipped, dropped = None, 0, time.time(), [], [], []
    for ci, cid in enumerate(cx_ids):
        if cid in done or cid in skip:
            continue
        pdb, g1, g2 = cid.split("_")
        path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            skipped.append((cid, "no pdb"))
            continue
        if a.inflight_file:                       # marker so the wrapper can blame an OOM on this cid
            open(a.inflight_file, "w").write(cid)
        try:
            rows, rec = _score_one(model, alphabet, path, pdb, g1, g2, keep[cid], a.max_residues,
                                   a.device)
        except Exception as e:
            skipped.append((cid, f"{type(e).__name__}: {e}"))
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True)
            if a.inflight_file:
                open(a.inflight_file, "w").write("")
            continue
        if isinstance(rec, tuple) and rec[0] == "TOOLARGE":
            dropped.append((cid, rec[1]))
            print(f"[score] {ci+1}/{len(cx_ids)} {cid} DROPPED (too large: {rec[1]} > "
                  f"{a.max_residues} residues) — logged, not scored", flush=True)
            if a.skip_file:
                open(a.skip_file, "a").write(cid + "\n")
            if a.inflight_file:
                open(a.inflight_file, "w").write("")
            continue
        if rec is not None:
            recs.append(rec)
        for r in rows:
            r = dict(complex_id=cid, **r)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if not done:
                    writer.writeheader()
            writer.writerow(r)
            n += 1
        fh.flush()
        if a.inflight_file:
            open(a.inflight_file, "w").write("")
        dt = time.time() - t0
        print(f"[score] {ci+1}/{len(cx_ids)} {cid} rec={rec:.3f} rows={len(rows)} "
              f"({dt:.0f}s, {dt/max(1,ci+1):.1f}s/cx)", flush=True)
    fh.close()
    if dropped:
        print(f"[score] DROPPED {len(dropped)} oversized complexes (>{a.max_residues} res): {dropped}",
              flush=True)
    print(f"\n[score] wrote {cache}: {n} rows; {len(skipped)} skipped; "
          f"mean complex top-1 recovery {np.mean(recs) if recs else float('nan'):.3f} "
          f"(ESM-IF1 healthy ~0.45-0.55; ~0.05 would mean a broken alphabet map)", flush=True)
    if skipped:
        print("  skipped:", skipped[:10])


def stage_analyse(a):
    """Reuse leverage_decomposition's verified analysis verbatim, pointed at the ESM-IF1 cache."""
    LD.PQ_SKEMPI = a.cache                       # <-- the only redirection; all stats are LD's
    rng = np.random.default_rng(SEED)
    rows = []

    pos, Lvec, lP, lQ = LD.position_frame()
    pos.to_csv("results/leverage_esmif_positions.csv", index=False,
               columns=[c for c in pos.columns if not c.startswith(("lP_", "lQ_"))])

    # model-agnostic identity control (the committed-KL match is MPNN-specific, so skipped here)
    newkl = (np.exp(lP) * (lP - lQ)).sum(axis=1)
    wi = pos.aa.map(IDX).to_numpy(); ar = np.arange(len(pos))
    ident = (np.exp(lP) * Lvec).sum(axis=1) + (lP - lQ)[ar, wi]
    id_err = float(np.abs(ident - newkl).max())
    print(f"  [+control] identity KL = E_P[L] + [logP(wt)-logQ(wt)]: max |Δ| = {id_err:.2e}")
    rows.append(dict(fixture="SKEMPI_esmif", test="algebraic_identity_KL_equals_EP_L_plus_r_wt",
                     stat=id_err, n=int(len(pos))))

    LD.theorem_demo(pos, Lvec, lP, rows, rng, name="SKEMPI_esmif")
    LD.position_level_cpi(pos, rows, rng, name="SKEMPI_esmif")
    del Lvec, lP, lQ
    gc.collect()

    sk = LD.build_skempi(rows)
    sk["destab"] = (sk.ddG >= LD.HOT_DDG).astype(int)
    sk.to_csv("results/leverage_esmif_mutations.csv", index=False,
              columns=[c for c in sk.columns if not c.startswith(("lP_", "lQ_"))])
    LD.run_fixture(sk, "SKEMPI (natural, ESM-IF1)", rows, rng)

    out = pd.DataFrame(rows)
    out["seed"] = SEED
    out["model"] = "esm_if1_gvp4_t16_142M_UR50"
    out["command"] = "python3 " + " ".join(sys.argv)
    out.to_csv(a.out, index=False)
    print(f"\n[done] wrote {a.out} ({len(out)} rows)")

    # headline: do the two signs replicate?
    sp = out[(out.fixture.str.startswith("SKEMPI")) & (out.test == "spearman_L_vs_ddG")]
    cpi_mut = out[out.test.str.startswith("CPI(LEVERAGE L (full) | burial+nbr+dSASA)", na=False)]
    cpi_pos = out[out.test.str.startswith("CPI_position_level(leverage L(->Ala)", na=False)]
    cpi_conf = out[out.test.str.startswith("CPI_position_level(confidence", na=False)]
    def show(df, k):
        return "  ".join(f"{r.stat:+.5f}[{getattr(r,'lo',float('nan')):+.5f},"
                         f"{getattr(r,'hi',float('nan')):+.5f}]" for r in df.itertuples()) or "n/a"
    print("[REPLICATION under ESM-IF1]")
    print(f"  Spearman(L, DDG)            = {show(sp,0)}   (MPNN: -0.295; theory <0)")
    print(f"  CPI(L | geometry) mutation  = {show(cpi_mut,0)}   (MPNN: +0.0588; theory >0)")
    print(f"  CPI(L->Ala|geom) position   = {show(cpi_pos,0)}   (MPNN: +0.00485)")
    print(f"  CPI(confidence|geom) pos    = {show(cpi_conf,0)}   (MPNN: +0.00023 blind)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="analyse", choices=["score", "analyse"])
    ap.add_argument("--cache", default=PQ_ESMIF)
    ap.add_argument("--out", default="results/leverage_esmif.csv")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="'cpu' (committed path) or 'cuda'")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N complexes (smoke test)")
    ap.add_argument("--max-residues", type=int, default=1200,
                    help="drop-and-log complexes larger than this (ESM-IF1 GVP OOM guard)")
    ap.add_argument("--skip-file", default="", help="file of complex_ids to skip; grows on OOM/oversize")
    ap.add_argument("--inflight-file", default="", help="marker file naming the complex being scored")
    a = ap.parse_args()
    {"score": stage_score, "analyse": stage_analyse}[a.stage](a)


if __name__ == "__main__":
    main()
