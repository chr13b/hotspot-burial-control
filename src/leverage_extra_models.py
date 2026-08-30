#!/usr/bin/env python3
"""Model-symmetry: run the SAME Confidence-Leverage Decomposition under PiFold and MIF, so the feature-class
law's POSITIVE half (leverage adds) rests on the same architecture panel as its NEGATIVE half (confidence
blind, shown on 5 models). Pure scorer swap into leverage_esmif's machinery — all statistics are
leverage_decomposition's, verbatim.

  python3 src/leverage_extra_models.py --model pifold --stage score --limit 4     # smoke test
  python3 src/leverage_extra_models.py --model pifold --stage score               # full
  python3 src/leverage_extra_models.py --model pifold --stage analyse --out results/leverage_pifold.csv
  python3 src/leverage_extra_models.py --model mif    --stage score --mif-seeds 6
"""
import argparse, csv, gc, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "models"))
import ftax_common as fc
import leverage_decomposition as LD
import leverage_esmif as LE
SEED, DATA, AA20, IDX = LD.SEED, LD.DATA, LD.AA20, LD.IDX


def make_scorer(model_name, mif_seeds):
    """Return lp(cx) -> [cx.n, 21] MPNN-alphabet log-probs, model loaded on CPU."""
    if model_name == "pifold":
        import ftax_pifold as fp
        model = fp.load_pifold(device="cpu")
        return lambda cx: fp.pifold_conditional_logprobs(model, cx, device="cpu").mean(axis=0)
    import ftax_mif as fm
    model, collater = fm.load_mif(device="cpu")
    return lambda cx: fm.mif_conditional_logprobs(model, collater, cx, seeds=range(mif_seeds),
                                                  device="cpu").mean(axis=0)


def _score_one(lp, path, pdb, g1, g2, keep, max_residues=0):
    cx = fc.load_complex(path, pdb, g1, g2)
    if cx is None:
        return [], None
    if max_residues and cx.n > max_residues:
        return [], ("TOOLARGE", int(cx.n))
    lP = LD.logdists(lp(cx))
    rec = float((lP.argmax(1) == np.array([IDX.get(s, -1) for s in cx.seq])).mean())
    lQ = np.full_like(lP, np.nan)
    for chains in (g1, g2):
        if not chains:
            continue
        mono = fc.load_complex(path, pdb, chains, "", require_both=False)
        if mono is None or mono.n < 5:
            continue
        lQm = LD.logdists(lp(mono))
        im = {(c, int(r), i): k for k, (c, r, i) in enumerate(zip(mono.chains, mono.resnums, mono.icodes))}
        for j in range(cx.n):
            k = im.get((cx.chains[j], int(cx.resnums[j]), cx.icodes[j]))
            if k is not None:
                lQ[j] = lQm[k]
        del lQm, mono
    rows = []
    for j in range(cx.n):
        key = (cx.chains[j], int(cx.resnums[j]), cx.icodes[j])
        if (keep is not None and key not in keep) or not np.isfinite(lQ[j]).all():
            continue
        r = dict(chain=key[0], resnum=key[1], icode=key[2], aa=cx.seq[j])
        for a in AA20:
            r[f"lP_{a}"] = float(lP[j, IDX[a]])
        for a in AA20:
            r[f"lQ_{a}"] = float(lQ[j, IDX[a]])
        rows.append(r)
    del lP, lQ, cx; gc.collect()
    return rows, rec


def stage_score(a):
    import torch; torch.set_num_threads(a.threads)
    keep = LE._keepset(); cx_ids = sorted(keep)
    if a.limit:
        cx_ids = cx_ids[:a.limit]
    lp = make_scorer(a.model, a.mif_seeds)
    print(f"[score:{a.model}] {len(cx_ids)} complexes ({sum(len(keep[c]) for c in cx_ids)} positions)", flush=True)
    done = set()
    if os.path.exists(a.cache) and not a.limit:
        try:
            done = set(pd.read_csv(a.cache, usecols=["complex_id"]).complex_id)
            print(f"[score] resuming, {len(done)} done", flush=True)
        except Exception:
            done = set()
    fh = open(a.cache, "a" if done else "w", newline=""); writer = None; n = 0; t0 = time.time(); recs = []
    for ci, cid in enumerate(cx_ids):
        if cid in done:
            continue
        pdb, g1, g2 = cid.split("_"); path = f"{DATA}/PDBs/{pdb}.pdb"
        if not os.path.exists(path):
            continue
        try:
            rows, rec = _score_one(lp, path, pdb, g1, g2, keep[cid], a.max_residues)
        except Exception as e:
            print(f"  skip {cid}: {type(e).__name__}: {e}", flush=True); continue
        if isinstance(rec, tuple):
            print(f"  drop {cid} (too large: {rec[1]})", flush=True); continue
        if rec is not None:
            recs.append(rec)
        for r in rows:
            r = dict(complex_id=cid, **r)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if not done:
                    writer.writeheader()
            writer.writerow(r); n += 1
        fh.flush()
        dt = time.time() - t0
        print(f"[score] {ci+1}/{len(cx_ids)} {cid} rec={rec:.3f} rows={len(rows)} ({dt/max(1,ci+1):.1f}s/cx)", flush=True)
    fh.close()
    print(f"[score:{a.model}] wrote {a.cache}: {n} rows; mean recovery {np.mean(recs) if recs else float('nan'):.3f} "
          f"(healthy ~0.35-0.55; ~0.05 = broken alphabet map)", flush=True)


def stage_analyse(a):
    LD.PQ_SKEMPI = a.cache
    rng = np.random.default_rng(SEED); rows = []
    pos, Lvec, lP, lQ = LD.position_frame()
    pos.to_csv(f"results/leverage_{a.model}_positions.csv", index=False,
               columns=[c for c in pos.columns if not c.startswith(("lP_", "lQ_"))])
    LD.theorem_demo(pos, Lvec, lP, rows, rng, name=f"SKEMPI_{a.model}")
    LD.position_level_cpi(pos, rows, rng, name=f"SKEMPI_{a.model}")
    del Lvec, lP, lQ; gc.collect()
    sk = LD.build_skempi(rows); sk["destab"] = (sk.ddG >= LD.HOT_DDG).astype(int)
    sk.to_csv(f"results/leverage_{a.model}_mutations.csv", index=False,
              columns=[c for c in sk.columns if not c.startswith(("lP_", "lQ_"))])
    LD.run_fixture(sk, f"SKEMPI (natural, {a.model})", rows, rng)
    out = pd.DataFrame(rows); out["seed"] = SEED; out["model"] = a.model
    out["command"] = "python3 " + " ".join(sys.argv); out.to_csv(a.out, index=False)
    sp = out[(out.fixture.str.startswith("SKEMPI")) & (out.test == "spearman_L_vs_ddG")]
    print(f"\n[REPLICATION under {a.model}] Spearman(L,ddG) & CPI rows:")
    print(out[out.test.str.contains("spearman_L_vs_ddG|CPI", na=False)][["test", "stat", "lo", "hi"]].to_string(index=False))
    print(f"[done] wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pifold", "mif"])
    ap.add_argument("--stage", default="analyse", choices=["score", "analyse"])
    ap.add_argument("--cache", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-residues", type=int, default=1200); ap.add_argument("--mif-seeds", type=int, default=6)
    a = ap.parse_args()
    a.cache = a.cache or f"results/leverage_pq_skempi_{a.model}.csv"
    a.out = a.out or f"results/leverage_{a.model}.csv"
    {"score": stage_score, "analyse": stage_analyse}[a.stage](a)


if __name__ == "__main__":
    main()
