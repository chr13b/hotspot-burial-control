"""Phase 1 - N_hot, the constellation cost at T = 0.1.

Tests BRIEF.md F2. BRIEF.md 5.5 says the analytic product exp(sum delta_i / T) assumes
positional independence, which is false, and that the discrepancy must be measured rather
than assumed away. This script measures it three ways.

Temperature scaling commutes with the log-softmax the model already returns
(softmax(logits/T) = softmax(log_probs/T), since a constant shift in logits is free), so
exact T-scaled conditionals come straight from forward passes.

  (1) BRIEF analytic      log10 N = sum_i delta_i / (T ln10),  delta_i = log p_mode - log p_native
                          computed on the across-decoding-order mixture marginals.

  (2) INDEPENDENT exact   each hotspot conditioned on ALL non-hotspot positions (native,
                          teacher-forced) but on NO other hotspot. One forward pass per
                          hotspot, each using a decoding order that puts every non-hotspot
                          first and that hotspot first among the hotspots.

  (3) CHAIN exact         hotspots decoded in sequence after the same non-hotspot context,
                          so each sees the earlier hotspots. One forward pass.

  (3) - (2) is the hotspot-hotspot positional correlation, holding the non-hotspot context
  fixed and native. It is exact, not sampled, and costs (1 + n_hot) forward passes per
  repetition.

  (4) DIRECT sampling     K real samples from model.sample() at T, on the subset of
                          complexes where a recovery is observable and affordable
                          (1.8 s/sample on this CPU). Validates (3) against the real sampler.

Usage:
  python3 src/p1_nhot.py --out results/p1 --positions results/p0_positions.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

T_SAMPLE = 0.1
N_ORDERS = 8
N_REPS = 4                 # independent non-hotspot context orders
SEED = 20260803


def _softmax_T(lp20, T):
    z = lp20 / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ordered_logprobs(model, cx, orders_np, device="cpu", max_batch=None):
    """Teacher-forced conditionals under a SET of explicit decoding orders. [n_orders, L, 21].

    The orders differ only in permutation, so they share X/S/mask entirely and can go through
    the model as one batch. Batching here rather than looping is a ~5x speedup: the encoder
    runs once per batch instead of once per order.
    """
    import torch
    orders_np = np.atleast_2d(orders_np)
    if max_batch is None:                      # keep B*L bounded on a 7.5 GB machine
        max_batch = int(max(1, min(6, 3000 // max(cx.n, 1))))
    X, S, mask, residue_idx, chain_enc = fc.featurize(cx, device)
    out = []
    with torch.no_grad():
        for s in range(0, len(orders_np), max_batch):
            chunk = orders_np[s:s + max_batch]
            b = len(chunk)
            order = torch.tensor(chunk, dtype=torch.long, device=device)
            mb = mask.repeat(b, 1)
            lp = model(X.repeat(b, 1, 1, 1), S.repeat(b, 1), mb, mb.clone(),
                       residue_idx.repeat(b, 1), chain_enc.repeat(b, 1),
                       torch.zeros(b, cx.n, device=device),
                       use_input_decoding_order=True, decoding_order=order)
            out.append(lp.cpu().numpy())
    return np.concatenate(out, axis=0)


def exact_constellation(model, cx, hot_idx, T=T_SAMPLE, n_reps=N_REPS, seed=SEED):
    """Independent-vs-chain constellation probability, exactly, over n_reps contexts.

    Returns per-repetition (log10 N_indep, log10 N_chain).
    """
    rng = np.random.default_rng(seed)
    L, k = cx.n, len(hot_idx)
    rest = np.setdiff1d(np.arange(L), hot_idx)
    nat = np.array([fc.MPNN_ALPHABET.index(cx.seq[i]) for i in hot_idx])
    out = []
    for _ in range(n_reps):
        ctx = rng.permutation(rest)              # shared non-hotspot context order
        hperm = rng.permutation(k)               # hotspot order for the chain variant

        # order 0      = chain    : hotspots after the context, each seeing the earlier ones
        # orders 1..k  = indep    : hotspot j first among the hotspots -> sees context only
        orders = [np.concatenate([ctx, hot_idx[hperm]])]
        for j in range(k):
            others = np.array([hot_idx[m] for m in hperm if hot_idx[m] != hot_idx[j]],
                              dtype=int)
            orders.append(np.concatenate([ctx, [hot_idx[j]], others]))
        lps = ordered_logprobs(model, cx, np.array(orders))

        pT = _softmax_T(lps[0][:, :20], T)
        log10_chain = float(-np.log10(pT[hot_idx, nat] + 1e-300).sum())
        lo = 0.0
        for j in range(k):
            pTj = _softmax_T(lps[j + 1][hot_idx[j], :20], T)
            lo += -np.log10(pTj[nat[j]] + 1e-300)
        out.append((float(lo), log10_chain))
    return out


def sample_sequences(model, cx, K, batch, temperature=T_SAMPLE, seed=SEED):
    """K sequences from the released autoregressive sampler at `temperature`. [K, L]."""
    import torch
    X, S, mask, residue_idx, chain_enc = fc.featurize(cx)
    L = cx.n
    omit = np.array([1.0 if a == "X" else 0.0 for a in fc.MPNN_ALPHABET], dtype=np.float32)
    bias = np.zeros(21, dtype=np.float32)
    out, done = [], 0
    torch.manual_seed(seed)
    with torch.no_grad():
        while done < K:
            b = min(batch, K - done)
            d = model.sample(
                X.repeat(b, 1, 1, 1), torch.randn(b, L), S.repeat(b, 1),
                torch.ones(b, L), chain_enc.repeat(b, 1), residue_idx.repeat(b, 1),
                mask=mask.repeat(b, 1), temperature=temperature,
                omit_AAs_np=omit, bias_AAs_np=bias, chain_M_pos=torch.ones(b, L),
                omit_AA_mask=None, pssm_coef=None, pssm_bias=None, pssm_multi=0.0,
                pssm_log_odds_flag=False, pssm_log_odds_mask=None,
                pssm_bias_flag=False, bias_by_res=torch.zeros(b, L, 21))
            out.append(d["S"].cpu().numpy())
            done += b
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p1")
    ap.add_argument("--positions", default="results/p0_positions.csv")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--mpnn-weights",
                    default=os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--direct-K", type=int, default=400)
    ap.add_argument("--direct-budget-s", type=float, default=5400,
                    help="wall-clock budget for the DIRECT sampling arm")
    ap.add_argument("--direct-max-log10", type=float, default=2.3,
                    help="only sample complexes whose exact N_hot is resolvable with K draws")
    ap.add_argument("--direct-n-marginal", type=int, default=8,
                    help="complexes to sample for the marginal-recovery check when the "
                         "joint constellation is unobservable everywhere")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cmd = "python3 " + " ".join(sys.argv)

    pos = pd.read_csv(a.positions)
    hot = pos[pos["label"].isin(["hot_loose", "hot_strict"]) & pos["is_interface"]]
    cx_ids = sorted(hot["complex_id"].unique())
    if a.limit:
        cx_ids = cx_ids[: a.limit]
    print(f"[p1] {len(cx_ids)} complexes carry >=1 interface hotspot "
          f"({len(hot)} hotspot positions); T={T_SAMPLE}")

    model, _ = fc.load_mpnn(a.mpnn_weights)

    # Stream the exact arm to disk, and resume from it if it already exists. Torch's CPU
    # allocator holds a high-water mark of several GB on this 7.5 GB machine, so an OOM
    # part-way through must not destroy hours of completed complexes.
    import csv
    import gc
    exact_csv = f"{a.out}_nhot_exact.csv"
    done = set()
    if os.path.exists(exact_csv):
        try:
            prev = pd.read_csv(exact_csv)
            done = set(prev["complex_id"])
            print(f"[p1] resuming: {len(done)} complexes already in {exact_csv}")
        except Exception:
            done = set()
    fh = open(exact_csv, "a" if done else "w", newline="")
    writer = None
    rows, t0 = [], time.time()

    for ci, cid in enumerate(cx_ids):
        if cid in done:
            continue
        pdb, g1, g2 = cid.split("_")
        path = os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            continue
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None:
            continue
        H = hot[hot["complex_id"] == cid].sort_values("idx")
        idx = H["idx"].values.astype(int)
        if idx.max() >= cx.n or not all(cx.seq[i] == aa for i, aa in zip(idx, H["aa"].values)):
            continue

        # (1) BRIEF analytic, on the across-order mixture marginals
        lp = fc.mpnn_conditional_logprobs(model, cx, seeds=range(N_ORDERS))
        mix = fc.order_mixture_logprobs(lp)[:, :20]
        mix = mix - np.log(np.exp(mix).sum(axis=1, keepdims=True))
        nat = np.array([fc.MPNN_ALPHABET.index(cx.seq[i]) for i in idx])
        delta = mix[idx, :].max(axis=1) - mix[idx, nat]
        log10_brief = float(delta.sum() / T_SAMPLE / np.log(10))

        # (2)+(3) exact independent vs chain
        ex = exact_constellation(model, cx, idx)
        li = np.array([e[0] for e in ex]); lc = np.array([e[1] for e in ex])

        rec = dict(
            complex_id=cid, pdb=pdb, n_hot=len(idx), L=cx.n,
            sum_delta_nats=float(delta.sum()), max_delta_nats=float(delta.max()),
            log10_Nhot_brief=log10_brief,
            log10_Nhot_indep=float(li.mean()), log10_Nhot_indep_sd=float(li.std(ddof=1)),
            log10_Nhot_chain=float(lc.mean()), log10_Nhot_chain_sd=float(lc.std(ddof=1)),
            positional_correlation_log10=float((lc - li).mean()),
            n_reps=N_REPS,
            hot_native="".join(cx.seq[idx]),
            hot_mode="".join(fc.MPNN_ALPHABET[m] for m in mix[idx, :].argmax(axis=1)),
            hot_resnums=";".join(f"{c}{r}" for c, r in zip(H["chain"], H["resnum"])),
        )
        rows.append(rec)
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(rec.keys()))
            if not done:
                writer.writeheader()
        writer.writerow(rec)
        fh.flush()
        del lp, mix
        gc.collect()

        if len(rows) % 10 == 0:
            el = time.time() - t0
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"[p1-exact] {len(rows)+len(done)}/{len(cx_ids)}  {el:.0f}s "
                  f"({el/len(rows):.1f}s/complex)  peak RSS {rss:.2f} GB", flush=True)

    fh.close()
    df = pd.read_csv(exact_csv)

    # ---------------------------------------------------- (4) direct sampling arm
    print(f"\n[p1-direct] budget {a.direct_budget_s:.0f}s, K={a.direct_K}")
    df["log10_Nhot_direct"] = np.nan
    df["direct_joint_freq"] = np.nan
    df["log10_Nhot_indep_from_samples"] = np.nan
    df["direct_K"] = 0
    df["direct_zero_recoveries"] = False

    df["direct_marg_obs"] = np.nan
    df["direct_marg_pred"] = np.nan

    elig = df[df["log10_Nhot_chain"] <= a.direct_max_log10].sort_values("L")
    print(f"[p1-direct] {len(elig)} complexes JOINT-resolvable at K={a.direct_K} "
          f"(log10 N_chain <= {a.direct_max_log10})")
    if len(elig) == 0:
        # The joint constellation is unobservable at any affordable K. The per-position
        # MARGINAL still is, and validating it checks the temperature-scaled conditionals
        # against the released sampler - the machinery N_hot is built from.
        elig = df.sort_values("L").head(a.direct_n_marginal)
        print(f"[p1-direct] joint unobservable everywhere; instead validating per-hotspot "
              f"MARGINAL recovery on the {len(elig)} smallest complexes")
    t1 = time.time()
    for _, r in elig.iterrows():
        if time.time() - t1 > a.direct_budget_s:
            print("[p1-direct] budget exhausted"); break
        cid = r["complex_id"]; pdb, g1, g2 = cid.split("_")
        cx = fc.load_complex(os.path.join(a.data_dir, "PDBs", f"{pdb}.pdb"), pdb, g1, g2)
        if cx is None:
            continue
        H = hot[hot["complex_id"] == cid].sort_values("idx")
        idx = H["idx"].values.astype(int)
        nat = np.array([fc.MPNN_ALPHABET.index(cx.seq[i]) for i in idx])
        batch = int(max(4, min(48, 12000 // max(cx.n, 1))))   # RAM-bound (7.5 GB machine)
        try:
            S = sample_sequences(model, cx, a.direct_K, batch)
        except Exception as e:
            print(f"[p1-direct] {cid} failed: {type(e).__name__}: {e}"); continue
        hits = (S[:, idx] == nat[None, :])
        joint = float(hits.all(axis=1).mean())
        marg = hits.mean(axis=0)
        # analytic per-position marginal at temperature T, from the same conditionals
        lp = fc.mpnn_conditional_logprobs(model, cx, seeds=range(N_ORDERS))
        mix = fc.order_mixture_logprobs(lp)[:, :20]
        mix = mix - np.log(np.exp(mix).sum(axis=1, keepdims=True))
        pred = _softmax_T(mix[idx, :], T_SAMPLE)[np.arange(len(idx)), nat]

        m = df["complex_id"] == cid
        df.loc[m, "direct_K"] = a.direct_K
        df.loc[m, "direct_joint_freq"] = joint
        df.loc[m, "direct_zero_recoveries"] = (joint == 0.0)
        df.loc[m, "log10_Nhot_direct"] = -np.log10(joint) if joint > 0 else np.nan
        df.loc[m, "direct_marg_obs"] = float(marg.mean())
        df.loc[m, "direct_marg_pred"] = float(pred.mean())
        if (marg > 0).all():
            df.loc[m, "log10_Nhot_indep_from_samples"] = float(-np.log10(marg).sum())
        print(f"[p1-direct] {cid:16s} L={cx.n:4d} k={len(idx)} joint={joint:.4f} "
              f"chain_pred_log10={r['log10_Nhot_chain']:.2f} | mean hotspot marginal: "
              f"observed={marg.mean():.3f} predicted={pred.mean():.3f}", flush=True)

    df["command"] = cmd
    df["T"] = T_SAMPLE
    df["mpnn_ckpt"] = os.path.basename(a.mpnn_weights)
    df.to_csv(f"{a.out}_nhot.csv", index=False)

    # ---------------------------------------------------------------- summary
    s = []
    def add(k, v):
        s.append(dict(metric=k, value=v)); print(f"  {k:46s} {v}")

    print("\n=== N_hot (T = 0.1) ===")
    add("n_complexes_with_hotspots", len(df))
    add("median_n_hot_positions", float(df["n_hot"].median()))
    add("median_log10_Nhot_brief_formula", round(float(df["log10_Nhot_brief"].median()), 3))
    add("median_log10_Nhot_independent_exact", round(float(df["log10_Nhot_indep"].median()), 3))
    add("median_log10_Nhot_chain_exact", round(float(df["log10_Nhot_chain"].median()), 3))
    add("IQR_log10_Nhot_chain_exact",
        f"[{df['log10_Nhot_chain'].quantile(.25):.2f}, {df['log10_Nhot_chain'].quantile(.75):.2f}]")
    add("frac_complexes_log10_Nhot_chain_ge_2", round(float((df["log10_Nhot_chain"] >= 2).mean()), 3))
    add("median_positional_correlation_log10",
        round(float(df["positional_correlation_log10"].median()), 3))
    add("IQR_positional_correlation_log10",
        f"[{df['positional_correlation_log10'].quantile(.25):.2f}, "
        f"{df['positional_correlation_log10'].quantile(.75):.2f}]")

    obs = df[df["log10_Nhot_direct"].notna()]
    add("n_complexes_direct_sampled", int((df["direct_K"] > 0).sum()))
    add("n_complexes_direct_resolved", len(obs))
    add("n_complexes_direct_zero_recoveries", int(df["direct_zero_recoveries"].sum()))
    mm = df[df["direct_marg_obs"].notna()]
    if len(mm):
        add("n_complexes_marginal_check", len(mm))
        add("mean_hotspot_marginal_OBSERVED", round(float(mm["direct_marg_obs"].mean()), 4))
        add("mean_hotspot_marginal_PREDICTED", round(float(mm["direct_marg_pred"].mean()), 4))
        add("marginal_obs_minus_pred", round(float((mm["direct_marg_obs"]
                                                    - mm["direct_marg_pred"]).mean()), 4))
    if len(obs):
        d = obs["log10_Nhot_direct"] - obs["log10_Nhot_chain"]
        add("median_direct_minus_chain_log10", round(float(d.median()), 3))
        add("median_abs_direct_minus_chain_log10", round(float(d.abs().median()), 3))
        d2 = obs["log10_Nhot_direct"] - obs["log10_Nhot_indep"]
        add("median_direct_minus_independent_log10", round(float(d2.median()), 3))
        d3 = obs["log10_Nhot_direct"] - obs["log10_Nhot_brief"]
        add("median_direct_minus_BRIEFformula_log10", round(float(d3.median()), 3))

    pd.DataFrame(s).assign(command=cmd).to_csv(f"{a.out}_summary.csv", index=False)
    med = float(df["log10_Nhot_chain"].median())
    print(f"\nSUMMARY | median log10 N_hot (chain, exact) = {med:.2f} over {len(df)} complexes"
          f" | F2 clause-1 (median log10 N_hot < 2) is {'TRUE' if med < 2 else 'FALSE'}")
    print(f"[done] wrote {a.out}_nhot.csv and {a.out}_summary.csv")


if __name__ == "__main__":
    main()
