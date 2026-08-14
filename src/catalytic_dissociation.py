#!/usr/bin/env python3
"""Catalytic dissociation — is inverse-folding CONFIDENCE blind to catalytic residues EVEN THOUGH a
sequence PLM (ESM-2) is not?

The nugget generalizes beyond binding hotspots. Prior work shows sequence PLMs concentrate low entropy at
catalytic/conserved sites, so "IF finds active sites" alone would be scooped. The NOVEL, defensible claim is
the DISSOCIATION: structure-conditioned IF confidence (ProteinMPNN log p(native|backbone)) is at chance for
catalytic residues, while ESM-2 sequence entropy predicts them — because IF confidence measures backbone
DETERMINACY (constraint), and catalytic residues, like interface hotspots, are frustrated.

M-CSA catalytic residues (130 enzymes, ~495 catalytic positions; ~/ftax/data/m-csa). Per enzyme chain we
score every position with (a) ProteinMPNN confidence and (b) ESM-2 negentropy (−H of its 20-aa distribution)
and logp(native); label = M-CSA catalytic. Enzyme-clustered bootstrap, seed 20260803.
  python3 src/catalytic_dissociation.py --out results/catalytic_dissociation.csv
"""
import argparse, csv, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ftax_common as fc

SEED = 20260803
MCSA = os.path.expanduser("~/ftax/data/m-csa")
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def esm2_entropy(model, alphabet, bc, seq, aa_idx, device="cpu"):
    """Per-position (negentropy, logp_native) over the 20 aa from one ESM-2 forward (unmasked)."""
    import torch
    _, _, toks = bc([("p", seq)])
    with torch.no_grad():
        logits = model(toks.to(device))["logits"][0]          # [L+2, vocab]
    lg = logits[1:1 + len(seq), aa_idx]                        # [L, 20], drop BOS/EOS
    logp = torch.log_softmax(lg, dim=1).numpy()
    p = np.exp(logp)
    negent = (p * logp).sum(1)                                 # = -H, higher = more confident/conserved
    nat = np.array([AA20.index(a) if a in AA20 else 0 for a in seq])
    lpn = logp[np.arange(len(seq)), nat]
    return negent, lpn


def build(labels, pdbdir, out_csv):
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    import esm
    model_e, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model_e.eval(); bc = alphabet.get_batch_converter()
    aa_idx = [alphabet.get_idx(a) for a in AA20]
    model_m, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    print("loaded ESM-2 (150M) + ProteinMPNN", flush=True)

    fh = open(out_csv, "w", newline=""); writer = None; t0 = time.time()
    pdbs = sorted(labels.pdb.unique())
    for pi, pdb in enumerate(pdbs):
        path = os.path.join(pdbdir, f"{pdb}.pdb")
        if not os.path.exists(path):
            continue
        lab = labels[labels.pdb == pdb]
        chain = lab.groupby("chain").catalytic.sum().idxmax()     # dominant catalytic chain
        catset = set(lab[(lab.chain == chain) & (lab.catalytic == 1)].resid.astype(int))
        try:
            cx = fc.load_complex(path, pdb, chain, "", require_both=False)
            if cx is None or cx.n < 20 or cx.n > 700:
                continue
            lp = fc.mpnn_unconditional_logprobs(model_m, cx)
            nat = np.array([fc.MPNN_ALPHABET.index(a) for a in cx.seq])
            mpnn_conf = lp[np.arange(cx.n), nat]
            negent, esm_lpn = esm2_entropy(model_e, alphabet, bc, "".join(cx.seq), aa_idx)
            for i in range(cx.n):
                row = dict(pdb=pdb, chain=cx.chains[i], resnum=int(cx.resnums[i]), aa=cx.seq[i],
                           is_catalytic=int(int(cx.resnums[i]) in catset),
                           mpnn_conf=float(mpnn_conf[i]), esm_negent=float(negent[i]),
                           esm_logp_native=float(esm_lpn[i]))
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys())); writer.writeheader()
                writer.writerow(row)
            fh.flush()
            if (pi + 1) % 20 == 0:
                print(f"  [{pi+1}/{len(pdbs)}] {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  skip {pdb}: {type(e).__name__}: {e}", flush=True)
    fh.close()


def auc(s, y):
    from scipy import stats
    s = np.asarray(s, float); y = np.asarray(y); m = np.isfinite(s); s, y = s[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/catalytic_dissociation.csv")
    ap.add_argument("--positions", default="results/catalytic_positions.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    labels = pd.read_csv(f"{MCSA}/mcsa_labels.csv")
    if not os.path.exists(a.positions):
        print("scoring enzymes ...")
        build(labels, f"{MCSA}/pdbs", a.positions)
    d = pd.read_csv(a.positions)
    y = d.is_catalytic.to_numpy(); g = d.pdb.to_numpy()
    print(f"\npositions {len(d)}, enzymes {d.pdb.nunique()}, catalytic {int(y.sum())} "
          f"({y.mean()*100:.1f}%)")

    def boot(score):
        ids = np.unique(g); idx = {c: np.where(g == c)[0] for c in ids}; out = []
        for _ in range(5000):
            t = np.concatenate([idx[c] for c in rng.choice(ids, len(ids), True)])
            v = auc(score[t], y[t])
            if np.isfinite(v):
                out.append(v)
        return np.percentile(out, [2.5, 97.5])

    rows = []
    feats = [("MPNN_confidence", d.mpnn_conf.values), ("ESM2_negentropy", d.esm_negent.values),
             ("ESM2_logp_native", d.esm_logp_native.values)]
    for name, s in feats:
        au = auc(s, y); lo, hi = boot(s)
        v = "~chance/BLIND" if lo <= 0.5 <= hi else ("PREDICTS" if lo > 0.5 else "anti")
        print(f"  AUROC {name:18s} = {au:.3f} [{lo:.3f},{hi:.3f}]  {v}")
        rows.append(dict(feature=name, auroc=round(au, 4), lo=round(lo, 4), hi=round(hi, 4), verdict=v))
    # the dissociation: ESM-2 negentropy − MPNN confidence, paired enzyme-bootstrap
    ids = np.unique(g); idx = {c: np.where(g == c)[0] for c in ids}; dd = []
    for _ in range(5000):
        t = np.concatenate([idx[c] for c in rng.choice(ids, len(ids), True)])
        a1 = auc(d.esm_negent.values[t], y[t]); a2 = auc(d.mpnn_conf.values[t], y[t])
        if np.isfinite(a1) and np.isfinite(a2):
            dd.append(a1 - a2)
    dd = np.array(dd); lo, hi, p = np.percentile(dd, 2.5), np.percentile(dd, 97.5), float(np.mean(dd > 0))
    diss = lo > 0
    print(f"\n  DISSOCIATION ESM2_negentropy − MPNN_confidence = {dd.mean():+.3f} [{lo:+.3f},{hi:+.3f}] P(>0)={p:.3f}")
    print(f"  -> {'DISSOCIATION CONFIRMED (PLM predicts catalytic, IF confidence does not)' if diss else 'no clean dissociation'}")
    rows.append(dict(feature="dissociation_esm_minus_mpnn", auroc=round(float(dd.mean()), 4),
                     lo=round(float(lo), 4), hi=round(float(hi), 4), p_gt0=round(p, 3),
                     verdict="CONFIRMED" if diss else "no"))

    # COMPOSITION CONTROL (this project's core lesson): do the scores predict catalytic BEYOND amino-acid
    # identity? Catalytic residues are enriched in H/D/E/C/S — ProteinMPNN's worst-recovered types.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    aa_dum = pd.get_dummies(d.aa).to_numpy().astype(float)
    zc = lambda v: ((v - v.mean()) / (v.std() + 1e-9)).to_numpy()

    def cv_auc(X):
        o = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            o[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        return o

    o_base = cv_auc(aa_dum)
    o_mpnn = cv_auc(np.column_stack([aa_dum, zc(d.mpnn_conf)]))
    o_esm = cv_auc(np.column_stack([aa_dum, zc(d.esm_negent)]))
    ab, am, ae = auc(o_base, y), auc(o_mpnn, y), auc(o_esm, y)
    ids = np.unique(g); idxb = {c: np.where(g == c)[0] for c in ids}

    def dboot(o1, o0):
        out = []
        for _ in range(3000):
            t = np.concatenate([idxb[c] for c in rng.choice(ids, len(ids), True)]); yy = y[t]
            if 0 < yy.sum() < len(yy):
                out.append(auc(o1[t], yy) - auc(o0[t], yy))
        return np.percentile(out, [2.5, 97.5])
    dm_lo, dm_hi = dboot(o_mpnn, o_base); de_lo, de_hi = dboot(o_esm, o_base)
    print(f"\n  COMPOSITION CONTROL (beyond amino-acid identity):")
    print(f"    aa-identity baseline AUROC {ab:.3f}")
    print(f"    +MPNN_confidence {am:.3f}  ΔAUROC {am-ab:+.4f} [{dm_lo:+.4f},{dm_hi:+.4f}]  "
          f"{'still anti (frustration beyond composition)' if dm_hi < 0 else 'vanishes -> composition'}")
    print(f"    +ESM2_negentropy {ae:.3f}  ΔAUROC {ae-ab:+.4f} [{de_lo:+.4f},{de_hi:+.4f}]  "
          f"{'still predicts (conservation beyond composition)' if de_lo > 0 else 'vanishes -> composition'}")
    rows += [dict(feature="aa_identity_baseline", auroc=round(ab, 4)),
             dict(feature="mpnn_over_aa_identity", auroc=round(am - ab, 4), lo=round(float(dm_lo), 4),
                  hi=round(float(dm_hi), 4)),
             dict(feature="esm_negent_over_aa_identity", auroc=round(ae - ab, 4), lo=round(float(de_lo), 4),
                  hi=round(float(de_hi), 4))]
    out = pd.DataFrame(rows); out["seed"] = SEED; out["n_pos"] = len(d)
    out["n_catalytic"] = int(y.sum()); out["n_enzymes"] = int(d.pdb.nunique())
    out["note"] = "M-CSA catalytic residues; IF confidence vs ESM-2 entropy dissociation (constraint vs function)"
    out["command"] = "python3 src/catalytic_dissociation.py"
    out.to_csv(a.out, index=False)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
