#!/usr/bin/env python3
"""ATLAS (TCR-pMHC) leverage replication — the SECOND natural ddG_bind fixture.

Pre-registration: results/PREREG_atlas.md (FROZEN). SEED=20260803. This is a bonus
generalization test; a null here bounds generalization and does not refute SKEMPI.

Reuses the committed machinery verbatim: ftax_common.load_complex / mpnn_unconditional_logprobs /
residue_sasa / neighbour_counts / relative_sasa, p0_burial_matched.atom_sasa (geometry),
leverage_decomposition.{logdists,leverage,_finish,cpi,run_fixture,zs,boot_stat}. The ESM-IF1 arm
swaps only the marginal source (models.ftax_esmif). Nothing about the leverage/CPI math is re-implemented.

Stages
  build-fixture : parse ATLAS -> results/atlas_fixture.csv. DIRECT author-number mapping
                  (ATLAS `num` == PDB author resnum on TCR_PDB_chain) + a MANDATORY WT-identity gate
                  (structure residue identity must equal the mutation's wt letter, else dropped+logged).
  score         : per interface/mutated TCR position, P=p(.|complex) and Q=p(.|TCR-only), leverage L_i(a)
                  and the P-functionals + geometry -> results/atlas_pq_{model}.csv (gitignored, derivable).
                  --model {mpnn,esmif}. Prints scorer positive controls (native recovery + bit-identical rescore).
  analyse       : H1 (Spearman(L,ddG)<0 & CPI(L|geometry)>placebo floor) and H2 (CPI(confidence|geometry)~0),
                  on the FULL set and the SKEMPI-non-overlapping subset, for both models. Writes the per-mutation
                  results/atlas_leverage.csv and the stats results/atlas_summary.csv.

  srun -p normal -c4 --mem=16G -t 00:40:00 bash -c 'source $SCRATCH/ftax/env.sh; \
       python3 src/leverage_atlas.py --stage build-fixture && python3 src/leverage_atlas.py --stage score --model mpnn'
  # ESM-IF1 arm under env_esmif.sh; then --stage analyse under env.sh.
"""
import argparse
import gc
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ftax_common as fc                      # noqa: E402
import leverage_decomposition as LD           # noqa: E402

SEED = 20260803
AA20 = LD.AA20
IDX = LD.IDX
HOT_DDG = LD.HOT_DDG                           # 1.0 kcal/mol destabilising-for-binding threshold
INTERFACE_DRSASA = 0.05                        # == p0_burial_matched.INTERFACE_DRSASA

ATLAS = f"{REPO}/data/atlas"
MUT_TSV = f"{ATLAS}/Mutants.tsv"
CDR_TXT = f"{ATLAS}/CDR_seqs.txt"
PDBDIR = f"{ATLAS}/pdb"
SKEMPI_POS = f"{REPO}/results/leverage_skempi_positions.csv"

FIXTURE_CSV = f"{REPO}/results/atlas_fixture.csv"
LEVERAGE_CSV = f"{REPO}/results/atlas_leverage.csv"
SUMMARY_CSV = f"{REPO}/results/atlas_summary.csv"

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
       'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
       'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
       'MSE': 'M', 'SEC': 'C', 'PYL': 'K'}


# --------------------------------------------------------------------------- shared parsing helpers

def parse_chain_residues(pdb_path, chain):
    """Ordered [(resnum:int, icode:str, aa:str)] over CA atoms of `chain` (stdlib; matches the WT gate)."""
    res, seen = [], set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[21] != chain:
                continue
            if line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
                continue
            aa = AA3.get(line[17:20].strip())
            if aa is None:
                continue
            rn, ic = int(line[22:26]), line[26].strip()
            if (rn, ic) in seen:
                continue
            seen.add((rn, ic))
            res.append((rn, ic, aa))
    return res


def load_cdr_groups():
    """{PDB: (pmhc_chain_str, tcr_chain_str)} from CDR_seqs.txt (authoritative chain grouping)."""
    out = {}
    with open(CDR_TXT) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {h: i for i, h in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= ix["TCR"]:
                continue
            out[f[ix["PDB ID"]].strip().upper()] = (f[ix["pMHC"]].strip(), f[ix["TCR"]].strip())
    return out


def parse_ddg(s):
    """ATLAS ddG cells carry uncertainty ('0.19 ± 0.11'); take the leading signed float (None if N/A)."""
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else None


def skempi_codes():
    codes = set()
    with open(SKEMPI_POS) as fh:
        next(fh)
        for line in fh:
            codes.add(line.split(",", 1)[0].split("_")[0].upper())
    return codes


# --------------------------------------------------------------------------- stage: build-fixture

def stage_build_fixture(a):
    groups = load_cdr_groups()
    skempi = skempi_codes()
    resid_cache = {}
    mut_re = re.compile(r"^([A-Z])(\d+)([A-Z])$")

    with open(MUT_TSV) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
    ix = {h: i for i, h in enumerate(hdr)}

    recs = []
    cur_pdb = None
    n_rows = n_mut = n_single = n_multi = 0
    gate_pass = gate_fail = no_struct = 0

    with open(MUT_TSV) as fh:
        next(fh)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                f += [""] * (len(hdr) - len(f))
            n_rows += 1
            tp = f[ix["true_PDB"]].strip()
            if tp and tp != "N/A":
                cur_pdb = tp.upper()                          # WT-anchor row sets the block structure
            ddg = f[ix["Delta_DeltaG_kcal_per_mol"]].strip()
            tcr_mut = f[ix["TCR_mut"]].strip()
            ddgv = parse_ddg(ddg)
            if ddgv is None or tcr_mut in ("", "WT"):
                continue
            n_mut += 1
            pdb = cur_pdb
            pmhc_pbid = f[ix["pMHC_PBID"]].strip()
            if (not pdb) and pmhc_pbid and pmhc_pbid != "N/A":
                pdb = pmhc_pbid.upper()
            m = mut_re.match(tcr_mut)
            if not m:
                n_multi += 1                                  # multi-point / non-canonical -> first pass skips
                continue
            n_single += 1
            wt, num, mt = m.group(1), int(m.group(2)), m.group(3)
            chain = f[ix["TCR_PDB_chain"]].strip()
            cdr = f[ix["CDR"]].strip()
            mut_chain = f[ix["TCR_mut_chain"]].strip()
            wtcdr = f[ix["wtCDRseq"]].strip()
            tcrname = f[ix["TCRname"]].strip()
            g1 = g2 = ""
            if pdb in groups:
                pmhc, tcr = groups[pdb]
                g1, g2 = tcr, pmhc                            # g1 = TCR (scored), g2 = pMHC (partner, ablated)
            pdbfile = f"{PDBDIR}/{pdb}.pdb"

            gp, struct_aa = False, ""
            if pdb and os.path.exists(pdbfile) and len(chain) == 1:
                key = (pdb, chain)
                if key not in resid_cache:
                    resid_cache[key] = parse_chain_residues(pdbfile, chain)
                hits = [(rn, ic, aa) for (rn, ic, aa) in resid_cache[key] if rn == num]
                struct_aa = ",".join(aa for _, _, aa in hits)
                gp = any(aa == wt for _, _, aa in hits)      # <-- the mandatory WT-identity gate
            else:
                no_struct += 1
            gate_pass += int(gp)
            gate_fail += int(not gp)

            # icode of the matched residue (prefer the blank-icode residue carrying wt)
            icode = ""
            if gp:
                for rn, ic, aa in resid_cache[(pdb, chain)]:
                    if rn == num and aa == wt:
                        icode = ic
                        break
            recs.append(dict(complex_id=f"{pdb}_{g1}_{g2}" if g1 else pdb, pdb=pdb, g1=g1, g2=g2,
                             chain=chain, resnum=num, icode=icode, wt=wt, mut=mt, num_atlas=num,
                             cdr=cdr, mut_chain=mut_chain, tcrname=tcrname, wtcdr=wtcdr,
                             ddG=ddgv, gate_pass=bool(gp), struct_aa=struct_aa,
                             is_overlap=int(pdb in skempi)))

    df = pd.DataFrame(recs)
    # aggregate replicate measurements of the SAME structural substitution (mean ddG, n_meas=count)
    keyc = ["complex_id", "pdb", "g1", "g2", "chain", "resnum", "icode", "wt", "mut",
            "num_atlas", "cdr", "mut_chain", "tcrname", "wtcdr", "gate_pass", "is_overlap", "struct_aa"]
    agg = (df.groupby(keyc, dropna=False)
             .agg(ddG=("ddG", "mean"), n_meas=("ddG", "size")).reset_index())
    agg = agg.sort_values(["is_overlap", "pdb", "chain", "resnum"]).reset_index(drop=True)
    agg.to_csv(FIXTURE_CSV, index=False)

    npass = int(agg.gate_pass.sum())
    print(f"[build-fixture] rows={n_rows} mut-rows={n_mut} single-point={n_single} multi/skip={n_multi}")
    print(f"[build-fixture] unique substitutions={len(agg)}  WT-gate PASS={npass} "
          f"({100*npass/max(len(agg),1):.1f}%)  fail={len(agg)-npass}  (no-structure rows={no_struct})")
    pas = agg[agg.gate_pass]
    print(f"[build-fixture] PASSING: {len(pas)} substitutions over {pas.pdb.nunique()} structures; "
          f"SKEMPI-overlap {int(pas.is_overlap.sum())} / non-overlap {int((pas.is_overlap == 0).sum())}")
    print("[build-fixture] passing per structure (overlap flag):")
    for pdb, g in pas.groupby("pdb"):
        print(f"    {pdb}  n={len(g):3d}  overlap={int(g.is_overlap.iloc[0])}  "
              f"g1(TCR)={g.g1.iloc[0]} g2(pMHC)={g.g2.iloc[0]}")
    print(f"[build-fixture] wrote {FIXTURE_CSV}")


# --------------------------------------------------------------------------- stage: score

def _score_positions(scorer, path, pdb, g1, g2, keep):
    """Mirror of LD._score_one but with a pluggable `scorer(cx)->[L,21] logprobs (MPNN-alphabet order)`,
    so MPNN (unconditional) and ESM-IF1 (native-conditional) share one partner-ablation path.
    P = p(.|complex); Q = p(.|group-in-isolation) (pMHC deleted for TCR residues)."""
    cx = fc.load_complex(path, pdb, g1, g2)
    if cx is None:
        return [], None
    lP = LD.logdists(scorer(cx))
    lQ = np.full_like(lP, np.nan)
    for chains in (g1, g2):
        if not chains:
            continue
        mono = fc.load_complex(path, pdb, chains, "", require_both=False)
        if mono is None or mono.n < 5:
            continue
        lQm = LD.logdists(scorer(mono))
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
        lPv, lQv = lP[j], lQ[j]
        wi = IDX.get(str(cx.seq[j]))
        if wi is None:
            continue
        rvec = lPv - lQv
        Lvec = rvec - rvec[wi]
        Ln = Lvec.copy()
        Ln[wi] = np.nan
        r = dict(chain=key[0], resnum=key[1], icode=key[2], aa=str(cx.seq[j]), group=int(cx.group[j]),
                 conf=float(lPv[wi]),
                 klP=float((np.exp(lPv) * rvec).sum()),
                 negH=float((np.exp(lPv) * lPv).sum()),
                 L_ala=float(Lvec[IDX["A"]]),
                 L_rms=float(np.sqrt(np.nanmean(Ln ** 2))),
                 L_min=float(np.nanmin(Ln)), L_mean=float(np.nanmean(Ln)))
        for a in AA20:
            r[f"lP_{a}"] = float(lPv[IDX[a]])
        for a in AA20:
            r[f"lQ_{a}"] = float(lQv[IDX[a]])
        rows.append(r)
    del lP, lQ, cx
    gc.collect()
    return rows, None


def _complex_geometry(path, pdb, g1, g2, cx):
    """burial / nbr / rsasa_complex / drsasa / is_interface per residue — replicates abbind_nugget.py:76-92."""
    import p0_burial_matched as p0
    all_ch = g1 + g2
    asa_b_atom = p0.atom_sasa(path, pdb, all_ch)
    asa_b = {}
    for (c, rn, ic, _an), v in asa_b_atom.items():
        asa_b[(c, rn, ic)] = asa_b.get((c, rn, ic), 0.0) + v
    asa_f1 = fc.residue_sasa(path, pdb, g1)
    asa_f2 = fc.residue_sasa(path, pdb, g2)
    nbr = fc.neighbour_counts(cx)
    geo = {}
    for i in range(cx.n):
        key = (cx.chains[i], int(cx.resnums[i]), cx.icodes[i])
        aa = str(cx.seq[i])
        sb = asa_b.get(key, np.nan)
        sf = (asa_f1 if cx.group[i] == 1 else asa_f2).get(key, np.nan)
        rb, rf = fc.relative_sasa(sb, aa), fc.relative_sasa(sf, aa)
        geo[key] = dict(rsasa_complex=rb, drsasa=(rf - rb), burial=-rb, nbr=int(nbr[i]),
                        is_interface=int((rf - rb) > INTERFACE_DRSASA))
    return geo


def _load_scorer(model, device):
    if model == "mpnn":
        mdl, _ = fc.load_mpnn(LD.MPNN_W)
        try:
            mdl = mdl.to(device)
        except Exception:
            pass
        # leverage uses the sequence-free unconditional marginal; recovery uses the standard
        # native-context conditional (8-order) readout, which is what the ~0.3-0.5 band refers to.
        return (lambda cx: fc.mpnn_unconditional_logprobs(mdl, cx, device=device),
                lambda cx: fc.mpnn_conditional_logprobs(mdl, cx, seeds=range(8),
                                                        device=device).mean(axis=0), mdl)
    elif model == "esmif":
        import models.ftax_esmif as fe
        ckpt = os.environ.get("FTAX_ESMIF_CKPT", getattr(fe, "DEFAULT_CKPT", None))
        mdl, alphabet = fe.load_esmif(ckpt, device=device)
        sc = (lambda cx: fe.esmif_conditional_logprobs(mdl, alphabet, cx, seeds=(0,),
                                                       device=device).mean(axis=0))
        return sc, sc, mdl                          # ESM-IF1 readout is already native-conditional
    raise SystemExit(f"unknown model {model}")


def stage_score(a):
    import torch
    torch.set_num_threads(a.threads)
    fx = pd.read_csv(FIXTURE_CSV)
    fx["icode"] = fx.icode.fillna("").astype(str)
    fx = fx[fx.gate_pass & (fx.g1 != "") & fx.g1.notna()].copy()
    scorer, rec_scorer, mdl = _load_scorer(a.model, a.device)
    comps = fx.groupby("complex_id").agg(pdb=("pdb", "first"), g1=("g1", "first"),
                                         g2=("g2", "first")).reset_index()

    rec_hit = rec_tot = 0
    all_rows = []
    for _, cr in comps.iterrows():
        cid, pdb, g1, g2 = cr.complex_id, cr.pdb, cr.g1, cr.g2
        path = f"{PDBDIR}/{pdb}.pdb"
        cx = fc.load_complex(path, pdb, g1, g2)
        if cx is None:
            print(f"  [skip] {cid}: load_complex None")
            continue
        geo = _complex_geometry(path, pdb, g1, g2, cx)
        sub = fx[fx.complex_id == cid]
        mutkeys = {(r.chain, int(r.resnum), str(r.icode)) for r in sub.itertuples()}
        keep = {k for k, v in geo.items() if v["is_interface"] == 1} | mutkeys
        rows, _ = _score_positions(scorer, path, pdb, g1, g2, keep)
        for r in rows:
            key = (r["chain"], r["resnum"], r["icode"])
            g = geo.get(key)
            if g is None:
                continue
            r.update(g)
            r["complex_id"] = cid
            all_rows.append(r)
        # scorer sanity: pooled TCR-interface native recovery via the native-context readout
        rdist = LD.logdists(rec_scorer(cx))
        for j in range(cx.n):
            key = (cx.chains[j], int(cx.resnums[j]), cx.icodes[j])
            if cx.group[j] == 1 and geo.get(key, {}).get("is_interface") == 1:
                rec_tot += 1
                rec_hit += int(AA20[int(np.argmax(rdist[j]))] == str(cx.seq[j]))

    out = pd.DataFrame(all_rows)
    out["model"] = a.model
    pq = f"{REPO}/results/atlas_pq_{a.model}.csv"
    out.to_csv(pq, index=False)
    print(f"[score:{a.model}] scored {out.complex_id.nunique()} complexes, {len(out)} positions "
          f"({int((out.is_interface == 1).sum())} interface) -> {pq}")
    if rec_tot:
        print(f"[score:{a.model}] POSITIVE CONTROL native recovery (native-context) on TCR interface "
              f"= {rec_hit/rec_tot:.3f} over {rec_tot} positions across {len(comps)} complexes "
              f"(expect ~0.3-0.5; TCR CDRs run lower)")

    # bit-identical re-score control on the first complex
    cr = comps.iloc[0]
    r1, _ = _score_positions(scorer, f"{PDBDIR}/{cr.pdb}.pdb", cr.pdb, cr.g1, cr.g2, None)
    r2, _ = _score_positions(scorer, f"{PDBDIR}/{cr.pdb}.pdb", cr.pdb, cr.g1, cr.g2, None)
    if r1 and r2:
        v1 = np.array([[x[f"lP_{aa}"] for aa in AA20] for x in r1])
        v2 = np.array([[x[f"lP_{aa}"] for aa in AA20] for x in r2])
        print(f"[score:{a.model}] POSITIVE CONTROL bit-identical re-score: max|Δlp|={np.abs(v1-v2).max():.2e} "
              f"(expect ~0)")


# --------------------------------------------------------------------------- stage: analyse

def _build_mut_frame(model):
    """Join gate-passing ATLAS mutations to their scored position -> the leverage_skempi_mutations schema."""
    pq = pd.read_csv(f"{REPO}/results/atlas_pq_{model}.csv")
    pq["icode"] = pq.icode.fillna("").astype(str)
    fx = pd.read_csv(FIXTURE_CSV)
    fx["icode"] = fx.icode.fillna("").astype(str)
    fx = fx[fx.gate_pass].copy()
    key = ["complex_id", "chain", "resnum", "icode"]
    m = fx.merge(pq, on=key, how="inner", suffixes=("", "_pq"))
    if len(m) == 0:
        return m
    # positive control: the scored native residue must equal the mutation wt letter
    bad = int((m["aa"] != m["wt"]).sum())
    if bad:
        print(f"  [WARN] {bad} merged rows with scored-aa != wt (dropped)")
        m = m[m.aa == m.wt].reset_index(drop=True)
    ar = np.arange(len(m))
    lP = m[[f"lP_{a}" for a in AA20]].to_numpy()
    lQ = m[[f"lQ_{a}" for a in AA20]].to_numpy()
    wi = m.wt.map(IDX).to_numpy()
    mi = m.mut.map(IDX).to_numpy()
    m["L"] = (lP[ar, mi] - lP[ar, wi]) - (lQ[ar, mi] - lQ[ar, wi])
    m["logP_mut"] = lP[ar, mi]
    m["destab"] = (m.ddG >= HOT_DDG).astype(int)
    m = LD._finish(m, "atlas")
    m["model"] = model
    keepcols = ["complex_id", "model", "pdb", "chain", "resnum", "icode", "wt", "mut", "cdr",
                "ddG", "n_meas", "aa", "L", "logP_mut", "conf", "klP", "negH", "nbr",
                "rsasa_complex", "drsasa", "is_interface", "burial", "blosum", "dvol", "dhydro",
                "is_overlap", "destab"]
    return m[[c for c in keepcols if c in m.columns]].copy()


def _placebo_floor(y, g, Zgeo, d, rng):
    """Mutation/position-level placebo floor: conservative = max upper-CI over information-free features
    (mirrors w_placebo_ladder.py but with the caller's label y)."""
    bur, nbr, ds = LD.zs(d.burial), LD.zs(d.nbr), LD.zs(d.drsasa)
    placebos = {"dup_dSASA": ds, "dup_nbr": nbr, "dSASA^2": LD.zs(ds ** 2),
                "nbr*dSASA": LD.zs(nbr * ds), "pure_noise": LD.zs(rng.standard_normal(len(d)))}
    his = []
    for _name, X in placebos.items():
        try:
            _c, _lo, hi, _p, _s, _cc = LD.cpi(y, g, Zgeo, np.asarray(X, float).copy(), rng)
            his.append(hi)
        except Exception:
            continue
    return (max(his) if his else float("nan"))


def _cpi_safe(y, g, Z, X, rng):
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 2:
        return None
    try:
        c, lo, hi, p, _s, _cc = LD.cpi(y, g, Z, np.asarray(X, float).copy(), rng)
        return dict(cpi=float(c), lo=float(lo), hi=float(hi), p=float(p))
    except Exception as e:
        print(f"    [cpi failed: {e}]")
        return None


def _spearman_boot(x, y, g, rng, nboot=5000):
    from scipy import stats
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = np.asarray(x)[ok], np.asarray(y)[ok], np.asarray(g)[ok]
    if len(x) < 4 or len(np.unique(g)) < 2:
        return (float(stats.spearmanr(x, y).correlation) if len(x) >= 4 else float("nan"),
                float("nan"), float("nan"))
    pt = float(stats.spearmanr(x, y).correlation)
    ids = np.unique(g)
    by = {k: np.where(g == k)[0] for k in ids}
    b = []
    for _ in range(nboot):
        idx = np.concatenate([by[k] for k in rng.choice(ids, len(ids), True)])
        b.append(stats.spearmanr(x[idx], y[idx]).correlation)
    b = np.array(b, float)
    b = b[np.isfinite(b)]
    return pt, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _analyse_subset(m, label, model, subset, rng, rows):
    """H1 (mutation-level) + H2 (position-level) on one (model, subset) slice; appends structured rows."""
    d = m[m.is_interface == 1].dropna(
        subset=["L", "conf", "klP", "burial", "nbr", "drsasa", "ddG"]).reset_index(drop=True)
    n, ncx = len(d), d.complex_id.nunique()
    npos = int((d.destab == 1).sum())
    print(f"\n=== {label} [{model}] : {n} interface mutations over {ncx} complexes "
          f"({npos} destabilising, {100*d.destab.mean() if n else 0:.0f}%) ===")
    base = dict(model=model, subset=subset, n=n, n_complexes=int(ncx), n_destab=npos)
    if n < 4 or ncx < 2:
        print("  INDETERMINATE: too few mutations/complexes for a stable estimate.")
        rows.append({**base, "test": "status", "note": "indeterminate (underpowered)"})
        return

    # ---- H1: Spearman(L, ddG)  (pre-registered: < 0)
    sp, slo, shi = _spearman_boot(d.L.to_numpy(), d.ddG.to_numpy(), d.complex_id.to_numpy(), rng)
    print(f"  H1 Spearman(L, ddG_bind) = {sp:+.4f} [{slo:+.4f},{shi:+.4f}]  (pre-reg: NEGATIVE)")
    rows.append({**base, "test": "spearman_L_vs_ddG", "stat": round(sp, 4),
                 "lo": round(slo, 4) if np.isfinite(slo) else "", "hi": round(shi, 4) if np.isfinite(shi) else ""})
    for f in ("conf", "klP"):
        s2, _, _ = _spearman_boot(d[f].to_numpy(), d.ddG.to_numpy(), d.complex_id.to_numpy(), rng)
        rows.append({**base, "test": f"spearman_{f}_vs_ddG", "stat": round(s2, 4)})

    # ---- H1: CPI(L | geometry) vs the placebo floor
    y = d.destab.to_numpy().astype(float)
    g = d.complex_id.to_numpy()
    Zgeo = np.column_stack([LD.zs(d.burial), LD.zs(d.nbr), LD.zs(d.drsasa)])
    cpi_L = _cpi_safe(y, g, Zgeo, LD.zs(d.L), rng)
    floor = _placebo_floor(y, g, Zgeo, d, rng)
    if cpi_L:
        clears = np.isfinite(floor) and cpi_L["lo"] > floor
        print(f"  H1 CPI(L | burial+nbr+dSASA) = {cpi_L['cpi']:+.5f} "
              f"[{cpi_L['lo']:+.5f},{cpi_L['hi']:+.5f}] P(>0)={cpi_L['p']:.3f}  "
              f"placebo-floor={floor:+.5f}  -> {'CLEARS floor' if clears else 'does NOT clear floor'}")
        rows.append({**base, "test": "CPI(L|geometry)", "stat": round(cpi_L["cpi"], 5),
                     "lo": round(cpi_L["lo"], 5), "hi": round(cpi_L["hi"], 5),
                     "p_gt0": round(cpi_L["p"], 3), "placebo_floor": round(floor, 5) if np.isfinite(floor) else "",
                     "clears_floor": bool(clears)})
    else:
        print("  H1 CPI(L|geometry): INDETERMINATE (cpi not estimable)")
        rows.append({**base, "test": "CPI(L|geometry)", "note": "indeterminate"})

    # ---- H2: position-level CPI(confidence | geometry) ~ 0  (positions = mutated+scored; is_hot observed)
    pos = d.groupby(["complex_id", "chain", "resnum", "icode"]).agg(
        is_hot=("destab", "max"), conf=("conf", "first"), burial=("burial", "first"),
        nbr=("nbr", "first"), drsasa=("drsasa", "first")).reset_index()
    yh = pos.is_hot.to_numpy().astype(float)
    gh = pos.complex_id.to_numpy()
    Zh = np.column_stack([LD.zs(pos.burial), LD.zs(pos.nbr), LD.zs(pos.drsasa)])
    cpi_c = _cpi_safe(yh, gh, Zh, LD.zs(pos.conf), rng)
    floor_h = _placebo_floor(yh, gh, Zh, pos, rng)
    if cpi_c:
        blind = not (np.isfinite(floor_h) and cpi_c["lo"] > floor_h)
        print(f"  H2 CPI(confidence | geometry) = {cpi_c['cpi']:+.5f} "
              f"[{cpi_c['lo']:+.5f},{cpi_c['hi']:+.5f}]  placebo-floor={floor_h:+.5f}  "
              f"-> {'BLIND (<=floor, as pre-reg)' if blind else 'confidence ADDS (unexpected)'}  "
              f"[{len(pos)} positions]")
        rows.append({**base, "test": "CPI(confidence|geometry)", "stat": round(cpi_c["cpi"], 5),
                     "lo": round(cpi_c["lo"], 5), "hi": round(cpi_c["hi"], 5),
                     "p_gt0": round(cpi_c["p"], 3), "n_positions": len(pos),
                     "placebo_floor": round(floor_h, 5) if np.isfinite(floor_h) else "",
                     "confidence_blind": bool(blind)})
    else:
        print("  H2 CPI(confidence|geometry): INDETERMINATE")
        rows.append({**base, "test": "CPI(confidence|geometry)", "note": "indeterminate",
                     "n_positions": len(pos)})


def stage_analyse(a):
    rng = np.random.default_rng(SEED)
    frames, rows = [], []
    for model in ("mpnn", "esmif"):
        pq = f"{REPO}/results/atlas_pq_{model}.csv"
        if not os.path.exists(pq):
            print(f"[analyse] {pq} missing; skipping {model}")
            continue
        m = _build_mut_frame(model)
        if len(m) == 0:
            print(f"[analyse] {model}: no merged mutations")
            continue
        frames.append(m)
        print(f"\n########## MODEL = {model} ##########")
        _analyse_subset(m, "ATLAS FULL set", model, "full", rng, rows)
        _analyse_subset(m[m.is_overlap == 0], "ATLAS SKEMPI-NON-OVERLAP", model, "nonoverlap", rng, rows)
        # cross-check with the committed run_fixture on the full set
        try:
            print(f"\n---- run_fixture cross-check (committed machinery), {model}, full ----")
            LD.run_fixture(m.copy(), f"ATLAS-full-{model}", [], rng)
        except Exception as e:
            print(f"  [run_fixture cross-check skipped: {e}]")

    if frames:
        allm = pd.concat(frames, ignore_index=True)
        allm.to_csv(LEVERAGE_CSV, index=False)
        print(f"\n[analyse] wrote {LEVERAGE_CSV}  ({len(allm)} mutation rows, "
              f"{allm[allm.model=='mpnn'].pdb.nunique() if (allm.model=='mpnn').any() else 0} structures)")
    if rows:
        srows = pd.DataFrame(rows)
        srows["seed"] = SEED
        srows.to_csv(SUMMARY_CSV, index=False)
        print(f"[analyse] wrote {SUMMARY_CSV}  ({len(srows)} stat rows)")

    # overlap headline (positive control #2, the critical one)
    if frames:
        m0 = frames[0]
        ov = int((m0.is_overlap == 1).sum())
        tot = len(m0)
        print(f"\n[analyse] SKEMPI-overlap (structure-level): {ov}/{tot} interface mutations "
              f"({100*ov/max(tot,1):.0f}%) are on SKEMPI structures; "
              f"non-overlap = {tot-ov} over {m0[m0.is_overlap==0].pdb.nunique()} structures.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build-fixture", "score", "analyse"])
    ap.add_argument("--model", default="mpnn", choices=["mpnn", "esmif"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "build-fixture":
        stage_build_fixture(a)
    elif a.stage == "score":
        stage_score(a)
    elif a.stage == "analyse":
        stage_analyse(a)


if __name__ == "__main__":
    main()
