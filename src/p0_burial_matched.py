"""Phase 0 - the burial-matched control, plus the frustration-vs-dynamics discrimination.

Tests BRIEF.md F0 and F1. Analysis choices are fixed in results/PREREG.md and are not
altered here.

Stage 1 (slow, cached): per-residue structural features + ProteinMPNN teacher-forced
conditional log-probabilities over 8 decoding orders, for every SKEMPI complex.
Stage 2 (fast): labels, matched pairs, complex-level bootstrap, F1, causal discrimination.

Usage:
  python3 src/p0_burial_matched.py --out results/p0 --data-dir ~/ftax/data
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

SEED_BOOT = 20260803
N_BOOT = 10000
N_ORDERS = 8
HOT_LOOSE, HOT_STRICT, NULL_ABS = 1.0, 2.0, 0.25
INTERFACE_DRSASA = 0.05


# =============================================================== stage 1

def atom_sasa(pdb_path, pdb_id, keep_chains):
    """Per-atom SASA, keyed (chain, resnum, icode, atomname)."""
    import freesasa
    from Bio.PDB import PDBParser
    freesasa.setVerbosity(freesasa.silent)
    model = next(iter(PDBParser(QUIET=True).get_structure(pdb_id, pdb_path)))
    st, keys = freesasa.Structure(), []
    for ch in keep_chains:
        if ch not in model:
            continue
        for res in model[ch]:
            name = res.get_resname().strip().upper()
            if name not in fc.THREE2ONE:
                continue
            _, rn, ic = res.id
            for atom in res:
                if atom.element == "H":
                    continue
                x, y, z = (float(v) for v in atom.get_coord())
                try:
                    st.addAtom(atom.get_name().ljust(4)[:4], name, str(rn), ch, x, y, z)
                except Exception:
                    continue
                keys.append((ch, int(rn), ic.strip(), atom.get_name().strip()))
    if st.nAtoms() == 0:
        return {}
    r = freesasa.calc(st)
    return {keys[i]: r.atomArea(i) for i in range(st.nAtoms())}


def frustration_features(pdb_path, pdb_id, chains, asa_bound):
    """Structure-derived frustration proxies, independent of ProteinMPNN.

    - buried_polar_frac : fraction of side-chain polar (N/O) atoms that are buried
    - n_unsat_bur_polar : buried side-chain polar atoms with no N/O partner within 3.5 A
    - chi1_strain       : |deviation| of chi1 from the nearest canonical rotamer well
    """
    from Bio.PDB import PDBParser
    model = next(iter(PDBParser(QUIET=True).get_structure(pdb_id, pdb_path)))
    CHI1 = {"ARG": "CG", "ASN": "CG", "ASP": "CG", "CYS": "SG", "GLN": "CG",
            "GLU": "CG", "HIS": "CG", "ILE": "CG1", "LEU": "CG", "LYS": "CG",
            "MET": "CG", "PHE": "CG", "PRO": "CG", "SER": "OG", "THR": "OG1",
            "TRP": "CG", "TYR": "CG", "VAL": "CG1"}
    bb = {"N", "CA", "C", "O", "OXT"}

    polar_xyz = []
    for ch in chains:
        if ch not in model:
            continue
        for res in model[ch]:
            if res.get_resname().strip().upper() not in fc.THREE2ONE:
                continue
            for a in res:
                if a.element in ("N", "O"):
                    polar_xyz.append(a.get_coord())
    polar_xyz = np.array(polar_xyz) if polar_xyz else np.zeros((0, 3))

    out = {}
    for ch in chains:
        if ch not in model:
            continue
        for res in model[ch]:
            name = res.get_resname().strip().upper()
            if name not in fc.THREE2ONE:
                continue
            _, rn, ic = res.id
            key = (ch, int(rn), ic.strip())

            sc_polar = [a for a in res
                        if a.element in ("N", "O") and a.get_name().strip() not in bb]
            nb, nunsat = 0, 0
            for a in sc_polar:
                s = asa_bound.get((ch, int(rn), ic.strip(), a.get_name().strip()), np.nan)
                if not np.isnan(s) and s < 1.0:
                    nb += 1
                    if len(polar_xyz):
                        d = np.linalg.norm(polar_xyz - a.get_coord(), axis=1)
                        if ((d > 0.1) & (d < 3.5)).sum() == 0:
                            nunsat += 1
            bpf = nb / len(sc_polar) if sc_polar else 0.0

            chi1s = np.nan
            tgt = CHI1.get(name)
            if tgt is not None and all(x in res for x in ("N", "CA", "CB")) and tgt in res:
                chi1 = _dihedral(res["N"].get_coord(), res["CA"].get_coord(),
                                 res["CB"].get_coord(), res[tgt].get_coord())
                chi1s = min(abs((chi1 - w + 180) % 360 - 180) for w in (-60.0, 60.0, 180.0))

            out[key] = dict(buried_polar_frac=bpf, n_sc_polar=len(sc_polar),
                            n_unsat_bur_polar=nunsat, chi1_strain=chi1s)
    return out


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def gnm_fluctuations(cx, cutoff=7.3):
    """Gaussian Network Model mean-square fluctuation per residue (predicted flexibility)."""
    d = np.linalg.norm(cx.CA[:, None, :] - cx.CA[None, :, :], axis=-1)
    A = ((d < cutoff) & (d > 0)).astype(float)
    K = np.diag(A.sum(1)) - A
    try:
        w, V = np.linalg.eigh(K)
        keep = w > 1e-8
        if keep.sum() == 0:
            return np.full(cx.n, np.nan)
        return (V[:, keep] ** 2 / w[keep]).sum(axis=1)
    except np.linalg.LinAlgError:
        return np.full(cx.n, np.nan)


def build_positions(data_dir, weights, out_csv, limit=None, verbose=True,
                    only_complexes=None):
    """One row per residue of every SKEMPI complex, with features and MPNN scores.

    Rows are STREAMED to `out_csv` one complex at a time. Accumulating them in memory
    costs >1 GB of Python dicts on this 7.5 GB machine and gets the process OOM-killed
    with nothing on disk; streaming also means a crash keeps the completed complexes.
    """
    import csv
    import gc
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    skempi = fc.parse_skempi(os.path.join(data_dir, "skempi_v2.csv"))
    model, noise = fc.load_mpnn(weights)
    if verbose:
        print(f"[stage1] checkpoint {os.path.basename(weights)} (train noise {noise})")

    groups = (skempi[["pdb", "group1", "group2"]].drop_duplicates()
              .sort_values(["pdb", "group1", "group2"]).reset_index(drop=True))
    if only_complexes:
        want = set(only_complexes)
        keep = groups.apply(lambda r: f"{r['pdb']}_{r['group1']}_{r['group2']}" in want, axis=1)
        groups = groups[keep].reset_index(drop=True)
        print(f"[stage1] restricted to {len(groups)} of {len(keep)} complexes")
    if limit:
        groups = groups.head(limit)

    skipped, t0, n_rows = [], time.time(), 0
    writer, fh = None, open(out_csv, "w", newline="")
    for gi, g in groups.iterrows():
        pdb, g1, g2 = g["pdb"], g["group1"], g["group2"]
        path = os.path.join(data_dir, "PDBs", f"{pdb}.pdb")
        if not os.path.exists(path):
            skipped.append((pdb, g1, g2, "no pdb file")); continue
        try:
            cx = fc.load_complex(path, pdb, g1, g2)
            if cx is None:
                skipped.append((pdb, g1, g2, "chain group empty")); continue
            if cx.n > 3000:
                skipped.append((pdb, g1, g2, f"too large ({cx.n})")); continue

            all_ch = g1 + g2
            asa_b_atom = atom_sasa(path, pdb, all_ch)
            asa_b = {}
            for (c, rn, ic, _an), v in asa_b_atom.items():
                asa_b[(c, rn, ic)] = asa_b.get((c, rn, ic), 0.0) + v
            asa_f1 = fc.residue_sasa(path, pdb, g1)
            asa_f2 = fc.residue_sasa(path, pdb, g2)

            ss = fc.kabsch_sander_ss(cx)
            nbr = fc.neighbour_counts(cx)
            gnm = gnm_fluctuations(cx)
            frus = frustration_features(path, pdb, all_ch, asa_b_atom)

            lp = fc.mpnn_conditional_logprobs(model, cx, seeds=range(N_ORDERS))
            lp_mean = lp.mean(axis=0)                       # avg log-lik over orders
            lp_mix = fc.order_mixture_logprobs(lp)          # normalised mixture
            lp_unc = fc.mpnn_unconditional_logprobs(model, cx)
            nat = np.array([fc.MPNN_ALPHABET.index(a) for a in cx.seq])

            # per-order native log-prob, so the estimate's order-spread can be measured
            nat_per_order = lp[:, np.arange(cx.n), nat]     # [orders, L]

            bz = (cx.bfac - np.nanmean(cx.bfac)) / (np.nanstd(cx.bfac) + 1e-9)

            for i in range(cx.n):
                key = (cx.chains[i], int(cx.resnums[i]), cx.icodes[i])
                aa = cx.seq[i]
                sb, sf = asa_b.get(key, np.nan), (asa_f1 if cx.group[i] == 1 else asa_f2).get(key, np.nan)
                rb, rf = fc.relative_sasa(sb, aa), fc.relative_sasa(sf, aa)
                f = frus.get(key, {})
                r = dict(
                    pdb=pdb, group1=g1, group2=g2, complex_id=f"{pdb}_{g1}_{g2}",
                    chain=key[0], resnum=key[1], icode=key[2], aa=aa, idx=i,
                    group=int(cx.group[i]), ss=ss[i], nbr=int(nbr[i]),
                    rsasa_complex=rb, rsasa_free=rf, drsasa=(rf - rb),
                    bfac=cx.bfac[i], bfac_z=bz[i], gnm_msf=gnm[i],
                    buried_polar_frac=f.get("buried_polar_frac", np.nan),
                    n_sc_polar=f.get("n_sc_polar", np.nan),
                    n_unsat_bur_polar=f.get("n_unsat_bur_polar", np.nan),
                    chi1_strain=f.get("chi1_strain", np.nan),
                    logp_native=lp_mean[i, nat[i]],
                    logp_native_sd_order=float(nat_per_order[:, i].std(ddof=1)),
                    logp_native_unc=lp_unc[i, nat[i]],
                    logp_mode_mix=float(lp_mix[i, :20].max()),
                    logp_native_mix=float(lp_mix[i, nat[i]]),
                    mode_aa=fc.MPNN_ALPHABET[int(lp_mix[i, :20].argmax())],
                    hydro=fc.KD_HYDRO.get(aa, np.nan),
                )
                for o in range(N_ORDERS):
                    r[f"logp_native_o{o}"] = float(nat_per_order[o, i])
                for a_i, a in enumerate(fc.MPNN_ALPHABET[:20]):
                    r[f"lp_{a}"] = float(lp_mean[i, a_i])
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                    writer.writeheader()
                writer.writerow(r)
                n_rows += 1
            fh.flush()
            del lp, lp_mean, lp_mix, lp_unc, nat_per_order, asa_b_atom
            gc.collect()
        except Exception as e:
            skipped.append((pdb, g1, g2, f"error: {type(e).__name__}: {e}"))
            continue

        if verbose and (gi + 1) % 10 == 0:
            el = time.time() - t0
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"[stage1] {gi+1}/{len(groups)} complexes, {n_rows} residues, "
                  f"{el:.0f}s ({el/(gi+1):.2f}s/complex), peak RSS {rss:.2f} GB",
                  flush=True)

    fh.close()
    return n_rows, pd.DataFrame(skipped, columns=["pdb", "g1", "g2", "reason"])


# =============================================================== stage 2

def position_labels(skempi):
    """Per (complex, chain, resnum, icode): alanine-scan ddG summary and label."""
    ala = skempi[skempi["n_mut"] == 1].copy()
    recs = []
    for _, r in ala.iterrows():
        m = fc.parse_mutation(r["muts"][0])
        if m is None or m["mut"] != "A" or m["wt"] == "A":
            continue
        recs.append(dict(complex_id=f"{r['pdb']}_{r['group1']}_{r['group2']}",
                         chain=m["chain"], resnum=m["resnum"], icode=m["icode"],
                         wt=m["wt"], ddG=r["ddG"], T_K=r["T_K"],
                         method=r["Method"], loc=r["iMutation_Location(s)"]))
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    g = (d.groupby(["complex_id", "chain", "resnum", "icode", "wt"])
         .agg(ddG=("ddG", "median"), ddG_min=("ddG", "min"), ddG_max=("ddG", "max"),
              n_meas=("ddG", "size"), T_K=("T_K", "median")).reset_index())

    g["label"] = "other"
    g.loc[g["ddG"] > HOT_LOOSE, "label"] = "hot_loose"
    g.loc[g["ddG"] > HOT_STRICT, "label"] = "hot_strict"
    g.loc[g["ddG"].abs() < NULL_ABS, "label"] = "null"
    # PREREG 2: drop positions whose repeat measurements straddle hotspot and null bands
    straddle = (g["ddG_max"] > HOT_LOOSE) & (g["ddG_min"].abs() < NULL_ABS)
    g["straddle"] = straddle
    return g[~straddle].copy()


def match_pairs(pos, hot_label="hot_loose", control="null", rs_tol=0.05, nbr_tol=1,
                strict=False, same_aa=False, hydro_tol=None):
    """Optimal 1:1 within-complex matching on rSASA_complex, SS class, neighbour count.

    `same_aa` / `hydro_tol` add the BRIEF 5.2 control for native amino-acid identity.
    """
    pairs = []
    for cid, sub in pos.groupby("complex_id"):
        if strict:
            H = sub[sub["label"] == "hot_strict"]
        else:
            H = sub[sub["label"].isin(["hot_loose", "hot_strict"])]
        if control == "null":
            C = sub[sub["label"] == "null"]
        elif control == "measured_nonhot":
            C = sub[sub["label"].isin(["null", "other"]) & sub["has_meas"]]
        else:
            C = sub[~sub["label"].isin(["hot_loose", "hot_strict"])]
        H = H[H["is_interface"]]
        C = C[C["is_interface"]]
        if len(H) == 0 or len(C) == 0:
            continue

        cost = np.full((len(H), len(C)), 1e6)
        hv, cv = H.reset_index(drop=True), C.reset_index(drop=True)
        for a in range(len(hv)):
            for b in range(len(cv)):
                drs = abs(hv.loc[a, "rsasa_complex"] - cv.loc[b, "rsasa_complex"])
                if not (drs <= rs_tol and hv.loc[a, "ss"] == cv.loc[b, "ss"]
                        and abs(hv.loc[a, "nbr"] - cv.loc[b, "nbr"]) <= nbr_tol):
                    continue
                if same_aa and hv.loc[a, "aa"] != cv.loc[b, "aa"]:
                    continue
                if hydro_tol is not None:
                    dh = abs(hv.loc[a, "hydro"] - cv.loc[b, "hydro"])
                    if not (dh <= hydro_tol):
                        continue
                cost[a, b] = drs
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] >= 1e6:
                continue
            h, c = hv.loc[a], cv.loc[b]
            pairs.append(dict(
                complex_id=cid, pdb=h["pdb"],
                hot_chain=h["chain"], hot_resnum=h["resnum"], hot_aa=h["aa"],
                ctl_chain=c["chain"], ctl_resnum=c["resnum"], ctl_aa=c["aa"],
                hot_ddG=h["ddG"], ctl_ddG=c["ddG"], ss=h["ss"],
                hot_rsasa=h["rsasa_complex"], ctl_rsasa=c["rsasa_complex"],
                d_rsasa=h["rsasa_complex"] - c["rsasa_complex"],
                hot_nbr=h["nbr"], ctl_nbr=c["nbr"], d_nbr=h["nbr"] - c["nbr"],
                d_logp=h["logp_native"] - c["logp_native"],
                d_logp_unc=h["logp_native_unc"] - c["logp_native_unc"],
                d_bfac_z=h["bfac_z"] - c["bfac_z"],
                d_gnm=h["gnm_msf"] - c["gnm_msf"],
                d_buried_polar=h["buried_polar_frac"] - c["buried_polar_frac"],
                d_unsat_polar=h["n_unsat_bur_polar"] - c["n_unsat_bur_polar"],
                d_chi1_strain=h["chi1_strain"] - c["chi1_strain"],
                d_hydro=h["hydro"] - c["hydro"], d_T=h.get("T_K", np.nan),
                **{f"d_logp_o{o}": h[f"logp_native_o{o}"] - c[f"logp_native_o{o}"]
                   for o in range(N_ORDERS)},
            ))
    return pd.DataFrame(pairs)


def complex_bootstrap(pairs, col="d_logp", n_boot=N_BOOT, seed=SEED_BOOT):
    """Bootstrap over COMPLEXES (the independent unit), not positions."""
    if pairs.empty:
        return dict(n_pairs=0, n_complexes=0, mean=np.nan, lo=np.nan, hi=np.nan)
    rng = np.random.default_rng(seed)
    cids = pairs["complex_id"].unique()
    by = {c: pairs.loc[pairs["complex_id"] == c, col].values for c in cids}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        v = np.concatenate([by[cids[i]] for i in pick])
        means[b] = np.nanmean(v)
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return dict(n_pairs=len(pairs), n_complexes=len(cids),
                mean=float(np.nanmean(pairs[col])), lo=float(lo), hi=float(hi),
                boot_sd=float(np.nanstd(means)), n_boot=n_boot, seed=seed)


def partial_spearman(x, y, z):
    """Spearman correlation of x,y after linearly removing rank(z) from both."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 20:
        return np.nan, np.nan, int(m.sum())
    rx, ry, rz = (stats.rankdata(v[m]) for v in (x, y, z))
    rz = np.column_stack([np.ones_like(rz), rz])
    res = lambda r: r - rz @ np.linalg.lstsq(rz, r, rcond=None)[0]
    rho, p = stats.spearmanr(res(rx), res(ry))
    return float(rho), float(p), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p0")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--mpnn-weights",
                    default=os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-complexes", default=None,
                    help="file with one complex_id per line; restricts stage 1")
    ap.add_argument("--cache", default=None, help="reuse an existing positions CSV")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    cmd = "python3 " + " ".join(sys.argv)
    pos_csv = f"{a.out}_positions.csv"

    if a.cache and os.path.exists(a.cache):
        print(f"[stage1] reusing {a.cache}")
        pos = pd.read_csv(a.cache)
        # blank insertion codes round-trip through CSV as NaN; without this the
        # label merge below silently matches nothing.
        pos["icode"] = pos["icode"].fillna("").astype(str)
        pos = pos.drop(columns=[c for c in ("label", "ddG", "n_meas", "T_K", "has_meas",
                                            "is_interface", "ddG_min", "ddG_max",
                                            "straddle") if c in pos.columns])
    else:
        only = None
        if a.only_complexes:
            only = [l.strip() for l in open(a.only_complexes) if l.strip()]
        n_rows, skipped = build_positions(a.data_dir, a.mpnn_weights, pos_csv,
                                         a.limit, only_complexes=only)
        skipped.to_csv(f"{a.out}_skipped.csv", index=False)
        pos = pd.read_csv(pos_csv)
        pos["icode"] = pos["icode"].fillna("").astype(str)
        print(f"[stage1] wrote {pos_csv}: {n_rows} residues, "
              f"{pos['complex_id'].nunique()} complexes; {len(skipped)} skipped")

    # ---- labels
    skempi = fc.parse_skempi(os.path.join(a.data_dir, "skempi_v2.csv"))
    lab = position_labels(skempi)
    pos["is_interface"] = pos["drsasa"] > INTERFACE_DRSASA
    pos = pos.merge(lab.drop(columns=["wt"]), how="left",
                    on=["complex_id", "chain", "resnum", "icode"])
    pos["label"] = pos["label"].fillna("unmeasured")
    pos["has_meas"] = pos["n_meas"].notna()
    pos.to_csv(pos_csv, index=False)

    # wild-type identity check: SKEMPI wt must equal the residue in the structure
    chk = pos.merge(lab[["complex_id", "chain", "resnum", "icode", "wt"]], how="inner",
                    on=["complex_id", "chain", "resnum", "icode"])
    mism = (chk["aa"] != chk["wt"]).sum()
    print(f"[check] SKEMPI wt vs structure: {len(chk)} mapped, {mism} mismatches "
          f"({mism/max(len(chk),1):.2%})")

    summary = []

    def record(tag, res, extra=None):
        row = dict(analysis=tag, **res)
        if extra:
            row.update(extra)
        summary.append(row)
        print(f"  {tag:38s} n_pairs={res['n_pairs']:4d} n_cx={res['n_complexes']:3d} "
              f"mean={res['mean']:+.4f} CI=[{res['lo']:+.4f},{res['hi']:+.4f}]")

    # ---- F0: the burial-matched control
    print("\n=== F0: burial-matched paired log-probability gap (hotspot - control) ===")
    print("PRIMARY (control = measured nulls, |ddG|<0.25):")
    hdr = pos[pos["T_K"].between(293, 303) | pos["T_K"].isna()]
    variants = {
        "PRIMARY_loose_null": dict(df=pos, control="null", strict=False),
        "PRIMARY_loose_null_T293_303": dict(df=hdr, control="null", strict=False),
        "strict_hot2_null": dict(df=pos, control="null", strict=True),
        "SECONDARY_A_measured_nonhot": dict(df=pos, control="measured_nonhot", strict=False),
        "SECONDARY_B_any_interface": dict(df=pos, control="any", strict=False),
        # BRIEF 5.2 - native amino-acid identity as a confound
        "AAMATCHED_any_interface": dict(df=pos, control="any", strict=False, same_aa=True),
        "HYDROMATCHED_any_interface": dict(df=pos, control="any", strict=False, hydro_tol=1.0),
        # matching tolerance sensitivity (reported, never substituted for the primary)
        "SENS_nbr_tol2_any_interface": dict(df=pos, control="any", strict=False, nbr_tol=2),
    }
    all_pairs = {}
    for tag, v in variants.items():
        pr = match_pairs(v["df"], control=v["control"], strict=v["strict"],
                         same_aa=v.get("same_aa", False), hydro_tol=v.get("hydro_tol"),
                         nbr_tol=v.get("nbr_tol", 1))
        all_pairs[tag] = pr
        if pr.empty:
            print(f"  {tag:38s} NO PAIRS")
            summary.append(dict(analysis=tag, n_pairs=0, n_complexes=0,
                                mean=np.nan, lo=np.nan, hi=np.nan))
            continue
        res = complex_bootstrap(pr)
        # decoding-order spread OF THE ESTIMATE (BRIEF 5.6)
        per_order = [pr[f"d_logp_o{o}"].mean() for o in range(N_ORDERS)]
        record(tag, res, dict(order_gap_sd=float(np.std(per_order, ddof=1)),
                              order_gap_min=float(np.min(per_order)),
                              order_gap_max=float(np.max(per_order)),
                              mean_d_rsasa=float(pr["d_rsasa"].mean()),
                              mean_d_nbr=float(pr["d_nbr"].mean()),
                              unconditional_mean=float(pr["d_logp_unc"].mean())))
        pr.to_csv(f"{a.out}_pairs_{tag}.csv", index=False)

    # ---- the uncontrolled comparison, for contrast with ProBID-Net (0.334 vs 0.472)
    print("\n=== UNCONTROLLED contrast (no matching) - the ProBID-Net-style comparison ===")
    iface = pos[pos["is_interface"]]
    contrasts = {
        # ProBID-Net's own definition: hotspot = interface residue with Ala ddG > 2
        "PROBIDNET_strict_vs_all_other_iface":
            (iface[iface["label"] == "hot_strict"],
             iface[iface["label"] != "hot_strict"]),
        "strict_vs_measured_nonhot_iface":
            (iface[iface["label"] == "hot_strict"],
             iface[iface["has_meas"] & ~iface["label"].isin(["hot_loose", "hot_strict"])]),
        "loose_vs_null_iface":
            (iface[iface["label"].isin(["hot_loose", "hot_strict"])],
             iface[iface["label"] == "null"]),
    }
    for tag, (H, C) in contrasts.items():
        if not len(H) or not len(C):
            continue
        rec_h = float((H["mode_aa"] == H["aa"]).mean())
        rec_c = float((C["mode_aa"] == C["aa"]).mean())
        print(f"  {tag}")
        print(f"    recovery  hot={rec_h:.3f} (n={len(H)})   nonhot={rec_c:.3f} (n={len(C)})"
              f"   gap={rec_h-rec_c:+.3f}")
        print(f"    logp      hot={H['logp_native'].mean():+.4f}  "
              f"nonhot={C['logp_native'].mean():+.4f}  "
              f"diff={H['logp_native'].mean()-C['logp_native'].mean():+.4f}")
        print(f"    rSASA     hot={H['rsasa_complex'].mean():.3f}  "
              f"nonhot={C['rsasa_complex'].mean():.3f}   <- the confound")
        summary.append(dict(analysis=f"UNCONTROLLED_{tag}", n_pairs=len(H) + len(C),
                            n_complexes=iface["complex_id"].nunique(),
                            mean=float(H["logp_native"].mean() - C["logp_native"].mean()),
                            lo=np.nan, hi=np.nan, recovery_hot=rec_h, recovery_null=rec_c,
                            n_hot=len(H), n_nonhot=len(C),
                            rsasa_hot=float(H["rsasa_complex"].mean()),
                            rsasa_null=float(C["rsasa_complex"].mean())))

    # ---- F1: burial-controlled partial Spearman, log-odds vs ddG
    print("\n=== F1: burial-controlled partial Spearman (log-odds vs ddG_bind) ===")
    single = skempi[skempi["n_mut"] == 1].copy()
    recs = []
    key = pos.set_index(["complex_id", "chain", "resnum", "icode"])
    for _, r in single.iterrows():
        m = fc.parse_mutation(r["muts"][0])
        if m is None:
            continue
        k = (f"{r['pdb']}_{r['group1']}_{r['group2']}", m["chain"], m["resnum"], m["icode"])
        if k not in key.index:
            continue
        p = key.loc[k]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[0]
        if p["aa"] != m["wt"]:
            continue
        recs.append(dict(complex_id=k[0], ddG=r["ddG"], T_K=r["T_K"],
                         logodds=p[f"lp_{m['mut']}"] - p[f"lp_{m['wt']}"],
                         rsasa=p["rsasa_complex"], nbr=p["nbr"],
                         is_interface=p["is_interface"], wt=m["wt"], mut=m["mut"]))
    f1 = pd.DataFrame(recs)
    f1.to_csv(f"{a.out}_f1_logodds.csv", index=False)
    for tag, sub in [("all_single", f1), ("interface_only", f1[f1["is_interface"]]),
                     ("interface_T293_303",
                      f1[f1["is_interface"] & f1["T_K"].between(293, 303)])]:
        if len(sub) < 20:
            continue
        raw, praw = stats.spearmanr(sub["logodds"], sub["ddG"])
        rho, p_, n = partial_spearman(sub["logodds"].values, sub["ddG"].values,
                                      sub["rsasa"].values)
        print(f"  {tag:22s} n={n:5d}  raw rho={raw:+.3f}  "
              f"burial-partial rho={rho:+.3f} (p={p_:.2e})  |rho|>=0.35? "
              f"{'YES -> F1 FIRES' if abs(rho) >= 0.35 else 'no'}")
        summary.append(dict(analysis=f"F1_{tag}", n_pairs=n, n_complexes=sub['complex_id'].nunique(),
                            mean=rho, lo=np.nan, hi=np.nan, raw_spearman=float(raw),
                            partial_p=p_))

    # ---- causal discrimination: frustration vs dynamics (BRIEF 4, exploratory:
    #      no pre-registered threshold attaches to this section)
    print("\n=== Frustration vs dynamics ===")
    PROXIES = [("frustration", "d_buried_polar"), ("frustration", "d_unsat_polar"),
               ("frustration", "d_chi1_strain"),
               ("dynamics", "d_bfac_z"), ("dynamics", "d_gnm")]
    pr = all_pairs.get("SECONDARY_B_any_interface")
    if pr is not None and len(pr) > 30:
        print("  (a) pair level - does the post-matching gap track each proxy?")
        for kind, col in PROXIES:
            m = np.isfinite(pr[col]) & np.isfinite(pr["d_logp"])
            if m.sum() < 20:
                continue
            rho, p_ = stats.spearmanr(pr.loc[m, col], pr.loc[m, "d_logp"])
            print(f"    d_logp vs {kind:11s} {col:16s} rho={rho:+.3f} p={p_:.1e} n={m.sum()}")
            summary.append(dict(analysis=f"CAUSAL_pair_{col}", n_pairs=int(m.sum()),
                                n_complexes=pr.loc[m, "complex_id"].nunique(),
                                mean=float(rho), lo=np.nan, hi=np.nan, partial_p=float(p_)))
        # joint standardised OLS: which family survives with the other in the model?
        cols = [c for _, c in PROXIES]
        sub = pr[np.isfinite(pr[cols]).all(axis=1) & np.isfinite(pr["d_logp"])]
        if len(sub) > 50:
            Z = sub[cols].values
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
            Z = np.column_stack([np.ones(len(Z)), Z])
            y = sub["d_logp"].values
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            resid = y - Z @ beta
            s2 = resid @ resid / (len(y) - Z.shape[1])
            se = np.sqrt(np.diag(s2 * np.linalg.pinv(Z.T @ Z)))
            print(f"    joint standardised OLS on d_logp (n={len(sub)}):")
            for j, c in enumerate(cols):
                t = beta[j + 1] / se[j + 1]
                print(f"      {c:18s} beta={beta[j+1]:+.4f} +/- {se[j+1]:.4f}  t={t:+.2f}")
                summary.append(dict(analysis=f"CAUSAL_jointOLS_{c}", n_pairs=len(sub),
                                    n_complexes=sub["complex_id"].nunique(),
                                    mean=float(beta[j + 1]), lo=float(beta[j + 1] - 1.96 * se[j + 1]),
                                    hi=float(beta[j + 1] + 1.96 * se[j + 1])))

    # (b) position level - all interface positions, burial+context residualised out.
    #     Far more power than the matched pairs; same question.
    print("  (b) position level - burial-residualised log-prob vs each proxy")
    ifc = pos[pos["is_interface"]].copy()
    base = ["rsasa_complex", "nbr"]
    ok = np.isfinite(ifc[base + ["logp_native"]]).all(axis=1)
    ifc = ifc[ok]
    if len(ifc) > 100:
        D = pd.get_dummies(ifc["ss"], prefix="ss", drop_first=True).astype(float)
        A = pd.get_dummies(ifc["aa"], prefix="aa", drop_first=True).astype(float)
        Xd = np.column_stack([np.ones(len(ifc)), ifc[base].values, D.values, A.values])
        yv = ifc["logp_native"].values
        ifc["logp_resid"] = yv - Xd @ np.linalg.lstsq(Xd, yv, rcond=None)[0]
        for kind, col in PROXIES:
            pcol = {"d_buried_polar": "buried_polar_frac",
                    "d_unsat_polar": "n_unsat_bur_polar",
                    "d_chi1_strain": "chi1_strain",
                    "d_bfac_z": "bfac_z",
                    "d_gnm": "gnm_msf"}[col]
            if pcol not in ifc.columns:
                print(f"    WARNING: proxy column {pcol} missing"); continue
            m = np.isfinite(ifc[pcol]) & np.isfinite(ifc["logp_resid"])
            if m.sum() < 50:
                continue
            rho, p_ = stats.spearmanr(ifc.loc[m, pcol], ifc.loc[m, "logp_resid"])
            print(f"    logp_resid vs {kind:11s} {pcol:16s} rho={rho:+.3f} "
                  f"p={p_:.1e} n={m.sum()}")
            summary.append(dict(analysis=f"CAUSAL_pos_{pcol}", n_pairs=int(m.sum()),
                                n_complexes=ifc.loc[m, "complex_id"].nunique(),
                                mean=float(rho), lo=np.nan, hi=np.nan, partial_p=float(p_)))
        ifc.to_csv(f"{a.out}_interface_resid.csv", index=False)

    sm = pd.DataFrame(summary)
    sm["command"] = cmd
    sm["mpnn_ckpt"] = os.path.basename(a.mpnn_weights)
    sm["n_orders"] = N_ORDERS
    sm.to_csv(f"{a.out}_summary.csv", index=False)

    prim = sm[sm["analysis"] == "PRIMARY_loose_null"]
    if len(prim) and np.isfinite(prim.iloc[0]["lo"]):
        r = prim.iloc[0]
        fired = (r["lo"] <= 0 <= r["hi"])
        print(f"\nSUMMARY | F0 primary gap {r['mean']:+.4f} nats "
              f"95%CI [{r['lo']:+.4f}, {r['hi']:+.4f}] over {int(r['n_complexes'])} complexes, "
              f"{int(r['n_pairs'])} pairs | F0 {'FIRES (CI contains 0)' if fired else 'does not fire'}")
    print(f"[done] wrote {a.out}_summary.csv")


if __name__ == "__main__":
    main()
