#!/usr/bin/env python3
"""T1 (venue decider) — does p(aa|complex) predict interface binding BEYOND an ALL-ATOM occlusion baseline?

Pre-registered in results/PREREG_bennett_hardening.md. The published Big-Idea-1 positive (P adds +0.025
over a *volume x contact* occlusion baseline, bennett_occlusion_energetics.csv) was flagged by a hostile
reviewer as a weak-baseline artifact: "repack the rotamers and it evaporates." This rebuilds the occlusion
baseline with a real ALL-ATOM, min-over-rotamer van-der-Waals clash of the substituted side chain against
the partner chain, and re-runs the identical CV-logistic ΔAUROC(P over geometry) test.

Side chains are built with rdkit (ETKDG conformers = rotamer samples), each rigid-superposed (Kabsch on
N,CA,C) onto the real backbone frame of the SSM position on the de-novo binder (chain A); clash = summed
vdW overlap vs all heavy atoms of the partner (chain B); MIN over conformers = best repack.

VALIDITY GATE (pre-registered; checked and printed BEFORE any ΔAUROC — T1 invalid if it fails):
  (1) the builder reconstructs NATIVE side chains present in the PDBs to median heavy-atom RMSD < 1.0 Å
      (element-matched, min over conformers) — i.e. the library spans the native rotamer, so a min-clash
      estimate is meaningful;
  (2) Gly clash == 0.
We ALSO report the fraction of substitutions with zero post-repack clash (occlusion prevalence) and the
est-vs-real native-clash Spearman as secondary diagnostics.

  python3 src/p_bennett_occlusion_allatom.py --out results/bennett_occlusion_allatom.csv
"""
import argparse, glob, os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc
import bennett_knows_where as bkw
import bennett_kl_detector as bkd

warnings.filterwarnings("ignore")
SEED = 20260803
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "F": 1.47, "CL": 1.75}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

SC = {"A": "C", "R": "CCCNC(N)=N", "N": "CC(N)=O", "D": "CC(=O)O", "C": "CS", "Q": "CCC(N)=O",
      "E": "CCC(=O)O", "H": "Cc1cnc[nH]1", "I": "C(C)CC", "L": "CC(C)C", "K": "CCCCN", "M": "CCSC",
      "F": "Cc1ccccc1", "P": None, "S": "CO", "T": "C(O)C", "V": "C(C)C", "W": "Cc1c[nH]c2ccccc12",
      "Y": "Cc1ccc(O)cc1"}


def aa_smiles(aa):
    if aa == "G":
        return "[N:1][CH2:2][C:3](=O)O"
    if aa == "P":
        return "O=[C:3](O)[C@@H:2]1CCC[N:1]1"
    return "[N:1][C@@H:2](%s)[C:3](=O)O" % SC[aa]


def kabsch(Pf, Pt):
    cf, ct = Pf.mean(0), Pt.mean(0)
    H = (Pf - cf).T @ (Pt - ct)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, ct - R @ cf


def build_library(K=40, seed=SEED):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    lib = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        m = Chem.AddHs(Chem.MolFromSmiles(aa_smiles(aa)))
        ps = AllChem.ETKDGv3(); ps.randomSeed = seed
        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=K, params=ps))
        if not cids:
            ps.useRandomCoords = True
            cids = list(AllChem.EmbedMultipleConfs(m, numConfs=K, params=ps))
        try:
            AllChem.MMFFOptimizeMoleculeConfs(m)
        except Exception:
            pass
        mp = {a.GetAtomMapNum(): a.GetIdx() for a in m.GetAtoms() if a.GetAtomMapNum() in (1, 2, 3)}
        Ni, CAi, Ci = mp[1], mp[2], mp[3]
        Oi = [nb.GetIdx() for nb in m.GetAtomWithIdx(Ci).GetNeighbors() if nb.GetSymbol() == "O"]
        bb = {Ni, CAi, Ci, *Oi}
        sci = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != "H" and a.GetIdx() not in bb]
        elems = [m.GetAtomWithIdx(i).GetSymbol().upper() for i in sci]
        radii = np.array([VDW.get(e, 1.70) for e in elems])
        ref, confs = None, []
        for cid in cids:
            cf = m.GetConformer(cid)
            ncac = np.array([list(cf.GetAtomPosition(i)) for i in (Ni, CAi, Ci)])
            if ref is None:
                ref = ncac
            R, t = kabsch(ncac, ref)
            if sci:
                scoord = np.array([list(cf.GetAtomPosition(i)) for i in sci])
                confs.append(scoord @ R.T + t)
        lib[aa] = dict(ref=ref if ref is not None else np.eye(3), confs=confs, radii=radii, elems=elems)
    return lib


def clash_of_cloud(cloud, radii, tree, pcoord, pradii, tol=0.4):
    if len(cloud) == 0:
        return 0.0
    tot = 0.0
    rmax = float(pradii.max()) if len(pradii) else 1.8
    for k in range(len(cloud)):
        near = tree.query_ball_point(cloud[k], radii[k] + rmax)
        if near:
            d = np.linalg.norm(pcoord[near] - cloud[k], axis=1)
            ov = (radii[k] + pradii[near] - tol) - d
            tot += ov[ov > 0].sum()
    return float(tot)


def _placed_clouds(aa, ncac_t, lib):
    L = lib[aa]
    if len(L["radii"]) == 0:
        return []
    R, t = kabsch(L["ref"], ncac_t)
    return [sc @ R.T + t for sc in L["confs"]]


def min_clash(aa, ncac_t, lib, tree, pcoord, pradii):
    L = lib[aa]
    if len(L["radii"]) == 0:
        return 0.0
    best = np.inf
    for cloud in _placed_clouds(aa, ncac_t, lib):
        best = min(best, clash_of_cloud(cloud, L["radii"], tree, pcoord, pradii))
        if best == 0.0:
            break
    return 0.0 if not np.isfinite(best) else best


def native_rmsd(aa, ncac_t, real_coords, real_elems, lib):
    """min over conformers of element-matched heavy-atom RMSD to the real side chain (pre-registered gate)."""
    L = lib[aa]
    if len(L["radii"]) == 0:
        return 0.0                                  # Gly: no side chain -> trivially reconstructed
    if len(real_coords) == 0:
        return np.nan
    belems = np.array(L["elems"]); relems = np.array(real_elems)
    best = np.inf
    for built in _placed_clouds(aa, ncac_t, lib):
        C = np.full((len(built), len(real_coords)), 1e9)
        for i in range(len(built)):
            same = np.where(relems == belems[i])[0]
            if len(same):
                C[i, same] = np.sum((built[i] - real_coords[same]) ** 2, axis=1)
        ri, ci = linear_sum_assignment(C)
        m = [C[i, j] for i, j in zip(ri, ci) if C[i, j] < 1e9]
        if m:
            best = min(best, float(np.sqrt(np.mean(m))))
    return best if np.isfinite(best) else np.nan


def parse_struct(pdb_path):
    from Bio.PDB import PDBParser
    s = PDBParser(QUIET=True).get_structure("x", pdb_path)[0]
    pc, pr = [], []
    for atom in (s["B"].get_atoms() if "B" in s else []):
        el = (atom.element.strip().upper() or atom.get_name()[0])
        if el == "H":
            continue
        pc.append(atom.get_coord()); pr.append(VDW.get(el, 1.70))
    chainA = {}
    if "A" in s:
        for res in s["A"]:
            d = {"bb": {}, "sc_xyz": [], "sc_r": [], "sc_e": []}
            for atom in res:
                el = (atom.element.strip().upper() or atom.get_name()[0])
                if el == "H":
                    continue
                nm = atom.get_name().strip()
                if nm in ("N", "CA", "C"):
                    d["bb"][nm] = atom.get_coord()
                elif nm not in BACKBONE_ATOMS:
                    d["sc_xyz"].append(atom.get_coord()); d["sc_r"].append(VDW.get(el, 1.70)); d["sc_e"].append(el)
            if {"N", "CA", "C"} <= set(d["bb"]):
                chainA[int(res.id[1])] = d
    return (np.array(pc, float), np.array(pr, float)), chainA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bennett_occlusion_allatom.csv")
    ap.add_argument("--pairs_out", default="results/bennett_occlusion_allatom_pairs.csv")
    ap.add_argument("--K", type=int, default=40)
    a = ap.parse_args()

    print(f"building rdkit rotamer library (K={a.K}) ...")
    lib = build_library(K=a.K)

    pdb_index = {os.path.basename(p)[:-4]: p for p in
                 glob.glob(f"{bkw.BEN}/design_models_ssm_natives/*/*.pdb")}
    model, _ = fc.load_mpnn(os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))

    recs, gate_rmsd, gate_estreal = [], [], []
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
                (pcoord, pradii), chainA = parse_struct(pdb)
            except Exception as e:
                print(f"  skip {parent[:22]}: {type(e).__name__} {e}"); continue
            if not pq or len(pcoord) == 0:
                continue
            tree = cKDTree(pcoord)
            for (par, pos), (subs, native) in lab.items():
                if par != parent or pos not in pq or pos not in bi or native is None or native not in bkw.IDX:
                    continue
                Pv, Qv, restype = pq[pos]
                if native != restype or pos not in chainA:
                    continue
                dsasa = bi[pos]["sasa_mono"] - bi[pos]["sasa_complex"]
                if dsasa <= 5:                       # interface layer only
                    continue
                res = chainA[pos]
                ncac = np.array([res["bb"]["N"], res["bb"]["CA"], res["bb"]["C"]], float)
                contact = int(len(tree.query_ball_point(res["bb"]["CA"], 8.0)))
                rc = np.array(res["sc_xyz"], float)
                gate_rmsd.append(native_rmsd(native, ncac, rc, res["sc_e"], lib))
                est_nat = min_clash(native, ncac, lib, tree, pcoord, pradii)
                real_nat = clash_of_cloud(rc, np.array(res["sc_r"]), tree, pcoord, pradii) if len(rc) else 0.0
                gate_estreal.append((est_nat, real_nat))
                for sub, binds in subs.items():
                    if sub not in bkw.IDX or sub == native:
                        continue
                    recs.append(dict(design=parent, resnum=int(pos), sub=sub, binds=int(binds),
                                     P=float(Pv[bkw.IDX[sub]]), Q=float(Qv[bkw.IDX[sub]]),
                                     dsasa=float(dsasa), sub_vol=bkw.VOL[sub], nat_vol=bkw.VOL[native],
                                     vol=-abs(bkw.VOL[sub] - bkw.VOL[native]),
                                     aa_clash=min_clash(sub, ncac, lib, tree, pcoord, pradii),
                                     contact=contact))
    d = pd.DataFrame(recs)
    d.to_csv(a.pairs_out, index=False)

    # ---- VALIDITY GATE (pre-registered: native reconstruction RMSD) ----
    gr = np.array([x for x in gate_rmsd if np.isfinite(x)])
    ge = np.array(gate_estreal)
    med_rmsd = float(np.median(gr)) if len(gr) else np.nan
    gly = min_clash("G", np.array([[0, 0, 0], [1.46, 0, 0], [2.0, 1.4, 0]], float),
                    lib, cKDTree(np.array([[1e3, 1e3, 1e3]])), np.array([[1e3, 1e3, 1e3]]), np.array([1.7]))
    frac_zero = float((d[d.dsasa > 5].aa_clash == 0).mean())
    rho_er = float(stats.spearmanr(ge[:, 0], ge[:, 1]).correlation) if len(ge) > 3 else np.nan
    gate_ok = (gly == 0.0) and np.isfinite(med_rmsd) and (med_rmsd < 1.0)
    print(f"\n=== VALIDITY GATE (pre-registered) ===")
    print(f"  native reconstruction RMSD (element-matched, min-over-rotamer): median = {med_rmsd:.3f} Å "
          f"(gate: <1.0)  n={len(gr)}  [p90={np.percentile(gr,90):.2f}]")
    print(f"  Gly clash = {gly:.3f} (expect 0)")
    print(f"  GATE {'PASS' if gate_ok else 'FAIL — T1 could-not-run (pre-registered)'}")
    print(f"  [secondary] post-repack occlusion prevalence: frac aa_clash==0 = {frac_zero:.3f}; "
          f"est-vs-real native-clash Spearman = {rho_er:+.3f}")

    di = d[d.dsasa > 5].reset_index(drop=True)
    y = di.binds.to_numpy(); g = di.design.to_numpy()
    Z = lambda c: ((di[c] - di[c].mean()) / (di[c].std() + 1e-9)).to_numpy()
    rows = [dict(metric="gate", feature="native_reconstruction_rmsd_median", value=round(med_rmsd, 4), n=len(gr)),
            dict(metric="gate", feature="gly_clash", value=round(gly, 4)),
            dict(metric="gate", feature="frac_zero_clash_postrepack", value=round(frac_zero, 4)),
            dict(metric="gate", feature="est_vs_real_native_clash_spearman", value=round(rho_er, 4)),
            dict(metric="gate", feature="PASS" if gate_ok else "FAIL", value=int(gate_ok))]
    for name, sc in [("P_complex", di.P.values), ("aa_clash", -di.aa_clash.values),
                     ("dSASA", -di.dsasa.values), ("vol_sim", di.vol.values),
                     ("old_vol_clash", -np.maximum(0.0, di.sub_vol - di.nat_vol).values)]:
        rows.append(dict(metric="auroc_standalone", feature=name, value=round(bkw.auc(sc, y), 4)))

    if gate_ok:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        di["clash_c"] = di.aa_clash * di.dsasa
        geo = np.column_stack([Z("aa_clash"), Z("clash_c"), Z("dsasa"), Z("vol"), Z("contact")])
        geoP = np.column_stack([geo, Z("P")])
        o_geo = np.zeros(len(y)); o_geoP = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(geo, y, g):
            o_geo[te] = LogisticRegression(max_iter=1000).fit(geo[tr], y[tr]).predict_proba(geo[te])[:, 1]
            o_geoP[te] = LogisticRegression(max_iter=1000).fit(geoP[tr], y[tr]).predict_proba(geoP[te])[:, 1]
        ag, agp = bkw.auc(o_geo, y), bkw.auc(o_geoP, y)
        ids = np.unique(g); pos = {u: np.where(g == u)[0] for u in ids}
        rng = np.random.default_rng(SEED); dd = []
        for _ in range(3000):
            idx = np.concatenate([pos[u] for u in rng.choice(ids, len(ids), True)]); yy = y[idx]
            if 0 < yy.sum() < len(yy):
                dd.append(bkw.auc(o_geoP[idx], yy) - bkw.auc(o_geo[idx], yy))
        dd = np.array(dd)
        lo, hi, p = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)), float(np.mean(dd > 0))
        survives = (lo > 0) and (agp - ag >= 0.010)
        verdict = "SURVIVES -> Spine B / ICLR" if survives else "COLLAPSES -> Spine A / TMLR"
        rows += [dict(metric="auroc_cv", feature="geometry_allatom(clash+clashxc+dSASA+vol+contact)", value=round(ag, 4)),
                 dict(metric="auroc_cv", feature="geometry_allatom+P", value=round(agp, 4)),
                 dict(metric="dAUROC_P_over_allatom_geometry", feature="P_adds_beyond_ALLATOM_occlusion",
                      value=round(agp - ag, 4), lo=round(lo, 4), hi=round(hi, 4), p_gt0=round(p, 3), verdict=verdict)]
        print(f"\ninterface: {len(di)} (pos,sub) pairs, {di.design.nunique()} designs, bind-rate {y.mean():.2f}")
        print(f"  AUROC standalone: P {bkw.auc(di.P.values,y):.3f}  aa_clash {bkw.auc(-di.aa_clash.values,y):.3f}  "
              f"dSASA {bkw.auc(-di.dsasa.values,y):.3f}")
        print(f"  geometry(all-atom) {ag:.3f}  |  geometry+P {agp:.3f}")
        print(f"  ΔAUROC(P over ALL-ATOM geometry) = {agp-ag:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}  -> {verdict}")
    else:
        print("\n  GATE FAILED — not computing ΔAUROC (pre-registered).")

    out = pd.DataFrame(rows); out["seed"] = SEED; out["K_conformers"] = a.K
    out["command"] = "python3 src/p_bennett_occlusion_allatom.py"
    out.to_csv(a.out, index=False)
    print(f"\nwrote {a.out} and {a.pairs_out}")


if __name__ == "__main__":
    main()
