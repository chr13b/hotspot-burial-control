#!/usr/bin/env python3
"""FoldX 5.1 physics ΔΔG_bind driver for both lanes (pre-reg: results/PREREG_foldx.md). Idempotent + shardable.

Reads results/foldx_worklist.csv (from foldx_prep.py). Two modes:
  --mode repair : RepairPDB each unique crystal once -> $SCRATCH/ftax/foldx/repaired/<pdb>_Repair.pdb (cache).
  --mode ddg    : per BuildModel SET, ΔΔG_bind = mean(IE_mut) - mean(IE_wt) over numberOfRuns, where IE =
                  AnalyseComplex "Interaction Energy" between the two chain groups g1,g2 (from complex_id).
                  Lane A: one set per single SKEMPI mutant. Lane B: one set per (complex, arm, k).

ΔΔG_bind sign: FoldX IE more negative = stronger binding, so ΔΔG_bind>0 = mutation WEAKENS binding (destabilising).
Verified against the smoke (1ACB E,I: mut IE -15.6855, wt IE -15.8433 -> ΔΔG_bind +0.158).

  python3 src/foldx_ddg.py --mode repair --shard 0 --nshards 30
  python3 src/foldx_ddg.py --mode ddg --lane B --shard 0 --nshards 60 --runs 5 --out results/_foldx_B_s0.csv
  python3 src/foldx_ddg.py --mode ddg --lane A --shard 0 --nshards 120 --runs 5 --out results/_foldx_A_s0.csv
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

FOLDX = os.environ.get("FOLDX_BIN", os.path.expandvars("$SCRATCH/ftax/foldx/v5_0/foldx_20261231"))
MOLEC = os.path.join(os.path.dirname(FOLDX), "molecules")
PDBS = os.environ.get("FOLDX_PDBS", os.path.expanduser("~/ftax/data/PDBs"))   # override for other fixtures (AB-Bind)
# repair cache is keyed by pdb-id only, so a distinct fixture that shares a pdb-id (e.g. AB-Bind vs SKEMPI, same id
# but a different / renumbered structure) MUST use a separate cache dir or it will silently reuse the wrong repair.
REPAIR = os.environ.get("FOLDX_REPAIR", os.path.expandvars("$SCRATCH/ftax/foldx/repaired"))
WORKBASE = os.path.expandvars("$SCRATCH/ftax/foldx/work")
WORKLIST = os.environ.get("FOLDX_WORKLIST", "results/foldx_worklist.csv")      # override to point at AB-Bind worklist


FOLDX_TIMEOUT = int(os.environ.get("FOLDX_TIMEOUT", "1800"))   # per-call cap; pathological structures -> NaN


def run_foldx(args, cwd):
    try:
        subprocess.run([os.path.join(cwd, "foldx")] + args, cwd=cwd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, timeout=FOLDX_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass   # e.g. 1KBH (~33k atoms): leave outputs absent -> the set resolves to NaN downstream


def link_env(d):
    os.makedirs(d, exist_ok=True)
    for name, tgt in (("foldx", FOLDX), ("molecules", MOLEC)):
        p = os.path.join(d, name)
        if not os.path.islink(p) and not os.path.exists(p):
            os.symlink(tgt, p)


def repair_one(pdb):
    out = os.path.join(REPAIR, f"{pdb}_Repair.pdb")
    if os.path.exists(out):
        return "cached"
    src = os.path.join(PDBS, f"{pdb}.pdb")
    if not os.path.exists(src):
        return "no_pdb"
    d = os.path.join(WORKBASE, f"repair_{pdb}")
    shutil.rmtree(d, ignore_errors=True)
    link_env(d)
    shutil.copy(src, os.path.join(d, f"{pdb}.pdb"))
    run_foldx(["--command=RepairPDB", f"--pdb={pdb}.pdb", "--pdb-dir=.", "--output-dir=."], cwd=d)
    rp = os.path.join(d, f"{pdb}_Repair.pdb")
    if os.path.exists(rp):
        os.makedirs(REPAIR, exist_ok=True)
        shutil.copy(rp, out)
        shutil.rmtree(d, ignore_errors=True)
        return "repaired"
    shutil.rmtree(d, ignore_errors=True)
    return "failed"


THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def read_repaired_residues(rp):
    """{(chain, resnum, icode): wt_1letter} from the repaired PDB (CA atoms) — for validating mutations
    against what FoldX will actually see (chain + author resnum + icode + residue name)."""
    res = {}
    with open(rp) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                aa = THREE_TO_ONE.get(line[17:20].strip())
                rs = line[22:26].strip()
                if aa and rs:
                    res[(line[21], int(rs), line[26].strip())] = aa
    return res


def parse_ie(summary_path):
    if not os.path.exists(summary_path):
        return None
    lines = open(summary_path).read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith("Pdb\t") and "Interaction Energy" in l and i + 1 < len(lines):
            cols = l.split("\t")
            j = cols.index("Interaction Energy")
            parts = lines[i + 1].split("\t")
            try:
                return float(parts[j])
            except (ValueError, IndexError):
                return None
    return None


def ddg_one(cid, muts, runs, setid):
    """muts: list of (chain, wt, resnum, icode, mut). Returns dict or None."""
    pdb, g1, g2 = cid.split("_")
    rp = os.path.join(REPAIR, f"{pdb}_Repair.pdb")
    if not os.path.exists(rp):
        return None
    respool = read_repaired_residues(rp)                    # drop numbering/wt mismatches (uniform across arms)
    valid = [m for m in muts if respool.get((m[0], int(m[2]), str(m[3]))) == m[1]]
    n_drop = len(muts) - len(valid)
    if not valid:
        return dict(ddg_bind=np.nan, ddg_bind_sd=np.nan, ie_mut=np.nan, ie_wt=np.nan,
                    n_ie_mut=0, n_ie_wt=0, n_used=0, n_dropped=n_drop)
    muts = valid
    d = os.path.join(WORKBASE, f"set_{setid}")
    shutil.rmtree(d, ignore_errors=True)
    link_env(d)
    shutil.copy(rp, os.path.join(d, f"{pdb}_Repair.pdb"))
    mutline = ",".join(f"{wt}{ch}{rn}{ic}{mt}" for (ch, wt, rn, ic, mt) in muts) + ";"
    open(os.path.join(d, "individual_list.txt"), "w").write(mutline + "\n")
    run_foldx(["--command=BuildModel", f"--pdb={pdb}_Repair.pdb", "--mutant-file=individual_list.txt",
               f"--numberOfRuns={runs}", "--pdb-dir=.", "--output-dir=."], cwd=d)
    # BuildModel output naming depends on numberOfRuns: '<pdb>_Repair_1.pdb' (runs=1) or
    # '<pdb>_Repair_1_<r>.pdb' (runs>1). Glob both; WT_* are the wt-remodelled references.
    ie_mut, ie_wt = [], []
    for prefix, lst in (("", ie_mut), ("WT_", ie_wt)):
        pdbs = (sorted(glob.glob(os.path.join(d, f"{prefix}{pdb}_Repair_1_*.pdb")))
                or sorted(glob.glob(os.path.join(d, f"{prefix}{pdb}_Repair_1.pdb"))))
        for mp in pdbs:
            tag = os.path.basename(mp)[:-4]
            run_foldx(["--command=AnalyseComplex", f"--pdb={tag}.pdb",
                       f"--analyseComplexChains={g1},{g2}", "--pdb-dir=.", "--output-dir=."], cwd=d)
            ie = parse_ie(os.path.join(d, f"Summary_{tag}_AC.fxout"))
            if ie is not None:
                lst.append(ie)
    shutil.rmtree(d, ignore_errors=True)
    if not ie_mut or not ie_wt:
        return None
    return dict(ddg_bind=round(float(np.mean(ie_mut) - np.mean(ie_wt)), 4),
                ddg_bind_sd=round(float(np.sqrt(np.var(ie_mut) + np.var(ie_wt))), 4),
                ie_mut=round(float(np.mean(ie_mut)), 4), ie_wt=round(float(np.mean(ie_wt)), 4),
                n_ie_mut=len(ie_mut), n_ie_wt=len(ie_wt), n_used=len(muts), n_dropped=n_drop)


def build_sets(lane, wl):
    """Return ordered list of (setid, cid, label_cols, muts)."""
    d = wl[wl.lane == lane].copy()
    d["icode"] = d.icode.fillna("").astype(str)
    sets = []
    if lane == "A":
        for r in d.itertuples():
            sid = f"{r.complex_id}__A__{r.chain}{r.resnum}{r.icode}{r.mut}"
            sets.append((sid, r.complex_id, dict(chain=r.chain, wt=r.wt, resnum=int(r.resnum),
                                                 icode=r.icode, mut=r.mut),
                         [(r.chain, r.wt, int(r.resnum), r.icode, r.mut)]))
    else:
        for (cid, arm, k), g in d.groupby(["complex_id", "arm", "k"]):
            sid = f"{cid}__{arm}__k{int(k)}"
            muts = [(x.chain, x.wt, int(x.resnum), x.icode, x.mut) for x in g.itertuples()]
            sets.append((sid, cid, dict(arm=arm, k=int(k), n_mut=len(muts)), muts))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["repair", "ddg"], required=True)
    ap.add_argument("--lane", choices=["A", "B"], default="B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="substring filter on complex_id (smoke/debug)")
    a = ap.parse_args()
    os.makedirs(WORKBASE, exist_ok=True)
    wl = pd.read_csv(WORKLIST)
    if a.only:
        wl = wl[wl.complex_id.str.contains(a.only)]

    if a.mode == "repair":
        pdbs = sorted({c.split("_")[0] for c in wl.complex_id.unique()})
        mine = [p for i, p in enumerate(pdbs) if i % a.nshards == a.shard]
        print(f"[repair] shard {a.shard}/{a.nshards}: {len(mine)} of {len(pdbs)} complexes", flush=True)
        for i, pdb in enumerate(mine):
            print(f"[repair] {i+1}/{len(mine)} {pdb} -> {repair_one(pdb)}", flush=True)
        return

    sets = build_sets(a.lane, wl)
    mine = [s for i, s in enumerate(sets) if i % a.nshards == a.shard]
    done = set()
    if a.out and os.path.exists(a.out):
        try:
            done = set(pd.read_csv(a.out).setid)
        except Exception:
            done = set()
    print(f"[ddg-{a.lane}] shard {a.shard}/{a.nshards}: {len(mine)} of {len(sets)} sets "
          f"({len(done)} already done), runs={a.runs}", flush=True)
    rows = []
    for i, (sid, cid, lab, muts) in enumerate(mine):
        if sid in done:
            continue
        res = ddg_one(cid, muts, a.runs, sid)
        row = dict(setid=sid, lane=a.lane, complex_id=cid, **lab)
        row.update(res if res else dict(ddg_bind=np.nan, ddg_bind_sd=np.nan, ie_mut=np.nan,
                                        ie_wt=np.nan, n_ie_mut=0, n_ie_wt=0, n_used=0, n_dropped=0))
        rows.append(row)
        if a.out and (len(rows) % 5 == 0 or i == len(mine) - 1):        # incremental append
            df = pd.DataFrame(rows)
            hdr = not os.path.exists(a.out)
            df.to_csv(a.out, mode="a", header=hdr, index=False)
            rows = []
        if (i + 1) % 10 == 0:
            print(f"[ddg-{a.lane}] {i+1}/{len(mine)} last={sid} ddg={row.get('ddg_bind')}", flush=True)
    if rows and a.out:
        pd.DataFrame(rows).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    print(f"[ddg-{a.lane}] shard {a.shard} done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
