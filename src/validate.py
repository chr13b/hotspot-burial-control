"""Positive controls for every computational path used in Phase 0/1.

CLAUDE.md ground rule 6: a negative from any search, query or filter is only as good
as a positive control run through the same path. Nothing downstream should be trusted
until this prints ALL PASS.

Usage:  python3 src/validate.py --data-dir ~/ftax/data --mpnn-weights <ckpt>
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_mutation_parser():
    print("\n== 1. SKEMPI mutation-string parser ==")
    cases = [
        ("LI38G", dict(wt="L", chain="I", resnum=38, icode="", mut="G")),
        ("TI17A", dict(wt="T", chain="I", resnum=17, icode="", mut="A")),
        ("YA100AF", dict(wt="Y", chain="A", resnum=100, icode="A", mut="F")),
        ("DE-2A", dict(wt="D", chain="E", resnum=-2, icode="", mut="A")),
    ]
    for tok, want in cases:
        got = fc.parse_mutation(tok)
        check(f"parse {tok}", got == want, f"got {got}")
    check("reject junk", fc.parse_mutation("nonsense") is None)


def test_secondary_structure(data_dir):
    """All-alpha myoglobin must come out mostly H; all-beta fibronectin-III mostly E."""
    print("\n== 2. Kabsch-Sander secondary structure (no DSSP binary available) ==")
    import urllib.request
    for pdb_id, chain, expect in [("1MBN", "A", "H"), ("1TEN", "A", "E")]:
        path = os.path.join(data_dir, f"{pdb_id}.pdb")
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(
                    f"https://files.rcsb.org/download/{pdb_id}.pdb", path)
            except Exception as e:
                check(f"{pdb_id} download", False, str(e))
                continue
        # BUGFIX: this previously called load_complex(path, pdb_id, chain, chain),
        # which puts the SAME chain in both groups and loads every residue twice
        # (1MBN: n=306 for a 153-residue chain). The SS gate was therefore computed on
        # duplicated coordinates, and _single_chain() below was unreachable dead code.
        cx = _single_chain(path, pdb_id, chain)
        assert cx.n < 400, f"{pdb_id}: n={cx.n} - residues loaded twice again?"
        ss = fc.kabsch_sander_ss(cx)
        fh = float((ss == "H").mean())
        fe = float((ss == "E").mean())
        detail = f"H={fh:.2f} E={fe:.2f} L={float((ss=='L').mean()):.2f} n={cx.n}"
        if expect == "H":
            check(f"{pdb_id} is helix-dominated", fh > 0.55 and fe < 0.10, detail)
        else:
            check(f"{pdb_id} is strand-dominated", fe > 0.30 and fh < 0.15, detail)


def _single_chain(path, pdb_id, chain):
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    model = next(iter(p.get_structure(pdb_id, path)))
    chains, resnums, icodes, seq, group = [], [], [], [], []
    Ns, CAs, Cs, Os, bfs = [], [], [], [], []
    for res in model[chain]:
        name = res.get_resname().strip().upper()
        if name not in fc.THREE2ONE:
            continue
        try:
            n, ca, c, o = res["N"], res["CA"], res["C"], res["O"]
        except KeyError:
            continue
        _, rn, ic = res.id
        chains.append(chain); resnums.append(int(rn)); icodes.append(ic.strip())
        seq.append(fc.THREE2ONE[name]); group.append(1)
        Ns.append(n.get_coord()); CAs.append(ca.get_coord())
        Cs.append(c.get_coord()); Os.append(o.get_coord())
        bfs.append(float(np.mean([a.get_bfactor() for a in res])))
    N = np.array(Ns, float); CA = np.array(CAs, float)
    C = np.array(Cs, float); O = np.array(Os, float)
    return fc.ComplexStruct(pdb=pdb_id, group1=chain, group2="", chains=np.array(chains),
                            resnums=np.array(resnums), icodes=np.array(icodes),
                            seq=np.array(seq), group=np.array(group), N=N, CA=CA, C=C,
                            O=O, CB=fc._virtual_cb(N, CA, C), bfac=np.array(bfs),
                            n=len(seq))


def test_sasa(data_dir):
    """Our addAtom-based SASA must agree with freesasa's own PDB reader."""
    print("\n== 3. SASA path (freesasa addAtom vs freesasa file reader) ==")
    import freesasa
    freesasa.setVerbosity(freesasa.silent)
    path = os.path.join(data_dir, "1MBN.pdb")
    if not os.path.exists(path):
        check("1MBN present", False)
        return
    ref_total = freesasa.calc(freesasa.Structure(path)).totalArea()
    ours = fc.residue_sasa(path, "1MBN", "A")
    our_total = sum(ours.values())
    rel = abs(our_total - ref_total) / ref_total
    check("total SASA agrees within 5%", rel < 0.05,
          f"ours={our_total:.0f} ref={ref_total:.0f} rel_diff={rel:.3%}")
    check("per-residue SASA non-empty", len(ours) > 100, f"{len(ours)} residues")
    # burial must actually vary, else matching is meaningless
    rs = [fc.relative_sasa(v, "A") for v in ours.values()]
    check("rSASA spans buried..exposed", min(rs) < 0.05 and max(rs) > 0.5,
          f"min={min(rs):.3f} max={max(rs):.3f}")


def test_interface_burial(data_dir):
    """Complexation must bury residues: some interface rSASA must drop."""
    print("\n== 4. Interface detection (bound vs free SASA) ==")
    path = os.path.join(data_dir, "PDBs", "1CSE.pdb")
    if not os.path.exists(path):
        check("1CSE present", False)
        return
    cx = fc.load_complex(path, "1CSE", "E", "I")
    check("1CSE loads with both groups", cx is not None)
    if cx is None:
        return
    bound = fc.residue_sasa(path, "1CSE", "EI")
    free_i = fc.residue_sasa(path, "1CSE", "I")
    keys = [k for k in free_i if k in bound]
    d = np.array([free_i[k] - bound[k] for k in keys])
    check("some chain-I residues are buried on binding", (d > 10).sum() >= 5,
          f"{(d>10).sum()} residues lose >10 A^2")
    check("free SASA >= bound SASA (no negatives beyond noise)", d.min() > -1.0,
          f"min delta={d.min():.2f}")


def test_mpnn(data_dir, weights):
    """ProteinMPNN must recover native sequence far above the 5% random baseline."""
    print("\n== 5. ProteinMPNN scoring path ==")
    path = os.path.join(data_dir, "PDBs", "1CSE.pdb")
    model, noise = fc.load_mpnn(weights)
    check("checkpoint loads", model is not None, f"training noise level {noise}")
    cx = fc.load_complex(path, "1CSE", "E", "I")
    lp = fc.mpnn_conditional_logprobs(model, cx, seeds=range(3))
    check("shape [orders, L, 21]", lp.shape == (3, cx.n, 21), str(lp.shape))
    mean_lp = lp.mean(axis=0)
    pred = mean_lp[:, :20].argmax(axis=1)
    native = np.array([fc.MPNN_ALPHABET.index(a) for a in cx.seq])
    rec = float((pred == native).mean())
    check("sequence recovery >> random (0.05)", 0.30 < rec < 0.85, f"recovery={rec:.3f}")
    # each single order must be a normalised distribution ...
    lse1 = np.log(np.exp(lp[0]).sum(axis=1))
    check("single-order log-probs normalised", np.allclose(lse1, 0, atol=1e-4),
          f"max|logsumexp|={np.abs(lse1).max():.2e}")
    # ... and so must the across-order MIXTURE (mean of probs), which is what
    # N_hot needs. The mean of LOG-probs is deliberately not normalised.
    mix = fc.order_mixture_logprobs(lp)
    lse2 = np.log(np.exp(mix).sum(axis=1))
    check("order-mixture log-probs normalised", np.allclose(lse2, 0, atol=1e-4),
          f"max|logsumexp|={np.abs(lse2).max():.2e}")
    # decoding order must actually change the conditionals
    spread = lp[:, :, :20].std(axis=0).mean()
    check("decoding order changes conditionals", spread > 1e-4, f"mean SD={spread:.4f}")
    unc = fc.mpnn_unconditional_logprobs(model, cx)
    check("unconditional path runs", unc.shape == (cx.n, 21), str(unc.shape))
    nat_c = mean_lp[np.arange(cx.n), native].mean()
    nat_u = unc[np.arange(cx.n), native].mean()
    check("sequence context helps (cond > uncond)", nat_c > nat_u,
          f"cond={nat_c:.3f} uncond={nat_u:.3f}")


def test_skempi(data_dir):
    """ddG sign convention: a known hotspot must come out strongly positive."""
    print("\n== 6. SKEMPI parsing and ddG sign ==")
    df = fc.parse_skempi(os.path.join(data_dir, "skempi_v2.csv"))
    check("rows parsed", len(df) > 5000, f"{len(df)} rows with ddG")
    check("complexes parsed", df["pdb"].nunique() > 200, f"{df['pdb'].nunique()} PDBs")

    # Barnase-barstar 1BRS: barstar D39A is a canonical hotspot (large positive ddG)
    sub = df[(df["pdb"] == "1BRS") & (df["n_mut"] == 1)]
    hits = sub[sub["Mutation(s)_cleaned"].str.strip() == "DD39A"]
    if len(hits):
        v = hits["ddG"].median()
        check("1BRS barstar D39A is a strong hotspot (ddG > 2)", v > 2.0,
              f"ddG={v:.2f} kcal/mol, n={len(hits)}")
    else:
        check("1BRS D39A present", False, "mutation not found")

    ala = df[(df["n_mut"] == 1) &
             df["Mutation(s)_cleaned"].str.strip().str.match(r"^[A-Z][A-Za-z0-9]-?\d+[A-Za-z]?A$")]
    check("alanine scan subset non-trivial", len(ala) > 1000, f"{len(ala)} Ala mutations")
    frac_pos = float((ala["ddG"] > 0).mean())
    check("most Ala mutations weaken binding (ddG>0)", frac_pos > 0.6,
          f"{frac_pos:.1%} positive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/ftax/data"))
    ap.add_argument("--mpnn-weights",
                    default=os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt"))
    a = ap.parse_args()

    test_mutation_parser()
    test_secondary_structure(a.data_dir)
    test_sasa(a.data_dir)
    test_interface_burial(a.data_dir)
    test_mpnn(a.data_dir, a.mpnn_weights)
    test_skempi(a.data_dir)

    print("\n" + "=" * 66)
    if FAILS:
        print(f"VALIDATION FAILED: {len(FAILS)} check(s) -> {FAILS}")
        sys.exit(1)
    print("ALL PASS - every computational path has a firing positive control.")


if __name__ == "__main__":
    main()
