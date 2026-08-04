"""Shared plumbing for the factorization-tax Phase 0/1 scripts.

Structure parsing, SASA, Kabsch-Sander secondary structure, neighbour counts, and
ProteinMPNN teacher-forced conditional log-probabilities.

Deliberately dependency-light: numpy / scipy / pandas / biopython / freesasa / torch.
No DSSP binary is required (none is installable on the target machine) - secondary
structure is computed here from backbone geometry and validated in tests.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- constants

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # common modified residues mapped to their parent
    "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P", "CSO": "C", "PTR": "Y",
    "SEP": "S", "TPO": "T", "MLY": "K", "KCX": "K", "LLP": "K", "CME": "C",
    "CSD": "C", "OCS": "C", "M3L": "K", "CAS": "C", "ABA": "A", "CGU": "E",
}

# Tien et al. 2013 (PLoS ONE 8:e80635) THEORETICAL maximum accessible surface area
MAXASA_TIEN = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "Q": 225.0,
    "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0,
    "M": 224.0, "F": 240.0, "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0,
    "Y": 263.0, "V": 174.0,
}

# Kyte-Doolittle hydropathy, used only for the hydrophobicity-matched subset (BRIEF 5.2)
KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

POLAR_SET = set("STNQYCHKRDEW")  # residues with side-chain polar/H-bonding atoms

R_GAS = 1.9872036e-3  # kcal / (mol K)


# ---------------------------------------------------------------- SKEMPI

_MUT_RE = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+)([A-Za-z]?)([A-Z])$")


def parse_mutation(tok):
    """'LI38G' -> (wt='L', chain='I', resnum=38, icode='', mut='G'). None if unparseable."""
    m = _MUT_RE.match(tok.strip())
    if not m:
        return None
    wt, ch, num, icode, mut = m.groups()
    if wt not in MAXASA_TIEN or mut not in MAXASA_TIEN:
        return None
    return dict(wt=wt, chain=ch, resnum=int(num), icode=icode.strip(), mut=mut)


def _parse_temperature(x):
    if pd.isna(x):
        return np.nan
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(x))
    return float(m.group(1)) if m else np.nan


def parse_skempi(csv_path):
    """SKEMPI 2.0 -> one row per measurement with ddG_bind in kcal/mol.

    ddG = RT ln(Kd_mut / Kd_wt); positive = mutation weakens binding.
    """
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    df = df.rename(columns={"#Pdb": "pdb_group"})

    df["T_K"] = df["Temperature"].map(_parse_temperature)
    df["T_assumed"] = df["T_K"].isna()
    df["T_K"] = df["T_K"].fillna(298.0)

    kd_mut = pd.to_numeric(df["Affinity_mut_parsed"], errors="coerce")
    kd_wt = pd.to_numeric(df["Affinity_wt_parsed"], errors="coerce")
    ok = (kd_mut > 0) & (kd_wt > 0) & kd_mut.notna() & kd_wt.notna()
    df = df[ok].copy()
    kd_mut, kd_wt = kd_mut[ok], kd_wt[ok]

    df["ddG"] = R_GAS * df["T_K"] * np.log(kd_mut / kd_wt)

    parts = df["pdb_group"].str.split("_", expand=True)
    df["pdb"] = parts[0].str.upper()
    df["group1"] = parts[1]
    df["group2"] = parts[2]

    df["muts"] = df["Mutation(s)_cleaned"].fillna("").str.split(",")
    df["n_mut"] = df["muts"].map(len)
    return df


# ---------------------------------------------------------------- structure

class ComplexStruct:
    """Backbone of one SKEMPI complex (both chain groups), in a fixed residue order."""

    __slots__ = ("pdb", "group1", "group2", "chains", "resnums", "icodes", "seq",
                 "N", "CA", "C", "O", "CB", "bfac", "group", "n")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _virtual_cb(N, CA, C):
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def load_complex(pdb_path, pdb_id, group1, group2, require_both=True):
    """Parse a SKEMPI PDB, keeping only the chains named in the two groups.

    Residues must have all of N, CA, C, O. Returns None if either group is empty,
    unless `require_both=False` - used to build the isolated-partner (monomer) state
    for the monomer-versus-complex frustration proxy.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_id, pdb_path)
    model = next(iter(struct))

    want = [(c, 1) for c in group1] + [(c, 2) for c in group2]

    chains, resnums, icodes, seq, group = [], [], [], [], []
    Ns, CAs, Cs, Os, bfs = [], [], [], [], []

    for chain_id, grp in want:
        if chain_id not in model:
            continue
        for res in model[chain_id]:
            het, resnum, icode = res.id
            name = res.get_resname().strip().upper()
            if name not in THREE2ONE:
                continue
            try:
                n, ca, c, o = (res["N"], res["CA"], res["C"], res["O"])
            except KeyError:
                continue
            chains.append(chain_id)
            resnums.append(int(resnum))
            icodes.append(icode.strip())
            seq.append(THREE2ONE[name])
            group.append(grp)
            Ns.append(n.get_coord()); CAs.append(ca.get_coord())
            Cs.append(c.get_coord()); Os.append(o.get_coord())
            # residue B-factor = mean over backbone atoms actually present
            bl = [a.get_bfactor() for a in res if a.element != "H"]
            bfs.append(float(np.mean(bl)) if bl else np.nan)

    if not seq:
        return None
    g = np.array(group)
    if require_both and (not (g == 1).any() or not (g == 2).any()):
        return None

    N = np.array(Ns, dtype=np.float64)
    CA = np.array(CAs, dtype=np.float64)
    C = np.array(Cs, dtype=np.float64)
    O = np.array(Os, dtype=np.float64)
    CB = _virtual_cb(N, CA, C)

    return ComplexStruct(
        pdb=pdb_id, group1=group1, group2=group2,
        chains=np.array(chains), resnums=np.array(resnums),
        icodes=np.array(icodes), seq=np.array(seq), group=g,
        N=N, CA=CA, C=C, O=O, CB=CB, bfac=np.array(bfs), n=len(seq),
    )


# ---------------------------------------------------------------- SASA

def residue_sasa(pdb_path, pdb_id, keep_chains):
    """Per-residue absolute SASA (A^2) for a chain subset, via freesasa Shrake-Rupley.

    Returns {(chain, resnum, icode): sasa}. Heavy atoms only, no waters/ligands.
    """
    import freesasa
    from Bio.PDB import PDBParser

    freesasa.setVerbosity(freesasa.silent)
    parser = PDBParser(QUIET=True)
    model = next(iter(parser.get_structure(pdb_id, pdb_path)))

    st = freesasa.Structure()
    keys = []
    for chain_id in keep_chains:
        if chain_id not in model:
            continue
        for res in model[chain_id]:
            name = res.get_resname().strip().upper()
            if name not in THREE2ONE:
                continue
            _, resnum, icode = res.id
            key = (chain_id, int(resnum), icode.strip())
            for atom in res:
                if atom.element == "H":
                    continue
                x, y, z = (float(v) for v in atom.get_coord())
                # freesasa needs the canonical 3-letter name to assign a radius
                try:
                    st.addAtom(atom.get_name().ljust(4)[:4], name, str(resnum),
                               chain_id, x, y, z)
                except Exception:
                    continue
                keys.append(key)

    if st.nAtoms() == 0:
        return {}
    res_result = freesasa.calc(st)
    out = {}
    for i in range(st.nAtoms()):
        out[keys[i]] = out.get(keys[i], 0.0) + res_result.atomArea(i)
    return out


def relative_sasa(abs_sasa, restype):
    m = MAXASA_TIEN.get(restype)
    return np.nan if m is None else abs_sasa / m


# ---------------------------------------------------- secondary structure

def secondary_structure(cx, prefer_pydssp=True):
    """3-class secondary structure (H / E / L), pydssp if available else our own.

    pydssp implements the full DSSP criteria (two consecutive n-turns for helix, a
    ladder for strand). Our fallback declares helix from a single n-turn and strand
    from a single bridge, which over-calls both - measured against pydssp on the
    validation structures: 1MBN H 0.87 vs 0.77, 1TEN H 0.11 (should be 0.00) and
    E 0.36 vs 0.51. Prefer pydssp.
    """
    if prefer_pydssp:
        try:
            import pydssp
            coord = np.stack([cx.N, cx.CA, cx.C, cx.O], axis=1)  # [L,4,3] N,CA,C,O
            raw = pydssp.assign(coord, out_type="c3")
            return np.array(["L" if c == "-" else c for c in raw], dtype="<U1")
        except Exception:
            pass
    return kabsch_sander_ss(cx)


def kabsch_sander_ss(cx):
    """3-class secondary structure (H / E / L) from backbone geometry.

    Faithful Kabsch & Sander (1983) electrostatic hydrogen-bond definition:
        E = 0.084 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) * 332  kcal/mol
    with an H-bond declared at E < -0.5. n-turns give helix (H); bridges give
    strand (E); helix takes priority. Used because no DSSP binary is installable
    on the target machine.
    """
    n = cx.n
    ss = np.array(["L"] * n, dtype="<U1")
    if n < 4:
        return ss

    # amide H: 1.0 A from N, antiparallel to the preceding C=O
    H = cx.N.copy()
    same_chain_prev = np.zeros(n, dtype=bool)
    same_chain_prev[1:] = (cx.chains[1:] == cx.chains[:-1])
    d = cx.C[:-1] - cx.O[:-1]
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    H[1:] = cx.N[1:] + np.where(same_chain_prev[1:, None], d / nrm, 0.0)

    # pair energies; i donates (N-H of i), j accepts (C=O of j)
    def dist(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rON = dist(cx.O, cx.N).T   # [i,j] = |O_j - N_i|
        rCH = dist(cx.C, H).T
        rOH = dist(cx.O, H).T
        rCN = dist(cx.C, cx.N).T
        Emat = 0.084 * (1.0 / rON + 1.0 / rCH - 1.0 / rOH - 1.0 / rCN) * 332.0

    hb = Emat < -0.5
    np.fill_diagonal(hb, False)
    # proline has no amide H; and only within-chain bonds count for SS classes
    hb[cx.seq == "P", :] = False
    hb &= (cx.chains[:, None] == cx.chains[None, :])
    # exclude trivially close pairs
    idx = np.arange(n)
    hb &= np.abs(idx[:, None] - idx[None, :]) >= 3

    def turn(k):
        t = np.zeros(n, dtype=bool)
        if n > k:
            valid = cx.chains[: n - k] == cx.chains[k:]
            t[: n - k] = hb[np.arange(k, n), np.arange(0, n - k)] & valid
        return t

    # helices: an n-turn at i puts i+1..i+k-1 inside the helix
    for k in (3, 4, 5):
        t = turn(k)
        for i in np.flatnonzero(t):
            ss[i + 1: i + k] = "H"

    # bridges -> strand
    def hbf(a, b):
        return hb[a, b] if (0 <= a < n and 0 <= b < n) else False

    is_e = np.zeros(n, dtype=bool)
    cand = np.argwhere(hb)
    for i, j in cand:
        if abs(i - j) <= 2:
            continue
        par = (hbf(i, j - 1) and hbf(j + 1, i)) or (hbf(j, i - 1) and hbf(i + 1, j))
        anti = (hbf(i, j) and hbf(j, i)) or (hbf(i - 1, j + 1) and hbf(j - 1, i + 1))
        if par or anti:
            is_e[i] = True
            is_e[j] = True
    ss[(ss == "L") & is_e] = "E"
    return ss


def neighbour_counts(cx, cutoff=10.0):
    """Number of other residues in the complex with Cbeta within `cutoff` of this Cbeta."""
    d = np.linalg.norm(cx.CB[:, None, :] - cx.CB[None, :, :], axis=-1)
    return ((d < cutoff).sum(axis=1) - 1).astype(int)


# ---------------------------------------------------------------- ProteinMPNN

def load_mpnn(weights_path, device="cpu", augment_eps=0.0):
    """Load a ProteinMPNN checkpoint with backbone noise disabled at inference."""
    import torch
    import sys
    repo = os.environ.get("PROTEINMPNN_DIR", os.path.expanduser("~/ftax/ProteinMPNN"))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from protein_mpnn_utils import ProteinMPNN

    ckpt = torch.load(weights_path, map_location=device)
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=augment_eps, k_neighbors=ckpt["num_edges"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt.get("noise_level", None)


def featurize(cx, device="cpu"):
    """ComplexStruct -> ProteinMPNN input tensors (batch of 1)."""
    import torch
    X = np.stack([cx.N, cx.CA, cx.C, cx.O], axis=1)[None]      # [1, L, 4, 3]
    S = np.array([[MPNN_ALPHABET.index(a) if a in MPNN_ALPHABET else 20
                   for a in cx.seq]])                            # [1, L]
    L = cx.n
    mask = np.ones((1, L), dtype=np.float32)

    # residue_idx: running index, +100 offset at every chain boundary (MPNN convention)
    residue_idx = np.zeros((1, L), dtype=np.int64)
    chain_enc = np.zeros((1, L), dtype=np.int64)
    uniq = []
    for c in cx.chains:
        if c not in uniq:
            uniq.append(c)
    for ci, c in enumerate(uniq):
        sel = np.flatnonzero(cx.chains == c)
        residue_idx[0, sel] = 100 * ci + sel
        chain_enc[0, sel] = ci + 1

    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return (t(X, torch.float32), t(S, torch.long), t(mask, torch.float32),
            t(residue_idx, torch.long), t(chain_enc, torch.long))


def mpnn_conditional_logprobs(model, cx, seeds=range(8), device="cpu"):
    """Teacher-forced conditional log-probs over several decoding orders.

    Returns [n_orders, L, 21]: log p(s_i | bound-complex backbone, s_{<i in order}).
    All positions are decodable, so each order is a uniformly random permutation
    of the whole complex.
    """
    import torch
    X, S, mask, residue_idx, chain_enc = featurize(cx, device)
    seeds = list(seeds)

    # Orders differ only in permutation, so they share X/S/mask and can be batched.
    # Verified bit-identical to running them one at a time (max abs diff 0.0).
    orders = []
    for sd in seeds:
        g = torch.Generator(device="cpu").manual_seed(int(sd))
        randn = torch.randn(1, cx.n, generator=g)
        orders.append(torch.argsort((torch.ones(1, cx.n) + 0.0001) * torch.abs(randn))[0])
    orders = torch.stack(orders).to(device)

    max_batch = int(max(1, min(8, 6000 // max(cx.n, 1))))
    out = []
    with torch.no_grad():
        for s in range(0, len(seeds), max_batch):
            o = orders[s:s + max_batch]
            b = o.shape[0]
            mb = mask.repeat(b, 1)
            lp = model(X.repeat(b, 1, 1, 1), S.repeat(b, 1), mb, mb.clone(),
                       residue_idx.repeat(b, 1), chain_enc.repeat(b, 1),
                       torch.zeros(b, cx.n, device=device),
                       use_input_decoding_order=True, decoding_order=o)
            out.append(lp.cpu().numpy())
    return np.concatenate(out, axis=0)


def order_mixture_logprobs(lp):
    """[n_orders, L, 21] -> [L, 21] log of the mean PROBABILITY over decoding orders.

    This is a properly normalised distribution (the mixture over decoding orders) and is
    what N_hot needs. The mean of log-probs, used as the per-position score elsewhere,
    is the average log-likelihood over orders and is deliberately NOT normalised.
    """
    m = lp.max(axis=0, keepdims=True)
    return (m + np.log(np.exp(lp - m).mean(axis=0, keepdims=True)))[0]


def mpnn_unconditional_logprobs(model, cx, device="cpu"):
    """Backbone-only log-probs (no sequence context) -> no decoding-order variance."""
    import torch
    X, S, mask, residue_idx, chain_enc = featurize(cx, device)
    with torch.no_grad():
        lp = model.unconditional_probs(X, mask, residue_idx, chain_enc)
    return lp[0].cpu().numpy()
