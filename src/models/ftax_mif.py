"""MIF (Yang et al. 2023, masked inverse folding) behind the ftax_common interface.

`mif_conditional_logprobs(model, collater, cx, seeds=...) -> [n_orders, L, 21]`
`mif_unconditional_logprobs(model, collater, cx)          -> [L, 21]`
both in fc.MPNN_ALPHABET order.

MIF is a MASKED model, not autoregressive, so there is no decoding order. The
analogue of ProteinMPNN's teacher-forced pass under a random decoding order --
where position i sees a random subset of the other residues -- is a random-subset
MASK. Each `seed` here draws a random half A and its complement B, runs two
forward passes (A masked / B masked), and reads each position off the pass in
which it was masked. Every position is therefore conditioned on the full complex
backbone plus a uniformly random ~half of the other native residues, which is the
expected conditioning set size under a random MPNN decoding order.

TWO TRAPS, both silent:
  1. The wrapper does NOT mask despite its docstring. Feeding the native sequence
     unmasked gives ~0.92 "recovery" -- that is the model copying its input, not
     inverse folding. Masking is done explicitly here.
  2. `sequence_models.pdb_utils.parse_PDB` keys residues by residue NUMBER only
     and silently merges chains with overlapping numbering. It is never used here;
     coordinates come from `fc.load_complex`.

MULTI-CHAIN -- and a correction. It is tempting to conclude from `pe=False`
(`self.pe = nn.Identity()`) and from `process_coords` building only PAIRWISE
trRosetta features that MIF is order-invariant and therefore chain-break-safe.
IT IS NOT. `gnn.get_node_features` (sequence_models/gnn.py:448-461) takes
`torch.diagonal(omega, offset=+1)` and `torch.diagonal(theta/phi, offset=+/-1)`,
i.e. five of the ten node features are angles to the SEQUENCE-ADJACENT residue.
Measured on 1CSE: a full residue permutation changes log-probs by up to 19.9, and
swapping the chain concatenation order [E,I] -> [I,E] changes chain E by up to
6.76. The effect is concentrated at the junction -- the two largest deviations sit
exactly at the chain termini; excluding +/-2 residues around each junction drops
the max to 1.71 and the mean to 0.13 (residual is message passing propagating the
junction perturbation over 3 decoder layers).

So MIF has the same class of junction artifact as PiFold. Use `junction_mask`.
`chain_order_sensitivity` measures it rather than assuming it away.

Inputs are N, CA, C only. O is never read.
"""

import os
import numpy as np
import torch

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
DEFAULT_HUB = "/home/chris/ftax/models/mif/torchhub"


def load_mif(name="mif", torch_home=DEFAULT_HUB, device="cpu"):
    """Returns (model, collater). Use 'mif', NOT 'mif_st' (404) and NOT 'mifst'
    (which pulls CARP-640M, a 2.57 GB extra download peaking ~5.1 GB -- will not
    fit in 4 GB)."""
    if torch_home:
        os.environ["TORCH_HOME"] = torch_home
    from sequence_models.pretrained import load_model_and_alphabet
    model, collater = load_model_and_alphabet(name)
    model.to(device).eval()
    return model, collater


def build_alphabet_map():
    """MPNN column j -> index into MIF's 30-token PROTEIN_ALPHABET. Looked up, never
    hardcoded: PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY' + 'BZX' + 'JOU' + '*-#@'."""
    from sequence_models.constants import PROTEIN_ALPHABET
    idx = np.asarray([PROTEIN_ALPHABET.index(aa) for aa in MPNN_ALPHABET], dtype=np.int64)
    if len(set(idx.tolist())) != len(MPNN_ALPHABET):
        raise ValueError("alphabet map not injective")
    return idx


def _features(cx):
    """cx -> the four trRosetta pair features MIF consumes. N/CA/C only."""
    from sequence_models.pdb_utils import process_coords
    coords = {"N": np.asarray(cx.N), "CA": np.asarray(cx.CA), "C": np.asarray(cx.C)}
    dist, omega, theta, phi = process_coords(coords)
    return [torch.tensor(x, dtype=torch.float) for x in (dist, omega, theta, phi)]


def _forward(model, collater, feats, seq_str, amap, device, renormalise=True):
    src, nodes, edges, connections, edge_mask = collater([[seq_str] + feats])
    with torch.no_grad():
        logits = model(src.to(device), nodes.to(device), edges.to(device),
                       connections.to(device), edge_mask.to(device), result="logits")
    lsm = torch.log_softmax(logits.float(), dim=-1)[0]        # [L, 30]
    sub = lsm[:, torch.as_tensor(amap)]                        # [L, 21]
    if renormalise:
        sub = sub - torch.logsumexp(sub, dim=1, keepdim=True)
    return sub.cpu().numpy()


def mif_unconditional_logprobs(model, collater, cx, device="cpu", renormalise=True):
    """All positions masked at once: log p(s_i | backbone) with NO sequence context.

    This is the exact analogue of `fc.mpnn_unconditional_logprobs`. One forward
    pass, zero variance.
    """
    amap = build_alphabet_map()
    feats = _features(cx)
    return _forward(model, collater, feats, "#" * cx.n, amap, device, renormalise)


def mif_conditional_logprobs(model, collater, cx, seeds=range(8), mask_frac=0.5,
                             device="cpu", renormalise=True):
    """[n_orders, L, 21]. Each seed = one complementary pair of random masks.

    Position i in order o is conditioned on the full complex backbone plus the
    native identity of a uniformly random subset (~1-mask_frac) of the others.
    Two forward passes per seed.
    """
    amap = build_alphabet_map()
    feats = _features(cx)
    seq = np.array(list("".join(cx.seq)))
    seeds = list(seeds)
    out = np.full((len(seeds), cx.n, 21), np.nan, dtype=np.float64)

    for oi, sd in enumerate(seeds):
        rng = np.random.RandomState(int(sd))
        pick = rng.rand(cx.n) < mask_frac        # group A masked in pass 1
        for grp in (pick, ~pick):
            s = seq.copy()
            s[grp] = "#"
            lp = _forward(model, collater, feats, "".join(s), amap, device, renormalise)
            out[oi, grp, :] = lp[grp]

    assert not np.isnan(out).any(), "some positions were never masked"
    return out


def junction_mask(cx, flank=2):
    """True for residues within `flank` of an artificial chain junction.

    MIF's node features include angles to the sequence-adjacent residue, so the
    residues either side of a concatenation junction get fabricated features.
    Same remedy as PiFold: drop them downstream.
    """
    m = np.zeros(cx.n, dtype=bool)
    for b in np.flatnonzero(cx.chains[1:] != cx.chains[:-1]):
        m[max(0, b - flank + 1): b + 1 + flank] = True
    return m


def chain_order_sensitivity(model, collater, cx_a, cx_b, device="cpu"):
    """Diagnostic, NOT an invariance test: MIF is order-DEPENDENT.

    Pass the same complex loaded with two different chain orders (e.g.
    load_complex(...,'E','I') and load_complex(...,'I','E')) and this returns the
    per-residue max|dlogp| for each chain, so the junction artifact can be
    quantified per complex instead of assumed.
    """
    amap = build_alphabet_map()
    a = _forward(model, collater, _features(cx_a), "#" * cx_a.n, amap, device)
    b = _forward(model, collater, _features(cx_b), "#" * cx_b.n, amap, device)
    out = {}
    for c in set(cx_a.chains):
        ia, ib = np.flatnonzero(cx_a.chains == c), np.flatnonzero(cx_b.chains == c)
        out[c] = np.abs(a[ia] - b[ib]).max(axis=1)
    return out
