"""PiFold (Gao et al., ICLR 2023) behind the ftax_common interface.

`pifold_conditional_logprobs(model, cx) -> [1, L, 21]` in fc.MPNN_ALPHABET order.

PiFold is ONE-SHOT / non-autoregressive: `forward()` does a single pass and never
consumes `decoding_order` (it is computed and discarded). It is bit-exactly
deterministic across seeds. So there is exactly one "order" and decoding-order
variance is structurally zero -- which is why it is worth having in the panel.

Caveats that must be reported, not buried:
  * PiFold has NO chain representation at all: no chain id, no residue index, no
    positional embedding (`num_positional_embeddings` is declared and never used).
    It was trained on CATH 4.2 single domains.
  * Concatenating chains therefore fabricates one bogus phi/psi/omega across each
    junction (`_dihedrals` reshapes to (B,3L,3) over consecutive triples).
    `junction_mask` below flags the affected residues; damage is local (measured:
    interior mean |dlogit| 0.088, junction termini 2.4-2.8).
  * Inter-chain contacts DO enter the graph as ordinary CA k-NN edges (measured
    4.28% of edges on 1CSE), so the partner is seen -- just not as a partner.
  * There is no X token: PiFold emits 20 logits. Column 20 is set to LOG_ZERO.
"""

import sys
import numpy as np
import torch

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PIFOLD_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"     # API/featurizer.py:16
DEFAULT_REPO = "/home/chris/ftax/models/pifold/repo"
DEFAULT_CKPT = "/home/chris/ftax/models/pifold/checkpoint.pth"
LOG_ZERO = -1000.0        # exp() underflows to exactly 0.0; finite so CSVs stay clean


def load_pifold(ckpt_path=DEFAULT_CKPT, repo=DEFAULT_REPO, device="cpu"):
    """Instantiate ProDesign_Model directly; the repo's Exp/cuda() helpers hardcode CUDA."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from methods.prodesign_model import ProDesign_Model

    class A:
        pass
    a = A()
    # defaults from the repo's parser.py; these are what the released ckpt expects
    a.hidden_dim = 128; a.node_features = 128; a.edge_features = 128
    a.k_neighbors = 30; a.dropout = 0.1; a.num_encoder_layers = 10
    a.updating_edges = 4
    a.node_dist = 1; a.node_angle = 1; a.node_direct = 1
    a.edge_dist = 1; a.edge_angle = 1; a.edge_direct = 1
    a.virtual_num = 3

    model = ProDesign_Model(a)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)          # strict: config must match exactly
    model.to(device).eval()
    return model


def build_alphabet_map():
    """MPNN column j -> PiFold logit column, or -1 for 'no such token' (X)."""
    idx = []
    for aa in MPNN_ALPHABET:
        idx.append(PIFOLD_ALPHABET.index(aa) if aa in PIFOLD_ALPHABET else -1)
    return np.asarray(idx, dtype=np.int64)


def junction_mask(cx, flank=2):
    """True for residues within `flank` of an artificial chain junction.

    PiFold sees the concatenated complex as one chain, so the residues either side
    of a junction get a fabricated backbone dihedral. Exclude these downstream.
    """
    m = np.zeros(cx.n, dtype=bool)
    brk = np.flatnonzero(cx.chains[1:] != cx.chains[:-1])   # last index of each chain
    for b in brk:
        m[max(0, b - flank + 1): b + 1 + flank] = True
    return m


def _to_pifold_batch(cx):
    d = {"title": str(cx.pdb), "seq": "".join(cx.seq)}
    for name, arr in (("N", cx.N), ("CA", cx.CA), ("C", cx.C), ("O", cx.O)):
        d[name] = np.asarray(arr, dtype=np.float64)
    bad = set(d["seq"]) - set(PIFOLD_ALPHABET)
    if bad:
        raise ValueError("PiFold featurizer cannot handle non-canonical residues %r "
                         "(API/featurizer.py:33 does alphabet.index())" % sorted(bad))
    return d


def pifold_conditional_logprobs(model, cx, device="cpu", renormalise=True):
    """[1, L, 21] log-probs in MPNN alphabet order. One-shot -> exactly one order."""
    if DEFAULT_REPO not in sys.path:
        sys.path.insert(0, DEFAULT_REPO)
    from API.featurizer import featurize_GTrans

    data = _to_pifold_batch(cx)
    with torch.no_grad():
        X, S, score, mask, lengths = featurize_GTrans([data])
        X, S, score, h_V, h_E, E_idx, batch_id, _bw, _fw, _do = \
            model._get_features(S, score, X=X, mask=mask)
        _logp, logits = model(h_V, h_E, E_idx, batch_id, return_logit=True)

    lsm = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()    # [L, 20]
    if lsm.shape[0] != cx.n:
        raise RuntimeError("PiFold returned %d rows for %d residues (masked residues "
                           "were dropped)" % (lsm.shape[0], cx.n))

    amap = build_alphabet_map()
    out = np.full((1, cx.n, 21), LOG_ZERO, dtype=np.float64)
    have = amap >= 0
    out[0][:, have] = lsm[:, amap[have]]
    if renormalise:
        m = out[0].max(axis=1, keepdims=True)
        out[0] -= m + np.log(np.exp(out[0] - m).sum(axis=1, keepdims=True))
    return out
