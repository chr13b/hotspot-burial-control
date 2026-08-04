"""ESM-IF1 (esm_if1_gvp4_t16_142M_UR50) behind the ftax_common interface.

Exposes `esmif_conditional_logprobs(model, alphabet, cx, ...) -> [n_orders, L, 21]`
with columns in fc.MPNN_ALPHABET order ('ACDEFGHIKLMNPQRSTVWYX'), i.e. the same
shape and the same alphabet as `fc.mpnn_conditional_logprobs`.

Important semantic differences from ProteinMPNN, stated up front:

  * ESM-IF1's decoder is a CAUSAL left-to-right transformer. There is no
    permutable decoding order. Position i is conditioned on residues 1..i-1 of
    the target chain only. `n_orders` here indexes permutations of the NON-target
    chains in the coordinate concatenation, which is the only ordering freedom
    the architecture has. For a 2-chain complex that is degenerate -> [1, L, 21].
  * Only N, CA, C are used (no O, no CB). ESM-IF1 never sees carbonyl oxygen.
  * Following esm.inverse_folding.multichain_util.score_sequence_in_complex, the
    target chain is placed FIRST in the concatenated coordinates and only the
    target chain's sequence is fed to the decoder. So a residue is conditioned on
    the full complex BACKBONE but NOT on the partner's SEQUENCE. ProteinMPNN, by
    contrast, sees partner sequence for partner positions decoded earlier. This
    is a real conditioning difference and must be reported, not hidden.
    `whole_complex_logprobs` below gives the partner-sequence-context variant.
"""

import numpy as np
import torch
import torch.nn.functional as F

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
DEFAULT_CKPT = "/home/chris/ftax/models/esm_if1_slim.pt"


# ------------------------------------------------------------------ loading

def load_esmif(ckpt_path=DEFAULT_CKPT, device="cpu"):
    """Load ESM-IF1 from a slimmed checkpoint ({'args','model'} only, 567 MB).

    The checkpoint published by FAIR is 1.7 GB because it carries Adam optimizer
    state; loading that directly peaks near 2.4 GB and OOMs on a 4 GB box.
    """
    import esm
    from esm.pretrained import load_model_and_alphabet_core

    model_data = torch.load(ckpt_path, map_location="cpu")
    model, alphabet = load_model_and_alphabet_core(
        "esm_if1_gvp4_t16_142M_UR50", model_data, None)
    model = model.to(device)
    model.eval()          # MANDATORY: the GVP encoder has dropout
    return model, alphabet


# ------------------------------------------------------- alphabet remapping

def build_alphabet_map(alphabet):
    """MPNN column j -> ESM vocab index. Built by lookup, never hardcoded.

    ESM-IF1's vocab order is
      <null_0> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <mask> <cath> <af2>
    which shares no prefix with 'ACDEFGHIKLMNPQRSTVWYX'. Getting this wrong
    silently produces ~5% recovery, so it is asserted below.
    """
    idx = []
    for aa in MPNN_ALPHABET:
        i = alphabet.get_idx(aa)
        if i == alphabet.unk_idx and aa != "<unk>":
            raise ValueError("amino acid %r missing from ESM alphabet" % aa)
        idx.append(i)
    idx = np.asarray(idx, dtype=np.int64)
    if len(set(idx.tolist())) != len(MPNN_ALPHABET):
        raise ValueError("alphabet map is not injective: %r" % idx.tolist())
    return idx


# ------------------------------------------------------------- featurising

def chain_blocks(cx):
    """[(chain_id, global_indices, coords[L,3,3] as N/CA/C, seq_str), ...] in cx order."""
    out, seen = [], []
    for c in cx.chains:
        if c not in seen:
            seen.append(c)
    for c in seen:
        gi = np.flatnonzero(cx.chains == c)
        crd = np.stack([cx.N[gi], cx.CA[gi], cx.C[gi]], axis=1).astype(np.float32)
        out.append((c, gi, crd, "".join(cx.seq[gi])))
    return out


def _concat(blocks, order, padding_length=10):
    """Concatenate coords in `order` (list of block positions) with NaN linkers."""
    pad = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    parts = [blocks[order[0]][2]]
    for k in order[1:]:
        parts.append(pad)
        parts.append(blocks[k][2])
    return np.concatenate(parts, axis=0)


def _decode_logprobs(model, alphabet, all_coords, seq, aamap, device,
                     renormalise=True):
    """One teacher-forced pass. Returns ([len(seq), 21] log-probs, off-alphabet mass).

    Mirrors esm.inverse_folding.util.get_sequence_loss exactly, but keeps the
    logits instead of collapsing them to a cross-entropy.
    """
    from esm.inverse_folding.util import CoordBatchConverter

    bc = CoordBatchConverter(alphabet)
    coords, confidence, _, tokens, padding_mask = bc([(all_coords, None, seq)], device=device)

    # prepend_bos=True, append_eos=False  =>  tokens = [<cath>] + seq, length L+1.
    # logits[:, :, i] therefore predicts seq[i] given <cath> + seq[:i].
    prev_output_tokens = tokens[:, :-1]
    with torch.no_grad():
        logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    lsm = F.log_softmax(logits.float(), dim=1)[0].transpose(0, 1)   # [L, vocab]
    assert lsm.shape[0] == len(seq), (lsm.shape, len(seq))

    sub = lsm[:, torch.as_tensor(aamap, device=lsm.device)]          # [L, 21]
    kept = torch.logsumexp(sub, dim=1)
    off_mass = float((1.0 - kept.exp()).clamp(min=0).mean())
    if renormalise:
        sub = sub - kept[:, None]        # proper 21-way distribution, MPNN semantics
    return sub.cpu().numpy(), off_mass


# --------------------------------------------------------------- public API

def esmif_conditional_logprobs(model, alphabet, cx, seeds=(0,), device="cpu",
                               padding_length=10, renormalise=True,
                               return_diag=False):
    """Teacher-forced conditional log-probs for every residue of the complex.

    Returns [n_orders, L, 21] in fc.MPNN_ALPHABET column order, where n_orders =
    len(seeds) and each "order" is a permutation of the NON-target chains (the
    target chain is always first, per the official multichain protocol).

    One forward pass per (chain x order). A 2-chain complex with seeds=(0,) is
    therefore 2 forward passes.
    """
    aamap = build_alphabet_map(alphabet)
    blocks = chain_blocks(cx)
    nb = len(blocks)
    out = np.full((len(seeds), cx.n, 21), np.nan, dtype=np.float64)
    diags = []

    for oi, sd in enumerate(seeds):
        rng = np.random.RandomState(int(sd))
        for bi, (cid, gi, _crd, seq) in enumerate(blocks):
            others = [k for k in range(nb) if k != bi]
            if len(others) > 1:
                others = list(rng.permutation(others))
            order = [bi] + others
            all_coords = _concat(blocks, order, padding_length)
            lp, off = _decode_logprobs(model, alphabet, all_coords, seq, aamap,
                                       device, renormalise)
            out[oi, gi, :] = lp
            diags.append((cid, off))

    assert not np.isnan(out).any(), "some positions were never scored"
    return (out, diags) if return_diag else out


def whole_complex_logprobs(model, alphabet, cx, chain_order=None, device="cpu",
                           padding_length=10, renormalise=True, linker_tok="<pad>"):
    """Variant: ONE causal pass over all chains concatenated, so a residue is also
    conditioned on the native sequence of every chain placed before it.

    This is the closer analogue of ProteinMPNN's conditioning (which sees partner
    sequence), but it is OFF-PROTOCOL: the official ESM-IF1 multichain code never
    feeds partner sequence. Use it as a sensitivity check, not as the headline
    number. Chains after the first are conditioned on real partner sequence;
    the first chain is not conditioned on anything but backbone.
    """
    from esm.inverse_folding.util import CoordBatchConverter

    aamap = build_alphabet_map(alphabet)
    blocks = chain_blocks(cx)
    order = list(range(len(blocks))) if chain_order is None else list(chain_order)

    all_coords = _concat(blocks, order, padding_length)
    # Sequence must stay 1:1 aligned with the coordinate axis, so linkers get a token.
    toks, gidx = [], []
    for n, k in enumerate(order):
        if n > 0:
            toks += [linker_tok] * padding_length
            gidx += [-1] * padding_length
        toks += list(blocks[k][3])
        gidx += list(blocks[k][1])
    gidx = np.asarray(gidx)

    bc = CoordBatchConverter(alphabet)
    coords, confidence, _, _, padding_mask = bc([(all_coords, None, None)], device=device)
    ids = [alphabet.get_idx(t) for t in toks]
    tokens = torch.tensor([[alphabet.get_idx("<cath>")] + ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model.forward(coords, padding_mask, confidence, tokens[:, :-1])
    lsm = F.log_softmax(logits.float(), dim=1)[0].transpose(0, 1)
    sub = lsm[:, torch.as_tensor(aamap)]
    if renormalise:
        sub = sub - torch.logsumexp(sub, dim=1, keepdim=True)
    sub = sub.cpu().numpy()

    out = np.full((1, cx.n, 21), np.nan)
    keep = gidx >= 0
    out[0, gidx[keep], :] = sub[keep]
    assert not np.isnan(out).any()
    return out
