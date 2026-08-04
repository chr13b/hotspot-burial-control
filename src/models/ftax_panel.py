"""One entry point for the multi-model panel.

    from ftax_panel import PANEL, load, score
    h  = load("esmif")                       # returns an opaque handle
    lp = score("esmif", h, cx)               # -> [n_orders, L, 21], MPNN alphabet

Every `score` returns the SAME object as `fc.mpnn_conditional_logprobs`:
a [n_orders, L, 21] float64 array of log-probabilities whose column order is
fc.MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX', normalised over those 21 columns.

`n_orders` means different things per model and this is not hidden:
  mpnn_*   autoregressive, n_orders = number of random decoding permutations
  esmif    causal L->R, NO permutable order; n_orders = permutations of the
           NON-target chains only (degenerate = 1 for a 2-chain complex)
  pifold   one-shot, n_orders is always 1 and the model is bit-exact deterministic
  mif      masked, n_orders = number of complementary random-mask pairs

`junction_flank` is the number of residues either side of an artificial chain
junction whose features are fabricated by that model, and which should be dropped
downstream. 0 means the model handles chains natively.
"""

import os
import sys
import numpy as np

sys.path.insert(0, "/mnt/c/Users/chris/Desktop/python_projects/personal_projects/factorization-tax/src")
import ftax_common as fc

MPNN_DIR = os.path.expanduser("~/ftax/ProteinMPNN")

PANEL = {
    # name          family                    params   junction_flank  native multichain
    "mpnn_vanilla": dict(family="GNN, autoregressive",  params=1.66e6, junction_flank=0, multichain=True),
    "mpnn_soluble": dict(family="GNN, autoregressive",  params=1.66e6, junction_flank=0, multichain=True),
    "esmif":        dict(family="GVP-transformer, causal", params=141.7e6, junction_flank=0, multichain=True),
    "pifold":       dict(family="GNN, one-shot",        params=6.61e6, junction_flank=2, multichain=False),
    "mif":          dict(family="GNN, masked",          params=3.44e6, junction_flank=2, multichain=False),
}


def load(name, device="cpu"):
    if name == "mpnn_vanilla":
        return fc.load_mpnn(os.path.join(MPNN_DIR, "vanilla_model_weights/v_48_020.pt"), device)[0]
    if name == "mpnn_soluble":
        return fc.load_mpnn(os.path.join(MPNN_DIR, "soluble_model_weights/v_48_020.pt"), device)[0]
    if name == "esmif":
        import ftax_esmif as fe
        return fe.load_esmif(device=device)
    if name == "pifold":
        import ftax_pifold as fp
        return fp.load_pifold(device=device)
    if name == "mif":
        import ftax_mif as fm
        return fm.load_mif(device=device)
    raise KeyError(name)


def score(name, handle, cx, n_orders=8, device="cpu"):
    """-> [n_orders_effective, L, 21] log-probs, fc.MPNN_ALPHABET column order."""
    if name.startswith("mpnn"):
        return fc.mpnn_conditional_logprobs(handle, cx, seeds=range(n_orders), device=device)
    if name == "esmif":
        import ftax_esmif as fe
        model, alphabet = handle
        return fe.esmif_conditional_logprobs(model, alphabet, cx, seeds=(0,), device=device)
    if name == "pifold":
        import ftax_pifold as fp
        return fp.pifold_conditional_logprobs(handle, cx, device=device)
    if name == "mif":
        import ftax_mif as fm
        model, collater = handle
        return fm.mif_conditional_logprobs(model, collater, cx, seeds=range(n_orders), device=device)
    raise KeyError(name)


def junction_mask(name, cx):
    """Residues whose features are corrupted by artificial chain concatenation."""
    flank = PANEL[name]["junction_flank"]
    if flank == 0:
        return np.zeros(cx.n, dtype=bool)
    import ftax_pifold as fp
    return fp.junction_mask(cx, flank=flank)


def verify_alphabet(name, handle):
    """Round-trip the model's own vocab through the MPNN column map. Raises on mismatch.

    Run this once per model per session. An alphabet slip is silent and costs the
    whole experiment; recovery drops to ~0.01-0.05 rather than erroring.
    """
    if name.startswith("mpnn"):
        return fc.MPNN_ALPHABET
    if name == "esmif":
        import ftax_esmif as fe
        _, alphabet = handle
        back = "".join(alphabet.get_tok(int(i)) for i in fe.build_alphabet_map(alphabet))
    elif name == "pifold":
        import ftax_pifold as fp
        back = "".join(fp.PIFOLD_ALPHABET[i] if i >= 0 else "X" for i in fp.build_alphabet_map())
    elif name == "mif":
        import ftax_mif as fm
        from sequence_models.constants import PROTEIN_ALPHABET
        back = "".join(PROTEIN_ALPHABET[i] for i in fm.build_alphabet_map())
    else:
        raise KeyError(name)
    if back != fc.MPNN_ALPHABET:
        raise AssertionError("%s alphabet map is WRONG: %r != %r" % (name, back, fc.MPNN_ALPHABET))
    return back
