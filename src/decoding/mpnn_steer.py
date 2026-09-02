"""Minimal steering layer over the RELEASED ProteinMPNN sampler.

Two capabilities needed for the KL-guided decoding experiment:

  (1) INJECTED DECODING ORDER  -- needs NO source change at all.
      protein_mpnn_utils.ProteinMPNN.sample() computes
          decoding_order = argsort((chain_mask + 1e-4) * |randn|)
      and `randn` is a free user-supplied argument used for NOTHING else.
      So setting
          randn[b, pos] = rank_of(pos) / (chain_mask[b, pos] + 1e-4)
      makes (chain_mask + 1e-4) * |randn| == rank exactly, hence
      argsort(...) == the requested permutation. Works for the general case
      where some positions are fixed (chain_mask == 0): ProteinMPNN forces
      those first anyway, and this encoding is consistent with that.

  (2) PER-POSITION TEMPERATURE -- DOES need a patched sample().
      In released sample(), `temperature` is a python scalar used in
          logits = W_out(h_V_t) / temperature
          probs  = softmax(logits - omit*1e8 + bias_AA/temperature
                                             + bias_by_res_t/temperature)
      h_V_t is [B, hidden] and the decoded position t = decoding_order[:, t_]
      varies per batch row, so no scalar/[B,1] argument can express a
      per-POSITION temperature. sample_ptemp() below is a verbatim copy of
      sample() with exactly three lines changed (marked ### PATCH), gathering
      a [B, L] temperature tensor at the decoded position each step.

Neither function modifies any project or ProteinMPNN file.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.environ.get("PROTEINMPNN_DIR", os.path.expanduser("~/ftax/ProteinMPNN"))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from protein_mpnn_utils import cat_neighbors_nodes, gather_nodes  # noqa: E402


# --------------------------------------------------------------- (1) order

def order_to_randn(order, chain_mask=None):
    """[B,L] long decoding order -> [B,L] float `randn` reproducing it in sample().

    order[b, t] = index of the position decoded at step t for batch row b.
    """
    order = torch.as_tensor(order, dtype=torch.long)
    if order.dim() == 1:
        order = order[None]
    B, L = order.shape
    rank = torch.empty(B, L, dtype=torch.float32)
    steps = torch.arange(1, L + 1, dtype=torch.float32).expand(B, L)
    rank.scatter_(1, order, steps)                 # rank[b, order[b,t]] = t+1
    if chain_mask is None:
        chain_mask = torch.ones(B, L)
    return rank / (chain_mask.float() + 1e-4)


def check_order_roundtrip(order, chain_mask=None):
    """Reproduce sample()'s own argsort and confirm it returns `order`."""
    order = torch.as_tensor(order, dtype=torch.long)
    if order.dim() == 1:
        order = order[None]
    cm = torch.ones_like(order, dtype=torch.float32) if chain_mask is None else chain_mask.float()
    randn = order_to_randn(order, cm)
    got = torch.argsort((cm + 0.0001) * torch.abs(randn))
    return bool((got == order).all()), got


# --------------------------------------------- (2) per-position temperature

def sample_ptemp(model, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx,
                 mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None,
                 chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None,
                 pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None,
                 pssm_bias_flag=None, bias_by_res=None):
    """Verbatim ProteinMPNN.sample() except `temperature` may be [B, L].

    Scalar temperature reproduces model.sample() bit-exactly (tested).
    """
    device = X.device
    E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
    h_E = model.W_e(E)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

    chain_mask = chain_mask * chain_M_pos * mask
    decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))
    mask_size = E_idx.shape[1]
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
    order_mask_backward = torch.einsum(
        'ij, biq, bjp->bqp',
        (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
        permutation_matrix_reverse, permutation_matrix_reverse)
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1. - mask_attend)

    N_batch, N_nodes = X.size(0), X.size(1)
    all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32)
    h_S = torch.zeros_like(h_V, device=device)
    S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
    h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device)
                         for _ in range(len(model.decoder_layers))]
    constant = torch.tensor(omit_AAs_np, device=device)
    constant_bias = torch.tensor(bias_AAs_np, device=device)
    omit_AA_mask_flag = omit_AA_mask is not None

    ### PATCH 1/3: accept scalar or [B, L] temperature
    temp_is_vec = torch.is_tensor(temperature) and temperature.dim() == 2
    if temp_is_vec:
        temperature = temperature.to(device=device, dtype=torch.float32)
        if temperature.shape[0] == 1 and N_batch > 1:
            temperature = temperature.expand(N_batch, -1)

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    for t_ in range(N_nodes):
        t = decoding_order[:, t_]
        chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
        mask_gathered = torch.gather(mask, 1, t[:, None])
        bias_by_res_gathered = torch.gather(
            bias_by_res, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

        ### PATCH 2/3: temperature at the position being decoded, [B,1]
        temp_t = torch.gather(temperature, 1, t[:, None]) if temp_is_vec else temperature

        if (mask_gathered == 0).all():
            S_t = torch.gather(S_true, 1, t[:, None])
        else:
            E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(
                1, 1, h_E.shape[-2], h_E.shape[-1]))
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:, None, None, None].repeat(
                1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
            mask_t = torch.gather(mask, 1, t[:, None])
            for l, layer in enumerate(model.decoder_layers):
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(
                    1, 1, h_V_stack[l].shape[-1]))
                h_ESV_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(
                    1, 1, mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                h_V_stack[l + 1].scatter_(1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                                          layer(h_V_t, h_ESV_t, mask_V=mask_t))
            h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(
                1, 1, h_V_stack[-1].shape[-1]))[:, 0]

            ### PATCH 3/3: the only arithmetic change -- temp_t replaces temperature
            logits = model.W_out(h_V_t) / temp_t
            probs = F.softmax(logits - constant[None, :] * 1e8
                              + constant_bias[None, :] / temp_t
                              + bias_by_res_gathered / temp_t, dim=-1)

            if pssm_bias_flag:
                pssm_coef_gathered = torch.gather(pssm_coef, 1, t[:, None])[:, 0]
                pssm_bias_gathered = torch.gather(
                    pssm_bias, 1, t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]))[:, 0]
                probs = ((1 - pssm_multi * pssm_coef_gathered[:, None]) * probs
                         + pssm_multi * pssm_coef_gathered[:, None] * pssm_bias_gathered)
            if pssm_log_odds_flag:
                g = torch.gather(pssm_log_odds_mask, 1, t[:, None, None].repeat(
                    1, 1, pssm_log_odds_mask.shape[-1]))[:, 0]
                pm = probs * g
                pm += probs * 0.001
                probs = pm / torch.sum(pm, dim=-1, keepdim=True)
            if omit_AA_mask_flag:
                g = torch.gather(omit_AA_mask, 1, t[:, None, None].repeat(
                    1, 1, omit_AA_mask.shape[-1]))[:, 0]
                pm = probs * (1.0 - g)
                probs = pm / torch.sum(pm, dim=-1, keepdim=True)
            S_t = torch.multinomial(probs, 1)
            all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21),
                               (chain_mask_gathered[:, :, None, ] * probs[:, None, :]).float())
        S_true_gathered = torch.gather(S_true, 1, t[:, None])
        S_t = (S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)).long()
        temp1 = model.W_s(S_t)
        h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)
        S.scatter_(1, t[:, None], S_t)
    return {"S": S, "probs": all_probs, "decoding_order": decoding_order}


# ------------------------------------------------------------------ driver

def draw(model, cx, K, batch, order=None, temperature=0.1, seed=0, use_patch=False,
         featurize=None, bias_by_res=None):
    """K samples. `order`: None (MPNN default random) or [L] / [B,L] permutation.

    `bias_by_res`: None (unbiased, default) or a [L,21] / [B,L,21] tensor added to the per-position logits
    before the softmax (this is how the CFG tilt +alpha*L is injected). Backward-compatible."""
    dev = next(model.parameters()).device                 # run on the model's device (CPU unless ported to GPU)
    X, S, mask, residue_idx, chain_enc = featurize(cx, device=dev)
    L = cx.n
    ALPHA = "ACDEFGHIKLMNPQRSTVWYX"
    omit = np.array([1.0 if a == "X" else 0.0 for a in ALPHA], dtype=np.float32)
    bias = np.zeros(21, dtype=np.float32)
    out, orders, done = [], [], 0
    torch.manual_seed(seed)
    with torch.no_grad():
        while done < K:
            b = min(batch, K - done)
            cm = torch.ones(b, L, device=dev)
            if order is None:
                rnd = torch.randn(b, L, device=dev)
            else:
                o = torch.as_tensor(order, dtype=torch.long)
                o = o[None].repeat(b, 1) if o.dim() == 1 else o[:b]
                rnd = order_to_randn(o, cm).to(dev)
            if bias_by_res is None:
                bbr = torch.zeros(b, L, 21, device=dev)
            else:
                _t = torch.as_tensor(bias_by_res, dtype=torch.float32, device=dev)
                bbr = _t[None].expand(b, -1, -1).contiguous() if _t.dim() == 2 else _t[:b]
            kw = dict(X=X.repeat(b, 1, 1, 1), randn=rnd, S_true=S.repeat(b, 1),
                      chain_mask=cm, chain_encoding_all=chain_enc.repeat(b, 1),
                      residue_idx=residue_idx.repeat(b, 1), mask=mask.repeat(b, 1),
                      temperature=temperature, omit_AAs_np=omit, bias_AAs_np=bias,
                      chain_M_pos=torch.ones(b, L, device=dev), omit_AA_mask=None, pssm_coef=None,
                      pssm_bias=None, pssm_multi=0.0, pssm_log_odds_flag=False,
                      pssm_log_odds_mask=None, pssm_bias_flag=False,
                      bias_by_res=bbr)
            d = sample_ptemp(model, **kw) if use_patch else model.sample(**kw)
            out.append(d["S"].cpu().numpy())
            orders.append(d["decoding_order"].cpu().numpy())
            done += b
    return np.concatenate(out, 0), np.concatenate(orders, 0)
