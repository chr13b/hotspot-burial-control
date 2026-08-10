"""Experiment B part 3 (F4): unmasking-order sweep on MultiFlow monomer co-design.

BRIEF §2.3's prescription: decide the highest-influence discrete positions while the continuous channel
is still hot. The released MultiFlow checkpoint is monomer-trained, so (with the user's agreement) the
demanding-position proxy is BURIAL (buried = geometrically demanding; Exp A showed burial dominates).

Design — SDEdit with a self-generated reference (in-distribution; no external featurization):
 1. Generate a reference monomer by unconditional co-design -> frames (trans_1, rotmats_1) + ref seq.
 2. Per-position burial from the backbone (Cα neighbour count); label buried (top tertile) vs exposed.
 3. SDEdit: corrupt the reference to t0 (both channels via the interpolant's own corruption), then
    denoise t0->1 with a chosen discrete unmasking ORDER, continuous schedule fixed:
      purity (dynamic model confidence, the released default) / burial_first / anti_burial / random.
 4. recovery(order) at buried vs exposed; headline = recovery(buried) - recovery(exposed) per order.

KILL F4: a full sweep of orders moves buried-minus-exposed recovery by < seed-to-seed SD -> knob inert.

Run in the multiflow conda env on a V100/RTX_3090 (cu117):
  python src/expB_ordering.py --n-ref 6 --t0 0.5 --seeds 0,1 --length 128 --out results/expB_ordering.csv
"""
import argparse
import os
import sys

import numpy as np


def load_multiflow(mf_dir, ckpt, num_timesteps, device):
    sys.path.insert(0, mf_dir)
    import torch
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from multiflow.models.flow_module import FlowModule
    with initialize_config_dir(version_base=None, config_dir=os.path.join(mf_dir, "multiflow/configs")):
        base_cfg = compose(config_name="inference_unconditional")
    ckpt_cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), "config.yaml"))
    OmegaConf.set_struct(base_cfg, False); OmegaConf.set_struct(ckpt_cfg, False)
    cfg = OmegaConf.merge(base_cfg, ckpt_cfg)
    cfg.interpolant = base_cfg.inference.interpolant
    cfg.interpolant.sampling.num_timesteps = num_timesteps
    # corruption fields needed by _corrupt_* for SDEdit (absent from the inference-only interpolant)
    cfg.interpolant.trans.batch_ot = False
    cfg.interpolant.trans.train_schedule = "linear"
    cfg.interpolant.rots.train_schedule = cfg.interpolant.rots.get("sample_schedule", "exp")
    import multiflow.experiments.utils as eu
    try:
        dcfg = eu.get_dataset_cfg(cfg)
    except Exception:
        dcfg = cfg.get("pdb_post2021_dataset", None)
    fm = FlowModule.load_from_checkpoint(checkpoint_path=ckpt, cfg=cfg, dataset_cfg=dcfg,
                                         folding_cfg=None, map_location=device, strict=False)
    fm.eval().to(device)
    fm.interpolant.set_device(device)
    return fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mf-dir", default=os.path.expandvars("$SCRATCH/expB/multiflow"))
    ap.add_argument("--ckpt", default=os.path.expandvars("$SCRATCH/expB/multiflow/weights/last.ckpt"))
    ap.add_argument("--n-ref", type=int, default=6)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--t0", type=float, default=0.5)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--num-timesteps", type=int, default=200)
    ap.add_argument("--out", default="results/expB_ordering.csv")
    a = ap.parse_args()
    cmd = "python3 " + " ".join(sys.argv)

    import torch
    from multiflow.data import utils as du
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fm = load_multiflow(a.mf_dir, a.ckpt, a.num_timesteps, device)
    ip = fm.interpolant
    model = fm.model
    MASK = du.MASK_TOKEN_INDEX
    L = a.length
    res_mask = torch.ones(1, L, device=device)
    res_idx = torch.arange(L, device=device)[None].float()
    chain_idx = torch.ones(1, L, device=device)

    def frames_from_atom37(a37):
        N, CA, C = a37[:, 0], a37[:, 1], a37[:, 2]
        e1 = C - CA; e1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
        v2 = N - CA; u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
        e2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
        e3 = torch.cross(e1, e2, dim=-1)
        return CA, torch.stack([e1, e2, e3], dim=-1)   # trans [L,3], rot [L,3,3]

    def model_pred(trans_t, rot_t, aa_t, t):
        b = {"res_mask": res_mask, "diffuse_mask": res_mask, "chain_idx": chain_idx, "res_idx": res_idx,
             "trans_sc": torch.zeros(1, L, 3, device=device),
             "aatypes_sc": torch.zeros(1, L, ip.num_tokens, device=device),
             "trans_t": trans_t, "rotmats_t": rot_t, "aatypes_t": aa_t}
        tt = torch.ones(1, 1, device=device) * t
        b["r3_t"] = tt; b["cat_t"] = tt
        b["so3_t"] = ip.rot_sample_kappa(tt) if ip._cfg.provide_kappa else tt
        with torch.no_grad():
            return model(b)

    def denoise(trans0, rot0, aa0, order_mode, static_key, t0):
        ts = torch.linspace(t0, 1.0, a.num_timesteps, device=device)
        trans_t, rot_t, aa_t = trans0.clone(), rot0.clone(), aa0.clone()
        for i in range(len(ts) - 1):
            t1 = ts[i].item(); d_t = (ts[i + 1] - ts[i]).item()
            out = model_pred(trans_t, rot_t, aa_t, t1)
            trans_t = ip._trans_euler_step(d_t, t1, out["pred_trans"], trans_t)
            rot_t = ip._rots_euler_step(d_t, t1, out["pred_rotmats"], rot_t)
            probs = torch.softmax(out["pred_logits"][0, :, :20] / ip._aatypes_cfg.temp, dim=-1)
            is_mask = (aa_t[0] == MASK)
            n_mask = int(is_mask.sum().item())
            if n_mask == 0:
                continue
            rate = min(1.0, d_t * (1 + ip._aatypes_cfg.noise * t1) / max(1e-6, 1 - t1))
            n_un = int(np.random.binomial(n_mask, rate))
            if n_un == 0:
                continue
            key = torch.log(probs.max(-1).values + 1e-9) if order_mode == "purity" else static_key
            key = torch.where(is_mask, key, torch.full_like(key, -1e9))
            pick = torch.argsort(key, descending=True)[:n_un]
            samp = torch.multinomial(probs, 1).squeeze(-1)
            aa_t = aa_t.clone(); aa_t[0, pick] = samp[pick].to(aa_t.dtype)
        return aa_t[0]

    orders = ["purity", "burial_first", "anti_burial", "random"]
    seeds = [int(s) for s in a.seeds.split(",")]
    t0t = torch.ones(1, 1, device=device) * a.t0
    rows = []
    for seed in seeds:
        for r in range(a.n_ref):
            torch.manual_seed(1000 + 37 * seed + r); np.random.seed(1000 + 37 * seed + r)
            with torch.no_grad():
                prot_traj, _ = ip.sample(1, L, model, diffuse_mask=res_mask, res_idx=res_idx, chain_idx=chain_idx)
            a37 = prot_traj[-1][0][0].to(device).float()
            ref_aa = prot_traj[-1][1][0].to(device).long()
            trans1, rot1 = frames_from_atom37(a37)
            ca = a37[:, 1].detach().cpu().numpy()
            d = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
            nb = ((d < 10.0).sum(1) - 1).astype(float)
            buried = torch.tensor(nb >= np.quantile(nb, 2 / 3.0), device=device)
            exposed = torch.tensor(nb <= np.quantile(nb, 1 / 3.0), device=device)
            keys = {"burial_first": torch.tensor(nb, device=device).float(),
                    "anti_burial": torch.tensor(-nb, device=device).float(),
                    "random": None, "purity": None}
            for od in orders:
                torch.manual_seed(9 * (1000 + 37 * seed + r)); np.random.seed(9 * (1000 + 37 * seed + r))
                with torch.no_grad():
                    trans0 = ip._corrupt_trans(trans1[None], t0t, res_mask, res_mask)
                    rot0 = ip._corrupt_rotmats(rot1[None], t0t, res_mask, res_mask)
                    aa0 = ip._corrupt_aatypes(ref_aa[None], t0t, res_mask, res_mask)
                sk = keys[od]
                if sk is None and od == "random":
                    sk = torch.rand(L, device=device)
                rec = denoise(trans0, rot0, aa0, od, sk if sk is not None else torch.zeros(L, device=device), a.t0)
                match = (rec.long() == ref_aa)
                rb = float(match[buried].float().mean()); re = float(match[exposed].float().mean())
                rows.append(dict(seed=seed, ref=r, order=od, rec_buried=rb, rec_exposed=re, rec_diff=rb - re))
                print(f"  seed{seed} ref{r} {od:13s} buried={rb:.3f} exposed={re:.3f} diff={rb-re:+.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows).assign(command=cmd)
    df.to_csv(a.out, index=False)
    g = df.groupby("order")["rec_diff"].agg(["mean", "std", "count"])
    print("\n=== buried-minus-exposed recovery by order ===\n" + g.to_string())
    sd = df.groupby("seed")["rec_diff"].std().mean()
    span = float(g["mean"].max() - g["mean"].min())
    print(f"\n[F4] order-span={span:.3f}  seed-SD~{sd:.3f}  ->  "
          f"F4 {'FIRES (knob inert)' if span < sd else 'does NOT fire (order matters)'}")
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()
