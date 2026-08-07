"""Mechanics tests for injected decoding order + per-position temperature."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/c/Users/chris/Desktop/python_projects/personal_projects/factorization-tax/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftax_common as fc          # noqa: E402
import mpnn_steer as ms           # noqa: E402

torch.set_num_threads(4)
DATA = os.path.expanduser("~/ftax/data/PDBs")
W = os.path.expanduser("~/ftax/ProteinMPNN/vanilla_model_weights/v_48_020.pt")
ALPHA = fc.MPNN_ALPHABET
OMIT = np.array([1.0 if a == "X" else 0.0 for a in ALPHA], dtype=np.float32)
BIAS = np.zeros(21, dtype=np.float32)
ok_all = True


def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")


def kw_for(cx, b, randn, temperature):
    X, S, mask, residue_idx, chain_enc = fc.featurize(cx)
    L = cx.n
    return dict(X=X.repeat(b, 1, 1, 1), randn=randn, S_true=S.repeat(b, 1),
                chain_mask=torch.ones(b, L), chain_encoding_all=chain_enc.repeat(b, 1),
                residue_idx=residue_idx.repeat(b, 1), mask=mask.repeat(b, 1),
                temperature=temperature, omit_AAs_np=OMIT, bias_AAs_np=BIAS,
                chain_M_pos=torch.ones(b, L), omit_AA_mask=None, pssm_coef=None,
                pssm_bias=None, pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=None, pssm_bias_flag=False,
                bias_by_res=torch.zeros(b, L, 21))


print("=" * 78)
print("T1  order_to_randn roundtrip (pure arithmetic, no model)")
rng = np.random.default_rng(0)
for L in (50, 96, 200, 442, 1000, 1700):
    o = torch.stack([torch.from_numpy(rng.permutation(L)) for _ in range(4)]).long()
    ok, got = ms.check_order_roundtrip(o)
    report(f"L={L:<5} B=4 exact permutation recovered", ok)
# with fixed positions (chain_mask == 0)
L = 200
o = torch.from_numpy(rng.permutation(L))[None].long()
cm = torch.ones(1, L)
cm[0, o[0, :20]] = 0.0                       # first 20 decoded are the fixed ones
ok, got = ms.check_order_roundtrip(o, cm)
report("L=200 with 20 fixed positions (chain_mask=0)", ok)

cx = fc.load_complex(os.path.join(DATA, "3EQY.pdb"), "3EQY", "A", "C")
cx2 = fc.load_complex(os.path.join(DATA, "1BRS.pdb"), "1BRS", "A", "D")
print(f"\nfixtures: 3EQY_A_C L={cx.n}   1BRS_A_D L={cx2.n}")
model, _ = fc.load_mpnn(W)

print("=" * 78)
print("T2  INJECTED ORDER reproduces model.sample() BIT-EXACTLY (no source change)")
B = 4
torch.manual_seed(1234)
r0 = torch.randn(B, cx.n)
torch.manual_seed(7)
with torch.no_grad():
    d_a = model.sample(**kw_for(cx, B, r0, 0.1))
order_a = d_a["decoding_order"]
r_inj = ms.order_to_randn(order_a, torch.ones(B, cx.n))
torch.manual_seed(7)
with torch.no_grad():
    d_b = model.sample(**kw_for(cx, B, r_inj, 0.1))
report("decoding_order identical", bool((d_b["decoding_order"] == order_a).all()))
report("sampled S bit-identical", bool((d_b["S"] == d_a["S"]).all()),
       f"(max|dS|={int((d_b['S']-d_a['S']).abs().max())})")
report("all_probs bit-identical", bool(torch.equal(d_b["probs"], d_a["probs"])),
       f"(max|dP|={float((d_b['probs']-d_a['probs']).abs().max()):.3e})")

print("=" * 78)
print("T3  ARBITRARY order (hotspot-first style) is honoured")
want = torch.stack([torch.arange(cx.n).flip(0),                       # reversed
                    torch.arange(cx.n),                               # N->C
                    torch.from_numpy(np.random.default_rng(3).permutation(cx.n)),
                    torch.from_numpy(np.concatenate([np.arange(10, 20), np.setdiff1d(
                        np.arange(cx.n), np.arange(10, 20))]))]).long()
torch.manual_seed(7)
with torch.no_grad():
    d_c = model.sample(**kw_for(cx, B, ms.order_to_randn(want, torch.ones(B, cx.n)), 0.1))
report("returned decoding_order == requested (incl. per-row different orders)",
       bool((d_c["decoding_order"] == want).all()))

print("=" * 78)
print("T4  PATCHED sampler reproduces released sample() BIT-EXACTLY (scalar T)")
torch.manual_seed(7)
with torch.no_grad():
    d_p = ms.sample_ptemp(model, **kw_for(cx, B, r0, 0.1))
report("S bit-identical to model.sample()", bool((d_p["S"] == d_a["S"]).all()))
report("probs bit-identical to model.sample()", bool(torch.equal(d_p["probs"], d_a["probs"])))

print("=" * 78)
print("T5  PER-POSITION temperature: constant vector == scalar, bit-exact")
tvec = torch.full((1, cx.n), 0.1)
torch.manual_seed(7)
with torch.no_grad():
    d_v = ms.sample_ptemp(model, **kw_for(cx, B, r0, tvec))
report("constant [1,L] vector T == scalar T, S bit-identical",
       bool((d_v["S"] == d_a["S"]).all()))
report("constant [1,L] vector T == scalar T, probs bit-identical",
       bool(torch.equal(d_v["probs"], d_a["probs"])))

print("=" * 78)
print("T6  PER-POSITION temperature actually acts, and ONLY where told")
hi = np.arange(0, cx.n, 5)                          # 20% of positions get T=1.0
tvec2 = torch.full((1, cx.n), 0.1)
tvec2[0, hi] = 1.0
Kb = 24
S_lo, _ = ms.draw(model, cx, Kb, Kb, order=None, temperature=0.1, seed=11,
                  use_patch=True, featurize=fc.featurize)
S_hi, _ = ms.draw(model, cx, Kb, Kb, order=None, temperature=tvec2, seed=11,
                  use_patch=True, featurize=fc.featurize)
ent_lo = np.array([len(set(S_lo[:, j])) for j in range(cx.n)])
ent_hi = np.array([len(set(S_hi[:, j])) for j in range(cx.n)])
d_hot = (ent_hi - ent_lo)[hi].mean()
d_cold = np.delete(ent_hi - ent_lo, hi).mean()
report("distinct-AA count rises at raised-T positions", d_hot > 0.5,
       f"(hot +{d_hot:.2f} vs cold {d_cold:+.2f} distinct AAs / {Kb} draws)")
report("cold positions ~unchanged", abs(d_cold) < 0.5 * abs(d_hot) + 0.3,
       "(leakage via context is expected but small)")

print("=" * 78)
print("T7  WALL-CLOCK on this CPU (torch threads = %d)" % torch.get_num_threads())
for c, name in ((cx, "3EQY_A_C"), (cx2, "1BRS_A_D")):
    for bsz in (1, 8, 24):
        n = max(bsz, 8)
        t0 = time.time()
        ms.draw(model, c, n, bsz, order=None, temperature=0.1, seed=1,
                use_patch=False, featurize=fc.featurize)
        dt = (time.time() - t0) / n
        print(f"    {name} L={c.n:4d}  batch={bsz:3d}  {dt*1000:7.1f} ms/sample "
              f"({dt*100:6.1f} s per K=100)")

import resource
print("\npeak RSS: %.2f GB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
print("=" * 78)
print("ALL PASS" if ok_all else "SOME TESTS FAILED")
