import torch, os, resource, time
src = "/tmp/scratch_models/torch_home/hub/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
dst = "/tmp/scratch_models/esm_if1_slim.pt"
t0 = time.time()
d = torch.load(src, map_location="cpu")
print("top-level keys:", sorted(d.keys()))
print("args.arch =", getattr(d["args"], "arch", None))
nm = sum(v.numel() for v in d["model"].values() if hasattr(v, "numel"))
print("model tensors: %d, params %.1fM (%.0f MB fp32)" % (len(d["model"]), nm/1e6, nm*4/1e6))
slim = {"args": d["args"], "model": d["model"]}
for k in list(d.keys()):
    if k not in ("args", "model"):
        d.pop(k)
torch.save(slim, dst)
print("wrote %s  %.0f MB   load+save %.0fs   peakRSS %.2f GB"
      % (dst, os.path.getsize(dst)/1e6, time.time()-t0,
         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
