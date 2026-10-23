"""Render one Krea 2 image end to end (text encoder -> Turbo DiT -> VAE) and report time / VRAM.

    venv\\Scripts\\python.exe tests\\smoke_krea2_sample.py [--base]
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oslora.config import RunConfig  # noqa: E402
from oslora.models import get_adapter  # noqa: E402

PROMPT = ("A vertical phone photo of a young Korean woman with long black hair and wispy bangs standing on a city "
          "sidewalk at dusk, wearing an oversized cream knit cardigan, soft natural light, slight smile.")


def main():
    base = "--base" in sys.argv
    cfg = RunConfig(model="krea2")
    ad = get_adapter("krea2")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    te = ad.text_encoder(cfg.paths.text_encoder)
    cond, uncond = te.encode([PROMPT, ""])
    te.unload()
    del te
    torch.cuda.empty_cache()
    print(f"text: {tuple(cond.shape)} in {time.time() - t0:.1f}s", flush=True)

    path = cfg.paths.dit if base else cfg.paths.inference_dit
    t0 = time.time()
    dit = ad.load_dit(path)
    print(f"DiT {os.path.basename(path)} loaded in {time.time() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB",
          flush=True)
    torch.cuda.reset_peak_memory_stats()
    steps, cfg_scale = (ad.base_sample_steps, ad.base_sample_cfg) if base else (ad.sample_steps, ad.sample_cfg)
    t0 = time.time()
    lat = ad.sample(dit, cond, uncond, 768, 1024, steps=steps, cfg=cfg_scale, seed=42)
    torch.cuda.synchronize()
    print(f"sample {steps} steps cfg {cfg_scale}: {time.time() - t0:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
          flush=True)
    del dit
    torch.cuda.empty_cache()
    vae = ad.vae(cfg.paths.vae)
    img = vae.decode(lat)[0]
    p = os.path.join(out_dir, f"krea2_{'base' if base else 'turbo'}.jpg")
    img.save(p, quality=92)
    print("saved", p)


if __name__ == "__main__":
    main()
