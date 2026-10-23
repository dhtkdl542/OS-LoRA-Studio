"""End-to-end check of the MOSS port: encode a prompt, sample with the int8 Base DiT, decode.

    venv\\Scripts\\python -m tests.smoke_moss_sample [out.png]
"""

import gc
import sys
import time

import torch

from oslora.models.moss.adapter import MossPaths, MossTextEncoder, MossVAE, load_dit, sample


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "smoke_moss.png"
    paths = MossPaths()
    prompt = (
        "A photo of a young woman with long black hair sitting in a sunlit cafe, holding a cup of "
        "coffee, soft natural window light, shallow depth of field."
    )
    t0 = time.time()
    te = MossTextEncoder(paths.text_encoder)
    print(f"TE loaded {time.time()-t0:.1f}s, vram {torch.cuda.memory_allocated()/2**30:.2f} GB")
    cond, uncond = te.encode([prompt, ""])
    print("embeds", cond.shape, uncond.shape, float(cond.float().std()))
    te.unload()
    del te
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    dit = load_dit(paths.dit)
    print(f"DiT loaded {time.time()-t0:.1f}s, vram {torch.cuda.memory_allocated()/2**30:.2f} GB")
    t0 = time.time()
    lat = sample(dit, cond, uncond, 768, 768, steps=int(sys.argv[2]) if len(sys.argv) > 2 else 24, seed=42)
    print(f"sampled {time.time()-t0:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    vae = MossVAE(paths.vae)
    vae.decode(lat)[0].save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
