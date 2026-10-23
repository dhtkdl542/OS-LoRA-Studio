"""Checkpoint comparison: every chosen checkpoint (plus the base model) on the same prompts and
seeds, each image scored for likeness (ArcFace vs. the training faces) and prompt adherence (CLIP).

    python -m oslora.compare <run_dir> <request.json>

request.json: {"prompts": [...], "checkpoints": ["base", "best", 500, 1000, ...], "seed": 42,
               "strength": 1.0, "width": 768, "height": 1024, "steps": 24}
Writes <run_dir>/compare/<id>/result.json (updated after every image) and the images.
"""

import json
import os
import re
import sys
import time
import traceback

import torch

from . import fsutil
from .config import load_config
from .data import AUG_PREFIX, LatentCache, apply_trigger, pick_holdout, scan_folder
from .lora import LoRANetwork
from .models import get_adapter
from .train import ROOT, flush


def find_checkpoints(run_dir: str, name: str):
    """[(label, step or None, path)], oldest first; "best" is the recommended file."""
    out = []
    try:
        with open(os.path.join(run_dir, "best_marks.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        raw = {}
    labels = {"likeness": "닮음 최고", "val": "홀드아웃 최저"}
    marks = {}
    for metric, v in raw.items():
        if isinstance(v, dict) and "step" in v:
            marks[int(v["step"])] = " · ".join(filter(None, [marks.get(int(v["step"])), labels.get(metric, metric)]))
    for f in os.listdir(run_dir):
        m = re.fullmatch(rf"{re.escape(name)}_(\d{{5}})\.safetensors", f)
        if m:
            s = int(m.group(1))
            tag = marks.get(s)
            out.append((f"step {s}" + (f" ({tag})" if tag else ""), s, os.path.join(run_dir, f)))
    out.sort(key=lambda r: r[1])
    best = os.path.join(run_dir, f"{name}.safetensors")
    if os.path.isfile(best):
        out.append(("추천", None, best))
    return out


def main():
    run_dir, req_path = sys.argv[1], sys.argv[2]
    with open(req_path, encoding="utf-8") as f:
        req = json.load(f)
    out_dir = os.path.dirname(req_path)
    res_path = os.path.join(out_dir, "result.json")
    cfg = load_config(os.path.join(run_dir, "config.yaml"))
    model = get_adapter(cfg.model)
    prompts = [apply_trigger(p, cfg.dataset.trigger) for p in req["prompts"]]

    available = {("best" if s is None else s): (label, s, p) for label, s, p in find_checkpoints(run_dir, cfg.name)}
    wanted = req.get("checkpoints") or ["base"] + list(available)
    columns = []
    for w in wanted:
        if w == "base":
            columns.append({"label": "베이스 (LoRA 없음)", "step": 0, "file": None})
        else:
            key = "best" if w == "best" else int(w)
            if key in available:
                label, s, p = available[key]
                columns.append({"label": label, "step": s, "file": os.path.basename(p)})
    # render on the model people generate with (SFT) unless asked for the training model
    use_base = req.get("model") == "base" or not (cfg.paths.inference_dit or "").strip()         or not os.path.isfile(cfg.paths.inference_dit)
    dit_path = cfg.paths.dit if use_base else cfg.paths.inference_dit
    result = {"prompts": req["prompts"], "columns": columns, "seed": req.get("seed", 42),
              "model": "base" if use_base else "sft", "model_file": os.path.basename(dit_path),
              "strength": req.get("strength", 1.0), "status": "preparing", "message": "준비 중",
              "cells": [[None] * len(columns) for _ in prompts], "started": time.time()}

    def save(**kw):
        result.update(kw)
        tmp = res_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        fsutil.replace(tmp, res_path, required=False)

    try:
        save(message="프롬프트 인코딩")
        cache = LatentCache(os.path.join(ROOT, "cache", cfg.model), cfg.model)
        texts = list(dict.fromkeys(prompts + [""]))
        missing = [t for t in texts if cache.get_text(t) is None]
        if missing:
            te = model.text_encoder(cfg.paths.text_encoder, log=lambda m: save(message=m))
            for t in missing:
                cache.put_text(t, te.encode([t])[0])
            te.unload()
            del te
            flush()
        emb = {t: cache.get_text(t) for t in texts}

        save(message="모델 로딩")
        from .models import quant

        quant.INT8_COMPUTE = bool(getattr(cfg.train, "int8_compute", False))
        dit = model.load_dit(dit_path, log=lambda m: save(message=m))
        n = cfg.network
        net = LoRANetwork(dit, model.lora_targets.get(n.targets, model.lora_targets["all"]), rank=n.rank, alpha=n.alpha)
        net.train(False)
        model.block_offload(dit, log=lambda m: save(message=m))
        vae = model.vae(cfg.paths.vae)

        judge = None
        try:
            from .evaluate import Evaluator

            pairs = scan_folder(cfg.dataset.folder, cfg.dataset.caption_ext)
            hold_file = os.path.join(run_dir, "holdout.json")
            if os.path.isfile(hold_file):
                held = set(json.load(open(hold_file, encoding="utf-8")))
                hold = {i for i, (p, _) in enumerate(pairs) if os.path.basename(p) in held}
            else:
                hold = set(pick_holdout(len(pairs), cfg.dataset.holdout, cfg.train.seed))
            judge = Evaluator([p for i, (p, _) in enumerate(pairs) if i not in hold
                               and not os.path.basename(p).startswith(AUG_PREFIX)], log=lambda m: save(message=m))
        except Exception as e:
            save(message=f"평가자 없이 진행 ({e})")

        total = len(prompts) * len(columns)
        done, t0 = 0, time.time()
        w, h, steps = int(req.get("width", 768)), int(req.get("height", 1024)), int(req.get("steps", 24))
        strength = float(req.get("strength", 1.0))
        for ci, col in enumerate(columns):
            if col["file"]:
                from safetensors.torch import load_file

                net.load_state_dict(load_file(os.path.join(run_dir, col["file"])))
                net.set_multiplier(strength)
            else:
                net.set_multiplier(0.0)
            for pi, prompt in enumerate(prompts):
                eta = (time.time() - t0) / done * (total - done) if done else None
                save(status="running", message=f"{col['label']} · 프롬프트 {pi + 1}/{len(prompts)} 그리는 중",
                     done=done, total=total, eta=eta)
                lat = model.sample(dit, emb[prompt], emb[""], w, h, steps=steps, cfg=cfg.sample.cfg,
                                  seed=result["seed"] + pi)
                img = vae.decode(lat)[0]
                fname = f"c{ci:02d}_p{pi:02d}.jpg"
                img.save(os.path.join(out_dir, fname), quality=92)
                cell = {"img": fname}
                if judge is not None:
                    q = judge.score([img], [prompt])
                    cell.update(likeness=q.get("likeness"), adherence=q.get("adherence"))
                result["cells"][pi][ci] = cell
                done += 1
        for ci, col in enumerate(columns):  # per-column averages for the table header
            likes = [c["likeness"] for c in (row[ci] for row in result["cells"]) if c and c.get("likeness") is not None]
            adh = [c["adherence"] for c in (row[ci] for row in result["cells"]) if c and c.get("adherence") is not None]
            col["likeness"] = sum(likes) / len(likes) if likes else None
            col["adherence"] = sum(adh) / len(adh) if adh else None
        save(status="done", message=f"완료: {total}장 ({(time.time() - t0) / 60:.1f}분)", done=total, total=total, eta=0)
    except Exception as e:
        save(status="error", message=str(e), trace=traceback.format_exc()[-1500:])
        raise
    finally:
        from . import memclean

        memclean.release_process_memory()
        memclean.schedule_after_exit([dit_path, cfg.paths.text_encoder, cfg.paths.vae])


if __name__ == "__main__":
    main()
