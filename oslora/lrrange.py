"""LR range per model / rank / training adapter (docs/PRINCIPLES.md, 2: safe mode).

Question: up to which LR does a higher LR buy faster learning without paying for it?

A rising-LR sweep mixes the LR with the time already trained, so the LRs are compared from the
same point instead, everything paired:

  trunk     TRUNK steps at REF_LR, then a snapshot (weights + optimizer state)
  branches  from the snapshot, BRANCH steps per LR in LADDER, same seed (same batches, same noise),
            probed at the same steps (every EVAL)
  compare   each branch against the REF_LR branch at equal steps, averaged over the second half:
            extra fit gain (benefit), holdout difference and leak ratio (costs)

Equal steps matter: the leak swings with the images a stretch of training happens to see (in the
first measurement it fell by a third over 200 steps at 1e-4), and only branches that saw the same
images cancel that out. Probes read a short EMA (PROBE_EMA, ~10 steps) against step-to-step jitter.

Rule: an LR passes when, over the second half of its branch,
    it learns more:            extra fit gain > GAIN_MIN x the reference branch's own gain
    the holdout is no worse:   holdout difference <= VAL_NOISE (the reference branch's jitter)
    it leaks at most LEAK_MAX x as much as the reference
and the edge is the highest LR that passes with every lower one passing too. The rule was set
after the MOSS measurement (2026-10-29: 1.5e-4 and 2e-4 learned no faster than 1e-4 but leaked
more with a worse holdout -- the long-run record); it is adopted only if it also gives Qwen 2.1
with its training adapter an edge of 2e-4 or above (Fizgig's standard range 1e-4..2e-4).

Numbers only: no LoRA is saved. Results go to cache/lrrange/<combo>.json and ranges.json.

RESULT 2026-10-29 (Damdeari, 512 px, rank 32): NOT ADOPTED -- nothing reads ranges.json.
  MOSS, no adapter:   2e-4 -6 % gain, holdout +0.0008, leak 1.51x  -> edge 1e-4
  Qwen 2.1 + adapter: 2e-4 -11 %, 3e-4 -24 %, 4e-4 -31 % gain, holdout +0.0007..0.0012 -> edge 1e-4
Every higher-LR branch first lost ground and then caught up (Qwen 3e-4 passed the reference by step
450): the branches measure the shock of raising the LR on an optimizer tuned to 1e-4, not the merit
of training at that LR, so the MOSS agreement with the long-run record proves nothing either. A
valid comparison needs runs that use each LR from the start. And the benefit here is fit loss,
which does not see identity (docs/PRINCIPLES.md): "learns faster" in loss is not "looks like her
sooner", which is what the v5 / Fizgig v3 question is about.

    python -m oslora.lrrange <dataset> --model moss_image
    python -m oslora.lrrange <dataset> --model qwen21 --adapter
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch

from .config import RunConfig
from .data import BucketSampler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "lrrange")
REF_LR = 1e-4
LADDER = (1e-4, 2e-4, 3e-4, 4e-4)
TRUNK, BRANCH, EVAL = 300, 150, 25
PROBE_EMA = 0.9
GAIN_MIN, VAL_NOISE, LEAK_MAX = 0.1, 3e-4, 1.5
RESOLUTION = 512

Probe = Tuple[int, float, Optional[float], float]  # step, fit, val, leak


def combo_key(model: str, rank: int, adapter: bool) -> str:
    return f"{model}|r{rank}|{'adapter' if adapter else 'plain'}"


def make_config(folder: str, model: str, rank: int, adapter: bool, seed: int) -> RunConfig:
    name = "lrrange_" + combo_key(model, rank, adapter).replace("|", "_")
    cfg = RunConfig(name=name, model=model, output_dir=OUT)
    cfg.dataset.folder = folder
    cfg.dataset.resolution = RESOLUTION
    cfg.dataset.photo_texture, cfg.dataset.photo_tone = 0.25, False  # as the proxy experiments
    cfg.network.rank = cfg.network.alpha = rank
    cfg.network.training_adapter = "auto" if adapter else ""
    cfg.train.seed = seed
    cfg.train.ema_decay = PROBE_EMA
    cfg.autopilot.enabled = False
    cfg.autopilot.quality_judges = False
    cfg.sample.prompts, cfg.sample.every = [], 0
    return cfg


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


class LRRange:
    """One measurement for one combination on one dataset."""

    def __init__(self, folder: str, model: str, rank: int = 32, adapter: bool = False, seed: int = 42, log=print):
        self.folder, self.model, self.rank, self.adapter, self.seed = folder, model, rank, adapter, seed
        self.key = combo_key(model, rank, adapter)
        self.log = log
        self.cfg = make_config(folder, model, rank, adapter, seed)
        self.tr = None
        self.fit0 = None

    # ------------------------------------------------------------------ pieces
    def gain(self, fit: float) -> float:
        return 1.0 - fit / self.fit0

    def _probe(self, step: int) -> Probe:
        with self.tr.using_ema():
            fit, val, leak = self.tr.run_probes()
        return step, fit, val, leak

    def _reseed(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        return BucketSampler(self.tr.train_items + self.tr.extra_items, self.cfg.train.batch_size, seed)

    def _train(self, sampler, lr: float, steps: int, start: Probe):
        """Train ``steps`` at ``lr``, probing every EVAL steps. Returns the probes (``start``
        first) and whether training collapsed."""
        tr, curve, t0 = self.tr, [start], time.time()
        for s in range(1, steps + 1):
            for group in tr.opt.param_groups:
                group["lr"] = lr
            loss, gnorm, _ = tr.forward_backward(sampler, self.shift, self.fixed_shift)
            if not (math.isfinite(loss) and math.isfinite(gnorm)):
                tr.opt.zero_grad(set_to_none=True)
                self.log(f"  LR {lr:.1e}: step {s}에서 무너짐 (NaN/Inf)")
                return curve, True
            tr.apply_update()
            if s % EVAL == 0 or s == steps:
                p = self._probe(start[0] + s)
                curve.append(p)
                self.log(f"  LR {lr:.1e} step {p[0]}: 학습량 {self.gain(p[1]):.4f}, 누출 {p[3]:.3e}"
                         + (f", 홀드아웃 {p[2]:.4f}" if p[2] is not None else "")
                         + f" ({(time.time() - t0) / s:.1f}초/step)")
        return curve, False

    def compare(self, ref: List[Probe], curve: List[Probe], collapsed: bool) -> Dict:
        """A branch against the reference at equal steps, averaged over the second half."""
        start = ref[0]
        late = [(r, c) for r, c in zip(ref, curve) if r[0] - start[0] >= BRANCH / 2]
        ref_gain = self.gain(ref[-1][1]) - self.gain(start[1])
        if collapsed or not late:
            return {"extra_gain": None, "d_val": None, "leak_ratio": None, "pass": False, "why": "무너짐"}
        extra = mean([self.gain(c[1]) - self.gain(r[1]) for r, c in late])
        d_val = mean([c[2] - r[2] for r, c in late]) if start[2] is not None else 0.0
        leak_ratio = mean([c[3] for _, c in late]) / max(1e-12, mean([r[3] for r, _ in late]))
        why = []
        if extra <= GAIN_MIN * ref_gain:
            why.append("더 배우지 않음")
        if d_val > VAL_NOISE:
            why.append("홀드아웃 나빠짐")
        if leak_ratio > LEAK_MAX:
            why.append("누출 큼")
        return {"extra_gain": extra, "extra_frac": extra / ref_gain if ref_gain > 0 else None,
                "d_val": d_val, "leak_ratio": leak_ratio, "pass": not why, "why": ", ".join(why) or "통과"}

    # ------------------------------------------------------------------ measurement
    def run(self) -> Dict:
        from .train import Trainer

        t_all = time.time()
        run_dir = os.path.join(OUT, self.cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        self.tr = tr = Trainer(self.cfg, run_dir)
        try:
            tr.prepare_data()
            tr.cache_everything()
            tr.build_model()
            tr.build_probes()
            t = self.cfg.train
            bw, bh = max(tr.train_items, key=lambda it: it.bucket[0] * it.bucket[1]).bucket
            self.fixed_shift = t.timestep_shift not in ("auto", None, "")
            self.shift = float(t.timestep_shift) if self.fixed_shift else tr.model.timestep_shift(bw, bh)
            tr.net.train(True)

            p0 = self._probe(0)
            self.fit0 = p0[1]
            self.log(f"[LR 범위] {self.key}: 공통 구간 {TRUNK} step (LR {REF_LR:.0e})")
            trunk, broke = self._train(self._reseed(self.seed), REF_LR, TRUNK, p0)
            if broke:
                raise RuntimeError("공통 구간에서 학습이 무너졌어요")
            snap = tr.take_snapshot()
            tr.rollback(snap)  # the EMA restarts from the snapshot's weights: probe the branch point as such
            start = self._probe(TRUNK)

            branches = []
            for lr in LADDER:
                tr.rollback(snap)
                self.log(f"[LR 범위] 갈래 LR {lr:.1e} ({BRANCH} step)")
                curve, broke = self._train(self._reseed(self.seed + 1), lr, BRANCH, start)
                branches.append({"lr": lr, "collapsed": broke, "curve": curve})
            ref = branches[0]["curve"]
            for b in branches:
                b.update(self.compare(ref, b["curve"], b["collapsed"]))
            branches[0]["pass"], branches[0]["why"] = True, "기준"
            edge = None
            for b in branches:  # the highest LR with every lower one passing too
                if not b["pass"]:
                    break
                edge = b["lr"]
            res = {"combo": self.key, "dataset": self.folder, "seed": self.seed, "resolution": RESOLUTION,
                   "trunk": {"steps": TRUNK, "lr": REF_LR, "fit0": self.fit0, "end": start, "curve": trunk},
                   "branch_steps": BRANCH, "branches": branches, "edge": edge, "probe_ema": PROBE_EMA,
                   "rule": {"gain_min": GAIN_MIN, "val_noise": VAL_NOISE, "leak_max": LEAK_MAX},
                   "date": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t_all) / 60, 1)}
            self.save(res)
            self.log(self.report(res))
            return res
        finally:
            tr.release_memory()

    def save(self, res: Dict):
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, res["combo"].replace("|", "_") + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        path = os.path.join(OUT, "ranges.json")
        try:
            with open(path, encoding="utf-8") as f:
                ranges = json.load(f)
        except (OSError, ValueError):
            ranges = {}
        ranges[res["combo"]] = {"edge": res["edge"], "dataset": os.path.basename(os.path.normpath(res["dataset"])),
                                "date": res["date"], "validated": False}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ranges, f, ensure_ascii=False, indent=1)

    @staticmethod
    def report(res: Dict) -> str:
        lines = [f"[LR 범위] {res['combo']} ({res['minutes']}분, 갈래 {res['branch_steps']} step, 후반 평균을 1e-4와 비교)"]
        for b in res["branches"][1:]:
            if b.get("extra_gain") is None:
                lines.append(f"  LR {b['lr']:.1e}: 무너짐 -> 탈락")
                continue
            lines.append(f"  LR {b['lr']:.1e}: 학습량 {b['extra_frac']:+.0%}, 홀드아웃 {b['d_val']:+.4f}, "
                         f"누출 {b['leak_ratio']:.2f}배 -> {b['why']}")
        lines.append(f"  상한: {res['edge']:.1e}" if res["edge"] else "  상한: 없음")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="LR range for one model / rank / adapter combination")
    ap.add_argument("folder")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--adapter", action="store_true", help="train with the model's training adapter")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    LRRange(a.folder, a.model, a.rank, a.adapter, a.seed, log=lambda m: print(m, flush=True)).run()


if __name__ == "__main__":
    main()
