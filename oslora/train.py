"""LoRA trainer.  ``python -m oslora.train <run_config.yaml>``

Writes into ``<output_dir>/<name>/``:
    config.yaml, log.txt, status.json, metrics.jsonl, samples/, *.safetensors
Create a file named STOP in that folder to stop gracefully (best weights are still saved).
"""

import copy
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import fsutil
from .autopilot import Autopilot
from .config import (
    RunConfig,
    auto_eval_every,
    auto_max_steps,
    auto_timestep_shift,
    load_config,
    resolve_int,
    save_config,
)
from .data import (AUG_PREFIX, LENGTH_WEIGHTS, BucketSampler, Item, LatentCache, apply_trigger, bucket_for, crop_sources,
                   file_key, load_caption_variants, load_image, pick_holdout, scan_folder)
from .lora import EMA, LoRANetwork
from .models import get_adapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_T = (0.4, 0.75)
LEAK_T = (0.3, 0.6, 0.9)
LEAK_RES = 512


class RunLog:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.log_f = open(os.path.join(run_dir, "log.txt"), "a", encoding="utf-8")
        self.metrics_f = open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        self.status: Dict = {"phase": "starting"}

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def metric(self, **kw):
        self.metrics_f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.metrics_f.flush()

    def set_status(self, **kw):
        self.status.update(kw)
        now = time.time()
        # progress messages from tight loops (text encoding, VAE caching): at most twice a second
        if set(kw) <= {"message"} and now - getattr(self, "_status_written", 0) < 0.5:
            return
        self._status_written = now
        self.status["updated"] = now
        tmp = os.path.join(self.run_dir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.status, f, ensure_ascii=False)
        # the web server reads status.json every few seconds: a collision must not stop training
        fsutil.replace(tmp, os.path.join(self.run_dir, "status.json"), tries=10, wait=0.05, required=False)

    def stop_requested(self) -> bool:
        return os.path.exists(os.path.join(self.run_dir, "STOP"))

    def pause_requested(self) -> bool:
        return os.path.exists(os.path.join(self.run_dir, "PAUSE"))


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def sample_timesteps(n: int, shift: float, device) -> torch.Tensor:
    t = torch.sigmoid(torch.randn(n, device=device))
    return shift * t / (1 + (shift - 1) * t)


def stack_text(embeds: List[torch.Tensor], device):
    length = max(e.shape[0] for e in embeds)
    out = torch.zeros(len(embeds), length, *embeds[0].shape[1:], device=device, dtype=torch.bfloat16)
    mask = torch.zeros(len(embeds), length, device=device, dtype=torch.bool)
    for i, e in enumerate(embeds):
        out[i, : e.shape[0]] = e.to(device)
        mask[i, : e.shape[0]] = True
    return out, mask


class Trainer:
    def __init__(self, cfg: RunConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.rl = RunLog(run_dir)
        self.device = torch.device("cuda")
        self.model = get_adapter(cfg.model)  # everything model-specific goes through this
        self.cache = LatentCache(os.path.join(ROOT, "cache", cfg.model), cfg.model)
        self.evaluator = None
        self.ap = None
        self.quality_prompts = []
        self.checks = {}
        self.extra_items = []
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    # ------------------------------------------------------------------ dataset
    def prepare_data(self):
        c = self.cfg.dataset
        pairs = scan_folder(c.folder, c.caption_ext)
        if not pairs:
            raise RuntimeError(f"이미지가 없습니다: {c.folder}")
        no_cap = sum(1 for _, cap in pairs if not cap)
        if no_cap:
            self.rl.log(f"⚠ 캡션 없는 이미지 {no_cap}개 (트리거워드만 사용됩니다)")
        from PIL import Image

        items = []
        for path, cap in pairs:
            with Image.open(path) as im:
                w, h = im.size
            bucket = bucket_for(w, h, c.resolution, step=self.model.bucket_step)
            if max(w / bucket[0], h / bucket[1]) < 0.75:
                self.rl.log(f"⚠ 저해상도 이미지(업스케일됨): {os.path.basename(path)} {w}x{h}")
            variants = load_caption_variants(path)
            if variants and cap and cap != variants.get("long"):
                variants["long"] = cap  # edited by hand in the caption editor (.txt): that wins
            alt =[(apply_trigger(variants[k], c.trigger), w) for k, w in LENGTH_WEIGHTS if k in variants]
            items.append(Item(image=path, caption=apply_trigger(cap, c.trigger), bucket=bucket,
                              key=file_key(path, f"{bucket[0]}x{bucket[1]}"), alt=alt if len(alt) > 1 else []))
        for i in self._holdout_indices([os.path.basename(p) for p, _ in pairs]):
            items[i].holdout = True
        self.items = items
        a = self.cfg.autopilot
        self.quality_prompts = ([apply_trigger(p, c.trigger) for p in a.quality_prompts]
                                if a.quality_judges and a.quality_prompts else [])
        self.checks = {}
        if self.quality_prompts:
            checks = {"style": apply_trigger(a.check_style_prompt, c.trigger),
                      "edit": apply_trigger(a.check_edit_prompt, c.trigger),
                      "bleed": a.check_bleed_prompt}  # someone else: no trigger
            self.checks = {k: v for k, v in checks.items() if v}
        self.train_items = [it for it in items if not it.holdout]
        # mixed resolution: the same training images again at the extra sizes (sampler only)
        self.extra_items = []
        for r in sorted({int(r) for r in (c.multires or []) if int(r) != c.resolution}):
            for it in self.train_items:
                with Image.open(it.image) as im:
                    b = bucket_for(*im.size, r, step=self.model.bucket_step)
                self.extra_items.append(Item(image=it.image, caption=it.caption, bucket=b,
                                             key=file_key(it.image, f"{b[0]}x{b[1]}"), alt=list(it.alt)))
        if self.extra_items:
            self.rl.log(f"해상도 섞기: {c.resolution}px에 더해 "
                        f"{sorted({int(r) for r in c.multires if int(r) != c.resolution})}px로도 학습해요 (고르게 섞음)")
        self.val_items = [it for it in items if it.holdout]
        buckets = sorted({it.bucket for it in self.train_items})
        self.rl.log(f"데이터셋: 학습 {len(self.train_items)}장, 검증(홀드아웃) {len(self.val_items)}장, 버킷 {buckets}")

    # best-of checkpoints: the step where a metric peaked is saved even between regular saves
    BEST_METRICS = {"likeness": ("닮음", True), "val": ("홀드아웃 loss", False)}
    BEST_FROM = 0.3  # earlier peaks are just the climb

    def _marks_path(self) -> str:
        return os.path.join(self.run_dir, "best_marks.json")

    def _load_marks(self, upto: Optional[int] = None):
        try:
            with open(self._marks_path(), encoding="utf-8") as f:
                marks = json.load(f)
        except (OSError, ValueError):
            marks = {}
        if upto is not None:  # continuing from an earlier step: later peaks were abandoned
            marks = {k: v for k, v in marks.items() if v.get("step", 0) <= upto}
        self.marks = marks
        self._save_marks()

    def _save_marks(self):
        with open(self._marks_path(), "w", encoding="utf-8") as f:
            json.dump(self.marks, f, ensure_ascii=False, indent=1)

    def _mark_best(self, metric: str, step: int, value, save_every: int, max_steps: int):
        """Save ``{name}_{step}.safetensors`` when ``metric`` beats its best so far (after the
        first BEST_FROM of the run). One extra file per metric: the superseded one goes unless
        it is a regular checkpoint or another metric's best."""
        if value is None or step < self.BEST_FROM * max_steps:
            return
        label, higher = self.BEST_METRICS[metric]
        prev = self.marks.get(metric)
        if prev and not (value > prev["value"] if higher else value < prev["value"]):
            return
        name = self.cfg.name
        path = os.path.join(self.run_dir, f"{name}_{step:05d}.safetensors")
        if not os.path.isfile(path):
            with self.using_ema():
                self.save_lora(path, extra={"ss_steps": step, "oslora_best": metric})
        note = ""
        if prev and prev["step"] != step:
            old = prev["step"]
            others = {v["step"] for k, v in self.marks.items() if k != metric}
            old_path = os.path.join(self.run_dir, f"{name}_{old:05d}.safetensors")
            if (not save_every or old % save_every) and old not in others and old != max_steps and os.path.isfile(old_path):
                os.remove(old_path)
                note = f", 이전 최고(step {old}) 파일은 정리"
        self.marks[metric] = {"step": step, "value": float(value)}
        self._save_marks()
        self.rl.log(f"★ {label} 최고 갱신: step {step} ({value:.4f}) → {name}_{step:05d}.safetensors 저장{note}")

    @staticmethod
    def _train_text(it: Item, plain: bool = False) -> torch.Tensor:
        """One of the long / medium / short captions (weighted), or the single caption; ``plain``:
        the version without image-finish statements, if there is one."""
        alt, single = (it.alt_text_plain, it.text_plain) if plain and it.text_plain is not None else (it.alt_text, it.text)
        if not alt:
            return single
        return random.choices([t for t, _ in alt], weights=[w for _, w in alt])[0]

    def _train_latent(self, it: Item) -> Tuple[torch.Tensor, bool]:
        """The clean latent, or (photo_texture of the time) the photo-textured one; True for the
        latter, whose caption then says nothing about the finish."""
        p = float(self.cfg.dataset.photo_texture or 0)
        if it.latent_photo is not None and p > 0 and random.random() < p:
            return it.latent_photo, True
        return it.latent, False

    def _holdout_indices(self, names: List[str]) -> List[int]:
        """Held-out images, fixed per run (holdout.json) so continuing keeps the same val set.
        Face crops go with their originals. Runs from before holdout.json keep the old pick."""
        path = os.path.join(self.run_dir, "holdout.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                saved = set(json.load(f))
            return [i for i, n in enumerate(names) if n in saved]
        c = self.cfg.dataset
        if getattr(self, "resuming", False):
            return pick_holdout(len(names), c.holdout, self.cfg.train.seed)
        sources = crop_sources(c.folder, names)
        picked = pick_holdout(len(names), c.holdout, self.cfg.train.seed, names, sources)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([names[i] for i in picked], f, ensure_ascii=False, indent=1)
        if sources:
            self.rl.log(f"얼굴 크롭 {len(sources)}장은 원본과 묶어서 홀드아웃을 골랐어요 (같은 얼굴이 학습·검증에 나뉘지 않게)")
        return picked

    def cache_everything(self):
        cfg = self.cfg
        self.rl.set_status(phase="caching", message="텍스트 임베딩 캐싱")
        # the photo texture with the tone: its steps get the captions without finish statements
        plain = {}
        if cfg.dataset.photo_tone and float(cfg.dataset.photo_texture or 0) > 0:
            from .augment import strip_finish

            for it in self.items + self.extra_items:
                if not os.path.basename(it.image).startswith(AUG_PREFIX):
                    for t in [it.caption] + [t for t, _ in it.alt]:
                        plain[t] = strip_finish(t)
        texts = ({it.caption for it in self.items + self.extra_items} | {t for it in self.items + self.extra_items for t, _ in it.alt} | {""}
                 | set(plain.values()) | set(cfg.sample.prompts)
                 | set(cfg.autopilot.leak_prompts) | set(self.quality_prompts) | set(self.checks.values()))
        missing = [t for t in texts if self.cache.get_text(t) is None]
        if missing:
            te = self.model.text_encoder(cfg.paths.text_encoder, log=self.rl.log)
            for i, t in enumerate(missing):
                self.cache.put_text(t, te.encode([t])[0])
                self.rl.set_status(message=f"텍스트 인코딩 {i + 1}/{len(missing)}")
            te.unload()
            del te
            flush()
        self.text = {t: self.cache.get_text(t) for t in texts}
        for it in self.items + self.extra_items:
            it.text = self.text[it.caption]
            it.alt_text = [(self.text[t], w) for t, w in it.alt]
            if it.caption in plain:
                it.text_plain = self.text[plain[it.caption]]
                it.alt_text_plain = [(self.text[plain[t]], w) for t, w in it.alt]
        if plain:
            n = sum(1 for t, p in plain.items() if p != t)
            self.rl.log(f"사진 질감 step의 캡션: 화질 문장(매끈함·노이즈 없음·보정된 피부 등)을 뺀 버전을 써요 "
                        f"(캡션 {len(plain)}개 중 {n}개에서 뺐어요)")
        n_alt = sum(1 for it in self.items if it.alt)
        if n_alt:
            self.rl.log(f"캡션 길이 섞기: {n_alt}장은 긴/중간/짧은 캡션 중 하나를 매번 골라요 "
                        f"({', '.join(f'{k} {w:.0%}' for k, w in LENGTH_WEIGHTS)})")

        self.rl.set_status(message="이미지 latent 캐싱")
        self.vae = self.model.vae(cfg.paths.vae, log=self.rl.log)
        todo = [it for it in self.items + self.extra_items if self.cache.get_latent(it.key) is None]
        for i, it in enumerate(todo):
            img = load_image(it.image, it.bucket)[None]
            self.cache.put_latent(it.key, self.vae.encode(img)[0])
            self.rl.set_status(message=f"VAE 인코딩 {i + 1}/{len(todo)}")
        for it in self.items + self.extra_items:
            it.latent = self.cache.get_latent(it.key)
        if float(cfg.dataset.photo_texture or 0) > 0:
            self._cache_photo()
        self.vae.to("cpu")
        flush()
        self.rl.log(f"캐싱 완료 (새로 인코딩: 텍스트 {len(missing)}, 이미지 {len(todo)})")

    def _cache_photo(self):
        """Photo-texture latents (sensor noise at the training size) for every item but the
        degraded copies; the VAE must be on the GPU."""
        from . import photoreal

        p = float(self.cfg.dataset.photo_texture or 0)
        tone = bool(self.cfg.dataset.photo_tone)
        textured = [it for it in self.items + self.extra_items if not os.path.basename(it.image).startswith(AUG_PREFIX)]
        new = 0
        for i, it in enumerate(textured):
            key = file_key(it.image, f"{it.bucket[0]}x{it.bucket[1]}|{photoreal.cache_tag(tone)}")
            if self.cache.get_latent(key) is None:
                img = load_image(it.image, it.bucket, photo=True, tone=tone)
                self.cache.put_latent(key, self.vae.encode(img[None])[0])
                new += 1
                self.rl.set_status(message=f"사진 질감 버전 인코딩 {i + 1}/{len(textured)}")
            it.latent_photo = self.cache.get_latent(key)  # from the cache: CPU, like the clean latents
        look = f"'{photoreal.TONE}' 톤과 '{photoreal.NOISE}' 센서 노이즈를" if tone else f"'{photoreal.NOISE}' 센서 노이즈만"
        self.rl.log(f"사진 질감: {len(textured)}장에 {look} 입힌 버전을 만들어 학습 step의 {p:.0%}에서 써요 "
                    f"(새로 인코딩 {new}장, 화질 변형 사본은 제외)")

    # ------------------------------------------------------------------ model
    def build_model(self):
        cfg = self.cfg
        from .models import quant

        quant.INT8_COMPUTE = bool(getattr(cfg.train, "int8_compute", False))
        if quant.INT8_COMPUTE:
            self.rl.log("빠른 int8 연산 켜짐: 원본 가중치를 int8 텐서코어로 바로 계산해요 (학습·샘플 모두)")
        self.dit = self.model.load_dit(cfg.paths.dit, log=self.rl.log)
        self.dit.gradient_checkpointing = cfg.train.gradient_checkpointing
        n = cfg.network
        targets = self.model.lora_targets
        self.net = LoRANetwork(self.dit, targets.get(n.targets, targets["all"]), rank=n.rank, alpha=n.alpha,
                               dropout=n.dropout, rank_dropout=n.rank_dropout, key_prefix=self.model.lora_prefix)
        params = self.net.parameters()
        for p in params:
            p.requires_grad_(True)
        count = sum(p.numel() for p in params)
        self.rl.log(f"LoRA: {len(self.net.modules)}개 레이어, rank {n.rank}, alpha {n.alpha}, 파라미터 {count / 1e6:.1f}M")
        self.offload = self.model.block_offload(self.dit, log=self.rl.log)
        t = cfg.train
        if t.optimizer == "adamw8bit":
            import bitsandbytes as bnb

            self.opt = bnb.optim.AdamW8bit(params, lr=1e-6, weight_decay=t.weight_decay, betas=(0.9, 0.99))
        else:
            self.opt = torch.optim.AdamW(params, lr=1e-6, weight_decay=t.weight_decay, betas=(0.9, 0.99))
        self.ema = EMA(params, t.ema_decay) if t.ema_decay > 0 else None
        self.swap = None
        inf = (cfg.paths.inference_dit or "").strip()
        if inf and os.path.normcase(os.path.abspath(inf)) != os.path.normcase(os.path.abspath(cfg.paths.dit)):
            if os.path.isfile(inf):
                self.swap = self.model.weight_swap(self.dit, inf, log=self.rl.log)
                self.rl.log(f"샘플·품질 확인은 {os.path.basename(inf)}로 그려요 (RAM에 대기, 그릴 때만 GPU로 교체)")
            else:
                self.rl.log(f"⚠ 추론용 DiT 파일이 없어서 샘플도 학습 모델로 그려요: {inf}")

    # ------------------------------------------------------------------ probes
    def _predict(self, latent, t: float, noise, text):
        x0 = latent[None].to(self.device, torch.float32)
        xt = (1 - t) * x0 + t * noise
        emb, mask = stack_text([text], self.device)
        tt = torch.tensor([t], device=self.device)
        return self.model.predict(self.dit, xt, tt, emb, mask).float(), noise - x0

    def build_probes(self):
        g = torch.Generator().manual_seed(1234)
        rng = random.Random(1234)
        n = self.cfg.autopilot.probe_images
        fit_items = rng.sample(self.train_items, min(n, len(self.train_items)))
        val_items = self.val_items[: max(n, 2)]

        def make(items):
            probes = []
            for it in items:
                for t in PROBE_T:
                    noise = torch.randn(it.latent[None].shape, generator=g)
                    probes.append((it, t, noise))
            return probes

        self.fit_probes = make(fit_items)
        self.val_probes = make(val_items)

        # leak probes: base-model images for unrelated prompts (cached on disk), then the
        # base prediction at fixed noise/t; the LoRA's deviation from it is the leak.
        self.leak_probes = []
        self.rl.set_status(message="누출 측정용 베이스 이미지 준비")
        tag = f"{os.path.basename(self.cfg.paths.dit)}|{LEAK_RES}"
        base_imgs = []
        for i, prompt in enumerate(self.cfg.autopilot.leak_prompts):
            key = "leak_" + __import__("hashlib").sha1(f"{tag}|{prompt}".encode()).hexdigest()[:16]
            lat = self.cache.get_latent(key)
            if lat is None:
                lat = self.model.sample(self.dit, self.text[prompt], self.text[""], LEAK_RES, LEAK_RES,
                                        steps=self.model.base_sample_steps, cfg=self.model.base_sample_cfg, seed=i,
                                        distilled=False)[0]
                self.cache.put_latent(key, lat)
            base_imgs.append(lat)
            for t in LEAK_T:
                noise = torch.randn(lat[None].shape, generator=g)
                with torch.no_grad():
                    base, _ = self._predict(lat, t, noise.to(self.device), self.text[prompt])
                self.leak_probes.append((lat, t, noise, prompt, base.cpu()))
        # keep a preview of what the base model does with the leak prompts
        try:
            self.vae.to(self.device)
            for i, lat in enumerate(base_imgs):
                self.vae.decode(lat[None])[0].save(os.path.join(self.run_dir, "samples", f"leak_base_{i}.jpg"), quality=90)
        finally:
            self.vae.to("cpu")
        self.rl.log(f"프로브 준비: fit {len(self.fit_probes)}, val {len(self.val_probes)}, leak {len(self.leak_probes)}")

    @torch.no_grad()
    def run_probes(self):
        def mse(probes):
            if not probes:
                return None
            total = 0.0
            for it, t, noise in probes:
                pred, target = self._predict(it.latent, t, noise.to(self.device), it.text)
                total += float(F.mse_loss(pred, target))
            return total / len(probes)

        fit = mse(self.fit_probes)
        val = mse(self.val_probes)
        leak = 0.0
        for lat, t, noise, prompt, base in self.leak_probes:
            pred, _ = self._predict(lat, t, noise.to(self.device), self.text[prompt])
            b = base.to(self.device)
            leak += float((pred - b).square().mean() / b.square().mean())
        leak /= max(1, len(self.leak_probes))
        return fit, val, leak

    # ------------------------------------------------------------------ helpers
    def lora_weights(self) -> List[torch.Tensor]:
        return [p.detach().cpu().clone() for p in self.net.parameters()]

    def load_lora_weights(self, weights: List[torch.Tensor]):
        with torch.no_grad():
            for p, w in zip(self.net.parameters(), weights):
                p.copy_(w.to(p.device, p.dtype))

    def _move_state(self, device):
        """Move everything the training loop keeps on the GPU (DiT + LoRA, optimizer moments,
        EMA shadow) to ``device``. Parameters move in place, so the optimizer keeps its refs."""
        self.dit.to(device)
        if self.offload and device != "cpu":
            self.offload.place()  # the swapped blocks go back to RAM
        for state in self.opt.state.values():
            for k, v in state.items():
                # AdamW keeps its scalar step counter on the CPU; leave it where it is
                if torch.is_tensor(v) and k != "step":
                    state[k] = v.to(device)
        if self.ema:
            self.ema.shadow = [s.to(device) for s in self.ema.shadow]
        flush()

    def pause(self, step: int) -> float:
        """Free the GPU until the PAUSE file disappears (or STOP appears), then carry on from
        the same step. Returns the seconds spent paused."""
        t0 = time.time()
        before = torch.cuda.memory_allocated() / 2**30
        self._move_state("cpu")
        after = torch.cuda.memory_reserved() / 2**30
        self.rl.log(f"⏸ 일시정지 (step {step}): GPU를 비웠어요 (VRAM {before:.1f} GB → {after:.1f} GB). "
                    "다시 시작하면 같은 지점부터 이어서 학습해요")
        self.rl.set_status(phase="paused", message=f"일시정지 중 (step {step}) - GPU 비움")
        while self.rl.pause_requested() and not self.rl.stop_requested():
            time.sleep(2)
        self.rl.set_status(phase="training", message="학습 재개 준비")
        try:
            self._live_updates = self._apply_overrides(step)  # settings changed while paused
        except Exception as e:
            self._live_updates = {}
            self.rl.log(f"⚠ 바뀐 설정을 반영하지 못했어요 (이전 설정으로 계속): {e}")
        self._move_state(self.device)
        paused = time.time() - t0
        self.rl.log(f"▶ 재개 (step {step}): {int(paused // 60)}분 {int(paused % 60)}초 동안 쉬었어요")
        self.rl.set_status(phase="training", message=f"학습 중 step {step}")
        return paused

    def _apply_overrides(self, step: int) -> Dict:
        """Settings changed in the UI while paused (``overrides.json``, written by the server on
        "resume with these settings"). Only what can change mid-run without restarting: int8
        compute, LR range, max steps, sample prompts/size/interval, photo texture. Returns the
        loop variables to update."""
        path = os.path.join(self.run_dir, "overrides.json")
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as f:
            o = json.load(f)
        os.remove(path)
        cfg, ap, done, upd = self.cfg, self.ap, [], {}
        if "int8_compute" in o and bool(o["int8_compute"]) != bool(cfg.train.int8_compute):
            from .models import quant

            quant.INT8_COMPUTE = cfg.train.int8_compute = bool(o["int8_compute"])
            done.append("빠른 int8 연산 " + ("켬" if quant.INT8_COMPUTE else "끔"))
        if ap is not None and ("min_lr" in o or "max_lr" in o):
            lo, hi = float(o.get("min_lr", ap.cfg.min_lr)), float(o.get("max_lr", ap.cfg.max_lr))
            if 0 < lo <= hi and (lo, hi) != (ap.cfg.min_lr, ap.cfg.max_lr):
                ap.cfg.min_lr, ap.cfg.max_lr = lo, hi
                cfg.autopilot.min_lr, cfg.autopilot.max_lr = lo, hi
                ap.base_lr = min(max(ap.base_lr, lo), hi)
                done.append(f"LR 범위 {lo:.0e}~{hi:.0e}")
        ms = str(o.get("max_steps", "")).strip()
        if ms.isdigit() and int(ms) > step and int(ms) != getattr(self, "max_steps", None):
            ms = int(ms)
            upd["max_steps"] = ms
            cfg.train.max_steps = ms
            if ap is not None:
                if step < ap.anneal_start:  # not winding down yet: move the wind-down with the end
                    ap.anneal_start = int(ms * (1 - ap.cfg.anneal_frac))
                ap.max_steps = ms
            done.append(f"최대 {ms} step")
        prompts = [p for p in (o.get("samples") or []) if p.strip()]
        if prompts and prompts != list(cfg.sample.prompts):
            missing = [p for p in prompts if p not in self.text]
            if missing:  # the DiT is on the CPU now, so the text encoder fits
                self.rl.set_status(message="새 샘플 프롬프트 인코딩")
                te = self.model.text_encoder(cfg.paths.text_encoder, log=self.rl.log)
                for p in missing:
                    emb = te.encode([p])[0]
                    self.cache.put_text(p, emb)
                    self.text[p] = emb
                te.unload()
                del te
                flush()
            cfg.sample.prompts = prompts
            done.append(f"샘플 프롬프트 {len(prompts)}개")
        for key, attr in (("sample_w", "width"), ("sample_h", "height")):
            if o.get(key) and int(o[key]) != getattr(cfg.sample, attr):
                setattr(cfg.sample, attr, int(o[key]))
                done.append(f"샘플 {attr} {int(o[key])}")
        if o.get("sample_every") and int(o["sample_every"]) != getattr(self, "sample_every", None):
            upd["sample_every"] = int(o["sample_every"])
            cfg.sample.every = int(o["sample_every"])
            done.append(f"샘플 {int(o['sample_every'])} step마다")
        if "photo_texture" in o:
            from . import photoreal

            p = photoreal.RATE if o["photo_texture"] else 0.0
            if p != float(cfg.dataset.photo_texture or 0):
                cfg.dataset.photo_texture = p
                cfg.dataset.photo_tone = p > 0  # the UI's recipe
                if p > 0:
                    self.vae.to(self.device)
                    self._cache_photo()
                    self.vae.to("cpu")
                    flush()
                done.append("사진 질감 " + ("켬" if p > 0 else "끔"))
        if done:
            save_config(cfg, os.path.join(self.run_dir, "config.yaml"))
            self.rl.log(f"⚙ 설정 반영 (step {step}): " + ", ".join(done))
        else:
            self.rl.log("⚙ 바뀐 설정이 없어서 그대로 이어가요")
        return upd

    # ------------------------------------------------------------------ resume state
    def _prune_saves(self, save_every: int, step: int):
        """Keep the newest ``save_keep`` regular checkpoints (LoRA file + its resume state); best-of
        files (★) and the steps after this one are never touched."""
        keep = int(getattr(self.cfg, "save_keep", 0) or 0)
        if keep <= 0:
            return
        name = self.cfg.name
        marked = {v.get("step") for v in getattr(self, "marks", {}).values()}
        regular = sorted(s for s in range(save_every, step + 1, save_every)
                         if s not in marked and os.path.isfile(os.path.join(self.run_dir, f"{name}_{s:05d}.safetensors")))
        gone = regular[:-keep]
        for s in gone:
            for p in (os.path.join(self.run_dir, f"{name}_{s:05d}.safetensors"),
                      os.path.join(self._ckpt_dir(), f"state_{s:05d}.pt")):
                if os.path.isfile(p):
                    os.remove(p)
        if gone:
            self.rl.log(f"중간 저장 정리: 최근 {keep}개만 남기고 step {', '.join(map(str, gone))} 파일을 지웠어요 (★ 파일은 그대로)")

    def _cleanup(self, step: int):
        """After the run: what only served resuming or the quality checks goes. The LoRA files,
        samples, metrics and the last step's small state (continue later, not bit-exact) stay."""
        import shutil

        freed = 0

        def size(p):
            if os.path.isfile(p):
                return os.path.getsize(p)
            return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)

        d = self._ckpt_dir()
        for f in os.listdir(d):
            if f != f"state_{step:05d}.pt":
                freed += size(os.path.join(d, f))
                os.remove(os.path.join(d, f))
        for sub in [os.path.join(self.run_dir, "samples", "quality")] + [
                os.path.join(self.run_dir, f) for f in os.listdir(self.run_dir) if f.startswith("_이어하기전_")]:
            if os.path.isdir(sub):
                freed += size(sub)
                shutil.rmtree(sub, ignore_errors=True)
        self.rl.log(f"정리: 이어하기용 상태·품질 검사 이미지 등 {freed / 2**30:.1f} GB를 지웠어요 "
                    f"(LoRA 파일과 샘플은 그대로, step {step}부터 이어서 학습은 가능하지만 옵티마이저 상태는 새로 시작해요)")

    def _ckpt_dir(self) -> str:
        d = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def save_state(self, step: int, ap: Autopilot, sampler, best_weights, best_step: int, full: bool):
        """Resume points. Every checkpoint gets a small ``state_<step>.pt`` (Autopilot state,
        RNG, the recommended weights so far); its LoRA weights are the checkpoint file itself.
        ``latest.pt`` additionally holds the raw weights, EMA and optimizer moments, so a
        stopped run continues exactly (overwritten each time, ~1.2 GB)."""
        d = self._ckpt_dir()
        light = {
            "step": step,
            "ap": {k: v for k, v in ap.__dict__.items() if k != "cfg"},
            "best_step": best_step,
            "best": [w.to(torch.bfloat16) for w in best_weights] if best_weights is not None else None,
            "epoch": sampler.epoch,
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "py": random.getstate()},
        }
        torch.save(light, os.path.join(d, f"state_{step:05d}.pt"))
        if full:
            snap = self.take_snapshot()
            heavy = dict(light, raw=snap[0], opt=snap[1],
                         ema=[s.cpu() for s in self.ema.shadow] if self.ema else None)
            tmp = os.path.join(d, "latest.pt.tmp")
            torch.save(heavy, tmp)
            fsutil.replace(tmp, os.path.join(d, "latest.pt"), tries=100)
            with open(os.path.join(d, "latest.json"), "w", encoding="utf-8") as f:
                json.dump({"step": step}, f)

    def restore(self, step: int, ap: Autopilot, sampler, max_steps: int):
        """Continue from a saved step: exactly from ``latest.pt`` when it is that step,
        otherwise from the checkpoint's LoRA weights with a fresh optimizer (re-warmed)."""
        from safetensors.torch import load_file

        d = self._ckpt_dir()
        latest = os.path.join(d, "latest.pt")
        heavy = torch.load(latest, weights_only=False) if os.path.isfile(latest) else None
        exact = heavy is not None and heavy["step"] == step
        if exact:
            state = heavy
            self.load_lora_weights(state["raw"])
            self.opt.load_state_dict(state["opt"])
            if self.ema and state.get("ema") is not None:
                self.ema.shadow = [s.float().to(self.device) for s in state["ema"]]
        else:
            state = torch.load(os.path.join(d, f"state_{step:05d}.pt"), weights_only=False)
            sd = load_file(os.path.join(self.run_dir, f"{self.cfg.name}_{step:05d}.safetensors"))
            self.net.load_state_dict(sd)  # the checkpoint holds EMA weights: start from them
            if self.ema:
                self.ema.shadow = [p.detach().float().clone() for p in self.net.parameters()]
        saved = state["ap"]
        old_max = saved["max_steps"]
        default_old = int(saved["max_steps"] * (1 - ap.cfg.anneal_frac))
        ap.__dict__.update(saved)
        ap.max_steps = max_steps
        if saved["anneal_start"] >= default_old:  # not pulled forward by strikes: follow the new length
            ap.anneal_start = int(max_steps * (1 - ap.cfg.anneal_frac))
        if step < ap.anneal_start:
            ap.anneal_from = None
        if not exact:
            ap.rewarm = (step, max(20, ap.eval_every // 2))  # fresh Adam moments: short re-warmup
        sampler.epoch = state.get("epoch", 0)
        rng = state.get("rng") or {}
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"])
            random.setstate(rng["py"])
        best = [w.float() for w in state["best"]] if state.get("best") is not None else None
        self._prune_after(step)
        how = "정확히 (옵티마이저까지)" if exact else "체크포인트 가중치에서 (옵티마이저 새로, 짧게 재워밍업)"
        self.rl.log(f"▶ step {step}부터 이어서 학습: {how}, 최대 {old_max} → {max_steps} step")
        return state["best_step"], best

    def _prune_after(self, step: int):
        """Move everything produced after ``step`` (abandoned future) out of the way, so the
        graphs and samples continue cleanly; nothing is deleted."""
        import re
        import shutil

        attic = os.path.join(self.run_dir, f"_이어하기전_step{step:05d}이후")
        moved = 0

        def stash(path):
            nonlocal moved
            rel = os.path.relpath(path, self.run_dir)
            dst = os.path.join(attic, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(path, dst)
            moved += 1

        name = re.escape(self.cfg.name)
        for f in os.listdir(self.run_dir):
            m = re.fullmatch(rf"{name}_(\d{{5}})\.safetensors", f)
            if (m and int(m.group(1)) > step) or f in (f"{self.cfg.name}.safetensors", f"{self.cfg.name}_last.safetensors"):
                stash(os.path.join(self.run_dir, f))
        lj = os.path.join(self.run_dir, "checkpoints", "latest.json")
        if os.path.isfile(lj):
            with open(lj, encoding="utf-8") as fh:
                latest_step = json.load(fh).get("step", 0)
            if latest_step > step:  # (moved only after the handle is closed: Windows locks open files)
                for f in ("latest.pt", "latest.json"):
                    if os.path.isfile(os.path.join(self.run_dir, "checkpoints", f)):
                        stash(os.path.join(self.run_dir, "checkpoints", f))
        for sub in ("samples", os.path.join("samples", "quality"), "checkpoints"):
            p = os.path.join(self.run_dir, sub)
            if not os.path.isdir(p):
                continue
            for f in os.listdir(p):
                m = re.match(r"(?:state_)?(\d{5})[_.]", f)
                if m and int(m.group(1)) > step:
                    stash(os.path.join(p, f))
        mp = os.path.join(self.run_dir, "metrics.jsonl")
        if os.path.isfile(mp):
            keep = []
            with open(mp, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        if json.loads(line).get("step", 0) <= step:
                            keep.append(line)
                    except Exception:
                        pass
            self.rl.metrics_f.close()
            with open(mp, "w", encoding="utf-8") as fh:
                fh.writelines(keep)
            self.rl.metrics_f = open(mp, "a", encoding="utf-8")
        if moved:
            self.rl.log(f"step {step} 이후에 만들어졌던 파일 {moved}개는 '{os.path.basename(attic)}' 폴더로 옮겼어요")

    def take_snapshot(self):
        """Weights + optimizer state on the CPU (rollback target)."""

        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu().clone()
            if isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            return copy.deepcopy(obj)

        return self.lora_weights(), to_cpu(self.opt.state_dict())

    def rollback(self, snapshot):
        self.load_lora_weights(snapshot[0])
        self.opt.load_state_dict(snapshot[1])
        if self.ema:
            self.ema.shadow = [w.float().to(self.device) for w in snapshot[0]]

    def using_ema(self):
        trainer = self

        class _Ctx:
            def __enter__(self):
                if trainer.ema:
                    trainer.ema.swap()
                trainer.net.train(False)

            def __exit__(self, *a):
                if trainer.ema:
                    trainer.ema.swap()
                trainer.net.train(True)

        return _Ctx()

    def save_lora(self, path: str, weights: Optional[List[torch.Tensor]] = None, extra: Optional[Dict] = None):
        if weights is not None:
            live = self.lora_weights()
            self.load_lora_weights(weights)
        meta = {
            "ss_base_model": self.model.base_model_tag,
            "ss_network_dim": self.cfg.network.rank,
            "ss_network_alpha": self.cfg.network.alpha,
            "oslora_trigger": self.cfg.dataset.trigger,
            "oslora_resolution": self.cfg.dataset.resolution,
            "oslora_name": self.cfg.name,
        }
        meta.update(extra or {})
        self.net.save(path, meta)
        if weights is not None:
            self.load_lora_weights(live)

    def _render(self, prompts, width, height, steps, cfg_scale, seed, out_dir, step, quality=92,
                seeds=None, names=None):
        """Render ``prompts`` with the EMA weights on the inference DiT (SFT) and save them as
        ``<step>_<i>.jpg`` (or ``<step>_<names[i]>.jpg``); seeds default to ``seed + i``."""
        seeds = seeds or [seed + i for i in range(len(prompts))]
        if self.swap:
            self.swap.swap()
        try:
            with self.using_ema():
                lats = [self.model.sample(self.dit, self.text[p], self.text[""], width, height, steps=steps,
                                          cfg=cfg_scale, seed=seeds[i], distilled=bool(self.swap))
                        for i, p in enumerate(prompts)]
        finally:
            if self.swap:
                self.swap.swap()
        torch.cuda.empty_cache()  # sampling leaves reserved-but-free blocks; the VAE needs them
        self.vae.to(self.device)
        images = []
        os.makedirs(out_dir, exist_ok=True)
        for i, lat in enumerate(lats):
            img = self.vae.decode(lat)[0]
            img.save(os.path.join(out_dir, f"{step:05d}_{names[i] if names else i}.jpg"), quality=quality)
            images.append(img)
        self.vae.to("cpu")
        flush()
        return images

    def sample_images(self, step: int):
        """The viewing samples (sample.every, plus step 0 and the final selected weights)."""
        s = self.cfg.sample
        if not s.prompts:
            return
        self.rl.set_status(phase="sampling", message=f"샘플 생성 (step {step})")
        self._render(s.prompts, s.width, s.height, s.steps, s.cfg, s.seed, os.path.join(self.run_dir, "samples"), step)
        self.rl.set_status(phase="training")

    def quality_round(self, step: int):
        """Small, fast renders scored by the judges and fed to the Autopilot, independent of the
        viewing samples so it can run far more often:
            likeness   every quality prompt x quality_seeds (median + spread)
            adherence / diversity / copy / aesthetic on the first seed (comparable across runs)
            checks     clean-look override, flexibility and identity bleed, each against its own
                       step-0 render (same seed, LoRA still zero)"""
        if self.evaluator is None:
            return None
        from .evaluate import frame_sharpness

        a = self.cfg.autopilot
        self.rl.set_status(phase="sampling", message=f"품질 확인 (step {step})")
        prompts = list(self.quality_prompts)
        seeds = max(1, int(a.quality_seeds or 1))
        jobs = [(p, 777 + k, f"{k * len(prompts) + n}") for k in range(seeds) for n, p in enumerate(prompts)]
        every = max(1, int(getattr(a, "check_every_rounds", 1) or 1))
        q_every = max(1, getattr(self, "quality_every", 1))
        do_checks = step == 0 or (step // q_every) % every == 0
        checks = dict(self.checks) if do_checks else {}
        jobs += [(p, 901 + n, f"check-{kind}") for n, (kind, p) in enumerate(checks.items())]
        images = self._render([j[0] for j in jobs], a.quality_width, a.quality_height, a.quality_steps,
                              self.cfg.sample.cfg, 777, os.path.join(self.run_dir, "samples", "quality"), step, 85,
                              seeds=[j[1] for j in jobs], names=[j[2] for j in jobs])
        n_id = seeds * len(prompts)
        q = self.evaluator.score(images[:len(prompts)], prompts)
        q.update(self.evaluator.likeness_spread(images[:n_id]))
        for (kind, _), im in zip(checks.items(), images[n_id:]):
            if kind == "style":
                q["style_sharp"] = frame_sharpness(im)
            elif kind == "edit":
                q["edit_match"] = self.evaluator.text_match(im, a.check_edit_text)
            elif kind == "bleed":
                q["bleed"] = self.evaluator.bleed(im)
        base = self._quality_baseline(step, q)
        if base.get("style_sharp") and "style_sharp" in q:
            q["style_ratio"] = q["style_sharp"] / base["style_sharp"]
        if base.get("edit_match") is not None and "edit_match" in q:
            q["edit_delta"] = q["edit_match"] - base["edit_match"]
        note = self.ap.quality(step, q) if self.ap is not None else None
        self.rl.metric(type="quality", step=step, final=False, note=note, **q)
        parts = []
        if q.get("likeness") is not None:
            parts.append(f"닮음 {q['likeness']:.2f}" + (f"±{q['likeness_iqr'] / 2:.2f}" if "likeness_iqr" in q else ""))
        parts += [f"프롬프트 준수 {q['adherence']:.1f}", f"학습이미지 유사도 {q['copy']:.2f}"]
        if "diversity" in q:
            parts.append(f"다양성 {q['diversity']:.2f}")
        if "aesthetic" in q:
            parts.append(f"미적 {q['aesthetic']:.2f}")
        if "style_ratio" in q:
            parts.append(f"화질 덮어쓰기 ×{q['style_ratio']:.2f}")
        if "edit_delta" in q:
            parts.append(f"유연성 {q['edit_delta']:+.1f}")
        if "bleed" in q:
            parts.append(f"번짐 {q['bleed']:.2f}")
        self.rl.log(f"[품질 {step}] " + ", ".join(parts) + (f" | {note}" if note else ""))
        if note:
            self.rl.set_status(last_reason=note, strikes=self.ap.strikes if self.ap else 0)
        self.rl.set_status(phase="training")
        return q

    def _quality_baseline(self, step: int, q: Dict) -> Dict:
        """The step-0 values of the checks (the base model through the same prompts and seeds);
        when continuing a run, read back from metrics.jsonl."""
        if step == 0:
            self.q_base = {k: q[k] for k in ("style_sharp", "edit_match", "bleed") if k in q}
        if getattr(self, "q_base", None) is None:
            self.q_base = {}
            try:
                with open(os.path.join(self.run_dir, "metrics.jsonl"), encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("type") == "quality" and r.get("step") == 0:
                            self.q_base = {k: r[k] for k in ("style_sharp", "edit_match", "bleed") if k in r}
                            break
            except (OSError, ValueError):
                pass
        return self.q_base

    # ------------------------------------------------------------------ main loop
    def train(self, resume_step: Optional[int] = None):
        cfg, t = self.cfg, self.cfg.train
        torch.manual_seed(t.seed)
        random.seed(t.seed)
        sampler = BucketSampler(self.train_items + self.extra_items, t.batch_size, t.seed + (resume_step or 0))
        spe = sampler.steps_per_epoch()
        max_steps = resolve_int(t.max_steps, auto_max_steps(len(self.train_items), t.batch_size))
        eval_every = resolve_int(cfg.autopilot.eval_every, auto_eval_every(max_steps, spe))
        sample_every = resolve_int(cfg.sample.every, eval_every * 5)
        save_every = resolve_int(cfg.save_every, eval_every * 5)
        self.max_steps, self.sample_every = max_steps, sample_every  # may change on resume-with-settings
        bw, bh = max(self.train_items, key=lambda it: it.bucket[0] * it.bucket[1]).bucket
        fixed_shift = t.timestep_shift not in ("auto", None, "")
        shift = float(t.timestep_shift) if fixed_shift else self.model.timestep_shift(bw, bh)
        ap_on = cfg.autopilot.enabled
        ap = Autopilot(cfg.autopilot, max_steps, eval_every)
        self.ap = ap if ap_on else None
        quality_every = resolve_int(cfg.autopilot.quality_every, max(eval_every, int(round(max_steps / 10 / 100)) * 100))
        self.quality_every = quality_every
        if self.quality_prompts:
            self.rl.set_status(message="품질 평가자 준비 (얼굴 인식 + CLIP)")
            try:
                from .evaluate import Evaluator

                # degraded copies would blur the reference face: originals and crops only
                refs = [it.image for it in self.train_items if not os.path.basename(it.image).startswith(AUG_PREFIX)]
                self.evaluator = Evaluator(refs, log=self.rl.log)
                if self.evaluator.face_self:
                    ap.like_target = 0.75 * self.evaluator.face_self
            except Exception as e:  # judges are a bonus; never block training on them
                self.rl.log(f"⚠ 샘플 평가자를 쓸 수 없어 건너뜁니다: {e}")
                self.evaluator = None
        self.rl.log(
            f"최대 {max_steps} step (epoch당 {spe} step), 평가 {eval_every}마다, 샘플 {sample_every}마다, "
            + (f"품질 확인 {quality_every}마다, " if self.evaluator else "") +
            f"timestep shift {shift:.2f}, Autopilot {'ON' if ap_on else 'OFF'}"
        )

        empty = self.text[""]
        params = self.net.parameters()
        self.net.train(True)

        if resume_step is None:
            fit0, val0, _ = self.run_probes()
            ap.set_baseline(fit0, val0)
            self.rl.metric(type="eval", step=0, fit=fit0, val=val0, leak=0.0, lr=ap.base_lr, action="baseline")
            self.rl.log(f"베이스라인: fit {fit0:.4f}" + (f", val {val0:.4f}" if val0 is not None else ""))
            best_weights, best_step = None, 0
            if cfg.sample.prompts:
                self.sample_images(0)
            self.quality_round(0)
            step = 0
        else:
            best_step, best_weights = self.restore(resume_step, ap, sampler, max_steps)
            step = resume_step
        self._load_marks(upto=resume_step)
        snapshot = self.take_snapshot()

        t_start = time.time() - 1e-3
        step0 = step  # speed / ETA are measured over this session's steps
        stop_reason = "완료"
        from .gpustat import GpuMonitor

        gpu = GpuMonitor()  # sampled only while training steps run (not evals / samples)
        span_time, span_steps = 0.0, 0
        while step < max_steps:
            if self.rl.pause_requested():
                t_start += self.pause(step)  # paused time does not count toward speed / ETA
                upd, self._live_updates = getattr(self, "_live_updates", None) or {}, {}
                if "max_steps" in upd:
                    max_steps = self.max_steps = upd["max_steps"]
                    self.rl.set_status(max_steps=max_steps)
                if "sample_every" in upd:
                    sample_every = self.sample_every = upd["sample_every"]
            if self.rl.stop_requested():
                stop_reason = "사용자 중지"
                break
            lr = ap.lr_at(step) if ap_on else self._cosine_lr(step, max_steps)
            for group in self.opt.param_groups:
                group["lr"] = lr
            gpu.active, t_step = True, time.time()

            batch = sampler.next()
            picks = [self._train_latent(it) for it in batch]
            x0 = torch.stack([lat for lat, _ in picks]).to(self.device, torch.float32)
            texts = [empty if random.random() < cfg.dataset.caption_dropout else self._train_text(it, plain=photo)
                     for it, (_, photo) in zip(batch, picks)]
            if cfg.dataset.flip and random.random() < 0.5:
                x0 = x0.flip(-1)
            emb, mask = stack_text(texts, self.device)
            # mixed resolution: each size gets its own shift (more tokens -> later noise levels)
            b_shift = shift if (fixed_shift or not self.extra_items) else self.model.timestep_shift(*batch[0].bucket)
            tt = sample_timesteps(len(batch), b_shift, self.device)
            noise = torch.randn_like(x0)
            xt = (1 - tt.view(-1, 1, 1, 1)) * x0 + tt.view(-1, 1, 1, 1) * noise
            pred = self.model.predict(self.dit, xt, tt, emb, mask)
            loss = F.mse_loss(pred.float(), (noise - x0).float())
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, t.grad_clip if t.grad_clip > 0 else 1e9))
            loss_v = float(loss)

            if ap.observe(loss_v, gnorm):
                self.opt.zero_grad(set_to_none=True)
                self.rollback(snapshot)
                self.rl.log(f"step {step}: NaN/Inf 감지 → 롤백")
                step += 1
                continue
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if t.max_norm > 0:
                self.net.max_norm_(t.max_norm)
            if self.ema:
                self.ema.update()
            step += 1
            gpu.active = False
            span_time, span_steps = span_time + time.time() - t_step, span_steps + 1

            if step % 5 == 0 or step == 1:
                self.rl.metric(type="step", step=step, loss=loss_v, lr=lr, gnorm=gnorm, t=float(tt[0]))
                sps = (time.time() - t_start) / max(1, step - step0)
                self.rl.set_status(phase="training", step=step, max_steps=max_steps, lr=lr, loss=loss_v,
                                   epoch=sampler.epoch, sec_per_step=sps, eta_sec=sps * (max_steps - step),
                                   message=f"학습 중 step {step}/{max_steps}")
            if step % 10 == 0 or step == step0 + 1:
                # console progress only (the UI reads status.json; log.txt stays readable)
                sps = (time.time() - t_start) / max(1, step - step0)
                eta = sps * (max_steps - step)
                print(f"[진행] step {step}/{max_steps} ({100 * step / max_steps:.0f}%) | loss {loss_v:.4f} | "
                      f"LR {lr:.2e} | {sps:.1f}초/step | 남은 시간 최대 "
                      f"{int(eta // 3600)}시간 {int(eta % 3600 // 60)}분", flush=True)

            if step % eval_every == 0 or step == max_steps:
                self.rl.set_status(message=f"평가 중 (step {step})")
                with self.using_ema():
                    fit, val, leak = self.run_probes()
                d = ap.evaluate(step, fit, val, leak)
                if not ap_on:  # still scored (best checkpoint), but no LR control
                    d.rollback, d.stop, d.lr = False, False, lr
                if d.rollback:
                    self.rollback(snapshot)
                else:
                    snapshot = self.take_snapshot()
                if d.is_best:
                    with self.using_ema():
                        best_weights, best_step = self.lora_weights(), step
                if not d.rollback:
                    self._mark_best("val", step, val, save_every, max_steps)
                # this span's training steps only: s/step, and how busy the GPU was meanwhile
                g, spt = gpu.take(), span_time / max(1, span_steps)
                span_time, span_steps = 0.0, 0
                self.rl.metric(type="eval", step=step, fit=fit, val=val, leak=leak, lr=d.lr, action=d.action,
                               reason=d.reason, score=d.score, best=d.is_best, sec_per_step=spt,
                               gpu_util=g and g["util"], gpu_watts=g and g["watts"])
                speed = f"{spt:.1f}초/step" + (f", GPU {g['util']:.0f}% {g['watts']:.0f}W" if g else "")
                self.rl.log(
                    f"[평가 {step}] fit {fit:.4f}" + (f" val {val:.4f}" if val is not None else "")
                    + f" leak {leak:.2e} score {d.score:.4f}{' ★' if d.is_best else ''} | LR {d.lr:.2e} | {d.reason}"
                    + f" | 학습 {speed}"
                )
                self.rl.set_status(best_step=best_step, strikes=ap.strikes, last_action=d.action, last_reason=d.reason)
                if d.stop:
                    stop_reason = "조기 종료: " + d.reason
                    break

            if save_every and step % save_every == 0 and step < max_steps:
                with self.using_ema():
                    self.save_lora(os.path.join(self.run_dir, f"{cfg.name}_{step:05d}.safetensors"), extra={"ss_steps": step})
                self.save_state(step, ap, sampler, best_weights, best_step, full=True)
                self._prune_saves(save_every, step)
            if sample_every and step % sample_every == 0 and step < max_steps:
                self.sample_images(step)
            if self.evaluator is not None and step % quality_every == 0 and step < max_steps:
                q = self.quality_round(step)
                self._mark_best("likeness", step, (q or {}).get("likeness"), save_every, max_steps)
                if ap_on and ap.should_stop(step):
                    stop_reason = "조기 종료: 품질 저하가 반복됨"
                    break

        # ---- finish
        self.rl.log(f"학습 종료 ({stop_reason}) at step {step}")
        with self.using_ema():
            last = self.lora_weights()
        self.save_lora(os.path.join(self.run_dir, f"{cfg.name}_last.safetensors"), last, {"ss_steps": step})
        # this step becomes a resume point too (as a checkpoint file + exact latest state)
        with self.using_ema():
            self.save_lora(os.path.join(self.run_dir, f"{cfg.name}_{step:05d}.safetensors"), extra={"ss_steps": step})
        self.save_state(step, ap, sampler, best_weights, best_step, full=True)
        self.rl.log(f"이어하기 지점 저장: step {step} (나중에 결과 탭에서 이어서 학습할 수 있어요)")
        if best_weights is None:
            best_weights, best_step = last, step
        self.save_lora(os.path.join(self.run_dir, f"{cfg.name}.safetensors"), best_weights,
                       {"ss_steps": best_step, "oslora_selected": "best"})
        self.rl.log(f"저장: {cfg.name}.safetensors (최고 점수 step {best_step}), {cfg.name}_last.safetensors (step {step})")
        if cfg.sample.prompts:
            # final samples from the selected (best) weights
            if self.ema:
                self.ema.shadow = [w.float().to(self.device) for w in best_weights]
            else:
                self.load_lora_weights(best_weights)
            self.sample_images(step)
        if cfg.cleanup:
            self._cleanup(step)
        self.rl.set_status(phase="done", step=step, best_step=best_step, message=f"완료 - {stop_reason}")

    def _cosine_lr(self, step: int, max_steps: int) -> float:
        lr = self.cfg.train.lr
        warm = max(10, int(0.04 * max_steps))
        if step < warm:
            return lr * (0.1 + 0.9 * step / warm)
        p = (step - warm) / max(1, max_steps - warm)
        return lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    def run(self, resume_step: Optional[int] = None):
        try:
            save_config(self.cfg, os.path.join(self.run_dir, "config.yaml"))
            self.rl.set_status(phase="starting", message="준비 중" if resume_step is None else f"step {resume_step}부터 이어서 학습 준비")
            self.resuming = resume_step is not None
            self.prepare_data()
            from . import rundb

            rundb.safe_ingest(self.run_dir, self.rl.log)  # the dataset as it is now
            self.cache_everything()
            self.build_model()
            self.build_probes()
            self.train(resume_step)
        except Exception as e:
            self.rl.log("오류: " + "".join(traceback.format_exception(e)))
            self.rl.set_status(phase="error", message=str(e))
            raise
        finally:
            from . import rundb

            rundb.safe_ingest(self.run_dir, self.rl.log)  # curves and outcome, also of failed / stopped runs
            self.release_memory()

    def release_memory(self):
        """Free everything this run holds, then have the model files dropped from the Windows
        file cache once the process has exited (other programs are never touched)."""
        from . import memclean

        for attr in ("dit", "net", "vae", "opt", "ema", "evaluator", "ap", "swap"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        for it in getattr(self, "items", []) + getattr(self, "extra_items", []):
            it.latent = it.latent_photo = it.text = it.text_plain = None
            it.alt_text, it.alt_text_plain = [], []
        self.text = {}
        self.fit_probes = self.val_probes = self.leak_probes = []
        memclean.release_process_memory()
        p = self.cfg.paths
        files = [p.dit, p.text_encoder, p.vae] + ([p.inference_dit] if p.inference_dit else [])
        if self.cfg.autopilot.quality_judges:
            from .evaluate import CLIP_REPO

            files += memclean.hf_cached_files(CLIP_REPO)
        try:
            memclean.schedule_after_exit(files, log=self.rl.log)
        except Exception as e:
            self.rl.log(f"⚠ RAM 정리 예약 실패: {e}")


def resumable_steps(run_dir: str) -> List[Dict]:
    """Steps a run can continue from: [{"step", "exact"}], newest first."""
    import re

    d = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(d):
        return []
    latest_step = None
    if os.path.isfile(os.path.join(d, "latest.pt")) and os.path.isfile(os.path.join(d, "latest.json")):
        try:
            with open(os.path.join(d, "latest.json"), encoding="utf-8") as f:
                latest_step = json.load(f).get("step")
        except Exception:
            latest_step = None
    out = []
    for f in os.listdir(d):
        m = re.fullmatch(r"state_(\d{5})\.pt", f)
        if m:
            s = int(m.group(1))
            out.append({"step": s, "exact": s == latest_step})
    return sorted(out, key=lambda r: -r["step"])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--resume", default=None, help="step to continue from, or 'latest'")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run_dir = os.path.join(cfg.output_dir if os.path.isabs(cfg.output_dir) else os.path.join(ROOT, cfg.output_dir), cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("STOP", "PAUSE"):
        if os.path.exists(os.path.join(run_dir, f)):
            os.remove(os.path.join(run_dir, f))
    resume = None
    if a.resume:
        steps = resumable_steps(run_dir)
        if not steps:
            raise SystemExit("이어서 학습할 지점이 없습니다 (checkpoints 폴더가 비어 있음)")
        resume = steps[0]["step"] if a.resume == "latest" else int(a.resume)
        if resume not in [s["step"] for s in steps]:
            raise SystemExit(f"step {resume}에는 이어하기 지점이 없습니다: {[s['step'] for s in steps]}")
    Trainer(cfg, run_dir).run(resume)


if __name__ == "__main__":
    main()
