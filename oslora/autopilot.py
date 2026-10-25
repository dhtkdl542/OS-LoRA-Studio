"""Autopilot: adaptive learning rate + overfitting guard.

Every ``eval_every`` steps the trainer measures three deterministic probes (fixed images,
noise and timesteps, so the numbers move only when the weights do):

    fit   flow-matching MSE on a fixed subset of *training* images   (want: down)
    val   the same on held-out images, if any                         (want: down)
    leak  how far the LoRA moves the model's prediction on prompts that have nothing to do
          with the subject, relative to the base model                (want: small)

and, on its own schedule, the quality judges (ArcFace likeness, CLIP prompt adherence,
diversity, similarity to the training images) score small renders (see ``quality``).

Lessons from calibration (Damdeari, 768px):
  * flow-matching loss is mostly irreducible noise; it moves ~1% over a whole run, and at a
    high LR single evaluations wobble by 0.05-0.1%. One-eval comparisons fire on noise.
  * val is insensitive to identity: it bottomed out at step 846 while ArcFace likeness kept
    rising to step ~1175. Stopping at val's minimum stops before the face is fully learned.

So the controller

  * warms up linearly to ``start_lr`` (1e-4 by default), within [min_lr, max_lr];
  * raises the LR (x1.25) after two windows learning at >= 50% of its peak pace -- but only
    while val (3-eval mean) is still making new bests; lowers it (x0.7) on a plateau;
  * on instability (non-finite loss, grad-norm or loss spike) rolls back to the last healthy
    snapshot and halves the LR;
  * uses val as a *brake*: when its 3-eval mean drifts above its best, or the training images
    keep improving while val stands still (3 evals without a real new best, the leak growing
    meanwhile), the LR is cut (x0.8) and the next stretch is watched from there. That is not
    an overfitting strike, and the LR can come back up once val improves again;
  * counts an overfitting *strike* (LR x0.6, final anneal pulled forward; ``patience``
    strikes stop the run) only on firm evidence: the quality judges confirm degradation
    (prompts no longer followed, copying the training images, collapse), val loses half of
    its improvement, or (no holdout) the leak keeps growing through a plateau;
  * recommends the *latest* evaluation that is still healthy (val within a quarter of its
    gain, no quality degradation): for a character LoRA later means a better-learned face.
"""

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional


@dataclass
class AutopilotConfig:
    # start at the fixed LR that worked in ai-toolkit (1e-4); little room above (a 3e-4
    # ceiling overfit by ~850 steps in calibration), more room below (lowering is the safe way)
    start_lr: Optional[float] = 1e-4  # None = geometric mean of min and max
    min_lr: float = 3e-5
    max_lr: float = 1e-4
    warmup_frac: float = 0.04
    anneal_frac: float = 0.15
    early_stop: bool = True
    patience: int = 3
    up_rate: float = 0.5  # fraction of the peak learning pace that counts as "learning well"
    plateau_rate: float = 0.15  # below this fraction of the peak pace it is a plateau
    val_window: int = 3  # val is judged on the mean of the last N evals (noise filter)
    brake_rise: float = 0.15  # val mean above its best by this fraction of its gain -> brake
    gap_frac: float = 0.1  # fit gain since val's best (fraction of the gain then) -> brake
    gap_stale: int = 3  # val without a new best for this many evals while fit improves and leak grows -> brake
    val_eps: float = 0.005  # a new val best must beat the old one by this fraction of val's gain
    stale_leak: float = 0.05  # ...the stall brake needs the leak up by this much over the stretch
    stale_floor: float = 0.5  # ...and never takes the LR below this fraction of the start LR
    val_collapse: float = 0.5  # val gave back this fraction of its gain -> strike
    keep_frac: float = 0.25  # recommendation: val may have given back at most this much
    leak_growth: float = 0.25  # leak growth over two windows that counts during a plateau
    leak_weight: float = 0.1  # display score: leak penalty (tie-breaker)
    min_steps_frac: float = 0.3  # never stop before this fraction of max_steps


@dataclass
class Decision:
    lr: float
    action: str = "hold"
    reason: str = ""
    rollback: bool = False
    stop: bool = False
    is_best: bool = False  # True = this evaluation becomes the recommended checkpoint
    score: float = float("nan")


class Autopilot:
    def __init__(self, cfg: AutopilotConfig, max_steps: int, eval_every: int):
        self.cfg = cfg
        self.max_steps = max_steps
        self.eval_every = eval_every
        start = cfg.start_lr if cfg.start_lr else math.sqrt(cfg.min_lr * cfg.max_lr)
        self.base_lr = min(max(start, cfg.min_lr), cfg.max_lr)
        self.warmup = max(10, int(max_steps * cfg.warmup_frac))
        self.anneal_start = int(max_steps * (1 - cfg.anneal_frac))
        self.anneal_from: Optional[float] = None
        self.good = self.bad = 0
        self.strikes = 0
        self.val_strikes = 0
        self.stability_hits = 0
        self.history: List[Dict] = []
        self.baseline: Optional[Dict] = None
        # val, judged on its running mean
        self.val_hist: List[float] = []
        self.best_val = float("inf")  # best running mean
        self.val_stale = 0  # evals since the running mean made a new best
        self.fit_at_val_best: Optional[float] = None
        self.leak_at_val_best: Optional[float] = None
        self.brake_cool = 0
        self.best_score = float("inf")
        self.best_step = 0
        self.leak_ref: Optional[float] = None
        self.rate_ema: Optional[float] = None
        self.peak_rate = 0.0
        self.window_loss: List[float] = []
        self.window_gnorm: List[float] = []
        self.gnorm_medians: List[float] = []
        self.loss_medians: List[float] = []
        self.nonfinite = 0
        # output-quality judges (see quality())
        self.like_target: Optional[float] = None
        self.q_history: List[Dict] = []
        self.q_ref: Optional[Dict] = None
        self.q_penalty = 0.0
        self.q_bad = 0
        self.q_degraded = False

    # -- per step -------------------------------------------------------------------
    def lr_at(self, step: int) -> float:
        if step < self.warmup:
            lr = self.base_lr * (0.1 + 0.9 * step / self.warmup)
        elif step >= self.anneal_start:
            if self.anneal_from is None:
                self.anneal_from = self.base_lr
            span = max(1, self.max_steps - self.anneal_start)
            p = min(1.0, (step - self.anneal_start) / span)
            lr = self.cfg.min_lr + (self.anneal_from - self.cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * p))
        else:
            lr = self.base_lr
        rewarm = getattr(self, "rewarm", None)  # resumed from a checkpoint with fresh Adam moments
        if rewarm and rewarm[0] <= step < rewarm[0] + rewarm[1]:
            lr *= 0.1 + 0.9 * (step - rewarm[0]) / rewarm[1]
        return lr

    def observe(self, loss: float, gnorm: float) -> bool:
        """Record one optimizer step; True means non-finite (caller rolls back now)."""
        if not (math.isfinite(loss) and math.isfinite(gnorm)):
            self.nonfinite += 1
            return True
        self.window_loss.append(loss)
        self.window_gnorm.append(gnorm)
        return False

    # -- helpers --------------------------------------------------------------------
    def set_baseline(self, fit: float, val: Optional[float]):
        self.baseline = {"fit": fit, "val": val}
        self.history.append({"step": 0, "fit": fit, "val": val, "leak": 0.0, "gain": 0.0, "lr": self.base_lr})
        if val is not None:
            self.val_hist.append(val)

    def gain(self, fit: float) -> float:
        return 1.0 - fit / self.baseline["fit"]

    def _score(self, fit: float, val: Optional[float], leak: float) -> float:
        """Display score (lower is better): val (or fit) relative to baseline plus small leak
        and prompt-adherence penalties. The recommendation itself is "latest healthy"."""
        b = self.baseline
        primary = (val / b["val"]) if (val is not None and b["val"]) else (fit / b["fit"])
        penalty = 0.0
        if self.leak_ref and leak > self.leak_ref:
            penalty = self.cfg.leak_weight * max(0.0, 1.0 - primary) * math.log2(leak / self.leak_ref)
        penalty += 0.5 * max(0.0, 1.0 - primary) * self.q_penalty / 0.1
        return primary + penalty

    def _strike(self, step: int, why: str) -> str:
        """Count one overfitting signal: cut the LR and pull the final anneal forward."""
        self.strikes += 1
        self.good = self.bad = 0
        self.base_lr = max(self.base_lr * 0.6, self.cfg.min_lr)
        if step < self.anneal_start:
            self.anneal_start = min(self.anneal_start, step + max(self.eval_every * 2, int(0.1 * (self.max_steps - step))))
            self.anneal_from = None
        return f"과적합 신호 {self.strikes}/{self.cfg.patience}: {why} → LR ×0.6"

    def _brake(self, why: str) -> str:
        self.good = 0
        self.base_lr = max(self.base_lr * 0.8, self.cfg.min_lr)
        self.brake_cool = 2
        return f"브레이크: {why} → LR ×0.8 (조기 종료 신호는 아님)"

    def should_stop(self, step: int) -> bool:
        return self.cfg.early_stop and self.strikes >= self.cfg.patience and step >= self.cfg.min_steps_frac * self.max_steps

    # -- quality judges ---------------------------------------------------------------
    def quality(self, step: int, q: Dict) -> Optional[str]:
        """Judge one round of quality renders. Some change is the point of a LoRA: a character
        LoRA *should* move CLIP adherence, diversity and similarity-to-training-images once the
        face becomes the character. So the reference is taken when the subject is learned
        (likeness >= like_target, or a clearly learned face that stopped improving, or the
        first LoRA round without faces) and only further degradation while likeness has
        stopped improving counts."""
        prev = self.q_history[-1] if self.q_history else None
        self.q_history.append({"step": step, **q})
        if step == 0 or prev is None:
            return None
        like = q.get("likeness")
        if like is not None and self.like_target:
            stalled = prev.get("likeness") is not None and like - prev["likeness"] < 0.02 and len(self.q_history) >= 4
            learned = like >= self.like_target or (like > 0.35 and stalled)
        else:
            learned = True
        if self.q_ref is None:
            if learned:
                self.q_ref = dict(q)
                return "대상 학습 완료 → 이 시점을 품질 기준점으로 삼음" if like is not None else None
            return None
        ref = self.q_ref
        adh_drop = (ref["adherence"] - q["adherence"]) / max(ref["adherence"], 1e-6)
        copy_rise = q["copy"] - ref["copy"]
        div_drop = (ref["diversity"] - q["diversity"]) if ("diversity" in q and "diversity" in ref) else 0.0
        self.q_penalty = max(0.0, adh_drop)
        like_gain = (like - prev["likeness"]) if (like is not None and prev.get("likeness") is not None) else 0.0
        if like is not None and like >= ref["likeness"] and q["adherence"] >= ref["adherence"]:
            self.q_ref = dict(q)  # strictly better: move the reference up
        problems = []
        if adh_drop > 0.05:
            problems.append(f"프롬프트 준수 -{adh_drop:.0%}")
        if copy_rise > 0.05:
            problems.append(f"학습 이미지와 유사도 +{copy_rise:.2f}")
        if div_drop > 0.08:
            problems.append(f"다양성 -{div_drop:.2f}")
        if problems and like_gain < 0.02:
            self.q_degraded = True
            self.q_bad += 1
            if self.q_bad >= 2 or adh_drop > 0.12:
                self.q_bad = 0
                return self._strike(step, "품질 저하 확인 (" + ", ".join(problems) + ")")
            return "품질 저하 관찰 중 (" + ", ".join(problems) + ")"
        self.q_bad = 0
        self.q_degraded = False
        return None

    # -- per evaluation -------------------------------------------------------------
    def _stable(self):
        if self.nonfinite:
            return False, f"NaN/Inf {self.nonfinite}회"
        if not self.window_gnorm:
            return True, ""
        g_med, l_med = median(self.window_gnorm), median(self.window_loss)
        if len(self.gnorm_medians) >= 2:
            ref_g, ref_l = median(self.gnorm_medians), median(self.loss_medians)
            if g_med > 3.0 * ref_g:
                return False, f"grad norm 급증 ({g_med:.3g} vs {ref_g:.3g})"
            if l_med > 2.0 * ref_l:
                return False, f"loss 급증 ({l_med:.4f} vs {ref_l:.4f})"
        return True, ""

    def evaluate(self, step: int, fit: float, val: Optional[float], leak: float) -> Decision:
        cfg = self.cfg
        stable, why = self._stable()
        if stable and self.window_gnorm:
            self.gnorm_medians.append(median(self.window_gnorm))
            self.loss_medians.append(median(self.window_loss))
        self.window_loss, self.window_gnorm, self.nonfinite = [], [], 0

        g = self.gain(fit)
        # learning curves decay roughly like 1/step, so measure progress per log-step: a
        # healthy run then keeps a roughly constant pace and a real plateau stands out
        prev_step = max(self.history[-1]["step"], 1)
        rate = (g - self.history[-1]["gain"]) / max(math.log(max(step, prev_step + 1) / prev_step), 1e-6)
        self.rate_ema = rate if self.rate_ema is None else 0.5 * self.rate_ema + 0.5 * rate
        self.peak_rate = max(self.peak_rate, self.rate_ema)
        rel = rate / self.peak_rate if self.peak_rate > 0 else 1.0
        if self.leak_ref is None and g > 0.01 and leak > 0:
            self.leak_ref = leak

        d = Decision(lr=self.base_lr)
        warm = step < self.warmup
        in_anneal = step >= self.anneal_start
        settled = step >= 2 * self.warmup and len(self.history) >= 4
        self.brake_cool = max(0, self.brake_cool - 1)

        # ---- val, on its running mean
        val_ok = True  # healthy enough to be recommended
        val_mean = None
        if val is not None:
            self.val_hist.append(val)
            window = self.val_hist[-cfg.val_window:]
            val_mean = sum(window) / len(window)
            b0 = self.baseline["val"]
            # a new best has to be a real step down, not noise-level creep (which would keep
            # resetting the stale count while the training images run away)
            eps = cfg.val_eps * max(b0 - self.best_val, 0.0) if math.isfinite(self.best_val) else 0.0
            if val_mean < self.best_val - eps:
                self.best_val = val_mean
                self.fit_at_val_best = fit
                self.leak_at_val_best = leak
                self.val_stale = 0
                if self.val_strikes > 0:  # recovered: forgive an earlier collapse strike
                    self.val_strikes -= 1
                    self.strikes -= 1
            else:
                self.val_stale += 1
            gain_v = max(b0 - self.best_val, 1e-4 * b0)
            given_back = max(0.0, val_mean - self.best_val) / gain_v
            val_ok = given_back <= cfg.keep_frac

        if not stable:
            self.stability_hits += 1
            self.base_lr = max(self.base_lr * 0.5, cfg.min_lr)
            d.action, d.reason, d.rollback = "rollback", f"불안정: {why} → 직전 정상 상태로 롤백, LR ½", True
        else:
            strike_why, brake_why = None, None
            if val is not None and settled:
                f_best = self.fit_at_val_best or self.baseline["fit"]
                gain_at_best = max(1.0 - f_best / self.baseline["fit"], 1e-6)
                gap = (f_best - fit) / self.baseline["fit"]
                if given_back >= cfg.val_collapse:
                    strike_why = f"홀드아웃이 개선분의 {given_back:.0%}를 잃음"
                    self.val_strikes += 1
                elif self.brake_cool == 0 and given_back > cfg.brake_rise:
                    brake_why = f"홀드아웃 평균이 최저점보다 상승 (개선분의 {given_back:.0%})"
                elif self.brake_cool == 0 and self.val_stale >= cfg.gap_stale and (
                        gap > cfg.gap_frac * gain_at_best
                        # late in a run the fit curve is flat, so a share of the early gain takes
                        # forever: the training images still improving + the leak growing while val
                        # stands still is memorising, whatever the size of the gap
                        or (gap > 0 and leak > (self.leak_at_val_best or 0.0) * (1 + cfg.stale_leak)
                            and self.base_lr > cfg.stale_floor * (cfg.start_lr or self.base_lr) * 1.001)):
                    brake_why = (f"학습 이미지만 좋아지고 홀드아웃은 {self.val_stale}회째 제자리"
                                 + (f", 누출 +{leak / self.leak_at_val_best - 1:.0%}" if self.leak_at_val_best else ""))
                    self.fit_at_val_best, self.leak_at_val_best = fit, leak  # measure the next stretch from here
                    self.val_stale = 0
            elif val is None and settled and len(self.history) >= 4 and step >= cfg.min_steps_frac * self.max_steps:
                # no holdout: leak growing through a sustained plateau is the best proxy
                leak_2ago = self.history[-2]["leak"]
                plateau = all(r < cfg.plateau_rate for r in (rel, self.history[-1].get("rel", 1.0),
                                                              self.history[-2].get("rel", 1.0)))
                if plateau and leak_2ago > 0 and leak > leak_2ago * (1 + cfg.leak_growth):
                    strike_why = f"학습 정체 중 누출 증가 (+{leak / leak_2ago - 1:.0%})"

            if strike_why:
                d.action, d.reason = "overfit", self._strike(step, strike_why)
            elif brake_why:
                d.action, d.reason = "brake", self._brake(brake_why)
            elif warm or in_anneal:
                d.action = "anneal" if in_anneal else "warmup"
                d.reason = "cosine 마무리 구간" if in_anneal else "워밍업"
            elif rel >= cfg.up_rate:
                self.good += 1
                self.bad = 0
                # never speed up on training-image progress alone: with a holdout, its running
                # mean must have made a new best in this evaluation as well
                if val is not None and self.val_stale > 0:
                    d.action, d.reason = "hold", "학습 이미지는 개선 중이지만 홀드아웃이 제자리 → LR 유지"
                elif self.good >= 2 and self.base_lr < cfg.max_lr:
                    self.base_lr = min(self.base_lr * 1.25, cfg.max_lr)
                    self.good = 0
                    d.action, d.reason = "up", f"순조롭게 학습 중 (속도 {rel:.0%}) → LR ×1.25"
                else:
                    d.action, d.reason = "hold", f"학습 중 (속도 {rel:.0%})"
            elif rel < cfg.plateau_rate:
                self.bad += 1
                self.good = 0
                if self.bad >= 2 and self.base_lr > cfg.min_lr:
                    self.base_lr = max(self.base_lr * 0.7, cfg.min_lr)
                    self.bad = 0
                    d.action, d.reason = "down", f"정체 (속도 {rel:.0%}) → LR ×0.7"
                else:
                    d.action, d.reason = "hold", f"정체 관찰 중 (속도 {rel:.0%})"
            else:
                self.good = self.bad = 0
                d.action, d.reason = "hold", f"학습 중 (속도 {rel:.0%})"

        d.score = self._score(fit, val, leak)
        # recommendation: the latest healthy evaluation (later = better-learned subject)
        if stable and not warm and val_ok and not self.q_degraded:
            self.best_score, self.best_step, d.is_best = d.score, step, True

        if self.should_stop(step):
            d.stop = True
            d.reason += " → 조기 종료"
        if self.stability_hits >= 4:
            d.stop = True
            d.reason += " → 반복적 불안정으로 중단"

        d.lr = self.lr_at(step)
        self.history.append({"step": step, "fit": fit, "val": val, "val_mean": val_mean, "leak": leak,
                             "gain": g, "rel": rel, "lr": d.lr})
        return d
