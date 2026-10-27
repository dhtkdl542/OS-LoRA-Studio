"""Overnight proxy experiments: short, small real trainings over many datasets, models and LR plans,
recorded in the history DB (see rundb.py). They answer what one full run per dataset can't:
how a dataset's traits and the LR plan shape likeness / leak / holdout -- per model.

A proxy run is 512 px, 1500 steps, evals every 100 steps, quality judges every 300, no progress
samples, no early stop, cleaned up when done: ~1.5 h (Qwen) on a 5060 Ti. 1500 steps reach the
stretch where holdout stands still while leak grows -- the part the Autopilot has to judge (600
steps ended before it: every decision was "hold"). Caches are per model and dataset, so the 2nd
and 3rd plan on the same pair skip the encoding.

    venv\\Scripts\\python -m oslora.experiments plan D:\\data\\A D:\\data\\B [--models qwen21,krea2]
    venv\\Scripts\\python -m oslora.experiments run [--hours 6.5 | --until 15:00]   (stop: experiments\\STOP)
    venv\\Scripts\\python -m oslora.experiments status

Plans: ``auto`` (the Autopilot), ``auto2`` (the same with another seed: how far two runs differ
by chance), ``mask`` (auto with masked training, on auto's seed), ``low`` / ``high`` (Autopilot
off, cosine from 4e-5 / 1.5e-4).

Seeds: random per dataset, so what a model learns from the runs isn't one seed's luck (with one
seed for everything, chance looks like a dataset trait and the spread looks smaller than it is).
The plans of one dataset share its seed -- auto vs low vs high then differ by the LR plan, not by
chance -- and ``auto2`` gets a second random one. Seeds are kept in the queue and the run config.

Right before its first run, a dataset nobody has prepared gets what the check-up's one-click fixes
add: face close-ups, captions (3 lengths, character mode, the default caption model) and degraded
copies; its check-up report goes into the dataset statistics. While a training started from the UI
is running, the queue waits.
"""

import argparse
import dataclasses
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "experiments")
QUEUE = os.path.join(EXP_DIR, "queue.json")
STOP = os.path.join(EXP_DIR, "STOP")
LOCK = os.path.join(EXP_DIR, "running.lock")

MODELS = ("qwen21", "krea2", "moss_image")
PLANS = {
    "auto": {"autopilot": True, "lr": 1e-4},
    "auto2": {"autopilot": True, "lr": 1e-4, "repeat": True},
    "mask": {"autopilot": True, "lr": 1e-4, "mask": True},  # auto with masked training, on auto's seed
    "low": {"autopilot": False, "lr": 4e-5},
    "high": {"autopilot": False, "lr": 1.5e-4},
}
PROXY = {"resolution": 512, "max_steps": 1500, "eval_every": 100, "quality_every": 300}
CAPTION = {"model": "qwen38-27b-gguf", "style": "moss", "subject": "character"}
# photo texture of every proxy run, fixed so all of them train the same way: the recipe of the
# first runs (noise only, a quarter of the steps), not whatever the UI default is at the time
EXP_PHOTO = {"rate": 0.25, "tone": False}
DEFAULT_MINUTES = {"qwen21": 40, "krea2": 55, "moss_image": 35}  # per 600 steps, until the queue has its own timings
CAPTION_SEC = 25  # per image, Qwen3.8 27B on a 5060 Ti


def _read(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def captioned(folder: str) -> int:
    from .data import scan_folder

    return sum(1 for _, cap in scan_folder(folder, ".txt") if cap)


def uncaptioned(folder: str) -> int:
    """Images still without a caption (a captioning that was cut short leaves some)."""
    from .data import scan_folder

    return sum(1 for _, cap in scan_folder(folder, ".txt") if not cap)


def run_name(folder: str, model: str, plan: str, steps: int = PROXY["max_steps"]) -> str:
    return f"EXP_{os.path.basename(os.path.normpath(folder))}_{model}_{plan}_{steps}"


def assign_seeds(q: Dict) -> None:
    """A seed for every item without one: the one its dataset's runs already use (repeats: the
    dataset's repeat seed), else a new random one."""
    import random

    def group(it):
        return it["folder"], bool(PLANS[it["plan"]].get("repeat"))

    seeds = {}
    for it in q["items"]:
        if "seed" in it:
            seeds.setdefault(group(it), it["seed"])
    for it in q["items"]:
        if "seed" not in it and it["state"] == "queued":
            it["seed"] = seeds.setdefault(group(it), random.SystemRandom().randrange(1, 2 ** 31))


def plan(folders: List[str], models=MODELS, plans=tuple(PLANS)) -> Dict:
    """Add every (dataset, model, plan) to the queue, autopilot runs of all datasets first (the
    most useful data early), then the fixed-LR contrasts. Folders without captions get them when
    their first run comes up."""
    q = _read(QUEUE, {"items": []})
    have = {it["name"] for it in q["items"]}
    skipped = []
    order = [(p, m, f) for p in plans for m in models for f in folders]
    for p, m, f in order:
        f = os.path.normpath(f.strip().strip('"'))
        if not os.path.isdir(f):
            skipped.append((f, "폴더 없음"))
            continue
        name = run_name(f, m, p)
        if name not in have:
            q["items"].append({"name": name, "folder": f, "model": m, "plan": p, "state": "queued",
                               "steps": PROXY["max_steps"]})
            have.add(name)
    assign_seeds(q)
    _write(QUEUE, q)
    return {"queued": sum(1 for it in q["items"] if it["state"] == "queued"),
            "skipped": sorted(set(skipped))}


def make_config(item: Dict):
    from .config import RunConfig
    from .models import get_adapter

    cfg = RunConfig(name=item["name"], model=item["model"], output_dir=EXP_DIR)
    ad = get_adapter(item["model"])
    p = PLANS[item["plan"]]
    cfg.dataset.folder = item["folder"]
    cfg.dataset.resolution = PROXY["resolution"]
    cfg.dataset.photo_texture, cfg.dataset.photo_tone = EXP_PHOTO["rate"], EXP_PHOTO["tone"]
    steps = item.get("steps", 600)
    cfg.train.max_steps = steps
    cfg.train.lr = p["lr"]
    cfg.network.training_adapter = "auto"
    cfg.train.seed = item.get("seed", cfg.train.seed)  # runs queued before per-dataset seeds: 42
    cfg.autopilot.enabled = p["autopilot"]
    cfg.mask.enabled = bool(p.get("mask"))
    cfg.autopilot.early_stop = False  # the whole curve: what happens after it would have stopped
    cfg.autopilot.start_lr = min(p["lr"], cfg.autopilot.max_lr)
    cfg.autopilot.eval_every = PROXY["eval_every"]
    cfg.autopilot.quality_every = PROXY["quality_every"]
    cfg.autopilot.quality_steps = ad.quality_steps
    cfg.sample.every = 0
    cfg.sample.prompts = []
    cfg.sample.steps, cfg.sample.cfg = ad.sample_steps, ad.sample_cfg
    cfg.save_every = steps
    cfg.save_keep = 1
    cfg.cleanup = True
    return cfg


def _env():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENCV_LOG_LEVEL="ERROR")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def ui_training() -> bool:
    """A training started from the UI (output/<run>/status.json updated in the last 3 minutes)."""
    for st in glob.glob(os.path.join(ROOT, "output", "*", "status.json")):
        s = _read(st, {})
        if s.get("phase") in ("starting", "caching", "training", "sampling", "paused") and \
                time.time() - os.path.getmtime(st) < 180:
            return True
    return False


def unprepared(folder: str) -> bool:
    """No degraded copies yet: nobody (neither the user nor the queue) has prepared this dataset.
    The copies are the last step, so a preparation cut short is picked up again next time."""
    from .data import AUG_PREFIX

    return not any(n.startswith(AUG_PREFIX) for n in os.listdir(folder))


def _apply(folder: str, action: str, log) -> None:
    """The check-up's one-click fix ``action``, if the check-up offers it."""
    from . import diagnose

    for f in diagnose.analyze(folder, PROXY["resolution"])["findings"]:
        if f.get("fix") == action and f["files"]:
            log(f"[실험] {f['fix_label']}: {diagnose.fix(folder, action, f['files'])['message']}")
            return


def prepare_dataset(folder: str, log=print) -> bool:
    """Make the dataset what the UI's check-up would: a dataset nobody has prepared gets face
    close-ups, then captions for every image without one (existing ones are kept), then degraded
    copies (they reuse the original's caption). Datasets prepared by hand are left as they are, and
    a dataset is prepared once only -- all its runs must see the same images. False if captioning
    failed."""
    fresh = unprepared(folder)
    if fresh:
        try:
            _apply(folder, "facecrop", log)
        except Exception as e:
            log(f"[실험] 얼굴 클로즈업 실패 (실험은 계속): {folder}: {e}")
    if uncaptioned(folder):
        log(f"[실험] 캡션 만들기: {folder} ({uncaptioned(folder)}장)")
        status = os.path.join(EXP_DIR, "caption_status.json")
        with open(os.path.join(EXP_DIR, "caption_log.txt"), "a", encoding="utf-8") as out:
            code = subprocess.call([sys.executable, "-m", "oslora.caption", "--folder", folder,
                                    "--model", CAPTION["model"], "--style", CAPTION["style"],
                                    "--subject", CAPTION["subject"], "--status", status],
                                   cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=_env())
        if code != 0 or captioned(folder) == 0:
            log(f"[실험] 캡션 실패: {folder} (experiments/caption_log.txt)")
            return False
    try:
        from . import diagnose

        if fresh:
            _apply(folder, "augment", log)
        diagnose.analyze(folder, PROXY["resolution"])  # the report as trained, for the dataset stats
    except Exception as e:
        log(f"[실험] 진단·변형 사본 실패 (실험은 계속): {folder}: {e}")
    return True


def drop_weights(run_dir: str) -> None:
    """An experiment keeps numbers only (log, metrics, config; the curves are in the DB): the LoRA
    files, the resume state (it holds the LoRA too) and the renders go, finished or not. Some
    datasets are of real people -- nothing that can draw them may stay behind."""
    import shutil

    for f in glob.glob(os.path.join(run_dir, "*.safetensors")):
        os.remove(f)
    for sub in ("checkpoints", "samples"):  # samples/ also holds the quality-check renders
        shutil.rmtree(os.path.join(run_dir, sub), ignore_errors=True)


def estimate_minutes(item: Dict, q: Dict) -> float:
    """Expected length of one queued experiment: the average of done runs of the same model and
    length (or a default), plus the captioning of its images that have none yet."""
    steps = item.get("steps", 600)
    done = [it["minutes"] for it in q["items"] if it.get("minutes") and it["model"] == item["model"]
            and it.get("steps", 600) == steps and it["state"] == "done"]
    m = sum(done) / len(done) if done else DEFAULT_MINUTES.get(item["model"], 45) * steps / 600
    todo = uncaptioned(item["folder"])
    if unprepared(item["folder"]):
        todo += 0.2 * (todo + captioned(item["folder"]))  # the face close-ups need captions too
    return m + todo * CAPTION_SEC / 60


def run(log=print, hours: float = 0) -> None:
    """Work through the queue, one training at a time. STOP (file) ends after the current run;
    a run that failed is marked and skipped, so one bad dataset can't block the night. With
    ``hours``, an experiment that wouldn't finish within that budget is not started."""
    from .config import save_config

    os.makedirs(EXP_DIR, exist_ok=True)
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 6 * 3600:
        raise SystemExit("실험이 이미 실행 중이에요 (experiments/running.lock)")
    open(LOCK, "w").close()
    t_start = time.time()
    try:
        while True:
            if os.path.exists(STOP):
                os.remove(STOP)
                log("[실험] STOP 파일을 보고 멈췄어요. 다시 run 하면 이어서 해요")
                return
            q = _read(QUEUE, {"items": []})
            item = next((it for it in q["items"] if it["state"] == "queued"), None)
            if item is None:
                log("[실험] 큐가 비었어요")
                return
            if hours:
                need = estimate_minutes(item, q)
                used = (time.time() - t_start) / 60
                if used + need > hours * 60:
                    log(f"[실험] 시간 예산 {hours}시간: 다음 실험({item['name']}, 약 {need:.0f}분)은 "
                        f"남은 {hours * 60 - used:.0f}분 안에 못 끝나서 여기서 멈춰요")
                    return
            if ui_training():
                log("[실험] 화면에서 시작한 학습이 끝날 때까지 기다려요")
                while ui_training() and not os.path.exists(STOP):
                    os.utime(LOCK)
                    time.sleep(60)
                continue
            if not prepare_dataset(item["folder"], log):
                q = _read(QUEUE, {"items": []})
                for it in q["items"]:  # every run on this folder needs the captions
                    if it["folder"] == item["folder"] and it["state"] == "queued":
                        it["state"] = "failed"
                _write(QUEUE, q)
                continue
            os.utime(LOCK)
            run_dir = os.path.join(EXP_DIR, item["name"])
            os.makedirs(run_dir, exist_ok=True)
            cfg_path = os.path.join(run_dir, "config.yaml")
            save_config(make_config(item), cfg_path)
            left = sum(1 for it in q["items"] if it["state"] == "queued")
            log(f"[실험] 시작: {item['name']} (남은 {left}개)")
            t0 = time.time()
            with open(os.path.join(run_dir, "stdout.txt"), "w", encoding="utf-8") as out:
                code = subprocess.call([sys.executable, "-m", "oslora.train", cfg_path], cwd=ROOT,
                                       stdout=out, stderr=subprocess.STDOUT, env=_env())
            os.utime(LOCK)
            drop_weights(run_dir)
            q = _read(QUEUE, {"items": []})
            for it in q["items"]:
                if it["name"] == item["name"]:
                    it["state"] = "done" if code == 0 else "failed"
                    it["minutes"] = round((time.time() - t0) / 60, 1)
            _write(QUEUE, q)
            log(f"[실험] {'끝' if code == 0 else '실패'}: {item['name']} ({(time.time() - t0) / 60:.0f}분)")
    finally:
        if os.path.exists(LOCK):
            os.remove(LOCK)


def status() -> Dict:
    q = _read(QUEUE, {"items": []})
    by = {}
    for it in q["items"]:
        by[it["state"]] = by.get(it["state"], 0) + 1
    done = [it for it in q["items"] if it.get("minutes")]
    per = {}
    for it in done:
        per.setdefault(it["model"], []).append(it["minutes"])
    left = [it for it in q["items"] if it["state"] == "queued"]
    est = sum(estimate_minutes(it, q) for it in left)
    # what one 6.5-hour night would get through, in queue order
    night, t = [], 0.0
    for it in left:
        t += estimate_minutes(it, q)
        if t > 6.5 * 60:
            break
        night.append(it["name"])
    return {"counts": by, "remaining_hours": round(est / 60, 1), "next_night_6_5h": night}


def main():
    ap = argparse.ArgumentParser(prog="python -m oslora.experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("folders", nargs="+")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--plans", default=",".join(PLANS))
    r = sub.add_parser("run")
    r.add_argument("--hours", type=float, default=0, help="time budget: don't start what wouldn't finish in it")
    r.add_argument("--until", default="", help="the same as a clock time, e.g. 15:00 (next such time)")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "plan":
        r = plan(a.folders, tuple(a.models.split(",")), tuple(a.plans.split(",")))
        print(f"큐에 {r['queued']}개 대기 중")
        for f, why in r["skipped"]:
            print(f"  건너뜀: {f} ({why})")
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    elif a.cmd == "run":
        hours = a.hours
        if a.until:
            import datetime

            now = datetime.datetime.now()
            hh, mm = map(int, a.until.split(":"))
            end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if end <= now:
                end += datetime.timedelta(days=1)
            hours = (end - now).total_seconds() / 3600
        run(hours=hours)
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
