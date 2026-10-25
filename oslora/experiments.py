"""Overnight proxy experiments: short, small real trainings over many datasets, models and LR plans,
recorded in the history DB (see rundb.py). They answer what one full run per dataset can't:
how a dataset's traits and the LR plan shape likeness / leak / holdout -- per model.

A proxy run is 512 px, 600 steps, evals every 50 steps, quality judges every 150, no progress
samples, cleaned up when done (only the final LoRA stays): ~30-55 min each on a 5060 Ti. Caches
are per model and dataset, so the 2nd and 3rd plan on the same pair skip the encoding.

    venv\\Scripts\\python -m oslora.experiments plan D:\\data\\A D:\\data\\B [--models qwen21,krea2]
    venv\\Scripts\\python -m oslora.experiments run        (stop: create experiments\\STOP; resumes later)
    venv\\Scripts\\python -m oslora.experiments status

Plans: ``auto`` (the Autopilot), ``low`` / ``high`` (Autopilot off, cosine from 4e-5 / 1.5e-4).
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
    "low": {"autopilot": False, "lr": 4e-5},
    "high": {"autopilot": False, "lr": 1.5e-4},
}
PROXY = {"resolution": 512, "max_steps": 600, "eval_every": 50, "quality_every": 150}


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


def run_name(folder: str, model: str, plan: str) -> str:
    return f"EXP_{os.path.basename(os.path.normpath(folder))}_{model}_{plan}"


def plan(folders: List[str], models=MODELS, plans=tuple(PLANS)) -> Dict:
    """Add every (dataset, model, plan) to the queue, autopilot runs of all datasets first (the
    most useful data early), then the fixed-LR contrasts. Folders without captions are reported."""
    q = _read(QUEUE, {"items": []})
    have = {it["name"] for it in q["items"]}
    skipped = []
    order = [(p, m, f) for p in plans for m in models for f in folders]
    for p, m, f in order:
        f = os.path.normpath(f.strip().strip('"'))
        if not os.path.isdir(f):
            skipped.append((f, "폴더 없음"))
            continue
        if captioned(f) == 0:
            skipped.append((f, "캡션 없음"))
            continue
        name = run_name(f, m, p)
        if name not in have:
            q["items"].append({"name": name, "folder": f, "model": m, "plan": p, "state": "queued"})
            have.add(name)
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
    from . import photoreal

    cfg.dataset.photo_texture = photoreal.RATE
    cfg.train.max_steps = PROXY["max_steps"]
    cfg.train.lr = p["lr"]
    cfg.autopilot.enabled = p["autopilot"]
    cfg.autopilot.start_lr = min(p["lr"], cfg.autopilot.max_lr)
    cfg.autopilot.eval_every = PROXY["eval_every"]
    cfg.autopilot.quality_every = PROXY["quality_every"]
    cfg.autopilot.quality_steps = ad.quality_steps
    cfg.sample.every = 0
    cfg.sample.prompts = []
    cfg.sample.steps, cfg.sample.cfg = ad.sample_steps, ad.sample_cfg
    cfg.save_every = PROXY["max_steps"]
    cfg.save_keep = 1
    cfg.cleanup = True
    return cfg


def _env():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENCV_LOG_LEVEL="ERROR")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run(log=print) -> None:
    """Work through the queue, one training at a time. STOP (file) ends after the current run;
    a run that failed is marked and skipped, so one bad dataset can't block the night."""
    from .config import save_config

    os.makedirs(EXP_DIR, exist_ok=True)
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 6 * 3600:
        raise SystemExit("실험이 이미 실행 중이에요 (experiments/running.lock)")
    open(LOCK, "w").close()
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
            if code == 0:  # the curves are in the DB and metrics.jsonl; the LoRA itself isn't needed
                for f in glob.glob(os.path.join(run_dir, "*.safetensors")):
                    os.remove(f)
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
    est = sum(sum(per.get(it["model"], [45])) / len(per.get(it["model"], [45])) for it in left)
    return {"counts": by, "remaining_hours": round(est / 60, 1), "next": [it["name"] for it in left[:5]]}


def main():
    ap = argparse.ArgumentParser(prog="python -m oslora.experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("folders", nargs="+")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--plans", default=",".join(PLANS))
    sub.add_parser("run")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "plan":
        r = plan(a.folders, tuple(a.models.split(",")), tuple(a.plans.split(",")))
        print(f"큐에 {r['queued']}개 대기 중")
        for f, why in r["skipped"]:
            print(f"  건너뜀: {f} ({why})")
        print(json.dumps(status(), ensure_ascii=False, indent=1))
    elif a.cmd == "run":
        run()
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
