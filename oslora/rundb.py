"""Training history database (``history/runs.sqlite``): every run's settings, dataset statistics
and curves, kept after the run folder is cleaned up or deleted.

Each eval is one row of (fit, val, leak, lr, action), each quality round one row of the judges'
scores, so runs can later be lined up as sequences -- the data a learned Autopilot (forecasting
the next evals from the history) would train on.

    runs     one row per run (name + creation time of its config): model, settings, outcome,
             Autopilot rules version, dataset statistics at the time, the full config
    evals    run, step, fit, val, leak, lr, action, reason, s/step, GPU util / watts
    quality  run, step, final, likeness ... (every judge the trainer logs)
    steps    run, step, loss, lr, gnorm, t (every 5th step)
    picks    the checkpoints the user picked ("use this one"), from cache/picks.jsonl

``ingest(run_dir)`` is idempotent: it replaces that run's rows. The trainer calls it at the
start (dataset snapshot) and at the end of a run; the server ingests every run folder at start.

    venv\\Scripts\\python -m oslora.rundb          ingest output/* and print a summary
"""

import json
import os
import re
import sqlite3
import time
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "history", "runs.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run TEXT PRIMARY KEY, name TEXT, model TEXT, kind TEXT, folder TEXT, resolution INTEGER,
    rank INTEGER, alpha REAL, max_steps INTEGER, final_step INTEGER, best_step INTEGER,
    phase TEXT, autopilot_version INTEGER, started REAL, updated REAL,
    config TEXT, dataset TEXT, marks TEXT
);
CREATE TABLE IF NOT EXISTS evals (
    run TEXT, step INTEGER, fit REAL, val REAL, leak REAL, lr REAL, action TEXT, reason TEXT,
    sec_per_step REAL, gpu_util REAL, gpu_watts REAL, dw REAL, PRIMARY KEY (run, step)
);
CREATE TABLE IF NOT EXISTS quality (
    run TEXT, step INTEGER, final INTEGER, likeness REAL, likeness_iqr REAL, adherence REAL,
    copy REAL, diversity REAL, aesthetic REAL, style_ratio REAL, style_sharp REAL,
    edit_delta REAL, edit_match REAL, bleed REAL, faces REAL, note TEXT,
    PRIMARY KEY (run, step, final)
);
CREATE TABLE IF NOT EXISTS steps (
    run TEXT, step INTEGER, loss REAL, lr REAL, gnorm REAL, t REAL, PRIMARY KEY (run, step)
);
CREATE TABLE IF NOT EXISTS picks (
    run TEXT, name TEXT, picked INTEGER, recommended INTEGER, time REAL, entry TEXT,
    PRIMARY KEY (name, time)
);
"""
QUALITY_COLS = ("likeness", "likeness_iqr", "adherence", "copy", "diversity", "aesthetic", "style_ratio",
                "style_sharp", "edit_delta", "edit_match", "bleed", "faces", "note")


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.executescript(SCHEMA)
    if "dw" not in [r[1] for r in con.execute("PRAGMA table_info(evals)")]:  # DBs from before the weight-change column
        con.execute("ALTER TABLE evals ADD COLUMN dw REAL")
    return con


def _read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def run_key(run_dir: str) -> Optional[str]:
    """``<name>@<creation time of config.yaml>``: a folder name reused later is another run."""
    cfg = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg):
        return None
    return f"{os.path.basename(os.path.normpath(run_dir))}@{int(os.stat(cfg).st_ctime)}"


def dataset_stats(folder: str) -> Dict:
    """What the dataset looked like (from its check-up report, if one exists)."""
    if not folder or not os.path.isdir(folder):
        return {}
    try:
        from . import diagnose

        rep = _read_json(diagnose._cache_path(folder))
    except Exception:
        rep = None
    if not rep:
        return {}
    items = [it for it in rep.get("items", []) if "error" not in it]
    orig = [it for it in items if not it.get("crop") and not it.get("aug")]
    sims = sorted(it["face"]["sim"] for it in orig if "sim" in it.get("face", {}))
    ratios = sorted(it["face"]["ratio"] for it in orig if "face" in it)
    sides = sorted(min(it["w"], it["h"]) for it in orig)

    def q(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    mean = sum(sims) / len(sims) if sims else None
    return {
        "images": len(items), "originals": len(orig),
        "crops": sum(1 for it in items if it.get("crop")), "augs": sum(1 for it in items if it.get("aug")),
        "likeness_median": q(sims, 0.5), "likeness_p10": q(sims, 0.1),
        "likeness_std": (sum((s - mean) ** 2 for s in sims) / len(sims)) ** 0.5 if sims else None,
        "below_0_7": sum(1 for s in sims if s < 0.7),
        "face_ratio_median": q(ratios, 0.5),
        "multi_face": sum(1 for it in orig if it.get("face", {}).get("count", 1) > 1),
        "no_face": sum(1 for it in orig if "face" not in it),
        "captioned": sum(1 for it in items if it.get("caption_words")),
        "caption_words_median": q(sorted(it.get("caption_words", 0) for it in orig), 0.5),
        "short_side_median": q(sides, 0.5),
    }


def ingest(run_dir: str, con: Optional[sqlite3.Connection] = None) -> Optional[str]:
    """Store (or refresh) one run. Returns its key, None if the folder isn't a run."""
    key = run_key(run_dir)
    if key is None:
        return None
    from .config import load_config

    try:
        cfg = load_config(os.path.join(run_dir, "config.yaml"))
    except Exception:
        return None
    try:
        from .autopilot import AUTOPILOT_VERSION
    except ImportError:
        AUTOPILOT_VERSION = 0
    status = _read_json(os.path.join(run_dir, "status.json"), {}) or {}
    own = con is None
    con = con or connect()
    old = con.execute("SELECT dataset FROM runs WHERE run=?", (key,)).fetchone()
    # the dataset snapshot is taken once, at the start: the folder may change afterwards
    dataset = old[0] if old and old[0] and old[0] != "{}" else json.dumps(dataset_stats(cfg.dataset.folder), ensure_ascii=False)
    import dataclasses

    evals, quality, steps = [], [], []
    mpath = os.path.join(run_dir, "metrics.jsonl")
    if os.path.isfile(mpath):
        with open(mpath, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("type")
                if t == "eval":
                    evals.append((key, r["step"], r.get("fit"), r.get("val"), r.get("leak"), r.get("lr"),
                                  r.get("action"), r.get("reason"), r.get("sec_per_step"), r.get("gpu_util"),
                                  r.get("gpu_watts"), r.get("dw")))
                elif t == "quality":
                    quality.append((key, r["step"], int(bool(r.get("final")))) + tuple(r.get(c) for c in QUALITY_COLS))
                elif t == "step":
                    steps.append((key, r["step"], r.get("loss"), r.get("lr"), r.get("gnorm"), r.get("t")))
    started = os.stat(os.path.join(run_dir, "config.yaml")).st_ctime
    con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        key, cfg.name, cfg.model, cfg.kind, cfg.dataset.folder, cfg.dataset.resolution, cfg.network.rank,
        cfg.network.alpha, cfg.train.max_steps, status.get("step"), status.get("best_step"), status.get("phase"),
        AUTOPILOT_VERSION, started, time.time(), json.dumps(dataclasses.asdict(cfg), ensure_ascii=False, default=str),
        dataset, json.dumps(_read_json(os.path.join(run_dir, "best_marks.json"), {}), ensure_ascii=False)))
    for table in ("evals", "quality", "steps"):
        con.execute(f"DELETE FROM {table} WHERE run=?", (key,))
    # a resumed run logs a step again after rolling back: the last record of a step wins
    con.executemany("INSERT OR REPLACE INTO evals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", evals)
    con.executemany(f"INSERT OR REPLACE INTO quality VALUES ({','.join('?' * (3 + len(QUALITY_COLS)))})", quality)
    con.executemany("INSERT OR REPLACE INTO steps VALUES (?,?,?,?,?,?)", steps)
    con.commit()
    if own:
        con.close()
    return key


def ingest_picks(con: sqlite3.Connection) -> int:
    from .learn import PICKS

    n = 0
    if os.path.isfile(PICKS):
        with open(PICKS, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                con.execute("INSERT OR REPLACE INTO picks VALUES (?,?,?,?,?,?)", (
                    None, r.get("run"), r.get("picked"), r.get("recommended"), r.get("time"),
                    json.dumps(r, ensure_ascii=False)))
                n += 1
    con.commit()
    return n


def ingest_all(output_dir: Optional[str] = None) -> Dict:
    """Every run folder under ``output_dir`` (default: output/) plus the picks."""
    output_dir = output_dir or os.path.join(ROOT, "output")
    con = connect()
    done = []
    if os.path.isdir(output_dir):
        for n in sorted(os.listdir(output_dir)):
            d = os.path.join(output_dir, n)
            if os.path.isdir(d):
                try:
                    k = ingest(d, con)
                except Exception as e:  # one broken folder must not stop the rest
                    print(f"[기록] {n}: {e}")
                    k = None
                if k:
                    done.append(k)
    picks = ingest_picks(con)
    con.close()
    return {"runs": done, "picks": picks}


def safe_ingest(run_dir: str, log=None):
    """For the trainer: never let bookkeeping break a run."""
    try:
        ingest(run_dir)
    except Exception as e:
        if log:
            log(f"⚠ 학습 기록 DB 저장 실패 (학습에는 영향 없음): {e}")


def main():
    r = ingest_all()
    con = connect()
    print(f"기록된 학습 {con.execute('SELECT COUNT(*) FROM runs').fetchone()[0]}개 "
          f"(이번에 갱신 {len(r['runs'])}개), 고른 체크포인트 {r['picks']}개 · {DB}")
    for row in con.execute("SELECT r.run, r.model, r.final_step, r.max_steps, r.phase, "
                           "(SELECT COUNT(*) FROM evals e WHERE e.run=r.run), "
                           "(SELECT COUNT(*) FROM quality q WHERE q.run=r.run), "
                           "(SELECT MAX(likeness) FROM quality q WHERE q.run=r.run) FROM runs r ORDER BY r.started"):
        print(f"  {row[0]}: {row[1]}, step {row[2]}/{row[3]} ({row[4]}), 평가 {row[5]}개, 품질 {row[6]}개, 닮음 최고 {row[7]}")
    con.close()


if __name__ == "__main__":
    main()
