"""OS LoRA Studio local server: a small JSON API + the static web app.

    python -m oslora.server        (http://127.0.0.1:7870)
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import webbrowser
from collections import OrderedDict
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import settings as app_settings
from .models import MODELS as MODEL_NAMES
from .models import get_adapter
from .caption import MODELS as CAPTION_MODELS
from .config import DEFAULT_SAMPLE_PROMPTS, RunConfig, save_config
from .data import IMAGE_EXTS, scan_folder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "output")
RUN_ROOTS = [OUTPUT, os.path.join(ROOT, "tests", "out")]
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CACHE = os.path.join(ROOT, "cache")
PY = sys.executable
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

PRESETS = {
    "character": {"label": "캐릭터", "hint": "1024px, 얼굴과 정체성 위주",
                  "resolution": 1024, "rank": 32, "alpha": 32, "start_lr": 1e-4, "min_lr": 3e-5, "max_lr": 1e-4,
                  "caption_dropout": 0.05, "max_steps": "auto"},
    "character_fast": {"label": "캐릭터 빠르게", "hint": "768px, 약 2배 빠름",
                       "resolution": 768, "rank": 32, "alpha": 32, "start_lr": 1e-4, "min_lr": 3e-5, "max_lr": 1e-4,
                       "caption_dropout": 0.05, "max_steps": "auto"},
    "style": {"label": "화풍", "hint": "낮은 LR, 캡션 드롭아웃 10%",
              "resolution": 1024, "rank": 32, "alpha": 32, "start_lr": 5e-5, "min_lr": 1e-5, "max_lr": 8e-5,
              "caption_dropout": 0.1, "max_steps": "auto"},
    "test": {"label": "빠른 테스트", "hint": "512px, 300 step만",
             "resolution": 512, "rank": 16, "alpha": 16, "start_lr": 1e-4, "min_lr": 3e-5, "max_lr": 1e-4,
             "caption_dropout": 0.05, "max_steps": 300},
}
DEFAULT_SAMPLES = list(DEFAULT_SAMPLE_PROMPTS)

PROCS: Dict[str, subprocess.Popen] = {}
STATE = {"train_run": None}
CHILD_ENV = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
             "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "OPENCV_LOG_LEVEL": "ERROR"}


def console(msg: str):
    """Print to the run.bat window (never fails on an odd console encoding)."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode(), flush=True)


BEST_LABELS = {"likeness": "닮음 최고", "val": "홀드아웃 최저"}


def best_marks(rd: str) -> Dict[int, str]:
    """step -> "닮음 최고 · 홀드아웃 최저" for the best-of checkpoints the trainer saved."""
    marks = read_json(os.path.join(rd, "best_marks.json")) or {}
    out: Dict[int, List[str]] = {}
    for metric, v in marks.items():
        if isinstance(v, dict) and "step" in v:
            out.setdefault(int(v["step"]), []).append(BEST_LABELS.get(metric, metric))
    return {s: " · ".join(ls) for s, ls in out.items()}


def _checkpoints(rd: str, run: str):
    """Comparable checkpoints of a run: [{"key", "label"}] (steps oldest first, then the pick)."""
    out = []
    if os.path.isdir(rd):
        marks = best_marks(rd)
        for f in os.listdir(rd):
            m = re.fullmatch(rf"{re.escape(run)}_(\d{{5}})\.safetensors", f)
            if m:
                s = int(m.group(1))
                out.append({"key": str(s), "label": f"step {s}" + (f" ({marks[s]})" if s in marks else "")})
        out.sort(key=lambda c: int(c["key"]))
        if os.path.isfile(os.path.join(rd, f"{run}.safetensors")):
            out.append({"key": "best", "label": "추천"})
    return out


def _resume_points(rd: str):
    try:
        from .train import resumable_steps

        return resumable_steps(rd)
    except Exception:
        return []


def echo_output(path: str, proc: subprocess.Popen, prefix: str, fmt=None):
    """Follow a child's output file and mirror each new line to the console until the child
    exits (then drain what is left). Starts at the file's current end (appended runs)."""
    start = os.path.getsize(path) if os.path.isfile(path) else 0

    def run():
        import time

        pos = start
        while True:
            done = proc.poll() is not None
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except FileNotFoundError:
                chunk = ""
            for line in chunk.splitlines():
                line = fmt(line) if fmt else line
                if line:
                    console(f"{prefix} {line}")
            if done:
                code = proc.returncode
                console(f"{prefix} {'끝났습니다' if code == 0 else f'비정상 종료 (코드 {code}) - 위 로그를 확인하세요'}")
                return
            time.sleep(0.5)

    threading.Thread(target=run, daemon=True).start()


def _caption_line(line: str) -> str:
    """The captioner prints JSON status lines; show the message (and the new caption)."""
    try:
        st = json.loads(line)
    except Exception:
        return line if not line.startswith(("Loading weights", "\r")) else ""
    msg = st.get("message", "")
    if st.get("caption"):
        msg += f" | {st.get('last')}: {st['caption'][:120]}"
    return msg

app = FastAPI(title="OS LoRA Studio")


# --------------------------------------------------------------------------- helpers
def _find_orphan(module: str) -> Optional[List[str]]:
    """A trainer / captioner started by an earlier server process (the server was restarted
    while it ran): return its command line so we keep treating it as running."""
    try:
        import psutil

        for p in psutil.process_iter(["cmdline"]):
            c = p.info["cmdline"] or []
            if module in c and p.pid != os.getpid():
                return c
    except Exception:
        pass
    return None


def running(key: str) -> bool:
    p = PROCS.get(key)
    if p is not None and p.poll() is None:
        return True
    orphan = _find_orphan({"train": "oslora.train", "caption": "oslora.caption", "compare": "oslora.compare"}[key])
    if orphan and key == "train":
        try:
            cfg = orphan[orphan.index("oslora.train") + 1]
            STATE["train_run"] = os.path.basename(os.path.dirname(cfg))
        except (ValueError, IndexError):
            pass
    return orphan is not None


PICKER = r'''
import os, sys, tkinter as tk
from tkinter import filedialog
kind, initial, title = sys.argv[1], sys.argv[2], sys.argv[3]
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
if kind == "folder":
    path = filedialog.askdirectory(parent=root, initialdir=initial or None, title=title, mustexist=True)
else:
    path = filedialog.askopenfilename(parent=root, initialdir=initial or None, title=title,
                                      filetypes=[("모델 파일", "*.safetensors"), ("모든 파일", "*.*")])
sys.stdout.write(os.path.normpath(path) if path else "")
'''


class PickReq(BaseModel):
    kind: str = "folder"  # folder | file
    initial: str = ""
    title: str = ""


@app.post("/api/pick")
def pick(req: PickReq):
    """Open the native Windows folder / file picker on this PC and return the chosen path.
    Runs in its own process so the dialog never blocks the server."""
    initial = req.initial.strip().strip('"')
    if initial and os.path.isfile(initial):
        initial = os.path.dirname(initial)
    if not initial or not os.path.isdir(initial):
        initial = os.path.expanduser("~")
    title = req.title or ("데이터셋 폴더 선택" if req.kind == "folder" else "모델 파일 선택")
    try:
        out = subprocess.run([PY, "-c", PICKER, req.kind, initial, title], capture_output=True, text=True,
                             encoding="utf-8", timeout=900, env=CHILD_ENV)
    except subprocess.TimeoutExpired:
        return {"path": ""}
    return {"path": out.stdout.strip()}


def read_json(path: str) -> Dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def run_path(run: str) -> str:
    if not run or "/" in run or "\\" in run or run.startswith("."):
        raise HTTPException(400, "잘못된 학습 이름입니다.")
    for root in RUN_ROOTS:
        p = os.path.join(root, run)
        if os.path.isfile(os.path.join(p, "config.yaml")):
            return p
    return os.path.join(OUTPUT, run)


def list_runs() -> List[Dict]:
    runs = {}
    for root in reversed(RUN_ROOTS):
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isfile(os.path.join(p, "config.yaml")):
                st = read_json(os.path.join(p, "status.json"))
                runs[d] = {
                    "name": d,
                    "phase": st.get("phase", "-"),
                    "step": st.get("step", 0),
                    "max_steps": st.get("max_steps"),
                    "updated": st.get("updated") or os.path.getmtime(p),
                    "test": root != OUTPUT,
                    "alive": STATE["train_run"] == d and running("train"),
                }
    return sorted(runs.values(), key=lambda r: r["updated"], reverse=True)


def allowed_image(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(IMAGE_EXTS + (".jpg",))


class _Thumbs:
    def __init__(self, cap=600):
        self.cap, self.data, self.lock = cap, OrderedDict(), threading.Lock()

    def get(self, path: str, size: int) -> bytes:
        key = (path, size, os.path.getmtime(path))
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=86)
        data = buf.getvalue()
        with self.lock:
            self.data[key] = data
            while len(self.data) > self.cap:
                self.data.popitem(last=False)
        return data


THUMBS = _Thumbs()


# --------------------------------------------------------------------------- basics
class ModelPathsReq(BaseModel):
    model: str = "moss_image"
    dit: Optional[str] = None
    inference_dit: Optional[str] = None
    text_encoder: Optional[str] = None
    vae: Optional[str] = None


@app.post("/api/settings/models")
def set_models(req: ModelPathsReq):
    """Remember the model file paths (settings.json) and report which ones exist."""
    if req.model not in app_settings.ALL_FILES:
        raise HTTPException(400, f"지원하지 않는 모델: {req.model}")
    app_settings.set_model_paths({k: v for k, v in req.dict().items() if k != "model"}, req.model)
    return {"models": app_settings.model_status(req.model)}


@app.get("/api/defaults")
def defaults():
    d = RunConfig()
    return {
        "presets": PRESETS,
        "samples": DEFAULT_SAMPLES,
        "sample_size": [d.sample.width, d.sample.height],
        # changes when the default prompts/size change, so browsers drop their remembered copy once
        "samples_sig": hashlib.sha1(json.dumps([DEFAULT_SAMPLES, d.sample.width, d.sample.height]).encode()).hexdigest()[:12],
        "paths": {"dit": d.paths.dit, "inference_dit": d.paths.inference_dit, "text_encoder": d.paths.text_encoder,
                  "vae": d.paths.vae},
        "models": app_settings.model_status(),
        # every trainable model: its files (paths + download links) and display names
        "model_list": {k: v for k, v in MODEL_NAMES.items()},
        "model_files": {k: app_settings.model_status(k) for k in MODEL_NAMES},
        "inference_labels": {k: get_adapter(k).inference_label for k in MODEL_NAMES},
        "fixed_steps": {k: get_adapter(k).sample_steps for k in MODEL_NAMES if get_adapter(k).distilled_steps},
        "caption_models": {k: v["label"] for k, v in CAPTION_MODELS.items()},
        # caption models run by llama.cpp -> their file set key (paths set / downloaded in the UI)
        "caption_llama": {k: v["gguf"] for k, v in CAPTION_MODELS.items() if v.get("quant") == "llama.cpp"},
        "busy": {"train": running("train"), "caption": running("caption"), "train_run": STATE["train_run"]},
    }


class _GPU:
    """NVML handle, opened lazily; None when there is no NVIDIA driver."""

    handle = None
    tried = False

    @classmethod
    def get(cls):
        if not cls.tried:
            cls.tried = True
            try:
                import pynvml

                pynvml.nvmlInit()
                cls.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                cls.name = pynvml.nvmlDeviceGetName(cls.handle)
            except Exception:
                cls.handle = None
        return cls.handle


@app.get("/api/system")
def system():
    import psutil

    vm = psutil.virtual_memory()
    out = {"cpu": psutil.cpu_percent(interval=None), "ram_used": vm.used / 2**30, "ram_total": vm.total / 2**30}
    h = _GPU.get()
    if h is not None:
        import pynvml

        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        out.update(gpu=pynvml.nvmlDeviceGetUtilizationRates(h).gpu, vram_used=mem.used / 2**30,
                   vram_total=mem.total / 2**30, gpu_name=_GPU.name, gpu_temp=pynvml.nvmlDeviceGetTemperature(h, 0))
        try:
            out.update(power_w=pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                       power_limit_w=pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000)
        except pynvml.NVMLError:
            pass
    return out


@app.get("/api/thumb")
def thumb(path: str, size: int = 384):
    if not allowed_image(path):
        raise HTTPException(404, "이미지를 찾을 수 없습니다.")
    return Response(THUMBS.get(path, max(64, min(size, 2048))), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


@app.get("/api/image")
def image(path: str):
    if not allowed_image(path):
        raise HTTPException(404, "이미지를 찾을 수 없습니다.")
    return FileResponse(path)


# --------------------------------------------------------------------------- dataset
class FolderReq(BaseModel):
    folder: str


@app.post("/api/dataset")
def dataset(req: FolderReq):
    folder = req.folder.strip().strip('"')
    if not os.path.isdir(folder):
        raise HTTPException(404, "폴더를 찾을 수 없습니다. 경로를 확인하세요.")
    items = []
    for path, cap in scan_folder(folder):
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        items.append({"path": path, "name": os.path.basename(path), "caption": cap, "w": w, "h": h})
    n = len(items)
    holdout = min(6, max(2, round(n * 0.08))) if n >= 16 else 0
    return {"folder": folder, "items": items, "holdout": holdout,
            "captioned": sum(1 for i in items if i["caption"]),
            "small": sum(1 for i in items if 0 < i["w"] * i["h"] < 700 * 700)}


DIAG: Dict = {"running": False, "folder": None, "done": 0, "total": 0, "current": "", "report": None, "error": None}


class DiagReq(BaseModel):
    folder: str
    resolution: int = 1024


@app.post("/api/diagnose")
def diagnose_start(req: DiagReq):
    """Dataset check-up in a background thread (CPU only; fine while training runs)."""
    folder = req.folder.strip().strip('"')
    if not os.path.isdir(folder):
        raise HTTPException(404, "폴더를 찾을 수 없습니다.")
    if DIAG["running"]:
        return {"ok": True}
    DIAG.update(running=True, folder=folder, done=0, total=0, current="", report=None, error=None)

    def work():
        from .diagnose import analyze

        try:
            def progress(i, n, name):
                DIAG.update(done=i, total=n, current=name)

            DIAG["report"] = analyze(folder, req.resolution, progress)
        except Exception as e:
            DIAG["error"] = str(e)
        finally:
            DIAG["running"] = False

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


@app.get("/api/diagnose/status")
def diagnose_status():
    rep = DIAG["report"]
    out = {k: v for k, v in DIAG.items() if k != "report"}
    if rep:
        out["findings"] = rep["findings"]
        out["count"] = len(rep["items"])
    return out


class AutoReq(BaseModel):
    folder: str
    kind: str = "character"
    trigger: str = ""
    model: str = "moss_image"


@app.post("/api/autoconfig")
def autoconfig(req: AutoReq):
    """Training settings chosen from the dataset check-up (runs the check-up if needed)."""
    from .autoconfig import recommend
    from .diagnose import analyze

    folder = req.folder.strip().strip('"')
    if not os.path.isdir(folder):
        raise HTTPException(404, "데이터셋 폴더를 먼저 불러오세요.")
    report = analyze(folder, 1024)
    return recommend(report, req.kind if req.kind in ("character", "style", "object") else "character",
                     req.trigger, OUTPUT, model=req.model if req.model in MODEL_NAMES else "moss_image")


class FixReq(BaseModel):
    folder: str
    action: str
    files: List[str]


@app.post("/api/diagnose/fix")
def diagnose_fix(req: FixReq):
    from .diagnose import fix

    folder = req.folder.strip().strip('"')
    if not os.path.isdir(folder):
        raise HTTPException(404, "폴더를 찾을 수 없습니다.")
    try:
        res = fix(folder, req.action, req.files)
    except ValueError as e:
        raise HTTPException(400, str(e))
    console(f"[진단] {res['message']}")
    return res


class CaptionSave(BaseModel):
    path: str
    caption: str


@app.post("/api/caption/save")
def caption_save(req: CaptionSave):
    if not allowed_image(req.path):
        raise HTTPException(404, "이미지를 찾을 수 없습니다.")
    with open(os.path.splitext(req.path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(req.caption.strip())
    return {"ok": True}


class CaptionStart(BaseModel):
    folder: str
    model: str = "qwen3vl-4b"
    style: str = "medium"
    subject: str = "character"
    trigger: str = ""
    extra: str = ""
    overwrite: bool = False


class LlamaPathsReq(BaseModel):
    key: str
    server: Optional[str] = None
    model: Optional[str] = None
    mmproj: Optional[str] = None


class LlamaDownloadReq(BaseModel):
    key: str
    parts: List[str]


def _llama_key(key: str) -> str:
    if key not in app_settings.LLAMA_MODELS:
        raise HTTPException(400, f"알 수 없는 캡션 모델: {key}")
    return key


@app.get("/api/caption/llama/{key}")
def llama_files(key: str):
    """llama.cpp caption model files (server / GGUF / mmproj) and the download in progress."""
    from . import downloads

    return {"status": app_settings.llama_status(_llama_key(key)), "download": downloads.status()}


@app.post("/api/caption/llama")
def llama_set(req: LlamaPathsReq):
    app_settings.set_llama_paths(_llama_key(req.key), server=req.server, model=req.model, mmproj=req.mmproj)
    return {"status": app_settings.llama_status(req.key)}


@app.post("/api/caption/llama/download")
def llama_download(req: LlamaDownloadReq):
    from . import downloads

    parts = [p for p in req.parts if p in ("server", "model", "mmproj")]
    if not parts:
        raise HTTPException(400, "받을 파일을 고르세요.")
    err = downloads.start(_llama_key(req.key), parts)
    if err:
        raise HTTPException(409, err)
    console(f"[다운로드] 캡션 모델 {req.key}: {', '.join(parts)}")
    return {"ok": True}


@app.post("/api/caption/llama/cancel")
def llama_cancel():
    from . import downloads

    downloads.cancel()
    return {"ok": True}


@app.post("/api/caption/start")
def caption_start(req: CaptionStart):
    folder = req.folder.strip().strip('"')
    if not os.path.isdir(folder):
        raise HTTPException(400, "데이터셋 폴더를 먼저 불러오세요.")
    if running("caption"):
        raise HTTPException(409, "캡션 작업이 이미 실행 중입니다.")
    if running("train"):
        raise HTTPException(409, "학습 중에는 VRAM이 부족해 캡션을 만들 수 없습니다. 학습이 끝난 뒤 실행하세요.")
    if running("compare"):
        raise HTTPException(409, "체크포인트 비교가 끝난 뒤 캡션을 만드세요.")
    spec = CAPTION_MODELS.get(req.model, {})
    try:
        if spec.get("quant") == "comfy-int8":
            app_settings.require_model("text_encoder")
        elif spec.get("quant") == "llama.cpp":
            app_settings.require_llama(spec["gguf"])
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    os.makedirs(CACHE, exist_ok=True)
    status = os.path.join(CACHE, "caption_status.json")
    if os.path.exists(status):
        os.remove(status)
    cmd = [PY, "-m", "oslora.caption", "--folder", folder, "--model", req.model, "--style", req.style,
           "--subject", req.subject, "--trigger", req.trigger, "--extra", req.extra, "--status", status]
    if req.overwrite:
        cmd.append("--overwrite")
    log_path = os.path.join(CACHE, "caption_log.txt")
    log = open(log_path, "w", encoding="utf-8")
    PROCS["caption"] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=NO_WINDOW, env=CHILD_ENV)
    console(f"\n[캡션] 시작: {folder} (모델 {req.model}, {req.style}/{req.subject})")
    echo_output(log_path, PROCS["caption"], "[캡션]", _caption_line)
    return {"ok": True}


@app.get("/api/caption/status")
def caption_status():
    st = read_json(os.path.join(CACHE, "caption_status.json"))
    p = PROCS.get("caption")
    st["running"] = running("caption")
    if p is not None and p.poll() not in (None, 0) and st.get("phase") not in ("done", "error"):
        try:
            with open(os.path.join(CACHE, "caption_log.txt"), encoding="utf-8") as f:
                tail = f.read()[-1200:]
        except Exception:
            tail = ""
        st.update(phase="error", message="캡션 프로세스가 비정상 종료됐습니다.", log=tail)
    return st


@app.post("/api/caption/stop")
def caption_stop():
    p = PROCS.get("caption")
    if p is not None and p.poll() is None:
        p.terminate()
    return {"ok": True}


# --------------------------------------------------------------------------- training
class TrainReq(BaseModel):
    name: str
    folder: str
    model: str = "moss_image"
    trigger: str = ""
    resolution: int = 1024
    rank: int = 32
    alpha: float = 32
    max_steps: str = "auto"
    autopilot: bool = True
    start_lr: float = 1e-4
    min_lr: float = 3e-5
    max_lr: float = 1e-4
    early_stop: bool = True
    holdout: str = "auto"
    caption_dropout: float = 0.05
    photo_texture: bool = False  # a quarter of the steps see the image with phone sensor noise
    save_keep: int = 3  # regular checkpoints kept (0 = all)
    cleanup: bool = True  # drop resume states / quality renders when done
    int8_compute: bool = False  # int8 tensor cores for the frozen weights (faster, experimental)
    multires: bool = False  # also train at 512 / 768 (sampled evenly with the main resolution)
    ema: float = 0.99
    optimizer: str = "adamw"
    lr: float = 1e-4
    samples: List[str] = []
    sample_every: int = 500
    sample_w: int = 640
    sample_h: int = 864
    dit: str = ""
    inference_dit: Optional[str] = None  # None = default SFT, "" = render on the training DiT
    text_encoder: str = ""
    vae: str = ""
    queue: bool = False  # add to the queue when something is running (instead of refusing)
    kind: str = "character"


# --------------------------------------------------------------------------- training queue
QUEUE_PATH = os.path.join(CACHE, "queue.json")
QUEUE: Dict = {"items": [], "paused": False, "last": None}


def load_queue():
    try:
        with open(QUEUE_PATH, encoding="utf-8") as f:
            QUEUE.update(json.load(f))
    except Exception:
        pass


def save_queue():
    os.makedirs(CACHE, exist_ok=True)
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(QUEUE, f, ensure_ascii=False)


def _queue_worker():
    """Start the next queued training whenever the GPU is free. A run that ended with an error
    or a user stop pauses the queue (the user resumes it from the UI)."""
    import time

    while True:
        time.sleep(5)
        try:
            if running("train") or running("caption") or running("compare"):
                continue
            last = QUEUE.get("last")
            if last:
                st = read_json(os.path.join(run_path(last), "status.json"))
                QUEUE["last"] = None
                if st.get("phase") == "error" or "사용자 중지" in st.get("message", ""):
                    if QUEUE["items"]:
                        QUEUE["paused"] = True
                        console(f"[대기열] '{last}'이(가) {'오류로' if st.get('phase') == 'error' else '중지로'} 끝나서 "
                                "대기열을 멈췄어요. 화면에서 '대기열 다시 시작'을 누르면 이어서 진행해요.")
                save_queue()
            if QUEUE["paused"] or not QUEUE["items"]:
                continue
            entry = QUEUE["items"].pop(0)
            save_queue()
            try:
                r = train_start(TrainReq(**entry))
                QUEUE["last"] = r["run"]
                save_queue()
                console(f"[대기열] 다음 학습 시작: {r['run']} (남은 대기 {len(QUEUE['items'])}개)")
            except HTTPException as e:
                console(f"[대기열] '{entry.get('name')}' 시작 실패: {e.detail} → 건너뛰어요")
        except Exception as e:  # never let the worker die
            console(f"[대기열] 오류: {e}")


@app.get("/api/queue")
def queue_list():
    items = [{"name": q["name"], "folder": q["folder"], "resolution": q["resolution"], "max_steps": q["max_steps"]}
             for q in QUEUE["items"]]
    return {"items": items, "paused": QUEUE["paused"]}


class QueueOp(BaseModel):
    op: str  # remove | up | down | resume | clear
    index: int = -1


@app.post("/api/queue")
def queue_op(req: QueueOp):
    items = QUEUE["items"]
    i = req.index
    if req.op == "remove" and 0 <= i < len(items):
        items.pop(i)
    elif req.op == "up" and 0 < i < len(items):
        items[i - 1], items[i] = items[i], items[i - 1]
    elif req.op == "down" and 0 <= i < len(items) - 1:
        items[i + 1], items[i] = items[i], items[i + 1]
    elif req.op == "resume":
        QUEUE["paused"] = False
    elif req.op == "clear":
        items.clear()
    save_queue()
    return queue_list()


@app.post("/api/train/start")
def train_start(req: TrainReq):
    name = req.name.strip()
    folder = req.folder.strip().strip('"')
    if not name or any(c in name for c in '\\/:*?"<>|'):
        raise HTTPException(400, "LoRA 이름에 쓸 수 없는 문자가 있습니다.")
    if not os.path.isdir(folder):
        raise HTTPException(400, "데이터셋 폴더를 먼저 불러오세요.")
    if req.model not in MODEL_NAMES:
        raise HTTPException(400, f"지원하지 않는 모델: {req.model}")
    d = RunConfig(model=req.model)
    paths = {"dit": req.dit or d.paths.dit, "text_encoder": req.text_encoder or d.paths.text_encoder, "vae": req.vae or d.paths.vae}
    paths = {k: p.strip().strip('"') for k, p in paths.items()}
    for key, p in paths.items():
        if not p or not os.path.isfile(p):
            info = app_settings.ALL_FILES[req.model][key]
            raise HTTPException(400, f"{info['label']} 파일이 {'지정되지 않았어요' if not p else '없어요: ' + p}. "
                                     f"학습 설정 > 모델 파일에서 경로를 지정하세요. 없다면 받기: {info['url']}")
    inference_dit = d.paths.inference_dit if req.inference_dit is None else req.inference_dit.strip().strip('"')
    if inference_dit and not os.path.isfile(inference_dit):
        raise HTTPException(400, f"샘플용 DiT 파일이 없습니다: {inference_dit} (비우면 학습 모델로 그려요)")
    app_settings.set_model_paths(dict(paths, inference_dit=inference_dit), req.model)
    run_dir = os.path.join(OUTPUT, name)
    if os.path.isfile(os.path.join(run_dir, f"{name}.safetensors")):
        raise HTTPException(409, f"'{name}' 결과가 이미 있습니다. 이름을 바꾸세요 (예: {name}_v2).")
    busy = running("train") or running("caption") or running("compare")
    if busy or req.queue:
        if any(q["name"] == name for q in QUEUE["items"]) or (busy and STATE.get("train_run") == name and running("train")):
            raise HTTPException(409, f"'{name}'은(는) 이미 실행 중이거나 대기열에 있어요. 이름을 바꾸세요.")
        if not req.queue:
            what = f"'{STATE['train_run']}' 학습" if running("train") else ("캡션 작업" if running("caption") else "체크포인트 비교")
            raise HTTPException(409, f"{what}이 실행 중입니다. 끝나거나 중지한 뒤 시작하세요.")
        entry = req.dict()
        entry["queue"] = False
        QUEUE["items"].append(entry)
        save_queue()
        console(f"[대기열] 추가: {name} ({len(QUEUE['items'])}번째)")
        return {"ok": True, "queued": True, "position": len(QUEUE["items"]), "run": name}

    cfg = RunConfig(name=name, output_dir=OUTPUT, model=req.model)
    cfg.paths.dit, cfg.paths.text_encoder, cfg.paths.vae = paths["dit"], paths["text_encoder"], paths["vae"]
    adapter = get_adapter(req.model)  # how this model renders samples (Krea 2 Turbo: 8 steps, no CFG)
    cfg.sample.steps, cfg.sample.cfg = adapter.sample_steps, adapter.sample_cfg
    cfg.autopilot.quality_steps = adapter.quality_steps
    cfg.paths.inference_dit = inference_dit
    cfg.dataset.folder, cfg.dataset.trigger = folder, req.trigger.strip()
    cfg.dataset.resolution = req.resolution
    cfg.dataset.holdout = req.holdout if req.holdout in ("auto", "off") else int(req.holdout)
    cfg.dataset.caption_dropout = req.caption_dropout
    from . import photoreal

    cfg.dataset.photo_texture = photoreal.RATE if req.photo_texture else 0.0
    cfg.train.int8_compute = bool(req.int8_compute)
    cfg.dataset.multires = [r for r in (512, 768) if r < req.resolution] if req.multires else []
    cfg.save_keep, cfg.cleanup = max(0, int(req.save_keep)), bool(req.cleanup)
    cfg.network.rank, cfg.network.alpha = req.rank, req.alpha
    ms = req.max_steps.strip()
    cfg.train.max_steps = "auto" if ms in ("", "auto") else int(ms)
    cfg.train.ema_decay, cfg.train.optimizer, cfg.train.lr = req.ema, req.optimizer, req.lr
    cfg.autopilot.enabled, cfg.autopilot.early_stop = req.autopilot, req.early_stop
    cfg.autopilot.min_lr, cfg.autopilot.max_lr = req.min_lr, req.max_lr
    if not req.min_lr <= req.start_lr <= req.max_lr:
        raise HTTPException(400, "시작 LR은 최소 LR과 최대 LR 사이여야 합니다.")
    cfg.autopilot.start_lr = req.start_lr
    cfg.kind = req.kind if req.kind in ("character", "style", "object") else "character"
    from . import learn

    taste = learn.summarize(cfg.kind, cfg.model)
    cfg.autopilot.keep_frac = round(taste["keep_frac"], 3)  # learned from the user's picks
    cfg.sample.prompts = [s.strip() for s in req.samples if s.strip()]
    cfg.sample.width, cfg.sample.height = req.sample_w, req.sample_h
    cfg.sample.every = max(25, int(req.sample_every))

    os.makedirs(run_dir, exist_ok=True)
    for f in ("metrics.jsonl", "status.json", "log.txt", "STOP", "PAUSE"):
        if os.path.exists(os.path.join(run_dir, f)):
            os.remove(os.path.join(run_dir, f))
    cfg_path = os.path.join(run_dir, "config.yaml")
    save_config(cfg, cfg_path)
    out_path = os.path.join(run_dir, "stdout.txt")
    out = open(out_path, "w", encoding="utf-8")
    PROCS["train"] = subprocess.Popen([PY, "-m", "oslora.train", cfg_path], cwd=ROOT, stdout=out,
                                      stderr=subprocess.STDOUT, creationflags=NO_WINDOW, env=CHILD_ENV)
    STATE["train_run"] = name
    QUEUE["last"] = name  # lets the queue see how this run ended
    console(f"\n[학습] 시작: {name} ({req.resolution}px, 최대 {cfg.train.max_steps} step, "
            f"LR {req.start_lr:g} ({req.min_lr:g}~{req.max_lr:g}))")
    echo_output(out_path, PROCS["train"], "[학습]", _train_line)
    return {"ok": True, "run": name}


def _train_line(line: str) -> str:
    """Drop library chatter (progress bars, known harmless warnings) from the console."""
    noise = ("Loading weights", "fast path is not available", "flash-linear-attention", "causal-conv1d",
             "force_upcast", "kept in float32", "warnings.warn", "UserWarning", "LOAD REPORT", "UNEXPECTED",
             "position_ids", "Notes:", "can be ignored", "Key ", "-----")
    if not line.strip() or any(n in line for n in noise):
        return ""
    return line


class RunReq(BaseModel):
    run: str


@app.post("/api/train/stop")
def train_stop(req: RunReq):
    open(os.path.join(run_path(req.run), "STOP"), "w").close()
    return {"ok": True}


class ContinueReq(BaseModel):
    run: str
    step: int
    max_steps: Optional[int] = None
    int8_compute: Optional[bool] = None  # switch the fast int8 path on/off for the rest of the run


@app.post("/api/train/continue")
def train_continue(req: ContinueReq):
    """Continue a finished / stopped run from one of its resume points, in the same folder."""
    from .config import load_config
    from .train import resumable_steps

    if running("train"):
        raise HTTPException(409, f"'{STATE['train_run']}' 학습이 실행 중입니다. 끝나거나 중지한 뒤 이어서 학습하세요.")
    if running("caption"):
        raise HTTPException(409, "캡션 작업이 끝난 뒤 이어서 학습하세요.")
    if running("compare"):
        raise HTTPException(409, "체크포인트 비교가 끝난 뒤 이어서 학습하세요.")
    rd = run_path(req.run)
    points = {p["step"]: p for p in resumable_steps(rd)}
    if req.step not in points:
        raise HTTPException(400, f"step {req.step}에는 이어하기 지점이 없습니다.")
    cfg_path = os.path.join(rd, "config.yaml")
    cfg = load_config(cfg_path)
    if req.int8_compute is not None:
        cfg.train.int8_compute = bool(req.int8_compute)
    if req.max_steps:
        if req.max_steps <= req.step:
            raise HTTPException(400, f"최대 step은 {req.step}보다 커야 합니다.")
        cfg.train.max_steps = int(req.max_steps)
    cfg.output_dir = os.path.dirname(rd)
    save_config(cfg, cfg_path)
    for f in ("STOP", "PAUSE"):
        if os.path.exists(os.path.join(rd, f)):
            os.remove(os.path.join(rd, f))
    out_path = os.path.join(rd, "stdout.txt")
    out = open(out_path, "a", encoding="utf-8")
    PROCS["train"] = subprocess.Popen([PY, "-m", "oslora.train", cfg_path, "--resume", str(req.step)], cwd=ROOT,
                                      stdout=out, stderr=subprocess.STDOUT, creationflags=NO_WINDOW, env=CHILD_ENV)
    STATE["train_run"] = req.run
    QUEUE["last"] = req.run
    exact = "정확히" if points[req.step]["exact"] else "체크포인트에서"
    console(f"\n[학습] 이어서 학습: {req.run} step {req.step}부터 ({exact}), 최대 {cfg.train.max_steps} step")
    echo_output(out_path, PROCS["train"], "[학습]", _train_line)
    return {"ok": True, "run": req.run}


class CompareReq(BaseModel):
    run: str
    prompts: List[str]
    checkpoints: List[str] = []  # "base", "best", "500", ...; empty = all
    seed: int = 42
    strength: float = 1.0
    width: int = 768
    height: int = 1024
    steps: int = 24
    model: str = "sft"  # sft = the inference DiT (what people generate with), base = training DiT


@app.post("/api/compare/start")
def compare_start(req: CompareReq):
    """Render the chosen checkpoints on the same prompts/seeds and score them (GPU job)."""
    import time as _t

    for key, what in (("train", "학습"), ("caption", "캡션 작업"), ("compare", "다른 비교")):
        if running(key):
            raise HTTPException(409, f"{what}이(가) 실행 중이라 GPU가 비어 있지 않아요. 끝난 뒤 비교하세요.")
    prompts = [p.strip() for p in req.prompts if p.strip()]
    if not prompts:
        raise HTTPException(400, "비교할 프롬프트를 하나 이상 넣어주세요.")
    rd = run_path(req.run)
    out = os.path.join(rd, "compare", _t.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    body = req.dict()
    body["prompts"] = prompts
    body["checkpoints"] = [c if c in ("base", "best") else int(c) for c in req.checkpoints] or None
    req_path = os.path.join(out, "request.json")
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    log_path = os.path.join(out, "stdout.txt")
    log = open(log_path, "w", encoding="utf-8")
    PROCS["compare"] = subprocess.Popen([PY, "-m", "oslora.compare", rd, req_path], cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, creationflags=NO_WINDOW, env=CHILD_ENV)
    n_ck = len(body["checkpoints"]) if body["checkpoints"] else "전체"
    console(f"\n[비교] 시작: {req.run} (체크포인트 {n_ck}, 프롬프트 {len(prompts)}개)")
    echo_output(log_path, PROCS["compare"], "[비교]", _train_line)
    return {"ok": True, "id": os.path.basename(out)}


@app.get("/api/compare/{run}")
def compare_result(run: str, id: str = ""):
    rd = os.path.join(run_path(run), "compare")
    ids = sorted((d for d in os.listdir(rd) if os.path.isdir(os.path.join(rd, d))), reverse=True) if os.path.isdir(rd) else []
    if not ids:
        return {"ids": [], "result": None}
    cid = id if id in ids else ids[0]
    res = read_json(os.path.join(rd, cid, "result.json"))
    if res:
        res["dir"] = os.path.join(rd, cid)
        if res.get("status") in ("preparing", "running") and not running("compare"):
            res.update(status="error", message="비교 프로세스가 중간에 끝났어요 (stdout.txt 확인)")
    return {"ids": ids, "id": cid, "result": res}


class PromoteReq(BaseModel):
    run: str
    file: str
    picked_likeness: Optional[float] = None  # from the comparison table, when picked there
    top_likeness: Optional[float] = None


def _run_facts(rd: str) -> Dict:
    """Facts about a run the taste model needs: training-image count, kind, settings."""
    from .config import load_config
    from .data import pick_holdout, scan_folder

    facts = {}
    try:
        cfg = load_config(os.path.join(rd, "config.yaml"))
        n = len(scan_folder(cfg.dataset.folder, cfg.dataset.caption_ext))
        facts = {"model": cfg.model, "n_train": n - len(pick_holdout(n, cfg.dataset.holdout, cfg.train.seed)),
                 "kind": getattr(cfg, "kind", "character") or "character", "rank": cfg.network.rank,
                 "resolution": cfg.dataset.resolution, "start_lr": cfg.autopilot.start_lr}
    except Exception:
        pass
    return facts


@app.post("/api/promote")
def promote(req: PromoteReq):
    """"I used this one": make the chosen checkpoint the run's recommended file (the old one
    is kept as a backup) and record the choice so the recommendations learn the user's taste.
    Choosing the current recommended file just records agreement."""
    import shutil
    import time as _t

    from . import learn

    rd = run_path(req.run)
    src = os.path.join(rd, os.path.basename(req.file))
    best = os.path.join(rd, f"{req.run}.safetensors")
    if not os.path.isfile(src) or not src.endswith(".safetensors"):
        raise HTTPException(404, "체크포인트 파일을 찾을 수 없어요.")
    st = read_json(os.path.join(rd, "status.json"))
    m = re.search(r"_(\d{5})\.safetensors$", src)
    agree = os.path.abspath(src) == os.path.abspath(best)
    picked = st.get("best_step") if agree else (int(m.group(1)) if m else None)
    if picked is None and src.endswith("_last.safetensors"):
        picked = st.get("step")
    learn.record({"run": req.run, "picked": picked, "autopilot_best": st.get("best_step"),
                  "max_steps": st.get("max_steps"), "picked_likeness": req.picked_likeness,
                  "top_likeness": req.top_likeness, **_run_facts(rd)})
    if agree:
        console(f"[결과] {req.run}: 추천 파일을 그대로 쓰셨다고 기록했어요")
        return {"ok": True, "message": "추천 파일을 쓰셨다고 기록했어요. 다음 추천에 반영돼요."}
    if os.path.isfile(best):
        shutil.copy2(best, os.path.join(rd, f"{req.run}_이전추천_{_t.strftime('%H%M%S')}.safetensors"))
    shutil.copy2(src, best)
    console(f"[결과] {req.run}: 추천 파일을 {os.path.basename(src)}(으)로 바꿨어요 (이전 추천은 백업, 선택 기록됨)")
    return {"ok": True, "message": f"{os.path.basename(src)}을(를) 추천 파일로 바꾸고 선택을 기록했어요. "
                                   "이전 추천 파일은 백업해 뒀어요."}


@app.get("/api/learn")
def learn_summary(kind: str = "character", model: str = "moss_image"):
    from . import learn

    return learn.summarize(kind, model)


@app.post("/api/train/pause")
def train_pause(req: RunReq):
    """The trainer finishes its current step, moves everything off the GPU and waits."""
    open(os.path.join(run_path(req.run), "PAUSE"), "w").close()
    console(f"[학습] 일시정지 요청: {req.run} (현재 step을 마치고 GPU를 비워요)")
    return {"ok": True}


class ResumeWithReq(BaseModel):
    run: str
    int8_compute: Optional[bool] = None
    photo_texture: Optional[bool] = None
    min_lr: Optional[float] = None
    max_lr: Optional[float] = None
    max_steps: Optional[str] = None
    samples: Optional[List[str]] = None
    sample_every: Optional[int] = None
    sample_w: Optional[int] = None
    sample_h: Optional[int] = None


@app.post("/api/train/resume_with")
def train_resume_with(req: ResumeWithReq):
    """Resume a paused run with changed settings: the trainer reads overrides.json when the
    PAUSE file disappears and applies what can change mid-run (see Trainer._apply_overrides)."""
    rd = run_path(req.run)
    pause = os.path.join(rd, "PAUSE")
    if not os.path.exists(pause):
        raise HTTPException(409, "일시정지 중인 학습이 아니에요. 먼저 모니터 탭에서 일시정지하세요.")
    o = {k: v for k, v in req.dict().items() if k != "run" and v is not None}
    tmp = os.path.join(rd, "overrides.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(rd, "overrides.json"))
    os.remove(pause)
    console(f"[학습] 설정 반영해서 재개: {req.run}")
    return {"ok": True}


@app.post("/api/train/resume")
def train_resume(req: RunReq):
    p = os.path.join(run_path(req.run), "PAUSE")
    if os.path.exists(p):
        os.remove(p)
    console(f"[학습] 재개 요청: {req.run}")
    return {"ok": True}


@app.get("/api/runs")
def runs():
    return {"runs": list_runs()}


@app.get("/api/runs/{run}")
def run_detail(run: str, since: int = 0):
    rd = run_path(run)
    if not os.path.isdir(rd):
        raise HTTPException(404, "학습 기록이 없습니다.")
    st = read_json(os.path.join(rd, "status.json"))
    steps, evals, quality = [], [], []
    mp = os.path.join(rd, "metrics.jsonl")
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                kind = m.get("type")
                if kind == "eval":
                    evals.append(m)
                elif kind == "quality":
                    quality.append(m)
                elif kind == "step":
                    steps.append([m["step"], m["loss"], m["lr"], m.get("gnorm", 0)])
    samples = []
    sd = os.path.join(rd, "samples")
    if os.path.isdir(sd):
        for n in sorted(os.listdir(sd)):
            if n[0].isdigit() and n.endswith(".jpg"):
                step, idx = n[:-4].split("_")
                samples.append({"step": int(step), "prompt": int(idx), "path": os.path.join(sd, n)})
    leak_base = [os.path.join(sd, n) for n in sorted(os.listdir(sd))] if os.path.isdir(sd) else []
    leak_base = [p for p in leak_base if os.path.basename(p).startswith("leak_base")]
    files = []
    marks = best_marks(rd)
    for n in sorted(os.listdir(rd)):
        if n.endswith(".safetensors"):
            kind = "best" if n == f"{run}.safetensors" else ("last" if n.endswith("_last.safetensors") else "checkpoint")
            m = re.search(r"_(\d{5})\.safetensors$", n)
            files.append({"name": n, "kind": kind, "mb": round(os.path.getsize(os.path.join(rd, n)) / 2**20, 1),
                          "tag": marks.get(int(m.group(1))) if m else None})
    try:
        with open(os.path.join(rd, "log.txt"), encoding="utf-8") as f:
            log = f.readlines()[-60:]
    except Exception:
        log = []
    cfg = {}
    try:
        import yaml

        with open(os.path.join(rd, "config.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    alive = STATE["train_run"] == run and running("train")
    return {
        "name": run, "dir": rd, "status": st, "alive": alive, "pause_requested": os.path.exists(os.path.join(rd, "PAUSE")), "resume_points": _resume_points(rd), "checkpoints": _checkpoints(rd, run), "steps": steps, "evals": evals, "quality": quality,
        "samples": samples, "sample_prompts": (cfg.get("sample") or {}).get("prompts", []),
        "leak_base": leak_base, "files": files, "log": "".join(log),
        "config": {"model": cfg.get("model") or "moss_image",
                   "resolution": (cfg.get("dataset") or {}).get("resolution"),
                   "rank": (cfg.get("network") or {}).get("rank"),
                   "trigger": (cfg.get("dataset") or {}).get("trigger"),
                   "folder": (cfg.get("dataset") or {}).get("folder")},
    }


@app.post("/api/open")
def open_folder(req: RunReq):
    path = run_path(req.run) if req.run else OUTPUT
    os.makedirs(path, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def _silence_proactor_disconnects():
    """Windows' asyncio proactor logs a scary traceback ("WinError 10022/10054") whenever a
    browser drops a polling connection it was about to close anyway. Swallow only that case."""
    if os.name != "nt":
        return
    from asyncio import proactor_events

    original = proactor_events._ProactorBasePipeTransport._call_connection_lost

    def quiet(self, exc):
        try:
            original(self, exc)
        except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
            if getattr(e, "winerror", None) not in (10022, 10053, 10054) and not isinstance(
                    e, (ConnectionResetError, ConnectionAbortedError)):
                raise

    proactor_events._ProactorBasePipeTransport._call_connection_lost = quiet


def main():
    import uvicorn

    _silence_proactor_disconnects()
    load_queue()
    threading.Thread(target=_queue_worker, daemon=True).start()
    os.makedirs(OUTPUT, exist_ok=True)

    def _history():  # keep the training history DB up to date with every run folder
        try:
            from . import rundb

            rundb.ingest_all(OUTPUT)
        except Exception as e:
            console(f"[기록] 학습 기록 DB 갱신 실패: {e}")

    threading.Thread(target=_history, daemon=True).start()
    port = int(os.environ.get("OSLORA_PORT", "7870"))
    console("=" * 64)
    console(f" OS LoRA Studio 실행 중  →  http://127.0.0.1:{port}")
    console(" 학습과 캡션 진행 상황이 이 창에도 표시돼요. 이 창을 닫으면 프로그램이 꺼져요.")
    console("=" * 64)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
