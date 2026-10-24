"""Downloads for the caption models run by llama.cpp: the GGUF model + vision projector from
Hugging Face, and ``llama-server.exe`` (CUDA build + its CUDA runtime DLLs) from llama.cpp's
GitHub releases. Files land under ``models/llm/`` and their paths go into settings.json.

Plain HTTP streaming (no Hugging Face cache): a ``.part`` file is renamed when complete, so an
interrupted download never looks finished.
"""

import json
import os
import re
import shutil
import threading
import urllib.request
import zipfile
from typing import Callable, Dict, List, Optional

from . import settings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_DIR = os.path.join(ROOT, "models", "llm")
SERVER_DIR = os.path.join(LLM_DIR, "llama.cpp")
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
# CUDA 12.4 builds run on any driver from 550 up (the 13.x builds need 580+)
SERVER_ASSET = re.compile(r"llama-b\d+-bin-win-cuda-12\.\d+-x64\.zip")
CUDART_ASSET = re.compile(r"cudart-llama-bin-win-cuda-12\.\d+-x64\.zip")

STATE: Dict = {"running": False}
_lock = threading.Lock()


class Cancelled(Exception):
    pass


def _fetch(url: str, dest: str, progress: Callable[[int, int], None]):
    req = urllib.request.Request(url, headers={"User-Agent": "OS-LoRA-Studio"})
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if STATE.get("cancel"):
                raise Cancelled()
            chunk = r.read(4 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            progress(done, total)
    os.replace(part, dest)


def _hf_url(repo_url: str, filename: str) -> str:
    repo = repo_url.rstrip("/").split("huggingface.co/")[1]
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _server_assets() -> Dict[str, str]:
    """{name: url} of the newest llama.cpp build that has both Windows CUDA 12 zips."""
    req = urllib.request.Request(RELEASES_API, headers={"User-Agent": "OS-LoRA-Studio",
                                                        "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.load(r)
    for rel in releases:
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        server = next((n for n in assets if SERVER_ASSET.fullmatch(n)), None)
        cudart = next((n for n in assets if CUDART_ASSET.fullmatch(n)), None)
        if server and cudart:
            return {server: assets[server], cudart: assets[cudart]}
    raise RuntimeError("llama.cpp 릴리스에서 Windows CUDA 빌드를 찾지 못했어요")


def _set(**kw):
    with _lock:
        STATE.update(kw)


def _job(key: str, parts: List[str]):
    info = settings.LLAMA_MODELS[key]
    try:
        for part in parts:
            if part == "server":
                assets = _server_assets()
                os.makedirs(SERVER_DIR, exist_ok=True)
                for name, url in assets.items():
                    _set(part=part, file=name, done=0, total=0, message=f"llama-server 받는 중: {name}")
                    z = os.path.join(SERVER_DIR, name)
                    _fetch(url, z, lambda d, t: _set(done=d, total=t))
                    _set(message=f"압축 푸는 중: {name}")
                    with zipfile.ZipFile(z) as zf:
                        for m in zf.infolist():
                            if m.is_dir():
                                continue
                            # flatten: some builds put everything in a sub-folder
                            with zf.open(m) as src, open(os.path.join(SERVER_DIR, os.path.basename(m.filename)), "wb") as dst:
                                shutil.copyfileobj(src, dst)
                    os.remove(z)
                exe = os.path.join(SERVER_DIR, "llama-server.exe")
                if not os.path.isfile(exe):
                    raise RuntimeError("받은 파일에 llama-server.exe가 없어요")
                settings.set_llama_paths(key, server=exe)
            else:
                filename = info["model_file" if part == "model" else "mmproj_file"]
                dest = os.path.join(LLM_DIR, key, filename)
                _set(part=part, file=filename, done=0, total=0, message=f"받는 중: {filename}")
                _fetch(_hf_url(info["url"], filename), dest, lambda d, t: _set(done=d, total=t))
                settings.set_llama_paths(key, **{part: dest})
        _set(running=False, phase="done", message="다운로드 완료")
    except Cancelled:
        _set(running=False, phase="cancelled", message="다운로드를 취소했어요")
    except Exception as e:
        _set(running=False, phase="error", message=f"다운로드 실패: {e}")


def start(key: str, parts: List[str]) -> Optional[str]:
    """Start downloading ``parts`` (server / model / mmproj) of a caption model in the background.
    Returns an error message if one is already running."""
    if key not in settings.LLAMA_MODELS:
        return f"알 수 없는 캡션 모델: {key}"
    with _lock:
        if STATE.get("running"):
            return "다른 다운로드가 진행 중이에요"
        STATE.clear()
        STATE.update(running=True, phase="downloading", key=key, parts=parts, done=0, total=0, message="시작하는 중")
    threading.Thread(target=_job, args=(key, parts), daemon=True).start()
    return None


def cancel():
    _set(cancel=True)


def status() -> Dict:
    with _lock:
        return dict(STATE)
