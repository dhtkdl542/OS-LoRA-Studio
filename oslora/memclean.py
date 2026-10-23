"""Release RAM after a run without touching any other program.

Windows keeps every file a process read in its standby cache ("Cached" in Task Manager);
after a training run that is the model weights (DiT, text encoder, VAE: 15+ GB). Opening a
file with FILE_FLAG_NO_BUFFERING makes the cache manager flush and drop the cached pages of
*that file only*, so we do exactly that for the files the run loaded. No admin rights, no
system-wide purge, no processes closed.

    python -m oslora.memclean <file> [<file> ...]
"""

import ctypes
import gc
import os
import sys
from typing import Iterable, List

GENERIC_READ = 0x80000000
FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
INVALID_HANDLE = ctypes.c_void_p(-1).value


def drop_file_cache(path: str) -> bool:
    """Ask Windows to purge ``path`` from the standby cache. True on success."""
    if os.name != "nt" or not os.path.isfile(path):
        return False
    k32 = ctypes.windll.kernel32
    k32.CreateFileW.restype = ctypes.c_void_p
    h = k32.CreateFileW(ctypes.c_wchar_p(os.path.abspath(path)), GENERIC_READ, FILE_SHARE_ALL, None,
                        OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None)
    if h in (None, INVALID_HANDLE):
        return False
    k32.CloseHandle(ctypes.c_void_p(h))
    return True


def standby_bytes() -> float:
    """Current standby-cache size in bytes (Windows performance counters), or -1."""
    try:
        import subprocess

        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "((Get-Counter '\\Memory\\Standby Cache Normal Priority Bytes','\\Memory\\Standby Cache Reserve Bytes',"
             "'\\Memory\\Standby Cache Core Bytes').CounterSamples | Measure-Object CookedValue -Sum).Sum"],
            capture_output=True, text=True, timeout=30, creationflags=0x08000000,
        )
        return float(out.stdout.strip())
    except Exception:
        return -1.0


def release_process_memory():
    """Free what this process still holds (Python garbage, CUDA cache, trimmed working set)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    if os.name == "nt":
        try:
            k32 = ctypes.windll.kernel32
            k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        except Exception:
            pass


def drop_files(paths: Iterable[str]) -> List[str]:
    """Files another program still has mapped are left alone by Windows (the purge fails
    for them), so this never pulls weights out from under e.g. a running ComfyUI."""
    return [p for p in dict.fromkeys(paths) if p and drop_file_cache(p)]


def hf_cached_files(repo: str) -> List[str]:
    """Weight files of a Hugging Face repo already in the local cache."""
    try:
        from huggingface_hub import scan_cache_dir

        for r in scan_cache_dir().repos:
            if r.repo_id == repo:
                return [str(f.file_path) for rev in r.revisions for f in rev.files
                        if str(f.file_path).endswith((".safetensors", ".bin", ".onnx"))]
    except Exception:
        pass
    return []


def schedule_after_exit(paths: Iterable[str], log=None):
    """Spawn a detached helper that waits for *this* process to exit (its mappings of the
    weight files are only released then) and drops those files from the standby cache."""
    paths = [p for p in dict.fromkeys(paths) if p and os.path.isfile(p)]
    if not paths or os.name != "nt":
        return
    import subprocess

    flags = 0x00000008 | 0x08000000 | 0x00000200  # DETACHED | NO_WINDOW | NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, "-m", "oslora.memclean", "--wait-pid", str(os.getpid()), *paths],
                     cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     creationflags=flags, close_fds=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    if log:
        total = sum(os.path.getsize(p) for p in paths) / 2**30
        log(f"RAM 정리 예약: 프로세스 종료 후 모델 파일 {len(paths)}개({total:.1f} GB)의 캐시를 해제합니다")


def _wait_for_pid(pid: int, timeout: float = 600.0):
    try:
        import psutil

        psutil.Process(pid).wait(timeout=timeout)
    except Exception:
        pass


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--wait-pid":
        _wait_for_pid(int(args[1]))
        args = args[2:]
    before = standby_bytes()
    done = drop_files(args)
    after = standby_bytes()
    print(f"캐시 해제: {len(done)}/{len(args)}개 파일, 대기 캐시 {before / 2**30:.1f} GB → {after / 2**30:.1f} GB")
