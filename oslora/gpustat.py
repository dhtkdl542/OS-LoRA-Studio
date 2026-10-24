"""GPU utilisation and power draw sampled in the background (NVML), averaged over chosen spans.

The trainer switches sampling on only while it computes training steps, so the averages say how
busy the GPU is during training itself: a step that waits on block-swap copies or the CPU shows
up as low utilisation and power at the same s/step, a compute-bound one near the power limit.
"""

import threading
import time
from typing import Dict, Optional


class GpuMonitor:
    def __init__(self, interval: float = 0.5, index: int = 0):
        self.handle = None
        self.active = False
        self.limit_w = None
        self._lock = threading.Lock()
        self._n = self._util = self._watts = 0.0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000
        except Exception:
            self.handle = None
            return
        threading.Thread(target=self._run, args=(interval,), daemon=True).start()

    def _run(self, interval: float):
        while True:
            if self.active:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                    watts = self._nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
                except Exception:
                    return
                with self._lock:
                    self._n += 1
                    self._util += util
                    self._watts += watts
            time.sleep(interval)

    def take(self) -> Optional[Dict[str, float]]:
        """Averages since the last call ({util %, watts, limit_w}), or None without samples."""
        with self._lock:
            n, util, watts = self._n, self._util, self._watts
            self._n = self._util = self._watts = 0.0
        if not n:
            return None
        return {"util": util / n, "watts": watts / n, "limit_w": self.limit_w}
