"""Sample total GPU usage, including Warp allocations outside PyTorch's allocator."""

import subprocess
import threading
import psutil


class ResourceMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak_device_vram_bytes = 0
        self.peak_process_ram_bytes = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            self.peak_process_ram_bytes = max(
                self.peak_process_ram_bytes, psutil.Process().memory_info().rss
            )
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if p.returncode == 0:
                self.peak_device_vram_bytes = max(
                    self.peak_device_vram_bytes, int(p.stdout.splitlines()[0]) * 1024**2
                )
            self.stop_event.wait(0.5)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
