"""Record each stage run (runtime, peak memory, hardware) to the runs table."""
import logging
import platform
import resource
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tpa_index.config import config_hash, load_labels, load_pipeline
from tpa_index.schema import write_table

log = logging.getLogger(__name__)


def device() -> str:
    import torch  # lazy: keeps the guard and CLI startup fast

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_name(dev: str | None = None) -> str:
    dev = dev or device()
    if dev == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine() or "unknown"


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9  # bytes on macOS, KiB on Linux


def _peak_mem_gb(dev: str) -> float | None:
    try:
        if dev == "cuda":
            import torch

            return torch.cuda.max_memory_allocated() / 1e9
        if dev == "mps":
            import torch

            # driver_allocated_memory is current, not peak; process peak RSS keeps it from under-reporting
            return max(torch.mps.driver_allocated_memory() / 1e9, _peak_rss_gb())
        return _peak_rss_gb()
    except Exception as e:  # unmeasurable: record null rather than a fake number
        log.warning("peak memory unavailable on %s: %s", dev, type(e).__name__)
        return None


@dataclass
class RunContext:
    stage: str
    run_id: str = field(default="")
    n_segments: int = 0


@contextmanager
def track_run(stage: str, model: str | None = None, cfg_hash: str | None = None, root=None):
    """Time a stage; on success write its runs row (marking the run complete). On error, write nothing and re-raise."""
    started = datetime.now(timezone.utc)
    ctx = RunContext(stage, run_id=f"{stage}-{started:%Y%m%dT%H%M%S}-{secrets.token_hex(2)}")
    dev = device()
    if dev == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield ctx
    except BaseException:
        log.error("run %s failed", ctx.run_id)
        raise
    write_table("runs", [{
        "run_id": ctx.run_id, "stage": stage, "model": model,
        "config_hash": cfg_hash or config_hash(load_labels(), load_pipeline()),
        "hardware": hardware_name(dev), "n_segments": ctx.n_segments,
        "runtime_s": max(time.perf_counter() - t0, 1e-6), "peak_mem_gb": _peak_mem_gb(dev), "started_at": started,
    }], ctx.run_id, root)
