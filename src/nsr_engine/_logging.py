from __future__ import annotations

import time


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    out[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def _read_rss() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return float("nan")


def memory_usage() -> dict[str, float]:
    mi = _read_meminfo()
    total = mi.get("MemTotal", 0)
    avail = mi.get("MemAvailable", 0)
    swap_total = mi.get("SwapTotal", 0)
    swap_free = mi.get("SwapFree", 0)
    return {
        "rss": _read_rss(),
        "sys_used": float(total - avail) if total else float("nan"),
        "sys_total": float(total) if total else float("nan"),
        "swap_used": float(swap_total - swap_free) if swap_total else float("nan"),
        "swap_total": float(swap_total) if swap_total else float("nan"),
    }


def format_memory() -> str:
    _GIB = 1024.0**3
    m = memory_usage()

    def g(x: float) -> str:
        return "?" if x != x else f"{x / _GIB:.1f}"

    return (
        f"rss={g(m['rss'])}GiB "
        f"phys={g(m['sys_used'])}/{g(m['sys_total'])}GiB "
        f"swap={g(m['swap_used'])}/{g(m['swap_total'])}GiB"
    )


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


class Heartbeat:
    """Throttled progress logger printing elapsed time and memory usage."""

    def __init__(self, label: str, *, interval_s: float = 30.0) -> None:
        self.label = label
        self.interval_s = interval_s
        self._start = time.monotonic()
        self._last = 0.0

    def beat(self, msg: str = "", *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last) < self.interval_s:
            return
        self._last = now
        elapsed = format_elapsed(now - self._start)
        suffix = f"  {msg}" if msg else ""
        print(
            f"[hb] {self.label}  t=+{elapsed}  {format_memory()}{suffix}",
            flush=True,
        )
