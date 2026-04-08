import functools
import threading
import time
import tracemalloc
from pathlib import Path

import psutil

_THRESHOLD_BYTES = 50 * 1024 * 1024
_PKG_ROOT = Path(__file__).parent.parent  # nablachem/


class _Section:
    def __init__(self, tracker: "PerformanceTracker", label: str):
        self._tracker = tracker
        self._label = label

    def __enter__(self):
        t = self._tracker
        entry_rss = t._process.memory_info().rss
        with t._lock:
            saved_parent_peak = t._peak_rss
            t._peak_rss = entry_rss
        self._entry_time = time.perf_counter()
        self._entry_rss = entry_rss
        self._saved_parent_peak = saved_parent_peak
        self._depth = len(t._stack)
        t._stack.append(self)
        self._record = {}
        t._results.append(self._record)
        return self

    def __exit__(self, *args):
        t = self._tracker
        exit_time = time.perf_counter()
        with t._lock:
            exit_peak = t._peak_rss
            t._peak_rss = max(self._saved_parent_peak, exit_peak)
        t._stack.pop()
        additional_mb = (exit_peak - self._entry_rss) / (1024**2)
        self._record.update(
            {
                "label": self._label,
                "depth": self._depth,
                "duration_s": exit_time - self._entry_time,
                "mem_at_entry_mb": self._entry_rss / (1024**2),
                "additional_mem_mb": max(0.0, additional_mb),
            }
        )

    def __call__(self, func):
        return self._tracker._wrap(func, self._label)


class PerformanceTracker:
    def __init__(self, poll_interval_s: float = 0.1):
        self._lock = threading.Lock()
        self._peak_rss: int = 0
        self._stack: list[_Section] = []
        self._results: list[dict] = []
        self._process = psutil.Process()
        self._poll_interval = poll_interval_s
        self._watchdog_thread: threading.Thread | None = None

    def start_memory_tracking(self) -> None:
        t = threading.Thread(target=self._watchdog, daemon=True)
        t.start()
        self._watchdog_thread = t

    def _watchdog(self):
        while True:
            time.sleep(self._poll_interval)
            rss = self._process.memory_info().rss
            with self._lock:
                if rss > self._peak_rss:
                    self._peak_rss = rss

    def _wrap(self, func, label: str):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _Section(self, label):
                return func(*args, **kwargs)

        return wrapper

    def track(self, label_or_func):
        if callable(label_or_func):
            return self._wrap(label_or_func, label_or_func.__name__)
        return _Section(self, label_or_func)

    @property
    def results(self) -> list[dict]:
        return list(self._results)

    def summary(self, mode: str) -> None:
        if mode == "compute":
            visible = [
                r for r in self._results if r["duration_s"] >= 1
            ]
            if not visible:
                print("No sections exceeded 1 s.")
                return
            label_col = max(len(r["label"]) + r["depth"] * 2 for r in visible)
            label_col = max(label_col, 7)
            header = f"{'Section':<{label_col}}  {'Duration':>10}"
            print(header)
            print("─" * len(header))
            for r in visible:
                indent = "  " * r["depth"]
                label = indent + r["label"]
                print(f"{label:<{label_col}}  {r['duration_s']:>9.3f} s")

        elif mode == "memory":
            visible = [
                r for r in self._results if r["additional_mem_mb"] >= 100
            ]
            if not visible:
                print("No sections exceeded 100 MB.")
            else:
                label_col = max(len(r["label"]) + r["depth"] * 2 for r in visible)
                label_col = max(label_col, 7)
                header = f"{'Section':<{label_col}}  {'+Memory':>10}"
                print(header)
                print("─" * len(header))
                for r in visible:
                    indent = "  " * r["depth"]
                    label = indent + r["label"]
                    print(f"{label:<{label_col}}  {r['additional_mem_mb']:>8.1f} MB")

            if not tracemalloc.is_tracing():
                return

            snapshot = tracemalloc.take_snapshot()

            # Merge all tracebacks by their outermost package frame so each
            # source line appears exactly once.
            by_location: dict[str, int] = {}
            for stat in snapshot.statistics("traceback"):
                location = None
                for frame in reversed(stat.traceback):
                    try:
                        if Path(frame.filename).is_relative_to(_PKG_ROOT):
                            rel = Path(frame.filename).relative_to(_PKG_ROOT)
                            location = f"{rel}:{frame.lineno}"
                            break
                    except ValueError:
                        continue
                if location is None:
                    frame = stat.traceback[-1]
                    location = f"{frame.filename}:{frame.lineno}"
                by_location[location] = by_location.get(location, 0) + stat.size

            large = [
                (loc, size)
                for loc, size in by_location.items()
                if size >= _THRESHOLD_BYTES
            ]
            large.sort(key=lambda t: t[1], reverse=True)

            if not large:
                return

            print(f"\nLive allocations by source line (>= 50 MB, total across all call paths):")
            print(f"{'Live memory':>12}  Location")
            print("─" * 60)
            for loc, size in large:
                size_mb = size / (1024**2)
                print(f"{size_mb:>10.1f} MB  {loc}")
