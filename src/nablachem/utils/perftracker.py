import functools
import threading
import time

import psutil


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
        t = threading.Thread(target=self._watchdog, daemon=True)
        t.start()

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

    def summary(self, min_duration_s: float = 1, min_memory_mb: float = 100.0) -> None:
        visible = [
            r
            for r in self._results
            if r["duration_s"] >= min_duration_s
            or r["additional_mem_mb"] >= min_memory_mb
        ]
        if not visible:
            print("No sections exceeded the reporting thresholds.")
            return

        label_col = max(len(r["label"]) + r["depth"] * 2 for r in visible)
        label_col = max(label_col, 7)
        header = f"{'Section':<{label_col}}  {'Duration':>10}  {'+Memory':>10}"
        print(header)
        print("─" * len(header))
        for r in visible:
            indent = "  " * r["depth"]
            label = indent + r["label"]
            duration = f"{r['duration_s']:.3f} s"
            memory = f"{r['additional_mem_mb']:.1f} MB"
            print(f"{label:<{label_col}}  {duration:>10}  {memory:>10}")
