from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from device_env import (
    get_train_settings,
    resolve_gpu_allocator,
    train_output_dir,
    train_spacy_args,
)

CONFIG = "config/config.cfg"
TRAIN_PATH = "data/train.spacy"
DEV_PATH = "data/dev.spacy"


class ResourceMonitor:
    def __init__(self, pid: int, mode: str, gpu_id: int, interval: float = 1.0) -> None:
        self.pid = pid
        self.mode = mode
        self.gpu_id = gpu_id
        self.interval = interval
        self.cpu_count = os.cpu_count() or 1
        self.cpu_capacity = min(
            float(self.cpu_count),
            self._read_cpu_quota_cores() or float(self.cpu_count),
        )
        self.memory_limit_mb = self._read_memory_limit_mb()
        self.clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.start_cpu_time = self._read_cpu_time() or 0.0
        self.last_cpu_time = self.start_cpu_time
        self.start_wall_time = time.perf_counter()
        self.last_wall_time = time.perf_counter()
        self.peak_cpu_cores = 0.0
        self.peak_rss_mb = 0.0
        self.gpu_samples: list[tuple[int, int, int]] = []
        self.gpu_error: str | None = None

    def start(self) -> None:
        self._sample_cpu()
        self._sample_gpu()
        self._thread.start()

    def stop(self) -> dict[str, str]:
        self._stop.set()
        self._thread.join(timeout=self.interval + 0.5)
        self._sample_cpu()
        self._sample_gpu()

        elapsed = max(0.001, time.perf_counter() - self.start_wall_time)
        total_cpu_seconds = max(0.0, self.last_cpu_time - self.start_cpu_time)
        return {
            "cpu_available": self._format_cpu_available(),
            "avg_cpu_cores": f"{self.avg_cpu_cores(total_cpu_seconds, elapsed):.2f}",
            "avg_cpu_percent": f"{self.avg_cpu_percent(total_cpu_seconds, elapsed):.1f}%",
            "peak_cpu_cores": f"{self.peak_cpu_cores:.2f}",
            "peak_cpu_percent": f"{self.peak_cpu_cores / self.cpu_capacity * 100:.1f}%",
            "peak_rss_mb": self._format_ram_peak(),
            **self.gpu_summary(),
        }

    def avg_cpu_cores(self, total_cpu_seconds: float, elapsed: float) -> float:
        return min(self.cpu_capacity, total_cpu_seconds / elapsed)

    def avg_cpu_percent(self, total_cpu_seconds: float, elapsed: float) -> float:
        return self.avg_cpu_cores(total_cpu_seconds, elapsed) / self.cpu_capacity * 100

    def gpu_summary(self) -> dict[str, str]:
        if self.mode != "gpu":
            return {}
        if not self.gpu_samples:
            return {"gpu": self.gpu_error or "n/a"}

        used_values = [sample[0] for sample in self.gpu_samples]
        total_mb = self.gpu_samples[-1][1]
        util_values = [sample[2] for sample in self.gpu_samples]
        baseline_mb = used_values[0]
        peak_mb = max(used_values)
        return {
            "vram_base": f"{baseline_mb} MB",
            "vram_peak": f"{peak_mb} / {total_mb} MB",
            "vram_delta": f"{max(0, peak_mb - baseline_mb)} MB",
            "gpu_util_avg": f"{sum(util_values) / len(util_values):.1f}%",
            "gpu_util_peak": f"{max(util_values)}%",
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample_cpu()
            self._sample_gpu()

    def _sample_cpu(self) -> None:
        now = time.perf_counter()
        cpu_time = self._read_cpu_time()
        if cpu_time is None:
            return

        wall_delta = max(0.001, now - self.last_wall_time)
        cpu_delta = max(0.0, cpu_time - self.last_cpu_time)
        self.peak_cpu_cores = min(
            self.cpu_capacity,
            max(self.peak_cpu_cores, cpu_delta / wall_delta),
        )
        self.last_cpu_time = cpu_time
        self.last_wall_time = now

        rss_mb = self._read_rss_mb()
        if rss_mb is not None:
            self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)

    def _sample_gpu(self) -> None:
        if self.mode != "gpu":
            return
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            first_line = result.stdout.strip().splitlines()[0]
            used_mb, total_mb, util = [int(part.strip()) for part in first_line.split(",")]
            self.gpu_samples.append((used_mb, total_mb, util))
        except Exception as exc:
            self.gpu_error = f"nvidia-smi unavailable: {exc}"

    def _read_cpu_time(self) -> float | None:
        try:
            stat = Path(f"/proc/{self.pid}/stat").read_text()
            after_name = stat[stat.rfind(")") + 2 :].split()
            user_ticks = int(after_name[11])
            system_ticks = int(after_name[12])
            return (user_ticks + system_ticks) / self.clock_ticks
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            return None

    def _read_rss_mb(self) -> float | None:
        try:
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            return None
        return None

    def _read_cpu_quota_cores(self) -> float | None:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        try:
            quota, period = cpu_max.read_text().split()[:2]
            if quota != "max":
                return int(quota) / int(period)
        except (FileNotFoundError, ValueError, IndexError, ZeroDivisionError):
            pass

        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        try:
            quota = int(quota_path.read_text())
            period = int(period_path.read_text())
            if quota > 0:
                return quota / period
        except (FileNotFoundError, ValueError, ZeroDivisionError):
            pass
        return None

    def _read_memory_limit_mb(self) -> float | None:
        for path in (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ):
            try:
                raw = path.read_text().strip()
                if raw and raw != "max":
                    limit_mb = int(raw) / 1024 / 1024
                    if limit_mb < 1024 * 1024:
                        return limit_mb
            except (FileNotFoundError, ValueError):
                pass

        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024
        except (FileNotFoundError, ValueError, IndexError):
            pass
        return None

    def _format_cpu_available(self) -> str:
        if abs(self.cpu_capacity - self.cpu_count) < 0.01:
            return f"{self.cpu_count} cores"
        return f"{self.cpu_capacity:.2f} cores ({self.cpu_count} visible)"

    def _format_ram_peak(self) -> str:
        if not self.memory_limit_mb:
            return f"{self.peak_rss_mb:.1f} MB"
        percent = self.peak_rss_mb / self.memory_limit_mb * 100
        return f"{self.peak_rss_mb:.1f} / {self.memory_limit_mb:.0f} MB ({percent:.1f}%)"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s"


def _line(width: int = 52) -> str:
    return "─" * width


def _print_block(title: str, rows: list[tuple[str, str]]) -> None:
    key_width = max([16, *(len(key) for key, _ in rows)])
    value_width = max([33, *(len(value) for _, value in rows)])
    width = key_width + value_width + 3
    print()
    print(f"╭{_line(width)}╮")
    print(f"│ {title:<{width - 2}} │")
    print(f"├{_line(width)}┤")
    for key, value in rows:
        print(f"│ {key:<{key_width}} {value:<{value_width}} │")
    print(f"╰{_line(width)}╯")


def _resource_rows(resources: dict[str, str], mode: str) -> list[tuple[str, str]]:
    rows = [
        ("CPU available", resources["cpu_available"]),
        ("CPU avg", f"{resources['avg_cpu_cores']} cores ({resources['avg_cpu_percent']})"),
        ("CPU peak", f"{resources['peak_cpu_cores']} cores ({resources['peak_cpu_percent']})"),
        ("RAM peak", resources["peak_rss_mb"]),
    ]

    if mode == "gpu":
        if "gpu" in resources:
            rows.append(("GPU", resources["gpu"]))
        else:
            rows.extend(
                [
                    ("VRAM base", resources["vram_base"]),
                    ("VRAM peak", resources["vram_peak"]),
                    ("VRAM delta", resources["vram_delta"]),
                    ("GPU util avg", resources["gpu_util_avg"]),
                    ("GPU util peak", resources["gpu_util_peak"]),
                ]
            )
    return rows


def main() -> int:
    if not Path(TRAIN_PATH).exists() or not Path(DEV_PATH).exists():
        print("Chưa có data .spacy. Chạy: make preprocess")
        return 1

    settings = get_train_settings()
    output_dir = train_output_dir(settings.mode)
    allocator, warn = resolve_gpu_allocator(settings)
    if warn:
        print(f"[WARN] {warn}")

    _print_block(
        "TRAIN",
        [
            ("NER_TRAIN_MODE", settings.mode.upper()),
            ("NER_GPU_ID", str(settings.gpu_id)),
            ("NER_GPU_ALLOCATOR", allocator),
            ("Output", f"{output_dir}/model-best"),
        ],
    )

    extra_args = list(sys.argv[1:])
    if not any(arg == "--gpu-id" for arg in extra_args):
        extra_args = train_spacy_args(settings, allocator) + extra_args

    cmd = [
        sys.executable,
        "-m",
        "spacy",
        "train",
        CONFIG,
        "--output",
        str(output_dir),
        "--paths.train",
        TRAIN_PATH,
        "--paths.dev",
        DEV_PATH,
        *extra_args,
    ]

    print(f"\n$ {' '.join(cmd)}\n")

    started = datetime.now().astimezone()
    t0 = time.perf_counter()
    process = subprocess.Popen(cmd)
    monitor = ResourceMonitor(process.pid, settings.mode, settings.gpu_id)
    monitor.start()
    try:
        returncode = process.wait()
    finally:
        elapsed = time.perf_counter() - t0
        finished = datetime.now().astimezone()
        resources = monitor.stop()

    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    _print_block(
        "TRAIN TIMING",
        [
            ("Started", started.strftime("%Y-%m-%d %H:%M:%S %Z")),
            ("Finished", finished.strftime("%Y-%m-%d %H:%M:%S %Z")),
            ("Duration", f"{_format_duration(elapsed)} ({elapsed:.1f}s)"),
            ("Status", status),
            ("Model", f"{output_dir}/model-best"),
        ],
    )
    _print_block("RESOURCE USAGE", _resource_rows(resources, settings.mode))
    print()

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
