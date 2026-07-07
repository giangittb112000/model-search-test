from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from device_env import get_train_settings, train_output_dir, train_spacy_args

CONFIG = "config/config.cfg"
TRAIN_PATH = "data/train.spacy"
DEV_PATH = "data/dev.spacy"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s"


def _line(width: int = 52) -> str:
    return "─" * width


def _print_block(title: str, rows: list[tuple[str, str]]) -> None:
    print()
    print(f"╭{_line()}╮")
    print(f"│ {title:<50} │")
    print(f"├{_line()}┤")
    for key, value in rows:
        print(f"│ {key:<16} {value:<33} │")
    print(f"╰{_line()}╯")


def main() -> int:
    if not Path(TRAIN_PATH).exists() or not Path(DEV_PATH).exists():
        print("Chưa có data .spacy. Chạy: make preprocess")
        return 1

    settings = get_train_settings()
    output_dir = train_output_dir(settings.mode)

    _print_block(
        "TRAIN",
        [
            ("NER_TRAIN_MODE", settings.mode.upper()),
            ("NER_GPU_ID", str(settings.gpu_id)),
            ("NER_GPU_ALLOCATOR", settings.gpu_allocator),
            ("Output", f"{output_dir}/model-best"),
        ],
    )

    extra_args = list(sys.argv[1:])
    if not any(arg == "--gpu-id" for arg in extra_args):
        extra_args = train_spacy_args(settings) + extra_args

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
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    finished = datetime.now().astimezone()

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
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
    print()

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
