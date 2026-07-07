from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DeviceMode = Literal["cpu", "gpu"]
GpuAllocator = Literal["null", "pytorch", "tensorflow"]

MODELS_ROOT = Path("models")


def _parse_mode(value: str | None, default: DeviceMode = "cpu") -> DeviceMode:
    mode = (value or default).strip().lower()
    if mode not in {"cpu", "gpu"}:
        return default
    return mode  # type: ignore[return-value]


def _parse_gpu_id() -> int:
    try:
        return int(os.environ.get("NER_GPU_ID", "0"))
    except ValueError:
        return 0


def _parse_gpu_allocator(train_mode: DeviceMode) -> GpuAllocator:
    """spaCy [system].gpu_allocator — chỉ có ý nghĩa khi train trên GPU."""
    if train_mode == "cpu":
        return "null"
    raw = (os.environ.get("NER_GPU_ALLOCATOR") or "auto").strip().lower()
    if raw in {"", "auto"}:
        return "pytorch"
    if raw in {"null", "none"}:
        return "null"
    if raw in {"pytorch", "tensorflow"}:
        return raw  # type: ignore[return-value]
    return "pytorch"


def train_output_dir(mode: DeviceMode) -> Path:
    return MODELS_ROOT / mode


def model_best_path() -> Path:
    mode = _parse_mode(os.environ.get("NER_TRAIN_MODE"), "cpu")
    return train_output_dir(mode) / "model-best"


@dataclass(frozen=True)
class TrainSettings:
    mode: DeviceMode
    gpu_id: int
    gpu_allocator: GpuAllocator


@dataclass(frozen=True)
class RunSettings:
    mode: DeviceMode
    model_mode: DeviceMode
    gpu_id: int


def get_train_settings() -> TrainSettings:
    mode = _parse_mode(os.environ.get("NER_TRAIN_MODE"), "cpu")
    return TrainSettings(
        mode=mode,
        gpu_id=_parse_gpu_id(),
        gpu_allocator=_parse_gpu_allocator(mode),
    )


def get_run_settings() -> RunSettings:
    train_mode = _parse_mode(os.environ.get("NER_TRAIN_MODE"), "cpu")
    return RunSettings(
        mode=_parse_mode(os.environ.get("NER_RUN_MODE"), "cpu"),
        model_mode=train_mode,
        gpu_id=_parse_gpu_id(),
    )


def train_spacy_args(settings: TrainSettings) -> list[str]:
    """
  spaCy train:
    --gpu-id        chọn CPU (-1) hoặc card GPU (0, 1, …)
    --system.gpu_allocator  cách CuPy chia bộ nhớ GPU (chỉ khi train GPU)
    """
    gpu_id = str(settings.gpu_id) if settings.mode == "gpu" else "-1"
    return [
        "--gpu-id",
        gpu_id,
        "--system.gpu_allocator",
        settings.gpu_allocator,
    ]


def apply_run_device(spacy_module, settings: RunSettings) -> str | None:
    if settings.mode != "gpu":
        return None
    try:
        spacy_module.require_gpu(settings.gpu_id)
        return None
    except Exception as exc:
        return str(exc)
