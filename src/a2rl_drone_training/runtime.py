"""Runtime setup that must run before importing JAX."""

from __future__ import annotations

import os
import sys


RTX_5050_DEFAULTS = {
    "device": "gpu",
    "num_envs": 256,
    "horizon": 256,
    "minibatches": 32,
    "gpu_memory_fraction": 0.60,
}


def configure_runtime(device: str, gpu_memory_fraction: float | None = None) -> None:
    if gpu_memory_fraction is not None and not 0 < gpu_memory_fraction <= 1:
        raise ValueError("--gpu-memory-fraction must be greater than 0 and at most 1")
    if device == "gpu":
        if sys.platform == "win32":
            raise ValueError(
                "JAX CUDA training requires Linux or WSL2. Native Windows Python "
                "cannot use this GPU backend. See the README GPU setup section."
            )
        # Require CUDA instead of silently falling back to CPU. Explicit CLI device
        # selection also keeps model allocations on the simulator's backend.
        os.environ["JAX_PLATFORMS"] = "cuda"
        if gpu_memory_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_memory_fraction)
    elif device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
