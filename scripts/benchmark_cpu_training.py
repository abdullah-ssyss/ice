from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    args: tuple[str, ...]


CASES = (
    BenchmarkCase("baseline_defaults", ()),
    BenchmarkCase("cpu_threads_8", ("--cpu-threads", "8")),
    BenchmarkCase("epochs_2_ablation", ("--update-epochs", "2")),
    BenchmarkCase(
        "motor_control_200hz",
        ("--sim-hz", "200", "--control-hz", "200", "--physics", "first_principles"),
    ),
)

GPU_CASES = tuple(
    BenchmarkCase(
        f"gpu_envs_{count}",
        (
            "--profile", "rtx-5050",
            "--device", "gpu",
            "--num-envs", str(count),
            "--minibatches", str(count // 8),
        ),
    )
    for count in (64, 128, 256, 512)
)


def _flag_value(args: list[str], flag: str, default: str) -> str:
    value = default
    for index, item in enumerate(args):
        if item == flag and index + 1 < len(args):
            value = args[index + 1]
    return value


def _run_case(
    *,
    case: BenchmarkCase,
    repo_root: Path,
    python: str,
    updates: int,
    num_envs: int,
    horizon: int,
    course: str,
    log_interval: int,
    warmup_updates: int,
    verbose_training_output: bool,
) -> tuple[int, float]:
    base_args = [
        "--device",
        "cpu",
        "--num-envs",
        str(num_envs),
        "--horizon",
        str(horizon),
        "--course",
        course,
        "--log-interval",
        str(log_interval),
        "--no-evaluation",
        "--no-checkpoint",
    ]
    run_args = [*base_args, *case.args]
    case_num_envs = int(_flag_value(run_args, "--num-envs", str(num_envs)))
    case_horizon = int(_flag_value(run_args, "--horizon", str(horizon)))
    measured_env_steps = case_num_envs * case_horizon * updates
    run_updates = warmup_updates + updates + 1
    total_env_steps = case_num_envs * case_horizon * run_updates
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + env["PYTHONPATH"]
    )
    with tempfile.TemporaryDirectory(prefix="a2rl-benchmark-") as directory:
        metrics_file = Path(directory) / "metrics.jsonl"
        cmd = [
            python,
            "-m",
            "a2rl_drone_training.train",
            *run_args,
            "--metrics-file",
            str(metrics_file),
            "--total-env-steps",
            str(total_env_steps),
        ]
        subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            check=True,
            stdout=None if verbose_training_output else subprocess.DEVNULL,
        )
        records = [
            json.loads(line)
            for line in metrics_file.read_text(encoding="utf-8").splitlines()
        ]
    first_measured_update = warmup_updates + 1
    last_measured_update = warmup_updates + updates
    measured = [
        record
        for record in records
        if first_measured_update <= int(record["update"]) <= last_measured_update
    ]
    if len(measured) != updates:
        raise RuntimeError(
            f"Expected {updates} measured updates for {case.name}, found {len(measured)}"
        )
    elapsed = sum(float(record["timing"]["update_seconds"]) for record in measured)
    return measured_env_steps, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU or GPU A2RL training presets.")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--course", default="arena_38m_stacked")
    parser.add_argument("--log-interval", type=int, default=1_000_000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--verbose-training-output", action="store_true")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be at least 1")
    if args.warmup_updates < 1:
        parser.error("--warmup-updates must be at least 1")

    repo_root = Path(__file__).resolve().parents[1]
    print(
        "case,measured_env_steps,seconds,env_steps_per_second",
        flush=True,
    )
    for case in GPU_CASES if args.device == "gpu" else CASES:
        total_env_steps, elapsed = _run_case(
            case=case,
            repo_root=repo_root,
            python=args.python,
            updates=args.updates,
            num_envs=args.num_envs,
            horizon=args.horizon,
            course=args.course,
            log_interval=args.log_interval,
            warmup_updates=args.warmup_updates,
            verbose_training_output=args.verbose_training_output,
        )
        steps_per_second = total_env_steps / max(elapsed, 1.0e-9)
        print(
            f"{case.name},{total_env_steps},{elapsed:.3f},{steps_per_second:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
