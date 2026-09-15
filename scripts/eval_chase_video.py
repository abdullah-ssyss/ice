from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/a2rl_matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from a2rl_drone_training.config import ObservationConfig, RacingEnvConfig, TrainingConfig
from a2rl_drone_training.actions import validate_checkpoint_actions
from a2rl_drone_training.course import GateCourse, course_by_name
from a2rl_drone_training.env import CrazyflowRacingEnv
from a2rl_drone_training.networks import mode_action
from a2rl_drone_training.normalization import (
    actor_normalization_spec,
    init_normalization_state,
    normalization_state_from_dict,
    normalize_actor_observation,
)


def latest_checkpoint(directory: Path) -> Path:
    candidates = sorted(directory.glob("checkpoint_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_*.pkl files found in {directory}")
    return candidates[-1]


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    validate_checkpoint_actions(payload)
    if "actor_params" not in payload:
        raise ValueError(f"Checkpoint {path} does not contain actor_params")
    checkpoint_config = payload.get("config")
    if checkpoint_config is not None:
        checkpoint_obs = getattr(checkpoint_config, "obs", None)
        checkpoint_core_dim = getattr(checkpoint_obs, "core_dim", None)
        current_core_dim = ObservationConfig().core_dim
        if checkpoint_core_dim is not None and checkpoint_core_dim != current_core_dim:
            raise ValueError(
                "Checkpoint observation architecture is incompatible with this eval script: "
                f"path={path} "
                f"checkpoint_core_dim={checkpoint_core_dim} "
                f"current_core_dim={current_core_dim}. "
                "Use a checkpoint created by the time-independent architecture."
            )
    return payload


def build_eval_config(
    checkpoint_config: TrainingConfig | None,
    *,
    num_envs: int,
    max_episode_time_s: float | None,
    device: str,
) -> tuple[RacingEnvConfig, ObservationConfig]:
    if checkpoint_config is None:
        env_config = RacingEnvConfig()
        obs_config = ObservationConfig()
    else:
        env_config = checkpoint_config.env
        obs_config = checkpoint_config.obs

    env_config = replace(
        env_config,
        num_envs=num_envs,
        device=device,
        max_episode_time_s=max_episode_time_s or env_config.max_episode_time_s,
        auto_reset=False,
        gate_window_scale=1.0,
        reset_distribution="evaluation",
        artificial_time_limit_s=None,
    )
    obs_config = replace(obs_config, pnp_dropout_prob=0.0, sensor_noise_scale=0.0)
    return env_config, obs_config


def rollout_policy(
    *,
    actor_params: Any,
    env_config: RacingEnvConfig,
    obs_config: ObservationConfig,
    course: GateCourse,
    seed: int,
    normalization_payload: dict[str, Any] | None,
    visualization_env: int = 0,
) -> dict[str, np.ndarray | int | float]:
    env = CrazyflowRacingEnv(env_config, obs_config, course=course)
    raw_obs = env.reset(seed=seed)
    if bool(jnp.any(env.reset_gate != 0)):
        raise RuntimeError("Evaluation visualization must start every environment at gate 1")
    normalization_state = (
        normalization_state_from_dict(normalization_payload, obs_config)
        if normalization_payload is not None
        else init_normalization_state(obs_config)
    )
    normalization_spec = actor_normalization_spec(obs_config)

    def normalize(x: jax.Array) -> jax.Array:
        return normalize_actor_observation(
            normalization_state,
            x,
            normalization_spec,
            obs_config.normalization_clip,
            obs_config.normalization_epsilon,
        )

    obs = normalize(raw_obs)

    active = jnp.ones((env_config.num_envs,), dtype=jnp.bool_)
    policy = jax.jit(lambda x: mode_action(actor_params, x, obs_config))

    positions = [np.asarray(jax.device_get(env.sim.data.states.pos[:, 0, :]))]
    velocities = [np.asarray(jax.device_get(env.sim.data.states.vel[:, 0, :]))]
    gate_counters = [np.asarray(jax.device_get(env.gate_counter))]
    rewards = []
    dones = []
    passes = []
    misses = []
    finishes = []

    max_steps = env_config.max_episode_steps
    for _ in range(max_steps):
        action = policy(obs)
        action = jnp.where(active[:, None], action, jnp.zeros_like(action))
        raw_obs, reward, terminated, truncated, info = env.step(action)
        obs = normalize(raw_obs)
        done = terminated | truncated
        active = active & ~done

        positions.append(np.asarray(jax.device_get(env.sim.data.states.pos[:, 0, :])))
        velocities.append(np.asarray(jax.device_get(env.sim.data.states.vel[:, 0, :])))
        gate_counters.append(np.asarray(jax.device_get(info["gate_counter"])))
        rewards.append(np.asarray(jax.device_get(reward)))
        dones.append(np.asarray(jax.device_get(done)))
        passes.append(np.asarray(jax.device_get(info["passed_gate"])))
        misses.append(np.asarray(jax.device_get(info["missed_gate"])))
        finishes.append(np.asarray(jax.device_get(info["finished"])))

        if not bool(jnp.any(active)):
            break

    env.close()

    positions_np = np.stack(positions)
    velocities_np = np.stack(velocities)
    gate_np = np.stack(gate_counters)
    reward_np = np.stack(rewards) if rewards else np.zeros((0, env_config.num_envs), dtype=np.float32)
    done_np = np.stack(dones) if dones else np.zeros((0, env_config.num_envs), dtype=bool)
    pass_np = np.stack(passes) if passes else np.zeros((0, env_config.num_envs), dtype=bool)
    miss_np = np.stack(misses) if misses else np.zeros((0, env_config.num_envs), dtype=bool)
    finish_np = np.stack(finishes) if finishes else np.zeros((0, env_config.num_envs), dtype=bool)

    selected_env = int(np.clip(visualization_env, 0, env_config.num_envs - 1))
    first_done = np.flatnonzero(done_np[:, selected_env])
    end_step = int(first_done[0] + 1) if first_done.size else int(positions_np.shape[0] - 1)
    end_step = max(end_step, 1)

    return {
        "positions": positions_np[: end_step + 1, selected_env],
        "velocities": velocities_np[: end_step + 1, selected_env],
        "gate_counters": gate_np[: end_step + 1, selected_env],
        "rewards": reward_np[:end_step, selected_env],
        "passes": pass_np[:end_step, selected_env],
        "misses": miss_np[:end_step, selected_env],
        "finishes": finish_np[:end_step, selected_env],
        "selected_env": selected_env,
        "end_step": end_step,
        "return": float(np.sum(reward_np[:end_step, selected_env])),
        "gates": int(np.max(gate_np[: end_step + 1, selected_env])),
        "finished": bool(np.any(finish_np[:end_step, selected_env])),
        "missed": bool(np.any(miss_np[:end_step, selected_env])),
    }


def _rect_points(center: np.ndarray, right: np.ndarray, width: float, height: float) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    hw = 0.5 * width
    hh = 0.5 * height
    corners = np.array(
        [
            center - hw * right - hh * up,
            center + hw * right - hh * up,
            center + hw * right + hh * up,
            center - hw * right + hh * up,
            center - hw * right - hh * up,
        ]
    )
    return corners


def _heading_series(positions: np.ndarray, velocities: np.ndarray, course: GateCourse) -> np.ndarray:
    headings = np.zeros((positions.shape[0], 2), dtype=np.float32)
    last = np.array([1.0, 0.0], dtype=np.float32)
    for i, (pos, vel) in enumerate(zip(positions, velocities, strict=True)):
        xy = vel[:2]
        norm = np.linalg.norm(xy)
        if norm < 0.25:
            gate_index = min(i, course.centers.shape[0] - 1)
            xy = course.centers[gate_index, :2] - pos[:2]
            norm = np.linalg.norm(xy)
        if norm >= 1.0e-6:
            target = xy / norm
            last = 0.85 * last + 0.15 * target
            last = last / max(np.linalg.norm(last), 1.0e-6)
        headings[i] = last
    return headings


def render_video(
    *,
    course: GateCourse,
    rollout: dict[str, np.ndarray | int | float],
    checkpoint: Path,
    output: Path,
    fps: int,
    stride: int,
    view_radius: float,
) -> None:
    positions = rollout["positions"]
    velocities = rollout["velocities"]
    gate_counters = rollout["gate_counters"]
    assert isinstance(positions, np.ndarray)
    assert isinstance(velocities, np.ndarray)
    assert isinstance(gate_counters, np.ndarray)

    headings = _heading_series(positions, velocities, course)
    frame_ids = np.arange(0, positions.shape[0], max(stride, 1), dtype=np.int32)
    if frame_ids[-1] != positions.shape[0] - 1:
        frame_ids = np.append(frame_ids, positions.shape[0] - 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.5, 6.2), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f7f7f7")

    def draw_static() -> None:
        ax.plot(
            course.nominal_racing_line[:, 0],
            course.nominal_racing_line[:, 1],
            course.nominal_racing_line[:, 2],
            color="#9a9a9a",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        for gate_idx, (center, right, normal, width, height, logical_id, opening) in enumerate(
            zip(
                course.centers,
                course.right_axes,
                course.normals,
                course.widths,
                course.heights,
                course.logical_gate_ids,
                course.openings,
                strict=True,
            ),
            start=1,
        ):
            rect = _rect_points(center, right, float(width), float(height))
            ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color="#111111", linewidth=2.2)
            outer = _rect_points(
                center,
                right,
                float(course.outer_widths[gate_idx - 1]),
                float(course.outer_heights[gate_idx - 1]),
            )
            ax.plot(outer[:, 0], outer[:, 1], outer[:, 2], color="#777777", linewidth=0.8, alpha=0.5)
            ax.quiver(
                center[0],
                center[1],
                center[2],
                normal[0],
                normal[1],
                normal[2],
                length=0.9,
                normalize=True,
                color="#1f77b4",
                linewidth=1.1,
            )
            label = str(gate_idx) if not opening else f"{gate_idx} {opening}"
            ax.text(
                center[0],
                center[1],
                center[2] + 0.82,
                label,
                fontsize=7,
                color="#111111",
                ha="center",
            )
            if opening:
                ax.text(
                    center[0],
                    center[1],
                    center[2] - 0.95,
                    f"L{logical_id}",
                    fontsize=6,
                    color="#555555",
                    ha="center",
                )

    def update(frame_number: int) -> None:
        frame = int(frame_ids[frame_number])
        pos = positions[frame]
        vel = velocities[frame]
        heading = headings[frame]
        speed = float(np.linalg.norm(vel))
        gate_counter = int(gate_counters[frame])
        current_gate = min(gate_counter + 1, course.num_gates)

        ax.clear()
        draw_static()
        trail_start = max(0, frame - 220)
        trail = positions[trail_start : frame + 1]
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="#d62728", linewidth=2.0)
        ax.scatter(pos[0], pos[1], pos[2], s=70, color="#d62728", edgecolor="white", linewidth=0.8)
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            heading[0],
            heading[1],
            0.0,
            length=1.1,
            normalize=True,
            color="#d62728",
            linewidth=1.5,
        )

        center = pos + np.array([heading[0] * 3.0, heading[1] * 3.0, 0.6], dtype=np.float32)
        ax.set_xlim(center[0] - view_radius, center[0] + view_radius)
        ax.set_ylim(center[1] - view_radius, center[1] + view_radius)
        ax.set_zlim(0.0, 5.2)
        azim = float(np.degrees(np.arctan2(-heading[1], -heading[0])))
        ax.view_init(elev=17.0, azim=azim)
        ax.set_box_aspect((1.0, 1.0, 0.35))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(
            f"{course.name} eval chase | gate {current_gate}/{course.num_gates} | "
            f"speed {speed:.1f} m/s",
            fontsize=10,
        )
        ax.text2D(
            0.015,
            0.95,
            f"checkpoint: {checkpoint.name}\n"
            f"step: {frame}  gates passed: {gate_counter}  return: {rollout['return']:.1f}",
            transform=ax.transAxes,
            fontsize=8,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
        )
        ax.grid(True, alpha=0.25)

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3000,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    animation.save(output, writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a chase-camera eval video for a drone policy.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_arena_path_alignment"))
    parser.add_argument("--course", default="arena_38m_stacked")
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval_chase_arena_38m_stacked.mp4"))
    parser.add_argument("--num-eval-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-episode-time", type=float, default=24.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--view-radius", type=float, default=8.0)
    parser.add_argument("--visualization-env", type=int, default=0)
    args = parser.parse_args()

    checkpoint = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    payload = load_checkpoint(checkpoint)
    course = course_by_name(args.course)
    env_config, obs_config = build_eval_config(
        payload.get("config"),
        num_envs=args.num_eval_envs,
        max_episode_time_s=args.max_episode_time,
        device=args.device,
    )
    rollout = rollout_policy(
        actor_params=jax.device_put(payload["actor_params"]),
        env_config=env_config,
        obs_config=obs_config,
        course=course,
        seed=args.seed,
        normalization_payload=payload.get("normalization_state"),
        visualization_env=args.visualization_env,
    )
    render_video(
        course=course,
        rollout=rollout,
        checkpoint=checkpoint,
        output=args.output,
        fps=args.fps,
        stride=args.stride,
        view_radius=args.view_radius,
    )
    print(
        f"wrote {args.output} "
        f"checkpoint={checkpoint} "
        f"selected_env={rollout['selected_env']} "
        f"gates={rollout['gates']} "
        f"return={rollout['return']:.3f} "
        f"finished={rollout['finished']} "
        f"missed={rollout['missed']}"
    )


if __name__ == "__main__":
    main()
