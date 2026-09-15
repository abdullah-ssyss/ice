from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from a2rl_drone_training.config import (
    CurriculumConfig,
    EvaluationConfig,
    NetworkConfig,
    ObservationConfig,
    PPOConfig,
    RacingEnvConfig,
    TrainingConfig,
)
from a2rl_drone_training.course import course_by_name
from a2rl_drone_training.runtime import RTX_5050_DEFAULTS, configure_runtime


def _resolve_cpu_threads(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "off", "none", "0"}:
        return None
    if normalized == "auto":
        return max(1, (os.cpu_count() or 2) // 2)
    try:
        threads = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--cpu-threads must be a positive integer, 'auto', or 'off'"
        ) from exc
    if threads < 1:
        raise argparse.ArgumentTypeError(
            "--cpu-threads must be a positive integer, 'auto', or 'off'"
        )
    return threads


def _configure_cpu_threads(value: str | None, device: str) -> int | None:
    if device.lower() != "cpu":
        return None
    threads = _resolve_cpu_threads(value)
    if threads is None:
        return None
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={threads}",
    )
    return threads


def _cpu_threads_arg(value: str) -> str:
    _resolve_cpu_threads(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an A2RL FPV racing policy.")
    parser.add_argument(
        "--profile",
        choices=["rtx-5050"],
        help="GPU starting preset; explicit flags override its defaults.",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=None,
        help="JAX GPU preallocation fraction, in (0, 1]; profile default: 0.60.",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--total-env-steps", type=int, default=20_000_000)
    parser.add_argument("--schedule-env-steps", type=int, default=20_000_000)
    parser.add_argument("--horizon", "--rollout-length", dest="horizon", type=int, default=256)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--actor-lr-end", type=float, default=3.0e-5)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr-end", type=float, default=3.0e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--entropy-coef-end", type=float, default=0.0003)
    parser.add_argument("--entropy-decay-fraction", type=float, default=0.75)
    parser.add_argument(
        "--exploration-std-start",
        type=float,
        default=0.60,
        help="Initial upper bound for each trainable action standard deviation.",
    )
    parser.add_argument(
        "--exploration-std-end",
        type=float,
        default=0.20,
        help="Final upper bound for each trainable action standard deviation.",
    )
    parser.add_argument(
        "--exploration-std-floor",
        type=float,
        default=0.08,
        help="Lower bound for each trainable action standard deviation.",
    )
    parser.add_argument(
        "--exploration-decay-fraction",
        type=float,
        default=0.75,
        help="Fraction of the PPO schedule over which the std ceiling cools.",
    )
    parser.add_argument(
        "--mean-action-alignment-coef",
        type=float,
        default=0.05,
        help="Weight for aligning deterministic means to positive-advantage samples.",
    )
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-clip-coef", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=_cpu_threads_arg,
        default="auto",
        help="CPU thread count set before JAX import; use auto or off.",
    )
    parser.add_argument(
        "--physics",
        default="first_principles",
        choices=["first_principles"],
    )
    parser.add_argument("--control-hz", type=int, default=500)
    parser.add_argument("--sim-hz", type=int, default=500)
    parser.add_argument("--max-episode-time", type=float, default=24.0)
    parser.add_argument(
        "--artificial-time-limit",
        type=float,
        default=None,
        help="Optional simulator time limit in seconds; treated as a truncation.",
    )

    parser.add_argument("--gate-context", type=int, default=3)
    parser.add_argument(
        "--sensor-noise-scale",
        "--obs-additive-noise-std",
        dest="sensor_noise_scale",
        type=float,
        default=1.0,
        help="Scale all feature-specific physical sensor noise; zero disables training noise.",
    )
    parser.add_argument("--gyro-noise-std", type=float, default=0.02)
    parser.add_argument("--specific-force-noise-std", type=float, default=0.15)
    parser.add_argument("--attitude-noise-std", type=float, default=0.005)
    parser.add_argument("--velocity-noise-std", type=float, default=0.03)
    parser.add_argument("--gate-position-noise-std", type=float, default=0.03)
    parser.add_argument("--gate-normal-noise-std", type=float, default=0.01)
    parser.add_argument("--gate-bearing-noise-std", type=float, default=0.005)
    parser.add_argument("--gate-distance-noise-std", type=float, default=0.03)
    parser.add_argument("--pnp-dropout-prob", type=float, default=0.02)

    parser.add_argument("--reward-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--gate-pass-bonus", type=float, default=6.0)
    parser.add_argument("--lap-finish-bonus", type=float, default=25.0)
    parser.add_argument("--miss-penalty", type=float, default=8.0)
    parser.add_argument("--crash-penalty", type=float, default=12.0)
    parser.add_argument(
        "--time-penalty",
        type=float,
        default=0.2,
        help="Full phase-D reward cost per physical second for reward v2.",
    )
    parser.add_argument(
        "--potential-reward-weight",
        type=float,
        default=1.25,
        help="Reward-v2 multiplier for potential-based course progress.",
    )
    parser.add_argument("--potential-track-scale", type=float, default=1.0)
    parser.add_argument(
        "--potential-backtrack-limit",
        type=float,
        default=0.50,
        help="Maximum pre-segment backtracking represented by the course coordinate.",
    )
    parser.add_argument("--potential-center-scale", type=float, default=0.25)
    parser.add_argument("--potential-plane-sigma", type=float, default=0.75)
    parser.add_argument("--potential-gamma", type=float, default=0.999)
    parser.add_argument(
        "--action-delta-penalty",
        type=float,
        default=0.0002,
        help="Reward-v2 coefficient on squared action changes.",
    )
    parser.add_argument("--gate-margin-penalty", type=float, default=6.0)
    parser.add_argument("--vehicle-radius", type=float, default=0.15)
    parser.add_argument("--position-uncertainty-k", type=float, default=2.0)
    # Reward-v1 ablation controls.
    parser.add_argument(
        "--late-gate-reward-base", type=float, default=1.0, help="Reward-v1 only."
    )
    parser.add_argument(
        "--late-gate-reward-cap", type=float, default=4.0, help="Reward-v1 only."
    )
    parser.add_argument(
        "--lookahead-progress-reward", type=float, default=0.35, help="Reward-v1 only."
    )
    parser.add_argument(
        "--centered-crossing-bonus", type=float, default=2.0, help="Reward-v1 only."
    )
    parser.add_argument(
        "--path-alignment-reward", type=float, default=0.02, help="Reward-v1 only."
    )
    parser.add_argument(
        "--stall-penalty", type=float, default=-0.02, help="Reward-v1 only."
    )

    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--phase-a-gate1-fraction", type=float, default=0.20)
    parser.add_argument("--phase-b-gate1-fraction", type=float, default=0.50)
    parser.add_argument("--phase-c-gate1-fraction", type=float, default=0.80)
    parser.add_argument("--phase-d-gate1-fraction", type=float, default=0.80)
    parser.add_argument("--phase-a-window-scale", type=float, default=1.80)
    parser.add_argument("--phase-b-window-scale", type=float, default=1.30)
    parser.add_argument("--phase-c-window-scale", type=float, default=1.00)
    parser.add_argument("--phase-b-time-scale", type=float, default=0.25)
    parser.add_argument("--phase-c-time-scale", type=float, default=0.50)
    parser.add_argument("--phase-a-min-pass-rate", type=float, default=0.80)
    parser.add_argument("--phase-b-min-pass-rate", type=float, default=0.85)
    parser.add_argument("--phase-c-min-pass-rate", type=float, default=0.90)
    parser.add_argument("--phase-b-completion-rate", type=float, default=0.50)
    parser.add_argument("--phase-c-completion-rate", type=float, default=0.80)
    parser.add_argument("--curriculum-min-gate-samples", type=int, default=32)
    parser.add_argument("--curriculum-hysteresis", type=int, default=2)
    parser.add_argument("--gate-window-rate-limit", type=float, default=0.02)
    parser.add_argument("--time-cost-rate-limit", type=float, default=0.02)
    parser.add_argument("--priority-alpha", type=float, default=2.0)
    parser.add_argument("--priority-epsilon", type=float, default=0.05)
    parser.add_argument("--priority-probability-floor", type=float, default=0.02)
    parser.add_argument("--skill-qualification-window", type=int, default=10)
    parser.add_argument("--phase-a-priority-mix", type=float, default=0.50)
    parser.add_argument(
        "--local-link-start-mix",
        type=float,
        default=0.50,
        help="Fraction of local resets assigned to predecessors of weak gates.",
    )

    parser.add_argument("--no-evaluation", action="store_true")
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=10,
        help="Evaluation interval in updates; zero runs only the final evaluation.",
    )
    parser.add_argument("--evaluation-envs", type=int, default=32)
    parser.add_argument("--evaluation-seed", type=int, default=123)
    parser.add_argument("--skill-evaluation-attempts-per-gate", type=int, default=8)
    parser.add_argument("--skill-evaluation-max-time", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--restore-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--course",
        default="arena_38m_stacked",
        choices=["arena_38m_stacked", "compact_slalom"],
    )
    parser.add_argument("--racing-line-spawn-min-distance", type=float, default=1.0)
    parser.add_argument("--racing-line-spawn-max-distance", type=float, default=3.0)
    parser.add_argument("--racing-line-spawn-speed", type=float, default=2.0)
    return parser


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    obs = replace(
        ObservationConfig(),
        gate_context=args.gate_context,
        sensor_noise_scale=args.sensor_noise_scale,
        gyro_noise_std_rad_s=args.gyro_noise_std,
        specific_force_noise_std_m_s2=args.specific_force_noise_std,
        attitude_noise_std_rad=args.attitude_noise_std,
        velocity_noise_std_m_s=args.velocity_noise_std,
        gate_position_noise_std_m=args.gate_position_noise_std,
        gate_normal_noise_std_rad=args.gate_normal_noise_std,
        gate_bearing_noise_std=args.gate_bearing_noise_std,
        gate_distance_noise_std_m=args.gate_distance_noise_std,
        pnp_dropout_prob=args.pnp_dropout_prob,
    )
    env = replace(
        RacingEnvConfig(),
        num_envs=args.num_envs,
        sim_hz=args.sim_hz,
        control_hz=args.control_hz,
        max_episode_time_s=args.max_episode_time,
        artificial_time_limit_s=args.artificial_time_limit,
        device=args.device,
        physics=args.physics,
        reward_version=args.reward_version,
        gate_pass_bonus=args.gate_pass_bonus,
        lap_finish_bonus=args.lap_finish_bonus,
        v2_miss_penalty=args.miss_penalty,
        v2_crash_penalty=args.crash_penalty,
        time_penalty=args.time_penalty,
        potential_reward_weight=args.potential_reward_weight,
        potential_track_scale=args.potential_track_scale,
        potential_backtrack_limit=args.potential_backtrack_limit,
        potential_center_scale=args.potential_center_scale,
        potential_plane_sigma_m=args.potential_plane_sigma,
        potential_gamma=args.potential_gamma,
        action_delta_penalty=args.action_delta_penalty,
        gate_margin_penalty=args.gate_margin_penalty,
        vehicle_radius_m=args.vehicle_radius,
        position_uncertainty_k=args.position_uncertainty_k,
        late_gate_reward_base=args.late_gate_reward_base,
        late_gate_reward_cap=args.late_gate_reward_cap,
        lookahead_progress_reward=args.lookahead_progress_reward,
        centered_crossing_bonus=args.centered_crossing_bonus,
        path_alignment_reward=args.path_alignment_reward,
        stall_penalty=args.stall_penalty,
        racing_line_spawn_min_distance_m=args.racing_line_spawn_min_distance,
        racing_line_spawn_max_distance_m=args.racing_line_spawn_max_distance,
        racing_line_spawn_speed_m_s=args.racing_line_spawn_speed,
    )
    ppo = replace(
        PPOConfig(),
        total_env_steps=args.total_env_steps,
        schedule_env_steps=args.schedule_env_steps,
        horizon=args.horizon,
        minibatches=args.minibatches,
        update_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        actor_lr=args.actor_lr,
        actor_lr_end=args.actor_lr_end,
        critic_lr=args.critic_lr,
        critic_lr_end=args.critic_lr_end,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        entropy_decay_fraction=args.entropy_decay_fraction,
        exploration_std_start=args.exploration_std_start,
        exploration_std_end=args.exploration_std_end,
        exploration_std_floor=args.exploration_std_floor,
        exploration_decay_fraction=args.exploration_decay_fraction,
        mean_action_alignment_coef=args.mean_action_alignment_coef,
        clip_coef=args.clip_coef,
        value_clip_coef=args.value_clip_coef,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    curriculum = replace(
        CurriculumConfig(),
        enabled=not args.no_curriculum,
        phase_a_gate1_fraction=args.phase_a_gate1_fraction,
        phase_b_gate1_fraction=args.phase_b_gate1_fraction,
        phase_c_gate1_fraction=args.phase_c_gate1_fraction,
        phase_d_gate1_fraction=args.phase_d_gate1_fraction,
        phase_a_window_scale=args.phase_a_window_scale,
        phase_b_window_scale=args.phase_b_window_scale,
        phase_c_window_scale=args.phase_c_window_scale,
        phase_b_time_cost_scale=args.phase_b_time_scale,
        phase_c_time_cost_scale=args.phase_c_time_scale,
        phase_a_min_pass_rate=args.phase_a_min_pass_rate,
        phase_b_min_pass_rate=args.phase_b_min_pass_rate,
        phase_c_min_pass_rate=args.phase_c_min_pass_rate,
        phase_b_completion_rate=args.phase_b_completion_rate,
        phase_c_completion_rate=args.phase_c_completion_rate,
        min_eval_samples_per_gate=args.curriculum_min_gate_samples,
        hysteresis_evaluations=args.curriculum_hysteresis,
        gate_window_rate_limit=args.gate_window_rate_limit,
        time_cost_rate_limit=args.time_cost_rate_limit,
        priority_alpha=args.priority_alpha,
        priority_epsilon=args.priority_epsilon,
        priority_probability_floor=args.priority_probability_floor,
        skill_qualification_window=args.skill_qualification_window,
        phase_a_priority_mix=args.phase_a_priority_mix,
        local_link_start_mix=args.local_link_start_mix,
    )
    evaluation = EvaluationConfig(
        enabled=not args.no_evaluation,
        interval_updates=args.evaluation_interval,
        num_envs=args.evaluation_envs,
        seed=args.evaluation_seed,
        skill_attempts_per_gate=args.skill_evaluation_attempts_per_gate,
        skill_max_time_s=args.skill_evaluation_max_time,
    )
    checkpoint_dir = None if args.no_checkpoint else args.checkpoint_dir
    metrics_file = args.metrics_file
    if metrics_file is None and checkpoint_dir is not None:
        metrics_file = checkpoint_dir / "metrics.jsonl"
    return TrainingConfig(
        env=env,
        obs=obs,
        net=NetworkConfig(),
        ppo=ppo,
        curriculum=curriculum,
        evaluation=evaluation,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        metrics_file=metrics_file,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile == "rtx-5050":
        parser.set_defaults(**RTX_5050_DEFAULTS)
        args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        configure_runtime(args.device, args.gpu_memory_fraction)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _configure_cpu_threads(args.cpu_threads, args.device)

    # Importing the trainer imports JAX, so it must happen after CPU runtime setup.
    from a2rl_drone_training.trainer import PPOTrainer

    trainer = PPOTrainer(
        _build_training_config(args),
        course=course_by_name(args.course),
        restore_checkpoint=args.restore_checkpoint,
    )
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
