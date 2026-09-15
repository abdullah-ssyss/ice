from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ObservationConfig:
    """Fixed-size state representation consumed by the policy."""

    gate_context: int = 3
    core_dim: int = 24
    gate_feature_dim: int = 10
    max_gate_distance: float = 25.0
    camera_fov_x_rad: float = 1.7453292519943295  # 100 deg
    camera_fov_y_rad: float = 1.3089969389957472  # 75 deg
    pnp_dropout_prob: float = 0.02
    sensor_noise_scale: float = 1.0
    gyro_noise_std_rad_s: float = 0.02
    specific_force_noise_std_m_s2: float = 0.15
    attitude_noise_std_rad: float = 0.005
    velocity_noise_std_m_s: float = 0.03
    gate_position_noise_std_m: float = 0.03
    gate_normal_noise_std_rad: float = 0.01
    gate_bearing_noise_std: float = 0.005
    gate_distance_noise_std_m: float = 0.03
    normalization_clip: float = 10.0
    normalization_epsilon: float = 1.0e-5

    @property
    def dim(self) -> int:
        return self.core_dim + self.gate_context * self.gate_feature_dim


@dataclass(frozen=True)
class RacingEnvConfig:
    """Crazyflow-backed racing task configuration."""

    num_envs: int = 64
    sim_hz: int = 500
    control_hz: int = 500
    max_episode_time_s: float = 24.0
    artificial_time_limit_s: float | None = None
    laps: int = 1
    device: str = "cpu"
    physics: Literal["first_principles", "so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"] = (
        "first_principles"
    )
    drone_model: str = "cf2x_L250"
    reward_version: Literal["v1", "v2"] = "v2"
    gate_pass_bonus: float = 6.0
    lap_finish_bonus: float = 25.0
    crash_penalty: float = -5.0
    miss_penalty: float = -2.0
    time_penalty: float = 0.2
    plane_progress_reward: float = 1.8
    distance_progress_reward: float = 0.8
    radial_progress_reward: float = 0.25
    centerline_reward: float = 0.06
    crossing_center_reward: float = 0.08
    path_alignment_reward: float = 0.02
    late_gate_reward_base: float = 1.0
    late_gate_reward_cap: float = 4.0
    lookahead_progress_reward: float = 0.35
    centered_crossing_bonus: float = 2.0
    stall_penalty: float = -0.02
    stall_progress_threshold_m: float = 0.005
    stall_patience_steps: int = 25
    action_penalty: float = 0.008
    potential_reward_weight: float = 1.25
    potential_track_scale: float = 1.0
    potential_backtrack_limit: float = 0.50
    potential_center_scale: float = 0.25
    potential_plane_sigma_m: float = 0.75
    potential_gamma: float = 0.999
    action_delta_penalty: float = 0.0002
    gate_margin_penalty: float = 6.0
    vehicle_radius_m: float = 0.15
    position_uncertainty_k: float = 2.0
    v2_miss_penalty: float = 8.0
    v2_crash_penalty: float = 12.0
    thrust_min_n: float | None = None
    thrust_hover_n: float | None = None
    thrust_max_n: float | None = None
    spawn_vel_noise_m_s: float = 0.15
    gate_miss_depth_m: float = 1.0
    auto_reset: bool = True
    reset_distribution: Literal["training", "evaluation"] = "training"
    gate_window_scale: float = 1.0
    racing_line_spawn_min_distance_m: float = 1.0
    racing_line_spawn_max_distance_m: float = 3.0
    racing_line_spawn_speed_m_s: float = 2.0
    spawn_distance_m: float = 1.3

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def max_episode_steps(self) -> int:
        return int(self.max_episode_time_s * self.control_hz)

    @property
    def artificial_time_limit_steps(self) -> int | None:
        if self.artificial_time_limit_s is None:
            return None
        return int(self.artificial_time_limit_s * self.control_hz)


@dataclass(frozen=True)
class NetworkConfig:
    action_dim: int = 4
    state_hidden: tuple[int, ...] = (128, 128)
    state_latent_dim: int = 128
    gate_hidden: tuple[int, ...] = (64, 64)
    gate_latent_dim: int = 64
    fusion_hidden: tuple[int, ...] = (256, 128)
    critic_hidden: tuple[int, ...] = (256, 256)
    initial_log_std: float = -0.5


@dataclass(frozen=True)
class PPOConfig:
    total_env_steps: int = 20_000_000
    schedule_env_steps: int = 20_000_000
    horizon: int = 256
    minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.999
    gae_lambda: float = 0.99
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 3.0e-4
    actor_lr_end: float = 3.0e-5
    critic_lr: float = 3.0e-4
    critic_lr_end: float = 3.0e-5
    entropy_coef_end: float = 0.0003
    entropy_decay_fraction: float = 0.75
    exploration_std_start: float = 0.60
    exploration_std_end: float = 0.20
    exploration_std_floor: float = 0.08
    exploration_decay_fraction: float = 0.75
    mean_action_alignment_coef: float = 0.05
    target_kl: float = 0.015
    seed: int = 7
    log_interval: int = 10


@dataclass(frozen=True)
class PrivilegedObservationConfig:
    """Noise-free simulator features available only to the value function."""

    dim: int = 37


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = True
    phase_a_gate1_fraction: float = 0.20
    phase_b_gate1_fraction: float = 0.50
    phase_c_gate1_fraction: float = 0.80
    phase_d_gate1_fraction: float = 0.80
    phase_a_window_scale: float = 1.80
    phase_b_window_scale: float = 1.30
    phase_c_window_scale: float = 1.00
    phase_d_window_scale: float = 1.00
    phase_b_time_cost_scale: float = 0.25
    phase_c_time_cost_scale: float = 0.50
    phase_a_min_pass_rate: float = 0.80
    phase_b_min_pass_rate: float = 0.85
    phase_c_min_pass_rate: float = 0.90
    phase_b_completion_rate: float = 0.50
    phase_c_completion_rate: float = 0.80
    min_eval_samples_per_gate: int = 32
    skill_qualification_window: int = 10
    phase_a_priority_mix: float = 0.50
    local_link_start_mix: float = 0.50
    hysteresis_evaluations: int = 2
    gate_window_rate_limit: float = 0.02
    time_cost_rate_limit: float = 0.02
    priority_epsilon: float = 0.05
    priority_alpha: float = 2.0
    priority_probability_floor: float = 0.02


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = True
    interval_updates: int = 10
    num_envs: int = 32
    seed: int = 123
    skill_attempts_per_gate: int = 8
    skill_max_time_s: float = 4.0


@dataclass(frozen=True)
class TrainingConfig:
    env: RacingEnvConfig = RacingEnvConfig()
    obs: ObservationConfig = ObservationConfig()
    critic_obs: PrivilegedObservationConfig = PrivilegedObservationConfig()
    net: NetworkConfig = NetworkConfig()
    ppo: PPOConfig = PPOConfig()
    curriculum: CurriculumConfig = CurriculumConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    checkpoint_dir: Path | None = Path("checkpoints")
    checkpoint_interval: int = 10
    metrics_file: Path | None = None
