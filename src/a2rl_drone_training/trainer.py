from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from a2rl_drone_training.config import TrainingConfig
from a2rl_drone_training.actions import ACTION_NAMES, ACTION_SPACE, validate_checkpoint_actions
from a2rl_drone_training.console import format_table, print_table
from a2rl_drone_training.course import GateCourse
from a2rl_drone_training.curriculum import CurriculumController
from a2rl_drone_training.env import CrazyflowRacingEnv
from a2rl_drone_training.networks import (
    critic_apply,
    init_critic,
    init_gc_actor,
    mode_action,
    sample_action_with_mode,
)
from a2rl_drone_training.normalization import (
    NormalizationState,
    actor_normalization_spec,
    init_normalization_state,
    normalization_state_dict,
    normalization_state_from_dict,
    normalize_actor_observation,
    update_normalization_state,
)
from a2rl_drone_training.optim import adam_init
from a2rl_drone_training.ppo import (
    Rollout,
    compute_gae,
    constrain_actor_exploration,
    ppo_update,
)
from a2rl_drone_training.rewards import REWARD_COMPONENT_NAMES


REWARD_COMPONENTS = REWARD_COMPONENT_NAMES


@dataclass
class TrainerState:
    actor_params: Any
    critic_params: Any
    actor_opt_state: Any
    critic_opt_state: Any
    obs: Array
    critic_obs: Array
    episode_smoothness_return: Array
    key: Array
    env_steps: int = 0
    updates: int = 0
    episodes_completed: int = 0
    schedule_steps: int = 0


@partial(jax.jit, static_argnames=("obs_config",))
def _sample_action_log_prob_and_value(
    actor_params: Any,
    critic_params: Any,
    actor_obs: Array,
    critic_obs: Array,
    key: Array,
    obs_config,
) -> tuple[Array, Array, Array, Array, Array]:
    key, action_key = jax.random.split(key)
    action, log_prob, mean_action = sample_action_with_mode(
        actor_params,
        actor_obs,
        action_key,
        obs_config,
    )
    value = critic_apply(critic_params, critic_obs)
    return key, action, log_prob, value, mean_action


@jax.jit
def _critic_value(critic_params: Any, critic_obs: Array) -> Array:
    return critic_apply(critic_params, critic_obs)


@jax.jit
def _accumulate_completed_episode_metric(
    running: Array,
    values: Array,
    done: Array,
) -> tuple[Array, Array]:
    def step(carry: Array, inputs: tuple[Array, Array]):
        value, step_done = inputs
        total = carry + value
        completed_value = jnp.where(step_done, total, jnp.nan)
        next_carry = jnp.where(step_done, 0.0, total)
        return next_carry, completed_value

    return jax.lax.scan(step, running, (values, done))


@partial(jax.jit, static_argnames=("obs_config",))
def _deterministic_action(actor_params: Any, obs: Array, obs_config) -> Array:
    return mode_action(actor_params, obs, obs_config)


class PPOTrainer:
    def __init__(
        self,
        config: TrainingConfig = TrainingConfig(),
        course: GateCourse | None = None,
        restore_checkpoint: Path | None = None,
    ):
        self.config = config
        self.course = course
        self.env = CrazyflowRacingEnv(config.env, config.obs, course=course)
        self.curriculum = CurriculumController(config.curriculum, self.env.course.num_gates)
        self.env.set_curriculum(self.curriculum.parameters())
        self.normalization_spec = actor_normalization_spec(config.obs)
        self.normalization_state: NormalizationState = init_normalization_state(config.obs)
        self.last_evaluation_metrics: dict[str, Any] | None = None
        self.last_skill_evaluation_metrics: dict[str, Any] | None = None
        self._eval_env: CrazyflowRacingEnv | None = None
        self._skill_eval_env: CrazyflowRacingEnv | None = None
        self._checkpoint_migrated = False
        self._pre_correction_v2_checkpoint = False
        self.run_id = f"{time.time_ns()}-{os.getpid()}"

        key = jax.random.key(config.ppo.seed)
        key, actor_key, critic_key = jax.random.split(key, 3)
        actor_params = init_gc_actor(actor_key, config.obs, config.net)
        critic_params = init_critic(critic_key, config.critic_obs.dim, config.net)
        raw_obs = self.env.reset(seed=config.ppo.seed)
        critic_obs = self.env.privileged_observe()
        self._validate_runtime_observations(raw_obs, critic_obs)
        self.normalization_state = update_normalization_state(
            self.normalization_state,
            raw_obs,
            self.normalization_spec,
        )
        obs = self._normalize(raw_obs)
        self.state = TrainerState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=adam_init(actor_params),
            critic_opt_state=adam_init(critic_params),
            obs=obs,
            critic_obs=critic_obs,
            episode_smoothness_return=jnp.zeros(
                (config.env.num_envs,), dtype=jnp.float32
            ),
            key=key,
        )
        if restore_checkpoint is not None:
            self.load_checkpoint(restore_checkpoint)
        else:
            self._apply_exploration_constraint()

    def train(self) -> None:
        cfg = self.config
        if self._pre_correction_v2_checkpoint and cfg.env.reward_version == "v2":
            raise RuntimeError(
                "Pre-correction Reward-V2 checkpoints cannot resume corrected training. "
                "Start a fresh Reward-V2 run; this checkpoint remains available for "
                "evaluation or Reward-V1 ablations."
            )
        steps_per_update = cfg.env.num_envs * cfg.ppo.horizon
        target_env_steps = max(cfg.ppo.total_env_steps, steps_per_update)
        if self.state.env_steps >= target_env_steps:
            print_table(
                [
                    (
                        "training",
                        [
                            ("status", "target already reached"),
                            ("total_timesteps", f"{self.state.env_steps:,}"),
                            ("target_timesteps", f"{target_env_steps:,}"),
                        ],
                    )
                ]
            )
            return
        total_updates = int(np.ceil(target_env_steps / steps_per_update))
        start_update = self.state.updates + 1
        checkpoint_dir = cfg.checkpoint_dir
        if cfg.ppo.log_interval < 1:
            raise ValueError("PPOConfig.log_interval must be at least 1")
        if cfg.ppo.schedule_env_steps < 1:
            raise ValueError("PPOConfig.schedule_env_steps must be at least 1")
        if checkpoint_dir is not None and cfg.checkpoint_interval < 1:
            raise ValueError("TrainingConfig.checkpoint_interval must be at least 1")
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        metrics_path = self._metrics_path()
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            self._startup_log_table(total_updates, steps_per_update, target_env_steps),
            flush=True,
        )
        run_start_time = time.perf_counter()
        run_start_env_steps = self.state.env_steps
        for update in range(start_update, total_updates + 1):
            update_start_time = time.perf_counter()
            self._apply_exploration_constraint()
            evaluation_due = cfg.evaluation.enabled and (
                update == total_updates
                or (
                    cfg.evaluation.interval_updates > 0
                    and update % cfg.evaluation.interval_updates == 0
                )
            )
            should_log = (
                update == start_update
                or update == total_updates
                or update % cfg.ppo.log_interval == 0
            )
            should_checkpoint = checkpoint_dir is not None and (
                update == total_updates or update % cfg.checkpoint_interval == 0
            )
            rollout, rollout_metrics = self.collect_rollout(
                collect_diagnostics=should_log or evaluation_due,
            )
            self.state.episodes_completed += int(rollout_metrics["episodes_completed"])
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.next_values,
                rollout.terminated,
                rollout.truncated,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
            )
            next_schedule_steps = self.state.schedule_steps + steps_per_update
            schedule_progress = min(
                next_schedule_steps / max(cfg.ppo.schedule_env_steps, 1),
                1.0,
            )
            (
                self.state.actor_params,
                self.state.critic_params,
                self.state.actor_opt_state,
                self.state.critic_opt_state,
                self.state.key,
                update_metrics,
            ) = ppo_update(
                actor_params=self.state.actor_params,
                critic_params=self.state.critic_params,
                actor_opt_state=self.state.actor_opt_state,
                critic_opt_state=self.state.critic_opt_state,
                rollout=rollout,
                advantages=advantages,
                returns=returns,
                key=self.state.key,
                obs_config=cfg.obs,
                config=cfg.ppo,
                schedule_progress=schedule_progress,
            )
            self.state.updates = update
            self.state.env_steps += steps_per_update
            self.state.schedule_steps = next_schedule_steps

            self.curriculum.step()
            evaluation: dict[str, Any] | None = None
            skill_evaluation: dict[str, Any] | None = None
            if evaluation_due:
                phase_before_evaluation = self.curriculum.phase
                evaluation = self.evaluate_policy()
                advanced = self.curriculum.record_evaluation(
                    attempts=evaluation["gate_attempts"],
                    passes=evaluation["gate_passes"],
                    episodes=int(evaluation["episodes"]),
                    finishes=int(evaluation["finishes"]),
                )
                if self.config.curriculum.enabled and phase_before_evaluation == "A":
                    skill_evaluation = self.evaluate_gate_skills()
                    advanced = self.curriculum.record_skill_evaluation(
                        attempts=skill_evaluation["gate_attempts"],
                        passes=skill_evaluation["gate_passes"],
                        strict_passes=skill_evaluation["strict_gate_passes"],
                    )
                    skill_evaluation["phase_advanced"] = advanced
                    skill_evaluation["update"] = update
                    self.last_skill_evaluation_metrics = skill_evaluation
                evaluation["phase_advanced"] = advanced
                evaluation["update"] = update
                self.last_evaluation_metrics = evaluation
            self.env.set_curriculum(self.curriculum.parameters())

            # PPO is dispatched asynchronously by JAX. Synchronize one update
            # output so the reported wall time includes the completed optimizer work.
            jax.block_until_ready(update_metrics["approx_kl"])
            update_seconds = time.perf_counter() - update_start_time
            run_seconds = time.perf_counter() - run_start_time
            run_env_steps = self.state.env_steps - run_start_env_steps
            timing_metrics = {
                "update_seconds": update_seconds,
                "env_steps_per_second": steps_per_update / max(update_seconds, 1.0e-9),
                "run_env_steps_per_second": run_env_steps / max(run_seconds, 1.0e-9),
                "updates_per_second": (update - start_update + 1) / max(run_seconds, 1.0e-9),
                "elapsed_seconds": run_seconds,
            }
            if should_log:
                self._log(update, total_updates, rollout_metrics, update_metrics, timing_metrics)
            self._write_metrics_record(
                update=update,
                total_updates=total_updates,
                detailed=should_log or evaluation_due,
                rollout_metrics=rollout_metrics,
                update_metrics=update_metrics,
                timing_metrics=timing_metrics,
                evaluation=evaluation,
                skill_evaluation=skill_evaluation,
            )
            if should_checkpoint:
                checkpoint_dir = Path(checkpoint_dir)
                self.save_checkpoint(checkpoint_dir / f"checkpoint_{update:06d}.pkl")
                self.save_checkpoint(checkpoint_dir / "checkpoint_latest.pkl")

    def collect_rollout(self, *, collect_diagnostics: bool = True) -> tuple[Rollout, dict[str, Any]]:
        obs_buf: list[Array] = []
        critic_obs_buf: list[Array] = []
        action_buf: list[Array] = []
        log_prob_buf: list[Array] = []
        value_buf: list[Array] = []
        reward_buf: list[Array] = []
        next_value_buf: list[Array] = []
        terminated_buf: list[Array] = []
        truncated_buf: list[Array] = []
        smoothness_buf: list[Array] = []
        done_buf: list[Array] = []
        course_finished_buf: list[Array] = []
        local_complete_buf: list[Array] = []
        final_returns: list[Array] = []
        final_lengths: list[Array] = []
        course_successful_returns: list[Array] = []
        local_successful_returns: list[Array] = []

        num_gates = self.env.course.num_gates
        gate_target_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_pass_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        strict_gate_pass_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_failure_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_clearance_sums = jnp.zeros((num_gates,), dtype=jnp.float32)
        curriculum_gate_clearance_sums = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_segment_time_sums = jnp.zeros((num_gates,), dtype=jnp.float32)
        gate_measurement_counts = jnp.zeros((num_gates,), dtype=jnp.float32)
        reward_sums = {name: jnp.asarray(0.0, dtype=jnp.float32) for name in REWARD_COMPONENTS}
        event_sums = {
            name: jnp.asarray(0.0, dtype=jnp.float32)
            for name in (
                "passed_gate",
                "missed_gate",
                "crashed",
                "out_of_bounds",
                "segment_complete",
                "course_finished",
                "local_segment_complete",
                "deadline",
                "stalled",
            )
        }
        episode_count = jnp.asarray(0.0, dtype=jnp.float32)
        reset_count = jnp.asarray(0.0, dtype=jnp.float32)
        local_reset_count = jnp.asarray(0.0, dtype=jnp.float32)
        action_saturation_count = jnp.asarray(0.0, dtype=jnp.float32)
        mean_action_saturation_count = jnp.asarray(0.0, dtype=jnp.float32)
        mean_action_gap_sums = jnp.zeros((self.config.net.action_dim,), dtype=jnp.float32)
        action_delta_squared_sums = jnp.zeros(
            (self.config.net.action_dim,), dtype=jnp.float32
        )
        speed_sum = jnp.asarray(0.0, dtype=jnp.float32)
        gate_counter_sum = jnp.asarray(0.0, dtype=jnp.float32)
        track_progress_sum = jnp.asarray(0.0, dtype=jnp.float32)
        forward_track_progress_sum = jnp.asarray(0.0, dtype=jnp.float32)
        backward_track_progress_sum = jnp.asarray(0.0, dtype=jnp.float32)

        obs = self.state.obs
        critic_obs = self.state.critic_obs
        for _ in range(self.config.ppo.horizon):
            (
                self.state.key,
                action,
                log_prob,
                value,
                mean_action,
            ) = _sample_action_log_prob_and_value(
                self.state.actor_params,
                self.state.critic_params,
                obs,
                critic_obs,
                self.state.key,
                self.config.obs,
            )
            next_raw_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            reset_critic_obs = self.env.privileged_observe()
            bootstrap_critic_obs = jnp.where(
                done[:, None],
                info["final_critic_observation"],
                reset_critic_obs,
            )
            next_value = _critic_value(self.state.critic_params, bootstrap_critic_obs)

            obs_buf.append(obs)
            critic_obs_buf.append(critic_obs)
            action_buf.append(action)
            log_prob_buf.append(log_prob)
            value_buf.append(value)
            reward_buf.append(reward)
            next_value_buf.append(next_value)
            terminated_buf.append(terminated)
            truncated_buf.append(truncated)
            smoothness_buf.append(info["reward_smoothness"])
            done_buf.append(done)
            course_finished_buf.append(info["course_finished"])
            local_complete_buf.append(info["local_segment_complete"])

            self.normalization_state = update_normalization_state(
                self.normalization_state,
                next_raw_obs,
                self.normalization_spec,
            )
            obs = self._normalize(next_raw_obs)
            critic_obs = reset_critic_obs

            one_hot = jax.nn.one_hot(info["gate_id"], num_gates, dtype=jnp.float32)
            passed = info["passed_gate"].astype(jnp.float32)
            task_failure = (terminated & ~info["finished"]).astype(jnp.float32)
            failure_gate_id = (
                info["gate_id"] + info["passed_gate"].astype(jnp.int32)
            ) % num_gates
            failure_one_hot = jax.nn.one_hot(
                failure_gate_id, num_gates, dtype=jnp.float32
            )
            gate_pass_counts += jnp.sum(one_hot * passed[:, None], axis=0)
            strict_gate_pass_counts += jnp.sum(
                one_hot * info["strict_passed_gate"].astype(jnp.float32)[:, None],
                axis=0,
            )
            gate_failure_counts += jnp.sum(
                failure_one_hot * task_failure[:, None], axis=0
            )
            episode_count += jnp.sum(done.astype(jnp.float32))
            reset_count += jnp.sum(info["reset_occurred"].astype(jnp.float32))
            local_reset_count += jnp.sum(
                info["reset_started_local"].astype(jnp.float32)
            )
            if collect_diagnostics:
                gate_target_counts += jnp.sum(one_hot, axis=0)
                valid_measurement = jnp.isfinite(info["gate_clearance"])
                measurement = valid_measurement.astype(jnp.float32)
                gate_clearance_sums += jnp.sum(
                    one_hot * jnp.nan_to_num(info["gate_clearance"])[:, None], axis=0
                )
                curriculum_gate_clearance_sums += jnp.sum(
                    one_hot
                    * jnp.nan_to_num(info["curriculum_gate_clearance"])[:, None],
                    axis=0,
                )
                gate_segment_time_sums += jnp.sum(
                    one_hot * jnp.nan_to_num(info["segment_time"])[:, None], axis=0
                )
                gate_measurement_counts += jnp.sum(one_hot * measurement[:, None], axis=0)
                for name in REWARD_COMPONENTS:
                    reward_sums[name] += jnp.sum(info[name])
                for name in event_sums:
                    event_sums[name] += jnp.sum(info[name].astype(jnp.float32))
                action_saturation_count += jnp.sum(
                    (jnp.abs(action) >= 0.98).astype(jnp.float32)
                )
                mean_action_saturation_count += jnp.sum(
                    (jnp.abs(mean_action) >= 0.98).astype(jnp.float32)
                )
                mean_action_gap_sums += jnp.sum(jnp.abs(action - mean_action), axis=0)
                action_delta_squared_sums += jnp.sum(
                    info["action_delta_squared"],
                    axis=0,
                )
                speed_sum += jnp.sum(info["speed"])
                gate_counter_sum += jnp.sum(info["gate_counter"].astype(jnp.float32))
                track_progress_sum += jnp.sum(info["track_progress_delta"])
                forward_track_progress_sum += jnp.sum(
                    jnp.maximum(info["track_progress_delta"], 0.0)
                )
                backward_track_progress_sum += jnp.sum(
                    jnp.minimum(info["track_progress_delta"], 0.0)
                )
                final_returns.append(info["final_return"])
                final_lengths.append(info["final_length"])
                course_successful_returns.append(
                    jnp.where(info["course_finished"], info["final_return"], jnp.nan)
                )
                local_successful_returns.append(
                    jnp.where(
                        info["local_segment_complete"],
                        info["final_return"],
                        jnp.nan,
                    )
                )

        (
            self.state.episode_smoothness_return,
            completed_smoothness_returns,
        ) = _accumulate_completed_episode_metric(
            self.state.episode_smoothness_return,
            jnp.stack(smoothness_buf),
            jnp.stack(done_buf),
        )
        course_smoothness_returns = jnp.where(
            jnp.stack(course_finished_buf),
            completed_smoothness_returns,
            jnp.nan,
        )
        local_smoothness_returns = jnp.where(
            jnp.stack(local_complete_buf),
            completed_smoothness_returns,
            jnp.nan,
        )

        self.state.obs = obs
        self.state.critic_obs = critic_obs
        rollout = Rollout(
            obs=jnp.stack(obs_buf),
            critic_obs=jnp.stack(critic_obs_buf),
            actions=jnp.stack(action_buf),
            log_probs=jnp.stack(log_prob_buf),
            values=jnp.stack(value_buf),
            rewards=jnp.stack(reward_buf),
            next_values=jnp.stack(next_value_buf),
            terminated=jnp.stack(terminated_buf),
            truncated=jnp.stack(truncated_buf),
        )
        gate_pass_np = _to_numpy(gate_pass_counts)
        strict_gate_pass_np = _to_numpy(strict_gate_pass_counts)
        gate_failure_np = _to_numpy(gate_failure_counts)
        self.curriculum.record_training(gate_pass_np, gate_failure_np)

        if not collect_diagnostics:
            return rollout, {
                "mean_reward": float(jnp.mean(rollout.rewards)),
                "episodes_completed": int(float(episode_count)),
                "gate_pass_counts": gate_pass_np,
                "strict_gate_pass_counts": strict_gate_pass_np,
                "gate_failure_counts": gate_failure_np,
                "reset_local_rate": _safe_ratio(local_reset_count, reset_count),
                "reset_event_count": float(reset_count),
            }

        sample_count = float(self.config.ppo.horizon * self.config.env.num_envs)
        action_count = sample_count * self.config.net.action_dim
        final_return_array = jnp.concatenate(final_returns)
        final_length_array = jnp.concatenate(final_lengths)
        gate_measurements_np = _to_numpy(gate_measurement_counts)
        metrics: dict[str, Any] = {
            "mean_reward": float(jnp.mean(rollout.rewards)),
            "episodic_return": _nanmean_or_zero(final_return_array),
            "episodic_length": _nanmean_or_zero(final_length_array),
            "episodes_completed": int(_finite_count(final_return_array)),
            "action_saturation_frequency": float(action_saturation_count / action_count),
            "mean_action_saturation_frequency": float(
                mean_action_saturation_count / action_count
            ),
            "mean_action_gap": _to_numpy(mean_action_gap_sums / sample_count),
            "action_delta_rms": _to_numpy(
                jnp.sqrt(action_delta_squared_sums / sample_count)
            ),
            "avg_speed": float(speed_sum / sample_count),
            "avg_gate_counter": float(gate_counter_sum / sample_count),
            "track_progress_delta": float(track_progress_sum / sample_count),
            "forward_track_progress": float(
                forward_track_progress_sum / sample_count
            ),
            "backward_track_progress": float(
                backward_track_progress_sum / sample_count
            ),
            "reset_local_rate": _safe_ratio(local_reset_count, reset_count),
            "reset_event_count": float(reset_count),
            "gate_window_scale": self.env.gate_window_scale,
            "time_cost_scale": self.env.time_cost_scale,
            "curriculum_phase": self.curriculum.phase,
            "gate_target_counts": _to_numpy(gate_target_counts),
            "gate_pass_counts": gate_pass_np,
            "strict_gate_pass_counts": strict_gate_pass_np,
            "gate_failure_counts": gate_failure_np,
            "gate_pass_rates": np.divide(
                gate_pass_np,
                gate_pass_np + gate_failure_np,
                out=np.zeros_like(gate_pass_np),
                where=(gate_pass_np + gate_failure_np) > 0,
            ),
            "strict_gate_pass_rates": np.divide(
                strict_gate_pass_np,
                gate_pass_np + gate_failure_np,
                out=np.zeros_like(strict_gate_pass_np),
                where=(gate_pass_np + gate_failure_np) > 0,
            ),
            "gate_clearance": np.divide(
                _to_numpy(gate_clearance_sums),
                gate_measurements_np,
                out=np.zeros_like(gate_measurements_np),
                where=gate_measurements_np > 0,
            ),
            "curriculum_gate_clearance": np.divide(
                _to_numpy(curriculum_gate_clearance_sums),
                gate_measurements_np,
                out=np.zeros_like(gate_measurements_np),
                where=gate_measurements_np > 0,
            ),
            "gate_segment_time": np.divide(
                _to_numpy(gate_segment_time_sums),
                gate_measurements_np,
                out=np.zeros_like(gate_measurements_np),
                where=gate_measurements_np > 0,
            ),
        }
        for name, value in reward_sums.items():
            metrics[name] = float(value / sample_count)
        metrics["successful_course_reward_total"] = _nanmean_or_zero(
            jnp.concatenate(course_successful_returns)
        )
        metrics["successful_course_reward_smoothness"] = _nanmean_or_zero(
            course_smoothness_returns
        )
        metrics["local_segment_reward_total"] = _nanmean_or_zero(
            jnp.concatenate(local_successful_returns)
        )
        metrics["local_segment_reward_smoothness"] = _nanmean_or_zero(
            local_smoothness_returns
        )
        # Compatibility aliases now refer only to true full-course success.
        metrics["successful_ep_reward_total"] = metrics[
            "successful_course_reward_total"
        ]
        metrics["successful_ep_reward_smoothness"] = metrics[
            "successful_course_reward_smoothness"
        ]
        for name, value in event_sums.items():
            metrics[f"{name}_rate"] = float(value / sample_count)
            metrics[f"{name}_count"] = int(float(value))
        return rollout, metrics

    def evaluate_policy(self) -> dict[str, Any]:
        """Run deterministic, strict, fixed-seed gate-1 evaluation."""

        env = self._get_evaluation_env()
        raw_obs = env.reset(seed=self.config.evaluation.seed)
        if bool(jnp.any(env.reset_gate != 0)):
            raise RuntimeError("Evaluation reset distribution must start every episode at gate 1")
        obs = self._normalize(raw_obs)
        active = jnp.ones((env.config.num_envs,), dtype=jnp.bool_)
        attempts = np.zeros((env.course.num_gates,), dtype=np.float64)
        passes = np.zeros((env.course.num_gates,), dtype=np.float64)
        clearance_sum = np.zeros((env.course.num_gates,), dtype=np.float64)
        segment_time_sum = np.zeros((env.course.num_gates,), dtype=np.float64)
        measurements = np.zeros((env.course.num_gates,), dtype=np.float64)
        attempts[0] = env.config.num_envs
        finishes = 0
        for _ in range(env.config.max_episode_steps):
            action = _deterministic_action(
                self.state.actor_params,
                obs,
                self.config.obs,
            )
            action = jnp.where(active[:, None], action, jnp.zeros_like(action))
            raw_obs, _, terminated, truncated, info = env.step(action)
            was_active = np.asarray(jax.device_get(active), dtype=bool)
            gate_ids = np.asarray(jax.device_get(info["gate_id"]), dtype=np.int32)
            passed = np.asarray(jax.device_get(info["passed_gate"]), dtype=bool) & was_active
            clearances = np.asarray(jax.device_get(info["gate_clearance"]), dtype=np.float32)
            segment_times = np.asarray(jax.device_get(info["segment_time"]), dtype=np.float32)
            for gate_index in range(env.course.num_gates):
                gate_pass = passed & (gate_ids == gate_index)
                count = int(np.sum(gate_pass))
                passes[gate_index] += count
                measurements[gate_index] += count
                if count:
                    clearance_sum[gate_index] += float(np.nansum(clearances[gate_pass]))
                    segment_time_sum[gate_index] += float(np.nansum(segment_times[gate_pass]))
                if gate_index + 1 < env.course.num_gates:
                    attempts[gate_index + 1] += count
            finishes += int(
                np.sum(
                    np.asarray(jax.device_get(info["course_finished"]), dtype=bool)
                    & was_active
                )
            )
            active = active & ~(terminated | truncated)
            obs = self._normalize(raw_obs)
            if not bool(jnp.any(active)):
                break

        pass_rates = np.divide(
            passes,
            attempts,
            out=np.zeros_like(passes),
            where=attempts > 0,
        )
        return {
            "episodes": env.config.num_envs,
            "finishes": finishes,
            "completion_rate": finishes / max(env.config.num_envs, 1),
            "gate_attempts": attempts,
            "gate_passes": passes,
            "gate_pass_rates": pass_rates,
            "gate_clearance": np.divide(
                clearance_sum,
                measurements,
                out=np.zeros_like(clearance_sum),
                where=measurements > 0,
            ),
            "gate_segment_time": np.divide(
                segment_time_sum,
                measurements,
                out=np.zeros_like(segment_time_sum),
                where=measurements > 0,
            ),
            "seed": self.config.evaluation.seed,
        }

    def evaluate_gate_skills(self) -> dict[str, Any]:
        """Run the deterministic Phase-A audit at the active training aperture."""

        attempts_per_gate = self.config.evaluation.skill_attempts_per_gate
        if attempts_per_gate < 1:
            raise ValueError("skill_attempts_per_gate must be at least 1")
        if self.config.evaluation.skill_max_time_s <= 0.0:
            raise ValueError("skill_max_time_s must be positive")
        env = self._get_skill_evaluation_env()
        env.set_skill_audit_window_scale(self.env.gate_window_scale)
        forced_gates = np.repeat(
            np.arange(env.course.num_gates, dtype=np.int32),
            attempts_per_gate,
        )
        audit_seed = self.config.evaluation.seed + 10_000
        raw_obs = env.reset(
            seed=audit_seed,
            forced_reset_gates=jnp.asarray(forced_gates),
        )
        if not np.array_equal(
            np.asarray(jax.device_get(env.reset_gate)),
            forced_gates,
        ):
            raise RuntimeError("Phase-A skill audit did not honor forced gate resets")
        obs = self._normalize(raw_obs)
        active = jnp.ones((env.config.num_envs,), dtype=jnp.bool_)
        passes = np.zeros((env.course.num_gates,), dtype=np.float64)
        strict_passes = np.zeros_like(passes)
        clearance_sum = np.zeros_like(passes)
        curriculum_clearance_sum = np.zeros_like(passes)
        segment_time_sum = np.zeros_like(passes)
        max_steps = min(
            env.config.max_episode_steps,
            max(
                1,
                int(np.ceil(self.config.evaluation.skill_max_time_s / env.config.dt)),
            ),
        )
        for _ in range(max_steps):
            action = _deterministic_action(
                self.state.actor_params,
                obs,
                self.config.obs,
            )
            action = jnp.where(active[:, None], action, jnp.zeros_like(action))
            raw_obs, _, terminated, truncated, info = env.step(action)
            was_active = np.asarray(jax.device_get(active), dtype=bool)
            gate_ids = np.asarray(jax.device_get(info["gate_id"]), dtype=np.int32)
            passed = (
                np.asarray(jax.device_get(info["passed_gate"]), dtype=bool)
                & was_active
                & (gate_ids == forced_gates)
            )
            strict_passed = (
                np.asarray(jax.device_get(info["strict_passed_gate"]), dtype=bool)
                & was_active
                & (gate_ids == forced_gates)
            )
            clearances = np.asarray(jax.device_get(info["gate_clearance"]), dtype=np.float32)
            curriculum_clearances = np.asarray(
                jax.device_get(info["curriculum_gate_clearance"]),
                dtype=np.float32,
            )
            segment_times = np.asarray(jax.device_get(info["segment_time"]), dtype=np.float32)
            for gate_index in range(env.course.num_gates):
                gate_pass = passed & (forced_gates == gate_index)
                count = int(np.sum(gate_pass))
                passes[gate_index] += count
                strict_passes[gate_index] += int(
                    np.sum(strict_passed & (forced_gates == gate_index))
                )
                if count:
                    clearance_sum[gate_index] += float(np.nansum(clearances[gate_pass]))
                    curriculum_clearance_sum[gate_index] += float(
                        np.nansum(curriculum_clearances[gate_pass])
                    )
                    segment_time_sum[gate_index] += float(
                        np.nansum(segment_times[gate_pass])
                    )
            active = active & ~(terminated | truncated | jnp.asarray(passed))
            obs = self._normalize(raw_obs)
            if not bool(jnp.any(active)):
                break

        attempts = np.full(
            (env.course.num_gates,),
            attempts_per_gate,
            dtype=np.float64,
        )
        return {
            "episodes": env.config.num_envs,
            "gate_attempts": attempts,
            "gate_passes": passes,
            "gate_pass_rates": passes / attempts,
            "strict_gate_passes": strict_passes,
            "strict_gate_pass_rates": strict_passes / attempts,
            "gate_clearance": np.divide(
                clearance_sum,
                passes,
                out=np.zeros_like(clearance_sum),
                where=passes > 0,
            ),
            "curriculum_gate_clearance": np.divide(
                curriculum_clearance_sum,
                passes,
                out=np.zeros_like(curriculum_clearance_sum),
                where=passes > 0,
            ),
            "gate_segment_time": np.divide(
                segment_time_sum,
                passes,
                out=np.zeros_like(segment_time_sum),
                where=passes > 0,
            ),
            "seed": audit_seed,
            "max_steps": max_steps,
            "gate_window_scale": env.gate_window_scale,
        }

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": 6,
            "action_space": ACTION_SPACE,
            "course_fingerprint": _course_fingerprint(self.env.course),
            "actor_params": jax.device_get(self.state.actor_params),
            "critic_params": jax.device_get(self.state.critic_params),
            "actor_opt_state": jax.device_get(self.state.actor_opt_state),
            "critic_opt_state": jax.device_get(self.state.critic_opt_state),
            "key": jax.device_get(self.state.key),
            "env_steps": self.state.env_steps,
            "updates": self.state.updates,
            "episodes_completed": self.state.episodes_completed,
            "schedule_steps": self.state.schedule_steps,
            "schedule_progress": self._schedule_progress(),
            "normalization_state": normalization_state_dict(self.normalization_state),
            "curriculum_state": self.curriculum.state_dict(),
            "last_evaluation_metrics": self.last_evaluation_metrics,
            "last_skill_evaluation_metrics": self.last_skill_evaluation_metrics,
            "run_id": self.run_id,
            "config": self.config,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as file:
                pickle.dump(payload, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        with path.open("rb") as file:
            payload = pickle.load(file)
        validate_checkpoint_actions(payload)
        required = ("actor_params", "critic_params", "actor_opt_state", "critic_opt_state")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Checkpoint {path} is missing required fields: {missing}")
        _validate_checkpoint_observation_compatibility(path, payload, self.config)
        checkpoint_fingerprint = payload.get("course_fingerprint")
        current_fingerprint = _course_fingerprint(self.env.course)
        if checkpoint_fingerprint is not None and checkpoint_fingerprint != current_fingerprint:
            raise ValueError(
                "Checkpoint course geometry is incompatible with this run: "
                f"path={path} checkpoint={checkpoint_fingerprint} "
                f"current={current_fingerprint}."
            )
        checkpoint_config = payload.get("config")
        checkpoint_env = getattr(checkpoint_config, "env", None)
        checkpoint_reward_version = getattr(checkpoint_env, "reward_version", None)
        if checkpoint_fingerprint is None and checkpoint_reward_version == "v2":
            self._pre_correction_v2_checkpoint = True
            warnings.warn(
                "Loading a pre-correction Reward-V2 checkpoint without a signed-course "
                "fingerprint. Keep it for evaluation or Reward-V1 ablations; do not resume "
                "corrected Reward-V2 training from it.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.state.actor_params = jax.device_put(payload["actor_params"])
        self.state.critic_params = jax.device_put(payload["critic_params"])
        self.state.actor_opt_state = jax.device_put(payload["actor_opt_state"])
        self.state.critic_opt_state = jax.device_put(payload["critic_opt_state"])
        self.state.env_steps = int(payload.get("env_steps", 0))
        self.state.updates = int(payload.get("updates", 0))
        self.state.episodes_completed = int(payload.get("episodes_completed", 0))
        if "schedule_steps" in payload:
            self.state.schedule_steps = max(int(payload["schedule_steps"]), 0)
        else:
            checkpoint_config = payload.get("config")
            checkpoint_ppo = getattr(checkpoint_config, "ppo", None)
            checkpoint_ppo_fields = (
                vars(checkpoint_ppo) if checkpoint_ppo is not None else {}
            )
            old_schedule_budget = max(
                int(
                    checkpoint_ppo_fields.get(
                        "schedule_env_steps",
                        checkpoint_ppo_fields.get(
                            "total_env_steps",
                            self.state.env_steps,
                        ),
                    )
                ),
                1,
            )
            old_progress = float(
                payload.get(
                    "schedule_progress",
                    min(self.state.env_steps / old_schedule_budget, 1.0),
                )
            )
            self.state.schedule_steps = max(int(round(old_progress * old_schedule_budget)), 0)
            self._checkpoint_migrated = True
        if "key" in payload:
            self.state.key = jax.device_put(payload["key"])
        if "normalization_state" in payload:
            self.normalization_state = normalization_state_from_dict(
                payload["normalization_state"], self.config.obs
            )
        if "curriculum_state" in payload:
            self.curriculum.load_state_dict(payload["curriculum_state"])
        self.last_evaluation_metrics = payload.get("last_evaluation_metrics")
        self.last_skill_evaluation_metrics = payload.get(
            "last_skill_evaluation_metrics"
        )
        self.run_id = str(payload.get("run_id", self.run_id))
        self._apply_exploration_constraint()
        self.env.set_curriculum(self.curriculum.parameters())
        raw_obs = self.env.reset(seed=self.config.ppo.seed + self.state.updates)
        self.state.obs = self._normalize(raw_obs)
        self.state.critic_obs = self.env.privileged_observe()
        self._validate_runtime_observations(raw_obs, self.state.critic_obs)
        print_table(
            [
                (
                    "checkpoint",
                    [
                        ("status", "restored"),
                        ("path", path),
                        ("total_timesteps", f"{self.state.env_steps:,}"),
                        ("updates", f"{self.state.updates:,}"),
                        ("episodes", f"{self.state.episodes_completed:,}"),
                        ("schedule_steps", f"{self.state.schedule_steps:,}"),
                        ("schedule_progress", f"{self._schedule_progress():.1%}"),
                        ("curriculum_phase", self.curriculum.phase),
                        ("migrated_schedule", self._checkpoint_migrated),
                        ("pre_correction_v2", self._pre_correction_v2_checkpoint),
                    ],
                )
            ]
        )

    def _schedule_progress(self) -> float:
        return min(
            self.state.schedule_steps / max(self.config.ppo.schedule_env_steps, 1),
            1.0,
        )

    def _apply_exploration_constraint(self) -> None:
        (
            self.state.actor_params,
            self.state.actor_opt_state,
            _,
        ) = constrain_actor_exploration(
            self.state.actor_params,
            self.state.actor_opt_state,
            self.config.ppo,
            self._schedule_progress(),
        )

    def _write_metrics_record(
        self,
        *,
        update: int,
        total_updates: int,
        detailed: bool,
        rollout_metrics: dict[str, Any],
        update_metrics: dict[str, Any],
        timing_metrics: dict[str, float],
        evaluation: dict[str, Any] | None,
        skill_evaluation: dict[str, Any] | None,
    ) -> None:
        path = self._metrics_path()
        if path is None:
            return
        qualification = self.curriculum.qualification_statistics()
        local_probability_components = (
            self.curriculum.local_gate_probability_components()
        )
        record = {
            "type": "update",
            "run_id": self.run_id,
            "update": update,
            "total_updates": total_updates,
            "env_steps": self.state.env_steps,
            "detailed": detailed,
            "timing": timing_metrics,
            "ppo": {
                **update_metrics,
                "schedule_steps": self.state.schedule_steps,
                "schedule_progress": self._schedule_progress(),
            },
            "rollout": rollout_metrics,
            "curriculum": {
                "phase": self.curriculum.phase,
                "gate_window_scale": self.env.gate_window_scale,
                "time_cost_scale": self.env.time_cost_scale,
                **qualification,
                "training_pass_rates": self.curriculum.training_pass_rates(),
                "local_gate_probabilities": local_probability_components["combined"],
                "direct_local_gate_probabilities": local_probability_components["direct"],
                "link_local_gate_probabilities": local_probability_components["link"],
            },
            "evaluation": evaluation,
            "skill_evaluation": skill_evaluation,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            json.dump(_json_safe(record), file, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()

    def _metrics_path(self) -> Path | None:
        if self.config.metrics_file is not None:
            return Path(self.config.metrics_file)
        if self.config.checkpoint_dir is None:
            return None
        return Path(self.config.checkpoint_dir) / "metrics.jsonl"

    def close(self) -> None:
        self.env.close()
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None
        if self._skill_eval_env is not None:
            self._skill_eval_env.close()
            self._skill_eval_env = None

    def _normalize(self, observations: Array) -> Array:
        return normalize_actor_observation(
            self.normalization_state,
            observations,
            self.normalization_spec,
            self.config.obs.normalization_clip,
            self.config.obs.normalization_epsilon,
        )

    def _get_evaluation_env(self) -> CrazyflowRacingEnv:
        if self._eval_env is None:
            env_config = replace(
                self.config.env,
                num_envs=self.config.evaluation.num_envs,
                reset_distribution="evaluation",
                auto_reset=False,
                gate_window_scale=1.0,
                artificial_time_limit_s=None,
            )
            obs_config = replace(
                self.config.obs,
                sensor_noise_scale=0.0,
                pnp_dropout_prob=0.0,
            )
            self._eval_env = CrazyflowRacingEnv(
                env_config,
                obs_config,
                course=self.env.course,
            )
        return self._eval_env

    def _get_skill_evaluation_env(self) -> CrazyflowRacingEnv:
        if self._skill_eval_env is None:
            num_envs = (
                self.env.course.num_gates
                * self.config.evaluation.skill_attempts_per_gate
            )
            env_config = replace(
                self.config.env,
                num_envs=num_envs,
                reset_distribution="evaluation",
                auto_reset=False,
                gate_window_scale=1.0,
                artificial_time_limit_s=None,
            )
            obs_config = replace(
                self.config.obs,
                sensor_noise_scale=0.0,
                pnp_dropout_prob=0.0,
            )
            self._skill_eval_env = CrazyflowRacingEnv(
                env_config,
                obs_config,
                course=self.env.course,
            )
        return self._skill_eval_env

    def _validate_runtime_observations(self, actor_obs: Array, critic_obs: Array) -> None:
        if actor_obs.shape[-1] != self.config.obs.dim:
            raise ValueError(
                f"Actor observation has {actor_obs.shape[-1]} features; expected {self.config.obs.dim}"
            )
        if critic_obs.shape[-1] != self.config.critic_obs.dim:
            raise ValueError(
                "Privileged critic observation has "
                f"{critic_obs.shape[-1]} features; expected {self.config.critic_obs.dim}"
            )

    def _log(
        self,
        update: int,
        total_updates: int,
        rollout_metrics: dict[str, Any],
        update_metrics: dict[str, Array],
        timing_metrics: dict[str, float],
    ) -> None:
        scalars = {name: float(value) for name, value in update_metrics.items()}
        target_steps = max(
            self.config.ppo.total_env_steps,
            self.config.env.num_envs * self.config.ppo.horizon,
        )
        steps_left = max(target_steps - self.state.env_steps, 0)
        rate = timing_metrics["run_env_steps_per_second"]
        eta = _format_duration(steps_left / rate) if rate > 0.0 else "n/a"
        std = np.exp(_to_numpy(self.state.actor_params["log_std"]))
        completion = (
            float(self.last_evaluation_metrics["completion_rate"])
            if self.last_evaluation_metrics is not None
            else 0.0
        )
        qualification = self.curriculum.qualification_statistics()
        curriculum_parameters = self.curriculum.parameters()
        sections: list[tuple[str, list[tuple[str, Any]]]] = [
            (
                "time",
                [
                    ("iterations", f"{update:,} / {total_updates:,}"),
                    ("progress", f"{100.0 * self.state.env_steps / max(target_steps, 1):.1f}%"),
                    ("fps", f"{timing_metrics['env_steps_per_second']:,.0f}"),
                    ("fps_avg", f"{rate:,.0f}"),
                    ("iteration_time", f"{timing_metrics['update_seconds']:.2f}s"),
                    ("iterations_per_sec", f"{timing_metrics['updates_per_second']:.3f}"),
                    ("time_elapsed", _format_duration(timing_metrics["elapsed_seconds"])),
                    ("eta", eta),
                    ("total_timesteps", f"{self.state.env_steps:,}"),
                    ("schedule_progress", f"{self._schedule_progress():.1%}"),
                ],
            ),
            (
                "rollout",
                [
                    ("ep_rew_mean", f"{rollout_metrics['episodic_return']:.3f}"),
                    ("ep_len_mean", f"{rollout_metrics['episodic_length']:.1f}"),
                    ("step_rew_mean", f"{rollout_metrics['mean_reward']:.5f}"),
                    ("episodes", f"{rollout_metrics['episodes_completed']:,}"),
                    (
                        "course_finishes",
                        f"{rollout_metrics['course_finished_count']:,}",
                    ),
                    (
                        "local_completions",
                        f"{rollout_metrics['local_segment_complete_count']:,}",
                    ),
                    ("speed_mean", f"{rollout_metrics['avg_speed']:.3f} m/s"),
                    ("action_saturation", f"{rollout_metrics['action_saturation_frequency']:.3%}"),
                    (
                        "mean_action_saturation",
                        f"{rollout_metrics['mean_action_saturation_frequency']:.3%}",
                    ),
                    (
                        "sample_mean_gap",
                        _format_named_values(
                            ACTION_NAMES,
                            rollout_metrics["mean_action_gap"],
                            precision=3,
                        ),
                    ),
                    (
                        "action_delta_rms",
                        _format_named_values(
                            ACTION_NAMES,
                            rollout_metrics["action_delta_rms"],
                            precision=3,
                        ),
                    ),
                    (
                        "track_progress",
                        f"{rollout_metrics['track_progress_delta']:+.5f} / step",
                    ),
                ],
            ),
            (
                "train",
                [
                    ("approx_kl", f"{scalars.get('approx_kl', 0.0):.6f}"),
                    ("clip_fraction", f"{scalars.get('clip_fraction', 0.0):.3f}"),
                    ("entropy", f"{scalars.get('entropy', 0.0):.3f}"),
                    ("actor_loss", f"{scalars.get('actor_loss', 0.0):.4f}"),
                    ("critic_loss", f"{scalars.get('critic_loss', 0.0):.4f}"),
                    ("explained_variance", f"{scalars.get('explained_variance', 0.0):.3f}"),
                    ("actor_grad_norm", f"{scalars.get('actor_grad_norm', 0.0):.3f}"),
                    ("critic_grad_norm", f"{scalars.get('critic_grad_norm', 0.0):.3f}"),
                    ("epochs", f"{scalars.get('epochs_completed', 0.0):.0f}"),
                    ("actor_lr", f"{scalars.get('actor_lr', 0.0):.2e}"),
                    ("critic_lr", f"{scalars.get('critic_lr', 0.0):.2e}"),
                    ("entropy_coef", f"{scalars.get('entropy_coef', 0.0):.5f}"),
                    (
                        "std_ceiling",
                        f"{scalars.get('exploration_std_ceiling', 0.0):.3f}",
                    ),
                    (
                        "mean_alignment",
                        f"{scalars.get('mean_action_alignment_loss', 0.0):.4f}",
                    ),
                    ("action_std", _format_named_values(ACTION_NAMES, std, precision=3)),
                ],
            ),
            (
                "curriculum",
                [
                    ("phase", self.curriculum.phase),
                    ("gate_window_scale", f"{self.env.gate_window_scale:.2f}x"),
                    ("time_cost_scale", f"{self.env.time_cost_scale:.2f}"),
                    ("local_reset_rate", f"{rollout_metrics['reset_local_rate']:.1%}"),
                    (
                        "local_reset_target",
                        f"{1.0 - curriculum_parameters.gate1_fraction:.1%}",
                    ),
                    ("reset_events", f"{rollout_metrics['reset_event_count']:,.0f}"),
                    ("qualification_source", qualification["source"]),
                    ("qualification_samples", f"{qualification['minimum_samples']:.0f} min"),
                    ("recent_active_pass", f"{qualification['minimum_pass_rate']:.1%} min"),
                    (
                        "recent_strict_pass",
                        f"{float(np.min(qualification['strict_pass_rates'])):.1%} min",
                    ),
                    (
                        "lifetime_active_pass",
                        f"{float(np.min(qualification['lifetime_pass_rates'])):.1%} min",
                    ),
                    (
                        "qualification_streak",
                        f"{qualification['qualification_streak']} / "
                        f"{self.config.curriculum.hysteresis_evaluations}",
                    ),
                    (
                        "gate1_completion",
                        f"{completion:.1%}" if self.last_evaluation_metrics is not None else "n/a",
                    ),
                ],
            ),
            (
                "rewards",
                [
                    (name[7:], f"{rollout_metrics[name]:+.5f}")
                    for name in REWARD_COMPONENTS
                ],
            ),
            (
                "successful_course",
                [
                    (
                        "reward_total",
                        f"{rollout_metrics['successful_course_reward_total']:+.3f}",
                    ),
                    (
                        "smoothness_total",
                        f"{rollout_metrics['successful_course_reward_smoothness']:+.3f}",
                    ),
                ],
            ),
            (
                "local_segment",
                [
                    (
                        "reward_total",
                        f"{rollout_metrics['local_segment_reward_total']:+.3f}",
                    ),
                    (
                        "smoothness_total",
                        f"{rollout_metrics['local_segment_reward_smoothness']:+.3f}",
                    ),
                ],
            ),
            (
                "gates",
                [
                    ("order", _format_gate_order(rollout_metrics["gate_pass_rates"])),
                    ("active_pass", _format_gate_series(rollout_metrics["gate_pass_rates"])),
                    ("strict_pass", _format_gate_series(rollout_metrics["strict_gate_pass_rates"])),
                    ("physical_clearance_m", _format_gate_series(rollout_metrics["gate_clearance"])),
                    (
                        "curriculum_clearance_m",
                        _format_gate_series(rollout_metrics["curriculum_gate_clearance"]),
                    ),
                    ("segment_time_s", _format_gate_series(rollout_metrics["gate_segment_time"])),
                    ("targets", _format_gate_count_series(rollout_metrics["gate_target_counts"])),
                    ("passes", _format_gate_count_series(rollout_metrics["gate_pass_counts"])),
                    ("failures", _format_gate_count_series(rollout_metrics["gate_failure_counts"])),
                    ("qualification_n", _format_gate_count_series(qualification["attempts"])),
                    ("recent_active_rate", _format_gate_series(qualification["pass_rates"])),
                    ("recent_strict_rate", _format_gate_series(qualification["strict_pass_rates"])),
                    (
                        "latest_course_rate",
                        _format_gate_series(
                            qualification["latest_evaluation_pass_rates"]
                        ),
                    ),
                    (
                        "lifetime_active_rate",
                        _format_gate_series(qualification["lifetime_pass_rates"]),
                    ),
                    (
                        "lifetime_strict_rate",
                        _format_gate_series(qualification["lifetime_strict_pass_rates"]),
                    ),
                    (
                        "weakest_recent_active",
                        _weakest_rate_summary(qualification["pass_rates"]),
                    ),
                    (
                        "weakest_recent_strict",
                        _weakest_rate_summary(qualification["strict_pass_rates"]),
                    ),
                    (
                        "weakest_lifetime_active",
                        _weakest_rate_summary(qualification["lifetime_pass_rates"]),
                    ),
                    (
                        "weakest_lifetime_strict",
                        _weakest_rate_summary(
                            qualification["lifetime_strict_pass_rates"]
                        ),
                    ),
                    (
                        "local_start_probability",
                        _format_gate_series(
                            self.curriculum.local_gate_probabilities()
                        ),
                    ),
                    ("weakest", _weak_gate_summary(rollout_metrics)),
                ],
            ),
        ]
        if self.last_evaluation_metrics is not None:
            evaluation = self.last_evaluation_metrics
            sections.append(
                (
                    "eval",
                    [
                        ("last_update", f"{int(evaluation.get('update', 0)):,}"),
                        ("seed", evaluation["seed"]),
                        ("completion_rate", f"{evaluation['completion_rate']:.1%}"),
                        ("pass_rate", _format_gate_series(evaluation["gate_pass_rates"])),
                        ("clearance_m", _format_gate_series(evaluation["gate_clearance"])),
                        ("segment_time_s", _format_gate_series(evaluation["gate_segment_time"])),
                    ],
                )
            )
        if self.last_skill_evaluation_metrics is not None:
            skill = self.last_skill_evaluation_metrics
            sections.append(
                (
                    "skill_eval",
                    [
                        ("last_update", f"{int(skill.get('update', 0)):,}"),
                        ("seed", skill["seed"]),
                        ("attempts_per_gate", self.config.evaluation.skill_attempts_per_gate),
                        ("gate_window_scale", f"{skill['gate_window_scale']:.2f}x"),
                        ("active_pass", _format_gate_series(skill["gate_pass_rates"])),
                        (
                            "strict_pass",
                            _format_gate_series(skill["strict_gate_pass_rates"]),
                        ),
                        (
                            "physical_clearance_m",
                            _format_gate_series(skill["gate_clearance"]),
                        ),
                        (
                            "curriculum_clearance_m",
                            _format_gate_series(skill["curriculum_gate_clearance"]),
                        ),
                        ("segment_time_s", _format_gate_series(skill["gate_segment_time"])),
                    ],
                )
            )
        print(format_table(sections), flush=True)

    def _startup_log_table(
        self,
        total_updates: int,
        steps_per_update: int,
        target_env_steps: int,
    ) -> str:
        actor_leaf = jax.tree_util.tree_leaves(self.state.actor_params)[0]
        devices = sorted(str(device) for device in actor_leaf.devices())
        return format_table(
            [
                (
                    "system",
                    [
                        ("device", self.config.env.device),
                        ("jax_backend", jax.default_backend()),
                        ("parameter_devices", ", ".join(devices)),
                        ("physics", self.config.env.physics),
                        ("sim_frequency", f"{self.config.env.sim_hz} Hz"),
                        ("control_frequency", f"{self.config.env.control_hz} Hz"),
                    ],
                ),
                (
                    "training",
                    [
                        ("algorithm", "PPO"),
                        ("action_space", ACTION_SPACE),
                        ("num_envs", f"{self.config.env.num_envs:,}"),
                        ("rollout_length", f"{self.config.ppo.horizon:,}"),
                        ("batch_size", f"{steps_per_update:,}"),
                        ("start_timesteps", f"{self.state.env_steps:,}"),
                        ("target_timesteps", f"{target_env_steps:,}"),
                        ("schedule_timesteps", f"{self.config.ppo.schedule_env_steps:,}"),
                        ("schedule_consumed", f"{self.state.schedule_steps:,}"),
                        ("total_iterations", f"{total_updates:,}"),
                        ("actor_obs_dim", self.config.obs.dim),
                        ("critic_obs_dim", self.config.critic_obs.dim),
                        ("reward_version", self.config.env.reward_version),
                        ("curriculum", self.config.curriculum.enabled),
                        ("strict_evaluation", self.config.evaluation.enabled),
                        (
                            "skill_attempts_gate",
                            self.config.evaluation.skill_attempts_per_gate,
                        ),
                        ("checkpoint_dir", self.config.checkpoint_dir),
                        ("checkpoint_interval", self.config.checkpoint_interval),
                        ("metrics_file", self._metrics_path()),
                    ],
                ),
            ]
        )


def _nanmean_or_zero(x: Array) -> float:
    array = _to_numpy(x)
    valid = array[np.isfinite(array)]
    return float(np.mean(valid)) if valid.size else 0.0


def _finite_count(x: Array) -> int:
    return int(np.isfinite(_to_numpy(x)).sum())


def _to_numpy(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (jax.Array, np.ndarray)):
        return _json_safe(np.asarray(jax.device_get(value)).tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _safe_ratio(numerator: Array, denominator: Array) -> float:
    num = float(numerator)
    den = float(denominator)
    return num / den if den > 0.0 else 0.0


def _course_fingerprint(course: GateCourse) -> str:
    """Hash signed geometry and traversal metadata in a platform-stable form."""

    digest = hashlib.sha256()
    digest.update(course.name.encode("utf-8"))
    for values, dtype in (
        (course.centers, "<f4"),
        (course.normals, "<f4"),
        (course.widths, "<f4"),
        (course.heights, "<f4"),
        (course.outer_widths, "<f4"),
        (course.outer_heights, "<f4"),
        (course.logical_gate_ids, "<i4"),
        (course.nominal_racing_line, "<f4"),
        (course.bounds_min, "<f4"),
        (course.bounds_max, "<f4"),
        (np.asarray([course.radius_m]), "<f4"),
        (np.asarray(course.world_up), "<f4"),
        (np.asarray(course.arena_size), "<f4"),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        digest.update(np.asarray(array.shape, dtype="<i4").tobytes())
        digest.update(array.tobytes())
    digest.update("\0".join(course.openings).encode("utf-8"))
    return digest.hexdigest()


def _validate_checkpoint_observation_compatibility(
    path: Path,
    payload: dict[str, Any],
    config: TrainingConfig,
) -> None:
    checkpoint_config = payload.get("config")
    checkpoint_obs = getattr(checkpoint_config, "obs", None)
    checkpoint_obs_dim = getattr(checkpoint_obs, "dim", None)
    if checkpoint_obs_dim is not None and checkpoint_obs_dim != config.obs.dim:
        raise ValueError(
            "Checkpoint observation dimension is incompatible with this run: "
            f"path={path} checkpoint_obs_dim={checkpoint_obs_dim} "
            f"current_obs_dim={config.obs.dim}. Start a fresh run or use a matching architecture."
        )
    critic_layers = payload.get("critic_params", {}).get("value", [])
    if critic_layers:
        checkpoint_critic_dim = int(np.asarray(critic_layers[0]["w"]).shape[0])
        if checkpoint_critic_dim != config.critic_obs.dim:
            raise ValueError(
                "Checkpoint critic observation dimension is incompatible with asymmetric PPO: "
                f"path={path} checkpoint_critic_obs_dim={checkpoint_critic_dim} "
                f"current_critic_obs_dim={config.critic_obs.dim}."
            )


def _format_duration(seconds: float) -> str:
    rounded = int(round(max(float(seconds), 0.0)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_gate_counts(values: Any) -> str:
    array = np.asarray(values, dtype=np.float32)
    return ",".join(f"{index + 1}:{int(round(value))}" for index, value in enumerate(array))


def _format_gate_values(values: Any) -> str:
    array = np.asarray(values, dtype=np.float32)
    return ",".join(f"{index + 1}:{value:.3f}" for index, value in enumerate(array))


def _format_named_values(names: tuple[str, ...], values: Any, *, precision: int) -> str:
    array = np.asarray(values, dtype=np.float32)
    return "  ".join(
        f"{name}={value:.{precision}f}" for name, value in zip(names, array)
    )


def _format_gate_order(values: Any) -> str:
    count = np.asarray(values).size
    return " ".join(f"G{index + 1:02d}".rjust(5) for index in range(count))


def _format_gate_series(values: Any) -> str:
    array = np.asarray(values, dtype=np.float32)
    return " ".join(f"{value:5.2f}" for value in array)


def _format_gate_count_series(values: Any) -> str:
    array = np.asarray(values, dtype=np.float32)
    return " ".join(f"{int(round(value)):5d}" for value in array)


def _weakest_rate_summary(values: Any) -> str:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return "unavailable"
    index = int(np.argmin(array))
    return f"G{index + 1} ({array[index]:.1%})"


def _weak_gate_summary(rollout_metrics: dict[str, Any]) -> str:
    target = np.asarray(rollout_metrics.get("gate_target_counts", []), dtype=np.float32)
    passed = np.asarray(rollout_metrics.get("gate_pass_counts", []), dtype=np.float32)
    failed = np.asarray(
        rollout_metrics.get("gate_failure_counts", rollout_metrics.get("gate_stop_counts", [])),
        dtype=np.float32,
    )
    missed = np.asarray(rollout_metrics.get("gate_miss_counts", failed), dtype=np.float32)
    if target.size == 0:
        return "unavailable"
    if missed.size and np.max(missed) > 0.0:
        weak_index = int(np.argmax(missed))
        weak_reason = "miss"
    elif failed.size and np.max(failed) > 0.0:
        weak_index = int(np.argmax(failed))
        weak_reason = "stop"
    else:
        weak_index = int(np.argmax(target - passed))
        weak_reason = "time"
    return f"G{weak_index + 1} ({weak_reason})"


def _gate_diagnostic_line(rollout_metrics: dict[str, Any]) -> str:
    target = np.asarray(rollout_metrics.get("gate_target_counts", []), dtype=np.float32)
    passed = np.asarray(rollout_metrics.get("gate_pass_counts", []), dtype=np.float32)
    failed = np.asarray(
        rollout_metrics.get("gate_failure_counts", rollout_metrics.get("gate_stop_counts", [])),
        dtype=np.float32,
    )
    missed = np.asarray(rollout_metrics.get("gate_miss_counts", failed), dtype=np.float32)
    if target.size == 0:
        return "gate_diag unavailable"
    if missed.size and np.max(missed) > 0.0:
        weak_index = int(np.argmax(missed))
        weak_reason = "miss"
    elif np.max(failed) > 0.0:
        weak_index = int(np.argmax(failed))
        weak_reason = "stop"
    else:
        weak_index = int(np.argmax(target - passed))
        weak_reason = "time"
    return (
        f"gate_diag weak=G{weak_index + 1}:{weak_reason} "
        f"target={_format_gate_counts(target)} pass={_format_gate_counts(passed)} "
        f"fail={_format_gate_counts(failed)}"
    )
