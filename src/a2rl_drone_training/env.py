from __future__ import annotations

import contextlib
import io
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from a2rl_drone_training.config import ObservationConfig, RacingEnvConfig
from a2rl_drone_training.course import GateCourse, arena_38m_stacked_course
from a2rl_drone_training.curriculum import CurriculumParameters, sample_reset_gates
from a2rl_drone_training.observations import (
    build_privileged_observation,
    build_racing_observation,
    yaw_to_quat_xyzw,
)
from a2rl_drone_training.rewards import (
    gate_clearance,
    progress_potential,
    rebase_progress_potential,
    reward_v1,
    reward_v2,
)


_OPTIONAL_WARP_IMPORT_PREFIXES = (
    "Failed to import warp:",
    "Failed to import mujoco_warp:",
)


def _import_crazyflow():
    """Import Crazyflow without printing irrelevant optional-Warp failures."""

    captured_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout):
            from crazyflow.control import Control
            from crazyflow.sim import Physics, Sim
    except ImportError as exc:
        _replay_unexpected_import_output(captured_stdout.getvalue())
        raise RuntimeError(
            "Crazyflow is required for the training environment. Install with `pip install -e .`."
        ) from exc
    _replay_unexpected_import_output(captured_stdout.getvalue())
    return Control, Physics, Sim


def _replay_unexpected_import_output(output: str) -> None:
    unexpected = [
        line
        for line in output.splitlines()
        if not line.startswith(_OPTIONAL_WARP_IMPORT_PREFIXES)
    ]
    if unexpected:
        print("\n".join(unexpected), flush=True)


def classify_completion(
    next_gate_counter: Array,
    total_gate_passes: int,
    started_local: Array,
) -> tuple[Array, Array, Array]:
    segment_complete = next_gate_counter >= int(total_gate_passes)
    course_finished = segment_complete & ~started_local
    local_segment_complete = segment_complete & started_local
    return segment_complete, course_finished, local_segment_complete


class CrazyflowRacingEnv:
    """Vectorized Crazyflow racing environment for PPO rollout collection."""

    def __init__(
        self,
        env_config: RacingEnvConfig = RacingEnvConfig(),
        obs_config: ObservationConfig = ObservationConfig(),
        course: GateCourse | None = None,
    ):
        self.config = env_config
        if env_config.physics != "first_principles":
            raise ValueError("Direct motor control requires --physics first_principles; so_rpy models consume attitude commands.")
        self.obs_config = obs_config
        self.course = course or arena_38m_stacked_course()
        self.total_gate_passes = self.course.num_gates * self.config.laps
        self.n_substeps = self.config.sim_hz // self.config.control_hz
        if self.n_substeps < 1 or self.config.sim_hz % self.config.control_hz:
            raise ValueError("sim_hz must be a positive integer multiple of control_hz")

        Control, Physics, Sim = _import_crazyflow()

        self.sim = Sim(
            n_worlds=self.config.num_envs,
            n_drones=1,
            drone_model=self.config.drone_model,
            physics=Physics(self.config.physics),
            control=Control.rotor_vel,
            freq=self.config.sim_hz,
            attitude_freq=self.config.control_hz,
            device=self.config.device,
        )
        self.gate_centers = jnp.asarray(self.course.centers, dtype=jnp.float32)
        self.gate_normals = jnp.asarray(self.course.normals, dtype=jnp.float32)
        self.gate_right_axes = jnp.asarray(self.course.right_axes, dtype=jnp.float32)
        self.gate_widths = jnp.asarray(self.course.widths, dtype=jnp.float32)
        self.gate_heights = jnp.asarray(self.course.heights, dtype=jnp.float32)
        self.bounds_min = jnp.asarray(self.course.bounds_min, dtype=jnp.float32)
        self.bounds_max = jnp.asarray(self.course.bounds_max, dtype=jnp.float32)
        (
            self.approach_starts,
            self.approach_dirs,
            self.approach_right_axes,
            self.approach_lengths,
        ) = self._build_racing_line_spawn_geometry()
        evaluation = self.config.reset_distribution == "evaluation"
        self.curriculum_phase = "evaluation" if evaluation else "A"
        self.gate1_start_fraction = 1.0 if evaluation else 0.2
        self.gate_window_scale = (
            1.0
            if evaluation
            else float(self.config.gate_window_scale)
        )
        self.time_cost_scale = 1.0 if evaluation else 0.0
        self.prioritized_local_starts = False
        local_gate_count = max(self.course.num_gates - 1, 0)
        self.local_gate_probabilities = jnp.full(
            (local_gate_count,),
            1.0 / max(local_gate_count, 1),
            dtype=jnp.float32,
        )
        self.rng = jax.random.key(0)
        self.gate_counter = jnp.zeros((self.config.num_envs,), dtype=jnp.int32)
        self.reset_gate = jnp.zeros((self.config.num_envs,), dtype=jnp.int32)
        self.started_local = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        self.elapsed_steps = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.segment_steps = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.last_action = jnp.zeros((self.config.num_envs, 4), dtype=jnp.float32)
        self.prev_vel = jnp.zeros((self.config.num_envs, 3), dtype=jnp.float32)
        self.current_acc = jnp.zeros((self.config.num_envs, 3), dtype=jnp.float32)
        self.stall_steps = jnp.zeros((self.config.num_envs,), dtype=jnp.int32)
        self.episode_return = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.episode_length = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        self.thrust_min, self.thrust_hover, self.thrust_max = self._resolve_thrust_bounds()
        from drone_models.transform import motor_force2rotor_vel

        coefficient = np.asarray(self.sim.data.params.rpm2thrust)
        if coefficient.shape != (3,) or not np.all(np.isfinite(coefficient)) or coefficient[2] <= 0:
            raise ValueError("Motor thrust curve must be a finite quadratic with a positive leading coefficient")
        rpm_bounds = motor_force2rotor_vel(
            jnp.asarray([self.thrust_min, self.thrust_hover, self.thrust_max]) / 4.0,
            self.sim.data.params.rpm2thrust,
        )
        bounds = np.asarray(rpm_bounds)
        if not np.all(np.isfinite(bounds)) or not 0 <= bounds[0] < bounds[1] < bounds[2]:
            raise ValueError("Model thrust limits do not define valid motor RPM bounds")
        self.motor_rpm_min, self.motor_rpm_hover, self.motor_rpm_max = rpm_bounds
        self.reset(seed=0)

    @property
    def observation_dim(self) -> int:
        return self.obs_config.dim

    @property
    def action_dim(self) -> int:
        return 4

    def reset(
        self,
        *,
        seed: int | None = None,
        mask: Array | None = None,
        forced_reset_gates: Array | None = None,
    ) -> Array:
        if seed is not None:
            self.rng = jax.random.key(seed)
            self.sim.seed(seed)
        if mask is None:
            mask = jnp.ones((self.config.num_envs,), dtype=jnp.bool_)
        else:
            mask = jnp.asarray(mask, dtype=jnp.bool_)
        if forced_reset_gates is not None:
            forced_reset_gates = jnp.asarray(forced_reset_gates, dtype=jnp.int32)
            if forced_reset_gates.shape != (self.config.num_envs,):
                raise ValueError("forced_reset_gates must provide one gate per environment")
            if bool(
                jnp.any(
                    (forced_reset_gates < 0)
                    | (forced_reset_gates >= self.course.num_gates)
                )
            ):
                raise ValueError("forced_reset_gates contains an invalid runtime gate")

        self.sim.reset(mask=mask)
        self.rng, reset_key = jax.random.split(self.rng)
        pos, vel, quat, reset_gate, started_local = self._sample_initial_state(
            reset_key,
            mask,
            forced_reset_gates,
        )
        states = self.sim.data.states
        mask3 = mask[:, None, None]
        mask4 = mask[:, None, None]
        new_states = states.replace(
            pos=jnp.where(mask3, pos[:, None, :], states.pos),
            vel=jnp.where(mask3, vel[:, None, :], states.vel),
            quat=jnp.where(mask4, quat[:, None, :], states.quat),
            ang_vel=jnp.where(mask3, jnp.zeros_like(states.ang_vel), states.ang_vel),
            rotor_vel=jnp.where(mask3, self.motor_rpm_hover, states.rotor_vel),
        )
        self.sim.data = self.sim.data.replace(states=new_states)

        self.gate_counter = jnp.where(mask, reset_gate, self.gate_counter)
        self.reset_gate = jnp.where(mask, reset_gate, self.reset_gate)
        self.started_local = jnp.where(mask, started_local, self.started_local)
        self.elapsed_steps = jnp.where(mask, 0.0, self.elapsed_steps)
        self.segment_steps = jnp.where(mask, 0.0, self.segment_steps)
        self.last_action = jnp.where(mask[:, None], 0.0, self.last_action)
        self.prev_vel = jnp.where(mask[:, None], vel, self.prev_vel)
        self.current_acc = jnp.where(mask[:, None], 0.0, self.current_acc)
        self.stall_steps = jnp.where(mask, 0, self.stall_steps)
        self.episode_return = jnp.where(mask, 0.0, self.episode_return)
        self.episode_length = jnp.where(mask, 0.0, self.episode_length)
        return self.observe()

    def observe(self, *, noisy: bool | None = None) -> Array:
        states = self.sim.data.states
        pos = states.pos[:, 0, :]
        quat = states.quat[:, 0, :]
        vel = states.vel[:, 0, :]
        ang_vel = states.ang_vel[:, 0, :]
        acc = self.current_acc
        if noisy is None:
            noisy = self.config.reset_distribution == "training"
        obs_key = None
        if noisy:
            self.rng, obs_key = jax.random.split(self.rng)
        yaw_error = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        stall_fraction = self.stall_steps.astype(jnp.float32) / max(
            float(self.config.stall_patience_steps), 1.0
        )
        return build_racing_observation(
            pos_world=pos,
            quat_xyzw=quat,
            vel_world=vel,
            ang_vel_world=ang_vel,
            acc_world=acc,
            gate_centers_world=self.gate_centers,
            gate_normals_world=self.gate_normals,
            gate_counter=self.gate_counter,
            total_gate_passes=self.total_gate_passes,
            last_action=self.last_action,
            remaining_time_fraction=(
                1.0 - self.elapsed_steps / max(float(self.config.max_episode_steps), 1.0)
            ),
            yaw_error=yaw_error,
            stall_fraction=stall_fraction,
            config=self.obs_config,
            key=obs_key,
        )

    def privileged_observe(self) -> Array:
        """Noise-free value-function input; this is never passed to the actor."""

        states = self.sim.data.states
        pos = states.pos[:, 0, :]
        quat = states.quat[:, 0, :]
        vel = states.vel[:, 0, :]
        yaw_error = jnp.zeros((self.config.num_envs,), dtype=jnp.float32)
        return build_privileged_observation(
            pos_world=pos,
            quat_xyzw=quat,
            vel_world=vel,
            ang_vel_world=states.ang_vel[:, 0, :],
            acc_world=self.current_acc,
            gate_centers_world=self.gate_centers,
            gate_normals_world=self.gate_normals,
            gate_right_axes_world=self.gate_right_axes,
            gate_counter=self.gate_counter,
            total_gate_passes=self.total_gate_passes,
            elapsed_steps=self.elapsed_steps,
            max_episode_steps=self.config.max_episode_steps,
            last_action=self.last_action,
            yaw_error=yaw_error,
            gate_window_scale=self.gate_window_scale,
            started_local=self.started_local,
        )

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict[str, Array]]:
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (self.config.num_envs, 4):
            raise ValueError("Motor actions must have shape (num_envs, 4)")
        action = jnp.clip(action, -1.0, 1.0)
        previous_action = self.last_action
        pos_before = self.sim.data.states.pos[:, 0, :]
        gate_id = self.gate_counter % self.course.num_gates
        lookahead_gate_id = (gate_id + 1) % self.course.num_gates
        plane_before, radial_before, _ = self._gate_metrics(pos_before, gate_id)
        distance_before = self._gate_distance(pos_before, gate_id)
        lookahead_distance_before = self._gate_distance(pos_before, lookahead_gate_id)
        _, track_phi_before, center_phi_before = progress_potential(
            pos=pos_before,
            gate_counter=self.gate_counter,
            gate_centers=self.gate_centers,
            gate_normals=self.gate_normals,
            gate_right_axes=self.gate_right_axes,
            segment_starts=self.approach_starts,
            segment_directions=self.approach_dirs,
            segment_lengths=self.approach_lengths,
            track_scale=self.config.potential_track_scale,
            center_scale=self.config.potential_center_scale,
            plane_sigma_m=self.config.potential_plane_sigma_m,
            backtrack_limit=self.config.potential_backtrack_limit,
        )
        if self.config.reward_version == "v2":
            _, track_phi_before = rebase_progress_potential(
                track_phi_before,
                track_phi_before,
                self.reset_gate,
                self.config.potential_track_scale,
            )

        cmd = self._action_to_motor_rpm(action)
        self.sim.rotor_vel_control(cmd[:, None, :])
        self.sim.step(self.n_substeps)

        states = self.sim.data.states
        pos = states.pos[:, 0, :]
        vel = states.vel[:, 0, :]
        self.current_acc = self._acceleration_world(vel)
        gate_id = self.gate_counter % self.course.num_gates
        speed = jnp.linalg.norm(vel, axis=-1)
        forward_speed = jnp.sum(vel * self.gate_normals[gate_id], axis=-1)
        time_to_gate = distance_before / jnp.maximum(jnp.abs(forward_speed), 0.1)
        plane_after, radial_after, inside_gate = self._gate_metrics(pos, gate_id)
        distance_after = self._gate_distance(pos, gate_id)
        lookahead_distance_after = self._gate_distance(pos, lookahead_gate_id)
        crossed_plane = (plane_before < 0.0) & (plane_after >= 0.0)
        cross_pos = self._gate_crossing_position(pos_before, pos, plane_before, plane_after)
        _, cross_radial, cross_inside = self._gate_metrics(cross_pos, gate_id)
        _, cross_lateral, cross_vertical = self._gate_offsets(cross_pos, gate_id)
        passed_gate = crossed_plane & cross_inside
        strict_inside = self._inside_gate(
            cross_lateral,
            cross_vertical,
            gate_id,
            window_scale=1.0,
        )
        strict_passed_gate = crossed_plane & strict_inside
        missed_gate = (plane_after > self.config.gate_miss_depth_m) & ~passed_gate & ~inside_gate
        next_gate_counter = self.gate_counter + passed_gate.astype(jnp.int32)
        segment_complete, course_finished, local_segment_complete = classify_completion(
            next_gate_counter,
            self.total_gate_passes,
            self.started_local,
        )

        crashed = pos[:, 2] < 0.05
        out_of_bounds = jnp.any((pos < self.bounds_min) | (pos > self.bounds_max), axis=-1)
        deadline = self.elapsed_steps + 1 >= self.config.max_episode_steps
        terminated = crashed | out_of_bounds | missed_gate | segment_complete | deadline
        artificial_limit = self.config.artificial_time_limit_steps
        if artificial_limit is None:
            truncated = jnp.zeros_like(terminated)
        else:
            truncated = (self.elapsed_steps + 1 >= artificial_limit) & ~terminated
        distance_progress = distance_before - distance_after
        if self.config.reward_version == "v1":
            stalled_step = (
                distance_progress < self.config.stall_progress_threshold_m
            ) & ~passed_gate
            next_stall_steps = jnp.where(stalled_step, self.stall_steps + 1, 0)
            stalled = next_stall_steps >= self.config.stall_patience_steps
        else:
            next_stall_steps = jnp.zeros_like(self.stall_steps)
            stalled = jnp.zeros_like(terminated)

        _, track_phi_after, _ = progress_potential(
            pos=pos,
            gate_counter=next_gate_counter,
            gate_centers=self.gate_centers,
            gate_normals=self.gate_normals,
            gate_right_axes=self.gate_right_axes,
            segment_starts=self.approach_starts,
            segment_directions=self.approach_dirs,
            segment_lengths=self.approach_lengths,
            track_scale=self.config.potential_track_scale,
            center_scale=self.config.potential_center_scale,
            plane_sigma_m=self.config.potential_plane_sigma_m,
            backtrack_limit=self.config.potential_backtrack_limit,
        )
        track_phi_after = jnp.where(
            segment_complete,
            self.config.potential_track_scale * next_gate_counter.astype(jnp.float32),
            track_phi_after,
        )
        # Keep the centering target fixed to the pre-transition gate. Track
        # progress alone advances to the next gate's course coordinate.
        _, _, center_phi_after = progress_potential(
            pos=pos,
            gate_counter=self.gate_counter,
            gate_centers=self.gate_centers,
            gate_normals=self.gate_normals,
            gate_right_axes=self.gate_right_axes,
            segment_starts=self.approach_starts,
            segment_directions=self.approach_dirs,
            segment_lengths=self.approach_lengths,
            track_scale=self.config.potential_track_scale,
            center_scale=self.config.potential_center_scale,
            plane_sigma_m=self.config.potential_plane_sigma_m,
            backtrack_limit=self.config.potential_backtrack_limit,
        )
        if self.config.reward_version == "v2":
            _, track_phi_after = rebase_progress_potential(
                track_phi_after,
                track_phi_after,
                self.reset_gate,
                self.config.potential_track_scale,
            )
        track_progress_delta = track_phi_after - track_phi_before
        physical_clearance = gate_clearance(
            cross_lateral,
            cross_vertical,
            self.gate_widths[gate_id],
            self.gate_heights[gate_id],
        )
        curriculum_clearance = gate_clearance(
            cross_lateral,
            cross_vertical,
            self.gate_widths[gate_id] * self.gate_window_scale,
            self.gate_heights[gate_id] * self.gate_window_scale,
        )
        velocity_alignment = self._horizontal_alignment(
            vel,
            self.gate_centers[gate_id] - pos,
        )
        if self.config.reward_version == "v1":
            reward, reward_components = reward_v1(
                plane_before=plane_before,
                plane_after=plane_after,
                radial_before=radial_before,
                radial_after=radial_after,
                distance_before=distance_before,
                distance_after=distance_after,
                lookahead_distance_before=lookahead_distance_before,
                lookahead_distance_after=lookahead_distance_after,
                cross_radial=cross_radial,
                velocity_alignment=velocity_alignment,
                stalled=stalled,
                action=action,
                gate_id=gate_id,
                passed_gate=passed_gate,
                missed_gate=missed_gate,
                crashed=crashed | out_of_bounds,
                finished=segment_complete,
                course_radius_m=self.course.radius_m,
                num_gates=self.course.num_gates,
                config=self.config,
            )
        else:
            deadline_failure = (
                deadline
                & ~segment_complete
                & ~missed_gate
                & ~crashed
                & ~out_of_bounds
            )
            reward, reward_components = reward_v2(
                track_phi_before=track_phi_before,
                track_phi_after=track_phi_after,
                center_phi_before=center_phi_before,
                center_phi_after=center_phi_after,
                action=action,
                previous_action=previous_action,
                clearance=physical_clearance,
                passed_gate=passed_gate,
                finished=course_finished,
                missed_gate=missed_gate | deadline_failure,
                crashed=crashed | out_of_bounds,
                dt=self.config.dt,
                time_cost_scale=self.time_cost_scale,
                position_noise_std_m=(
                    max(float(self.obs_config.sensor_noise_scale), 0.0)
                    * max(float(self.obs_config.gate_position_noise_std_m), 0.0)
                ),
                config=self.config,
            )
        done = terminated | truncated

        self.gate_counter = jnp.minimum(next_gate_counter, self.total_gate_passes - 1)
        self.elapsed_steps = self.elapsed_steps + 1.0
        segment_time = (self.segment_steps + 1.0) * self.config.dt
        self.segment_steps = jnp.where(passed_gate | done, 0.0, self.segment_steps + 1.0)
        self.last_action = action
        self.stall_steps = next_stall_steps

        final_return = self.episode_return + reward
        final_length = self.episode_length + 1.0
        self.episode_return = jnp.where(done, 0.0, final_return)
        self.episode_length = jnp.where(done, 0.0, final_length)

        # These are captured before auto-reset. GAE may bootstrap from the critic
        # version on truncation, but never from a reset observation.
        final_observation = self.observe()
        final_critic_observation = self.privileged_observe()
        self.prev_vel = vel

        info = {
            "passed_gate": passed_gate,
            "strict_passed_gate": strict_passed_gate,
            "missed_gate": missed_gate,
            "crashed": crashed,
            "out_of_bounds": out_of_bounds,
            # ``finished`` retains the legacy segment-end meaning for V1 consumers.
            "finished": segment_complete,
            "segment_complete": segment_complete,
            "course_finished": course_finished,
            "local_segment_complete": local_segment_complete,
            "deadline": deadline,
            "gate_counter": next_gate_counter,
            "gate_id": gate_id,
            "episode_gate_progress": next_gate_counter - self.reset_gate,
            "reset_gate": self.reset_gate,
            "started_gate1": ~self.started_local,
            "started_local": self.started_local,
            # Compatibility aliases for existing dashboards.
            "random_start": self.started_local,
            "focused_start": jnp.zeros_like(self.started_local),
            "speed": speed,
            "forward_speed": forward_speed,
            "time_to_gate": time_to_gate,
            "stalled": stalled,
            "gate_window_scale": jnp.full(
                (self.config.num_envs,),
                self.gate_window_scale,
                dtype=jnp.float32,
            ),
            "time_cost_scale": jnp.full(
                (self.config.num_envs,),
                self.time_cost_scale,
                dtype=jnp.float32,
            ),
            "gate_clearance": jnp.where(passed_gate, physical_clearance, jnp.nan),
            "curriculum_gate_clearance": jnp.where(
                passed_gate,
                curriculum_clearance,
                jnp.nan,
            ),
            "segment_time": jnp.where(passed_gate, segment_time, jnp.nan),
            "track_progress_delta": track_progress_delta,
            "action_delta_squared": jnp.square(action - previous_action),
            "final_return": jnp.where(done, final_return, jnp.nan),
            "final_length": jnp.where(done, final_length, jnp.nan),
            "final_observation": final_observation,
            "final_critic_observation": final_critic_observation,
            **reward_components,
        }

        reset_occurred = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        reset_started_local = jnp.zeros((self.config.num_envs,), dtype=jnp.bool_)
        if self.config.auto_reset and bool(jnp.any(done)):
            obs = self.reset(mask=done)
            reset_occurred = done
            reset_started_local = done & self.started_local
        else:
            obs = final_observation
        info["reset_occurred"] = reset_occurred
        info["reset_started_local"] = reset_started_local
        info["reset_random_start"] = reset_started_local
        info["reset_focused_start"] = jnp.zeros_like(reset_started_local)
        return obs, reward, terminated, truncated, info

    def render(self, **kwargs: Any) -> Any:
        return self.sim.render(**kwargs)

    def close(self) -> None:
        self.sim.close()

    def set_curriculum(self, parameters: CurriculumParameters) -> None:
        if self.config.reset_distribution == "evaluation":
            self.curriculum_phase = "evaluation"
            self.gate1_start_fraction = 1.0
            self.gate_window_scale = 1.0
            self.time_cost_scale = 1.0
            self.prioritized_local_starts = False
            return
        self.curriculum_phase = parameters.phase
        self.gate1_start_fraction = float(np.clip(parameters.gate1_fraction, 0.0, 1.0))
        self.gate_window_scale = float(parameters.gate_window_scale)
        self.time_cost_scale = float(np.clip(parameters.time_cost_scale, 0.0, 1.0))
        self.prioritized_local_starts = bool(parameters.prioritized_local_starts)
        probabilities = np.asarray(parameters.local_gate_probabilities, dtype=np.float32)
        if probabilities.shape != (max(self.course.num_gates - 1, 0),):
            raise ValueError("Curriculum local gate probabilities do not match the course")
        self.local_gate_probabilities = jnp.asarray(probabilities)

    def set_skill_audit_window_scale(self, scale: float) -> None:
        """Apply Phase-A aperture only to the dedicated noise-free audit env."""

        if self.config.reset_distribution != "evaluation" or self.config.auto_reset:
            raise RuntimeError("Skill-audit aperture requires a non-auto-reset evaluation env")
        if scale < 1.0:
            raise ValueError("Skill-audit gate window scale cannot be smaller than physical")
        self.gate_window_scale = float(scale)

    def _resolve_thrust_bounds(self) -> tuple[float, float, float]:
        thrust_min = self.config.thrust_min_n
        thrust_max = self.config.thrust_max_n
        if thrust_min is None or thrust_max is None:
            try:
                from drone_controllers.mellinger.params import ForceTorqueParams

                params = ForceTorqueParams.load(self.config.drone_model)
                thrust_min = float(params.thrust_min * 4.0) if thrust_min is None else thrust_min
                thrust_max = float(params.thrust_max * 4.0) if thrust_max is None else thrust_max
            except Exception as exc:
                raise ValueError("Cannot resolve motor thrust limits for the configured drone model") from exc

        if self.config.thrust_hover_n is not None:
            thrust_hover = self.config.thrust_hover_n
        else:
            try:
                mass = float(jnp.mean(self.sim.data.params.mass))
                thrust_hover = mass * 9.81
            except Exception:
                thrust_hover = 0.5 * (float(thrust_min) + float(thrust_max))
        if not all(np.isfinite(x) for x in (thrust_min, thrust_hover, thrust_max)) or not 0 <= thrust_min < thrust_hover < thrust_max:
            raise ValueError("Thrust limits must satisfy 0 <= min < hover < max")
        return float(thrust_min), thrust_hover, float(thrust_max)

    def _action_to_motor_rpm(self, action: Array) -> Array:
        """Map each motor independently: -1=min RPM, 0=hover RPM, +1=max RPM."""
        action = jnp.clip(action, -1.0, 1.0)
        return self.motor_rpm_hover + jnp.where(
            action >= 0.0,
            action * (self.motor_rpm_max - self.motor_rpm_hover),
            action * (self.motor_rpm_hover - self.motor_rpm_min),
        )

    def _build_racing_line_spawn_geometry(self) -> tuple[Array, Array, Array, Array]:
        centers = np.asarray(self.course.centers, dtype=np.float32)
        line = np.asarray(self.course.nominal_racing_line, dtype=np.float32)
        if line.shape[0] >= self.course.num_gates + 1:
            starts = line[: self.course.num_gates]
        else:
            starts = np.empty_like(centers)
            starts[0] = centers[0] - self.config.spawn_distance_m * self.course.normals[0]
            starts[1:] = centers[:-1]

        normals = np.asarray(self.course.normals, dtype=np.float32)
        vectors = centers - starts
        lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
        line_dirs = vectors / np.maximum(lengths, 1.0e-6)
        normal_alignment = np.sum(line_dirs * normals, axis=-1, keepdims=True)
        use_line_dir = (lengths > 1.0e-6) & (normal_alignment > 0.25)
        dirs = np.where(use_line_dir, line_dirs, normals)
        lengths = np.where(
            use_line_dir,
            lengths,
            np.maximum(lengths, float(self.config.racing_line_spawn_max_distance_m)),
        )

        up = np.asarray(self.course.world_up, dtype=np.float32)
        right = np.cross(up[None, :], dirs)
        right_lengths = np.linalg.norm(right, axis=-1, keepdims=True)
        fallback_right = np.asarray(self.course.right_axes, dtype=np.float32)
        right = np.where(
            right_lengths > 1.0e-6,
            right / np.maximum(right_lengths, 1.0e-6),
            fallback_right,
        )

        return (
            jnp.asarray(starts, dtype=jnp.float32),
            jnp.asarray(dirs, dtype=jnp.float32),
            jnp.asarray(right, dtype=jnp.float32),
            jnp.asarray(lengths[:, 0], dtype=jnp.float32),
        )

    def _sample_initial_state(
        self,
        key: Array,
        reset_mask: Array,
        forced_reset_gates: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        (
            reset_key,
            distance_key,
            lateral_key,
            vertical_key,
            vel_key,
            yaw_key,
        ) = jax.random.split(key, 6)
        if forced_reset_gates is None:
            reset_gate, started_local = sample_reset_gates(
                key=reset_key,
                reset_mask=reset_mask,
                current_reset_gate=self.reset_gate,
                gate1_fraction=self.gate1_start_fraction,
                local_gate_probabilities=self.local_gate_probabilities,
                prioritized=self.prioritized_local_starts,
                evaluation=self.config.reset_distribution == "evaluation",
            )
        else:
            reset_gate = jnp.where(
                reset_mask,
                forced_reset_gates,
                self.reset_gate,
            )
            started_local = reset_mask & (reset_gate != 0)
        gate_id = reset_gate % self.course.num_gates
        centers = self.gate_centers[gate_id]
        approach_dirs = self.approach_dirs[gate_id]
        right_axes = self.approach_right_axes[gate_id]
        approach_lengths = self.approach_lengths[gate_id]
        widths = self.gate_widths[gate_id] * self.gate_window_scale
        heights = self.gate_heights[gate_id] * self.gate_window_scale
        spawn_min = min(
            float(self.config.racing_line_spawn_min_distance_m),
            float(self.config.racing_line_spawn_max_distance_m),
        )
        spawn_max = max(
            float(self.config.racing_line_spawn_min_distance_m),
            float(self.config.racing_line_spawn_max_distance_m),
        )
        max_distance = jnp.minimum(spawn_max, jnp.maximum(0.1, 0.85 * approach_lengths))
        min_distance = jnp.minimum(spawn_min, max_distance)
        approach_distance = jax.random.uniform(
            distance_key,
            (self.config.num_envs,),
            minval=min_distance,
            maxval=max_distance,
        )
        lateral_noise = (
            jax.random.uniform(lateral_key, (self.config.num_envs,), minval=-0.35, maxval=0.35)
            * widths
        )
        vertical_noise = (
            jax.random.uniform(vertical_key, (self.config.num_envs,), minval=-0.25, maxval=0.25)
            * heights
        )
        pos = (
            centers
            - approach_distance[:, None] * approach_dirs
            + lateral_noise[:, None] * right_axes
            + vertical_noise[:, None] * jnp.array([0.0, 0.0, 1.0])
        )
        pos = pos.at[:, 2].set(jnp.maximum(pos[:, 2], 0.35))
        vel = (
            jax.random.normal(vel_key, (self.config.num_envs, 3))
            * self.config.spawn_vel_noise_m_s
        )
        vel = vel + jnp.where(
            started_local[:, None],
            self.config.racing_line_spawn_speed_m_s * approach_dirs,
            0.0,
        )
        fallback_normals = self.gate_normals[gate_id]
        yaw_dirs = jnp.where(
            jnp.linalg.norm(approach_dirs[:, :2], axis=-1, keepdims=True) > 1.0e-5,
            approach_dirs,
            fallback_normals,
        )
        course_yaw = jnp.arctan2(yaw_dirs[:, 1], yaw_dirs[:, 0])
        yaw = course_yaw + jax.random.normal(yaw_key, (self.config.num_envs,)) * 0.08
        quat = yaw_to_quat_xyzw(yaw)
        return (
            pos.astype(jnp.float32),
            vel.astype(jnp.float32),
            quat.astype(jnp.float32),
            reset_gate,
            started_local,
        )

    def _acceleration_world(self, vel: Array) -> Array:
        sim_acc = getattr(self.sim.data.states_deriv, "acc", None)
        if sim_acc is not None:
            return sim_acc[:, 0, :]
        return (vel - self.prev_vel) / self.config.dt

    def _gate_metrics(self, pos: Array, gate_id: Array) -> tuple[Array, Array, Array]:
        plane, lateral, vertical = self._gate_offsets(pos, gate_id)
        radial = jnp.sqrt(jnp.square(lateral) + jnp.square(vertical))
        inside = self._inside_gate(
            lateral,
            vertical,
            gate_id,
            window_scale=self.gate_window_scale,
        )
        return plane, radial, inside

    def _inside_gate(
        self,
        lateral: Array,
        vertical: Array,
        gate_id: Array,
        *,
        window_scale: float,
    ) -> Array:
        widths = self.gate_widths[gate_id] * float(window_scale)
        heights = self.gate_heights[gate_id] * float(window_scale)
        return (jnp.abs(lateral) <= 0.5 * widths) & (
            jnp.abs(vertical) <= 0.5 * heights
        )

    def _gate_offsets(self, pos: Array, gate_id: Array) -> tuple[Array, Array, Array]:
        rel = pos - self.gate_centers[gate_id]
        plane = jnp.sum(rel * self.gate_normals[gate_id], axis=-1)
        lateral = jnp.sum(rel * self.gate_right_axes[gate_id], axis=-1)
        vertical = rel[..., 2]
        return plane, lateral, vertical

    def _gate_distance(self, pos: Array, gate_id: Array) -> Array:
        centers = self.gate_centers[gate_id]
        return jnp.linalg.norm(pos - centers, axis=-1)

    @staticmethod
    def _gate_crossing_position(
        pos_before: Array,
        pos_after: Array,
        plane_before: Array,
        plane_after: Array,
    ) -> Array:
        denom = plane_after - plane_before
        t = jnp.where(jnp.abs(denom) > 1.0e-6, -plane_before / denom, 1.0)
        t = jnp.clip(t, 0.0, 1.0)
        return pos_before + t[:, None] * (pos_after - pos_before)

    @staticmethod
    def _horizontal_alignment(a: Array, b: Array) -> Array:
        horizontal_mask = jnp.array([1.0, 1.0, 0.0], dtype=a.dtype)
        a_xy = a * horizontal_mask
        b_xy = b * horizontal_mask
        a_xy = a_xy / jnp.maximum(
            jnp.linalg.norm(a_xy, axis=-1, keepdims=True),
            1.0e-6,
        )
        b_xy = b_xy / jnp.maximum(
            jnp.linalg.norm(b_xy, axis=-1, keepdims=True),
            1.0e-6,
        )
        return jnp.sum(a_xy * b_xy, axis=-1)
