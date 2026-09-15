import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import (
    EvaluationConfig,
    NetworkConfig,
    ObservationConfig,
    PPOConfig,
    RacingEnvConfig,
    TrainingConfig,
)
from a2rl_drone_training.course import course_by_name
from a2rl_drone_training.curriculum import CurriculumParameters
from a2rl_drone_training.env import CrazyflowRacingEnv, classify_completion
from a2rl_drone_training.observations import quat_to_yaw_xyzw, wrap_pi
from a2rl_drone_training.rewards import course_coordinate


def _env_config(**overrides):
    values = {
        "num_envs": 10,
        "sim_hz": 200,
        "control_hz": 100,
        "physics": "first_principles",
        "max_episode_time_s": 1.0,
        "device": "cpu",
    }
    values.update(overrides)
    return RacingEnvConfig(**values)


class EnvironmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.course = course_by_name("compact_slalom")
        self.obs_config = ObservationConfig(sensor_noise_scale=0.0, pnp_dropout_prob=0.0)

    def test_training_resets_obey_active_curriculum(self):
        env = CrazyflowRacingEnv(_env_config(), self.obs_config, self.course)
        try:
            local_count = self.course.num_gates - 1
            env.set_curriculum(
                CurriculumParameters(
                    phase="B",
                    gate1_fraction=0.5,
                    gate_window_scale=1.3,
                    time_cost_scale=0.25,
                    prioritized_local_starts=False,
                    local_gate_probabilities=np.ones((local_count,)) / local_count,
                )
            )
            env.reset(seed=17)
            self.assertEqual(int(jnp.sum(env.reset_gate == 0)), 5)
            self.assertEqual(int(jnp.sum(env.started_local)), 5)
            self.assertAlmostEqual(env.gate_window_scale, 1.3)
        finally:
            env.close()

    def test_evaluation_resets_are_gate_one_strict_and_deterministic(self):
        env = CrazyflowRacingEnv(
            _env_config(reset_distribution="evaluation", auto_reset=False),
            self.obs_config,
            self.course,
        )
        try:
            first = env.reset(seed=123)
            second = env.reset(seed=123)
            self.assertFalse(bool(jnp.any(env.reset_gate)))
            self.assertFalse(bool(jnp.any(env.started_local)))
            self.assertEqual(env.gate_window_scale, 1.0)
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        finally:
            env.close()

    def test_skill_audit_can_force_deterministic_gate_coverage_at_active_window(self):
        env = CrazyflowRacingEnv(
            _env_config(reset_distribution="evaluation", auto_reset=False),
            self.obs_config,
            self.course,
        )
        try:
            forced = jnp.arange(env.config.num_envs, dtype=jnp.int32) % self.course.num_gates
            first = env.reset(seed=123, forced_reset_gates=forced)
            first_gates = np.asarray(env.reset_gate)
            second = env.reset(seed=123, forced_reset_gates=forced)
            np.testing.assert_array_equal(first_gates, np.asarray(forced))
            np.testing.assert_array_equal(np.asarray(env.reset_gate), np.asarray(forced))
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
            self.assertEqual(env.gate_window_scale, 1.0)
            env.set_skill_audit_window_scale(1.8)
            self.assertEqual(env.gate_window_scale, 1.8)
        finally:
            env.close()

    def test_relaxed_crossing_advances_but_uses_physical_clearance_and_margin(self):
        course = course_by_name("arena_38m_stacked")
        env = CrazyflowRacingEnv(_env_config(num_envs=1), self.obs_config, course)
        try:
            env.set_curriculum(
                CurriculumParameters(
                    phase="A",
                    gate1_fraction=0.2,
                    gate_window_scale=1.8,
                    time_cost_scale=0.0,
                    prioritized_local_starts=False,
                    local_gate_probabilities=np.ones((11,)) / 11.0,
                )
            )
            env.reset(seed=4, forced_reset_gates=jnp.array([1]))
            states = env.sim.data.states
            center = env.gate_centers[1]
            normal = env.gate_normals[1]
            right = env.gate_right_axes[1]
            crossing_start = center - 0.02 * normal + 0.90 * right
            pos = states.pos.at[:, 0, :].set(crossing_start[None, :])
            vel = states.vel.at[:, 0, :].set(5.0 * normal[None, :])
            env.sim.data = env.sim.data.replace(states=states.replace(pos=pos, vel=vel))

            _, _, _, _, info = env.step(jnp.zeros((1, 4), dtype=jnp.float32))
            self.assertTrue(bool(info["passed_gate"][0]))
            self.assertFalse(bool(info["strict_passed_gate"][0]))
            self.assertLess(float(info["gate_clearance"][0]), 0.0)
            self.assertGreater(float(info["curriculum_gate_clearance"][0]), 0.0)
            self.assertEqual(float(info["reward_gate"][0]), 6.0)
            self.assertAlmostEqual(float(info["reward_margin"][0]), -6.0, places=5)
        finally:
            env.close()

    def test_corrected_stacked_transition_preserves_course_coordinate(self):
        course = course_by_name("arena_38m_stacked")
        env = CrazyflowRacingEnv(_env_config(num_envs=1), self.obs_config, course)
        try:
            epsilon = 1.0e-3
            before_pos = env.gate_centers[10][None, :] - epsilon * env.gate_normals[10][None, :]
            after_pos = env.gate_centers[10][None, :] + epsilon * env.gate_normals[10][None, :]
            before = course_coordinate(
                before_pos,
                jnp.array([10], dtype=jnp.int32),
                env.approach_starts,
                env.approach_dirs,
                env.approach_lengths,
            )
            after = course_coordinate(
                after_pos,
                jnp.array([11], dtype=jnp.int32),
                env.approach_starts,
                env.approach_dirs,
                env.approach_lengths,
            )
            self.assertGreaterEqual(float(after[0]), float(before[0]))
            self.assertLess(float(after[0] - before[0]), 0.01)
            g12_plane = jnp.sum(
                (after_pos[0] - env.gate_centers[11]) * env.gate_normals[11]
            )
            self.assertLess(float(g12_plane), 0.0)
        finally:
            env.close()

    def test_local_segment_completion_is_not_course_finish(self):
        segment, course, local = classify_completion(
            jnp.array([12, 12], dtype=jnp.int32),
            12,
            jnp.array([True, False]),
        )
        np.testing.assert_array_equal(np.asarray(segment), [True, True])
        np.testing.assert_array_equal(np.asarray(course), [False, True])
        np.testing.assert_array_equal(np.asarray(local), [True, False])

    def test_final_gate_rewards_depend_on_episode_origin(self):
        course = course_by_name("arena_38m_stacked")
        self.assertEqual(course.num_gates, 12)
        env = CrazyflowRacingEnv(
            _env_config(num_envs=1), self.obs_config, course
        )

        def cross_final_gate():
            states = env.sim.data.states
            center = env.gate_centers[-1]
            normal = env.gate_normals[-1]
            pos = states.pos.at[:, 0, :].set(center[None, :] - 0.02 * normal[None, :])
            vel = states.vel.at[:, 0, :].set(5.0 * normal[None, :])
            env.sim.data = env.sim.data.replace(states=states.replace(pos=pos, vel=vel))
            return env.step(jnp.zeros((1, 4), dtype=jnp.float32))

        try:
            env.reset(seed=3, forced_reset_gates=jnp.array([11]))
            _, _, terminated, truncated, info = cross_final_gate()
            self.assertTrue(bool(info["segment_complete"][0]))
            self.assertTrue(bool(info["local_segment_complete"][0]))
            self.assertFalse(bool(info["course_finished"][0]))
            self.assertEqual(float(info["reward_gate"][0]), 6.0)
            self.assertEqual(float(info["reward_finish"][0]), 0.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))

            env.reset(seed=3, forced_reset_gates=jnp.array([0]))
            env.gate_counter = jnp.array([11], dtype=jnp.int32)
            _, _, terminated, truncated, info = cross_final_gate()
            self.assertTrue(bool(info["segment_complete"][0]))
            self.assertFalse(bool(info["local_segment_complete"][0]))
            self.assertTrue(bool(info["course_finished"][0]))
            self.assertEqual(float(info["reward_gate"][0]), 6.0)
            self.assertEqual(float(info["reward_finish"][0]), 25.0)
            self.assertTrue(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
        finally:
            env.close()

    def test_direct_motors_are_independent_and_bounded(self):
        env = CrazyflowRacingEnv(_env_config(auto_reset=False), self.obs_config, self.course)
        try:
            obs = env.reset(seed=5)
            np.testing.assert_allclose(np.asarray(obs[:, 22]), 0.0)
            actions = jnp.zeros((env.config.num_envs, 4)).at[:, 0].set(2.0).at[:, 1].set(-2.0)
            rpm = env._action_to_motor_rpm(actions)
            np.testing.assert_allclose(np.asarray(rpm[:, 0]), env.motor_rpm_max, rtol=1e-6)
            np.testing.assert_allclose(np.asarray(rpm[:, 1]), env.motor_rpm_min, atol=0.01)
            np.testing.assert_allclose(np.asarray(rpm[:, 2:]), env.motor_rpm_hover, rtol=1e-6)
            env.step(actions)
            np.testing.assert_allclose(np.asarray(env.sim.data.controls.rotor_vel[:, 0]), rpm, rtol=1e-6)
            self.assertGreater(float(jnp.max(jnp.abs(env.sim.data.states.ang_vel))), 0.0)
            np.testing.assert_allclose(np.asarray(env.observe()[:, 22]), 0.0)
        finally:
            env.close()

    def test_artificial_limit_truncates_and_preserves_final_pre_reset_state(self):
        env = CrazyflowRacingEnv(
            _env_config(artificial_time_limit_s=0.01, auto_reset=True),
            self.obs_config,
            self.course,
        )
        try:
            env.reset(seed=9)
            _, _, terminated, truncated, info = env.step(
                jnp.zeros((env.config.num_envs, 4), dtype=jnp.float32)
            )
            self.assertFalse(bool(jnp.any(terminated)))
            self.assertTrue(bool(jnp.all(truncated)))
            self.assertTrue(bool(jnp.all(info["reset_occurred"])))
            self.assertEqual(info["final_critic_observation"].shape[-1], 37)
            self.assertFalse(
                np.array_equal(
                    np.asarray(info["final_critic_observation"]),
                    np.asarray(env.privileged_observe()),
                )
            )
        finally:
            env.close()

    def test_competition_deadline_is_terminal(self):
        env = CrazyflowRacingEnv(
            _env_config(max_episode_time_s=0.01, artificial_time_limit_s=None),
            self.obs_config,
            self.course,
        )
        try:
            _, _, terminated, truncated, info = env.step(
                jnp.zeros((env.config.num_envs, 4), dtype=jnp.float32)
            )
            self.assertTrue(bool(jnp.all(terminated)))
            self.assertFalse(bool(jnp.any(truncated)))
            self.assertTrue(bool(jnp.all(info["deadline"])))
        finally:
            env.close()


class TrainerSmokeTests(unittest.TestCase):
    def test_vectorized_training_checkpoint_and_deterministic_evaluation(self):
        from a2rl_drone_training.trainer import PPOTrainer

        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig(
                env=_env_config(
                    num_envs=4,
                    max_episode_time_s=0.05,
                    reward_version="v2",
                ),
                obs=ObservationConfig(sensor_noise_scale=0.0, pnp_dropout_prob=0.0),
                net=NetworkConfig(
                    state_hidden=(8,),
                    state_latent_dim=8,
                    gate_hidden=(8,),
                    gate_latent_dim=8,
                    fusion_hidden=(8,),
                    critic_hidden=(8,),
                ),
                ppo=PPOConfig(
                    total_env_steps=48,
                    schedule_env_steps=96,
                    horizon=4,
                    minibatches=1,
                    update_epochs=1,
                    log_interval=10,
                    target_kl=0.015,
                ),
                evaluation=EvaluationConfig(
                    enabled=True,
                    interval_updates=0,
                    num_envs=2,
                    seed=77,
                    skill_attempts_per_gate=1,
                    skill_max_time_s=0.02,
                ),
                checkpoint_dir=Path(directory),
                checkpoint_interval=10,
            )
            trainer = PPOTrainer(config, course=self._compact_course())
            try:
                trainer.train()
                self.assertTrue(np.isfinite(np.asarray(trainer.state.obs)).all())
                self.assertTrue(np.isfinite(np.asarray(trainer.state.critic_obs)).all())
                self.assertEqual(trainer.curriculum.state.updates, 3)
                first_eval = trainer.evaluate_policy()
                second_eval = trainer.evaluate_policy()
                np.testing.assert_array_equal(
                    first_eval["gate_passes"], second_eval["gate_passes"]
                )
                first_skill = trainer.evaluate_gate_skills()
                second_skill = trainer.evaluate_gate_skills()
                np.testing.assert_array_equal(
                    first_skill["gate_passes"], second_skill["gate_passes"]
                )
                np.testing.assert_array_equal(
                    first_skill["strict_gate_passes"],
                    second_skill["strict_gate_passes"],
                )
                self.assertEqual(first_skill["gate_window_scale"], 1.8)
                self.assertEqual(trainer._get_evaluation_env().gate_window_scale, 1.0)
                checkpoint = Path(directory) / "checkpoint_000003.pkl"
                self.assertTrue(checkpoint.exists())
                self.assertTrue((Path(directory) / "checkpoint_latest.pkl").exists())
                self.assertEqual(list(Path(directory).glob(".*.tmp")), [])
                metrics_path = Path(directory) / "metrics.jsonl"
                self.assertTrue(metrics_path.exists())
                records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
                self.assertEqual(len(records), 3)
                self.assertFalse(records[1]["detailed"])
                self.assertIsNone(records[1]["evaluation"])
                self.assertEqual(records[-1]["update"], 3)
                self.assertTrue(records[-1]["detailed"])
                self.assertIsNotNone(records[-1]["evaluation"])
                self.assertIsNotNone(records[-1]["skill_evaluation"])
                normalization_count = float(trainer.normalization_state.count)
                phase = trainer.curriculum.phase
                skill_attempts = trainer.curriculum.state.skill_attempts.copy()
                recent_attempts = trainer.curriculum.state.skill_recent_attempts.copy()
            finally:
                trainer.close()

            restored = PPOTrainer(
                config,
                course=self._compact_course(),
                restore_checkpoint=checkpoint,
            )
            try:
                self.assertEqual(float(restored.normalization_state.count), normalization_count)
                self.assertEqual(restored.curriculum.phase, phase)
                self.assertEqual(restored.state.env_steps, 48)
                self.assertEqual(restored.state.updates, 3)
                self.assertEqual(restored.state.schedule_steps, 48)
                self.assertEqual(restored._schedule_progress(), 0.5)
                np.testing.assert_array_equal(
                    restored.curriculum.state.skill_attempts,
                    skill_attempts,
                )
                np.testing.assert_array_equal(
                    restored.curriculum.state.skill_recent_attempts,
                    recent_attempts,
                )
            finally:
                restored.close()

            with checkpoint.open("rb") as file:
                legacy_payload = pickle.load(file)
            self.assertEqual(legacy_payload["checkpoint_version"], 6)
            self.assertIn("course_fingerprint", legacy_payload)
            legacy_payload.pop("schedule_steps")
            legacy_payload.pop("course_fingerprint")
            legacy_payload["checkpoint_version"] = 2
            legacy_payload["schedule_progress"] = 0.5
            for key in tuple(legacy_payload["curriculum_state"]):
                if key.startswith("skill_recent_") or key == "skill_strict_passes":
                    legacy_payload["curriculum_state"].pop(key)
            legacy_ppo = legacy_payload["config"].ppo
            object.__setattr__(legacy_ppo, "total_env_steps", 96)
            object.__delattr__(legacy_ppo, "schedule_env_steps")
            legacy_checkpoint = Path(directory) / "checkpoint_legacy.pkl"
            with legacy_checkpoint.open("wb") as file:
                pickle.dump(legacy_payload, file)
            with self.assertWarnsRegex(RuntimeWarning, "pre-correction Reward-V2"):
                migrated = PPOTrainer(
                    config,
                    course=self._compact_course(),
                    restore_checkpoint=legacy_checkpoint,
                )
            try:
                self.assertTrue(migrated._checkpoint_migrated)
                self.assertTrue(migrated._pre_correction_v2_checkpoint)
                self.assertEqual(migrated.state.schedule_steps, 48)
                self.assertEqual(migrated._schedule_progress(), 0.5)
                self.assertEqual(migrated.curriculum.state.skill_recent_count, 0)
                with self.assertRaisesRegex(RuntimeError, "cannot resume"):
                    migrated.train()
            finally:
                migrated.close()

            mismatched = self._compact_course()
            mismatched_normals = mismatched.normals.copy()
            mismatched_normals[0] *= -1.0
            mismatched = replace(mismatched, normals=mismatched_normals)
            mismatch_trainer = PPOTrainer(config, course=mismatched)
            try:
                with self.assertRaisesRegex(ValueError, "course geometry is incompatible"):
                    mismatch_trainer.load_checkpoint(checkpoint)
            finally:
                mismatch_trainer.close()

    @staticmethod
    def _compact_course():
        return course_by_name("compact_slalom")


if __name__ == "__main__":
    unittest.main()
