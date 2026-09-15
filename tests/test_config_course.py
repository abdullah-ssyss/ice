import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from a2rl_drone_training.config import (
    CurriculumConfig,
    ObservationConfig,
    PPOConfig,
    RacingEnvConfig,
    TrainingConfig,
)
from a2rl_drone_training.course import arena_38m_stacked_course, course_by_name, default_a2rl_course


class ConfigCourseTests(unittest.TestCase):
    def test_observation_dimension_tracks_gate_context(self):
        cfg = ObservationConfig(gate_context=4)
        self.assertEqual(cfg.core_dim, 24)
        self.assertEqual(cfg.dim, cfg.core_dim + 4 * cfg.gate_feature_dim)

    def test_default_course_has_unit_normals(self):
        course = default_a2rl_course()
        self.assertEqual(course.centers.shape[1], 3)
        self.assertEqual(course.normals.shape, course.centers.shape)
        norms = (course.normals**2).sum(axis=-1) ** 0.5
        self.assertTrue(((norms > 0.999) & (norms < 1.001)).all())

    def test_env_dt_matches_control_rate(self):
        cfg = RacingEnvConfig(control_hz=200)
        self.assertEqual(cfg.dt, 0.005)

    def test_default_episode_horizon_is_24_seconds(self):
        cfg = RacingEnvConfig()
        self.assertEqual(cfg.max_episode_time_s, 24.0)
        self.assertEqual(cfg.max_episode_steps, 12000)

    def test_reward_v2_and_long_credit_assignment_are_defaults(self):
        cfg = RacingEnvConfig()
        ppo = PPOConfig()
        self.assertEqual(cfg.reward_version, "v2")
        self.assertEqual(cfg.late_gate_reward_base, 1.0)
        self.assertEqual(cfg.late_gate_reward_cap, 4.0)
        self.assertEqual(cfg.time_penalty, 0.2)
        self.assertEqual(cfg.action_delta_penalty, 0.0002)
        self.assertEqual(cfg.potential_reward_weight, 1.25)
        self.assertEqual(cfg.potential_backtrack_limit, 0.5)
        self.assertEqual(cfg.gate_margin_penalty, 6.0)
        self.assertEqual(ppo.total_env_steps, 20_000_000)
        self.assertEqual(ppo.schedule_env_steps, 20_000_000)
        self.assertEqual(ppo.horizon, 256)
        self.assertEqual(ppo.gamma, 0.999)
        self.assertEqual(ppo.gae_lambda, 0.99)
        self.assertEqual(ppo.update_epochs, 4)
        self.assertEqual(ppo.entropy_coef, 0.003)
        self.assertEqual(ppo.entropy_coef_end, 0.0003)
        self.assertEqual(ppo.exploration_std_start, 0.60)
        self.assertEqual(ppo.exploration_std_end, 0.20)
        self.assertEqual(ppo.exploration_std_floor, 0.08)
        self.assertEqual(ppo.mean_action_alignment_coef, 0.05)

    def test_default_curriculum_reset_mixtures(self):
        cfg = CurriculumConfig()
        self.assertAlmostEqual(cfg.phase_a_gate1_fraction, 0.2)
        self.assertAlmostEqual(cfg.phase_b_gate1_fraction, 0.5)
        self.assertAlmostEqual(cfg.phase_c_gate1_fraction, 0.8)
        self.assertAlmostEqual(cfg.phase_d_gate1_fraction, 0.8)
        self.assertEqual(cfg.skill_qualification_window, 10)
        self.assertAlmostEqual(cfg.phase_a_priority_mix, 0.5)
        self.assertAlmostEqual(cfg.local_link_start_mix, 0.5)
        env = RacingEnvConfig()
        self.assertAlmostEqual(env.racing_line_spawn_min_distance_m, 1.0)
        self.assertAlmostEqual(env.racing_line_spawn_max_distance_m, 3.0)
        self.assertAlmostEqual(env.racing_line_spawn_speed_m_s, 2.0)

    def test_arena_38m_stacked_course_metadata(self):
        course = arena_38m_stacked_course()
        self.assertEqual(course.name, "arena_38m_stacked")
        self.assertEqual(course.num_gates, 12)
        self.assertEqual(course.logical_gate_ids.tolist().count(7), 2)
        self.assertEqual(course.logical_gate_ids.tolist().count(10), 2)
        self.assertEqual(course.openings[6], "top")
        self.assertEqual(course.openings[7], "bottom")
        self.assertEqual(course.openings[10], "bottom")
        self.assertEqual(course.openings[11], "top")
        self.assertEqual(course.bounds_min[:2].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(course.bounds_min[2]), 0.05)
        self.assertEqual(course.bounds_max.tolist(), [38.0, 38.0, 5.0])
        self.assertEqual(course.arena_size, (38.0, 38.0))
        self.assertAlmostEqual(float(course.widths[0]), 1.5)
        self.assertAlmostEqual(float(course.heights[0]), 1.5)
        self.assertAlmostEqual(float(course.outer_widths[0]), 2.7)
        self.assertAlmostEqual(float(course.outer_heights[0]), 2.7)
        self.assertEqual(course.nominal_racing_line.shape, (14, 3))
        np.testing.assert_allclose(course.normals[10], -course.normals[11], atol=1.0e-6)
        just_after_g11 = course.centers[10] + 0.01 * course.normals[10]
        g12_plane_distance = np.dot(
            just_after_g11 - course.centers[11],
            course.normals[11],
        )
        self.assertLess(g12_plane_distance, 0.0)

    def test_course_by_name(self):
        self.assertEqual(course_by_name("arena_38m_stacked").num_gates, 12)
        self.assertEqual(course_by_name("compact_slalom").name, "compact_slalom")

    def test_train_parser_exposes_cpu_and_observation_noise_args(self):
        from a2rl_drone_training.train import build_parser

        args = build_parser().parse_args(
            [
                "--cpu-threads",
                "8",
                "--obs-additive-noise-std",
                "0.0",
                "--pnp-dropout-prob",
                "0.0",
            ]
        )
        self.assertEqual(args.cpu_threads, "8")
        self.assertEqual(args.sensor_noise_scale, 0.0)
        self.assertEqual(args.pnp_dropout_prob, 0.0)

    def test_time_penalty_cli_reaches_reward_config(self):
        from a2rl_drone_training.train import _build_training_config, build_parser

        args = build_parser().parse_args(["--time-penalty", "0.37", "--reward-version", "v2"])
        config = _build_training_config(args)
        self.assertEqual(config.env.time_penalty, 0.37)
        self.assertEqual(config.env.reward_version, "v2")

    def test_schedule_skill_audit_and_persistence_cli_reach_config(self):
        from a2rl_drone_training.train import _build_training_config, build_parser

        args = build_parser().parse_args(
            [
                "--schedule-env-steps",
                "12000000",
                "--skill-evaluation-attempts-per-gate",
                "4",
                "--skill-evaluation-max-time",
                "2.5",
                "--checkpoint-interval",
                "7",
                "--checkpoint-dir",
                "saved",
                "--skill-qualification-window",
                "6",
                "--phase-a-priority-mix",
                "0.35",
                "--local-link-start-mix",
                "0.4",
                "--exploration-std-end",
                "0.18",
                "--mean-action-alignment-coef",
                "0.07",
                "--potential-backtrack-limit",
                "0.65",
            ]
        )
        config = _build_training_config(args)
        self.assertEqual(config.ppo.schedule_env_steps, 12_000_000)
        self.assertEqual(config.evaluation.skill_attempts_per_gate, 4)
        self.assertEqual(config.evaluation.skill_max_time_s, 2.5)
        self.assertEqual(config.checkpoint_interval, 7)
        self.assertEqual(config.curriculum.skill_qualification_window, 6)
        self.assertEqual(config.curriculum.phase_a_priority_mix, 0.35)
        self.assertEqual(config.curriculum.local_link_start_mix, 0.4)
        self.assertEqual(config.ppo.exploration_std_end, 0.18)
        self.assertEqual(config.ppo.mean_action_alignment_coef, 0.07)
        self.assertEqual(config.env.potential_backtrack_limit, 0.65)
        self.assertEqual(config.checkpoint_dir, Path("saved"))
        self.assertEqual(config.metrics_file, Path("saved/metrics.jsonl"))

        disabled = _build_training_config(build_parser().parse_args(["--no-checkpoint"]))
        self.assertIsNone(disabled.checkpoint_dir)
        self.assertIsNone(disabled.metrics_file)

    def test_cpu_thread_config_sets_unset_env_vars(self):
        from a2rl_drone_training.train import _configure_cpu_threads

        with patch.dict(os.environ, {}, clear=True):
            threads = _configure_cpu_threads("8", "cpu")
            self.assertEqual(threads, 8)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "8")
            self.assertEqual(
                os.environ["XLA_FLAGS"],
                "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8",
            )

    def test_cpu_thread_config_respects_existing_env_vars(self):
        from a2rl_drone_training.train import _configure_cpu_threads

        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "3",
                "XLA_FLAGS": "--already-set=true",
            },
            clear=True,
        ):
            threads = _configure_cpu_threads("8", "cpu")
            self.assertEqual(threads, 8)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
            self.assertEqual(os.environ["XLA_FLAGS"], "--already-set=true")

    def test_gate_diagnostic_line_marks_weak_gate(self):
        try:
            from a2rl_drone_training.trainer import _gate_diagnostic_line
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("trainer diagnostics require jax")
            raise

        line = _gate_diagnostic_line(
            {
                "gate_target_counts": np.array([100, 20, 15], dtype=np.float32),
                "gate_pass_counts": np.array([4, 1, 0], dtype=np.float32),
                "gate_miss_counts": np.array([0, 3, 1], dtype=np.float32),
                "gate_stop_counts": np.array([0, 3, 1], dtype=np.float32),
                "gate_stall_counts": np.array([0, 5, 0], dtype=np.float32),
            }
        )
        self.assertIn("weak=G2:miss", line)
        self.assertIn("target=1:100,2:20,3:15", line)
        self.assertIn("pass=1:4,2:1,3:0", line)
        self.assertIn("fail=1:0,2:3,3:1", line)

    def test_training_eta_duration_format(self):
        try:
            from a2rl_drone_training.trainer import _format_duration
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("trainer diagnostics require jax")
            raise

        self.assertEqual(_format_duration(4.4), "4s")
        self.assertEqual(_format_duration(65.0), "1m05s")
        self.assertEqual(_format_duration(3661.0), "1h01m")

    def test_observation_builder_is_time_independent(self):
        try:
            import jax.numpy as jnp

            from a2rl_drone_training.observations import build_racing_observation
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("observation builder requires jax")
            raise

        cfg = ObservationConfig(sensor_noise_scale=0.0, pnp_dropout_prob=0.0)
        course = arena_38m_stacked_course()
        obs = build_racing_observation(
            pos_world=jnp.zeros((2, 3), dtype=jnp.float32),
            quat_xyzw=jnp.array(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=jnp.float32,
            ),
            vel_world=jnp.zeros((2, 3), dtype=jnp.float32),
            ang_vel_world=jnp.zeros((2, 3), dtype=jnp.float32),
            acc_world=jnp.zeros((2, 3), dtype=jnp.float32),
            gate_centers_world=jnp.asarray(course.centers, dtype=jnp.float32),
            gate_normals_world=jnp.asarray(course.normals, dtype=jnp.float32),
            gate_counter=jnp.array([0, 5], dtype=jnp.int32),
            total_gate_passes=course.num_gates,
            last_action=jnp.zeros((2, 4), dtype=jnp.float32),
            remaining_time_fraction=jnp.ones((2,), dtype=jnp.float32),
            yaw_error=jnp.zeros((2,), dtype=jnp.float32),
            stall_fraction=jnp.zeros((2,), dtype=jnp.float32),
            config=cfg,
        )
        self.assertEqual(obs.shape, (2, cfg.dim))

    def test_checkpoint_observation_dimension_guard(self):
        try:
            from a2rl_drone_training.trainer import (
                _validate_checkpoint_observation_compatibility,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("checkpoint compatibility guard requires jax")
            raise

        payload = {"config": TrainingConfig(obs=ObservationConfig(core_dim=22))}
        with self.assertRaisesRegex(ValueError, "checkpoint_obs_dim=52"):
            _validate_checkpoint_observation_compatibility(
                Path("old_checkpoint.pkl"),
                payload,
                TrainingConfig(),
            )

    def test_balanced_reset_sampler_targets_active_ratio(self):
        try:
            import jax
            import jax.numpy as jnp

            from a2rl_drone_training.curriculum import sample_reset_gates
        except ModuleNotFoundError as exc:
            if exc.name == "jax":
                self.skipTest("balanced reset sampler requires jax")
            raise

        key = jax.random.key(0)
        full_reset = jnp.ones((10,), dtype=jnp.bool_)
        current_gate = jnp.zeros((10,), dtype=jnp.int32)
        gates, selected = sample_reset_gates(
            key=key,
            reset_mask=full_reset,
            current_reset_gate=current_gate,
            gate1_fraction=0.6,
            local_gate_probabilities=jnp.ones((3,), dtype=jnp.float32) / 3.0,
            prioritized=False,
            evaluation=False,
        )
        self.assertEqual(int(jnp.sum(selected)), 4)
        self.assertEqual(int(jnp.sum(gates == 0)), 6)

        partial_reset = jnp.array(
            [False, False, False, False, False, False, True, True, True, True],
            dtype=jnp.bool_,
        )
        current = jnp.array(
            [2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=jnp.int32,
        )
        _, selected = sample_reset_gates(
            key=key,
            reset_mask=partial_reset,
            current_reset_gate=current,
            gate1_fraction=0.6,
            local_gate_probabilities=jnp.ones((3,), dtype=jnp.float32) / 3.0,
            prioritized=False,
            evaluation=False,
        )
        self.assertEqual(int(jnp.sum(selected)), 2)


if __name__ == "__main__":
    unittest.main()
