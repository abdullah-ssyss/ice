import os
import unittest
from unittest.mock import patch

from a2rl_drone_training.runtime import configure_runtime
from a2rl_drone_training.train import parse_args


class RuntimeTests(unittest.TestCase):
    def test_profile_preserves_explicit_overrides_in_either_order(self):
        for flags in (
            ["--num-envs", "128", "--profile", "rtx-5050"],
            ["--profile", "rtx-5050", "--num-envs", "128"],
        ):
            args = parse_args(flags)
            self.assertEqual(args.num_envs, 128)
            self.assertEqual(args.device, "gpu")
            self.assertEqual(args.gpu_memory_fraction, 0.60)
        args = parse_args([])
        self.assertEqual((args.device, args.num_envs, args.minibatches), ("cpu", 64, 8))

    def test_profile_preserves_minibatch_size(self):
        cpu = parse_args([])
        gpu = parse_args(["--profile", "rtx-5050"])
        self.assertEqual(cpu.num_envs * cpu.horizon // cpu.minibatches,
                         gpu.num_envs * gpu.horizon // gpu.minibatches)

    def test_gpu_requires_cuda_and_applies_memory_fraction(self):
        with patch("sys.platform", "linux"), patch.dict(os.environ, {}, clear=True):
            configure_runtime("gpu", 0.60)
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda")
            self.assertEqual(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.6")

    def test_cpu_overrides_inherited_cuda_selection(self):
        with patch.dict(os.environ, {"JAX_PLATFORMS": "cuda"}):
            configure_runtime("cpu")
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")

    def test_native_windows_gpu_error_explains_wsl(self):
        with patch("sys.platform", "win32"):
            with self.assertRaisesRegex(ValueError, "WSL2"):
                configure_runtime("gpu")

    def test_invalid_memory_fraction_rejected(self):
        for fraction in (0, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                configure_runtime("gpu", fraction)
