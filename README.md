# A2RL Racing Drone Training

This package trains an autonomous FPV racing policy with Crazyflow simulation and a plain-JAX PPO implementation. The default setup targets CPU training and the 38 m A2RL arena course.

## System Shape

- Environment: vectorized Crazyflow `Sim` in direct rotor-velocity control mode.
- Policy action: normalized `[motor_1, motor_2, motor_3, motor_4]` in `[-1, 1]`.
- Crazyflow command: `[rpm_1, rpm_2, rpm_3, rpm_4]` in the simulator model's native motor order.
- Actor: gate-conditioned deployment policy using noisy estimator and gate-PnP-compatible features.
- Critic: asymmetric value network using exact, noise-free simulator state during training only.
- Trainer: 256-step rollouts, truncation-correct GAE, clipped PPO, scheduled learning rates and entropy, and KL early stopping.

Each action independently commands one motor: `-1` maps to its minimum RPM,
`0` to hover RPM, and `+1` to maximum RPM, with linear interpolation on either
side of hover. RPM bounds come from the configured model's per-motor thrust limits
and quadratic thrust curve. This bypasses attitude and force/torque controllers;
first-principles physics models the resulting forces, torques, and rotor dynamics.
Resets initialize rotor speeds to hover. The former yaw-error observation channel
is reserved and always zero; actor/critic dimensions remain 54/37.

Direct motor control requires `--physics first_principles`. The default simulation
and action rates are both 500 Hz. A 256-step rollout now covers 0.512 seconds;
discounting and training schedules are still expressed in steps. The action-change
penalty defaults to 0.0002 to preserve its nominal per-second budget at the higher
action rate (previously 0.001 at 100 Hz). Existing attitude-control checkpoints are
incompatible even though both interfaces have four outputs; start fresh in a new
checkpoint directory, such as `--checkpoint-dir checkpoints_motors`.

WSL Ubuntu, with evaluation enabled for curriculum progression:

```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050 \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500 \
  --checkpoint-dir checkpoints_motors
```


## Install

Run commands from the repository root. Use the examples for your shell: Bash on
Linux, or Command Prompt (`cmd.exe`) on Windows (including a Command Prompt tab in Windows Terminal).
Command Prompt uses a caret (`^`) for line continuation; it must be the last
character on the line, with no trailing spaces.

Linux (Bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows (Command Prompt):

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The Windows examples use the virtual environment's Python directly, so activation
is unnecessary. The setup command assumes Python 3.11 is installed and available
through the `py` launcher.

The optional Warp backend is not required for the configured JAX/Crazyflow CPU path.

## GPU Setup And RTX 5050 Laptop Preset

JAX CUDA runs on Linux; Windows users need WSL2 (listed as experimental by
[JAX](https://docs.jax.dev/en/latest/installation.html)). The native Windows
commands elsewhere in this README are for CPU training. Keep the Windows NVIDIA
driver installed; WSL uses that driver, as described in the
[NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

If WSL is not installed, run this once in an administrator Command Prompt, then
restart if prompted and complete Ubuntu's first-launch setup:

```bat
wsl --install -d Ubuntu
```

From Command Prompt in the repository root, open Ubuntu at the same location:

```bat
wsl -d Ubuntu
```

Inside Ubuntu, create a separate Linux environment (do not reuse the Windows
`.venv`) and install the project together with CUDA-enabled JAX:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install --upgrade pip
.venv-wsl/bin/python -m pip install -e . "jax[cuda13]"
.venv-wsl/bin/python -m pip check
.venv-wsl/bin/python -c "import jax; print(jax.devices('gpu'))"
exit
```

The final check must list a GPU. CUDA 13 is the current JAX installation path;
see the linked JAX guide for driver requirements. Dependency resolution and GPU
execution must succeed before starting a long run.

Back in Command Prompt, launch the acceptance stage through WSL:

```bat
wsl -d Ubuntu -- .venv-wsl/bin/python -m a2rl_drone_training.train ^
  --profile rtx-5050 ^
  --total-env-steps 2000000 ^
  --schedule-env-steps 20000000 ^
  --course arena_38m_stacked
```

The preset is a starting point for the 8 GB laptop GPU, not a measured optimum:

| Setting | Value | Purpose |
| --- | --- | --- |
| Device | GPU | Require CUDA for simulation and model allocations |
| Parallel environments | 256 | Amortize Python dispatch across more simulated drones |
| Rollout horizon | 256 | Retain the existing temporal rollout length |
| Minibatches | 32 | Keep 2,048 samples per minibatch, matching CPU defaults |
| GPU memory preallocation | 60% | Leave room for the laptop display and other applications |

Explicit flags override the preset, for example `--num-envs 128 --minibatches 16`.
`--gpu-memory-fraction` controls JAX's allocation pool, not a hard limit on total
process GPU memory; see [JAX memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
Physics uses the first-principles motor model at 500 Hz, and PPO retains four epochs and float32
networks. The larger rollout batch changes update frequency per environment step;
evaluate learning quality as well as throughput. Evaluation and checkpoint intervals
are still measured in updates.

Compare 64, 128, 256, and 512 environments on the actual laptop:

```bat
wsl -d Ubuntu -- .venv-wsl/bin/python scripts/benchmark_cpu_training.py ^
  --device gpu ^
  --warmup-updates 2 ^
  --updates 5
```

Despite its historical filename, the benchmark supports both CPU and GPU. It
excludes warmup updates, synchronizes optimizer completion, and disables evaluation
and checkpoints to compare training throughput at the same physics fidelity.
GPU cases use a fixed environment-count sweep; `--num-envs` configures CPU cases.
Use the fastest count that fits in memory, then verify the acceptance metrics with
evaluation enabled. The environment still has a Python rollout loop and synchronizes
on episode resets, so higher GPU utilization and a speedup are not guaranteed.

## Recommended Staged Training

Start with a two-million-step acceptance stage while keeping every PPO schedule on
the full twenty-million-step clock:

Linux (Bash):

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --num-envs 64 \
  --horizon 256 \
  --minibatches 8 \
  --update-epochs 4 \
  --gamma 0.999 \
  --gae-lambda 0.99 \
  --total-env-steps 2000000 \
  --schedule-env-steps 20000000 \
  --course arena_38m_stacked \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500
```
```bash
./.venv/bin/python3.13 -m a2rl_drone_training.train \
  --profile rtx-5050 \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500 \
  --no-evaluation
```

Windows (Command Prompt):

```bat
python -m a2rl_drone_training.train ^
  --device cpu ^
  --cpu-threads 8 ^
  --num-envs 64 ^
  --horizon 256 ^
  --minibatches 8 ^
  --update-epochs 4 ^
  --gamma 0.999 ^
  --gae-lambda 0.99 ^
  --total-env-steps 2000000 ^
  --schedule-env-steps 20000000 ^
  --course arena_38m_stacked ^
  --physics first_principles ^
  --sim-hz 500 ^
  --control-hz 500
```

After reviewing the acceptance metrics, resume the same schedule:

Linux (Bash):

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --total-env-steps 20000000 \
  --schedule-env-steps 20000000 \
  --restore-checkpoint checkpoints/checkpoint_latest.pkl
```

Windows (Command Prompt):

```bat
.\.venv\Scripts\python.exe -m a2rl_drone_training.train ^
  --device cpu ^
  --cpu-threads 8 ^
  --total-env-steps 20000000 ^
  --schedule-env-steps 20000000 ^
  --restore-checkpoint checkpoints/checkpoint_latest.pkl
```

Continue only when PPO values remain finite, at least 1,000 reset events are within
`80% +/- 3%` local starts, linked G12 crossings are present, no recent active-window
gate is below 50%, and the minimum recent rate is moving toward 80%. Also verify
that sampled-action saturation and the sampled/mean action gap trend down with the
scheduled exploration ceiling.

For CPU training without evaluation (benchmarking only):

Linux (Bash):

```bash
a2rl-drone-train \
  --device cpu \
  --cpu-threads 8 \
  --num-envs 64 \
  --horizon 256 \
  --minibatches 8 \
  --update-epochs 4 \
  --physics first_principles \
  --sim-hz 500 \
  --control-hz 500 \
  --no-evaluation
```

Windows (Command Prompt):

```bat
.\.venv\Scripts\python.exe -m a2rl_drone_training.train ^
  --device cpu ^
  --cpu-threads 8 ^
  --num-envs 64 ^
  --horizon 256 ^
  --minibatches 8 ^
  --update-epochs 4 ^
  --physics first_principles ^
  --sim-hz 500 ^
  --control-hz 500 ^
  --no-evaluation
```

```bat
python -m a2rl_drone_training.train ^
  --device gpu ^
  --profile rtx-5050 ^
  --cpu-threads 8 ^
  --num-envs 64 ^
  --horizon 256 ^
  --minibatches 8 ^
  --update-epochs 4 ^
  --physics first_principles ^
  --sim-hz 500 ^
  --control-hz 500 ^
  --no-evaluation
```

Training logs use an aligned SB3-style table. The `time/` section reports current and run-average FPS, iteration time, elapsed time, and ETA; the remaining sections retain every reward component, PPO diagnostic, curriculum state, and per-gate metric.

Tables automatically fit the terminal width, including after resizing the window.
Long labels and values are shortened with `...` to keep columns aligned without
line wrapping. Full diagnostic values remain available in `metrics.jsonl`.

Checkpoints default to `checkpoints/`, including atomic numbered saves and `checkpoint_latest.pkl`. Structured update records are appended to `checkpoints/metrics.jsonl`. Use `--no-checkpoint` for disposable benchmark runs.

## Reward V2

`reward_version=v2` is the default. It combines:

- Potential-based progress along a continuous course coordinate, scaled to about `+1.25` shaping return per gate.
- Gate-centering pressure localized near the active gate plane.
- A curriculum-scaled physical time cost, reaching `-0.2 reward/second` in racing phase D.
- Squared action-change cost with a default coefficient of `0.0002`, rather than raw throttle or action magnitude cost.
- `+6` gate pass and `+25` true full-course finish rewards.
- `-8` missed-gate and competition-deadline penalties.
- `-12` crash or out-of-bounds penalties.
- A safety-margin penalty only when crossing clearance is below `vehicle_radius + k * position_uncertainty`.

The gate-margin penalty is capped at `6` by default, so a crossing on or outside a
physical inner edge can cancel the `+6` relaxed-window gate reward. Curriculum-scaled
dimensions decide whether Phase-A training advances; physical 1.0x dimensions decide
strict-pass telemetry, clearance, and margin reward.

There is no raw speed reward, exact-center bonus, overlapping distance/lookahead shaping, or stall reward counter in v2. Use `--reward-version v1` to reproduce the legacy reward formula for ablations.

The existing `--time-penalty` option now controls the full phase-D v2 cost in reward per physical second. Phase A starts at zero, phases B and C use configurable fractions, and phase D ramps to the configured value.

The progress potential is rebased by the episode's reset gate, so local starts retain
continuous gate-to-gate shaping without a larger negative offset at later gates.
Forward motion is capped at one segment per transition, while up to half a segment of
backtracking remains visible to the reward instead of being flattened at the segment
start. A local episode that reaches gate 12 terminates successfully but receives no
`+25` finish bonus; that bonus and course-completion credit are reserved for episodes
that began at gate 1. Reward V1 retains its legacy behavior.

## Curriculum

Curriculum advancement is driven by deterministic evaluation, not environment steps
or a selected best rollout. Reset mixtures are sampled from the environments that
reset on each event, with stochastic rounding for small reset batches.

| Phase | Gate-1 starts | Local starts | Gate window | Time cost |
| --- | ---: | ---: | ---: | ---: |
| A: gate skill | 20% | 80% stratified | 1.8x | 0% |
| B: course linking | 50% | 50% stratified | toward 1.3x | 25% |
| C: reliable course | 80% | 20% prioritized | toward 1.0x | 50% |
| D: racing | 80% | 20% prioritized | 1.0x | ramps to 100% |

Phase A qualification uses a separate deterministic, noise-free local skill audit with
eight fixed-seed attempts per runtime gate. The audit uses the active Phase-A opening,
while recording physical strict passes from the same crossings. Qualification uses
the last ten audits, requires at least 32 samples per gate, an approximately 80%
minimum active-window pass rate, and two consecutive qualifying audits. Lifetime
rates remain diagnostics only, so old failures cannot permanently lock the phase.
After coverage, Phase-A direct local starts mix 50% uniform coverage with 50% recent
audit-failure priority. Half of local starts are additionally allocated to predecessor
gates for weak strict-evaluation gates, training the transition into the failure rather
than repeatedly spawning at it. This direct/link blend is configurable. Phase B
additionally requires roughly 50% official full-course
completion and 85% minimum strict gate pass rate. Phase C requires roughly 80%
completion and 90% minimum strict pass rate. Hysteresis and rate-limited
gate-window/time-cost changes prevent rapid transitions.

Prioritized local starts use per-gate failure rates with configurable exponent, epsilon, and a minimum probability floor. Stacked top/bottom openings are separate runtime gates and therefore retain independent statistics and sampling probabilities.

Use `--no-curriculum` for strict gate-1 starts, a 1.0x gate opening, and the full configured time cost.

## Observations

The actor receives only deployment-compatible features:

- Body gyro and specific force.
- Attitude quaternion, body velocity, and body gravity.
- Previous action, course progress, remaining competition time, a reserved zero channel, and legacy-v1 stall state.
- Relative gate pose, normal, image-plane bearing, visibility, and distance for the next `N` gates.

Noise is applied in physical units per sensor/feature before normalization. `--sensor-noise-scale` scales all configured noise models; the legacy `--obs-additive-noise-std` spelling is retained as an alias for this scale. PnP dropout masks all PnP-derived geometry. Running normalization is used only for unbounded channels; bounded quaternions, normals, visibility, progress, and controller channels use fixed transforms.

The critic separately receives exact pose, attitude, velocity, angular velocity, acceleration, active and next gate geometry, elapsed time, gate progress, previous motor action, a reserved zero channel, gate-window scale, and reset-start type. Those features never enter the actor network or actor loss.

Normalization statistics are updated during training, frozen during evaluation, and stored in checkpoints.

## Episode Semantics

True terminals are crashes, out-of-bounds failures, missed gates, local segment completion, true course completion, and the real competition deadline set by `--max-episode-time`. Local segment completion is a successful terminal but not a full-course finish. `--artificial-time-limit` is an optional simulator truncation.

GAE bootstraps through truncations from the final pre-reset privileged observation. It does not bootstrap through true terminals, and it never uses an automatically reset initial observation as the final state. The PPO rollout boundary by itself is neither a terminal nor a truncation.

## PPO Defaults

- 64 environments, 256-step rollouts, batch size 16,384.
- 4 epochs and 8 minibatches.
- `gamma=0.999`, `gae_lambda=0.99`.
- Policy and value clipping `0.2`.
- Actor and critic learning rates linearly decay from `3e-4` to `3e-5` over the independent `--schedule-env-steps` budget, default 20 million steps.
- Entropy decays from `0.003` to `0.0003` over 75% of that schedule budget.
- Target KL `0.015`; remaining epochs stop after an over-target epoch.
- Global gradient clipping `0.5`.
- Independent trainable exploration standard deviation for each action, constrained by
  a ceiling that cools from `0.60` to `0.20` over 75% of the schedule and a `0.08`
  floor.
- A small positive-advantage mean-action alignment loss keeps deterministic evaluation
  behavior close to successful sampled behavior.

The CLI exposes rollout length, gamma, lambda, all schedule endpoints, KL target,
clipping, exploration bounds, curriculum thresholds, evaluation cadence, and reward-v2
coefficients for ablations. `--skill-qualification-window` controls the rolling audit
horizon, `--phase-a-priority-mix` controls the uniform/failure-priority blend, and
`--local-link-start-mix` controls predecessor-link replay.

## Evaluation

Strict evaluation always:

- Starts every environment at gate 1.
- Uses the configured fixed evaluation seed.
- Uses a strict 1.0x opening.
- Disables sensor noise and PnP dropout.
- Freezes actor normalization.
- Uses deterministic policy actions.

The Phase-A skill audit is separate from official evaluation. It uses forced local gate
starts at the active Phase-A opening solely for curriculum qualification and never
contributes to gate-1 course-completion statistics. Official evaluation remains
gate-1-only and strict 1.0x.

The chase-video script visualizes a fixed environment index, default `0`; it never chooses the best parallel environment.

Linux (Bash):

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_chase_video.py \
  --checkpoint-dir checkpoints \
  --course arena_38m_stacked \
  --output artifacts/eval_chase_arena_38m_stacked.mp4 \
  --num-eval-envs 32 \
  --visualization-env 0 \
  --seed 123 \
  --device cpu
```

Windows (Command Prompt):

```bat
set "PYTHONPATH=src"
.\.venv\Scripts\python.exe scripts/eval_chase_video.py ^
  --checkpoint-dir checkpoints ^
  --course arena_38m_stacked ^
  --output artifacts/eval_chase_arena_38m_stacked.mp4 ^
  --num-eval-envs 32 ^
  --visualization-env 0 ^
  --seed 123 ^
  --device cpu
```

## Checkpoints

Schema-v6 checkpoints include actor and critic parameters, optimizer state, actor
normalization statistics, curriculum phase, official and skill-audit counters, rolling
audit history, latest strict-evaluation gate results and priority state, consumed PPO
schedule steps, evaluation counters,
total environment steps, update count, episode count, RNG state, and a signed course
geometry fingerprint.

Schema-v6 also records the action-space identifier
`motor_rpm_hover_centered_v1`. Training and video evaluation reject checkpoints
without that identifier, including all older attitude-control checkpoints. Motor
checkpoints restore through `--restore-checkpoint`; the total-step target and the
schedule-step budget remain independent. Course-fingerprint checks still apply.

## Tests And Benchmark

Linux (Bash):

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/benchmark_cpu_training.py --updates 3
```

Windows (Command Prompt):

```bat
.\.venv\Scripts\python.exe -m pip install pytest
set "PYTHONPATH=src"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/benchmark_cpu_training.py --updates 3
```

In Command Prompt, `set "PYTHONPATH=src"` applies to subsequent commands in the current
terminal session.
