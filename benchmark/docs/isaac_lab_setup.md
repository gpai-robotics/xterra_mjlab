# Isaac Lab setup on this server

Isaac Lab 2.3.2 / Isaac Sim 5.1 are installed at `/opt/IsaacLab` with the launcher
at `/usr/local/bin/isaaclab`. Out of the box the simulation app **fails to start**:
Isaac Sim's bundled torch (2.7.0+cu128, python 3.11) cannot load because three CUDA
runtime libraries are absent from the system.

Symptom — `isaaclab -p <script>` aborts during `AppLauncher`, cascading through
`isaacsim.core.cloner`, `isaaclab_assets` and `isaaclab_tasks`:

    ImportError: libcusparseLt.so.0: cannot open shared object file
    ImportError: libnccl.so.2: cannot open shared object file
    ImportError: libtorch_cpu.so: undefined symbol: cuptiActivityEnableDriverApi,
                 version libcupti.so.12

Note `import isaaclab` on its own SUCCEEDS. Only launching the simulation app
fails, so an import check is not a sufficient readiness test.

## Fix

Install the three missing libraries into a user-owned directory and put exactly
those three on `LD_LIBRARY_PATH`:

    pip install --target=$HOME/.cache/robustlab-libs/pkg \
        nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-cuda-cupti-cu12

    B=$HOME/.cache/robustlab-libs/pkg/nvidia
    export LD_LIBRARY_PATH="$B/cusparselt/lib:$B/nccl/lib:$B/cuda_cupti/lib:$LD_LIBRARY_PATH"

**Add only those three.** Putting the full `nvidia-*-cu12` set on the path shadows
Isaac Sim's own CUDA libraries with newer builds and produces a different failure:

    libcusparse.so.12: undefined symbol: __nvJitLinkGetErrorLogSize_12_9

## Verified working

With the three libraries on the path, URDF -> USD conversion succeeds:

    ROBUSTLAB_ISAACLAB_ROOT=/opt/IsaacLab \
    robustlab robot convert-urdf <urdf> <output.usd>

produced `assets/cache/usd/go2.usd` (binary USDC, physics + robot prims,
referencing `configuration/go2_sensor.usd`).

## Other environment notes

- Running the test suite needs `MUJOCO_GL=osmesa`. Loading torch before MuJoCo's
  default GL backend aborts the interpreter during offscreen rendering.
- `/opt/isaacsim/.../DerivedDataCache` cannot be locked when another Isaac process
  is running; this is logged as an error but is not fatal.

## Known issue: episodes do not yet complete

As of this commit the Isaac Lab **adapter** starts but an episode terminates
before producing samples:

    status: error / simulator_error
    diagnostics: "simulator process terminated before worker finalization"
    samples: 0

What is fixed and verified:
- The missing CUDA libraries (above) — Isaac Sim starts, both GPUs active.
- Module shadowing: `runtime._runtime_command` used to pass the in-package
  `src/robustness_lab/backends/worker.py` to `isaaclab -p`. That puts the
  package directory on `sys.path[0]`, where our own `isaaclab.py` shadows the
  real Isaac Lab package, so `from isaaclab.app import AppLauncher` imported our
  adapter and failed with "attempted relative import with no known parent
  package". Now launched through a shim written outside the package.
- The worker always leaves a structured `episode_log.json` even on hard
  teardown, instead of vanishing silently.

What is NOT yet working: the simulation process exits ~3.4s after start, right
after the GPU banner, with no Python traceback. Still to investigate.

Contributing factors observed, neither confirmed fatal:
- `/opt/isaacsim/5.1.0/kit/cache/DerivedDataCache` is owned by another user and
  is not writable, so Kit logs "Failed to create local file data store". The
  URDF->USD conversion succeeded despite this, so it is probably not the cause.
- `/usr/local/bin/isaaclab -p` masks the child's exit status (its shell branch
  ends in a successful `break`), so a crashed worker reports return code 0.
  Never treat that launcher's exit code as evidence of success.

MuJoCo runs the full compiled-policy path end to end and is unaffected.

## Running the FoReSight native path on a clean machine

The `foresight_native` backend launches the policy's own Isaac Lab task. Beyond
the CUDA libraries above, five further blockers were hit on a machine other than
the original author's. None were recorded in the repository; all are now handled
by shims in `scripts/play_foresight_on_world.py` or by environment variables.

1. **`rsl_rl` version.** The source repo imports `EmpiricalNormalization` from
   `rsl_rl.modules`, which moved to `rsl_rl.networks` in rsl_rl 3.1.
   **Do not downgrade rsl_rl to 2.x to fix this.** The source repo's own
   `on_policy_runner_with_extractor.py` calls `PPO.init_storage(training_type,
   num_envs, num_steps, obs, [num_actions])` — the 5-argument, unified-`obs`
   signature introduced in 3.x. rsl_rl 2.3.1 requires 6 arguments and fails.
   Isaac Sim 5.1's bundled rsl_rl 3.1.2 is the correct version; the bridge
   re-exports the one moved symbol.

2. **`isaaclab.utils.pretrained_checkpoint`** moved to `isaaclab_rl.utils` in
   Isaac Lab 2.3. The bridge registers a LAZY proxy — resolving eagerly pulls in
   `isaaclab.utils.mesh`, which needs `pxr`, and USD bindings only exist once
   Isaac Sim's app has started.

3. **Isaac Lab's log directory** defaults to
   `os.path.join(tempfile.gettempdir(), "isaaclab", "logs")`. On a shared
   machine another user may own `/tmp/isaaclab`, giving PermissionError. Set
   `TMPDIR` to a directory you own.

4. **Do not set `CUDA_VISIBLE_DEVICES`.** Isaac warns that it does so, and with
   it set the GPUs are reported as "CUDA being in bad state" and skipped.

5. **`--summary_json` is required.** Without it the bridge raises
   `TypeError: expected str, bytes or os.PathLike object, not NoneType`.

### Verified working invocation

```bash
cd robustlab                  # repo root
R=$FORESIGHT_ROOT             # foresight_go2_parkour checkout
B=$HOME/.cache/robustlab-libs/pkg/nvidia
env TMPDIR=$HOME/.cache/robustlab-tmp \
    LD_LIBRARY_PATH="$B/cusparselt/lib:$B/nccl/lib:$B/cuda_cupti/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="$PWD/src" \
    /usr/local/bin/isaaclab -p scripts/play_foresight_on_world.py \
      --world <world.json> --bridge-manifest <bridge.json> \
      --foresight-root "$R" --benchmark-steps 300 \
      --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
      --checkpoint "$R/logs/rsl_rl/unitree_go2_parkour/2026-08-08_14-55-43_baseline_teacher_seed1_scaled/model_6300.pt" \
      --summary_json <out>/summary.json --seed 7 --num_envs 1 --headless
```

Observed result on RS4, difficulty 0.3, 300 steps: the policy loaded, the height
scan raycaster sampled continuously, and `summary.json` recorded
`num_goals: 8`, `max_goal_index: 4`, `success: false`,
`environment_physics_runtime.status: "verified"` with no missing effects.
`success: false` reflects the deliberately short 300-step budget, not a policy
verdict; the production config uses 1000.
