#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 BATCH_MANIFEST.json" >&2
  exit 2
fi

batch_path=$(realpath "$1")
batch_steps=$(jq -r '.items[0].steps // 1000' "$batch_path")
gpu_index=${GPU_INDEX:-1}
lab_root=$(realpath "$(dirname "$0")/..")
foresight_root=${FORESIGHT_ROOT:?FORESIGHT_ROOT must be set}
batch_name=$(basename "$(dirname "$batch_path")")
scratch_python=${ROBUSTLAB_ISAACLAB_PYTHON:?ROBUSTLAB_ISAACLAB_PYTHON must be set}
isaacsim_root=${ROBUSTLAB_ISAACSIM_ROOT:?ROBUSTLAB_ISAACSIM_ROOT must be set}
scratch_root=${ROBUSTLAB_ISAAC_SCRATCH_ROOT:-/tmp}
keep_scratch=${KEEP_ISAAC_TEMP:-0}
runner_pid=

export PYTHONPATH="$foresight_root/parkour_tasks:$foresight_root:$lab_root/src:${PYTHONPATH:-}"
"$scratch_python" -m robustness_lab.scratch cleanup --root "$scratch_root" --stale-after 0 >/dev/null
scratch_args=(create --root "$scratch_root" --label "batch-$batch_name" --owner-pid "$$")
if [[ "$keep_scratch" == "1" ]]; then
  scratch_args+=(--preserve)
fi
isaac_scratch=$("$scratch_python" -m robustness_lab.scratch "${scratch_args[@]}")

cleanup_scratch() {
  "$scratch_python" -m robustness_lab.scratch release "$isaac_scratch" --root "$scratch_root" >/dev/null 2>&1 || true
}

terminate_runner() {
  local exit_code=$1
  trap - INT TERM
  if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
    kill -TERM "$runner_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
  fi
  exit "$exit_code"
}

trap cleanup_scratch EXIT
trap 'terminate_runner 130' INT
trap 'terminate_runner 143' TERM

cd /tmp
set +u
source "$isaacsim_root/setup_conda_env.sh"
set -u
export PYTHONPATH="$foresight_root/parkour_tasks:$foresight_root:$lab_root/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export ISAACSIM_PORTABLE_ROOT="$isaac_scratch"
cd "$lab_root"

"$scratch_python" scripts/play_foresight_on_world.py \
  --batch-manifest "$batch_path" \
  --foresight-root "$foresight_root" \
  --benchmark-steps "$batch_steps" \
  --headless --enable_cameras --device cuda:0 --num_envs 1 --seed 101 --hide_debug_markers &
runner_pid=$!
wait "$runner_pid"
