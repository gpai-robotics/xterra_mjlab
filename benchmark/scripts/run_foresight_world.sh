#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 WORLD.json CHECKPOINT.pt OUTPUT_DIR [TASK]" >&2
  exit 2
fi

world_path=$(realpath "$1")
checkpoint_path=$(realpath "$2")
output_path=$(realpath -m "$3")
task_name=${4:-Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0}
gpu_index=${GPU_INDEX:-1}
video_length=${VIDEO_LENGTH:-1000}
record_video=${RECORD_VIDEO:-1}
run_seed=${RUN_SEED:-101}
lab_root=$(realpath "$(dirname "$0")/..")
foresight_root=${FORESIGHT_ROOT:?FORESIGHT_ROOT must be set}
scratch_python=${ROBUSTLAB_ISAACLAB_PYTHON:?ROBUSTLAB_ISAACLAB_PYTHON must be set}
isaacsim_root=${ROBUSTLAB_ISAACSIM_ROOT:?ROBUSTLAB_ISAACSIM_ROOT must be set}
scratch_root=${ROBUSTLAB_ISAAC_SCRATCH_ROOT:-/tmp}
keep_scratch=${KEEP_ISAAC_TEMP:-0}
runner_pid=

mkdir -p "$output_path/video"
export PYTHONPATH="$foresight_root/parkour_tasks:$foresight_root:$lab_root/src:${PYTHONPATH:-}"
"$scratch_python" -m robustness_lab.scratch cleanup --root "$scratch_root" --stale-after 0 >/dev/null
scratch_args=(create --root "$scratch_root" --label "world-$(basename "$output_path")" --owner-pid "$$")
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

runner_args=(
  --world "$world_path"
  --foresight-root "$foresight_root"
  --bridge-manifest "$output_path/native_bridge_manifest.json"
  --benchmark-steps "$video_length"
  --task "$task_name"
  --checkpoint "$checkpoint_path"
  --summary_json "$output_path/summary.json"
  --headless --device cuda:0 --num_envs 1 --seed "$run_seed" --hide_debug_markers
)
if [[ "$record_video" == "1" ]]; then
  runner_args+=(--video --video_length "$video_length" --video_dir "$output_path/video")
fi

"$scratch_python" scripts/play_foresight_on_world.py "${runner_args[@]}" &
runner_pid=$!
wait "$runner_pid"
