#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXAMINATION_STATE.json" >&2
  exit 2
fi

lab_root=$(realpath "$(dirname "$0")/..")
state_path=$(realpath "$1")
python_bin=${ROBUSTLAB_ISAACLAB_PYTHON:?ROBUSTLAB_ISAACLAB_PYTHON must be set}

cd "$lab_root"
export PER_REGION_MATERIALS="${PER_REGION_MATERIALS:-1}"

# A detached coordinator can be stopped while Isaac is still inside a batch.
# Complete that immutable batch manifest first; resume will import it exactly
# once from batch_status.json and continue with newly planned batches.
active_manifest=$(jq -r '.active_batch.manifest // empty' "$state_path")
if [[ -n "$active_manifest" ]]; then
  active_status=$(jq -r '.status // empty' "$(dirname "$active_manifest")/batch_status.json" 2>/dev/null || true)
  if [[ "$active_status" != "completed" ]]; then
    bash scripts/run_foresight_batch.sh "$active_manifest"
  fi
fi

export PYTHONPATH="$lab_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m robustness_lab.cli examine resume "$state_path"
