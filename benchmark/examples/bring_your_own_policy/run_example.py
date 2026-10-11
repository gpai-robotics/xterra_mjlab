"""End-to-end bring-your-own-policy example.

Runs the exact flow an external submitter follows: a TorchScript policy plus a
scaffolded manifest, evaluated on three terrains, producing metrics and an HTML
report. Deliberately BOUNDED (3 terrains, 6s each) -- it demonstrates the
pipeline; it is not the full RS1-RS11 benchmark.

    MUJOCO_GL=osmesa python examples/bring_your_own_policy/run_example.py
"""

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from robustness_lab.aggregate import aggregate_runs                # noqa: E402
from robustness_lab.io import dump_json, load_data                 # noqa: E402
from robustness_lab.models import RobotProfile                     # noqa: E402
from robustness_lab.report import generate_aggregate_report        # noqa: E402
from robustness_lab.robot import inspect_mjcf                      # noqa: E402
from robustness_lab.scaffold import scaffold_policy_manifest       # noqa: E402
from robustness_lab.runtime import execute_episode                 # noqa: E402
from robustness_lab.worlds import generate_terrain                 # noqa: E402

TERRAINS = ["rs1", "rs3", "rs8"]
ROBOT = os.path.join(REPO, "assets/cache/mujoco_menagerie/unitree_go2/go2.xml")


class ExamplePolicy(nn.Module):
    """Stands in for a submitted policy: 45 observations -> 12 joint actions."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(45, 128), nn.ELU(),
                                 nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 12))

    def forward(self, x):
        return torch.tanh(self.net(x))


def main():
    work = tempfile.mkdtemp(prefix="byop_example_")

    # 1. The submitter's artifact.
    artifact = os.path.join(work, "policy.pt")
    torch.jit.script(ExamplePolicy()).save(artifact)

    # 2. The robot profile the benchmark scales its worlds to.
    profile = inspect_mjcf(ROBOT)
    profile_path = os.path.join(work, "robot.json")
    dump_json(profile_path, profile)

    # 3. The manifest -- drafted from the weights, not written by hand.
    manifest = os.path.join(work, "policy.yaml")
    drafted = scaffold_policy_manifest(artifact, joint_order=profile.controller_joint_order,
                                       output=manifest, robot_profile=profile)
    print("probed %d -> %d" % (drafted["input_size"], drafted["output_size"]))
    for item in drafted["needs_review"]:
        print("  confirm: %s" % item)

    # 4. Evaluate on a few terrains.
    runs = []
    for terrain in TERRAINS:
        world = generate_terrain(terrain, profile, 0.3, 7)
        world_path = os.path.join(work, "world_%s.json" % terrain)
        dump_json(world_path, world)
        episode = {
            "episode_id": "byop-%s" % terrain,
            "backend": "mujoco",
            "robot_profile": profile_path,
            "robot_asset": ROBOT,
            "robot_format": "mjcf",
            "world": world_path,
            "controller": {"type": "compiled_policy", "manifest": manifest,
                           "command": [0.6, 0.0, 0.0]},
            "output_dir": os.path.join(work, "run_%s" % terrain),
            "timeout_s": 6.0,
        }
        episode_path = os.path.join(work, "ep_%s.yaml" % terrain)
        with open(episode_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(episode, handle)
        result = execute_episode(episode_path, os.path.join(REPO, "configs/runtimes.yaml"))
        log = json.load(open(os.path.join(episode["output_dir"], "episode_log.json"),
                             encoding="utf-8"))
        print("%-5s status=%-8s reason=%-16s samples=%d"
              % (terrain, result.status, log.get("terminal_reason"), len(log.get("samples", []))))
        runs.append(episode["output_dir"])

    # 5. Aggregate into a dashboard.
    aggregate = aggregate_runs(runs)
    aggregate_path = os.path.join(work, "aggregate.json")
    dump_json(aggregate_path, aggregate)
    dashboard = os.path.join(work, "dashboard.html")
    generate_aggregate_report(aggregate_path, dashboard)
    print("\nstatus counts: %s" % aggregate.get("status_counts"))
    print("dashboard: %s" % dashboard)


if __name__ == "__main__":
    main()
