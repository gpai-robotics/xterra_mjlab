# Third-party licenses

`xterra_mjlab` is licensed under Apache-2.0 (see the top-level `LICENSE` and
`NOTICE`). It depends on the following third-party components at runtime; their
licenses are collected here for convenience, with canonical text upstream.

| Component | License | File | Redistributed here |
|-----------|---------|------|--------------------|
| mjlab (`mjlab`) | Apache-2.0 | [mjlab-Apache-2.0.txt](mjlab-Apache-2.0.txt) | No (pip install) |
| rsl_rl (`rsl-rl-lib`) | BSD-3-Clause | [rsl_rl-BSD-3-Clause.txt](rsl_rl-BSD-3-Clause.txt) | No (pip install) |
| MuJoCo / MuJoCo-Warp / Warp | Apache-2.0 | [MuJoCo-and-Warp-Apache-2.0.txt](MuJoCo-and-Warp-Apache-2.0.txt) | No (pip install) |
| PyTorch, Gymnasium, NumPy | BSD / MIT | (see each package's distribution) | No (pip install) |

None of these are vendored into this repository. The SvanM2 robot description
(MJCF + STL meshes) under `xterra_mjlab/assets/svanm2/` is xTerra's own and
released under this repository's Apache-2.0 license (see `NOTICE`).

The robustness benchmark under `benchmark/` is MIT-licensed
([`benchmark/LICENSE`](../benchmark/LICENSE)) and vendors these third-party files:

| Component | License | File | Location |
|-----------|---------|------|----------|
| Unitree Go2 URDF + meshes (`unitree_ros`) | BSD-3-Clause | [LICENSE](../benchmark/assets/cache/unitree_ros/LICENSE) | `benchmark/assets/cache/unitree_ros/` |
| Unitree Go2 MJCF + meshes (MuJoCo Menagerie) | BSD-3-Clause | [LICENSE](../benchmark/assets/cache/mujoco_menagerie/unitree_go2/LICENSE) | `benchmark/assets/cache/mujoco_menagerie/unitree_go2/` |
| CTS and MoE-CTS weights ([RoboGauge](https://github.com/wty-yy/RoboGauge)) | MIT | [RoboGauge-MIT.txt](RoboGauge-MIT.txt) | `benchmark/external_policies/{cts,moe_cts}/weights/` |
| DreamWaQ and MoE-CTS weights ([go2_rl_robotlab](https://github.com/APLaS-Plus/go2_rl_robotlab)) | Apache-2.0 | [go2_rl_robotlab-Apache-2.0.txt](go2_rl_robotlab-Apache-2.0.txt) | `benchmark/external_policies/{dreamwaq,moe_cts}/weights/` |
