# Evaluation workflow

## 1. Verify the machine and assets

```bash
export PYTHONPATH="$PWD/src"
python -m robustness_lab doctor
python -m robustness_lab assets verify
```

`ready` means CPU/core evaluation can run. `full_cross_sim_ready` additionally requires all configured runtimes and the NVIDIA driver.

## 2. Inspect each robot representation

```bash
python -m robustness_lab robot inspect \
  assets/cache/unitree_ros/robots/go2_description/urdf/go2_description.urdf \
  --format urdf --override configs/robots/go2.yaml \
  --output assets/cache/go2_urdf_profile.json

python -m robustness_lab robot inspect \
  assets/cache/mujoco_menagerie/unitree_go2/go2.xml \
  --format mjcf --override configs/robots/go2_mjcf.yaml \
  --output assets/cache/go2_mjcf_profile.json

python -m robustness_lab robot compare \
  assets/cache/go2_urdf_profile.json assets/cache/go2_mjcf_profile.json \
  --output reports/go2_robot_conformance.json
```

To create the Isaac Lab asset while preserving the URDF as provenance:

```bash
python -m robustness_lab robot convert-urdf \
  assets/cache/unitree_ros/robots/go2_description/urdf/go2_description.urdf \
  --output assets/cache/go2_isaaclab/go2.usd
```

The official URDF and Menagerie MJCF currently differ in fixed-link topology,
total mass (about 5.47%), nominal pose, and declared actuator velocity limits;
their reported COM also narrowly exceeds the 5 mm default. The supplied
conformance report therefore blocks a cross-engine equivalence claim until one
canonical model/pose contract is chosen.

## 3. Generate worlds

```bash
python -m robustness_lab world generate --terrain rs8 \
  --robot assets/cache/go2_urdf_profile.json --difficulty 0.5 --seed 7 \
  --output runs/go2-rs8-world.json
python -m robustness_lab world preview runs/go2-rs8-world.json \
  --output reports/go2-rs8.svg
```

Generate the reproducible Go2 visual variation catalogue (world JSON, MuJoCo
XML, headless PNG, manifest, contact sheet, and clickable HTML):

```bash
python -m robustness_lab gallery configs/galleries/go2_world_variations.yaml \
  --robot assets/cache/go2_mjcf_profile.json \
  --robot-asset assets/cache/mujoco_menagerie/unitree_go2/go2.xml \
  --output reports/world_variation_gallery
```

For an AE cell, add `--purple rs3 --brown rs10`. For AEE, add `--layout exhaustive`, `sparse`, `random`, `long`, or `grid`. A connected four-port grid can be generated with `--layout grid --rows 3 --columns 3 --pair-mode cycle`; use `--pair-mode repeat --purple rs11 --brown rs4` to repeat one AE entity on every edge.

## 4. Materialize and run a suite

```bash
python -m robustness_lab suite configs/suites/quick.yaml \
  --robot assets/cache/go2_mjcf_profile.json --backend mujoco \
  --robot-asset assets/cache/mujoco_menagerie/unitree_go2/go2.xml \
  --robot-format mjcf --controller configs/controllers/pd_hold.yaml \
  --output runs/suites/go2-mujoco-quick
```

The result is a frozen `suite_manifest.json`, world files, and one episode YAML per trial. Run episode YAMLs with `robustlab run`. This two-step design permits cluster schedulers to distribute trials without changing the experiment matrix.

## 5. Calibrate and report

```bash
python -m robustness_lab calibrate generate \
  --robot assets/cache/go2_mjcf_profile.json \
  --output runs/calibration-worlds-mjcf
python -m robustness_lab report runs/fake-smoke \
  --output reports/fake-smoke.html
```

Use `calibrate analyze` after executing all fixture episodes. Use `aggregate` with the conformance and calibration files to obtain confidence intervals, capability curves, and validity-gated cross-backend deltas.

Render that aggregate as the suite-level dashboard with:

```bash
python -m robustness_lab report reports/reference_aggregate.json \
  --output reports/reference-dashboard.html
```

## 6. Run the discrete pruned examination

```bash
python -m robustness_lab examine start \
  configs/examiners/discrete_go2_fake.yaml --max-trials 3
python -m robustness_lab examine resume \
  runs/discrete_examiner/go2_fake/examination_state.json --max-trials 25
```

Open `runs/discrete_examiner/go2_fake/dashboard/index.html`. The page is
regenerated after every trial and separates behavioral failure counts from
simulator errors, skips, and unsupported physics. See
[discrete_pruned_benchmark.md](discrete_pruned_benchmark.md) for the exact
99-single-terrain/330-pair catalogue and real-policy configurations.
