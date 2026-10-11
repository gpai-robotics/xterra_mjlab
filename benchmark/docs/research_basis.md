# Research basis

The implementation was grounded in these benchmark and evaluation patterns:

- [Barkour](https://arxiv.org/abs/2305.14654): timed courses, obstacle-level outcomes, penalties, and repeated trials.
- [Robot Parkour Learning](https://proceedings.mlr.press/v229/zhuang23a/zhuang23a.pdf): success/distance outcomes, multiple seeds, OOD obstacle sweeps, and dynamics/perception perturbations.
- [Extreme Parkour](https://arxiv.org/abs/2309.14341): robot-normalized obstacle scales and explicit control/sensor timing.
- [Robust Robot Walker](https://robust-robot-walker.github.io/resources/rrw_paper.pdf): success, pass time, distance, stuck, fall, and rollover outcomes.
- [Robust recovery study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063321/): recovery probability versus disturbance magnitude and gait phase.
- [NIST robot test methods](https://www.nist.gov/el/intelligent-systems-division-73500/test-methods): reproducible ramps, gaps, stairs, repetitions, and threshold capability.
- [EUROBENCH performance indicators](https://eurobench.github.io/sofware_documentation/latest/pi_spec.html): protocol/data/indicator separation.
- [SimBenchmark](https://leggedrobotics.github.io/SimBenchmark/) and [Humanoid-Gym](https://arxiv.org/abs/2404.05695): explicit simulation-engine comparison and sim-to-sim validation.

These are design references, not bundled datasets. External robot/course assets retain their upstream licenses and exact revisions in `assets/sources.yaml`.
