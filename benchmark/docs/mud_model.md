# RS11 depth-resistive mud

New RS11 worlds use the `depth_resistive_mud_v1` tier. The model is a
reduced-order vertical visco-elasto-plastic approximation based on Chen, Yi and
Shan, [*A Reduced-Order Resistive Force Model for Robotic Foot-Mud
Interactions*](https://arxiv.org/abs/2403.02617) (2024).

For penetration depth `z > 0`, foot area `A`, and upward vertical velocity `v`,
the signed world-frame vertical force is

```text
F = alpha * (z / H)^beta * A - b * v * A - sigma_y * A * s
```

The suction state is active only during withdrawal (`v > 0`) and advances as
`s = 1 - (1 - s) exp(-dt / tau)`. It resets when withdrawal stops or the foot
leaves the mud. The final force is clamped to `max_force_n` in either direction.

The recorded parameters are:

- `support_stress_pa` (`alpha`): support stress at the reference depth;
- `depth_exponent` (`beta`): exponent of the rate-independent depth law;
- `reference_depth_m` (`H`): depth normalisation;
- `damping_pa_s_per_m` (`b`): vertical velocity damping;
- `yield_stress_pa` (`sigma_y`): maximum suction stress;
- `suction_time_constant_s` (`tau`): suction build-up time; and
- `max_force_n`: per-foot numerical and physical safety clamp.

Difficulty stays within the paper's measured ranges for `alpha` (0.01–0.17
MPa), `beta` (0.38–0.56), and `sigma_y` (2–17 kPa). Increasing difficulty makes
the mud softer underfoot by reducing `alpha` and grabbier during withdrawal by
increasing `sigma_y`.

The visible mud volume is non-colliding so a foot can penetrate it. A thin rigid
box at the volume bottom is a geometric backstop: sinkage cannot exceed the mud
thickness even if the reduced-order force saturates.

## Limits

This is a per-foot vertical force model, not a deformable-material simulation.
It does not model rut creation or persistent terrain deformation, thixotropic
breakdown and recovery, displaced material, pore-fluid flow, or a shear-strength
traction law. Horizontal traction still comes from the ordinary MuJoCo contact
model at the rigid backstop; the mud law itself supplies no lateral force.
Successive footsteps therefore do not inherit a changed surface from earlier
contacts.
