# Metric and event definitions

Metric definitions are versioned independently from simulator code. `events-v1` defaults can be overridden in `EpisodeSpec.event_thresholds`, and the resolved values belong in the run manifest.

## Terminal events

- **Success:** goal distance at most `0.25 × body length`, mandatory route conditions satisfied, and no terminal failure.
- **Fall:** trunk/forbidden-body contact, base height below `0.45 × standing height`, or absolute roll/pitch above 60 degrees for at least 0.2 seconds.
- **Stuck:** command speed above 0.2 m/s while progress is below `0.05 × body length` over two seconds.
- **Slip:** a contacting foot has tangential speed above `max(0.05 m/s, 0.1 × command speed)`.
- **Recovery:** speed error is within 10% of command and roll/pitch are within five degrees for 0.5 seconds after a transition or disturbance.

Timeout, backend unavailability, and simulator error are separate outcomes.

## Primary metric families

| Family | Examples |
|---|---|
| Outcome | success, time, displacement, progress, fall/stuck reason |
| Tracking | velocity/yaw RMSE and p95 error |
| Stability | roll/pitch RMS, maximum orientation, body-height range |
| Contact | slip ratio/distance, contact impulse, peak force, body contacts |
| Efficiency | absolute and positive work, mechanical CoT, torque saturation, thermal proxy |
| Recovery | recovery count/time and disturbance outcome |
| Affordance | central/bypass choice and route evidence |
| Perception | observation age and exposed estimate errors |
| Systems | inference p50/p95/p99, deadline misses, simulator step p95, real-time factor |
| Robustness | success/severity curve, normalized AUC, C50, conservative C90 |

Mechanical cost of transport uses:

`CoT = integral(sum(abs(torque × joint_velocity)), dt) / (mass × gravity × displacement)`

Contact impulse is preferred over a raw force peak because force peaks depend strongly on solver step and contact formulation. Peak force remains available with its sampling metadata.

## Statistics

- Binary proportions use Wilson 95% intervals.
- Continuous values use deterministic bootstrap intervals.
- Matched world/seed comparisons use paired records.
- C90 is the greatest tested severity whose lower Wilson bound remains at least 90%.
- Quick suites use 10 trials and standard simulation suites use 30 trials per fixed condition.
- Controller training seeds are an outer experimental unit and must not be pooled as ordinary episode samples.

Invalid metrics remain in output with a reason; they are never converted to zero.

