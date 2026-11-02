# Fresh PTZ precision profile

**Current local default (2026-11-02):**
`unified_motion_20261101/control-v3-turn-training-20261102/deploy.pt`.
The selected weights are unchanged. As of 2026-11-03 the online entrypoint also
enables camera centering v4; see [`FRONT_CENTERING_FIX.md`](FRONT_CENTERING_FIX.md).
Use `--no-centering-v4` to retain the preceding camera governor.
Camera v3 adds zoom lookahead, edge-axis priority and bounded last-RGB-motion
camera search, plus 768 new turn transitions and 384 original P-DQN updates.
Detection and ReID weights remain fixed. Deployment association additionally
includes the 2026-11-03 turn-reassociation fix in
[`TURN_REASSOCIATION_FIX.md`](TURN_REASSOCIATION_FIX.md). The older v2 candidate and
the published stable weights below remain available for rollback. Focused
results and limitations are in
`unified_motion_20261101/control-v3-validation-20261102/CONTROL_V3_RESULT.md`.

`precision_policy_v2.py` defines a separate 15-input observation contract. It
constructs the original author's P-DQN Q network, duration network and two target
copies from a new seed. No legacy PTZ weights or replay are loaded. Both Adam
optimizers start empty. The hidden layers remain 128/256/128/64; the original
source update equations and gradient inversion are unchanged.

The frozen whole-vehicle detector and FastReID checkpoint remain the existing
ones. The original training used the frozen RGB identity association; deployment
now adds the separately tested turn-reassociation fix. The explicit motor/scale
safety governor is unchanged. Every transition records the raw proposal, governor
decision, actual executed discrete action and actual selected duration. A low
center error alone therefore does not prove that the neural proposal improved.

The current attempt uses a two-stage execution curriculum. The first 32 training
layouts retain legacy 40 ms maximum pan/tilt pulses. Their checkpoint continues
the same freshly initialized networks, optimizers and newly collected replay;
it does not import the released pretrained policy. Subsequent layouts allow up
to 100 ms pulses and an RGB angular-speed-based gain floor, retaining the
remaining-image-error bound and 1 ms minimum. This removes an actuator-duty
limit exposed by fast near-field samples. Stage configurations and sources are
saved separately. A random-initialization online control with the same final
governor is needed to separate neural learning from this engineering change.

## Observation and objective changes

- Actual executed action and seconds replace the legacy lagged action/index-time
  encoding. Actual RGB area is used directly, without the old 4%-to-2% remap.
- RGB angular-Kalman velocity replaces a short per-command fit that can return
  zero when fewer than three command observations are available.
- Image age and known own-camera motion are explicit. Position is projected
  forward using RGB velocity and image age, never a target pose or route.
- The desired area is 1.6%, with a 2% upper guard. Rewards use actual visible
  pixel distance, scale error, progress, and overshoot; invisible targets carry
  a penalty. Predicted boxes are never used as visible reward observations.

## Procedural data

The intended first complete run uses 16 support layouts × 64 transitions and 96
training layouts × 96 transitions: 10,240 new real UE command transitions. Eight
families are balanced: static fine correction, slow movement/stops, fast motion
up to 36 m/s, acceleration, turns, inclines, physical truck occlusion with an
identical-colour follower, and tiny-target acquisition. Camera height, view,
offset, route direction, speed, scale and colour vary by deterministic seed.
Fast-motion layouts include a short acceleration phase to initialize RGB motion
estimates before sustained fast movement.

Training uses a controlled synchronous simulation clock. The world advances
through explicitly sampled image delays and command cycles. During pulses and
holds, intermediate real RGB frames update the tracker in <=33.3 ms simulation
slices; policy state history is built once per completed command. GPU learning
does not advance the world. This collection speed is **not** online FPS.

Train, support, development and check layouts have disjoint seed ranges. Scene
geometry is used only to construct UE layouts. Policy inputs, action selection
and reward do not consume target ground truth, actor identifiers, speed, route,
scenario family or phase.

## Commands

Use the existing `50mamba` interpreter and run from this directory. GPU/UE runs
must be sequential; do not run evaluation alongside training.

```powershell
python train_precision_from_scratch.py --output unified_motion_20261101/NEW-TRAINING
python evaluate_precision_policy.py --checkpoint unified_motion_20261101/NEW-TRAINING/last.pt --output unified_motion_20261101/NEW-HELDOUT
python live_fast_online_results.py --scene viaduct30 --seconds 28 --fps 30 --perception-fps 30 --checkpoint unified_motion_20261101/NEW-TRAINING/last.pt --output unified_motion_20261101/NEW-ONLINE
```

The live runner selects the observation contract from checkpoint metadata and
rejects unsupported precision profiles. The old default checkpoint and released
private repository are not replaced by a training run.

## Perception is fixed for camera control work

Following the user's clarification, the live control entry point retains the
original full FP32 detection, fresh FP32 ReID and original robust association.
Experimental flow interleaving, AMP and operator fusion are not enabled or
offered by this entry point. Failed performance experiments remain in the saved
comparison folder with frozen sources and genuine online videos.

`--latest-frame` remains a separate capture experiment; it is off by default.
The final camera-only run uses the original capture selection as well.

`benchmark_fresh_reid_amp.py` measures synchronized FP32/AMP/fused inference on
real archived detector crops. This is numerical/performance screening, not an
identity robustness result. Evaluation videos are produced only by the genuine
online recorder; subsequent encoding does not rerun inference or draw boxes.

`evaluate_camera_control_only.py` uses saved native diagnostics only AFTER the
run to score geometric target centering on every source frame, including
occlusion/no-box frames. It reports pixel and angular errors, target departures
from the view and command reversals. A target still in view without a selected
box is recorded separately as a perception event; departures from the view
remain camera-control failures. No GT is supplied to the policy or reward.

## Saved candidate and rejected calibration

The preceding v2 candidate is
`unified_motion_20261101/precision-v2-fresh-r4-20261102/last.pt`.
`run_precision_control_candidate.py --output NEW-FOLDER` now runs the validated
local v3 default with original perception on the 30 m/s scene. The previous
published checkpoint file is not overwritten.

An auxiliary fit of only the four pan/tilt duration output rows reduced internal
duration-label error but FAILED closed-loop camera tracking. It remains saved
under `precision-duration-calibration-20261102` for diagnosis and is rejected;
the opt-in launcher does not load it. Its losses are explicitly supervised from
actual executed motor durations, not original-source P-DQN updates. Both Q
networks and all perception were unchanged. This failure shows why small fitting
error alone is not a valid camera-control acceptance criterion.
