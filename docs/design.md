# Immortals SSL Simulator — Design

Status: implemented (round 2, 2026-09-07). Supersedes `architecture.md`.
See §10 for known gaps and deliberate deviations from this spec.

This document is the specification implementation agents work from. It is
informed by a source-level review of grSim (ODE), ER-Force's simulator (Bullet)
and TIGERs Sumatra's internal simulator (analytical). Review notes with
file:line references live outside the repo; the conclusions are folded in here.

## 1. Goals

1. Drop-in replacement for grSim for the Immortals software (Tyr and the older
   `Software` tree): standard SSL simulation protocol on 10300/10301/10302,
   vision multicast on 224.5.23.2:10020, and the legacy grSim packet protocol
   on 20011 (Tyr's `grsim.cpp` sender still uses it).
2. Physically faithful where it matters for strategy: ball two-phase
   slide/roll model and chip bounce model that are *the same constants the
   geometry packet advertises*; robots with acceleration limits and wheel slip;
   a dribbler that can lose the ball; a kicker with charge time.
3. Fast: 22 robots at 1 kHz physics, target ≥ 50× real time headless on one
   core. Faster-than-realtime and lock-step stepping are first class.
4. Deterministic: integer simulation time, fixed step, seeded RNG per concern,
   no wall clock anywhere in the core. Same seed + same inputs ⇒ identical
   output bytes.
5. Realistic vision: multiple cameras with overlap, floor-projected flying ball,
   partial occlusion, Gaussian noise, dropouts, false dribbler balls, latency.
   All optional via a realism config with named presets.
6. Embeddable: the core is a plain Rust library with no I/O, no threads and no
   protobuf; the network layer and CLI are thin adapters.

Non-goals for this round: 3D rendering, robots tipping over, a GUI editor.
A minimal 2D egui viewer (`ssl-sim-viewer`) is included; it only consumes the
network output and sends standard `SimulatorCommand`s, so it stays optional.

## 2. What we learned from the three simulators

| Topic | grSim (ODE) | ER-Force (Bullet) | Sumatra (analytical) | Decision |
|---|---|---|---|---|
| Ball deceleration | hand-rolled constant 0.49 m/s², advertised model not simulated | hand-rolled 0.49 m/s² on top of Bullet friction, advertised −0.35 | closed-form two-phase, spin-driven switch, same code as AI predictor | **Sumatra's model, closed form, spin state** |
| Chip | vz set directly, friction applied in air (bug) | ballistic Bullet, no bounce model | fixed-loss hops, first/other hop damping | **Sumatra fixed-loss model, per-bounce damping, chips clear walls/goals by height** |
| Robot drive | physical wheels, torque-capped velocity motors | single body, PD force controller, hardcoded inertia, no wheels | kinematic bang-bang / constant accel | **Wheel force model on a 2D rigid body (no wheel bodies), plus an "ideal" kinematic mode** |
| Robot hull | 0.073 m sphere for ball contacts (!) | compound convex hulls w/ mouth | disc + flat front chord | **Disc + flat front chord (2D), height-gated for the ball** |
| Dribbler | anchorless hinge joint (snaps ball) | hinge motor on 7 mm roller or glue constraint | rigid position snap | **Traction model (holding force budget, slip → release) + optional glue mode** |
| Kicker | set velocity, no cooldown | impulse, 100 ms boolean charge, clamp | impulse on contact, no charge | **Impulse, contact zone test, configurable charge/cooldown, clamp to specs** |
| Vision split | 4 quadrants, 0.2 m overlap | Manhattan-Voronoi, 2·overlap band | none | **Manhattan-Voronoi + overlap band (ER-Force rule, test-pinned)** |
| Ball height in vision | pinhole floor projection | floor projection + area | none | **Floor projection; `z` omitted by default (option to emit true z)** |
| Occlusion | none | 15×15 disc sampling, centroid of visible part | none | **Same idea, analytic ray-vs-cylinder** |
| Time | wall clock, drifts | 5 ms ticks, delay via QTimer per packet | 10 ms cam / 1 ms sub | **u64 ns sim time, 1 ms substep, ordered delay queue keyed on sim time** |
| Sync mode | none | none (FastSimulator test-only) | SimNet TCP (proprietary) | **`SimulationSyncRequest/Response` implemented** |
| Specs change | full world restart | full world rebuild | n/a | **Applied live** |

## 3. Crate layout

```
ssl-sim-core    deterministic simulation library (no I/O, no proto, no threads)
ssl-sim-proto   generated protobuf: ssl-simulation-protocol (+ ER-Force custom
                messages), ssl-vision (wrapper, tracked), legacy grSim packets
ssl-sim-net     proto <-> core translation, UDP endpoints, vision publisher,
                sync mode, legacy grSim adapter
ssl-sim-cli     `ssl-sim` binary: config file + flags, run loop, logging
```

Dependency direction: `cli → net → {core, proto}`. `core` depends only on
`glam` (or nalgebra) for vectors, `rand_xoshiro`/`rand` for RNG, `thiserror`.
No physics engine crate. Rapier is removed.

## 4. Units and conventions

- Core is SI: metres, seconds, radians, kilograms. Wire conversion (mm, ns) is
  done only in `ssl-sim-net`.
- World frame: x along the field length, y along the width, z up. Blue defends
  −x by default (configurable side).
- Robot frame: +x forward (kicker direction), +y left. Orientation θ is the
  angle of robot +x from world +x, CCW positive.
- `MoveLocalVelocity{forward, left, angular}` maps to robot-frame (vx, vy, ω).
- Wheel i has mounting angle φᵢ measured CCW from robot +x to the wheel's
  radial direction; its drive direction is the tangent (−sin φᵢ, cos φᵢ).
  Protocol `RobotWheelAngles` is documented clockwise; the net layer negates.
  Default angles (front_left, back_left, back_right, front_right) =
  (60°, 135°, 225°, 300°) in the CCW convention, i.e. grSim's layout.
- Simulation time `SimTime(u64)` in nanoseconds. Substep `dt` default 1 ms.

## 5. Core model

### 5.1 World and stepping

```
World {
  time: SimTime, frame: u64,
  field: FieldGeometry, params: Params (ball, robot defaults, realism),
  ball: Ball, robots: BTreeMap<RobotId, Robot>,
  pending: teleports/by_force movers, rng: Rngs,
  events: Vec<Event>  (kick, goal, ball out, collisions …)
}
World::step(&mut self)           // exactly one substep
World::step_for(&mut self, dur)  // n substeps, rejects non-multiples
```

Order inside one substep:

1. Apply pending control (teleports, `by_force` movers, spec changes).
2. Robot actuators: sample latest command (with command timeout → coast),
   firmware limiter, wheel forces, kicker/dribbler logic.
3. Integrate robots (semi-implicit Euler), then resolve robot–robot and
   robot–boundary contacts (positional correction + restitution/friction
   impulses, 4 iterations, mass-weighted).
4. Ball: collision sweep against robots / walls / goal frames ordered by time
   of impact, then closed-form advance of the remaining dt.
5. Bookkeeping: events, `time += dt`, `frame += 1`.

Commands arrive asynchronously (net layer pushes into the world between
substeps) or synchronously (sync mode). Robots hold the last command; after
`command_timeout` (default 0.1 s) without a command the robot coasts to a stop
(motors off, decel 8 m/s², 50 rad/s²).

### 5.2 Ball

State: `pos: Vec3, vel: Vec3, spin: Vec2` (ground-contact spin such that
rolling ⇔ `vel_xy == spin·r`). Parameters (defaults, all configurable and all
advertised verbatim in the geometry packet):

| Param | Default | Note |
|---|---|---|
| radius | 0.0215 m | |
| mass | 0.046 kg | |
| acc_slide | −3.0 m/s² | Sumatra −3.0, ER-Force −3.9, real field measurements −2.5…−3.4 |
| acc_roll | −0.30 m/s² | Sumatra −0.26…−0.45, ER-Force −0.35 |
| inertia_distribution p | 0.5 | ⇒ published k_switch = 1/(1+p) = 0.667 |
| chip_damping_xy_first_hop | 0.75 | TIGERs field files |
| chip_damping_xy_other_hops | 0.95 | |
| chip_damping_z | 0.50 | |
| min_hop_height | 0.01 m | below this the ball is grounded |
| rest_speed | 0.01 m/s | snap to rest |

**Flat motion** (Sumatra `FlatBallTrajectory`): from `(v0, s0)` compute contact
slip `c = v0 − s0·r`; if `|c| < ε` the ball rolls with `a = v̂·acc_roll`;
otherwise it slides with `a = ĉ·acc_slide`, spin accelerates by
`a/(r·p)`, and the switch happens after `t_sw = |(s0·r − v0)·p/(1+p)| / |a|`
(component along the dominant axis), then rolls. Rest time is analytic.
Implemented as `BallTrajectory::from_state(state, params)` with
`state_at(t)`, `time_by_distance(d)`, `time_by_velocity(v)`, and the chip
inverses (`vel_for_distance`, `vel_for_touchdown(n, d)`, `vel_for_height`).
The simulator advances the ball with `trajectory.state_at(dt)`; the same
type is exported for prediction use by clients.

**Chip motion**: ballistic with g = 9.81 while `z > 0` or `vz > 0`; on
touchdown `vel *= (d_xy, d_xy, −d_z)` with `d_xy` = first-hop damping if spin
is zero, else other-hops damping; spin set to rolling; hop loop ends when the
next apex would be below `min_hop_height`, then flat model. No air drag or
Magnus (documented limitation, hook left for later).

**Collision kernel** (Sumatra, with spin kept): for contact normal n̂, object
surface velocity u (including ω × r of the robot), ball velocity v:
`if n̂·u ≤ n̂·v: no effect` (receding); `v'_n = n̂·u + (n̂·u − n̂·v)(1 − k_n)`,
`v'_t = k_t·(t̂·u) + (1 − k_t)(t̂·v)`, `v'_z` scaled by `(1 − k_t)` for
vertical walls. Pairs: ball–robot hull `k_n 0.5`, `k_t 0.0`; ball–kicker face
`k_n 0.6`, `k_t 0.3`; ball–wall `k_n 0.5`; ball–goal frame `k_n 0.5`. Spin is
reflected and damped by 0.4 rather than discarded. The robot receives the
equal and opposite impulse (negligible but keeps momentum honest).

**Height gating**: a ball contacts a robot only if `z_ball − r < robot.height`;
field boundary boards have height `wall_height` (default 0.10 m), goal frames
`goal_height` (0.155 m). Above those the ball passes. A tall invisible "room"
box at field + 1.0 m margin keeps the ball from leaving the world; the floor
is infinite.

**Out of play** is not enforced (that is the referee's job) but a
`BallLeftField` / `Goal` event is emitted for tooling.

### 5.3 Robot

Per-robot `RobotSpecs` (all fields settable live from the protocol):

| Field | Default | Source |
|---|---|---|
| radius | 0.09 m | proto default |
| height | 0.15 m | proto default |
| mass | 2.5 kg | Sumatra |
| inertia_z | ½·m·r² ≈ 0.0101 kg·m² | derived unless overridden |
| center_to_dribbler | 0.075 m | Sumatra physics value; mouth half-angle acos(c2d/r) |
| dribbler_width | 0.07 m | ER-Force `RobotSpecErForce` |
| shoot_radius | 0.0885 m | seated ball centre = c2d + r_ball − seat_depth (in front of the solid chord) |
| max_linear_kick_speed | 6.5 m/s | ER-Force gen-2020 |
| max_chip_kick_speed | 5.5 m/s | |
| limits.vel_absolute_max / vel_angular_max | 3.5 m/s / 20 rad/s | |
| limits.acc_speedup_absolute_max / brake | 4.0 / 6.0 m/s² | |
| limits.acc_speedup_angular_max / brake | 50 / 50 rad/s² | |
| wheel_angles | (60,135,225,300)° CCW | grSim layout |
| wheel_radius | 0.027 m | grSim ini |
| motor.max_wheel_speed | derived from vel_absolute_max·1.3 | |
| motor.max_force_per_wheel | 6 N (≈ 0.16 N·m torque at 0.027 m) | tuned so 4 wheels give ≈ 7–8 m/s² peak |
| motor.velocity_gain | 60 N per m/s of wheel-speed error | |
| traction.mu_drive | 0.8 | grSim wheel tangent |
| traction.mu_lateral | 0.05 | grSim roller axis |
| kicker.charge_time | 0.1 s (configurable; realistic ≈ 1–2 s) | ER-Force |
| kicker.max_ball_height | 0.05 m | ER-Force |
| dribbler.max_speed_rpm | 10 000 | normalisation only |
| dribbler.hold_accel | 4.0 m/s² at full speed | traction budget |

**Hull**: disc of `radius` with a flat front chord at distance
`center_to_dribbler`; the chord width follows. Used for ball contact (2D, with
height gate) and for robot–robot contact (disc only; the chord is ignored
between robots, as every simulator does).

**Drive, `wheels` mode (default)**: robot is a 2D rigid body `(x, y, θ, vx,
vy, ω)` with mass and `inertia_z`. Each substep:

1. *Firmware limiter*: the commanded local twist is rate-limited toward the
   previous setpoint using `RobotLimits` (separate speed-up and brake
   accelerations, absolute and angular), then clamped to the velocity limits.
   This is what real firmware does and what strategy code assumes.
2. *Inverse kinematics*: wheel surface speed setpoints
   `u_i* = −vx sin φᵢ + vy cos φᵢ + R·ω` (R = wheel mounting radius = radius −
   half wheel thickness ≈ 0.0875 m).
3. *Wheel forces*: actual surface speed `u_i` from the body twist by the same
   formula; drive force along the wheel tangent
   `F_i = clamp(K_v (u_i* − u_i), ±F_max)`; traction clamp
   `|F_i| ≤ mu_drive·N_i`, `N_i = m g / 4`; lateral roller friction
   `F_lat = −mu_lateral·N_i·sign(v_lat)` (small). Sum to body force/torque.
4. *Integrate* with semi-implicit Euler.

Emergent: acceleration is limited by motor force and traction, so an
over-demanding command slips instead of teleporting; `MoveWheelVelocity`
(m/s per the protocol) skips step 2; `MoveGlobalVelocity` rotates into the
robot frame first.

**Drive, `ideal` mode**: skip 3; integrate the limited setpoint directly. For
fast CI and for teams that want grSim-like behaviour. Robots still collide.

**Kicker**: `kick_speed > 0` in the current command fires when charged and
the ball is in the kick zone (ball centre within `[c2d − 0.005, c2d +
r_ball + 0.015]` along the heading, within `±dribbler_width/2` laterally,
`z_ball < max_ball_height`). Impulse sets the ball velocity to
`R(θ)·(cos α, 0, sin α)·speed` plus the robot's own velocity, after cancelling
the ball's incoming normal component; `speed` clamped to `[0.05,
max_linear|chip]`, `α` = `kick_angle` degrees (any value accepted; 0 straight,
45 typical chip). Spin reset to zero. Discharge, recharge after
`charge_time`. Dribbler is released for that substep. Feedback gets a
`kick` event for the legacy `Robots_Status`.

**Dribbler** (`dribbler_speed > 0`, RPM): if the ball is in the *mouth zone*
(kick zone extended 0.01 m inward), the dribbler applies to the ball a force
toward the seated point `(shoot_radius, 0)` in the robot frame plus the robot
surface velocity as the target velocity, with a force budget
`m_ball · hold_accel · min(1, rpm / max_speed_rpm)`. If the required force to
keep the ball seated (from the relative acceleration of the robot and the
ball's lateral velocity) exceeds the budget, the ball slips and leaves. Ball
spin is driven backward (`−ω_dribbler·r_roller`), which shortens a subsequent
kick's slide phase. `glue` mode (realism `simulate_dribbling = false`) instead
snaps the ball to the seated point and gives it the surface velocity (Sumatra
behaviour). `dribbler_ball_contact` feedback is the barrier test: ball centre
within `[c2d, c2d + 0.0235]` along the heading and `±(r·sin half-angle)`
laterally (Sumatra/TIGERs IR geometry), evaluated at reply time.

### 5.4 Field

`FieldGeometry` with Division A and B presets (SSL rules 2024+):

| | Div A | Div B |
|---|---|---|
| length × width | 12.0 × 9.0 | 9.0 × 6.0 |
| goal width × depth | 1.8 × 0.18 | 1.0 × 0.18 |
| goal height / wall thickness | 0.155 / 0.02 | same |
| boundary width | 0.3 | 0.3 |
| penalty area depth × width | 1.8 × 3.6 | 1.0 × 2.0 |
| centre circle radius | 0.5 | 0.5 |
| line thickness | 0.01 | 0.01 |

Note: the penalty mark is not on the wire and not needed by the simulator. Lines
emitted: the 2018+ set (`TopTouchLine, BottomTouchLine, LeftGoalLine,
RightGoalLine, HalfwayLine, CenterLine, LeftPenaltyStretch,
RightPenaltyStretch, LeftFieldLeftPenaltyStretch, LeftFieldRightPenaltyStretch,
RightFieldLeftPenaltyStretch, RightFieldRightPenaltyStretch`) and arc
`CenterCircle`. Collision geometry: boundary boards at the outer edge of the
boundary area (height `wall_height`), goal frames (two posts + back plate,
20 mm thick, height 0.155), room box. Default formations for 0…15 robots per
team (grSim "inside" formation; configurable list).

### 5.5 Vision model

Calibrated against ten 2026 division-A/B game logs; `docs/calibration/vision.md`
§8 is the value table and §9 the list of model changes the data demanded.

Cameras: `default_camera_count` (default **2**, the only rig seen on a
division-A field; 1 for Div B, 4 supported), auto-placed at
`(±default_camera_x_fraction·L, 0)` for two and `(±fraction·L, ±W/4)` for four,
at `default_camera_height` (**6.4 m**), all configurable or overridden by an
explicit `[[vision.cameras]]` list. Frame period **1/73.3 s** per camera.

Each camera has its **own** capture instant: `camera_phase = locked` uses
`phase_offsets[i]` within the period, `free_running` draws one uniform phase per
camera from the seeded `calibration` stream at first capture. Every capture is
its own `VisionOutput` (exactly one `DetectionFrame`) and its own
`SSL_WrapperPacket`; capture instants are quantised to the substep.

Static per-camera calibration state, drawn once from the `calibration` stream:
a constant position offset of magnitude `object_position_offset` in a random
direction, a smooth position warp (sum of low-frequency plane waves, std
`calibration_warp_stddev`, wavelength `calibration_warp_length`, normalised over
the field), a constant orientation offset with random sign, and an orientation
warp. This is the dominant error a client sees (20 mm between cameras versus
0.4 mm of white noise) and is applied before the Gaussian noise.

Per frame, per camera:

1. Region test: Manhattan-Voronoi with band `2·camera_overlap`
   (`own ≤ min + 2·overlap`, ER-Force rule and test vectors) **and** a hard
   field-of-view disc of `fov_radius` (6.6 m) around the nadir.
2. Robots: pose + static calibration error + Gaussian `stddev_robot_p`,
   `stddev_robot_phi`; dropped with `missing_robot_detections`. Height field =
   `RobotSpecs::height`. `confidence ~ N(robot_confidence_mean,
   confidence_stddev)` clamped to (0, 1]. With probability
   `duplicate_robot_rate` a second entry with the same id and independent noise
   is emitted.
3. Ball: floor projection through the camera pinhole using the true camera
   position: `p' = c_xy + (p_xy − c_xy)·c_z/(c_z − min(z − r, 0.9 c_z))`;
   `area = area_at_nadir_px · visibility · height_term` (no horizontal distance
   term — measured `area` is flat over the working area; `PIXEL_PER_AREA = 1`,
   `focal_length_px` is the real calibrated focal length and only enters the
   height term and the geometry packet) plus `stddev_ball_area`; occlusion (if
   enabled): 15×15 samples on the ball disc facing the camera, each ray tested
   analytically against every robot cylinder; visible fraction <
   `ball_visibility_threshold` ⇒ dropped, else reported at the centroid of
   visible samples; then noise and `missing_ball_detections`. `z` omitted unless
   `report_ball_z`; `area` omitted entirely unless `report_area`.
4. Spurious dribbler balls with rate `dribbler_ball_detections` per robot per
   second, on the robot's centreline `spurious_ball_forward` (0.13 m) ahead of
   its centre with lateral `N(0, spurious_ball_lateral_stddev)` and area
   `spurious_ball_area_px`.
5. Multiple balls are shuffled with the seeded RNG.
6. `frame_number` per camera, `t_capture = t + delay − processing`,
   `t_sent = t + delay` (sim seconds; plus a configurable epoch offset so the
   values look like Unix time for tools that care).

While sim time is inside a configured `[[vision.outages]]` window nothing is
emitted at all; frame numbers and the geometry cadence keep running, so
geometry goes out on the first frame after the gap.

Packets are queued with due time `t + vision_delay` and released when the
loop's sim time passes it (so fast mode and sync mode behave identically).
One `SSL_WrapperPacket` per camera capture is always emitted, even if empty. The
geometry packet (field, lines, arcs, ball models — with the *advertised*
`acc_roll` for the wire's constant-deceleration model — one calibration per
camera with correct `derived_camera_world_t*` and a self-consistent quaternion,
optionally perturbed by `camera_position_error`) is attached to camera 0's
packets every `geometry_every_n_frames` of that camera (default 73 = 1.00 s;
1 also supported).

A ground-truth stream (`TrackedWrapperPacket` on 224.5.23.2:10010) is optional
and off by default; it carries exact ball 3D pos/vel and robot pos/vel with
`source_name = "ssl-sim-truth"`.

### 5.6 Realism config

Flat struct `Realism`, loadable from TOML and from the wire via
`RealismConfigErForce` packed in `RealismConfig.custom` (so Ra-style tooling
and existing team scripts work). Fields: ER-Force's 17, plus the calibration
warp (`calibration_warp_stddev`, `calibration_warp_length`,
`calibration_orientation_offset`, `calibration_orientation_warp`), the kick
imperfections (`kick_direction_stddev`, `chip_angle_stddev`,
`kick_speed_factor_stddev`) and `command_delay`; the extras are config-file
only, since the ER-Force message has no room for them. Rig geometry and what a
camera reports live in `VisionConfig` instead.

Presets: `none`, **`realistic` (default, measured from the 2026 corpus)**,
`go26` and `rc26` (its two venues), and ER-Force's own values under
`erforce_friendly`, `erforce_realistic`, `erforce_rc2021` (aliases `friendly`,
`rc2021`). `BallParams::preset` and `RobotLimits::preset` are the equivalents
for the physics tables, exposed as `--ball-preset` and `--robot-limits`. Robot
command loss / response loss are applied in the net layer with their own seeded
RNG stream.

### 5.7 Randomness and determinism

`Rngs { vision_noise, vision_dropout, packet_loss, shuffle, physics,
calibration }`, each a `Xoshiro256++` seeded from `seed` + stream id.
`calibration` is drawn from exactly once per camera rig (phases, offset
directions, warp coefficients), so per-frame randomness never shifts the static
calibration and vice versa. `seed` comes from config/CLI
(default: fixed 0 so runs are reproducible unless the user asks for
`--seed random`). Robot iteration is `BTreeMap` order. No floating point
reductions depend on hash order. The core has a `state_hash()` used by the
determinism test.

## 6. Network layer (`ssl-sim-net`)

- One thread per socket, blocking `recv_from`, decodes and forwards
  `(message, reply_to)` over a channel to the sim thread; the sim thread
  applies between substeps and sends responses to the exact `reply_to`
  address of that datagram (never "last sender").
- Endpoints: control 10300 (`SimulatorCommand` → `SimulatorResponse`, also
  `SimulationSyncRequest` → `SimulationSyncResponse`); blue 10301 and yellow
  10302 (`RobotControl` → `RobotControlResponse`, and sync requests carrying
  `robot_control` for that team); legacy grSim 20011 (`grSim_Packet`, team
  from `isteamyellow`, replacement = teleports, `Robots_Status` sent back to
  the sender on 30011/30012 semantics: to the sender's address at the
  configured status port); vision multicast 224.5.23.2:10020 (port changeable
  via `SimulatorConfig.vision_port`, `--vision-addr` for localhost).
- Legacy command mapping: `veltangent` → forward, `velnormal` → left,
  `velangular` → angular, `kickspeedx/kickspeedz` → speed = hypot, angle =
  atan2; `spinner` → dribbler 1 or 0 (scaled to max rpm); `wheelsspeed` with
  `wheel1..4` in rad/s (grSim semantics) converted to m/s.
- `SimulatorControl.simulation_speed` sets the real-time scaling (0 = pause).
- `TeleportBall.teleport_safely`: ER-Force semantics (evict overlapping
  robots, zero robots within 1.5 m). `by_force`: sticky mover with impulse
  gain 0.1 (ball) / 1/6 (robot) until cancelled by a message with
  `by_force = false`. `roll`: set spin to rolling.
- `SimulatorConfig.geometry`: parse field size/goal from the packet and
  rebuild collision geometry **without** touching ball/robot state.
  `robot_specs`: applied live per robot; missing fields keep current values
  (partial specs allowed). Unknown `custom` Any types ignored with a warning
  error code.
- Errors use stable codes: `UNREADABLE`, `PARTIAL_COORD`, `VELOCITY_FORCE`,
  `TELEPORT_SAFELY_PARTIAL`, `CREATE_NOPOS_ROBOT`, `INVALID_SPEC`,
  `UNSUPPORTED` (with message). Per-robot errors go to the owning team's
  socket.
- Sync mode: on `SimulationSyncRequest` the loop applies `simulator_command`
  and `robot_control`, steps `round(sim_step / dt)` substeps (error if not a
  multiple), flushes the vision delay queue *as of the new time*, and answers
  with all detection frames generated during the step plus the robot control
  response. While a sync client is active the free-running clock is paused.

## 7. CLI (`ssl-sim`)

```
ssl-sim run [--config sim.toml] [--division a|b] [--robots 11]
            [--mode realtime|fast|sync] [--speed 1.0] [--step-ms 1]
            [--realism none|realistic|go26|rc26|erforce_*|<file>]
            [--ball-preset <name>] [--robot-limits <name>] [--cameras 1|2|4]
            [--seed N|random] [--vision-addr 224.5.23.2:10020] [--localhost]
            [--truth] [--no-legacy-grsim] [--log-level info]
ssl-sim presets      # default config as TOML + every preset name and what it is
```

Realtime loop: accumulate wall time × speed, run whole substeps, publish due
vision packets, sleep the remainder (spin-wait the last ~0.5 ms on Windows for
timer accuracy). Fast mode: no sleeping; vision still generated on sim time.
Status line once per second: sim time, real-time factor, ball, robot count,
substep cost (µs).

## 8. Testing and acceptance

Unit (core): closed-form flat trajectory vs numeric integration; chip
inverses round-trip; collision kernel energy bounds; wheel IK/FK round-trip;
kick speed within 0.10 m/s of command for 2/4/6/8 m/s and 13 m/s clamps to the
spec (ER-Force tolerances); teleport within 0.01 m; robot velocity command
reaches within 2e-2 m/s and 5e-2 m position error over 1.8 s (ideal mode);
determinism: two worlds with the same seed and inputs produce equal
`state_hash()` after 10 000 substeps; camera overlap: the 17 ER-Force
assertions; occlusion: ball behind a robot invisible, half-covered ball
reported shifted away from the robot.

Integration (net): protobuf round trips for every message; legacy grSim
packet mapping; sync request with `sim_step = 0.016` returns 4 detection
frames; vision packet rate over 2 s within ±5 % of configured.

Bench: `cargo bench` on `World::step` with 22 robots; CI threshold 20 µs per
substep on the reference machine.

## 9. Work plan and agent assignment

Phase 0 (lead): repo restructure, skeleton types, this document.
Phase 1 (parallel):
- **Fable** — `core::ball` (trajectory, chip, inverses), `core::collision`
  (sweep + kernel + TOI ordering, robot–robot impulses), `core::robot::drive`
  (wheel model, firmware limiter). With tests.
- **Opus A** — `core::field`, `core::vision` (cameras, projection, occlusion,
  noise, delay queue), `core::rng`, realism presets, formations.
- **Opus B** — `ssl-sim-proto` expansion, `ssl-sim-net` adapters and sockets,
  sync mode, legacy grSim, CLI, config file.
- **Opus C** — `ssl-sim-viewer`: egui 2D viewer over the vision multicast and
  the optional truth stream; teleport / speed / pause via `SimulatorCommand`.
Phase 2 (lead + Opus): integration of `World::step`, acceptance tests, bench,
README, memory notes.

## 10. Implementation status and known gaps (2026-09-07)

Implemented and tested: everything in §3–§8 except the items below. Full
workspace: 144 tests, clippy clean. Throughput on the reference machine:
about 17 µs per substep with 22 commanded robots in release (60× real time
including 73.3 Hz vision on two cameras), 20 µs measured by the running CLI.

Deviations and simplifications, all deliberate:
- `realism.command_delay` is accepted in config but not applied; commands
  take effect at the next substep.
- Ball–robot hull contact uses the exact Minkowski sum (rounded chord
  corners; corner hits use hull damping, not kicker damping).
- Dribbler: forward push is rigid, only pull and lateral components consume
  the traction budget; imparted back-spin is capped at 1 m/s contact speed;
  a held ball may penetrate the chord by `seat_depth`; regrab has 80 %
  hysteresis. Barrier test also requires ball centre below 6 cm.
- `by_force` movers are critically damped springs (ER-Force gains converted
  to continuous stiffness); the ball is forced to rolling spin while dragged.
- Robot–robot collision events fire only when the pair approaches faster
  than 2 cm/s; goal frames act as walls for robots; robots are clamped to
  the room box.
- Spin on contact is mirrored about the contact plane and scaled by
  `spin_retention`, not reflected with angular-momentum transfer.
- Vision: robots are never occluded; positional noise is not a function of
  ball height (only `area` carries it); spurious dribbler balls skip the
  occlusion re-test; the *advertised* `camera_position_error` is still a
  deterministic offset (the error a client can see is the drawn per-camera
  calibration warp instead); the geometry packet still advertises a fixed
  (300, 300) principal point, zero distortion and no image size, and one focal
  length for the whole rig, while real rigs mix sensors; capture instants are
  quantised to the substep (±0.5 ms at the 1 ms default).
- Goal posts are zero-thickness double-sided segments on the post centre line.
- `TeleportRobot` with position but no orientation is accepted (ER-Force
  rejects it) because the Immortals `Software` client sends that.
- Sync mode snaps `sim_step` to the nearest whole substep (f32 on the wire).

