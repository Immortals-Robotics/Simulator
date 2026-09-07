# Immortals SSL Simulator

A small, fast, deterministic RoboCup Small Size League simulator written in
Rust. It is a drop-in replacement for grSim for the Immortals software: the
standard SSL simulation protocol on 10300/10301/10302, vision multicast on
224.5.23.2:10020, and the legacy grSim packet protocol on 20011.

The design document is [`docs/design.md`](docs/design.md); it is the
specification this code is written against.

## Crates

| Crate | What it is |
|---|---|
| `ssl-sim-core` | The simulation library. SI units, no I/O, no threads, no protobuf. |
| `ssl-sim-proto` | Generated protobuf bindings: `sim`, `grsim`, `tracked`. |
| `ssl-sim-net` | Protocol adapters, UDP endpoints, vision publisher, sync mode, the run loop. |
| `ssl-sim-cli` | The `ssl-sim` binary. |
| `ssl-sim-viewer` | Optional 2D viewer; it only consumes the network output. |

Dependency direction is `cli → net → {core, proto}`.

## Quick start

```powershell
cargo run -p ssl-sim-cli --release -- run
```

That starts Division A with 11 robots per team, 1 ms physics substeps, the
`realistic` realism preset (the measured 2026 one), a two-camera 73.3 Hz rig,
vision on the multicast group and the legacy grSim listener enabled.

```powershell
# Everything on loopback, no noise, as fast as the machine allows.
cargo run -p ssl-sim-cli --release -- run --localhost --realism none --mode fast

# Lock-step: the world only advances on SimulationSyncRequest.
cargo run -p ssl-sim-cli --release -- run --mode sync

# Print the built-in defaults as a config file you can edit.
cargo run -p ssl-sim-cli -- presets > sim.toml
```

## CLI

```
ssl-sim run [--config sim.toml] [--division a|b] [--robots N]
            [--mode realtime|fast|sync] [--speed X] [--step-ms 1]
            [--realism none|realistic|go26|rc26|erforce_friendly|
                       erforce_realistic|erforce_rc2021|<file.toml>]
            [--ball-preset default|go26|rc26|erforce|tigers|grsim]
            [--robot-limits default|tigers|erforce|kiks|fast|grsim]
            [--cameras 1|2|4]
            [--seed N|random] [--vision-addr HOST:PORT] [--localhost]
            [--truth] [--no-legacy-grsim] [--duration SECS]
            [--log-level info]
ssl-sim presets
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | — | TOML file deserialised into `SimConfig`; every key optional. |
| `--division` | `a` | `a` = 12 × 9 m, `b` = 9 × 6 m. |
| `--robots` | `11` | Robots per team at start, 0..=16. |
| `--mode` | `realtime` | See [Modes](#modes). |
| `--speed` | `1.0` | Real-time scaling. `0` pauses; `SimulatorControl.simulation_speed` changes it at runtime. |
| `--step-ms` | `1` | Physics substep in milliseconds. |
| `--realism` | `realistic` | Preset name or a TOML file with the `Realism` keys at the top level. |
| `--ball-preset` | `default` | Replaces the whole `[ball]` table with a named `BallParams` preset. |
| `--robot-limits` | `default` | Replaces `limits` in both teams' default robot specs with a named `RobotLimits` preset. |
| `--cameras` | `2` | Cameras to auto-place (1, 2 or 4). Clears any explicit `[[vision.cameras]]` from the config file. |
| `--seed` | `0` | RNG seed. `random` derives one from the wall clock. |
| `--vision-addr` | `224.5.23.2:10020` | Vision destination. |
| `--localhost` | off | Publish vision to `127.0.0.1` instead of the multicast group. |
| `--truth` | off | Also publish the ground-truth tracker stream on port 10010. |
| `--no-legacy-grsim` | off | Do not bind the legacy grSim port 20011. |
| `--duration` | — | Stop after N seconds (handy for benchmarks and CI). |
| `--log-level` | `info` | `error`/`warn`/`info`/`debug`/`trace` or an `RUST_LOG`-style filter. `RUST_LOG` wins if set. |

`ssl-sim presets` writes the full default `SimConfig` as TOML on stdout, with
every preset name and a one-line description of it in the header comment.
[`config/example.toml`](config/example.toml) is the same content, annotated
with where each measured value comes from.

Once a second the runner logs a status line:

```
INFO sim 12.35s | rtf 1.00x | substep 18.4us | robots 22 | mode realtime
```

## Ports and protocols

| Purpose | Transport | Default | Message in | Message out |
|---|---|---|---|---|
| Simulation control | UDP, bound on `0.0.0.0` | `10300` | `SimulatorCommand` or `SimulationSyncRequest` | `SimulatorResponse` / `SimulationSyncResponse` |
| Blue robot control | UDP | `10301` | `RobotControl` or `SimulationSyncRequest` | `RobotControlResponse` / `SimulationSyncResponse` |
| Yellow robot control | UDP | `10302` | same, for yellow | same |
| Legacy grSim | UDP | `20011` | `grSim_Packet` | — |
| Legacy status, blue | UDP | `30011` | — | `Robots_Status` to the sender's IP |
| Legacy status, yellow | UDP | `30012` | — | `Robots_Status` to the sender's IP |
| Vision | UDP multicast, TTL 1 | `224.5.23.2:10020` | — | `SSL_WrapperPacket` |
| Ground truth (`--truth`) | UDP multicast, TTL 1 | `224.5.23.2:10010` | — | `TrackerWrapperPacket` |

Every reply goes to the **exact source address of the datagram that caused
it**, so several clients can talk to the simulator without stealing each
other's responses. The vision port is changeable at runtime through
`SimulatorConfig.vision_port`.

Both the control port and the team ports accept a `SimulationSyncRequest` in
place of their normal message; the runner tells them apart from the protobuf
tags. On a team port the request's `robot_control` is attributed to that team;
on the control port there is no team to attribute it to, so a `robot_control`
there is answered with `UNSUPPORTED`.

### Units on the wire

| Message | Wire | Core |
|---|---|---|
| `TeleportBall`, `TeleportRobot`, `RobotControl`, `RobotSpecs` | metres, m/s, rad, rpm, degrees for `kick_angle` | same |
| `SSL_DetectionFrame` / `SSL_GeometryData` | millimetres (ball model constants in m/s²) | metres |
| `RealismConfigErForce.vision_delay`, `vision_processing_time` | nanoseconds | seconds |
| `TrackerWrapperPacket` | metres, m/s | same |
| legacy `grSim_Packet` | metres, degrees for `dir`, **rad/s** for `wheel1..4` | metres, radians, m/s |

`RobotWheelAngles` is documented **clockwise** by the protocol. The core stores
CCW mounting angles, so `ssl-sim-net` negates every angle on the way in and out.

### Errors

Responses carry `SimulatorError { code, message }` with stable codes:
`UNREADABLE`, `PARTIAL_COORD`, `VELOCITY_FORCE`, `TELEPORT_SAFELY_PARTIAL`,
`CREATE_NOPOS_ROBOT`, `INVALID_SPEC`, `UNKNOWN_ROBOT`, `INVALID_STEP`,
`UNSUPPORTED`. Per-robot errors go to the owning team's socket. Unknown robot
ids in a `RobotControl` are reported but do not stop the rest of the message
from being applied.

Deviation from ER-Force, on purpose: a `TeleportRobot` with `x` and `y` but no
`orientation` is accepted (ER-Force answers `PARTIAL_COORD`). The Immortals
clients teleport that way.

### Legacy grSim mapping

| grSim field | Maps to |
|---|---|
| `isteamyellow` | team of the whole `grSim_Commands` block |
| `veltangent` / `velnormal` / `velangular` | `MoveLocalVelocity.forward` / `.left` / `.angular` |
| `kickspeedx`, `kickspeedz` | `kick_speed = hypot(x, z)`, `kick_angle = atan2(z, x)` in degrees; both ≈ 0 means no kick |
| `spinner` | dribbler at the robot's `max_speed_rpm` (10 000 if unknown), else off |
| `wheelsspeed` + `wheel1..4` | wheel velocities, rad/s → m/s with the wheel radius |
| `grSim_BallReplacement` | ball teleport, metres |
| `grSim_RobotReplacement` | robot teleport, metres and degrees; `turnon` → `present` |

**Wheel order.** grSim builds its wheels at mounting angles 60°, 135°, 225°,
300° measured CCW from forward, so `wheel1..4` are front-left, back-left,
back-right, front-right. The SSL simulation protocol orders wheels
`(front_right, back_right, back_left, front_left)` — exactly the reverse — so
`front_right = wheel4`, `back_right = wheel3`, `back_left = wheel2`,
`front_left = wheel1`.

`Robots_Status` is sent to the legacy sender's IP address on 30011 (blue) /
30012 (yellow) whenever a robot's status changes: `infrared` is the break-beam
ball contact, and `flat_kick` / `chip_kick` stay set for 10 vision frames after
a kick (grSim's 10-frame counter).

## Modes

| Mode | Behaviour |
|---|---|
| `realtime` | Accumulates wall time × `speed`, runs whole substeps, publishes due vision packets, then sleeps the remainder. The last ~0.5 ms is spin-waited because Windows timers are only millisecond-accurate. Catch-up is capped at 250 ms so a stall cannot spiral. |
| `fast` | Never sleeps. Vision is still generated on simulation time, so packets stay at the configured rate *in simulation seconds* and simply arrive sooner. |
| `sync` | The world only advances on `SimulationSyncRequest`. Each request applies `simulator_command`, applies `robot_control`, steps `sim_step` (which must be a whole number of substeps, else `INVALID_STEP`), and answers with every detection frame the step released plus the robot control response. |

In `realtime` and `fast` the free-running clock is paused while a sync request
has been seen within the last second, so a lock-step client can take over a
running simulator without racing it.

## Configuration

The config file is a `SimConfig`; see `config/example.toml` for a fully
commented copy of the defaults. Flags override the file
(`--robots`, `--step-ms`, `--seed`, `--realism`).

Everything except the substep, the seed and the initial robot count can also be
changed at runtime over the control port:

* `SimulatorConfig.geometry` — field size, goals, boundary and penalty area.
  The collision geometry is rebuilt; the ball and the robots are **not** moved.
  The penalty area comes from `penalty_area_depth` / `penalty_area_width` when
  present, otherwise it is derived from the penalty stretch lines.
* `SimulatorConfig.robot_specs` — partial specs are allowed; only fields that
  are actually set on the wire override the current value. The ER-Force
  `RobotSpecErForce` custom message (`shoot_radius`, `dribbler_width`) is
  unpacked; unknown `custom` types are ignored. A spec for robot N of a team is
  applied to that robot *and* becomes the team default for robots created
  later; if the robot does not exist only the default changes.
* `SimulatorConfig.realism_config` — a `RealismConfigErForce` packed into
  `custom`. Unknown custom types are reported with `UNSUPPORTED` and ignored.
* `SimulatorConfig.vision_port` — moves the vision publisher.
* `SimulatorControl.simulation_speed` — real-time scaling; `0` pauses.

### Realism presets

`--realism` accepts a preset name or a path to a TOML file with the `Realism`
keys at the top level.

**`realistic` is now the measured 2026 preset, not ER-Force's numbers.** It is
fitted to ten division-A/B game logs (German Open 2026 + RoboCup 2026, 6.6 h of
play, 3.2 M detection frames, 31.4 M robot detections) — see
[`docs/calibration/vision.md`](docs/calibration/vision.md) §8 for the table and
§9 for the model changes the data forced. ER-Force's own presets are still
here, prefixed `erforce_`, for clients that were tuned against their simulator.

| Preset | What it is |
|---|---|
| `none` | No noise, no loss, no calibration error. |
| `realistic` | **Default.** Pooled 2026 measurement. |
| `go26` | German Open 2026 venue: no `area` on the wire, many dribbler-LED false balls, warped rather than offset cameras. |
| `rc26` | RoboCup 2026 venue: `area` reported, few false balls, a 2–2.6 cm constant offset between the cameras. |
| `erforce_friendly` (`friendly`) | ER-Force "Friendly". |
| `erforce_realistic` | ER-Force "Realistic" — what our `realistic` used to be. |
| `erforce_rc2021` (`rc2021`) | ER-Force "RC2021", glued dribbler. |

| Field | none | realistic (measured) | go26 | rc26 | erforce_realistic |
|---|---|---|---|---|---|
| `stddev_ball_p` [m] | 0 | **0.0007** | 0.0007 | 0.0007 | 0.0014 |
| `stddev_robot_p` [m] | 0 | **0.0005** | 0.0005 | 0.0005 | 0.0013 |
| `stddev_robot_phi` [rad] | 0 | **0.005** | 0.005 | 0.005 | 0.01 |
| `stddev_ball_area` [px] | 0 | **3.3** | 3.3 | 3.3 | 6.5 |
| `camera_overlap` [m] | 0.3 | **0.8** | 0.8 | 0.8 | 1.0 |
| `dribbler_ball_detections` [1/s/robot] | 0 | **0.02** | **0.06** | **0.0015** | 0.05 |
| `camera_position_error` [m] | 0 | **0.02** | 0.02 | 0.02 | 0.1 |
| `robot_command_loss` | 0 | 0.01 | 0.01 | 0.01 | 0.03 |
| `robot_response_loss` | 0 | 0.01 | 0.01 | 0.01 | 0.1 |
| `missing_ball_detections` | 0 | **0.007** | 0.007 | 0.007 | 0.05 |
| `missing_robot_detections` | 0 | **0.002** | 0.002 | 0.002 | 0.02 |
| `vision_delay` [s] | 0.035 | **0.022** | 0.022 | 0.022 | 0.035 |
| `vision_processing_time` [s] | 0.005 | **0.0073** | 0.0073 | 0.0073 | 0.010 |
| `simulate_dribbling` | true | true | true | true | true |
| `object_position_offset` [m] | 0 | **0.02** | **0.005** | **0.023** | 0.02 |
| `calibration_warp_stddev` [m] | 0 | **0.012** | **0.014** | **0.009** | 0 |
| `calibration_warp_length` [m] | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| `calibration_orientation_offset` [rad] | 0 | **0.02** | 0.02 | 0.02 | 0 |
| `calibration_orientation_warp` [rad] | 0 | **0.02** | 0.02 | 0.02 | 0 |
| `kick_direction_stddev` [rad] | 0 | **0.044** (2.5°) | 0.044 | 0.044 | 0 |
| `chip_angle_stddev` [rad] | 0 | **0.105** (6°) | 0.105 | 0.105 | 0 |
| `kick_speed_factor_stddev` | 0 | **0.10** | 0.10 | 0.10 | 0 |

The headline of the calibration study is in the last block: **each camera's
per-frame noise is 0.4 mm, but two cameras place the same robot 20 mm apart**
(28.9 mrad in orientation). That static error — a constant per-camera offset
plus a smooth spatially varying warp — is what a team's world model actually
has to cope with, and it is two orders of magnitude larger than the white
noise. `object_position_offset` carries the constant part and
`calibration_warp_*` the varying part; both are drawn once per camera from the
seed and never change during a run.

`robot_command_loss` and `robot_response_loss` are applied by the network layer
using the world's seeded `packet_loss` stream, so a run with a fixed seed stays
reproducible. `command_delay` is present in the config but **not implemented
yet**: commands take effect at the next substep regardless of its value. The
ER-Force `RealismConfigErForce` wire message only covers the first 17 keys; the
calibration-warp and kick keys are config-file only.

### Vision model

The camera rig and everything a camera reports live in `[vision]`; the defaults
are measured (`docs/calibration/vision.md` §2, §8):

| Knob | Default | Why |
|---|---|---|
| `default_camera_count` | `2` | Every division-A field in the corpus used exactly two cameras; division B used one. |
| `default_camera_height` | `6.4` m | Measured 5.97–6.52 m in every log (we used to assume 4.0). |
| `default_camera_x_fraction` | `0.20` | Two-camera rigs split along x only, at ±2.4 m on a 12 m field — inside the classic ±L/4. |
| `fov_radius` | `6.6` m | Detection completeness is flat to 6.5 m from the nadir and then falls off a cliff (22 % missed at 6.75 m, 92 % at 7.25 m). The Manhattan region rule alone reaches 8.9 m, so cameras used to report objects real ones never see. `0` disables it. |
| `frame_rate` | `73.3` Hz | 73.11–73.29 Hz on all 19 streams. |
| `camera_phase` / `phase_offsets` | `locked`, all 0 | Real rigs are either hardware-locked with per-camera offsets up to 5.8 ms (RoboCup 2026) or free-running with a uniform phase, σ = T/√12 = 3.94 ms (German Open). |
| `geometry_every_n_frames` | `73` | 1.00 s, the German Open cadence (RoboCup 2026 used 1.50 s ⇒ 110). |
| `area_at_nadir_px` | `63.0` px | Measured mean blob area of a ball on the floor. |
| `focal_length_px` | `1420` | The real calibrated focal length; with `PIXEL_PER_AREA = 1` the reported `area` is simply the blob's pixel area. |
| `report_area` | `true` | The German Open reported **no** `area` on any of its 941 367 ball detections. Set `false` to exercise that path. |
| `report_ball_z` | `false` | 0 of 1 820 478 real ball detections carried `z`. |
| `spurious_ball_forward` / `_lateral_stddev` / `_area_px` | `0.13` m / `0.07` m / `32` px | False "dribbler balls" sit on the robot's **centreline** 0.12–0.14 m ahead of its centre (an IR break beam firing into the camera), and are 2.6× smaller than a real ball. |
| `robot_confidence_mean` / `ball_confidence_mean` / `confidence_stddev` | `0.9` / `0.9` / `0.05` | Real confidences average 0.896 and 0.913; we used to emit a constant 1.0. |
| `duplicate_robot_rate` | `3e-5` | 932 duplicate ids in 31.4 M detections. Rare, but a consumer that assumes uniqueness per frame breaks on real logs. |
| `outages` | none | Nine of ten games contain a 303–421 s vision outage. Script one with `[[vision.outages]] start = …, duration = …`; frame numbers keep counting through it and geometry is re-sent on the first frame after. |

Two consequences worth knowing as a client:

* **One `SSL_WrapperPacket` per camera capture, not per rig.** Each camera has
  its own capture instant (that is the point of `camera_phase`), so packets
  from different cameras carry different `t_capture`. The geometry packet rides
  on camera 0's packets.
* **The ball `area` model has no horizontal distance term.** Measured `area` is
  flat to ±8 % from the nadir out to 5 m, where a pinhole 1/d² predicts a 37 %
  fall — a flat sensor's obliquity nearly cancels the range term. Height still
  scales it, slightly steeper than an ideal pinhole, which is what the logs
  show.

The advertised ball model is the simulated one with one translation: the wire's
`SSL_BallModelStraightTwoPhase` has a single constant rolling deceleration,
while the core rolls with a speed-dependent one, so the geometry packet carries
`acc_roll - roll_speed_coefficient * advertise_roll_speed`
(`BallParams::advertised_acc_roll()`, −0.2875 m/s² by default) together with
`k_switch` and the first-hop chip damping factors.

### Determinism

`--seed N` (default `0`) seeds every random stream. Same seed plus same inputs
produce identical output, including packet loss. `--seed random` opts out.

## Pointing the Immortals software at it

Both clients speak the protocols this simulator implements without any change.

**Tyr** (`source/sender/simulator.cpp`, `grsim.cpp`) — set in Tyr's config:

```
network.blue_robot_simulation_address   = 127.0.0.1:10301
network.yellow_robot_simulation_address = 127.0.0.1:10302
network.grsim_address                   = 127.0.0.1:20011
network.vision_address                  = 224.5.23.2:10020
```

Tyr's `simulator.cpp` sends `MoveLocalVelocity`, `kick_angle` 0 for straight
kicks and 45 for chips, and a dribbler speed in rpm — all handled. Its
`grsim.cpp` sends the legacy `grSim_Packet` to 20011; run without
`--no-legacy-grsim` (the default) to keep that path working.

**Software** (`source/soccer/output/ssl_simulator/ssl_simulator.cpp`) — set:

```
control_simulation_address       = 127.0.0.1:10300
blue_robot_simulation_address    = 127.0.0.1:10301
yellow_robot_simulation_address  = 127.0.0.1:10302
```

It teleports robots with `x`/`y` and `by_force` but no orientation; that is
accepted here.

If the clients run on the same machine, start the simulator with `--localhost`
so vision goes to `127.0.0.1:10020` instead of the multicast group, and point
the clients' vision address at `127.0.0.1:10020` too. Otherwise leave the
multicast default; TTL is 1, so traffic stays on the local link.

## Building and testing

The official SSL simulation protocol `.proto` files are vendored under
`protocol/`, so a normal checkout builds without extra setup:

```powershell
cargo check
cargo test
cargo clippy --all-targets
```

To build against another protocol checkout:

```powershell
$env:SSL_SIMULATION_PROTO_DIR = "D:\dev\Robotics\Others\SSL\simulation-protocol\proto"
cargo check
```

The network tests cover protobuf round trips for every message, the proto ↔
core conversions (units, wheel-angle negation, spec merging, realism unpacking,
the vision packet and the advertised ball model), the legacy grSim mapping, and
end-to-end UDP request/response against a running simulator in sync mode. Tests
that need a *stepping* world — the vision rate over real sockets, sync-mode
frame counts, kick events — live in `crates/ssl-sim-net/tests/live.rs`:

```powershell
cargo test -p ssl-sim-net --test live
```

The vision model's own tests (field-of-view clipping, camera phase, the static
calibration warp, spurious balls, the area model, confidences, duplicate ids,
outages) are in `ssl-sim-core`:

```powershell
cargo test -p ssl-sim-core --lib -- vision
```
