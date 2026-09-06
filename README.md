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
`realistic` realism preset, vision on the multicast group and the legacy grSim
listener enabled.

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
            [--realism none|friendly|realistic|rc2021|<file.toml>]
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
| `--seed` | `0` | RNG seed. `random` derives one from the wall clock. |
| `--vision-addr` | `224.5.23.2:10020` | Vision destination. |
| `--localhost` | off | Publish vision to `127.0.0.1` instead of the multicast group. |
| `--truth` | off | Also publish the ground-truth tracker stream on port 10010. |
| `--no-legacy-grsim` | off | Do not bind the legacy grSim port 20011. |
| `--duration` | — | Stop after N seconds (handy for benchmarks and CI). |
| `--log-level` | `info` | `error`/`warn`/`info`/`debug`/`trace` or an `RUST_LOG`-style filter. `RUST_LOG` wins if set. |

`ssl-sim presets` writes the full default `SimConfig` as TOML on stdout.
[`config/example.toml`](config/example.toml) is the same content with comments.

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

`--realism` accepts `none`, `friendly`, `realistic` (default), `rc2021`, or a
path to a TOML file with the `Realism` keys at the top level. The presets carry
ER-Force's values:

| Field | none | friendly | realistic | rc2021 |
|---|---|---|---|---|
| `stddev_ball_p` [m] | 0 | 0.0004 | 0.0014 | 0.0010 |
| `stddev_robot_p` [m] | 0 | 0.0003 | 0.0013 | 0.0013 |
| `stddev_robot_phi` [rad] | 0 | 0.003 | 0.01 | 0.01 |
| `stddev_ball_area` [px] | 0 | 1 | 6.5 | 6.5 |
| `camera_overlap` [m] | 0.3 | 1 | 1 | 1 |
| `dribbler_ball_detections` [1/s/robot] | 0 | 0.001 | 0.05 | 0.02 |
| `camera_position_error` [m] | 0 | 0.05 | 0.1 | 0 |
| `robot_command_loss` | 0 | 0.01 | 0.03 | 0.03 |
| `robot_response_loss` | 0 | 0.01 | 0.1 | 0.1 |
| `missing_ball_detections` | 0 | 0.05 | 0.05 | 0.05 |
| `missing_robot_detections` | 0 | 0.02 | 0.02 | 0 |
| `vision_delay` [s] | 0.035 | 0.035 | 0.035 | 0.035 |
| `vision_processing_time` [s] | 0.005 | 0.010 | 0.010 | 0.010 |
| `simulate_dribbling` | true | true | true | **false** (glue) |
| `object_position_offset` [m] | 0 | 0.02 | 0.02 | 0 |

`robot_command_loss` and `robot_response_loss` are applied by the network layer
using the world's seeded `packet_loss` stream, so a run with a fixed seed stays
reproducible. `command_delay` is present in the config but **not implemented
yet**: commands take effect at the next substep regardless of its value.

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
core conversions (units, wheel-angle negation, spec merging, realism unpacking),
the legacy grSim mapping, and end-to-end UDP request/response against a running
simulator in sync mode. Tests that need a *stepping* world live in
`crates/ssl-sim-net/tests/live.rs` and are `#[ignore]`d until the core physics
modules land:

```powershell
cargo test -p ssl-sim-net --test live -- --ignored
```
