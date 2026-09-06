# ssl-sim-viewer

A minimal 2D viewer for the Immortals SSL simulator, built with `egui` /
`eframe`. It is a pure protocol client: it listens to the SSL vision multicast
and the optional ground-truth tracker stream, and sends standard
`SimulatorCommand`s to the simulator's control port. Nothing links against
`ssl-sim-core`, so the viewer also works against grSim and ER-Force's
simulator.

## Usage

```sh
cargo run -p ssl-sim-viewer --release
```

Flags (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--vision ADDR:PORT` | `224.5.23.2:10020` | Vision stream to subscribe to |
| `--truth ADDR:PORT` | `224.5.23.2:10010` | Ground-truth `TrackerWrapperPacket` stream |
| `--no-truth` | off | Do not subscribe to the tracker stream at all |
| `--control HOST:PORT` | `127.0.0.1:10300` | Where `SimulatorCommand`s are sent |

Examples:

```sh
# Simulator publishing vision on localhost instead of multicast
cargo run -p ssl-sim-viewer -- --vision 127.0.0.1:10020 --no-truth

# Watch a remote simulator, control it locally over the network
cargo run -p ssl-sim-viewer -- --control 10.0.0.5:10300
```

Multicast sockets are bound to `0.0.0.0:PORT` with `SO_REUSEADDR` and then
join the group, so the viewer coexists with the team software (Tyr, the older
`Software` tree, ssl-vision clients) already listening on the same port.

## What it shows

- **Field** drawn from the geometry packet: lines, arcs, goals and the run-off
  boundary, fitted to the window (metres to pixels, y up). Until a geometry
  packet arrives a default Division A field is drawn.
- **Robots** as discs with the flat front cut at 0.075 m from the centre,
  rotated by the detected `orientation`, blue/yellow, with the robot id.
- **Balls** as orange dots; the dot grows with the reported `area` and fades
  when `confidence < 0.5`.
- **Ground truth** (when enabled) as thin magenta outlines on top; a ball off
  the ground gets a second ring scaled by its height.
- **Side panel**: per-camera packet rate and totals, connection status, the
  truth ball's position and height, view toggles, the simulation-speed slider
  and Pause/Resume.

Detections are kept per camera and merged for drawing; a camera that stops
sending for 1.5 s drops out of the merged view and greys out in the panel.
Enable "show camera ids" to see which camera reported each object.

## Mouse and keyboard

| Gesture | Effect |
|---|---|
| Left-drag on the ball | `TeleportBall{x, y}` streamed at 60 Hz, `teleport_safely = false` |
| Right-drag on a robot | `TeleportRobot{id, x, y}` streamed at 60 Hz, orientation preserved |
| Scroll wheel over a robot | Rotates it in place (`TeleportRobot` with the new orientation) |
| Middle-click on the field | Teleports the ball there with zero velocity |
| Speed slider / Pause / Resume | `SimulatorCommand{control{simulation_speed}}` (Pause = 0, Resume = 1) |

The robot under the cursor is outlined in white; that is the robot the scroll
wheel rotates. Commands are fire-and-forget UDP; send failures are shown in
the panel rather than taken as fatal, so the viewer can be started before the
simulator.

## Implementation notes

- One blocking receiver thread per stream (`std::net::UdpSocket`, no async
  runtime). Each decodes with `prost` into a shared `Arc<Mutex<State>>` and
  calls `Context::request_repaint`, so new data shows up immediately.
- The UI takes one snapshot of the state per frame so the mutex is never held
  while painting, and asks for a repaint every ~16 ms.
- Wire units are kept as they arrive: detections stay in millimetres, tracker
  frames in metres. `view::Transform` does the single conversion to pixels.

## Tests

```sh
cargo test -p ssl-sim-viewer
```

Covers the coordinate transform (fit, y-up orientation, screen/world round
trip, length scaling), the robot outline's flat front, the robot hit test, and
decoding a hand-built `SSL_WrapperPacket` off the wire into `State`.

An opt-in smoke test joins the real vision multicast group and checks a
locally sent packet arrives:

```sh
cargo test -p ssl-sim-viewer -- --ignored
```
