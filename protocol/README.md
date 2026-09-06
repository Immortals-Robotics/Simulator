# Protocol Files

Vendored protobuf definitions, so a plain checkout builds without submodules.

| Directory | Source | Notes |
|---|---|---|
| `ssl-simulation-protocol/proto` | RoboCup-SSL/ssl-simulation-protocol (2026-04) | control, config, robot control/feedback, sync, vision wrapper |
| `erforce` | same repo, `proto/erforce` | `RealismConfigErForce`, `RobotSpecErForce` (packed in `Any`) |
| `grsim` | RoboCup-SSL/grSim `src/proto` | legacy `grSim_Packet` / `Robots_Status` |
| `ssl-vision` | RoboCup-SSL/ssl-vision `src/shared/proto` | tracked-frame packets for the ground-truth stream |

Set `SSL_SIMULATION_PROTO_DIR` to point at another checkout of the simulation
protocol when testing against a newer draft.
