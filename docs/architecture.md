# Architecture

## Goals

- Start with true 3D physics because chipped balls are part of normal SSL play.
- Keep the simulator embeddable by making the physics core independent from UDP,
  protobuf generation, rendering, and process management.
- Use the SSL simulation protocol as the boundary for teams, tools, and future UI
  clients.
- Prefer simple data flow over a large framework.

## Shape

```text
ssl-sim-core
  3D physics world, fixed stepping, robot commands, teleports, snapshots

ssl-sim-proto
  generated protobuf bindings from ssl-simulation-protocol

ssl-sim-cli
  UDP sockets on the standard ports, protobuf translation, realtime or fast loop

future optional UI
  separate process or crate that uses the same protocol messages
```

## Physics

The field uses SI units internally:

- x/y: field plane in meters
- z: height in meters
- gravity: negative z

`rapier3d` is used from the start. The first model is intentionally simple:

- the ball is a dynamic 3D sphere
- robots are kinematic 3D bodies driven by protocol velocity commands
- straight and chip kicks set the ball velocity from robot orientation and kick
  angle
- snapshots project 3D state into protocol-facing data when needed

This keeps chipped balls native to the physics state instead of adding a second
ball model later.

## Time

The core only exposes `step(dt_seconds)`. Callers choose the clock:

- realtime mode accumulates wall-clock time and steps fixed slices
- fast mode repeatedly applies the configured fixed step without sleeping
- tests can call `step` directly for deterministic simulation

## Protocol Boundary

The protocol adapter should stay thin:

- decode protobuf command
- translate to `ssl-sim-core` command or teleport
- step the world elsewhere in the loop
- encode protobuf response

Unsupported protocol features should return `SimulatorError` with stable error
codes instead of silently pretending to work.
