# Immortals SSL Simulator

A small, embeddable RoboCup Small Size League simulator written in Rust.

The project starts with a headless 3D physics core and a CLI UDP adapter for the
standard SSL simulation protocol. GUI, 2D, and 3D views should stay optional and
talk to the simulator through the same protocol messages wherever practical.

## Crates

- `ssl-sim-core`: embeddable 3D simulation core using SI units.
- `ssl-sim-proto`: generated Rust bindings for the SSL simulation protocol.
- `ssl-sim-cli`: headless command line simulator with standard UDP endpoints.

## Protocols

The simulator vendors the official SSL simulation protocol `.proto` files under
`protocol/ssl-simulation-protocol/proto`, so a normal checkout can build without
extra setup:

```powershell
cargo check
```

To test against another protocol checkout, override the proto directory:

```powershell
$env:SSL_SIMULATION_PROTO_DIR = "D:\dev\Robotics\Others\SSL\simulation-protocol\proto"
cargo check
```

## CLI

Run the headless simulator with the default SSL ports:

```powershell
cargo run -p ssl-sim-cli -- run
```

Default UDP endpoints:

- simulation control: `0.0.0.0:10300`
- blue robot control: `0.0.0.0:10301`
- yellow robot control: `0.0.0.0:10302`
- vision multicast output: `224.5.23.2:10020`

The core supports both wall-clock stepping and fixed-step fast-forward. The CLI
currently exposes this as:

```powershell
cargo run -p ssl-sim-cli -- run --mode realtime
cargo run -p ssl-sim-cli -- run --mode fast --step-ms 2
```

By default the simulator starts with 11 blue and 11 yellow robots in a symmetric
in-field formation based on grSim's inside formation.
