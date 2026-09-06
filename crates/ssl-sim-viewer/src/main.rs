//! 2D viewer for the SSL simulator. OWNER: general agent C.
//!
//! Subscribes to the vision multicast (and optionally the ground-truth
//! tracker stream), draws field / robots / ball with egui, and sends
//! `SimulatorCommand`s (teleport, speed, pause) to the control port.
//!
//! The viewer is a pure client of the standard protocols: it never talks to
//! `ssl-sim-core` directly, so it works against grSim and ER-Force's
//! simulator too.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod app;
mod net;
mod state;
mod view;

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use clap::Parser;

use crate::{app::ViewerApp, net::ControlSender, state::State};

/// Minimal 2D viewer for the RoboCup SSL simulator.
#[derive(Debug, Parser)]
#[command(name = "ssl-sim-viewer", version, about)]
struct Args {
    /// Vision multicast group and port to listen on.
    #[arg(long, value_name = "ADDR:PORT", default_value = "224.5.23.2:10020")]
    vision: SocketAddr,

    /// Ground-truth tracker multicast group and port.
    #[arg(long, value_name = "ADDR:PORT", default_value = "224.5.23.2:10010")]
    truth: SocketAddr,

    /// Do not subscribe to the ground-truth tracker stream.
    #[arg(long)]
    no_truth: bool,

    /// Simulator control endpoint for `SimulatorCommand`s.
    #[arg(long, value_name = "HOST:PORT", default_value = "127.0.0.1:10300")]
    control: SocketAddr,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let truth_addr = (!args.no_truth).then_some(args.truth);

    let state = Arc::new(Mutex::new(State::default()));
    let control = ControlSender::new(args.control)?;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 400.0])
            .with_title("ssl-sim-viewer"),
        ..Default::default()
    };

    let vision_addr = args.vision;
    let thread_state = Arc::clone(&state);
    eframe::run_native(
        "ssl-sim-viewer",
        options,
        Box::new(move |cc| {
            // Receivers wake the UI so it repaints as soon as data lands.
            let wake_ctx = cc.egui_ctx.clone();
            net::spawn_vision(vision_addr, Arc::clone(&thread_state), move || {
                wake_ctx.request_repaint();
            })?;
            if let Some(addr) = truth_addr {
                let wake_ctx = cc.egui_ctx.clone();
                net::spawn_truth(addr, Arc::clone(&thread_state), move || {
                    wake_ctx.request_repaint();
                })?;
            }
            Ok(Box::new(ViewerApp::new(
                thread_state,
                control,
                vision_addr,
                truth_addr,
            )))
        }),
    )
    .map_err(|err| anyhow::anyhow!("eframe: {err}"))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use prost::Message as _;
    use ssl_sim_proto::sim::{
        SslDetectionBall, SslDetectionFrame, SslDetectionRobot, SslFieldLineSegment,
        SslGeometryData, SslGeometryFieldSize, SslWrapperPacket, Vector2f,
    };

    use crate::{state::State, view::FieldGeometry};

    fn robot(id: u32, x: f32, y: f32, orientation: f32) -> SslDetectionRobot {
        SslDetectionRobot {
            confidence: 1.0,
            robot_id: Some(id),
            x,
            y,
            orientation: Some(orientation),
            pixel_x: 0.0,
            pixel_y: 0.0,
            height: Some(0.15),
        }
    }

    /// Build a wrapper packet by hand, put it through the wire encoding, and
    /// check that the receive path lands it in the shared state.
    #[test]
    fn decodes_a_wrapper_packet_into_the_state() {
        let packet = SslWrapperPacket {
            detection: Some(SslDetectionFrame {
                frame_number: 42,
                t_capture: 1.5,
                t_sent: 1.51,
                camera_id: 2,
                balls: vec![SslDetectionBall {
                    confidence: 0.9,
                    area: Some(120),
                    x: -1500.0,
                    y: 250.0,
                    z: None,
                    pixel_x: 0.0,
                    pixel_y: 0.0,
                }],
                robots_blue: vec![robot(3, 1000.0, -500.0, 0.25)],
                robots_yellow: vec![robot(7, -2000.0, 750.0, -1.0)],
            }),
            geometry: Some(SslGeometryData {
                field: SslGeometryFieldSize {
                    field_length: 12000,
                    field_width: 9000,
                    goal_width: 1800,
                    goal_depth: 180,
                    boundary_width: 300,
                    field_lines: vec![SslFieldLineSegment {
                        name: "HalfwayLine".to_owned(),
                        p1: Vector2f { x: 0.0, y: -4500.0 },
                        p2: Vector2f { x: 0.0, y: 4500.0 },
                        thickness: 10.0,
                        r#type: None,
                    }],
                    field_arcs: vec![],
                    penalty_area_depth: Some(1800),
                    penalty_area_width: Some(3600),
                },
                calib: vec![],
                models: None,
            }),
            source: None,
        };
        let bytes = packet.encode_to_vec();

        let decoded = SslWrapperPacket::decode(bytes.as_slice()).expect("decode");
        let now = Instant::now();
        let mut state = State::default();
        state.apply_wrapper(decoded, now);

        // Geometry is latched and converted to metres for drawing.
        let field = FieldGeometry::from_packet(state.geometry.as_ref().expect("geometry"));
        assert_eq!(field.length, 12.0);
        assert_eq!(field.width, 9.0);
        assert_eq!(field.goal_width, 1.8);
        assert_eq!(field.boundary_width, 0.3);
        assert_eq!(state.geometry.as_ref().unwrap().field.field_lines.len(), 1);

        // The detection frame is filed under its camera id with a live counter.
        let camera = state.cameras.get(&2).expect("camera 2");
        assert_eq!(camera.frame.frame_number, 42);
        assert_eq!(camera.rate.total, 1);
        assert!(camera.rate.is_live(now));
        assert!(!state.cameras.contains_key(&0));
        assert!(state.vision_live(now));

        // ... and shows up in the merged, ready-to-draw view.
        let merged = state.merged(now);
        assert_eq!(merged.balls.len(), 1);
        assert_eq!(merged.balls[0].camera_id, 2);
        assert_eq!(merged.balls[0].value.area, Some(120));
        assert_eq!(merged.blue.len(), 1);
        assert_eq!(merged.blue[0].value.robot_id, Some(3));
        assert_eq!(merged.yellow.len(), 1);
        assert_eq!(merged.yellow[0].value.robot_id, Some(7));

        // A second packet from another camera does not evict the first.
        let mut other = packet.clone();
        other.geometry = None;
        if let Some(detection) = other.detection.as_mut() {
            detection.camera_id = 0;
            detection.balls.clear();
        }
        state.apply_wrapper(other, now);
        assert_eq!(state.cameras.len(), 2);
        assert_eq!(state.merged(now).balls.len(), 1);
        assert_eq!(state.merged(now).blue.len(), 2);
    }

    #[test]
    fn garbage_bytes_are_counted_not_fatal() {
        // The receive path only ever sees `Result`s; make sure a malformed
        // datagram decodes to an error rather than a panic.
        assert!(SslWrapperPacket::decode(&[0xff, 0xff, 0xff, 0xff][..]).is_err());
    }
}
