//! The egui application: side panel, field rendering and mouse interaction.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use egui::{Color32, PointerButton, Sense, Vec2};
use ssl_sim_proto::{
    sim::{RobotId, SimulatorCommand, SimulatorControl, Team, TeleportBall, TeleportRobot},
    tracked::TrackedFrame,
};

use crate::{
    net::ControlSender,
    state::{Merged, State},
    view::{self, DrawOptions, FieldGeometry, Transform, DEFAULT_FIELD, ROBOT_RADIUS},
};

/// Teleports are streamed while dragging; cap them at 60 per second.
const TELEPORT_PERIOD: Duration = Duration::from_micros(16_667);
/// Target frame period for the repaint request.
const FRAME_PERIOD: Duration = Duration::from_micros(16_667);
/// How close (in metres) the cursor must be to grab the ball.
const BALL_GRAB_RADIUS: f32 = 0.15;
/// Radians of rotation per scroll-wheel unit.
const SCROLL_TO_RADIANS: f32 = 0.01;

/// Identifies a robot in the viewer: team flag plus id.
type RobotKey = (bool, u32);

#[derive(Debug, Clone, Copy, PartialEq)]
enum Drag {
    Ball,
    Robot(RobotKey),
}

/// Snapshot of the shared state taken once per frame, so the mutex is not
/// held while painting.
struct Snapshot {
    field: FieldGeometry,
    geometry: Option<ssl_sim_proto::sim::SslGeometryData>,
    merged: Merged,
    truth: Option<TrackedFrame>,
    truth_source: Option<String>,
    truth_hz: f32,
    truth_live: bool,
    cameras: Vec<CameraStat>,
    vision_live: bool,
    vision_decode_errors: u64,
    last_error: Option<String>,
}

struct CameraStat {
    id: u32,
    hz: f32,
    total: u64,
    live: bool,
    balls: usize,
    robots: usize,
}

/// The viewer.
pub struct ViewerApp {
    state: Arc<Mutex<State>>,
    control: ControlSender,
    vision_addr: SocketAddr,
    truth_addr: Option<SocketAddr>,

    show_truth: bool,
    show_camera_ids: bool,
    sim_speed: f32,
    sent_speed: f32,

    drag: Option<Drag>,
    last_teleport: Option<Instant>,
    selected: Option<RobotKey>,
    /// Orientation the user has scrolled a robot to, overriding detection.
    rotation: Option<(RobotKey, f32)>,
}

impl ViewerApp {
    /// Build the app around an already-populated shared state.
    pub fn new(
        state: Arc<Mutex<State>>,
        control: ControlSender,
        vision_addr: SocketAddr,
        truth_addr: Option<SocketAddr>,
    ) -> Self {
        Self {
            state,
            control,
            vision_addr,
            truth_addr,
            show_truth: true,
            show_camera_ids: false,
            sim_speed: 1.0,
            sent_speed: 1.0,
            drag: None,
            last_teleport: None,
            selected: None,
            rotation: None,
        }
    }

    fn snapshot(&self, now: Instant) -> Snapshot {
        let state = self.state.lock().expect("state mutex poisoned");
        let field = state
            .geometry
            .as_ref()
            .map(FieldGeometry::from_packet)
            .unwrap_or(DEFAULT_FIELD);
        let cameras = state
            .cameras
            .iter()
            .map(|(&id, camera)| CameraStat {
                id,
                hz: camera.rate.hz,
                total: camera.rate.total,
                live: camera.rate.is_live(now),
                balls: camera.frame.balls.len(),
                robots: camera.frame.robots_blue.len() + camera.frame.robots_yellow.len(),
            })
            .collect();
        Snapshot {
            field,
            geometry: state.geometry.clone(),
            merged: state.merged(now),
            truth: state.truth.as_ref().map(|t| t.frame.clone()),
            truth_source: state.truth.as_ref().and_then(|t| t.source.clone()),
            truth_hz: state.truth.as_ref().map_or(0.0, |t| t.rate.hz),
            truth_live: state.truth.as_ref().is_some_and(|t| t.rate.is_live(now)),
            cameras,
            vision_live: state.vision_live(now),
            vision_decode_errors: state.vision_decode_errors,
            last_error: state.last_error.clone(),
        }
    }

    fn send_control(&mut self, control: SimulatorControl) {
        self.control.send(&SimulatorCommand {
            control: Some(control),
            config: None,
        });
    }

    fn send_speed(&mut self, speed: f32) {
        self.sim_speed = speed;
        self.sent_speed = speed;
        self.send_control(SimulatorControl {
            simulation_speed: Some(speed),
            ..Default::default()
        });
    }

    fn send_teleport_ball(&mut self, x: f32, y: f32) {
        self.send_control(SimulatorControl {
            teleport_ball: Some(TeleportBall {
                x: Some(x),
                y: Some(y),
                z: Some(0.0),
                vx: Some(0.0),
                vy: Some(0.0),
                vz: Some(0.0),
                teleport_safely: Some(false),
                roll: None,
                by_force: Some(false),
            }),
            ..Default::default()
        });
    }

    fn send_teleport_robot(&mut self, key: RobotKey, x: f32, y: f32, orientation: f32) {
        let (blue, id) = key;
        self.send_control(SimulatorControl {
            teleport_robot: vec![TeleportRobot {
                id: RobotId {
                    id: Some(id),
                    team: Some(if blue { Team::Blue } else { Team::Yellow } as i32),
                },
                x: Some(x),
                y: Some(y),
                orientation: Some(orientation),
                v_x: Some(0.0),
                v_y: Some(0.0),
                v_angular: Some(0.0),
                present: Some(true),
                by_force: Some(false),
            }],
            ..Default::default()
        });
    }

    /// True if enough time has passed since the last streamed teleport.
    fn teleport_due(&mut self, now: Instant) -> bool {
        let due = self
            .last_teleport
            .is_none_or(|last| now.saturating_duration_since(last) >= TELEPORT_PERIOD);
        if due {
            self.last_teleport = Some(now);
        }
        due
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, snapshot: &Snapshot) {
        ui.heading("ssl-sim-viewer");
        ui.add_space(4.0);

        ui.label(format!("vision  {}", self.vision_addr));
        status_dot(ui, snapshot.vision_live, "receiving", "waiting for packets");
        egui::Grid::new("cameras")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                ui.label("cam");
                ui.label("Hz");
                ui.label("pkts");
                ui.label("objs");
                ui.end_row();
                for camera in &snapshot.cameras {
                    let color = if camera.live {
                        view::camera_color(camera.id)
                    } else {
                        Color32::DARK_GRAY
                    };
                    ui.colored_label(color, camera.id.to_string());
                    ui.label(format!("{:.1}", camera.hz));
                    ui.label(camera.total.to_string());
                    ui.label(format!("{}+{}", camera.robots, camera.balls));
                    ui.end_row();
                }
            });
        if snapshot.cameras.is_empty() {
            ui.weak("no cameras seen yet");
        }
        if snapshot.geometry.is_none() {
            ui.weak("no geometry packet yet (drawing default field)");
        }
        if snapshot.vision_decode_errors > 0 {
            ui.colored_label(
                Color32::LIGHT_RED,
                format!("{} undecodable packets", snapshot.vision_decode_errors),
            );
        }

        ui.separator();
        match self.truth_addr {
            Some(addr) => {
                ui.label(format!("truth  {addr}"));
                status_dot(ui, snapshot.truth_live, "receiving", "no tracker packets");
                if let Some(source) = &snapshot.truth_source {
                    ui.weak(source);
                }
                ui.label(format!("{:.1} Hz", snapshot.truth_hz));
                match snapshot.truth.as_ref().and_then(|f| f.balls.first()) {
                    Some(ball) => {
                        ui.monospace(format!("ball  x {:+.3}  y {:+.3}", ball.pos.x, ball.pos.y));
                        ui.monospace(format!("      z {:+.3} m", ball.pos.z));
                        if let Some(vel) = ball.vel {
                            ui.monospace(format!(
                                "  |v| {:.2} m/s",
                                (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt()
                            ));
                        }
                    }
                    None => {
                        ui.weak("no tracked ball");
                    }
                }
            }
            None => {
                ui.label("truth  disabled");
            }
        }

        ui.separator();
        ui.checkbox(&mut self.show_truth, "show truth overlay");
        ui.checkbox(&mut self.show_camera_ids, "show camera ids");

        ui.separator();
        ui.label(format!("control  {}", self.control.target()));
        let slider = ui.add(
            egui::Slider::new(&mut self.sim_speed, 0.0..=5.0)
                .text("speed")
                .fixed_decimals(2),
        );
        if slider.changed() && (self.sim_speed - self.sent_speed).abs() > f32::EPSILON {
            let speed = self.sim_speed;
            self.send_speed(speed);
        }
        ui.horizontal(|ui| {
            if ui.button("Pause").clicked() {
                self.send_speed(0.0);
            }
            if ui.button("Resume").clicked() {
                self.send_speed(1.0);
            }
        });
        ui.weak(format!("{} commands sent", self.control.sent));
        if let Some(err) = &self.control.last_error {
            ui.colored_label(Color32::LIGHT_RED, err);
        }

        ui.separator();
        ui.weak("left-drag ball · right-drag robot");
        ui.weak("scroll: rotate · middle-click: place ball");
        if let Some(err) = &snapshot.last_error {
            ui.colored_label(Color32::LIGHT_RED, err);
        }
    }

    fn field_panel(&mut self, ui: &mut egui::Ui, snapshot: &Snapshot, now: Instant) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let t = Transform::fit(response.rect, snapshot.field.half_extent(), 8.0);

        view::draw_field(&painter, &t, &snapshot.field, snapshot.geometry.as_ref());
        let options = DrawOptions {
            show_camera_ids: self.show_camera_ids,
            selected: self.selected,
        };
        view::draw_detections(&painter, &t, &snapshot.merged, &options);
        if self.show_truth {
            if let Some(frame) = &snapshot.truth {
                view::draw_truth(&painter, &t, frame);
            }
        }

        self.handle_input(&response, &t, snapshot, now);

        // Cursor read-out, bottom left.
        if let Some(pos) = response.hover_pos() {
            let (wx, wy) = t.to_world(pos);
            painter.text(
                response.rect.left_bottom() + Vec2::new(6.0, -6.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{wx:+.3}, {wy:+.3} m"),
                egui::FontId::monospace(12.0),
                Color32::from_white_alpha(180),
            );
        }
    }

    fn handle_input(
        &mut self,
        response: &egui::Response,
        t: &Transform,
        snapshot: &Snapshot,
        now: Instant,
    ) {
        let Some(pointer) = response.interact_pointer_pos().or(response.hover_pos()) else {
            if !response.dragged() {
                self.drag = None;
            }
            return;
        };
        let (wx, wy) = t.to_world(pointer);

        // Hover selection, frozen while dragging a robot.
        match self.drag {
            Some(Drag::Robot(key)) => self.selected = Some(key),
            _ => {
                if response.hovered() {
                    self.selected = nearest_robot(&snapshot.merged, wx, wy).map(|hit| hit.key);
                }
            }
        }

        if response.drag_started_by(PointerButton::Primary) {
            self.drag = near_ball(&snapshot.merged, wx, wy).then_some(Drag::Ball);
        }
        if response.drag_started_by(PointerButton::Secondary) {
            self.drag = nearest_robot(&snapshot.merged, wx, wy).map(|hit| Drag::Robot(hit.key));
        }

        let dragging_ball =
            self.drag == Some(Drag::Ball) && response.dragged_by(PointerButton::Primary);
        let dragging_robot = match self.drag {
            Some(Drag::Robot(key)) if response.dragged_by(PointerButton::Secondary) => Some(key),
            _ => None,
        };
        if dragging_ball && self.teleport_due(now) {
            self.send_teleport_ball(wx, wy);
        } else if let Some(key) = dragging_robot.filter(|_| self.teleport_due(now)) {
            let orientation = self.orientation_of(key, &snapshot.merged);
            self.send_teleport_robot(key, wx, wy, orientation);
        }
        if response.drag_stopped() {
            self.drag = None;
            self.last_teleport = None;
        }

        if response.clicked_by(PointerButton::Middle) {
            self.send_teleport_ball(wx, wy);
        }

        // Scroll wheel rotates the selected robot in place.
        if response.hovered() {
            let scroll = response.ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                if let Some(key) = self.selected {
                    if let Some(hit) = find_robot(&snapshot.merged, key) {
                        let orientation = wrap_angle(
                            self.orientation_of(key, &snapshot.merged) + scroll * SCROLL_TO_RADIANS,
                        );
                        self.rotation = Some((key, orientation));
                        self.send_teleport_robot(key, hit.x, hit.y, orientation);
                    }
                }
            }
        }
    }

    /// Orientation to keep when teleporting: whatever the user last scrolled
    /// this robot to, otherwise the orientation from the latest detection.
    fn orientation_of(&self, key: RobotKey, merged: &Merged) -> f32 {
        if let Some((rot_key, orientation)) = self.rotation {
            if rot_key == key {
                return orientation;
            }
        }
        find_robot(merged, key).map_or(0.0, |hit| hit.orientation)
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let snapshot = self.snapshot(now);

        egui::Panel::right("side")
            .default_size(270.0)
            .show(ui, |ui| self.side_panel(ui, &snapshot));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.field_panel(ui, &snapshot, now));

        ui.ctx().request_repaint_after(FRAME_PERIOD);
    }
}

fn status_dot(ui: &mut egui::Ui, ok: bool, yes: &str, no: &str) {
    let (color, text) = if ok {
        (Color32::from_rgb(80, 220, 120), yes)
    } else {
        (Color32::from_rgb(220, 140, 60), no)
    };
    ui.colored_label(color, format!("\u{25cf} {text}"));
}

/// A robot found by a hit test, in metres.
struct RobotHit {
    key: RobotKey,
    x: f32,
    y: f32,
    orientation: f32,
}

fn robot_hits(merged: &Merged) -> impl Iterator<Item = RobotHit> + '_ {
    let blue = merged.blue.iter().map(|seen| (true, seen));
    let yellow = merged.yellow.iter().map(|seen| (false, seen));
    blue.chain(yellow).filter_map(|(is_blue, seen)| {
        seen.value.robot_id.map(|id| RobotHit {
            key: (is_blue, id),
            x: view::mm(seen.value.x),
            y: view::mm(seen.value.y),
            orientation: seen.value.orientation.unwrap_or(0.0),
        })
    })
}

/// Nearest robot whose body contains the point, if any.
fn nearest_robot(merged: &Merged, x: f32, y: f32) -> Option<RobotHit> {
    robot_hits(merged)
        .map(|hit| {
            let d2 = (hit.x - x).powi(2) + (hit.y - y).powi(2);
            (d2, hit)
        })
        .filter(|(d2, _)| *d2 <= ROBOT_RADIUS * ROBOT_RADIUS)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, hit)| hit)
}

fn find_robot(merged: &Merged, key: RobotKey) -> Option<RobotHit> {
    robot_hits(merged).find(|hit| hit.key == key)
}

fn near_ball(merged: &Merged, x: f32, y: f32) -> bool {
    merged.balls.iter().any(|seen| {
        let (bx, by) = (view::mm(seen.value.x), view::mm(seen.value.y));
        (bx - x).powi(2) + (by - y).powi(2) <= BALL_GRAB_RADIUS * BALL_GRAB_RADIUS
    })
}

fn wrap_angle(a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let wrapped = a.rem_euclid(tau);
    if wrapped > std::f32::consts::PI {
        wrapped - tau
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Seen;
    use ssl_sim_proto::sim::SslDetectionRobot;

    fn robot(id: u32, x: f32, y: f32) -> Seen<SslDetectionRobot> {
        Seen {
            camera_id: 0,
            value: SslDetectionRobot {
                confidence: 1.0,
                robot_id: Some(id),
                x,
                y,
                orientation: Some(0.5),
                pixel_x: 0.0,
                pixel_y: 0.0,
                height: Some(0.15),
            },
        }
    }

    #[test]
    fn hit_test_picks_the_closest_robot_inside_its_body() {
        let merged = Merged {
            blue: vec![robot(3, 1000.0, 0.0), robot(4, 1120.0, 0.0)],
            ..Default::default()
        };
        // 1.05 m is inside both bodies (0.09 m radius) but closer to robot 3.
        assert_eq!(nearest_robot(&merged, 1.05, 0.0).unwrap().key, (true, 3));
        // 1.07 m is closer to robot 4.
        assert_eq!(nearest_robot(&merged, 1.07, 0.0).unwrap().key, (true, 4));
        // Just outside both bodies in y.
        assert!(nearest_robot(&merged, 1.06, 0.1).is_none());
        // Far away: nothing.
        assert!(nearest_robot(&merged, 4.0, 3.0).is_none());
    }

    #[test]
    fn wrap_angle_stays_in_pi_range() {
        for &(input, expected) in &[
            (0.0_f32, 0.0_f32),
            (std::f32::consts::TAU + 1.0, 1.0),
            (-std::f32::consts::FRAC_PI_2, -std::f32::consts::FRAC_PI_2),
        ] {
            assert!((wrap_angle(input) - expected).abs() < 1e-4);
        }
    }
}
