//! Metres-to-pixels transform and all the `Painter` drawing.

use egui::{epaint::PathShape, Align2, Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};
use ssl_sim_proto::{
    sim::{SslDetectionBall, SslDetectionRobot, SslGeometryData},
    tracked::TrackedFrame,
};

use crate::state::{Merged, Seen};

/// Robot body radius \[m\] used for drawing (proto default).
pub const ROBOT_RADIUS: f32 = 0.09;
/// Distance from the robot centre to the flat front \[m\].
pub const ROBOT_CENTER_TO_DRIBBLER: f32 = 0.075;
/// Ball radius \[m\].
pub const BALL_RADIUS: f32 = 0.0215;

/// Default Division A field, used until a geometry packet arrives.
pub const DEFAULT_FIELD: FieldGeometry = FieldGeometry {
    length: 12.0,
    width: 9.0,
    goal_width: 1.8,
    goal_depth: 0.18,
    boundary_width: 0.3,
};

const BLUE: Color32 = Color32::from_rgb(60, 120, 235);
const YELLOW: Color32 = Color32::from_rgb(235, 200, 60);
const ORANGE: Color32 = Color32::from_rgb(255, 140, 20);
const GRASS: Color32 = Color32::from_rgb(22, 72, 38);
const BOUNDARY: Color32 = Color32::from_rgb(14, 46, 24);
const LINE: Color32 = Color32::from_rgb(226, 232, 226);
const TRUTH: Color32 = Color32::from_rgb(255, 90, 200);

/// Colour-blind-ish palette for the per-camera dots.
const CAMERA_COLORS: [Color32; 6] = [
    Color32::from_rgb(255, 255, 255),
    Color32::from_rgb(120, 255, 200),
    Color32::from_rgb(255, 170, 120),
    Color32::from_rgb(180, 160, 255),
    Color32::from_rgb(255, 240, 120),
    Color32::from_rgb(120, 220, 255),
];

/// Colour used for the camera-id marker of `camera_id`.
pub fn camera_color(camera_id: u32) -> Color32 {
    CAMERA_COLORS[camera_id as usize % CAMERA_COLORS.len()]
}

/// Field dimensions in metres, extracted from the geometry packet (which uses
/// millimetres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldGeometry {
    /// Playing field length (goal line to goal line) \[m\].
    pub length: f32,
    /// Playing field width (touch line to touch line) \[m\].
    pub width: f32,
    /// Inner goal width \[m\].
    pub goal_width: f32,
    /// Goal depth \[m\].
    pub goal_depth: f32,
    /// Run-off area outside the touch/goal lines \[m\].
    pub boundary_width: f32,
}

impl FieldGeometry {
    /// Read the field size out of a geometry packet.
    pub fn from_packet(geometry: &SslGeometryData) -> Self {
        let field = &geometry.field;
        Self {
            length: mm(field.field_length as f32),
            width: mm(field.field_width as f32),
            goal_width: mm(field.goal_width as f32),
            goal_depth: mm(field.goal_depth as f32),
            boundary_width: mm(field.boundary_width as f32),
        }
    }

    /// Half extents of everything that must be visible, goals included.
    pub fn half_extent(&self) -> Vec2 {
        Vec2::new(
            self.length / 2.0 + self.boundary_width.max(self.goal_depth + 0.05),
            self.width / 2.0 + self.boundary_width,
        )
    }
}

/// Millimetres to metres.
pub fn mm(value: f32) -> f32 {
    value / 1000.0
}

/// Affine map from world metres (x right, y up, origin at field centre) to
/// screen pixels (y down).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Pixels per metre.
    pub scale: f32,
    /// Screen position of the world origin.
    pub origin: Pos2,
}

impl Transform {
    /// Fit a world box of `half_extent` metres (centred on the origin) into
    /// `screen`, preserving aspect ratio and leaving `margin` pixels of slack.
    pub fn fit(screen: Rect, half_extent: Vec2, margin: f32) -> Self {
        let avail_x = (screen.width() - 2.0 * margin).max(1.0);
        let avail_y = (screen.height() - 2.0 * margin).max(1.0);
        let scale = (avail_x / (2.0 * half_extent.x).max(1e-6))
            .min(avail_y / (2.0 * half_extent.y).max(1e-6))
            .max(1e-6);
        Self {
            scale,
            origin: screen.center(),
        }
    }

    /// World metres to screen pixels.
    pub fn to_screen(self, x: f32, y: f32) -> Pos2 {
        Pos2::new(
            self.origin.x + x * self.scale,
            self.origin.y - y * self.scale,
        )
    }

    /// Screen pixels back to world metres.
    pub fn to_world(self, pos: Pos2) -> (f32, f32) {
        (
            (pos.x - self.origin.x) / self.scale,
            (self.origin.y - pos.y) / self.scale,
        )
    }

    /// A length in metres expressed in pixels.
    pub fn len(self, metres: f32) -> f32 {
        metres * self.scale
    }
}

/// What to draw on top of the raw detections.
#[derive(Debug, Clone, Copy)]
pub struct DrawOptions {
    /// Draw a colour-coded dot per detection showing the reporting camera.
    pub show_camera_ids: bool,
    /// Robot highlighted by the mouse, if any.
    pub selected: Option<(bool, u32)>,
}

/// Draw the pitch: run-off area, playing surface, lines, arcs and goals.
pub fn draw_field(
    painter: &Painter,
    t: &Transform,
    field: &FieldGeometry,
    geometry: Option<&SslGeometryData>,
) {
    let outer = field.half_extent();
    painter.rect_filled(
        Rect::from_min_max(
            t.to_screen(-outer.x, outer.y),
            t.to_screen(outer.x, -outer.y),
        ),
        0.0,
        BOUNDARY,
    );
    let (hx, hy) = (field.length / 2.0, field.width / 2.0);
    let inner = Vec2::new(hx + field.boundary_width, hy + field.boundary_width);
    painter.rect_filled(
        Rect::from_min_max(
            t.to_screen(-inner.x, inner.y),
            t.to_screen(inner.x, -inner.y),
        ),
        0.0,
        GRASS,
    );

    match geometry {
        Some(geometry) if !geometry.field.field_lines.is_empty() => {
            for line in &geometry.field.field_lines {
                painter.line_segment(
                    [
                        t.to_screen(mm(line.p1.x), mm(line.p1.y)),
                        t.to_screen(mm(line.p2.x), mm(line.p2.y)),
                    ],
                    Stroke::new(t.len(mm(line.thickness)).max(1.0), LINE),
                );
            }
            for arc in &geometry.field.field_arcs {
                draw_arc(
                    painter,
                    t,
                    (mm(arc.center.x), mm(arc.center.y)),
                    mm(arc.radius),
                    arc.a1,
                    arc.a2,
                    Stroke::new(t.len(mm(arc.thickness)).max(1.0), LINE),
                );
            }
        }
        _ => draw_fallback_lines(painter, t, field),
    }

    // Goals, which the geometry packet does not describe as lines.
    let stroke = Stroke::new(t.len(0.02).max(1.5), LINE);
    let gh = field.goal_width / 2.0;
    for side in [-1.0_f32, 1.0] {
        let x0 = side * hx;
        let x1 = side * (hx + field.goal_depth);
        painter.line_segment([t.to_screen(x0, gh), t.to_screen(x1, gh)], stroke);
        painter.line_segment([t.to_screen(x0, -gh), t.to_screen(x1, -gh)], stroke);
        painter.line_segment([t.to_screen(x1, gh), t.to_screen(x1, -gh)], stroke);
    }
}

/// Minimal field markings for when no geometry packet has arrived yet.
fn draw_fallback_lines(painter: &Painter, t: &Transform, field: &FieldGeometry) {
    let (hx, hy) = (field.length / 2.0, field.width / 2.0);
    let stroke = Stroke::new(t.len(0.01).max(1.0), LINE);
    let corners = [
        t.to_screen(-hx, -hy),
        t.to_screen(hx, -hy),
        t.to_screen(hx, hy),
        t.to_screen(-hx, hy),
    ];
    painter.add(PathShape::closed_line(corners.to_vec(), stroke));
    painter.line_segment([t.to_screen(0.0, -hy), t.to_screen(0.0, hy)], stroke);
    draw_arc(
        painter,
        t,
        (0.0, 0.0),
        0.5,
        0.0,
        std::f32::consts::TAU,
        stroke,
    );
}

/// Polyline approximation of a circular arc in world coordinates.
fn draw_arc(
    painter: &Painter,
    t: &Transform,
    center: (f32, f32),
    radius: f32,
    a1: f32,
    a2: f32,
    stroke: Stroke,
) {
    let sweep = a2 - a1;
    let steps = ((sweep.abs() * radius * t.scale / 4.0).ceil() as usize).clamp(6, 256);
    let points: Vec<Pos2> = (0..=steps)
        .map(|i| {
            let a = a1 + sweep * (i as f32 / steps as f32);
            t.to_screen(center.0 + radius * a.cos(), center.1 + radius * a.sin())
        })
        .collect();
    painter.add(PathShape::line(points, stroke));
}

/// Outline of a robot: a disc with the front chopped off at
/// [`ROBOT_CENTER_TO_DRIBBLER`], oriented by `orientation` \[rad\].
pub fn robot_outline(t: &Transform, x: f32, y: f32, orientation: f32) -> Vec<Pos2> {
    let half_mouth = (ROBOT_CENTER_TO_DRIBBLER / ROBOT_RADIUS)
        .clamp(-1.0, 1.0)
        .acos();
    let sweep = std::f32::consts::TAU - 2.0 * half_mouth;
    let steps = ((t.len(ROBOT_RADIUS) / 1.5).ceil() as usize).clamp(8, 64);
    (0..=steps)
        .map(|i| {
            let a = orientation + half_mouth + sweep * (i as f32 / steps as f32);
            t.to_screen(x + ROBOT_RADIUS * a.cos(), y + ROBOT_RADIUS * a.sin())
        })
        .collect()
}

/// Draw one detected robot.
fn draw_robot(
    painter: &Painter,
    t: &Transform,
    seen: &Seen<SslDetectionRobot>,
    is_blue: bool,
    options: &DrawOptions,
) {
    let robot = &seen.value;
    let (x, y) = (mm(robot.x), mm(robot.y));
    let orientation = robot.orientation.unwrap_or(0.0);
    let base = if is_blue { BLUE } else { YELLOW };
    let selected = options.selected == Some((is_blue, robot.robot_id.unwrap_or(u32::MAX)));
    let stroke = if selected {
        Stroke::new(2.5, Color32::WHITE)
    } else {
        Stroke::new(1.0, Color32::from_black_alpha(160))
    };
    painter.add(PathShape::convex_polygon(
        robot_outline(t, x, y, orientation),
        base,
        stroke,
    ));

    // Heading tick from the centre through the flat front.
    painter.line_segment(
        [
            t.to_screen(x, y),
            t.to_screen(
                x + ROBOT_CENTER_TO_DRIBBLER * orientation.cos(),
                y + ROBOT_CENTER_TO_DRIBBLER * orientation.sin(),
            ),
        ],
        Stroke::new(1.5, Color32::from_black_alpha(200)),
    );

    let font_size = t.len(0.09);
    if font_size >= 7.0 {
        if let Some(id) = robot.robot_id {
            painter.text(
                t.to_screen(x, y),
                Align2::CENTER_CENTER,
                id,
                FontId::proportional(font_size),
                if is_blue {
                    Color32::WHITE
                } else {
                    Color32::BLACK
                },
            );
        }
    }
    if options.show_camera_ids {
        painter.circle_filled(
            t.to_screen(x, y + ROBOT_RADIUS * 1.35),
            (t.len(0.02)).max(2.0),
            camera_color(seen.camera_id),
        );
    }
}

/// Draw one detected ball.
fn draw_ball(
    painter: &Painter,
    t: &Transform,
    seen: &Seen<SslDetectionBall>,
    options: &DrawOptions,
) {
    let ball = &seen.value;
    let (x, y) = (mm(ball.x), mm(ball.y));
    // `area` is in pixels; grow the dot with its square root so a ball high in
    // the air (bigger blob) reads as bigger without exploding.
    let area_scale = ball
        .area
        .map(|a| ((a as f32).sqrt() / 6.0).clamp(0.8, 3.0))
        .unwrap_or(1.0);
    let radius = (t.len(BALL_RADIUS) * area_scale).max(2.0);
    let color = if ball.confidence < 0.5 {
        ORANGE.gamma_multiply(0.4)
    } else {
        ORANGE
    };
    painter.circle(
        t.to_screen(x, y),
        radius,
        color,
        Stroke::new(1.0, Color32::from_black_alpha(120)),
    );
    if options.show_camera_ids {
        painter.circle_filled(
            t.to_screen(x, y + BALL_RADIUS * 3.0),
            (t.len(0.015)).max(1.5),
            camera_color(seen.camera_id),
        );
    }
}

/// Draw all merged detections.
pub fn draw_detections(painter: &Painter, t: &Transform, merged: &Merged, options: &DrawOptions) {
    for seen in &merged.blue {
        draw_robot(painter, t, seen, true, options);
    }
    for seen in &merged.yellow {
        draw_robot(painter, t, seen, false, options);
    }
    for seen in &merged.balls {
        draw_ball(painter, t, seen, options);
    }
}

/// Draw the ground-truth frame as thin outlines on top of the detections.
pub fn draw_truth(painter: &Painter, t: &Transform, frame: &TrackedFrame) {
    let stroke = Stroke::new(1.5, TRUTH);
    for robot in &frame.robots {
        painter.add(PathShape::line(
            robot_outline(t, robot.pos.x, robot.pos.y, robot.orientation),
            stroke,
        ));
    }
    for ball in &frame.balls {
        painter.circle_stroke(
            t.to_screen(ball.pos.x, ball.pos.y),
            t.len(BALL_RADIUS).max(2.0),
            stroke,
        );
        // A ball off the ground: mark its true (unprojected) height.
        if ball.pos.z > 0.02 {
            painter.circle_stroke(
                t.to_screen(ball.pos.x, ball.pos.y),
                t.len(BALL_RADIUS + ball.pos.z * 0.25).max(3.0),
                Stroke::new(1.0, TRUTH.gamma_multiply(0.5)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "{a} != {b}");
    }

    #[test]
    fn fit_centres_and_scales_uniformly() {
        // 800x400 viewport, 12.6 x 9.6 m world, no margin.
        let screen = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(800.0, 400.0));
        let t = Transform::fit(screen, Vec2::new(6.3, 4.8), 0.0);

        // Height is the binding constraint: 400 px / 9.6 m.
        approx(t.scale, 400.0 / 9.6);
        approx(t.origin.x, 400.0);
        approx(t.origin.y, 200.0);

        // World origin maps to the centre of the viewport.
        let c = t.to_screen(0.0, 0.0);
        approx(c.x, 400.0);
        approx(c.y, 200.0);
    }

    #[test]
    fn y_axis_points_up_and_x_right() {
        let screen = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(800.0, 400.0));
        let t = Transform::fit(screen, Vec2::new(6.3, 4.8), 0.0);

        let up = t.to_screen(0.0, 1.0);
        let right = t.to_screen(1.0, 0.0);
        assert!(up.y < t.origin.y, "+y must move up the screen");
        assert!(right.x > t.origin.x, "+x must move right");
        approx(t.origin.y - up.y, t.scale);
        approx(right.x - t.origin.x, t.scale);
    }

    #[test]
    fn screen_and_world_round_trip() {
        let screen = Rect::from_min_size(Pos2::new(37.0, 11.0), Vec2::new(1024.0, 768.0));
        let t = Transform::fit(screen, Vec2::new(6.3, 4.8), 20.0);
        for &(x, y) in &[(0.0, 0.0), (4.5, -3.0), (-6.0, 4.4), (0.123, -0.456)] {
            let (rx, ry) = t.to_world(t.to_screen(x, y));
            approx(rx, x);
            approx(ry, y);
        }
    }

    #[test]
    fn lengths_scale_with_the_transform() {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(960.0, 960.0));
        let t = Transform::fit(screen, Vec2::new(6.0, 6.0), 0.0);
        approx(t.scale, 80.0);
        approx(t.len(0.09), 7.2);
    }

    #[test]
    fn robot_outline_has_a_flat_front() {
        let t = Transform {
            scale: 100.0,
            origin: Pos2::ZERO,
        };
        // Facing +x: the mouth chord is the gap between the last and first point.
        let outline = robot_outline(&t, 0.0, 0.0, 0.0);
        let first = outline[0];
        let last = outline[outline.len() - 1];
        // Both chord ends sit at x = +0.075 m => +7.5 px.
        approx(first.x, ROBOT_CENTER_TO_DRIBBLER * 100.0);
        approx(last.x, ROBOT_CENTER_TO_DRIBBLER * 100.0);
        // ... and are mirrored about y.
        approx(first.y, -last.y);
        // Every point is on the robot circle.
        for p in &outline {
            approx(p.to_vec2().length(), ROBOT_RADIUS * 100.0);
        }
    }
}
