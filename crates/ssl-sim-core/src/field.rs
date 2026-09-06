//! Field geometry: dimensions, line/arc sets for the geometry packet, and the
//! collision primitives (boundary boards, goal frames, room box).
//!
//! OWNER: general agent A. Replace the `todo!()` bodies; keep the signatures.

use serde::{Deserialize, Serialize};

use crate::types::{Team, Vec2};

/// SSL division.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Division {
    /// 12 x 9 m, 11 robots.
    A,
    /// 9 x 6 m, 6 robots.
    B,
}

/// Field dimensions [m]. `length` is along x, `width` along y, centre at the origin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FieldGeometry {
    /// Goal line to goal line.
    pub length: f64,
    /// Touch line to touch line.
    pub width: f64,
    /// Between the goal posts (inner).
    pub goal_width: f64,
    /// Goal line to the back of the goal.
    pub goal_depth: f64,
    /// Height of the goal frame (crossbar).
    pub goal_height: f64,
    /// Thickness of the goal posts and back wall.
    pub goal_wall_thickness: f64,
    /// Field line to the boundary boards.
    pub boundary_width: f64,
    /// Penalty area depth (along x).
    pub penalty_area_depth: f64,
    /// Penalty area width (along y).
    pub penalty_area_width: f64,
    /// Centre circle radius.
    pub center_circle_radius: f64,
    /// Painted line thickness.
    pub line_thickness: f64,
}

impl Default for FieldGeometry {
    fn default() -> Self {
        Self::division(Division::A)
    }
}

/// A painted line segment for the geometry packet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldLine {
    /// SSL standard name, e.g. `TopTouchLine`.
    pub name: String,
    /// Start [m].
    pub p1: Vec2,
    /// End [m].
    pub p2: Vec2,
    /// Thickness [m].
    pub thickness: f64,
}

/// A painted arc for the geometry packet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldArc {
    /// SSL standard name, e.g. `CenterCircle`.
    pub name: String,
    /// Centre [m].
    pub center: Vec2,
    /// Radius [m].
    pub radius: f64,
    /// Start angle [rad].
    pub a1: f64,
    /// End angle [rad].
    pub a2: f64,
    /// Thickness [m].
    pub thickness: f64,
}

/// A vertical wall segment used for ball and robot collisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallSegment {
    /// Start point [m].
    pub a: Vec2,
    /// End point [m].
    pub b: Vec2,
    /// Height above the floor [m]; the ball passes over when higher.
    pub height: f64,
    /// Outward normal (the side objects live on). Unit length.
    pub normal: Vec2,
    /// True for goal posts / back walls (used for goal events and ball-only collisions).
    pub is_goal: bool,
}

impl FieldGeometry {
    /// Standard dimensions for a division.
    pub fn division(div: Division) -> Self {
        match div {
            Division::A => Self {
                length: 12.0,
                width: 9.0,
                goal_width: 1.8,
                goal_depth: 0.18,
                goal_height: 0.155,
                goal_wall_thickness: 0.02,
                boundary_width: 0.3,
                penalty_area_depth: 1.8,
                penalty_area_width: 3.6,
                center_circle_radius: 0.5,
                line_thickness: 0.01,
            },
            Division::B => Self {
                length: 9.0,
                width: 6.0,
                goal_width: 1.0,
                goal_depth: 0.18,
                goal_height: 0.155,
                goal_wall_thickness: 0.02,
                boundary_width: 0.3,
                penalty_area_depth: 1.0,
                penalty_area_width: 2.0,
                center_circle_radius: 0.5,
                line_thickness: 0.01,
            },
        }
    }

    /// Half length.
    pub fn half_length(&self) -> f64 {
        self.length * 0.5
    }

    /// Half width.
    pub fn half_width(&self) -> f64 {
        self.width * 0.5
    }

    /// The 2018+ SSL line set (touch lines, goal lines, halfway, centre, penalty stretches).
    ///
    /// Names are exactly the `SSL_FieldShapeType` enumerators. "Left" and
    /// "Right" in the penalty stretch names are as seen from the centre of the
    /// field looking towards that goal, which puts
    /// `LeftFieldLeftPenaltyStretch` at `-y` and `RightFieldLeftPenaltyStretch`
    /// at `+y`.
    pub fn lines(&self) -> Vec<FieldLine> {
        let hl = self.half_length();
        let hw = self.half_width();
        let d = self.penalty_area_depth;
        let pw = self.penalty_area_width * 0.5;
        let t = self.line_thickness;
        let line = |name: &str, x1: f64, y1: f64, x2: f64, y2: f64| FieldLine {
            name: name.to_string(),
            p1: Vec2::new(x1, y1),
            p2: Vec2::new(x2, y2),
            thickness: t,
        };
        vec![
            line("TopTouchLine", -hl, hw, hl, hw),
            line("BottomTouchLine", -hl, -hw, hl, -hw),
            line("LeftGoalLine", -hl, -hw, -hl, hw),
            line("RightGoalLine", hl, -hw, hl, hw),
            line("HalfwayLine", 0.0, -hw, 0.0, hw),
            line("CenterLine", -hl, 0.0, hl, 0.0),
            line("LeftPenaltyStretch", -hl + d, -pw, -hl + d, pw),
            line("RightPenaltyStretch", hl - d, -pw, hl - d, pw),
            line("LeftFieldLeftPenaltyStretch", -hl, -pw, -hl + d, -pw),
            line("LeftFieldRightPenaltyStretch", -hl, pw, -hl + d, pw),
            line("RightFieldLeftPenaltyStretch", hl, pw, hl - d, pw),
            line("RightFieldRightPenaltyStretch", hl, -pw, hl - d, -pw),
        ]
    }

    /// The arc set (centre circle).
    pub fn arcs(&self) -> Vec<FieldArc> {
        vec![FieldArc {
            name: "CenterCircle".to_string(),
            center: Vec2::ZERO,
            radius: self.center_circle_radius,
            a1: 0.0,
            a2: std::f64::consts::TAU,
            thickness: self.line_thickness,
        }]
    }

    /// Collision walls: the four boundary boards (height `wall_height`, at the
    /// outer edge of the boundary area) and both goal frames (two posts and a
    /// back wall each, `goal_wall_thickness` thick, `goal_height` tall).
    /// The boundary boards have an opening where the goal is, so the ball can
    /// enter the goal.
    ///
    /// Goal posts are modelled as zero-thickness segments on the post centre
    /// line (`|y| = goal_width/2 + thickness/2`), each emitted twice with
    /// opposite normals so both faces collide.
    pub fn walls(&self, wall_height: f64) -> Vec<WallSegment> {
        let hl = self.half_length();
        let hw = self.half_width();
        let ox = hl + self.boundary_width;
        let oy = hw + self.boundary_width;
        let gh = self.goal_width * 0.5;
        let th = self.goal_wall_thickness;
        let mut walls = Vec::with_capacity(16);

        let board = |a: Vec2, b: Vec2, normal: Vec2| WallSegment {
            a,
            b,
            height: wall_height,
            normal,
            is_goal: false,
        };

        // Boards behind the touch lines; normals point back onto the field.
        walls.push(board(
            Vec2::new(-ox, oy),
            Vec2::new(ox, oy),
            Vec2::new(0.0, -1.0),
        ));
        walls.push(board(
            Vec2::new(-ox, -oy),
            Vec2::new(ox, -oy),
            Vec2::new(0.0, 1.0),
        ));

        // Boards behind the goal lines, split so the goal mouth is open.
        for s in [-1.0f64, 1.0] {
            let n = Vec2::new(-s, 0.0);
            walls.push(board(Vec2::new(s * ox, -oy), Vec2::new(s * ox, -gh), n));
            walls.push(board(Vec2::new(s * ox, gh), Vec2::new(s * ox, oy), n));
        }

        // Goal frames.
        for s in [-1.0f64, 1.0] {
            let x0 = s * hl;
            let x1 = s * (hl + self.goal_depth);
            for sy in [-1.0f64, 1.0] {
                let y = sy * (gh + th * 0.5);
                let a = Vec2::new(x0, y);
                let b = Vec2::new(x1, y);
                // Inner face (towards the goal mouth) and outer face.
                walls.push(WallSegment {
                    a,
                    b,
                    height: self.goal_height,
                    normal: Vec2::new(0.0, -sy),
                    is_goal: true,
                });
                walls.push(WallSegment {
                    a,
                    b,
                    height: self.goal_height,
                    normal: Vec2::new(0.0, sy),
                    is_goal: true,
                });
            }
            walls.push(WallSegment {
                a: Vec2::new(x1, -(gh + th * 0.5)),
                b: Vec2::new(x1, gh + th * 0.5),
                height: self.goal_height,
                normal: Vec2::new(-s, 0.0),
                is_goal: true,
            });
        }

        walls
    }

    /// Axis-aligned "room" box half extents [m] beyond which nothing may travel
    /// (field + 1 m margin). Tall (implicitly infinite).
    pub fn room_half_extents(&self) -> Vec2 {
        Vec2::new(
            self.half_length() + self.boundary_width + self.goal_depth + 1.0,
            self.half_width() + self.boundary_width + 1.0,
        )
    }

    /// Which team's goal (if any) contains this point: `x` beyond the goal
    /// line and inside the goal mouth and depth.
    ///
    /// Blue defends `-x`, so the goal at `-x` is blue's.
    pub fn goal_containing(&self, p: Vec2) -> Option<Team> {
        let hl = self.half_length();
        if p.x.abs() <= hl || p.x.abs() >= hl + self.goal_depth {
            return None;
        }
        if p.y.abs() >= self.goal_width * 0.5 {
            return None;
        }
        Some(if p.x < 0.0 { Team::Blue } else { Team::Yellow })
    }

    /// True if the point is outside the field lines (touch or goal lines).
    pub fn is_outside_field(&self, p: Vec2) -> bool {
        p.x.abs() > self.half_length() || p.y.abs() > self.half_width()
    }
}

/// grSim-like "inside" formation for Division A, robot numbers 1..=15
/// (number 0 is the goalkeeper and is derived from the field length instead).
/// Positions are for the `-x` team; three staggered ranks entirely inside the
/// own half, clear of the centre circle.
const FORMATION_INSIDE_DIV_A: [(f64, f64); 15] = [
    (-4.0, 0.0),
    (-4.0, 1.0),
    (-4.0, -1.0),
    (-4.0, 2.0),
    (-4.0, -2.0),
    (-2.5, 0.5),
    (-2.5, -0.5),
    (-2.5, 1.5),
    (-2.5, -1.5),
    (-2.5, 2.5),
    (-2.5, -2.5),
    (-1.2, 1.0),
    (-1.2, -1.0),
    (-1.2, 2.0),
    (-1.2, -2.0),
];

/// Default starting formations, mirrored for the two teams. Index = robot number.
/// Blue is placed at `-x`, yellow at `+x`, both facing the centre.
///
/// Number 0 is the goalkeeper, 0.3 m in front of its own goal line; the rest are
/// the grSim "inside" formation scaled to the division's field size.
pub fn default_formation(div: Division, number: u8) -> Option<Vec2> {
    if number > 15 {
        return None;
    }
    let field = FieldGeometry::division(div);
    let hl = field.half_length();
    let hw = field.half_width();
    if number == 0 {
        return Some(Vec2::new(-hl + 0.3, 0.0));
    }
    let reference = FieldGeometry::division(Division::A);
    let sx = hl / reference.half_length();
    let sy = hw / reference.half_width();
    let (x, y) = FORMATION_INSIDE_DIV_A[number as usize - 1];
    Some(Vec2::new(x * sx, y * sy))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE_NAMES: [&str; 12] = [
        "TopTouchLine",
        "BottomTouchLine",
        "LeftGoalLine",
        "RightGoalLine",
        "HalfwayLine",
        "CenterLine",
        "LeftPenaltyStretch",
        "RightPenaltyStretch",
        "LeftFieldLeftPenaltyStretch",
        "LeftFieldRightPenaltyStretch",
        "RightFieldLeftPenaltyStretch",
        "RightFieldRightPenaltyStretch",
    ];

    #[test]
    fn field_line_set_is_the_2018_twelve() {
        for div in [Division::A, Division::B] {
            let f = FieldGeometry::division(div);
            let lines = f.lines();
            assert_eq!(lines.len(), 12, "{div:?}");
            let names: Vec<&str> = lines.iter().map(|l| l.name.as_str()).collect();
            assert_eq!(names, LINE_NAMES);
            // Every endpoint must sit on the field or its markings.
            for l in &lines {
                for p in [l.p1, l.p2] {
                    assert!(p.x.abs() <= f.half_length() + 1e-9);
                    assert!(p.y.abs() <= f.half_width() + 1e-9);
                }
                assert!((l.thickness - f.line_thickness).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn field_line_endpoints_are_correct() {
        let f = FieldGeometry::division(Division::A);
        let lines = f.lines();
        let get = |name: &str| lines.iter().find(|l| l.name == name).unwrap().clone();

        assert_eq!(get("TopTouchLine").p1, Vec2::new(-6.0, 4.5));
        assert_eq!(get("TopTouchLine").p2, Vec2::new(6.0, 4.5));
        assert_eq!(get("BottomTouchLine").p1, Vec2::new(-6.0, -4.5));
        assert_eq!(get("LeftGoalLine").p1, Vec2::new(-6.0, -4.5));
        assert_eq!(get("LeftGoalLine").p2, Vec2::new(-6.0, 4.5));
        assert_eq!(get("RightGoalLine").p1, Vec2::new(6.0, -4.5));
        // Halfway line spans the width at x = 0; centre line spans the length at y = 0.
        assert_eq!(get("HalfwayLine").p1, Vec2::new(0.0, -4.5));
        assert_eq!(get("HalfwayLine").p2, Vec2::new(0.0, 4.5));
        assert_eq!(get("CenterLine").p1, Vec2::new(-6.0, 0.0));
        assert_eq!(get("CenterLine").p2, Vec2::new(6.0, 0.0));
        // Penalty area: 1.8 deep, 3.6 wide.
        assert_eq!(get("LeftPenaltyStretch").p1, Vec2::new(-4.2, -1.8));
        assert_eq!(get("LeftPenaltyStretch").p2, Vec2::new(-4.2, 1.8));
        assert_eq!(get("RightPenaltyStretch").p1, Vec2::new(4.2, -1.8));
        assert_eq!(get("LeftFieldLeftPenaltyStretch").p1, Vec2::new(-6.0, -1.8));
        assert_eq!(get("LeftFieldLeftPenaltyStretch").p2, Vec2::new(-4.2, -1.8));
        assert_eq!(get("LeftFieldRightPenaltyStretch").p1, Vec2::new(-6.0, 1.8));
        assert_eq!(get("RightFieldLeftPenaltyStretch").p1, Vec2::new(6.0, 1.8));
        assert_eq!(get("RightFieldLeftPenaltyStretch").p2, Vec2::new(4.2, 1.8));
        assert_eq!(
            get("RightFieldRightPenaltyStretch").p1,
            Vec2::new(6.0, -1.8)
        );
    }

    #[test]
    fn field_has_exactly_one_arc() {
        let f = FieldGeometry::division(Division::A);
        let arcs = f.arcs();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].name, "CenterCircle");
        assert_eq!(arcs[0].center, Vec2::ZERO);
        assert_eq!(arcs[0].radius, 0.5);
        assert_eq!(arcs[0].a1, 0.0);
        assert!((arcs[0].a2 - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn field_walls_leave_an_opening_exactly_the_goal_width() {
        let f = FieldGeometry::division(Division::A);
        let walls = f.walls(0.1);
        let ox = f.half_length() + f.boundary_width;

        for side in [-1.0f64, 1.0] {
            let boards: Vec<&WallSegment> = walls
                .iter()
                .filter(|w| {
                    !w.is_goal
                        && (w.a.x - side * ox).abs() < 1e-12
                        && (w.b.x - side * ox).abs() < 1e-12
                })
                .collect();
            assert_eq!(boards.len(), 2, "goal-line board must be split in two");
            let mut inner: Vec<f64> = boards
                .iter()
                .map(|w| {
                    if w.a.y.abs() < w.b.y.abs() {
                        w.a.y
                    } else {
                        w.b.y
                    }
                })
                .collect();
            inner.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((inner[1] - inner[0] - f.goal_width).abs() < 1e-12);
            // The opening is centred on y = 0.
            assert!((inner[0] + inner[1]).abs() < 1e-12);
            // No board covers the middle of the goal mouth.
            for w in boards {
                let lo = w.a.y.min(w.b.y);
                let hi = w.a.y.max(w.b.y);
                assert!(!(lo < 0.0 && hi > 0.0));
            }
        }
    }

    #[test]
    fn field_walls_have_boards_and_two_goal_frames() {
        let f = FieldGeometry::division(Division::B);
        let walls = f.walls(0.1);
        let boards: Vec<&WallSegment> = walls.iter().filter(|w| !w.is_goal).collect();
        let goal: Vec<&WallSegment> = walls.iter().filter(|w| w.is_goal).collect();
        assert_eq!(
            boards.len(),
            6,
            "2 touch boards + 2x2 split goal-line boards"
        );
        // 2 goals x (2 posts x 2 faces + 1 back wall).
        assert_eq!(goal.len(), 10);
        for w in &boards {
            assert!((w.height - 0.1).abs() < 1e-12);
            assert!((w.normal.length() - 1.0).abs() < 1e-12);
        }
        for w in &goal {
            assert!((w.height - f.goal_height).abs() < 1e-12);
            assert!((w.normal.length() - 1.0).abs() < 1e-12);
        }
        // Touch-line board normals point back onto the field.
        let top = boards
            .iter()
            .find(|w| w.a.y > 0.0 && w.b.y > 0.0 && (w.a.y - w.b.y).abs() < 1e-12)
            .unwrap();
        assert_eq!(top.normal, Vec2::new(0.0, -1.0));
        // Posts extend from the goal line to goal_depth behind it.
        let post = goal
            .iter()
            .find(|w| w.a.x < 0.0 && (w.a.y - w.b.y).abs() < 1e-12)
            .unwrap();
        assert!((post.a.x + f.half_length()).abs() < 1e-12);
        assert!((post.b.x + f.half_length() + f.goal_depth).abs() < 1e-12);
        assert!(
            (post.a.y.abs() - (f.goal_width * 0.5 + f.goal_wall_thickness * 0.5)).abs() < 1e-12
        );
    }

    #[test]
    fn field_goal_containment() {
        let f = FieldGeometry::division(Division::A);
        // Inside the blue (-x) goal.
        assert_eq!(f.goal_containing(Vec2::new(-6.1, 0.0)), Some(Team::Blue));
        assert_eq!(f.goal_containing(Vec2::new(-6.17, 0.8)), Some(Team::Blue));
        // Inside the yellow (+x) goal.
        assert_eq!(f.goal_containing(Vec2::new(6.1, -0.5)), Some(Team::Yellow));
        // On the field, in front of the goal.
        assert_eq!(f.goal_containing(Vec2::new(-5.9, 0.0)), None);
        assert_eq!(f.goal_containing(Vec2::new(0.0, 0.0)), None);
        // Past the goal line but wide of the posts.
        assert_eq!(f.goal_containing(Vec2::new(-6.1, 1.0)), None);
        // Behind the back of the goal.
        assert_eq!(f.goal_containing(Vec2::new(-6.2, 0.0)), None);
        // Exactly on the goal line / post is not "in".
        assert_eq!(f.goal_containing(Vec2::new(-6.0, 0.0)), None);
        assert_eq!(f.goal_containing(Vec2::new(-6.1, 0.9)), None);
    }

    #[test]
    fn formation_fits_inside_the_own_half_without_overlap() {
        for div in [Division::A, Division::B] {
            let f = FieldGeometry::division(div);
            let robot_radius = 0.09;
            let positions: Vec<Vec2> = (0..16)
                .map(|n| default_formation(div, n).expect("formation for 0..=15"))
                .collect();
            assert_eq!(positions.len(), 16);
            assert!(default_formation(div, 16).is_none());

            // Keeper on the goal line side.
            assert!((positions[0] - Vec2::new(-f.half_length() + 0.3, 0.0)).length() < 1e-12);

            for (i, p) in positions.iter().enumerate() {
                assert!(p.x < 0.0, "{div:?} #{i} must be in the -x half: {p:?}");
                assert!(
                    p.x > -f.half_length() + robot_radius,
                    "{div:?} #{i} inside the field: {p:?}"
                );
                assert!(
                    p.y.abs() < f.half_width() - robot_radius,
                    "{div:?} #{i} inside the field: {p:?}"
                );
                assert!(
                    p.length() > f.center_circle_radius + robot_radius,
                    "{div:?} #{i} must clear the centre circle: {p:?}"
                );
            }
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    let d = (positions[i] - positions[j]).length();
                    assert!(d >= 0.2, "{div:?} #{i} and #{j} only {d} m apart");
                }
            }
        }
    }

    #[test]
    fn formation_min_separation_meets_the_spec() {
        for div in [Division::A, Division::B] {
            let positions: Vec<Vec2> = (0..16)
                .map(|n| default_formation(div, n).unwrap())
                .collect();
            let mut min = f64::INFINITY;
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    min = min.min((positions[i] - positions[j]).length());
                }
            }
            assert!(min >= 0.25, "{div:?} closest pair is {min} m apart");
        }
    }
}
