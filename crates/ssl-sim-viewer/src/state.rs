//! Shared world snapshot fed by the receiver threads and read by the UI.
//!
//! Everything in here is wire-shaped: detection coordinates stay in the
//! millimetres the SSL vision protocol uses, tracker coordinates stay in
//! metres. Conversion to drawing space happens in [`crate::view`].

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use ssl_sim_proto::{
    sim::{SslDetectionBall, SslDetectionFrame, SslDetectionRobot, SslGeometryData},
    tracked::{TrackedFrame, TrackerWrapperPacket},
};

/// A stream is considered live if a packet arrived within this long.
pub const LIVE_TIMEOUT: Duration = Duration::from_millis(1500);

/// Length of the sliding window used for the packet-rate counters.
const RATE_WINDOW: Duration = Duration::from_millis(1000);

/// Sliding-window packet rate counter.
#[derive(Debug, Clone)]
pub struct RateCounter {
    window_start: Instant,
    window_count: u32,
    /// Packets per second over the last completed window.
    pub hz: f32,
    /// Total packets seen since start.
    pub total: u64,
    /// When the most recent packet arrived.
    pub last: Instant,
}

impl RateCounter {
    /// A counter that has just seen its first packet at `now`.
    pub fn new(now: Instant) -> Self {
        Self {
            window_start: now,
            window_count: 1,
            hz: 0.0,
            total: 1,
            last: now,
        }
    }

    /// Record one packet received at `now`.
    pub fn tick(&mut self, now: Instant) {
        self.total += 1;
        self.window_count += 1;
        self.last = now;
        let elapsed = now.saturating_duration_since(self.window_start);
        if elapsed >= RATE_WINDOW {
            self.hz = self.window_count as f32 / elapsed.as_secs_f32();
            self.window_start = now;
            self.window_count = 0;
        }
    }

    /// Whether a packet arrived recently enough for the stream to count as up.
    pub fn is_live(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.last) < LIVE_TIMEOUT
    }
}

/// The latest detection frame from one camera plus its rate counter.
#[derive(Debug, Clone)]
pub struct CameraState {
    /// Most recent detection frame from this camera.
    pub frame: SslDetectionFrame,
    /// Packet rate for this camera.
    pub rate: RateCounter,
}

/// Ground truth from the optional tracker stream.
#[derive(Debug, Clone)]
pub struct TruthState {
    /// Most recent tracked frame.
    pub frame: TrackedFrame,
    /// Name reported by the source, if any.
    pub source: Option<String>,
    /// Packet rate for the tracker stream.
    pub rate: RateCounter,
}

/// Everything the UI draws, updated by the receiver threads.
#[derive(Debug, Default)]
pub struct State {
    /// Latest geometry packet seen on the vision stream.
    pub geometry: Option<SslGeometryData>,
    /// Latest detection frame per camera id.
    pub cameras: BTreeMap<u32, CameraState>,
    /// Latest ground-truth frame, if the tracker stream is enabled.
    pub truth: Option<TruthState>,
    /// Vision packets that failed to decode.
    pub vision_decode_errors: u64,
    /// Tracker packets that failed to decode.
    pub truth_decode_errors: u64,
    /// Last socket/decode error message, for the status panel.
    pub last_error: Option<String>,
}

/// One detection with the camera that reported it.
#[derive(Debug, Clone, Copy)]
pub struct Seen<T> {
    /// Camera that reported the detection.
    pub camera_id: u32,
    /// The detection itself.
    pub value: T,
}

/// Detections merged across all cameras, ready for drawing.
#[derive(Debug, Default, Clone)]
pub struct Merged {
    /// All balls from all cameras.
    pub balls: Vec<Seen<SslDetectionBall>>,
    /// All blue robots from all cameras.
    pub blue: Vec<Seen<SslDetectionRobot>>,
    /// All yellow robots from all cameras.
    pub yellow: Vec<Seen<SslDetectionRobot>>,
}

impl State {
    /// Fold a decoded vision wrapper packet into the state.
    ///
    /// Geometry is latched (the simulator only sends it every N frames) and
    /// detection frames are kept per camera, newest wins.
    pub fn apply_wrapper(&mut self, packet: ssl_sim_proto::sim::SslWrapperPacket, now: Instant) {
        if let Some(geometry) = packet.geometry {
            self.geometry = Some(geometry);
        }
        if let Some(frame) = packet.detection {
            let camera_id = frame.camera_id;
            match self.cameras.get_mut(&camera_id) {
                Some(camera) => {
                    camera.frame = frame;
                    camera.rate.tick(now);
                }
                None => {
                    self.cameras.insert(
                        camera_id,
                        CameraState {
                            frame,
                            rate: RateCounter::new(now),
                        },
                    );
                }
            }
        }
    }

    /// Fold a decoded tracker wrapper packet into the state.
    pub fn apply_tracker(&mut self, packet: TrackerWrapperPacket, now: Instant) {
        let Some(frame) = packet.tracked_frame else {
            return;
        };
        match &mut self.truth {
            Some(truth) => {
                truth.frame = frame;
                truth.source = packet.source_name;
                truth.rate.tick(now);
            }
            None => {
                self.truth = Some(TruthState {
                    frame,
                    source: packet.source_name,
                    rate: RateCounter::new(now),
                });
            }
        }
    }

    /// Drop detection frames from cameras that went quiet, so the merged view
    /// does not keep drawing stale robots forever.
    pub fn merged(&self, now: Instant) -> Merged {
        let mut merged = Merged::default();
        for (&camera_id, camera) in &self.cameras {
            if !camera.rate.is_live(now) {
                continue;
            }
            merged
                .balls
                .extend(camera.frame.balls.iter().map(|value| Seen {
                    camera_id,
                    value: *value,
                }));
            merged
                .blue
                .extend(camera.frame.robots_blue.iter().map(|value| Seen {
                    camera_id,
                    value: *value,
                }));
            merged
                .yellow
                .extend(camera.frame.robots_yellow.iter().map(|value| Seen {
                    camera_id,
                    value: *value,
                }));
        }
        merged
    }

    /// Whether any camera has produced a packet recently.
    pub fn vision_live(&self, now: Instant) -> bool {
        self.cameras.values().any(|c| c.rate.is_live(now))
    }
}
