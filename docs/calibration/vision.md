# Vision calibration from recorded SSL game logs

Measured with `ssl-logtool vision <logs...> --out docs/calibration/vision.json`
(source: `crates/ssl-sim-logtool/src/vision/`). Every number below comes from
that JSON; this file is the reading of it.

**Corpus** — 10 division-A/B games, one streaming pass each (45 s wall clock for
all 10, ~410 MB/s decompressed):

| tournament | logs | field (mm) | cameras | detection frames | robot dets | ball dets |
|---|---|---|---|---|---|---|
| German Open 2026 | 6 | 12020 × 9020 | 2 | 1.797 M | 16.89 M | 0.941 M |
| RoboCup 2026 div A | 3 | 12000 × 9000 | 2 | 1.240 M | 12.98 M | 0.724 M |
| RoboCup 2026 div B | 1 | 9000 × 6000 | 1 | 0.159 M | 1.55 M | 0.155 M |
| **total** | **10** | | **19 streams** | **3.196 M** | **31.4 M** | **1.82 M** |

6.6 hours of play. Wire units (mm, s) are converted to SI on read; everything
below is metres / seconds / radians / pixels.

**How the estimators work** (and where they can lie)

* *Noise* — for each `(camera, object)` a rolling 1 s window of consecutive
  detections is fitted with a straight line and the residual standard deviation
  is taken as the per-frame noise. `slow` requires a fitted speed < 30 mm/s and
  < 20 mm end-to-end travel; `static` additionally requires referee HALT/TIMEOUT
  for > 2 s and < 5 mm travel. Detrending removes any error that is constant or
  slowly varying over a second, so these numbers are the *white* noise only —
  the static, position-dependent calibration error shows up instead in the
  camera-to-camera disagreement below, and is an order of magnitude larger.
* *Dropouts* — an object detected by a camera in both the previous and the next
  frame of **that same camera** but missing in between. This needs no external
  reference and no camera-region model, and it is exactly the estimator that
  matches the simulator's per-object per-frame Bernoulli draw.
* *Tracker as reference* — the autoref/team tracker stream is used only to
  identify which raw ball is the real one, to get robot velocities (to
  compensate the capture-time offset between cameras) and to get the ball
  height. It is a filtered, extrapolating estimate, so it is never used as a
  position ground truth. Where a number depends on it, it is labelled.
  Sources whose `timestamp` runs on a monotonic uptime clock rather than the
  vision PC's unix clock are rejected automatically (ER-Force's tracker does
  this in the RoboCup 2026 logs).

---

## 1. Timing

### Frame period

Every camera in every log runs at the **same 73.2–73.3 Hz**:

| corpus | clean mean period | rate | p99 | p99.9 | long gaps (> 1.8 T) | max gap |
|---|---|---|---|---|---|---|
| German Open (12 streams) | 13.644 ms | 73.29 Hz | 14.05–14.15 ms | 14.3–16.0 ms | 0–4 per log | 312–359 s |
| RoboCup 2026 (7 streams) | 13.653–13.678 ms | 73.11–73.24 Hz | 21.1–21.5 ms | 22.7–23.5 ms | 13–132 per log | 0–421 s |

"clean mean" excludes gaps longer than 3 periods. The per-log *average* rate
printed by the tool (60.4–73.3 Hz) is lower than 73.3 only because most games
contain **one multi-minute vision outage** (303–421 s); the cameras themselves
never change rate. Jitter around the nominal period is tiny in the German Open
(σ = 0.14–0.25 ms) and ~5× larger at RoboCup 2026 (σ = 1.24–1.34 ms, with ~1 %
of frames arriving one-and-a-half periods late).

Sanity checks that all came back clean: **0** negative capture-time gaps, **0**
zero gaps, `frame_number` increments by exactly +1 on 3,195,926 of 3,195,941
consecutive pairs (15 jumps, all at the outages), **148** frames with no
detection at all (all in one log), **0** geometry changes in 18,609 geometry
packets.

### Processing time `t_sent − t_capture`

| corpus | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| German Open | 5.6–6.8 ms | | | 8.3–10.4 ms | 16–34 ms |
| RoboCup 2026 | 8.7–8.9 ms | | | 11.6–13.2 ms | 25–28 ms |
| all | **7.26 ms** | 6.75 ms | 8.94 ms | 10.48 ms | 36.5 ms |

### Transport latency

The logger writes a wall-clock receive timestamp, but the logger clock is *not*
synchronised to the vision PC in these recordings — the per-log median of
`recv − t_sent` ranges from −121 s to +4.4 s. Only the spread is meaningful:
`p99 − p1` of `recv − t_sent` is **1.0 ms** (best log) to **67 ms** (worst),
mostly logger scheduling rather than network. Absolute vision latency is
therefore **not observable from a log**; see §9.

### Camera phase

Cross-camera capture-time offsets, measured by pairing each frame with the
nearest frame of the other camera:

| corpus | σ(Δt) | p1 … p99 | verdict |
|---|---|---|---|
| German Open (6 logs) | 3.80–4.73 ms | −6.75 … +6.76 ms | **free-running** — a uniform phase over one period has σ = T/√12 = 3.94 ms, which is what is measured; the offset covers the whole ±T/2 |
| RoboCup 2026 (3 two-camera logs) | 1.53–2.16 ms | −6.5 … +8.9 ms | **phase-locked**, with a per-log constant offset of +0.17, +1.36 and −5.83 ms (p50) |

So both regimes exist in the wild. Our model emits all cameras at one instant,
which matches RoboCup 2026 with a 0 ms offset but not the German Open at all.

---

## 2. Camera geometry

| log set | cam | `derived_camera_world_t` (m) | focal (px) | principal point (px) | image (px) | distortion |
|---|---|---|---|---|---|---|
| German Open | 0 | (−2.454, −0.132, **5.968**) | 721.0 | (612, 512) | 1224 × 1024 | −1.86e−3 |
| German Open | 1 | (+2.295, −0.051, **6.132**) | 1478.0 | (1224, 1024) | 2448 × 2048 | −1.32e−5 |
| RC 2026 div A | 0 | (−2.515, +0.291, **6.478**) | 1416.5 | (1274, 1057) | 2448 × 2048 | −2.23e−3 |
| RC 2026 div A | 1 | (+1.429, +0.272, **6.518**) | 1424.0 | (1252, 819) | 2448 × 2048 | −4.04e−3 |
| RC 2026 div B | 0 | (−0.004, +0.072, **6.403**) | 1473.7 | (1226, 1024) | 2448 × 2048 | 0 |

Notes that matter for the model:

* **Camera height is 5.97–6.52 m**, not the 4.0 m our `default_camera_height`
  assumes.
* Two-camera rigs split along **x only**, at x ≈ ±2.4 m — noticeably *inside*
  the ±L/4 = ±3.0 m our `auto_cameras` uses, and **not symmetric** at RoboCup
  2026 (−2.515 / +1.429, i.e. the pair midpoint is at x = −0.54 m).
* The German Open rig mixes a 1224 × 1024 and a 2448 × 2048 camera. Any model
  that assumes one focal length for the whole rig is wrong there — and the ball
  `area` a camera reports scales with *its* focal length squared.
* The principal point is not always the image centre (RC 2026 cam 1: (1252, 819)
  on a 2448 × 2048 sensor). Our geometry packet hard-codes (300, 300).
* Geometry cadence: the German Open sends **both** calibrations in every packet
  **every 1.0010 s** (σ = 0.4 ms); RoboCup 2026 sends **one** calibration per
  packet every 1.5013 s on the two-camera fields (so 3 s per camera) and every
  3.0025 s on the single-camera field. Nothing ever changed mid-game.

### Visible region and overlap

Coverage is measured as a 100 mm grid of robot detections per camera, thresholded
at 2 % of that camera's 99th-percentile cell count.

| log | cam 0 dense bbox (m) | cam 1 dense bbox (m) | overlap area frac | band width along x, p50 / p90 / max |
|---|---|---|---|---|
| GO 2026-03-12_16-30 | x −6.55…+1.65, y ±4.75 | x −1.45…+6.55 | 0.124 | 0.70 / 1.40 / 1.80 m |
| GO 2026-03-13_11-33 | x −6.55…+1.65 | x −1.45…+6.25 | 0.160 | 0.80 / 1.80 / 3.00 m |
| GO 2026-03-14_09-01 | x −6.45…+1.65 | x −1.45…+6.45 | 0.241 | 1.40 / 3.10 / 3.20 m |
| RC 2026-07-02 | x −6.55…+2.35 | x −2.05…+6.45 | 0.083 | 0.10 / 0.40 / 0.80 m |
| RC 2026-07-04_17-31 | x −6.55…+2.45 | x −2.05…+6.45 | 0.175 | 1.00 / 1.80 / 2.70 m |

The bounding boxes overlap by 3.1 m (German Open) to 4.5 m (RoboCup 2026), but
the region where *both* cameras produce detections at a useful density is a band
of only **0.1–3.1 m** wide, and its width varies strongly with y (wide near the
centre line, narrow at the goal ends) — the real overlap is lens-shaped, not a
strip.

Against our ER-Force Manhattan rule (visible iff `manhattan ≤ min + 2 ·
camera_overlap`), the implied `camera_overlap` is **0.05–0.80 m** from the median
band and **0.20–1.55 m** from the p90 band; the aggregate the tool reports is
**0.87 m**.

The Manhattan rule also over-reaches: robots are still detected out to ~6.5 m
from a camera's nadir, but beyond that they are not (see §4), whereas the
Manhattan cell of a two-camera 12 × 9 m field extends to 8.9 m.

---

## 3. Detection noise

All figures are per-frame residual σ after removing a 1 s linear trend.

| quantity | static referee | slow (any referee) | per-camera range (static) | windows |
|---|---|---|---|---|
| robot x, y | **0.42 mm** | 0.66 / 0.68 mm | 0.25 – 0.49 mm | 138 498 |
| robot orientation φ | **4.29 mrad** | 6.32 mrad | 3.05 – 5.56 mrad | 138 498 |
| ball (at rest, tracker-gated) | — | **0.71 mm** | 0.53 – 0.96 mm | 17 611 |
| ball `area` (at rest) | — | **3.30 px** (5.4 % of mean area) | 2.86 – 4.57 px | 9 184 |

The `slow` numbers are inflated by robots that were actually creeping (a 1 s
linear fit does not absorb real acceleration); `static` is the cleaner estimate
of the sensor noise, `slow` is the practical upper bound.

### Noise versus distance from the camera nadir

This was the hypothesis to test, and the data says **no**:

| nadir distance (m) | 0.25 | 1.25 | 2.25 | 3.25 | 4.25 | 5.25 | 6.25 | 6.75 |
|---|---|---|---|---|---|---|---|---|
| robot σ_p (mm) | 0.49 | 0.50 | 0.61 | 0.76 | 0.76 | 0.60 | 0.59 | 0.79 |
| robot σ_φ (mrad) | 6.26 | 5.63 | 5.83 | 6.45 | 7.32 | 6.48 | 5.16 | 6.83 |
| ball σ (mm) | 0.76 | 0.85 | 0.71 | 0.58 | 0.71 | 0.66 | 0.74 | 0.54 |

Flat within ±0.2 mm / ±1 mrad across the full 0–7 m range. A single
distance-independent Gaussian is the right model; adding a distance term would
be fitting noise. (Orientation noise *does* rise slightly, ~6.0 → 7.3 mrad,
between 3.5 m and 5 m, but not monotonically.)

### Confidence and height

* Robot `confidence`: mean 0.896, σ 0.080, min 0.20; 48 % of detections above
  0.95, essentially none below 0.5.
* Ball `confidence`: mean 0.913, σ 0.121 — but strongly bimodal by venue. The
  German Open peaks at 0.65–0.75 and reaches down to 0.2; RoboCup 2026 sits at
  0.90–1.00. We report a constant 1.0.
* Robot `height`: 135–150 mm, mean 146.2 mm (135 / 140 / 150 mm clusters).
  Always present.
* Ball `z`: **never reported** — 0 of 1,820,478 ball detections. `report_ball_z
  = false` is correct.
* Ball `area`: present on all 879,111 RoboCup 2026 detections, **absent on all
  941,367 German Open detections**. Consumers must cope with a missing `area`.

---

## 4. Dropouts

| estimator | robots | balls |
|---|---|---|
| single-frame (seen in prev and next frame of the same camera) | **0.169 %** | **1.43 %** |
| including multi-frame gaps in a detection run | 0.79 % | 6.21 % |
| tracker says the object is in this camera's region, within 5 m of nadir | 0.46 % | 14.6 % (includes occlusion) |

Per camera, the single-frame robot dropout is 0.003–0.012 % in the German Open
and 0.07–0.74 % at RoboCup 2026 — a 30–100× venue difference. Gap-length
histogram: 1-frame gaps dominate (roughly 5:1 over 2-frame).

For the ball the single-frame dropout is almost entirely occlusion, and splitting
it by the distance to the nearest robot shows that clearly:

| ball → nearest robot (m) | 0.025 | 0.075 | 0.125 | 0.175 | 0.275 | 0.375 | 0.575 |
|---|---|---|---|---|---|---|---|
| single-frame dropout | 8.7 % | 4.4 % | 1.8 % | 1.3 % | 0.70 % | 0.72 % | 0.48 % |

So the *unconditional* ball dropout, with no robot nearby, is **0.5–0.7 %**; the
rest is occlusion, which we model separately.

### The field-of-view edge

Gating on the tracker ("the tracker says this robot is inside this camera's
Manhattan region — did the camera report it?") and binning by the distance from
the camera's nadir exposes where a real camera actually stops seeing:

| nadir distance (m) | 0.25 | 1.75 | 2.25 | 3.25 | 4.25 | 5.25 | 6.25 | **6.75** | **7.25** | 7.75 |
|---|---|---|---|---|---|---|---|---|---|---|
| robot miss rate | 0.16 % | 0.58 % | 0.99 % | 0.14 % | 0.90 % | 1.23 % | 0.39 % | **22.1 %** | **92.5 %** | 95.8 % |
| samples | 358 k | 4 796 k | 3 414 k | 3 309 k | 2 610 k | 507 k | 422 k | 264 k | 199 | 24 |

Flat below 1.3 % out to 6.5 m, then a cliff. That cliff is the lens, not a
dropout, and it is the thing the Manhattan region rule does not model.

**Duplicate detections of the same robot id in one frame: 932** out of 31.4 M
robot detections (3.0e−5). Rare, but real, and our model never produces them.

---

## 5. Multi-camera disagreement

Same robot id seen by both cameras within 0.6 of a frame period, with the
capture-time offset compensated using the tracked velocity, mismatches beyond
0.15 m rejected (872 of 3.23 M):

| log | Δx mean ± σ (mm) | Δy mean ± σ (mm) | \|Δ\| mean / p50 / p90 / p99 (mm) | Δφ σ (mrad) | gross Δφ > 0.5 rad |
|---|---|---|---|---|---|
| GO 2026-03-12_16-30 | −6.6 ± 12.8 | −6.7 ± 7.8 | 15.4 / 14.7 / 22.7 / 50.7 | 31.3 | 6 |
| GO 2026-03-13_13-32 | −1.3 ± 14.6 | −5.9 ± 9.4 | 15.1 / 12.8 / 29.7 / 51.6 | 27.1 | 0 |
| GO 2026-03-14_11-19 | +0.2 ± 15.1 | −4.5 ± 9.5 | 15.6 / 13.2 / 28.7 / 51.1 | 26.8 | 0 |
| RC 2026-07-02 | **+25.8 ± 10.2** | −9.4 ± 7.1 | 28.3 / 26.2 / 42.9 / 59.6 | 29.8 | 40 |
| RC 2026-07-03 | **−20.4 ± 8.4** | +7.3 ± 6.5 | 22.8 / 21.0 / 34.0 / 47.9 | 27.6 | 15 |
| RC 2026-07-04_17-31 | **−23.6 ± 9.7** | +6.8 ± 7.0 | 25.6 / 22.9 / 41.2 / 52.6 | 25.0 | 2 |
| **aggregate** | −2.7 ± 20.0 | −1.9 ± 10.7 | **20.3** / 15.9 / 31.7 / 54.0 | **28.9** | 75 |

This is the headline result of the whole study: **each camera's per-frame noise
is 0.4 mm, but the two cameras disagree about where the same robot is by 20 mm
rms.** The error that matters for a real team's world model is a static,
position-dependent calibration error two orders of magnitude larger than the
white noise, and at RoboCup 2026 it had a clear **constant component: a 2.0–2.6
cm offset along x between the two cameras** (σ of only 8–10 mm around it). The
German Open rig had no such constant offset (±7 mm) but a larger spatially
varying part (σ 13–15 mm).

Orientation disagrees by 25–32 mrad rms (p1…p99 ≈ ±60 mrad) versus 4.3 mrad of
per-frame noise — same story. 75 gross disagreements (> 0.5 rad) out of 3.23 M
comparisons are pattern misreads.

Measured against the *tracker* instead (which fuses both cameras, so common-mode
cancels), the per-camera radial bias is only **+0.68 mm** outward, mean;
tangential −0.50 mm. That confirms the disagreement is between cameras rather
than a global scale error.

---

## 6. Ball

### Multiplicity and spurious detections

Balls per detection frame, all logs: **0 → 44.9 %**, 1 → 53.4 %, 2 → 1.57 %,
3 → 0.090 %, 4 → 0.024 %, ≥5 → 0.008 % (max 8).

For every multi-ball frame the real ball is identified by matching the tracker's
primary ball; the remainder are "extra". Their position in the nearest robot's
own frame (x forward, y left) is the test of the break-beam-LED hypothesis:

| log set | rate (per robot per second) | extras within 0.3 m of a robot | forward (m) | lateral (m) | extra `area` | real ball `area` |
|---|---|---|---|---|---|---|
| German Open (6 logs) | **0.042 – 0.089** | 1625 – 5156 per log | **+0.119 … +0.141** ± 0.07–0.12 | **−0.036 … +0.008** ± 0.06–0.09 | n/a | n/a |
| RoboCup 2026 (4 logs) | **0.0015 – 0.0063** | 32 – 160 per log | −0.092 … +0.065 ± 0.11–0.15 | −0.053 … +0.042 ± 0.14–0.17 | 24–38 px | 55–68 px |

**The hypothesis is confirmed for the German Open and rejected for RoboCup
2026.** In the German Open the extra balls sit on the robot's centreline
(lateral 0.000 ± 0.07 m) at **0.12–0.14 m in front of the robot centre** — i.e.
4.5–6.5 cm in front of the kicker face at `center_to_dribbler` = 0.075 m. The
aggregate forward histogram has sharp spikes at +0.095 m and +0.145 m and a
third at +0.255 m. That is exactly the geometry of an IR break-beam emitter
firing into the camera, and the German Open vision reported **no `area` field at
all**, so a size filter could not reject it. At RoboCup 2026, where `area` is
reported, the extras are 2.6× smaller than a real ball (32 px vs 63 px) and are
scattered rather than dribbler-localised: generic false positives, 30× rarer.

Aggregate: 9,900 extra detections attributable to a robot over 460,273
robot-seconds of observation = **0.0343 per robot per second**.

### `area` model

| quantity | value |
|---|---|
| mean reported `area`, ball on the floor | **63.1 px** (σ 12.1 px, range 16–264) |
| `k = area · d²`, d = 3-D distance to the camera | 3565 ± 928 px·m² |
| implied focal length at `PIXEL_PER_AREA = 10` | **495 px** |
| implied focal length at `PIXEL_PER_AREA = 1` | **1567 px** vs a calibrated 1416–1474 px |

`area` versus distance from the nadir (aggregate, RoboCup 2026 only since the
German Open reports none):

| nadir distance (m) | 0.25 | 1.25 | 2.25 | 3.25 | 4.25 | 5.25 | 5.75 | 6.25 | 6.75 | 7.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean `area` (px) | 62.6 | 59.8 | 68.2 | 68.4 | 66.7 | 60.4 | 56.7 | 55.1 | 48.7 | 36.2 |

**The 1/d² law is not visible in the data.** Going from the nadir to 5 m out at a
6.5 m camera height the range grows from 6.5 m to 8.2 m, so a pure pinhole model
predicts the area should fall by 37 % — instead it is flat to within ±8 % and
only collapses past 5.5 m, at the edge of the field of view. The obliquity of a
flat sensor (a sphere projects to an ellipse of area ∝ 1/cos³θ, which is ×2.1 at
38°) very nearly cancels the 1/d² term over the whole working area; barrel
distortion (k ≈ −2e−3) trims the rest.

Against the *ball height* the pinhole law does hold, using the tracker's ball z:

| tracker ball z (m) | 0.025 | 0.125 | 0.275 | 0.425 | 0.575 | 0.825 | 0.975 |
|---|---|---|---|---|---|---|---|
| mean `area` (px) | 63.1 | 72.1 | 79.8 | 83.4 | 88.0 | 92.4 | 99.9 |

A pure 1/(H − z)² with H = 6.45 m predicts 63.1 → 76 px at z = 0.575 m and 85 px
at z = 0.9 m; measured 88 px and 93–100 px (thin samples above 0.6 m). The height
term is therefore the right shape, running ~15 % steeper than the ideal pinhole,
and it is the only part of the geometry that should drive `area`.

### Ball visibility versus occlusion

P(the camera that owns the region and has the ball within 5 m of its nadir
reports a ball within 0.15 m of the tracked ball), by distance from the ball to
the nearest robot centre:

| ball → nearest robot (m) | 0.01 | 0.05 | 0.07 | 0.09 | 0.11 | 0.13 | 0.15 | 0.19 | 0.25 | 0.31 | 0.45 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P(detected) | 0.019 | 0.030 | 0.180 | **0.630** | **0.731** | 0.787 | 0.845 | 0.909 | 0.954 | 0.931 | 0.969 |
| samples | 2 217 | 7 046 | 14 869 | 57 930 | 54 628 | 71 848 | 44 936 | 27 612 | 39 501 | 18 671 | 29 975 |

Under 0.06 m the "ball" is under the robot (the tracker is extrapolating) and is
never seen. The interesting range is 0.09–0.13 m — a ball on the dribbler —
where detection drops to **0.63–0.79**.

The direct dribbling measurement (tracker ball within 0.11 m of a robot centre,
in front of it, and that same camera reported that robot in that frame):

| corpus | P(ball detected while dribbled) | P(detected, free, > 0.3 m from any robot) | ratio |
|---|---|---|---|
| German Open | 0.316 – 0.629 | 0.661 – 0.941 | ≈ 0.55 |
| RoboCup 2026 | 0.607 – 0.853 | 0.751 – 0.993 | ≈ 0.82 |
| aggregate | **0.571** | **0.891** | **0.64** |

Also worth recording: P(detected) versus *ball speed* is **lowest for a slow
ball** (0.85 below 0.25 m/s) and rises to 0.94–0.96 above 2 m/s. Motion blur is
not a problem for a 73 Hz camera; a slow ball is a ball being dribbled.

---

## 7. Referee split

Across the 10 logs (seconds): HALT 7313, STOP 2612, ball placement 5781,
timeouts 1512, direct free kicks 4292, normal start 720, force start 329,
kickoff/penalty prep 351. 1,128,177 of 3,195,960 camera frames (35 %) fall under
a referee command that has been HALT or TIMEOUT for more than 2 s, which is what
feeds the `static` noise estimator.

---

## 8. Recommended `Realism` / `VisionConfig` values

Bold = change from our current `Realism::realistic()` / `VisionConfig::default()`.

| parameter | current default | **measured / recommended** | evidence |
|---|---|---|---|
| `stddev_ball_p` | 0.0014 | **0.0007** | §3, ball at rest, 0.53–0.96 mm per camera |
| `stddev_robot_p` | 0.0013 | **0.0005** | §3, 0.42 mm static / 0.66 mm slow |
| `stddev_robot_phi` | 0.01 | **0.005** | §3, 4.29 mrad static / 6.32 mrad slow |
| `stddev_ball_area` | 6.5 | **3.3** | §3, detrended area residual on a resting ball (5.4 % of 63 px) |
| `missing_robot_detections` | 0.02 | **0.002** | §4, single-frame estimator 0.00169; 0.0079 including multi-frame gaps |
| `missing_ball_detections` | 0.05 | **0.007** | §4, single-frame dropout far from any robot; the raw 0.0143 includes occlusion, which we model separately |
| `dribbler_ball_detections` | 0.05 | **0.02** (venue range 0.0015–0.089) | §6; keep 0.05 for a "no area filter" preset, 0.002 for a modern one |
| `camera_overlap` | 1.0 | **0.8** | §2, p90 overlap band width / 2; range 0.05–1.55 m |
| `object_position_offset` | 0.02 | **0.012** | §5, 20.3 mm mean pairwise disagreement ⇒ ~10 mm per camera; RoboCup 2026 shows a 2.0–2.6 cm constant inter-camera offset |
| `camera_position_error` | 0.1 | **0.02** | not observable from a log (we never see the true camera pose). Only affects the advertised geometry; keep it small and let `object_position_offset` carry the visible error |
| `vision_delay` | 0.035 | **0.022** | §1: 7.3 ms measured `t_sent − t_capture` plus a conventional 15 ms of camera exposure/readout ahead of `t_capture` and LAN hop, which a log cannot show |
| `vision_processing_time` | 0.010 | **0.0073** | §1, directly measured (5.6–8.9 ms by venue) |
| `frame_rate` | 60.0 | **73.3** | §1, 13.644–13.678 ms period on all 19 streams |
| `cameras` / `default_camera_count` | 4 | **2** (1 for division B) | §2; every division-A field in the corpus used exactly 2 |
| `ball_visibility_threshold` | 0.4 | keep 0.4, then **tune to hit P = 0.65 at 0.10 m and P = 0.93 at > 0.25 m** | §6 visibility table; the absolute threshold is only meaningful against our own occlusion sampler, so calibrate against the curve, not the number |

Additional `VisionConfig` values the data pins down:

| parameter | current | **recommended** | evidence |
|---|---|---|---|
| `default_camera_height` | 4.0 | **6.4** | §2, 5.97–6.52 m in every log |
| `focal_length_px` | 390 | **450** (or 1420 with `PIXEL_PER_AREA = 1`) | §6; 390 px under-reports `area` by 31 % |
| `PIXEL_PER_AREA` | 10.0 | **1.0** | §6: with `PIXEL_PER_AREA = 1` and the real calibrated focal length (1416 px) the formula gives 68.9 px against a measured 63.1 px — 9 % — so the reported `area` is simply the pixel area, and the ×10 fudge forces `focal_length_px` to be unphysical |
| `geometry_every_n_frames` | 30 | **73** (1.00 s, German Open) or **110** (1.50 s, RoboCup 2026) | §2, measured intervals 1.0010 s / 1.5013 s / 3.0025 s |
| `report_ball_z` | false | **false** (confirmed) | §3, 0 of 1.82 M detections carry `z` |
| camera x placement | ±L/4 = ±3.005 m | **±2.4 m** | §2 |

---

## 9. Model changes the data demands

1. **Per-camera frame phase.** Six of the ten logs have free-running cameras
   whose capture instants are uniformly distributed over the frame period
   (σ = 3.94 ms = T/√12, matched to 3 %); the other four are hardware-locked to
   within 1.5–2.2 ms with a per-log constant offset of up to 5.8 ms. We emit
   every camera at one instant. Add a per-camera phase offset with two modes
   (locked with a fixed offset, or free-running) — a consumer that fuses two
   cameras behaves measurably differently in the two regimes. *§1.*

2. **The `area` formula's distance term is wrong in the horizontal direction.**
   Reported `area` is flat (±8 %) from the nadir out to 5 m and then falls off a
   cliff, where our pinhole 1/d² predicts a 37 % monotone decline. The flat
   sensor's 1/cos³θ obliquity cancels the range term over the working area.
   Either drop the horizontal term (report `A0 · visibility`, `A0 ≈ 63 px`) or
   add the obliquity factor. The *height* term is correct and should be kept.
   *§6.*

3. **`PIXEL_PER_AREA = 10` is not a real scale factor.** The reported `area` is
   the raw blob pixel area: with `PIXEL_PER_AREA = 1` and the camera's real
   calibrated focal length our own formula lands within 9 % of the measured
   mean. Setting the constant to 1 lets `focal_length_px` be the physical
   calibration value instead of a compensating fudge. *§6.*

4. **Noise does *not* grow with distance from the nadir.** Robot σ_p stays in
   0.49–0.79 mm and σ_φ in 5.2–7.3 mrad across the whole 0–7 m range; ball σ
   stays in 0.54–0.85 mm. A single distance-independent Gaussian is right — do
   not add a radial term. *§3.*

5. **White noise is not the dominant error; inter-camera calibration error is.**
   Per-frame noise is 0.4 mm but the two cameras place the same robot 20 mm apart
   (28.9 mrad in orientation). Our `object_position_offset` is the right knob but
   it currently applies a *purely radial, purely constant* offset; the data shows
   a constant part (2.0–2.6 cm along x at RoboCup 2026) **plus** a
   spatially-varying part of similar size (σ 8–20 mm). Consider a smooth
   per-camera 2-D warp (e.g. a low-order polynomial in field position seeded per
   camera) instead of a single radial constant. *§5.*

6. **The camera region is a disc/rectangle around the nadir, not a Manhattan
   cell.** Robot detection completeness is flat at 98.8–99.8 % out to 6.5 m from
   the nadir, then falls to 77.9 % in the 6.75 m bin and 7.5 % beyond 7.25 m — a
   hard field-of-view edge. The Manhattan rule extends a two-camera 12 × 9 m field to
   8.9 m, so we generate detections in places real cameras never see. Clip the
   region by a nadir radius (or an image-plane test with the real focal length
   and image size) in addition to the Manhattan rule. *§2, §4.*

7. **The spurious dribbler ball sits on the centreline, not at a corner.** Our
   `dribbler_corner()` offsets the false ball by half a dribbler width to the
   right. Measured lateral offset is 0.000 ± 0.07 m and the forward offset is
   0.119–0.141 m from the robot centre (spikes at +0.095 and +0.145 m), i.e.
   ~2–7 cm in front of the kicker face. Drop the lateral offset and widen the
   forward one. *§6.*

8. **The spurious-ball rate is a property of the vision configuration, not of
   robots.** It varies 60× between venues (0.0015 vs 0.089 per robot per second)
   and correlates exactly with whether `area` is reported at all. Expose it as a
   preset dimension rather than a single default. *§6.*

9. **`area` is optional on the wire and was absent for a whole tournament.**
   941,367 German Open ball detections carry no `area`. Anything downstream that
   assumes `area` is present will silently break on real data — and our
   simulator always emits it, so it never exercises that path. *§3.*

10. **Duplicate robot ids within a single frame happen** (932 in 31.4 M
    detections) and our model cannot produce them. Low priority, but a consumer
    that assumes uniqueness per frame will crash on a real log, not on ours.
    *§4.*

11. **Confidence is not 1.0.** Robot confidence averages 0.896 (down to 0.20) and
    ball confidence is bimodal by venue (German Open 0.65–0.75 typical, RoboCup
    2026 0.90–1.00). We emit a constant 1.0. If any consumer weights by
    confidence, our stream is unrepresentative. *§3.*

12. **Robot `height` is 135–150 mm and varies per robot**, not a single value —
    fine as-is, but worth driving from `RobotSpecs::height` rather than a
    constant. *§3.*

13. **Multi-minute vision outages are normal.** Nine of ten games contain a
    303–421 s gap in the detection stream, plus 13–132 single dropped frames per
    RoboCup 2026 game. Nothing in our model produces an outage; a team's world
    model should be tested against one. *§1.*

---

## 10. Limitations

* **The tracker is a filter, not truth.** Every number gated on the tracker
  (ball at rest, ball height, ball visibility, region completeness, velocity
  compensation for the camera-pair comparison) inherits its smoothing and its
  extrapolation through occlusions. The ball-visibility numbers are the worst
  affected: when the tracker holds a ball position through a long occlusion, the
  "camera missed it" counter fires even though there was nothing to see. That is
  why the headline dropout rates use the tracker-free prev/next estimator, and
  why the visibility table is presented as a *curve* (whose shape is robust)
  rather than a single number.
* **Detrending removes real signal.** A 1 s linear fit absorbs any error that
  varies slowly, so the σ values in §3 are a lower bound on the total position
  error. §5 is the complementary upper-bound view.
* **Absolute latency is unmeasurable.** The logger clock is not synchronised to
  the vision PC in any of these recordings (offsets from −121 s to +4.4 s), and
  `t_capture` itself is a camera-driver timestamp whose relation to the shutter
  is unknown. Only `t_sent − t_capture` is solid; the 15 ms added to reach
  `vision_delay` is convention, not measurement.
* **Two venues, two vision configurations.** Almost every "surprising" number
  (spurious ball rate, `area` presence, camera phase lock, dropout rate,
  confidence distribution) splits cleanly by tournament rather than varying
  within one. Treat the aggregate as a midpoint between two presets, not as a
  universal constant.
* **Small camera counts.** No log in the corpus has more than 2 cameras, so the
  overlap and region conclusions are only tested for a 1- and 2-camera split
  along x. A 4-camera division-A rig may behave differently.
* **Quantile merging is approximate.** Per-log quantiles are exact (reservoirs
  hold 200 k samples); the aggregate quantiles concatenate reservoirs, so they
  are weighted by reservoir size rather than by sample count. Aggregate means
  and standard deviations are exact (Welford merge).
