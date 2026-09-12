# AI Driver Drowsiness Detection System

A real-time driver drowsiness detection system that works in normal lighting **and**
in low-light / dark conditions, using a modified NoIR webcam with 850 nm infrared
illumination.

Final-year B.E. project. Built and committed stage by stage — the commit history is
the development log.

> **Status: Stage 5 of 15 complete** (camera → landmarks → EAR / MAR → head pose → frame validity → eye crops ready for the CNN).
> No detection results are reported yet. Every number published in this README will
> come from a real experiment; nothing is estimated or copied from other papers.

---

## 1. Objective

Detect driver drowsiness early enough to warn the driver, by combining three
independent evidence layers instead of relying on any single cue:

| Layer | Signals | Why it is needed |
|---|---|---|
| **Geometric** | EAR, MAR, head pose | Fast, interpretable, no training data required |
| **Machine learning** | Eye-state CNN (OPEN / CLOSED) | Robust where EAR fails: glasses, squinting, odd angles, poor light |
| **Temporal** | PERCLOS, blink duration, yawn rate, head-nod count over a 60 s sliding window | A single closed-eye frame is a blink, not drowsiness |

A state machine fuses the three layers into **ALERT → MILD → DROWSY**, with
hysteresis so the state cannot oscillate around a threshold.

## 2. Target architecture

```
Camera (laptop webcam → USB webcam → modified NoIR webcam + 850 nm IR)
   ↓
OpenCV  →  grayscale preprocessing
   ↓
MediaPipe Face Mesh
   ↓
┌────────────┬────────────┬────────────┐
│  EAR/MAR   │  Eye CNN   │ Head pose  │
└────────────┴────────────┴────────────┘
   ↓
Temporal aggregation (60 s sliding window)
   ↓
PERCLOS / blink duration / yawn rate / nod count / invalid-frame rate
   ↓
State machine:  ALERT → MILD → DROWSY
   ↓
┌──────────────┴──────────────┐
│                             │
Laptop alert            ESP32 (USB serial)
(visual → beep → voice)       ↓
                          Buzzer

  +  Streamlit dashboard      +  Timestamped event logging
```

Frames enter the pipeline through the `FrameSource` interface in
[`src/capture.py`](src/capture.py), so changing the physical camera never changes
the detection code.

## 3. Repository layout

```
AI-Drowsiness-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── capture.py          camera abstraction + live preview   [Stage 1]
│   ├── landmarks.py        MediaPipe Face Landmarker           [Stage 2]
│   ├── features.py         EAR / MAR, frame validity gate      [Stages 3-4]
│   ├── headpose.py         head pose: yaw / pitch / roll       [Stage 4]
│   ├── eye_cnn.py          eye crop preprocessing [Stage 5], CNN [Stages 7-8]
│   ├── temporal.py         PERCLOS / blink / yawn / nod + FSM  [Stage 9]
│   ├── alert.py            escalating laptop alerts            [Stage 10]
│   ├── hardware.py         ESP32 serial link                   [Stage 11]
│   └── app.py              Streamlit dashboard                 [Stage 12]
├── training/               MRL dataset prep + CNN training     [Stages 6-7]
├── models/
│   ├── face_landmarker.task  MediaPipe face model (official)   [Stage 2]
│   └── eye_cnn.pt            trained eye-state CNN             [Stage 7]
├── hardware/               ESP32 sketch, wiring, photos        [Stage 11]
├── evaluation/
│   ├── camera_baseline.py  camera acceptance test + baseline   [Stage 1]
│   ├── results/            committed baseline measurements
│   └── evaluate.py         full system evaluation              [Stage 15]
├── data/                   local only, not committed
└── docs/                   report, poster, demo material
```

Files appear as their stage is built; the layout above is the final target.

## 4. Hardware

| Component | Role | Used from |
|---|---|---|
| Laptop webcam | Development camera for all software stages | Stage 1 |
| ESP32-WROOM-32 | Physical alarm controller (no AI runs on it) | Stage 11 |
| Buzzer + breadboard + jumper wires | Physical drowsiness alarm | Stage 11 |
| USB webcam (IR-cut filter removed later) | Final NoIR camera | Stage 13 |
| 850 nm 3-LED IR illuminator | Invisible illumination in darkness | Stage 13 |
| Adjustable mount | Stable driver-facing camera/IR position | Stage 13 |

The webcam is **not** modified until every software stage works on the unmodified
camera. A diffuser is a conditional purchase — only if Stage 14 testing shows
uneven IR illumination or hotspots.

## 5. Setup (Windows, Python 3.10 or 3.11)

```bat
git clone <your-repo-url>
cd AI-Drowsiness-Detection
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements use **`opencv-contrib-python`** (MediaPipe's dependency), not
`opencv-python`. Never install both: they unpack two different builds into the
same `cv2/` folder and the result is corrupt. If `opencv-python` is already
present, `pip uninstall -y opencv-python` first.

## 6. Running the current stage

### Stage 5 — eye crops, preprocessed exactly as the CNN will see them

```bat
python -m src.features                                :: live: everything so far + both eye crops in a panel
python -m src.features --dump-crops data\eye_crops\s1 :: also save valid crops from VALID frames every 30 frames
python -m src.features --crop-scale 1.7 --eye-size 48 :: try other crop geometry / output size
python -m src.eye_cnn --image path\to\eye.png         :: run the shared preprocessing on one file (MRL sample, saved crop)
python -m src.eye_cnn --self-test                     :: geometry, alignment, contract and I/O checks, no camera
```

Keys added: **`c`** hide / show the crop panel · **`e`** save both crops now
(64 × 64 gray to `data/eye_crops/`, un-resized raw to `data/eye_crops/raw/`).

**What Stage 5 produces** ([`src/eye_cnn.py`](src/eye_cnn.py)):

| Step | Detail |
|---|---|
| Locate | Eye centre = mean of the six EAR landmarks (33/160/158/133/153/144 right, 362/385/387/263/373/380 left); eye width = corner distance |
| Crop | Square, side = **1.5 ×** eye width, rotated so the eye corners are horizontal (`align_roll`), edge-replicated if it leaves the frame — one `warpAffine` does rotation, crop and padding |
| Grayscale | `cv2.COLOR_BGR2GRAY`; already-gray input (MRL) passes through |
| Resize | **64 × 64**, `INTER_AREA` when shrinking, `INTER_LINEAR` when enlarging |
| Normalise | float32 in [0, 1], then per-image standardisation to mean 0 / std 1 (flat image → zeros). Optional CLAHE (`equalize`, off) kept for the Stage 7 ablation |
| Validity | geometric only: eye width < 15 px or crop partly outside the frame → `valid=False` with a reason; the image is still produced for display. Stage 8 will additionally require Stage 4's frame to be VALID |

**The one rule that prevents train/inference mismatch:**
`preprocess_eye_image(image, config)` is the *only* function allowed to prepare
CNN input. Live crops go through it inside `crop_eye`; the MRL training
pipeline (Stage 6) and inference (Stage 8) must call the same function with
the same `EyePreprocessConfig`. The self-test asserts this contract bit for
bit: the shared function applied to a live raw crop reproduces the live
tensor exactly, and a crop saved to PNG re-preprocesses to the identical
tensor.

Two decisions that bind later stages, both deliberate:

- **No left / right flipping.** MRL does not label which eye an image shows, so
  the classifier must be side-agnostic; horizontal flips become a training
  augmentation instead.
- **`crop_scale = 1.5` and `size = 64` are initial values.** MRL images are
  tight, roughly square eye crops. Stage 6 puts real MRL samples next to crops
  saved by this stage (`e` key or `--dump-crops`) and adjusts the scale until
  they look alike — *before* any training. Per-image standardisation, rather
  than a dataset mean/std, is what makes MRL's infrared sensors and our
  webcam / NoIR camera comparable in brightness and contrast.

**Measured so far (Stage 5):**

| Run | Result |
|---|---|
| Self-test, synthetic eye tilted 20° | crop centred within 2 px; tilt after alignment **0.0°** (19.9° without); contract holds bit for bit; saved PNG re-preprocesses identically |
| True-aspect test portrait video, 300 frames | 600 crops, **100 % valid**; eye width 32–36 px → raw crop 51 px, *enlarged* to 64 × 64 |
| Live test with the developer's face | **pending** — at laptop distance the eyes measured ~50–60 px wide in Stage 3, so raw crops of ~75–90 px will be *shrunk* to 64 × 64, close to MRL's native resolution |

The portrait's eyes are small enough that its crops are upsampled — soft but
usable. Below the 15 px eye-width gate a crop is flagged invalid rather than
fed onward.

### Stage 4 — head pose and frame validity

```bat
python -m src.features                        :: live: EAR / MAR + yaw / pitch / roll + VALID / INVALID + invalid rate
python -m src.features --max-yaw 25           :: change the yaw limit (default 30 deg - initial value, untuned)
python -m src.features --pose-method pnp      :: cross-check estimator ('p' key cycles live)
python -m src.features --record data\s4.csv   :: per-frame CSV now also has yaw, pitch, roll, valid, reasons
python -m src.headpose                        :: head-pose self-test: rotation maths, PnP round trip
python -m src.features --self-test            :: formulas + validity gate + invalid-frame tracker
```

The Stage 3 demo grew into this one (same command). New on the HUD: the yaw /
pitch / roll line with an arrow from the nose in the facing direction, a green
**VALID FRAME** or red **INVALID: …reasons…** banner, and the invalid-frame rate
over a rolling 60 s window and for the session. On an INVALID frame the EAR and
MAR values are still shown but greyed "(not interpreted)" and are excluded from
the traces — they must never be read as eye or mouth state.

Keys: **`q`** quit · **`m`** mesh mode · **`g`** grayscale input · **`p`** pose
method · **`z`** hide / show the driver-zone ellipse (start hidden with
`--no-zone`) · **`s`** snapshot.

**Head pose** ([`src/headpose.py`](src/headpose.py)). Convention: **yaw > 0 =
turned to the subject's left, pitch > 0 = looking up, roll > 0 = tilt toward the
left shoulder.** Two independent estimators are implemented so one can check the
other:

| Method | How | Role |
|---|---|---|
| `matrix` | Angles from the 4 × 4 transformation MediaPipe fits to its canonical face model (all 468 points) | **default** |
| `pnp` | `cv2.solvePnP` on six landmarks (nose tip 1, chin 152, eye corners 33 / 263, mouth corners 61 / 291) against a generic 3-D face model, focal length assumed = frame width | cross-check |

**Why `matrix` is the default — measured, not assumed.** Both estimators were
put through the same tests (full numbers in
[`evaluation/results/stage4_pose_validation.csv`](evaluation/results/stage4_pose_validation.csv)):

| Test | `pnp` | `matrix` |
|---|---|---|
| Mirror the image → yaw and roll flip sign, pitch unchanged | ✅ | ✅ |
| Rotate the image ±15° in-plane → roll changes by exactly ∓15° | −15.0 / +15.5 | −15.0 / +15.5 |
| 7 real head-turn frames (developer's Stage 3 snapshots, direction labelled) → yaw sign correct | 7 / 7 | 7 / 7 |
| Strong turns | 45–61° | 40–57° |
| Moderate turn smaller than strong ones | ✅ | ✅ |
| **5 frontal frames → ǀyawǀ < 20°** | ❌ +10 … **+39°** | ✅ +3 … +12° |
| Agreement across the 12 frames | r = 0.96, mean ǀΔǀ 9.9°, `matrix` ≈ 0.83 × `pnp` | |

PnP's yaw read **+34° and +39° on the two frontal frames with the mouth wide
open** — opening the mouth moves the chin and mouth-corner landmarks, which the
generic closed-mouth model interprets as a head turn. That would flip a 30°
validity gate during every yawn, exactly when Stage 9 needs valid frames most.
`matrix` read +7° and +10° on the same frames. Two more findings: planar
perspective warps of a photo are **not** a valid proxy for head rotation
(MediaPipe's learned 3-D prior largely ignores them, so that part of the test
battery was discarded — only real turns and exact transforms count); and the
developer's frontal frames read **pitch ≈ −11°** with both methods, because a
laptop camera sits above the line of sight — Stage 9's nod detection must work
on pitch *changes* relative to a baseline, never on absolute pitch.

**Frame validity** ([`src/features.py`](src/features.py), `assess_frame`). A
frame is **INVALID** when any of these hold — and the HUD lists every reason:

| Check | Reason key | Initial threshold (configurable, **untuned**) |
|---|---|---|
| No face detected | `no_face` | — |
| Faces present but none inside the driver zone | `no_driver_in_zone` | zone radius 0.40 (Stage 2) |
| Face too small to measure reliably | `face_too_small` | width < 80 px |
| Face partly outside the frame | `face_at_edge` | any landmark within 4 px of the border |
| Degenerate landmark geometry (NaN EAR / MAR) | `degenerate_landmarks` | — |
| Eyes too narrow for EAR to be precise | `eye_too_small` | narrowest eye < 15 px |
| Head pose could not be estimated | `pose_unavailable` | — |
| Head turned too far | `yaw` | ǀyawǀ > 30° |
| Head tilted too far up / down (optional) | `pitch` | **off** by default — nods vs looking down is Stage 9's call |

`InvalidFrameTracker` keeps the rate over a rolling window (60 s, the future
temporal window) and for the session, plus a histogram of reason keys printed
at exit — that histogram is how the thresholds will be tuned from real
sessions.

**Measured so far (Stage 4):**

| Run | Result |
|---|---|
| Live webcam, nobody in view, 250 frames | 250 / 250 INVALID, reason `no_face`; 3.5 ms inference — the no-face path verified on real hardware |
| Synthetic two-face video (test portrait scaled to a ~75 px face) | 100 % INVALID: `face_too_small` (75 px < 80) and `eye_too_small` (14 px < 15) — the gate correctly refuses a face at roughly 2 m equivalent |
| Live test with a real face — yaw distribution while driving-like head movement, resulting invalid rate | **pending** (Stage 4 live test) |

### Stage 3 — Eye Aspect Ratio and Mouth Aspect Ratio

```bat
python -m src.features                               :: live camera: landmarks + EAR / MAR + FPS
python -m src.features --record data\features.csv    :: also log every frame's values to CSV
python -m src.features --no-window --max-frames 300  :: headless: statistics only
python -m src.features --self-test                   :: formula checks on known geometry, no camera
```

Same keys as Stage 2. The HUD shows EAR for the left eye, right eye and their
mean, MAR, FPS and inference time, plus the exact landmarks and distances the
formulas use (yellow for eyes, magenta for mouth) and two rolling traces.
`--record` writes one row per frame — this is how the EAR/MAR distributions for
open eyes, closed eyes and yawns will be *measured* before Stage 9 sets any
threshold.

**Landmarks and formulas** ([`src/features.py`](src/features.py)). All distances
are in pixels on the un-mirrored frame; L/R are the subject's own left/right.

| Feature | Landmarks (MediaPipe indices) | Formula |
|---|---|---|
| EAR, right eye | p1…p6 = 33, 160, 158, 133, 153, 144 | EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖) |
| EAR, left eye | p1…p6 = 362, 385, 387, 263, 373, 380 | same |
| MAR | corners 61, 291 · upper inner lip 82, 13, 312 · lower inner lip 87, 14, 317 | MAR = mean(‖82−87‖, ‖13−14‖, ‖312−317‖) / ‖61−291‖ |

EAR (Soukupová & Čech, 2016) averages two vertical eyelid gaps and divides by
the eye width, so it is invariant to distance and in-plane roll and falls
towards 0 as the eye closes. MAR is ≈ 0 with the lips touching and rises as
the mouth opens; three lip pairs are averaged to damp landmark jitter. Both
return NaN rather than dividing by zero if a width degenerates.

**Measured so far** — raw values, no thresholds applied:

| Condition | EAR (mean of both eyes) | MAR | Notes |
|---|---|---|---|
| Static test portrait, **vertically squashed** to 640 × 480 (aspect factor 0.53 — see note below); 883 frames over two runs | median 0.182, std 0.003 — absolute value is an artefact of the squash | median 0.005 | Noise floor: frame-to-frame ǀΔEARǀ median 0.0007; the two eyes agree to ǀL−Rǀ = 0.005 |
| Same portrait at its **true aspect ratio** (driver face in the two-face test video); 244 frames | median **0.323**, std 0.008, range 0.317–0.343 | median 0.001 | Eyes open, smiling, mouth closed |
| Live webcam, developer's face, 116 frames with a face (face in view 46 % of the run) | median **0.272**, range 0.176–0.416 | median **0.016**, max 0.044 (mouth closed) | One 2-frame dip to 0.176, consistent with a blink |
| Live test, frontal, **eyes open** | L 0.400 · R 0.357 · mean **0.379** | 0.005 | Blinks visible as sharp dips in the trace |
| Live test, frontal, eyes half-closed / looking down | L 0.293 · R 0.258 · mean 0.275 | 0.004 | |
| Live test, frontal, **eyes closed** | L 0.191 · R 0.131 · mean **0.161** | 0.003 | 2.4× below the open value |
| Live test, frontal, **mouth wide open** (two frames) | mean 0.407 / 0.447 | **0.869 / 1.001** | ~200× the closed-mouth value |
| Live test, head turned hard left or right (seven frames) | far eye 0.405 – **1.265**, near eye **0.113** – 0.373; mean 0.26 – 0.81 | 0.007 – 0.162 | See yaw note below |

The live-test rows are single-frame HUD readings from the developer's Stage 3
test snapshots (2026-09-13); the full list is in
[`evaluation/results/stage3_live_observations.csv`](evaluation/results/stage3_live_observations.csv).

**EAR is only meaningful near-frontal.** The formula is symmetric — on the
frontal portrait the two eyes agree to 0.005 — but head yaw foreshortens the
eye that is turning away, so the *far* eye's EAR inflates (to a meaningless
1.265 at extreme yaw) while the *near* eye can read as low as 0.113, a false
"eye closed". Both error directions appeared in the live test. A large per-eye
gap is therefore a head-pose signal, not noise: Stage 4 measures yaw directly
and marks such frames INVALID, and Stage 9 must never treat them as eye closure.

**Open-eye EAR differs between people — and image aspect ratio changes it.**
The first portrait test video was resized to 640 × 480 without preserving
aspect ratio, compressing it vertically by 0.53×; its EAR of 0.18 is that
artefact, since the same face at true aspect measures 0.32 (0.32 × 0.53 ≈ 0.18).
A useful warning in its own right: EAR values are only comparable between
frames of the same aspect ratio, so the camera resolution must never change
mid-session. At true aspect the portrait (0.32) and the developer (0.38) still
differ, so a single fixed threshold from the literature is unlikely to transfer
between drivers — an input to Stage 9's design (per-session calibration is one
option), not something applied here.

### Stage 2 — live facial landmarks

```bat
python -m src.landmarks                :: live camera, face contours + irises drawn
python -m src.landmarks --mode mesh    :: full 468-triangle tesselation
python -m src.landmarks --gray         :: feed grayscale frames (rehearsal for the IR camera)
python -m src.landmarks --no-window --max-frames 300   :: headless: detection rate + timing only
python -m src.landmarks --anchor 0.5 0.5 --zone 0.4    :: where the driver's face is expected (see below)
python -m src.landmarks --self-test    :: model integrity + code checks, no camera needed
```

Keys: **`q`**/**`Esc`** quit · **`m`** cycle draw mode (contours → mesh → points → off) ·
**`g`** toggle grayscale input · **`z`** hide / show the driver-zone ellipse ·
**`s`** snapshot. The HUD shows FPS, MediaPipe
inference time, FACE FOUND / NO FACE DETECTED, and the running detection rate.

**What Stage 2 produces for later stages.** `FaceLandmarkDetector.process(frame)`
returns a `FaceLandmarks` object (or `None` when no face is present) holding a
`(478, 3)` array in MediaPipe's canonical face-mesh topology, plus `pixels`,
`points(indices)` and `bounding_box()` helpers. Two rules downstream code relies on:

- **Left/right mean the *subject's* left and right.** That only holds if the
  model sees the un-mirrored camera frame, so detection always runs on the raw
  frame and the mirror-image view is applied to the *display* afterwards.
  Verified empirically: landmark 263 (subject's left eye) sits at larger *x*
  than landmark 33 (right eye) in the raw frame.
- **No face → `None`, never a stale or zero result.** Stage 4 will turn this
  into an explicit INVALID frame; nothing downstream should ever interpret a
  missing face as closed eyes.
- **Two people in view → the driver is chosen, the other is ignored.** Up to
  two faces are detected. Only faces inside a configurable *driver zone* around
  the expected driver position (`--anchor`, default frame centre; `--zone`
  radius 0.40 in normalised units) are candidates. Once chosen, the driver is
  kept while a face stays within 0.15 of their last position — a continuity
  lock, so tracking cannot flip between two people — otherwise the candidate
  nearest the anchor wins. Other faces are boxed **IGNORED** on screen and
  never measured; faces present but all outside the zone give "no driver".
  Verified on a synthetic two-face video built from MediaPipe's test portrait
  ([`evaluation/two_face_test.py`](evaluation/two_face_test.py)): the driver
  was kept in **100 % of 200 two-face frames**, including while the second face
  slid onto the anchor itself; a lone face inside the zone was selected in
  100 %; a lone face outside the zone was rejected in 100 %. Known limitation:
  if the driver's face is lost while a passenger sits inside the zone, the
  passenger is selected — Stage 4's head-pose gating and Stage 14's camera
  placement narrow that window. Also found and handled: MediaPipe occasionally
  returns *two* landmark sets for the **same** face right after a re-detection
  (8 of 150 single-face frames); duplicates are removed by box overlap
  (IoU ≥ 0.5) before selection. All radii are initial values, not tuned.

Measured on this laptop (640 × 480, CPU): MediaPipe inference **~4 ms** per frame
with no face in view, **7–8 ms** on a static test portrait, and **10–13 ms live
with a real, moving face** (live run, 2026-09-13). The live figure is higher most
likely because VIDEO mode re-runs the face detector whenever tracking confidence
dips, which a static image never triggers. 478 landmarks; grayscale input detected
the same face at the same rate as RGB. End-to-end throughput is set by the camera —
**20 FPS live** — not by MediaPipe: the same loop runs at 54 FPS with rendering and
96 FPS headless when reading from a video file. At 20 FPS each frame has a 50 ms
budget, of which landmarks use ~12 ms, leaving ~38 ms for everything later stages add.

### Stage 1 — camera only

```bat
python -m src.capture                 :: live camera preview
python -m src.capture --list          :: list working camera indices
python -m src.capture --device 1      :: use a different camera
python -m src.capture --self-test     :: headless check, no camera needed
```

In the preview window: **`q`** or **`Esc`** quits, **`s`** saves a snapshot to
`data/snapshots/`. Closing the window with the **X** button also exits cleanly.
(`Alt+F4` does *not* close it — OpenCV's HighGUI window class ignores it.)

### Measured baseline — unmodified laptop webcam

Recorded 2026-09-13 so that Stage 13 has a real before/after reference for the
NoIR conversion. Windows 11, Python 3.11.9, OpenCV 4.11.0.86, NumPy 1.26.4,
indoor artificial lighting. All figures are measured, not estimated. (Stage 2
moved the environment to NumPy 2.4.6 and `opencv-contrib-python` 4.11.0.86; the
acceptance test was re-run afterwards and passed 13/13 with identical behaviour.)

| Property | Measured value |
|---|---|
| Cameras detected | 1 — index 0, DSHOW backend |
| Resolution | 640 × 480 as requested; driver-reported 30 FPS |
| Camera open time | 1.60 s; 1.54–1.66 s on reopen, no handle leak over 3 cycles |
| Throughput, headless capture | **30.0 FPS**, 0 of 120 frames dropped |
| Throughput, with preview window | 19.0–25.4 FPS — the cost of `cv2.imshow` rendering |
| Frame read latency | median 31.0 ms, p95 47.6 ms, max 50.0 ms |
| Mean frame brightness | 129.7 / 255 |
| Channel means (B / G / R) | 124.0 / 130.0 / 131.5 — IR-cut filter in place |
| Focus, Laplacian variance | 86.8 |
| Saturated pixels (hotspot measure) | 0.00 % |
| Illumination uniformity (dimmest ÷ brightest 3×3 cell) | 0.406 |

Produced by [`evaluation/camera_baseline.py`](evaluation/camera_baseline.py); the
raw results are committed in [`evaluation/results/`](evaluation/results/).
Reproduce with:

```bat
python evaluation\camera_baseline.py --label unmodified-laptop-webcam
```

Two notes for anyone repeating this. **Let the camera settle** — a cold first
run measured 22.5 FPS / 48 ms median latency, versus 30.0 FPS / 31 ms once
auto-exposure had stabilised. And **repeatability is tight**: two consecutive
runs differed by 0.01 FPS and 0.05 brightness levels, so a real change between
hardware conditions will stand out clearly from measurement noise.

See [evaluation/README.md](evaluation/README.md) for the Stage 13 before/after
protocol and what each metric reveals about the NoIR/IR conversion.

## 7. Build stages

| # | Stage | Status |
|---|---|---|
| 1 | Project foundation + webcam capture | ✅ done |
| 2 | MediaPipe Face Landmarker — 478 real-time landmarks | ✅ done |
| 3 | EAR + MAR from landmarks, live display + CSV recording | ✅ done |
| 4 | Head pose (yaw / pitch / roll) + INVALID-frame gate + invalid-frame rate | ✅ done |
| 5 | Eye-region cropping + preprocessing shared with training | ✅ done |
| 6 | MRL eye dataset preparation (subject-independent split) | ⬜ |
| 7 | Eye-state CNN training and evaluation | ⬜ |
| 8 | CNN integrated into the live pipeline | ⬜ |
| 9 | Temporal analysis + ALERT/MILD/DROWSY state machine | ⬜ |
| 10 | Escalating laptop alert system | ⬜ |
| 11 | ESP32 + buzzer physical alarm | ⬜ |
| 12 | Streamlit dashboard + event logging | ⬜ |
| 13 | NoIR camera conversion + 850 nm IR illumination | ⬜ |
| 14 | Full hardware integration + day/dim/IR testing | ⬜ |
| 15 | Final evaluation, ablation study, documentation | ⬜ |

## 8. Evaluation plan (Stage 15)

Measured on a self-collected test set (~8–10 subjects) across alert/acted-drowsy
behaviour, bright/dim/IR-only lighting, and with/without glasses:
accuracy, precision, recall, F1, confusion matrix, FPS, processing latency,
eye-closure-to-alarm latency, false alarms per hour and invalid-frame rate —
broken down by lighting condition and by glasses/no-glasses, plus an ablation
study (EAR-only, CNN-only, EAR+CNN, with/without temporal smoothing, with IR).

## 9. Limitations

To be filled in from actual observations as stages complete — not in advance.
