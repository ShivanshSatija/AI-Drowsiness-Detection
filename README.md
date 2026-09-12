# AI Driver Drowsiness Detection System

A real-time driver drowsiness detection system that works in normal lighting **and**
in low-light / dark conditions, using a modified NoIR webcam with 850 nm infrared
illumination.

Final-year B.E. project. Built and committed stage by stage — the commit history is
the development log.

> **Status: Stage 3 of 15 complete** (camera → landmarks → live EAR / MAR).
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
│   ├── features.py         EAR / MAR / head pose               [Stages 3-4]
│   ├── eye_cnn.py          eye-state CNN inference             [Stages 7-8]
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
| Static frontal test portrait — eyes open, smiling, mouth closed; 883 frames over two runs | median **0.182**, std 0.003, range 0.173–0.193 | median **0.005**, max 0.025 | Noise floor: frame-to-frame ǀΔEARǀ median 0.0007; the two eyes agree to ǀL−Rǀ = 0.005 |
| Live webcam, developer's face, 116 frames with a face (face in view 46 % of the run) | median **0.272**, range 0.176–0.416 | median **0.016**, max 0.044 (mouth closed) | One 2-frame dip to 0.176, consistent with a blink |

Closed-eye and open-mouth values are **not yet recorded** — they come from the
Stage 3 live test.

**EAR depends on head yaw.** On the frontal portrait the two eyes agree to
0.005, but in the live run the left eye read a persistent 0.08–0.15 *lower*
than the right in every 25-frame bin — the sitter was looking at the screen
rather than into the camera, which foreshortens one eye. The formula itself is
symmetric, so a per-eye gap of that size is a head-pose signal, not noise.
Stage 4 measures yaw directly and marks such frames INVALID; Stage 9 must not
treat them as eye closure.

### Stage 2 — live facial landmarks

```bat
python -m src.landmarks                :: live camera, face contours + irises drawn
python -m src.landmarks --mode mesh    :: full 468-triangle tesselation
python -m src.landmarks --gray         :: feed grayscale frames (rehearsal for the IR camera)
python -m src.landmarks --no-window --max-frames 300   :: headless: detection rate + timing only
python -m src.landmarks --self-test    :: model integrity + code checks, no camera needed
```

Keys: **`q`**/**`Esc`** quit · **`m`** cycle draw mode (contours → mesh → points → off) ·
**`g`** toggle grayscale input · **`s`** snapshot. The HUD shows FPS, MediaPipe
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
| 4 | Head pose + invalid-frame handling | ⬜ |
| 5 | Eye-region cropping and preprocessing | ⬜ |
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
