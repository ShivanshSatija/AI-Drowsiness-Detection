# AI Driver Drowsiness Detection System

A real-time driver drowsiness detection system that works in normal lighting **and**
in low-light / dark conditions, using a modified NoIR webcam with 850 nm infrared
illumination.

Final-year B.E. project. Built and committed stage by stage — the commit history is
the development log.

> **Status: Stage 1 of 15 complete** (camera capture + live preview).
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
│   ├── landmarks.py        MediaPipe Face Mesh                 [Stage 2]
│   ├── features.py         EAR / MAR / head pose               [Stages 3-4]
│   ├── eye_cnn.py          eye-state CNN inference             [Stages 7-8]
│   ├── temporal.py         PERCLOS / blink / yawn / nod + FSM  [Stage 9]
│   ├── alert.py            escalating laptop alerts            [Stage 10]
│   ├── hardware.py         ESP32 serial link                   [Stage 11]
│   └── app.py              Streamlit dashboard                 [Stage 12]
├── training/               MRL dataset prep + CNN training     [Stages 6-7]
├── models/                 trained eye_cnn.pt                  [Stage 7]
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

## 6. Running the current stage

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
indoor artificial lighting. All figures are measured, not estimated.

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
| 2 | MediaPipe Face Mesh landmarks | ⬜ |
| 3 | EAR + MAR | ⬜ |
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
