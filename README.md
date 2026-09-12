# AI Driver Drowsiness Detection System

A real-time driver drowsiness detection system that works in normal lighting **and**
in low-light / dark conditions, using a modified NoIR webcam with 850 nm infrared
illumination.

Final-year B.E. project. Built and committed stage by stage — the commit history is
the development log.

> **Status: Stages 1–6 complete; Stage 7 training in progress; Stage 8 and 9
> code complete and verified, awaiting the trained model for their live tests.**
> Camera → landmarks → EAR / MAR → head pose → frame validity → eye crops →
> (CNN) → 60 s temporal analysis → ALERT / MILD / DROWSY all run end to end.
> The first real CNN training run is under way and its measured results will
> be added here when it finishes. **No eye-state accuracy is claimed until
> then, and every temporal threshold is an untuned initial value.**
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
│   ├── temporal.py         60 s window: PERCLOS / blinks / yawns / nods, FSM + hysteresis, replay [Stage 9]
│   ├── alert.py            escalating laptop alerts            [Stage 10]
│   ├── hardware.py         ESP32 serial link                   [Stage 11]
│   └── app.py              Streamlit dashboard                 [Stage 12]
├── training/
│   ├── inspect_mrl.py      report the dataset's real structure  [Stage 6]
│   ├── prepare_mrl.py      subject-independent split -> .npz   [Stage 6]
│   ├── splits/             committed subject lists + split_stats.json
│   ├── train_eye_cnn.py    training + evaluation logic          [Stage 7]
│   └── train_eye_cnn.ipynb thin Colab driver around the script [Stage 7]
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

### Stage 9 — temporal drowsiness analysis and the ALERT / MILD / DROWSY state machine

```bat
python -m src.features                                    :: live: everything so far + temporal panel (top right)
python -m src.features --record data\session.csv          :: record a session (acted drowsiness) for replay / tuning
python -m src.temporal --replay data\session.csv --plot data\session.png   :: offline replay: metrics, transitions, plot
python -m src.features --fusion ear                       :: eye-closure source: cnn | ear | fused (default)
python -m src.features --perclos-mild 0.12 --perclos-drowsy 0.25 --microsleep 1.0 --yawn-mar 0.5
python -m src.temporal --self-test                        :: 11 synthetic scenarios, no camera
```

**What it computes** ([`src/temporal.py`](src/temporal.py)) — all over a
**60 s sliding window**, using only frames Stage 4 marked VALID:

| Quantity | Definition |
|---|---|
| Per-frame eye closure | fused probability = mean of the CNN's P(CLOSED) over the available eyes and a soft EAR indicator, sigmoid((0.20 − EAR) / 0.03); closed when ≥ 0.5. `--fusion cnn` / `ear` isolate one source (Stage 15 ablation) |
| **PERCLOS** | closed valid frames ÷ valid frames with an eye reading |
| Closure events | a run of closed frames: < 80 ms is noise, 80–500 ms a **blink**, longer a **prolonged closure**, ≥ 1.5 s a **microsleep**. An invalid gap ≤ 0.3 s inside a run pauses it instead of splitting it |
| **Mean blink duration**, blink rate | over blinks completed inside the window |
| **Yawn rate** | MAR ≥ 0.60 sustained ≥ 1.0 s, counted per minute of window |
| **Head nod count** | pitch dropping ≥ 15° below the window's median pitch for 0.3–3 s and recovering; longer drops are "looking down", not nods |
| **Invalid-frame rate** | invalid ÷ all frames in the window — the only statistic invalid frames contribute to |

**A single closed frame never means drowsiness:** it moves PERCLOS by one frame
and can at most start a closure event; only window statistics and event
durations reach the state machine. Before 60 valid frames (~3 s), or when more
than 60 % of the window is invalid, the metrics are flagged *insufficient* and
the state holds.

**State machine with hysteresis.** Enter and exit thresholds differ (Schmitt
trigger), escalation needs its condition to hold for **2 s**, de-escalation
needs the exit condition for **10 s** plus a minimum stay of 5 s in MILD /
10 s in DROWSY. The one exception is a microsleep — a closure ≥ 1.5 s, i.e.
~30 consecutive frames — which enters DROWSY immediately.

| Transition | Condition |
|---|---|
| → **MILD** | PERCLOS ≥ 0.15, **or** ≥ 2 yawns, **or** ≥ 2 nods, **or** ≥ 2 closures ≥ 0.5 s in the window |
| → **DROWSY** | PERCLOS ≥ 0.30, **or** PERCLOS ≥ 0.15 with ≥ 3 yawns + nods, **or** a microsleep |
| DROWSY → MILD / ALERT | PERCLOS < 0.22 and no microsleep in the window, held 10 s, after ≥ 10 s in DROWSY |
| MILD → ALERT | PERCLOS < 0.10 and no supporting events, held 10 s, after ≥ 5 s in MILD |

**Threshold provenance — read this before quoting any number.**

| Value | Setting | Status |
|---|---|---|
| EAR "closed" centre 0.20 (soft width 0.03) | `ear_closed_thr` | **INITIAL, bracketed by MEASURED data**: developer open 0.34–0.40, closed 0.05–0.19 (Stages 3/5). Per-person — needs calibration |
| MAR yawn 0.60 for ≥ 1.0 s | `mar_yawn_thr`, `min_yawn_s` | **INITIAL, bracketed by MEASURED data**: closed ≤ 0.05, wide open 0.87–1.00. Talking untested → **TUNE** |
| Blink 80–500 ms | `min_blink_s`, `max_blink_s` | INITIAL (literature 100–400 ms; one frame at 20 FPS = 50 ms) |
| Microsleep 1.5 s | `microsleep_s` | INITIAL, conservative (literature 0.5–1 s +) → **TUNE** |
| PERCLOS 0.15 / 0.10 (MILD enter / exit), 0.30 / 0.22 (DROWSY) | `perclos_*` | INITIAL from the driver-monitoring literature (PERCLOS levels of Wierwille et al. / NHTSA) → **TUNE on own recordings** |
| 2 yawns, 2 nods or 2 long closures → MILD; 3 yawns + nods lift PERCLOS-mild to DROWSY | `yawns_mild`, `nods_mild`, `long_closures_mild`, `support_for_drowsy` | INITIAL → **TUNE** |
| Nod = 15° drop, 0.3–3 s | `nod_drop_deg`, `min/max_nod_s` | INITIAL → **TUNE** (uses pitch *changes* — Stage 4 measured a −11° rest offset on the laptop) |
| Dwell 2 s up / 10 s down; holds 5 s / 10 s | `*_dwell_s`, `*_min_hold_s` | INITIAL engineering values → **TUNE** |
| Window 60 s; sufficiency 60 valid frames, ≤ 60 % invalid | `window_s`, `min_valid_frames`, `max_invalid_rate` | roadmap requirement; INITIAL |

**Verified so far:** the 11-scenario self-test (20 FPS synthetic streams) —
eyes open → ALERT; one closed frame → no event, PERCLOS 0.3 %; 150 ms blinks
every 4 s → 15 blinks/min, mean 157 ms, PERCLOS 3.9 %, ALERT; a 2 s closure →
still ALERT at 1 s, DROWSY at 1.5 s, held; 40 % closure → DROWSY after the
minimum 5 s (3 s sufficiency + 2 s dwell), then graded recovery DROWSY → MILD
(while the long closures age out of the window) → ALERT with no flapping;
PERCLOS hovering 12–16 % → at most 2 transitions in 120 s (no
flapping); 3 yawns → MILD; 3 nods against a −10° baseline; 30 % invalid frames
with closed eyes → PERCLOS stays 0; 70 % invalid → flagged insufficient; fusion
fallbacks; CSV replay of a 3 s closure → DROWSY. The integrated live loop and
the replay tool ran on the test-portrait video (eyes open: ALERT, PERCLOS 0 %).
**Not yet measured:** behaviour on a real acted-drowsiness recording — that is
the Stage 9 live test, and its recording is what the thresholds get tuned on.

### Stage 6 — MRL Eye Dataset: what it really is, and how it was split

```bat
python training\inspect_mrl.py data\mrl\mrlEyes_2018_01.zip                        :: report the real structure
python training\prepare_mrl.py --source data\mrl\mrlEyes_2018_01.zip --out data\mrl_prepared --preview
```

**Verified facts about the archive** (from the inspector and the archive's own
`annotation.txt`, not from memory): `mrlEyes_2018_01.zip`, 341.9 MB, SHA-256
`17ff8992…fdfc`; one folder per subject `s0001…s0037`; **84,898** 8-bit grayscale
PNGs, square, 56–278 px (median 87); every filename matches
`sXXXX_YYYYY_G_GL_ST_RF_LT_SN` (gender, glasses, eye state 0 = closed / 1 = open,
reflections none/low/high, lighting bad/good, sensor 01 RealSense SR300 · 02 IDS ·
03 Aptina). Overall: **41,946 closed / 42,952 open**, glasses in 28.3 % of
images, 5 of 37 subjects female, lighting labelled "good" in only 36.8 %,
sensor 02 in six subjects and sensor 03 in two (s0012, s0014).

**Why the split needed a search, not a shuffle.** Subjects are wildly unequal:
382 to 10,257 images each, and several are effectively single-class
(s0004: 1,069 closed / 0 open; s0006: 1,011 / 1; s0008: 832 / 0; s0028: 13 / 723;
s0035: 21 / 621). A random subject shuffle can give a test set that is mostly
one class or mostly one person. `prepare_mrl.py` therefore evaluates
**20,000 seeded random partitions** (whole subjects only, greedy fill towards
70 / 15 / 15 of images) and keeps the one with the lowest cost: deviation from
the image fractions, deviation of the validation and test closed ratio and
glasses ratio from the global values, and penalties for a validation or test
split without a female subject, without sensor-02 images, or with fewer than
2,000 images of either class. Seed 0, best trial 14,152, cost 0.2217 (fraction
0.020 · closed ratio 0.019 · glasses ratio 0.184 · all penalties 0).

| Split | Subjects | Images | Closed | Glasses | Female subj. | Sensor 02 / 03 imgs |
|---|---|---|---|---|---|---|
| train | 25 | 59,012 (69.5 %) | 49.3 % | 27.4 % | 3 | 2,995 / 1,121 |
| val | 4 — s0011 s0012 s0027 s0031 | 12,779 (15.1 %) | 49.6 % | 25.6 % | 1 | 2,835 / 1,643 |
| test | 8 — s0004 s0006 s0007 s0008 s0009 s0015 s0016 s0032 | 13,107 (15.4 %) | 49.8 % | 34.8 % | 1 | 6,162 / 0 |

Guarantees: no subject appears in two splits — asserted in `prepare_mrl.py`
before anything is written, again when the `.npz` packs are re-opened, and a
third time at the start of every training run. The subject lists and every
statistic above are committed in [`training/splits/`](training/splits/) so the
split is reproducible and auditable. Known limitations: the test set contains
no Aptina (sensor 03) images — only two subjects have any and both cannot be
held out — and its glasses share (34.8 %) is above the global 28.3 %; the
validation set has only four subjects.

**Preprocessing** used exactly `src.eye_cnn.preprocess_eye_image` (grayscale →
64 × 64, `INTER_AREA` since almost every image shrinks) and packed uint8 images
with all fields: `train.npz` 125.9 MB, `val.npz` 28.8 MB, `test.npz` 35.5 MB
(git-ignored under `data/mrl_prepared/`; copy to Google Drive for Colab).
Mean intensity differs between splits (81.5 / 91.8 / 115.6) — a concrete reason
per-image standardisation is applied at load time rather than a dataset mean.

**Crop-scale check.** Twelve MRL samples (open / closed, with / without glasses)
were laid beside the eight live crops saved in Stage 5: with `crop_scale = 1.5`
the live crops frame the eye like MRL does (eye spanning roughly 65–70 % of the
tile, skin margin above and below), so the value stays. The visible domain gap
is elsewhere — the webcam crops are softer and lower-contrast than MRL's
infrared frames — which standardisation plus the blur / gamma augmentation
address, and which the NoIR camera in Stage 13 should narrow.

### Stage 7 — eye-state CNN (code ready; first real run training)

```bat
python -m src.eye_cnn --self-test                       :: preprocessing, augmentation, model, checkpoint round trip
python training\train_eye_cnn.py --data <npz dir> --out runs\run01 --export models\eye_cnn.pt --epochs 30
python -m src.eye_cnn --predict path\to\eye.png         :: classify one image with models/eye_cnn.pt
```

Colab: open [`training/train_eye_cnn.ipynb`](training/train_eye_cnn.ipynb) — a
thin driver that mounts Drive, clones this repo and runs the same script on a
GPU, writing checkpoints, curves and metrics to Drive after every epoch.

| Piece | Where | What |
|---|---|---|
| Model | `src/eye_cnn.py` `build_model` | four double-conv blocks (w, 2w, 4w, 4w) + BN/ReLU, global average pooling, dropout, 2 logits. Width 32 → **582,562 parameters** |
| Classes | `CLASSES = ("CLOSED", "OPEN")` | index 0 = closed, 1 = open — the same coding as MRL's eye-state field |
| Augmentation | `augment_eye`, `AugmentConfig` | applied to the **uint8 image before standardisation**: horizontal flip (dataset does not label eye side), ±10° rotation, 0.85–1.15 scale, ±4 px shift, contrast / brightness, gamma 0.7–2.2, **low-light branch** (×0.25–0.6 intensity, then sensor noise σ ≤ 14 — before standardisation so the signal-to-noise ratio is realistically low), blur, specular spot (glasses reflection), small dark cutout (frame edge) |
| Checkpoint | `save_model` / `load_model` | self-describing file: weights + architecture + the exact `EyePreprocessConfig` + class names + metadata; Stage 8 refuses a mismatched one |
| Inference API | `EyeStateClassifier` | `predict(tensors)`, `predict_crop(EyeCrop)`, `predict_image(file)` — all through `preprocess_eye_image` |
| Training | `training/train_eye_cnn.py` | re-asserts subject disjointness, class-weighted cross-entropy, AdamW + one-cycle LR, early stopping on validation accuracy, `last.pt`/`best.pt` every epoch, `history.json`, `curves.png`; final evaluation on the **test split of unseen subjects**: accuracy, precision, recall, F1 (CLOSED as the safety-relevant positive class, plus OPEN and macro), confusion matrix, breakdown by glasses; `metrics.json` / `metrics.md` / `confusion_matrix.png`; `--export` writes the final model |

**Measured on this laptop (CPU, 12 threads):** one forward pass on both eyes
takes **3.8 ms** at width 32 (2.5 ms at width 16) — well inside the ~38 ms per
frame left after landmarks, so the default width stays 32.

**Pipeline smoke test — not a result.** To prove the code path, the script
was run for 3 epochs on a *synthetic* stand-in for the splits (3,000 / 600 /
600 images of a bright ring vs a dark arc, disjoint fake subjects). It reached
100 % because the task is trivially separable; the point is that data
loading, augmentation, checkpointing, curves, metrics, export and reload all
worked, and `EyeStateClassifier` reproduced the training labels on the
exported file. **Real eye-state figures will appear here only after the Colab
run on the MRL split.** The roadmap's > 95 % target will not be engineered
towards; whatever the unseen-subject test set gives is what gets reported.

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
| Live test, developer's face, 8 snapshots ([`evaluation/results/stage5_live_observations.csv`](evaluation/results/stage5_live_observations.csv)) | Eye width **33–41 px** at the normal laptop distance (~33–37 cm by the matrix estimate) → raw crops 50–60 px, almost 1 : 1 with the 64 × 64 output; open-eye tiles sharp, centred and level under head tilt; closed-eye tiles unmistakably closed. At ~46 cm the eyes shrink to **24–25 px** and the tiles turn soft (2.7× upsampling). Every crop geometrically valid |

**Eye size drives crop quality, and it is set by camera distance and
resolution.** At 640 × 480 the eyes were 38 px at ~35 cm and 24 px at ~46 cm;
a dashboard mount at 60–80 cm would leave ~15–20 px — at the validity floor
and far too soft for a classifier. Input to Stage 13/14: run the USB camera at
**1280 × 720** (doubling eye pixels) and/or mount it closer, and re-measure
with the same HUD. Below the 15 px eye-width gate a crop is flagged invalid
rather than fed onward.

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
| Live test, developer's face, 8 snapshots (same CSV as Stage 5) | Frontal yaw **+2…+9°**, pitch **−8…+6°** (camera slightly above the eye line), roll −2…−6°; turns read **−24°** and **+30°** and stayed VALID (the gate fires only *beyond* 30°); invalid-frame rate 8–23 % over the session, the high values from start-up and edge moments |

Two things the live frames added. **EAR is already badly distorted well
inside the 30° gate:** at yaw −24° the far eye read 0.471 and at +30° it read
0.555, against 0.34 frontal — a 40–60 % inflation while the frame still
counts as VALID. Stage 9 must either tighten the yaw gate (≈ 20–25°) or
compensate EAR for yaw; the measurement is recorded so the choice can be made
from data. **One roll reading is suspect:** in the −24° frame the face is
visibly tilted by roughly 20° yet the matrix reported roll −2°. Roll gates
nothing and the eye crops are aligned from the image eye-line angle (their
tiles were level), so nothing downstream is affected — but re-check `matrix`
roll under strong yaw before ever using it.

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
| 6 | MRL eye dataset: inspected, subject-independent split built and verified, packs written | ✅ done |
| 7 | Eye-state CNN training and evaluation | 🔶 code complete; first real run training — results pending |
| 8 | CNN integrated into the live pipeline | 🔶 integrated and verified with interim weights; live test waits for the final model |
| 9 | Temporal analysis + ALERT/MILD/DROWSY state machine | 🔶 implemented, 11 synthetic scenarios pass, replay tool; thresholds are initial values — live acted-drowsiness test pending |
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
