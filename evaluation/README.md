# evaluation/

Measurement tooling and committed measurements. Currently holds the Stage 1
camera baseline and the Stage 3 EAR/MAR observations; the full system
evaluation (Stage 15) will be added here later.

## two_face_test.py

Checks that the landmark detector keeps measuring the **driver** when a second
person is in view. It builds a 400-frame synthetic video from one frontal
portrait pasted at two sizes — driver alone, second face appearing, second face
sliding onto the driver anchor, driver gone, second face outside the zone — and
asserts which face is returned in each phase and why (`single`, `lock`,
`nearest anchor`, or none). No camera needed:

```bat
python evaluation\two_face_test.py --portrait path\to\frontal_photo.png
```

Any clear frontal photo works; the project's own run used MediaPipe's test
image (`business-person.png` from the MediaPipe assets bucket), which is not
committed here. The generated video lands in `data/` (git-ignored) and can be
replayed in any demo with `--device data\two_faces.avi`.

Result on 2026-09-13: all checks passed — driver kept in 100 % of the 200
two-face frames, lone face inside the zone selected 100 %, lone face outside
the zone rejected 100 %.

## results/stage3_live_observations.csv

EAR and MAR values read from the HUD of the developer's Stage 3 test
snapshots (single frames, 2026-09-13). The `condition` column was labelled by
inspecting each frame. Head-turn rows show the far eye's EAR inflating and the
near eye's collapsing — the evidence behind the yaw gating in Stage 4. Frame
images are not committed.

## camera_baseline.py

A repeatable acceptance test for the camera capture path. It measures the
camera only — no face detection, no eye state, no drowsiness logic.

Two purposes:

1. **Acceptance test.** Proves the camera works: opens, delivers correctly
   shaped frames, the image is live rather than black or frozen, frames are not
   dropped, and the device can be released and reopened without leaking a
   handle. Exit code is `0` on pass, `1` if any check fails.
2. **Baseline record.** Writes every measurement to a JSON file in
   `results/`, so the identical run can be repeated after a hardware change and
   the two compared numerically.

### Running it

```bat
venv\Scripts\activate
python evaluation\camera_baseline.py --label unmodified-laptop-webcam --notes "indoor lighting"
```

Useful options:

| Option | Purpose |
|---|---|
| `--label NAME` | Names the run and the output file. Use a descriptive condition name. |
| `--notes "..."` | Lighting, distance, glasses, IR on/off — anything needed to reproduce it. |
| `--device N` | Camera index. Use `python -m src.capture --list` to find it. |
| `--frames N` | Frames to measure (default 120). |
| `--save-frame` | Keep one sample image in `data/snapshots/` (git-ignored — it may contain a face). |
| `--compare A.json B.json` | Print a side-by-side diff of two runs. |

### What each metric means

| Metric | Why it is recorded |
|---|---|
| `measured_fps`, `latency_ms` | Real throughput of the capture path. The frame budget every later stage must fit inside. |
| `drop_rate` | Dropped reads. A USB camera on a long extension may start dropping frames. |
| `open_time_s`, `reopen_times_s` | Device open cost, and proof no handle is leaked across restarts. |
| `mean_brightness`, `std_brightness` | How much light reaches the sensor. The headline number for any low-light comparison. |
| `channel_means` (B/G/R) | **Key NoIR indicator.** With the IR-cut filter in place the channels differ normally. Once the filter is removed, IR leaks into all three channels and they converge. |
| `focus_laplacian_var` | Sharpness. Use it to refocus the lens after reassembling the camera — turn the lens until this number peaks on a detailed, well-lit target. |
| `saturated_fraction` | Fraction of pixels at/near 255. **This is the IR hotspot measure.** |
| `grid_brightness`, `grid_uniformity_ratio` | 3×3 brightness map and dimmest/brightest cell ratio. **This is the illumination-evenness measure** that decides whether a diffuser is needed. |

### Interpreting a comparison

These are things to *look for*, not results — record whatever actually happens.

- **After removing the IR-cut filter:** expect `channel_means` to move closer
  together, and daylight images to look washed out or pink-tinted.
- **After adding 850 nm illumination in darkness:** expect `mean_brightness` to
  rise substantially versus the same dark scene with IR off.
- **Hotspots:** a high `saturated_fraction` combined with a low
  `grid_uniformity_ratio` (a bright centre cell, dark corners) is the evidence
  that would justify buying a diffuser. A uniformity ratio near 1.0 means even
  illumination and no diffuser needed.
- **Focus:** if `focus_laplacian_var` drops sharply after reassembly, the lens
  was disturbed — refocus before doing anything else.

Compare like with like. A run in a dark room is not comparable to one in
daylight; that is why `--label` and `--notes` exist.

### Measurement discipline

- Let the camera run for a few seconds before recording — auto-exposure needs
  to settle. A cold first run measured 22.5 FPS / 48 ms median latency, while
  settled runs measured 30.0 FPS / 31 ms on the same hardware.
- Repeatability under identical conditions is very tight (two runs differed by
  0.01 FPS and 0.05 brightness levels), so a meaningful change between
  conditions will stand out clearly from noise.
- Keep the result JSON files. They are small, they are committed, and they are
  the evidence behind the numbers in the report.

## Stage 13 protocol (NoIR + IR)

Run the same command at each step, changing only `--label` and `--notes`.
Keep the camera position, distance and framing identical.

| # | When | Suggested label |
|---|---|---|
| 1 | Before touching the hardware (**done** — see `results/`) | `unmodified-laptop-webcam` |
| 2 | USB webcam, unmodified, normal room light | `usb-webcam-unmodified` |
| 3 | Same camera after removing the IR-cut filter, daylight | `noir-daylight` |
| 4 | NoIR, dim room, IR **off** | `noir-dim-no-ir` |
| 5 | NoIR, dark room, IR **off** | `noir-dark-no-ir` |
| 6 | NoIR, dark room, IR **on** | `noir-dark-ir-on` |

Then compare the pairs that answer a question:

```bat
python evaluation\camera_baseline.py --compare evaluation\results\<step2>.json evaluation\results\<step3>.json
python evaluation\camera_baseline.py --compare evaluation\results\<step5>.json evaluation\results\<step6>.json
```

A bare filename also works — anything not found as given is looked up in
`evaluation/results/`.

Step 2 → 3 shows what removing the filter did. Step 5 → 6 shows what the IR
illuminator actually buys in darkness — the core low-light claim of the project.

## Checks a human still has to perform

The script is headless and cannot verify the interactive parts. Run
`python -m src.capture` and confirm by eye:

- [ ] The preview window opens and shows a live image
- [ ] Motion is smooth with no lag building up
- [ ] `q` exits cleanly
- [ ] `Esc` exits cleanly
- [ ] The window **X** button exits cleanly
- [ ] `s` saves a snapshot to `data/snapshots/`
- [ ] Image quality is acceptable by eye (focus, framing, exposure)

`Alt+F4` does **not** close the window — OpenCV's HighGUI window class ignores
it. That is expected, not a defect.
