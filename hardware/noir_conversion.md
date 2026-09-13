# Stage 13 — NoIR conversion of the USB webcam and 850 nm IR illumination

The detector is finished in software (Stages 1–12) and runs on the unmodified
laptop webcam. Stage 13 replaces the camera with a USB webcam whose **IR-cut
filter has been removed** ("NoIR") and adds an **850 nm 3-LED illuminator**, so
the driver's face is visible to the camera in darkness without visible light.

**Working rule for this stage: one physical step at a time, then stop and
report.** Every step below ends with a STOP line. Do not continue past a STOP
until the results of that step have been looked at. The detection code is not
touched in this stage.

Hardware in play:

| Part | Role | Notes |
|---|---|---|
| USB webcam | the camera that gets modified | model unknown to this document — the checks in §2 decide whether it is suitable |
| MYADDICTION 850 nm 3-LED IR illuminator board | invisible illumination | small board, deliberately not a high-power CCTV array; **read its label** for voltage and current before powering it |
| adjustable mount | holds camera + illuminator at the driver | |
| power supply / cable for the illuminator | | must match the label; polarity matters |
| TV remote (any) | IR test source | its LED is ~940 nm or ~850 nm — invisible to the eye, visible to a NoIR camera |

Tooling used throughout (all in the repo):

```bat
python evaluation\stage13_camera_check.py identify                                   :: which cameras, which OpenCV index, which controls
python evaluation\stage13_camera_check.py measure --device N --label LABEL --notes "..."   :: capture + image + detector numbers -> JSON
python evaluation\stage13_camera_check.py live --device N                            :: focus / saturation / evenness / IR-spot readouts
python evaluation\stage13_camera_check.py compare LABEL_A LABEL_B                    :: before/after table
python evaluation\camera_baseline.py --device N --label LABEL                        :: Stage 1 capture-path acceptance test (still valid)
```

Results land in `evaluation/results/stage13/` and are committed — they are the
evidence for the report. Snapshots (`--save-frame`, `s` key) go to
`data/snapshots/`, which is git-ignored because they may show a face.

---

## 1. Before touching the hardware — software checks

Plug the USB webcam in (a rear or direct port, not a hub if possible). Then:

### 1.1 Confirm the webcam works normally (Windows)

Open the Windows **Camera** app, switch to the USB webcam, confirm a live,
focused, correctly exposed image. If Windows does not show it, nothing below
will either — fix the cable/port/driver first.

### 1.2 Confirm OpenCV detects it

```bat
python evaluation\stage13_camera_check.py identify
```

Expected: two PnP cameras (the laptop's "Integrated Webcam" and the USB one,
with its own VID:PID) and two OpenCV indices, each delivering frames. Unplug
the USB webcam and run again — the index that disappears is the USB webcam.
Write that index down; it is `N` in every command below. Note the **controls**
line: whether `exposure`, `gain` and `auto_exposure` are reported (values other
than −1). Cameras whose auto-exposure cannot be disabled sometimes fight IR
illumination at night; knowing this now avoids confusion later.

Then the Stage 1 acceptance test on that index:

```bat
python evaluation\camera_baseline.py --device N --label usb-webcam-unmodified --notes "room light, 60 cm"
```

It must end with `VERDICT: PASS`.

### 1.3 Record normal webcam performance

Sit at the driving distance (50–70 cm), normal room light, face the camera,
eyes open, and run:

```bat
python evaluation\stage13_camera_check.py measure --device N --label usb-unmodified-room --notes "room light, 60 cm, no glasses"
```

This records the capture path, the image statistics **and** the detector on
this camera: FPS, landmark inference time, face rate, valid-frame rate, EAR
median/std, eye width in pixels, head pose. It is the reference every later
step is compared against. Also do the **remote test with the filter still in
place** so there is a before/after pair:

```bat
python evaluation\stage13_camera_check.py live --device N --label usb-unmodified-remote
```

Hold a TV remote ~1 m from the camera, pointing at it, hold a button down.
With the IR-cut filter present the LED is invisible or a faint dot; the `IR
spot` line stays `none` or shows a few pixels. Press `r` to record, `q` to
quit. Optional but useful: `--save-frame` on the measure run.

### 1.4 Identify whether the camera is physically suitable

This cannot be decided by software. Look at the camera from the outside and
answer, with photos:

| Question | Suitable answer | Doubtful | Unsuitable |
|---|---|---|---|
| How does the housing open? | visible screws (often under rubber feet / the label) or snap-fit seams | glued housing that must be pried | ultrasonically welded housing with no seam |
| Lens type? | a **separate small lens barrel** that screws into a holder over the sensor (most 720p/1080p desk webcams: you can see a threaded ring around the lens) | fixed-focus lens glued into the holder (focus cannot be redone by turning) | **autofocus module** (voice-coil lens sealed with the sensor — Logitech C920-class): the filter is inside a sealed unit |
| Any filter glass visible at the front? | no — the filter is inside, behind the lens | a coated glass window at the front (that is usually just a cover, not the IR filter) | |
| Cables inside? | one ribbon or a few wires to the USB board | | |

A webcam with a screw-in lens barrel over a sensor board is the standard
candidate and the assumption behind §3. If the camera turns out to have an
autofocus module or a sealed sensor package, **stop**: the alternative is a
camera that ships IR-sensitive (a "NoIR" USB camera module), not forcing
this one open.

**STOP 1 — report:** the `identify` output, the `measure` numbers (FPS, face
rate, EAR median), the remote-test result, photos of the camera's outside and
the answers to the suitability table. The physical work starts only after that.

---

## 2. What the IR-cut filter is and how to recognise it

A colour camera sensor is sensitive well into the near infrared (to ~1000 nm).
Left alone, IR would wash out colours and defocus the image, so manufacturers
place a thin **IR-cut filter** — a piece of glass with a dielectric coating
that transmits visible light and reflects/absorbs wavelengths above roughly
650–700 nm — somewhere between the lens and the sensor. Removing it is the
whole conversion: afterwards the sensor sees 850 nm light almost as well as
visible light.

How to recognise the filter once the camera is open (do **not** assume where
it is — check each candidate):

| Where it can be | What it looks like | Removable? |
|---|---|---|
| **A. Glued to the back of the lens barrel** (most common on cheap webcams) | a tiny square or round glass, 3–6 mm, on the rear opening of the lens module; **look through it at a lamp: slightly cyan / blue-green tint; tilt it in reflected light: red, magenta or green sheen**, unlike plain glass | yes — the usual case |
| **B. A separate glass in the lens holder, sitting over the sensor** | a small square glass loose or lightly glued in a recess of the plastic holder, between holder and sensor | yes — lift it out |
| **C. On the sensor package itself** (the sensor's cover glass is the filter) | no separate glass anywhere; the sensor window has the coloured sheen | **no** — removing it destroys the sensor |
| **D. Inside the lens stack** (between lens elements) | no glass at the back or in the holder, yet the remote LED stays dim | **no** — not without destroying the lens |

Two tests tell you whether you found it: (1) the **sheen test** above — a
plain protective window is neutral/clear, the IR-cut filter is not; (2) the
**remote test**: with the lens (and its filter) removed and only the bare
sensor exposed to the remote's LED, the LED appears as a bright spot in the
live helper. If it appears bright with the lens off but dim with the lens on,
the filter is in the lens (A or D). If it stays dim even with the lens off,
the filter is on the sensor (C) — the camera is unsuitable.

## 3. Risks — read before opening anything

| Risk | Consequence | Mitigation |
|---|---|---|
| Dust, fingerprint or scratch on the **sensor** | permanent dark spots / blur on every frame | never touch the sensor window; work on a clean table; keep the lens off the sensor for as short a time as possible; blow dust off with a bulb blower, never canned air at close range, never a cloth on the sensor |
| Static discharge | dead sensor / USB chip | touch a grounded metal object first; avoid carpets and synthetic clothing; hold boards by the edges |
| **Filter glass shatters** while prying | shards on the sensor, in the eye | safety glasses; pry over a tray; if it cracks, remove all fragments with tweezers, then blow clean |
| Lens barrel comes out at an unknown position | focus lost; refocusing takes trial and error | **count and note the turns** while unscrewing; mark the barrel and holder with a fine marker before moving anything |
| Thread-lock / glue on the lens barrel | barrel cannot be turned; forcing strips the plastic thread | a drop of isopropyl alcohol on the thread, wait a minute, gentle pressure; if it will not move, stop and report |
| Ribbon cable / microphone wire torn | camera dead | photograph before disconnecting; pull connectors straight, never by the wires |
| Lost screws | housing will not close | a lidded container; one photo per disassembly step |
| Focus shift from the missing glass | the glass added optical path; without it the sharp plane moves — the image **will** be slightly out of focus after reassembly | expected; §5 refocuses with the live helper |
| Colour is wrong afterwards | daylight looks pink / washed out; auto white balance drifts | expected and harmless: the detector uses greyscale landmarks (`--gray` also available); the change is a known, intended trade-off |
| Camera never works again | | it is a cheap USB webcam and the modification is irreversible — accept that before starting; the laptop webcam remains the fallback for every software demo |
| Warranty | void | accepted |

Tools: small Phillips / Torx / flat screwdrivers (check what the screws are
first), a plastic spudger or guitar pick, fine tweezers, a craft knife or
razor blade, a bulb blower, isopropyl alcohol, safety glasses, a lidded
container for screws, good light and a phone for photos at every step, and
ideally a magnifier.

---

## 4. Procedure — removing the IR-cut filter

Each step ends with STOP. Photograph before and after each step.

**Step A — open the housing.** Unplug the camera. Peel the label / rubber
feet if the screws hide there; undo the screws; separate the halves with the
spudger at the seam. Photograph the inside: the sensor board, the lens barrel,
any cables, any glass. Do not yet touch the lens.
**STOP A — report photos and describe: is the lens a screw-in barrel? Any
glass visible at the back of the lens or over the sensor?**

**Step B — free the lens barrel.** Mark the barrel's orientation against the
holder with a marker dot. Unscrew the barrel **counting the turns** (e.g.
"2¾ turns") — this number is the coarse refocus setting later. If it is glued
in place, use the alcohol trick; if it still will not move, STOP and report.
As soon as the barrel is out, cover the exposed sensor opening (lens cap,
clean tape over the holder rim, or simply put the barrel back loosely) —
every second the sensor is open is a dust risk.
**STOP B — report the number of turns, and whether a glass is now visible on
the rear of the barrel (case A), in the holder (case B), or nowhere (C/D).**

**Step C — confirm it is the filter.** Sheen test on the candidate glass;
then the bare-sensor remote test: plug the camera in with the barrel off, run
`live`, point the remote at the sensor from ~30 cm — a bright spot means the
sensor sees IR and the filter is in what you removed. Unplug again.
**STOP C — report which case (A/B/C/D) it is. C or D: the camera is not
suitable; do not proceed.**

**Step D — remove the filter.**
*Case A (glued to the barrel):* hold the barrel rear side up over a tray. Warm
it gently for ~20 s with a hair dryer on low (softens the glue; do not heat
until the plastic deforms). Slide the tip of the craft knife under one edge
of the glass and lift with a steady twist. It usually pops off in one piece;
if it cracks, remove every fragment. Do not scratch the rear lens element.
Blow the barrel clean; look through it at a lamp — it must now be clear with
no coloured sheen.
*Case B (glass in the holder):* lift it out with tweezers; blow the holder
and sensor clean.
**STOP D — report: filter out in one piece? Any residue or damage visible on
the rear lens element? Sensor clean?**

## 5. Reassembly and refocus

**Step E — reassemble.** Screw the barrel back in to the marked orientation
and turn count, close the housing loosely (you may need to reopen it to
adjust focus unless the barrel is reachable from the front — many webcams
allow turning the lens through the front bezel).
**STOP E — the camera is closed; plug it in and confirm `identify` still lists
it and delivers frames.**

**Step F — refocus.** The filter glass added optical path; without it, the
plane of sharp focus has moved slightly, so the image will be soft until the
barrel is re-set. Tape a page of printed text at the driving distance
(50–70 cm), room light, and run:

```bat
python evaluation\stage13_camera_check.py live --device N --label noir-refocus
```

Turn the lens barrel a fraction of a turn at a time and watch the **FOCUS
centre** number — turn until it peaks (the text goes green when within 5 % of
the best value seen; `f` resets the best). Overshoot on purpose once to be
sure you passed the peak, then return to it. Press `r` to record. Then fix the
barrel position lightly (a small piece of tape on the barrel/holder junction
— glue only after every test in §6 has passed) and close the housing.
**STOP F — report the recorded focus values before (from step 1.3) and after.**

## 6. Post-modification tests

Same seat, same distance, same framing as step 1.3 for every run.

| # | Step | Command / action | What to look for |
|---|---|---|---|
| 1 | Remote test | `live --device N --label noir-remote`; remote at 1 m, button held; press `r` | `IR spot: YES` with a blob of tens to hundreds of pixels, magenta circle on the LED — versus `none` before the conversion |
| 2 | Daylight | `measure --device N --label noir-daylight --notes "..."` | colours look pink/washed (expected); `channel_spread` **drops** versus `usb-unmodified-room`; face rate, valid rate and EAR std should be about the same as before — that is the "conversion did not hurt detection in daylight" claim |
| 3 | Dim lighting | dim the room (one small lamp, or curtains); `measure --label noir-dim-ir-off` | brightness falls; note face rate and EAR std — this is where an unmodified camera starts to fail |
| 4 | Darkness | lights off, no IR; `measure --label noir-dark-ir-off` | expect near-black: mean brightness < 10–20, face rate near 0. **This is the reference the illuminator is judged against**; it must be recorded, not assumed |

```bat
python evaluation\stage13_camera_check.py compare usb-unmodified-room noir-daylight
```

**STOP G — report the four measurements and the compare table.**

## 7. Integrating the 850 nm illuminator

### 7.1 Safety — 850 nm is invisible, so the eye does not protect itself

- The beam is barely visible (a faint red glow at most). There is **no blink
  or aversion reflex**, so a bright IR source can be stared into without
  discomfort. Treat the board like a bright lamp you cannot see.
- This project deliberately uses a **small 3-LED board**, not a CCTV array.
  Still: keep it **at least 50 cm from any eye**, never point it into an eye
  from close range, never look into the LEDs from a few centimetres to "check
  if it is on" — use the NoIR camera or a phone camera (most phone cameras
  show 850 nm as a faint purple glow) instead. Limit test sessions to minutes,
  not hours, until the exposure is characterised.
- **Power:** read the board's label. CCTV IR boards are commonly 12 V DC; some
  small ones are 5 V. Use a supply of exactly the labelled voltage, correct
  polarity, and a current rating above the board's draw. Wrong voltage burns
  the LEDs or the driver. Never feed it from the ESP32's pins.
- **Heat:** after 5 minutes on, the board should be warm, not too hot to
  touch. If it is, stop and reconsider the supply.
- Many boards have a **light sensor (LDR)** that switches the LEDs on only in
  darkness. For daytime tests, cover the LDR with a piece of black tape.
- Do not power the illuminator while handling the camera internals.

### 7.2 Placement

Mount camera and illuminator together on the adjustable mount, illuminator
5–10 cm beside the lens, both aimed at the driver's face from 50–70 cm.
Do **not** mount the illuminator on-axis right next to the lens if the driver
wears glasses — the reflection of the LEDs in the lenses can cover the eyes;
an offset of 10–15 cm and a slight angle usually moves the reflection off the
pupils. **No diffuser yet** — the first question is whether one is needed.

### 7.3 Evenness test

Lights off, illuminator on, same seat and distance:

```bat
python evaluation\stage13_camera_check.py live --device N --label noir-dark-ir-on
python evaluation\stage13_camera_check.py measure --device N --label noir-dark-ir-on --notes "IR on, X cm, lights off"
python evaluation\stage13_camera_check.py compare noir-dark-ir-off noir-dark-ir-on
python evaluation\stage13_camera_check.py compare usb-unmodified-room noir-dark-ir-on
```

Read the 3×3 map and the numbers with the face filling the centre cell:

| Observation | Meaning | Action |
|---|---|---|
| `mean_brightness` rises from near-black to a usable level; face rate and valid rate close to the daylight run; EAR std similar | the illuminator does the job | keep as is |
| `saturated_fraction` above ~1 % **on the face**, or a bright washed-out forehead/nose with dark cheeks in the map, or `uniformity ratio` below ~0.5 with the centre cell brightest | hotspot: too much light on-axis, too little at the edges | first try more distance or a slight angle; if that does not fix it, **that** is the evidence for buying the diffuser |
| brightness rises but face rate stays low | landmarks fail on the IR image (too dark, too noisy, or glasses reflections) | check exposure/gain controls found in §1.2; move the illuminator; check reflections |
| `EAR std` much higher than daylight | noisy landmarks in IR | same as above; compare `--gray` |

The thresholds in that table are starting points for judgement, not tuned
values — record what actually happens.

**STOP H — report the dark/IR-on measurement, the compare tables and a
snapshot of the 3×3 map. The diffuser decision is made here, on evidence.**

## 8. Reading a `compare` table

| Metric | Expected direction after filter removal (same light) | After IR on (dark room) |
|---|---|---|
| `channel spread` (max−min of B/G/R means) | **down** — IR leaks into all channels | small (near-monochrome scene) |
| `mean brightness` | up a little in daylight | **up strongly** versus IR off |
| `focus (Laplacian)` | down right after reassembly, back near the old value after refocus | lower than daylight is normal (less texture) |
| `saturated fraction`, `largest bright blob px` | remote test: **up** (the LED becomes a spot) | hotspot indicator |
| `uniformity ratio` | unchanged | the evenness verdict |
| `face rate`, `valid rate`, `EAR std`, `eye width px` | about unchanged in daylight | the low-light claim of the project |
| `CNN P(closed)` | only once `models/eye_cnn.pt` exists | the CNN was trained on IR eye images (MRL) — this is where it should shine |

## 9. What Stage 13 delivers when done

- `evaluation/results/stage13/` with the identify JSON and one measure JSON per
  step (`usb-unmodified-room`, `noir-daylight`, `noir-dim-ir-off`,
  `noir-dark-ir-off`, `noir-dark-ir-on`, plus the `live-*` remote/refocus
  records), committed.
- Photos of the disassembly in `hardware/photos/`.
- The README's Stage 13 section filled with the measured numbers — not with
  the expectations written in this document.
- A decision, on evidence, about the diffuser (Stage 14 buys it only if §7.3
  says so).
