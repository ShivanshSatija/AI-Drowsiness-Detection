# Stage 11 — physical alarm: ESP32-WROOM-32 + buzzer

> Stage 13 (NoIR webcam conversion + 850 nm illuminator) is documented separately in [noir_conversion.md](noir_conversion.md).

The laptop runs the entire AI. The ESP32 runs **no** MediaPipe, **no** CNN and
no decision logic: it receives one word over USB serial and drives one buzzer.

```
Python AI (src/features.py → src/hardware.py)
   ↓  USB serial, 115200 8N1, ASCII lines
ESP32-WROOM-32 (hardware/drowsiness_alarm/drowsiness_alarm.ino)
   ↓  GPIO 23
Buzzer
```

If the board is unplugged, the laptop's own Stage 10 alerts (banner, beeps,
voice) keep working unchanged. The buzzer is an *additional* channel.

## 1. Parts

| Part | Notes |
|---|---|
| ESP32-WROOM-32 development board | 30- or 38-pin "DevKit" style, micro-USB or USB-C, CP2102 or CH340 USB bridge |
| Small buzzer | **Active** buzzer preferred (two legs, sounds with plain DC, often a sticker on top). A **passive** buzzer / piezo disc also works — set `BUZZER_PASSIVE 1` in the sketch |
| Breadboard | half size is enough |
| Jumper wires | 2 male-male (buzzer on the breadboard) or 2 female-female (buzzer legs straight to the header) |
| USB **data** cable | many phone cables are charge-only; if no COM port appears, the cable is the first suspect |

Optional, only if the buzzer is rated 5 V or draws more than ~12 mA: one NPN
transistor (2N2222 / S8050 / BC547) and a 1 kΩ resistor — see §8.

## 2. Wiring and pin numbers

```
   ESP32-WROOM-32 DevKit                     Buzzer
   ┌─────────────────────┐
   │              GPIO23 ├────────────────── (+)  longer leg / "+" mark
   │                 GND ├────────────────── (−)
   │  GPIO2 (blue LED)   │   on-board, mirrors the buzzer — no wiring
   │  micro-USB ─────────┼──── USB data cable ──── laptop
   └─────────────────────┘
```

| Signal | ESP32 pin | Why this pin |
|---|---|---|
| Buzzer + | **GPIO 23** | plain general-purpose output; not a boot-strapping pin (0, 2, 5, 12, 15), not input-only (34–39), not a flash pin (6–11), so it stays LOW and silent at boot |
| Buzzer − | **GND** | any GND pin |
| Status LED | **GPIO 2** | the DevKit's on-board blue LED; lights whenever the buzzer is on, blinks slowly when the serial link is lost |

Push the ESP32 into the breadboard so its two pin rows straddle the centre
gutter. Plug the buzzer into two free rows, then jumper `GPIO23 → buzzer +`
and `GND → buzzer −`. Polarity matters for an active buzzer (it simply stays
silent if reversed); a passive piezo does not care.

**Some boards label GPIO 23 as `D23`, others as `23` or `IO23`.** It is on the
right-hand header of the common 30-pin DevKit, next to GPIO 22 and GND.

## 3. Serial protocol

ASCII, one command per line, terminated by `\n` (a trailing `\r` is ignored),
case-insensitive, 115200 baud, 8 data bits, no parity, 1 stop bit.

| Laptop → ESP32 | Meaning | ESP32 → laptop | Buzzer |
|---|---|---|---|
| `HELLO` | the laptop's program is present; arms the link watchdog | `OK HELLO` | unchanged |
| `ALERT` | driver alert | `OK ALERT` | silent |
| `MILD` | mild drowsiness | `OK MILD` | one **100 ms chirp every 2 s** |
| `DROWSY` | drowsy | `OK DROWSY` | **fast beeping, 150 ms on / 150 ms off** (two alternating tones on a passive buzzer) |
| `CLEAR` | silence now (driver dismissed the alert, or the program is exiting) | `OK CLEAR` | silent until the next `MILD` / `DROWSY` |
| `PING` | link check | `PONG <uptime_ms>` | unchanged |
| `STATUS` | ask | `STATUS <state> buzzer=<0/1> age_ms=<ms since last command>` | unchanged |
| `TEST` | local demo, no laptop program needed | `OK TEST` … `TEST DONE` | MILD pattern 3 s, DROWSY pattern 3 s, silence |
| anything else | | `ERR unknown <text>` | unchanged |
| *(board boots)* | | `READY drowsiness_alarm v1 pin=23 passive=0 watchdog_ms=3000` | silent |
| *(no command for 3 s after `HELLO`)* | | `LINK LOST` | **forced silent**, LED blinks slowly |

Rules the two sides agree on:

* **State words are idempotent.** The laptop re-sends the current word every
  second as a heartbeat; repeating `DROWSY` does not restart the pattern.
* **Watchdog is fail-silent.** After `HELLO`, three seconds without any command
  means the AI program has crashed, hung or been unplugged. The board silences
  the buzzer and blinks the LED. A stuck alarm after a crash would be a false
  alarm, and the laptop's Stage 10 alerts remain the primary channel. Before
  `HELLO` (e.g. you typing in the Serial Monitor) there is no watchdog, so a
  manually typed `MILD` keeps chirping until you type something else.
* **`CLEAR` is not `ALERT`.** `ALERT` means "the driver is alert". `CLEAR`
  means "whatever the state, be quiet now" — used when the driver mutes the
  laptop alert with `d` (the buzzer is muted with it) and at program exit.
  Both leave the buzzer silent; the difference is recorded in the board's
  `STATUS` reply and in the logs.
* **The laptop maps AI → word** in `src.hardware.word_for()`: audio muted by
  the driver → `CLEAR`; otherwise the Stage 9 state `ALERT` / `MILD` / `DROWSY`.

## 4. Arduino IDE setup (once)

1. Install **Arduino IDE 2.x** from arduino.cc.
2. *File → Preferences → Additional boards manager URLs*, add
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
3. *Tools → Board → Boards Manager*, search **esp32**, install **"esp32 by
   Espressif Systems"** (3.x is current; the sketch also compiles on 2.x).
4. Plug the board in with the data cable. Windows normally installs the
   CP2102 / CH340 driver by itself; if no new COM port appears in Device
   Manager after a minute, install the driver from Silicon Labs (CP210x) or
   WCH (CH340), then try a different cable.
5. *Tools → Board → esp32 → "ESP32 Dev Module"*. *Tools → Port → the new
   COM port* (the one that is **not** "Standard Serial over Bluetooth link").
   Leave upload speed 921600, flash frequency 80 MHz, partition scheme default.
6. *File → Open* → `hardware/drowsiness_alarm/drowsiness_alarm.ino`. The
   sketch must stay inside a folder of the same name — that is an Arduino
   rule, which is why the file is not directly in `hardware/`.
7. If your buzzer is passive (clicks instead of beeping), change
   `#define BUZZER_PASSIVE 0` to `1`.
8. **Upload** (→ button). If the IDE hangs at `Connecting........`, hold the
   board's **BOOT** button until the dots stop, then release. Some boards need
   this every time; it is normal.
9. *Tools → Serial Monitor*, set **115200 baud** and the line ending to
   **"Newline"** (or "Both NL & CR"). Press the board's **EN**/RST button: you
   must see `READY drowsiness_alarm v1 pin=23 passive=0 watchdog_ms=3000`.

## 5. Test procedure

Run the three steps in order. Do not move to the next until the current one
behaves exactly as described.

### Step 1 — ESP32 and buzzer alone (Serial Monitor, no Python)

| Type in the Serial Monitor | Expect on screen | Expect from the hardware |
|---|---|---|
| *(press EN)* | `READY drowsiness_alarm v1 …` | silence, LED off |
| `PING` | `PONG 1234` (uptime) | — |
| `MILD` | `OK MILD` | a short chirp every 2 s, LED flashes with it, **keeps going** (no watchdog before HELLO) |
| `DROWSY` | `OK DROWSY` | fast beeping, LED flickers |
| `CLEAR` | `OK CLEAR` | silence |
| `TEST` | `OK TEST`, 6 s later `TEST DONE` | MILD pattern for 3 s, DROWSY pattern for 3 s, silence |
| `xyz` | `ERR unknown xyz` | — |
| `HELLO`, then `DROWSY`, then wait | `OK HELLO`, `OK DROWSY`, ~3 s later `LINK LOST` | beeping stops by itself, LED blinks slowly |

If `MILD` produces only faint clicks, the buzzer is passive: set
`BUZZER_PASSIVE 1`, re-upload. If nothing at all: check polarity, check the
jumper really is on GPIO 23, check the LED — if the LED flashes but the buzzer
is silent the wiring is wrong, not the code.

**Close the Serial Monitor before step 2** — only one program can hold the port.

### Step 2 — Python → ESP32 (no AI yet)

```bat
pip install pyserial                       :: once (already in requirements.txt)
python -m src.hardware --list              :: the ESP32 port is marked "<- ESP32?"; Bluetooth ports are never chosen
python -m src.hardware --test              :: scripted protocol test, ~15 s
python -m src.hardware --monitor           :: interactive: type words, see replies
python -m src.hardware --send DROWSY       :: one word
```

`--test` prints every exchange and ends with `TEST PASSED`. You should hear:
3 s of chirps (MILD), 3 s of fast beeping (DROWSY), silence (CLEAR), and about
4 s later the board reports `LINK LOST` because the test deliberately stops
sending. Pass `--port COM9` if auto-detect picks nothing (unusual USB bridge).

### Step 3 — integrated with the drowsiness state machine

```bat
python -m src.features                     :: --serial auto is the default: connects if a board is found, keeps looking if not
python -m src.features --serial COM9       :: fixed port
python -m src.features --no-serial         :: Stage 10 behaviour, no board
```

The start-up banner prints the link state and the window shows an `ESP32 …`
line at the bottom right. Then act drowsy exactly as in the Stage 10 test:

| You | Laptop (Stage 10) | Buzzer (Stage 11) |
|---|---|---|
| eyes open, normal | nothing | silent |
| state reaches MILD | amber banner, one soft beep | chirp every 2 s |
| state reaches DROWSY | red flashing banner, beeps, then voice | fast beeping |
| press `d` (mute audio) | banner stays, laptop audio off | **silent** (`CLEAR`), resumes when the mute expires or the level escalates |
| press `r` / click DISMISS | everything clears, state → ALERT | silent (`ALERT`) |
| unplug the USB cable while DROWSY | keeps alerting | stops within 3 s (board watchdog, if still powered) or immediately (unpowered); the console prints `disconnected` and the link retries every 3 s |
| plug it back in | — | reconnects, `HELLO`, resumes the current state within ~3 s |
| quit with `q` | — | `CLEAR` is sent, then the port closes: silent |

## 6. Expected buzzer behaviour — summary

| Word | Sound | LED |
|---|---|---|
| `ALERT` / `CLEAR` / boot | silent | off |
| `MILD` | 100 ms chirp, every 2 s | flashes with the chirp |
| `DROWSY` | 150 ms on / 150 ms off, continuous | flickers with it |
| link lost | silent | 100 ms blink every 1.5 s |

These patterns are engineering choices for a first hardware test, not measured
results. Loudness and annoyance are judged in Stage 14 in the real setting.

## 7. Safety notes

* **Power.** The buzzer is powered from the ESP32's 3.3 V GPIO. An ESP32 pin
  can source about 40 mA absolute maximum and ~12 mA comfortably. Small 3 V
  active buzzers draw 10–30 mA — fine for a test, but if the buzzer is marked
  **5 V**, or is warm, or the board resets when it sounds, drive it through
  a transistor: GPIO 23 → 1 kΩ → NPN base; emitter → GND; buzzer between
  VIN/5V (+) and collector (−). Never connect the buzzer between GPIO 23 and 5 V.
* **Never power the ESP32 from two sources at once** (USB plus VIN) and never
  feed 5 V into a GPIO pin.
* **Unplug USB before rewiring.** A jumper slipping onto 3V3 or VIN while
  powered can kill the board.
* **Boot pins.** Keep the buzzer off GPIO 0, 2, 12 and 15 — a load there can
  stop the board from booting or flashing. GPIO 23 is chosen for that reason.
* **Fail-silent by design.** If the laptop program dies, the buzzer stops
  within 3 s. This is deliberate: a physical alarm that cannot be stopped is a
  hazard, and the driver-facing laptop alert remains the primary channel.
  It also means the buzzer is only as reliable as the USB link — in the
  vehicle (Stage 14) the cable must be secured.
* **Hearing.** Test at arm's length; some piezos exceed 85 dB at 10 cm.
* **This is a development prototype**, not an automotive-grade device. It must
  never be the only safeguard against drowsy driving.

## 8. Files

| File | Role |
|---|---|
| `hardware/drowsiness_alarm/drowsiness_alarm.ino` | ESP32 firmware (Arduino). Non-blocking `millis()` patterns, line parser, watchdog |
| `src/hardware.py` | Python side: port discovery (VID/PID, Bluetooth excluded), `BuzzerLink` thread with heartbeat and reconnect, `word_for()` mapping, `--list/--test/--monitor/--send/--self-test` |
| `src/features.py` | `--serial auto|COMx`, `--no-serial`; one `set_state()` call per frame |
| `hardware/wiring/` | put a photo of your breadboard here for the report |
| `hardware/photos/` | build photos |
