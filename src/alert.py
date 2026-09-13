"""Escalating laptop alerts driven by the drowsiness state (Stage 10).

Pipeline slice implemented here::

    TemporalState (Stage 9)  ->  AlertManager.update()  ->  AlertStatus (what to show / play)
      state ALERT/MILD/DROWSY        policy + cooldowns          |            |
      closure_now_s, reasons         + dismissal                 v            v
                                                          draw_alert_overlay  AudioWorker thread
                                                          (main thread, cv2)  (winsound / TTS)
                                                                              |
                                                                              v
                                                                        alert log (CSV)

Escalation ladder
-----------------
    level 0  NONE    state ALERT                nothing
    level 1  VISUAL  state MILD                 amber banner (optionally one soft beep on entry)
    level 2  BEEP    state DROWSY               red flashing banner + beep every ``beep_interval_s``
    level 3  VOICE   DROWSY for >= voice_after_s, or eyes closed right now for >= voice_closure_s
                                                banner + spoken warning every ``voice_cooldown_s``
                                                (beeps continue between utterances)

Why a closed-eye shortcut to voice: a driver whose eyes are shut cannot see a
banner, and a 250 ms beep is easy to sleep through; speech is the modality that
still works. The threshold (2 s) is deliberately above the Stage 9 microsleep
threshold (1.5 s) so the state machine, not the alert layer, decides drowsiness.

Design rules (the roadmap's requirements)
-----------------------------------------
* **Detection and alerting are separate.** This module never sees frames, EAR
  or CNN outputs; it only reads ``TemporalState``. ``src.features`` wires the
  two together in ~6 lines.
* **Never freeze the vision pipeline.** ``AlertManager.update()`` is pure
  bookkeeping plus a non-blocking ``queue.put``. All sound (blocking
  ``winsound.Beep``, text-to-speech) runs on one daemon ``AudioWorker`` thread.
  The overlay is a few ``cv2`` rectangles and strings on the main thread.
* **No continuous alarm.** Each audible level has a cooldown; the queue holds
  at most one pending sound per kind; a per-episode beep cap is available.
* **Manual dismissal** (``d`` key): silences audio for ``dismiss_s`` seconds
  and cuts off the sound that is playing. The visual warning stays - a driver
  cannot dismiss what the camera sees. Audio re-arms automatically when the
  level *escalates* (e.g. MILD -> DROWSY, or voice becomes due) or when the
  silence expires while the state is still MILD/DROWSY.
* **Every alert event is logged** with wall-clock ISO timestamp and the
  pipeline's monotonic time: RAISED, ESCALATED, BEEP, VOICE, DISMISSED,
  REARMED, CLEARED, plus SUPPRESSED (why a sound was *not* played), so the
  cooldown logic is auditable after a session.
* **Laptop only.** Nothing here knows about the ESP32 (Stage 11); that will be
  another consumer of the same ``AlertStatus``.

Time base: decisions use ``TemporalState.t`` (the detector's monotonic seconds),
so ``--replay`` of a recorded CSV reproduces exactly the alerts a live run
would have raised, at the recorded frame times. The log adds wall-clock time.

Run directly::

    python -m src.alert --self-test                     # policy, cooldowns, dismissal, threading; silent
    python -m src.alert --test-sounds                   # play beep (mild), beep (drowsy), voice once
    python -m src.alert --replay data/drowsy_session.csv [--audio] [--log logs/replay_alerts.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.temporal import ALERT, DROWSY, MILD, TemporalState

NONE, VISUAL, BEEP, VOICE = 0, 1, 2, 3
LEVEL_NAMES = {NONE: "NONE", VISUAL: "VISUAL", BEEP: "BEEP", VOICE: "VOICE"}
LOG_DIR = Path("logs")
LOG_FIELDS = ["timestamp", "t_s", "event", "level", "state", "detail", "perclos", "closure_now_s", "reasons"]


# --- configuration -------------------------------------------------------------

@dataclass
class AlertConfig:
    """Alert policy. All times in seconds. Every value is an engineering choice
    to be confirmed in the Stage 10 live test (none is a measured result)."""

    # level 1: MILD
    mild_beep_on_entry: bool = True        # one soft beep when MILD is first entered
    mild_beep_hz: int = 660
    mild_beep_ms: int = 150

    # level 2: DROWSY beeps
    beep_hz: int = 1000
    beep_ms: int = 250
    beep_interval_s: float = 3.0           # cooldown between beeps while DROWSY
    max_beeps_per_episode: int = 0         # 0 = unlimited (cooldown alone limits the rate)

    # level 3: voice
    voice_after_s: float = 6.0             # DROWSY sustained this long -> voice
    voice_closure_s: float = 2.0           # eyes closed right now this long -> voice immediately
    voice_cooldown_s: float = 15.0
    voice_text: str = "Wake up. You are showing signs of drowsiness. Pull over and take a break."
    voice_repeat_text: str = "Wake up. Please pull over and rest."
    tts_backend: str = "auto"              # auto | pyttsx3 | powershell | none

    # dismissal
    dismiss_s: float = 30.0                # audio silence after 'd'; visual stays

    # visual
    flash_hz: float = 2.0                  # DROWSY banner flash rate

    # output
    audio: bool = True                     # False = fully silent (log + overlay only)
    log_path: Optional[Path] = None        # None = logs/alerts_<timestamp>.csv


@dataclass
class AlertStatus:
    """What the rest of the program needs each frame: drawn by the overlay,
    consumed later by the ESP32 link (Stage 11) and the dashboard (Stage 12)."""

    t: float
    level: int                             # NONE / VISUAL / BEEP / VOICE
    state: str
    level_since: float                     # when the current level began
    episode_since: float                   # when the current MILD/DROWSY episode began (NaN if none)
    dismissed_until: float                 # monotonic time until which audio is muted (NaN if not)
    beeps_this_episode: int
    voices_this_episode: int
    message: str                           # banner text
    reasons: List[str] = field(default_factory=list)

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES[self.level]

    @property
    def dismissed(self) -> bool:
        return not math.isnan(self.dismissed_until) and self.t < self.dismissed_until

    def as_row(self) -> Dict[str, object]:
        return {"alert_level": self.level_name, "alert_dismissed": int(self.dismissed),
                "alert_beeps": self.beeps_this_episode, "alert_voices": self.voices_this_episode}


# --- audio backends -----------------------------------------------------------

def beep_winsound(hz: int, ms: int) -> None:
    import winsound
    winsound.Beep(int(hz), int(ms))


def beep_sounddevice(hz: int, ms: int) -> None:
    """Fallback for non-Windows machines (sounddevice is already in the venv)."""
    import sounddevice as sd
    rate = 44100
    n = int(rate * ms / 1000.0)
    t = np.arange(n) / rate
    tone = 0.4 * np.sin(2 * math.pi * hz * t)
    fade = min(200, n // 4)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    sd.play(tone.astype(np.float32), rate)
    sd.wait()


def make_beeper() -> Tuple[Callable[[int, int], None], str]:
    if platform.system() == "Windows":
        try:
            import winsound  # noqa: F401
            return beep_winsound, "winsound"
        except ImportError:
            pass
    try:
        import sounddevice  # noqa: F401
        return beep_sounddevice, "sounddevice"
    except Exception:  # pragma: no cover - no audio device at all
        return (lambda hz, ms: time.sleep(ms / 1000.0)), "silent"


class Speaker:
    """Text-to-speech with three interchangeable backends.

    pyttsx3    Windows SAPI5 (also NSSpeech / espeak elsewhere). Fast after the
               first call (~0.2 s vs ~2 s for the first utterance, measured on
               the dev laptop). Cannot be interrupted mid-sentence.
    powershell Windows System.Speech via a child process. No dependency, ~1.5 s
               start-up per utterance, but ``stop()`` kills it instantly - so
               dismissal cuts the voice off.
    none       prints the text (machines without TTS); ``mute`` does nothing (tests).

    The engine is created lazily *on the audio thread* (pyttsx3/COM objects are
    thread-affine on Windows).
    """

    def __init__(self, backend: str = "auto") -> None:
        self.requested = backend
        self.backend = self._resolve(backend)
        self._engine = None
        self._proc: Optional[subprocess.Popen] = None

    @staticmethod
    def _resolve(backend: str) -> str:
        if backend != "auto":
            return backend
        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except ImportError:
            pass
        if platform.system() == "Windows" and shutil.which("powershell"):
            return "powershell"
        return "none"

    def speak(self, text: str) -> None:
        if self.backend == "pyttsx3":
            import pyttsx3
            if self._engine is None:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 175)
                self._engine.setProperty("volume", 1.0)
            self._engine.say(text)
            self._engine.runAndWait()
        elif self.backend == "powershell":
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      "$s.Rate = 1; $s.Volume = 100; $s.Speak('{}')".format(text.replace("'", "''")))
            self._proc = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc.wait()
            self._proc = None
        elif self.backend == "none":
            print("[alert] (voice) {}".format(text))

    def stop(self) -> None:
        """Cut off the current utterance where the backend allows it."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


class AudioWorker:
    """One daemon thread that owns every blocking sound call.

    ``enabled=False`` makes it a dry run: nothing is played, ``play()`` still
    accepts every request, so ``--replay`` without ``--audio`` logs the exact
    alert timeline a live run would produce.

    ``play()`` never blocks: it drops the request if a sound of the same kind is
    already waiting (there is no point queuing beeps behind beeps) and returns.
    ``silence()`` invalidates everything queued and stops the current voice.
    """

    def __init__(self, beeper: Optional[Callable[[int, int], None]] = None, speaker: Optional[Speaker] = None,
                 enabled: bool = True) -> None:
        self.enabled = enabled
        self.beeper, self.beeper_name = (beeper, "custom") if beeper else make_beeper()
        self.speaker = speaker or Speaker("auto" if enabled else "none")
        self._queue: "queue.Queue[Optional[Tuple[int, str, tuple]]]" = queue.Queue()
        self._generation = 0
        self._pending: Dict[str, int] = {"beep": 0, "voice": 0}
        self._lock = threading.Lock()
        self.played: List[Tuple[float, str]] = []      # (wall time, kind) for tests / stats
        self.errors: List[str] = []
        self._thread = threading.Thread(target=self._run, name="alert-audio", daemon=True)
        self._thread.start()

    def play(self, kind: str, *args) -> bool:
        """Queue a sound. Returns False only when a sound of that kind is already
        pending. With audio disabled every request is accepted and dropped, so the
        manager's decisions and log are identical in a silent (dry) run."""
        if not self.enabled:
            return True
        with self._lock:
            if self._pending[kind] > 0:
                return False
            self._pending[kind] += 1
            self._queue.put_nowait((self._generation, kind, args))
        return True

    def silence(self) -> None:
        with self._lock:
            self._generation += 1
        self.speaker.stop()

    def close(self, timeout: float = 2.0) -> None:
        self.silence()
        self._queue.put_nowait(None)
        self._thread.join(timeout)

    @property
    def busy(self) -> bool:
        return self._pending["beep"] > 0 or self._pending["voice"] > 0

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, kind, args = item
            try:
                with self._lock:
                    stale = generation != self._generation
                if stale:
                    continue
                if kind == "beep":
                    self.beeper(*args)
                elif kind == "voice":
                    self.speaker.speak(*args)
                self.played.append((time.time(), kind))
            except Exception as exc:  # audio must never take the pipeline down
                self.errors.append("{}: {}".format(kind, exc))
            finally:
                with self._lock:
                    self._pending[kind] -= 1


# --- logging --------------------------------------------------------------------

class AlertLog:
    """Append-only CSV of alert events. Opened lazily on the first event so a
    session with no alerts leaves no file behind."""

    def __init__(self, path: Optional[Path] = None, echo: bool = True) -> None:
        self.path = path or LOG_DIR / "alerts_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.echo = echo
        self.rows = 0
        self._file = None
        self._writer = None

    def write(self, t: float, event: str, level: int, state: str, detail: str, perclos: float,
              closure_now_s: float, reasons: List[str]) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(LOG_FIELDS)
        stamp = datetime.now().isoformat(timespec="milliseconds")
        self._writer.writerow([stamp, round(t, 3), event, LEVEL_NAMES[level], state, detail,
                               round(perclos, 4), round(closure_now_s, 2), "; ".join(reasons)])
        self._file.flush()
        self.rows += 1
        if self.echo:
            print("[alert] {} t={:7.1f}s {:<10} {:<6} {:<6} {}".format(
                stamp[11:23], t, event, LEVEL_NAMES[level], state, detail))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# --- the manager ------------------------------------------------------------------

class AlertManager:
    """Turns the Stage 9 state stream into escalating alerts.

    Call ``update(ts)`` once per frame with the ``TemporalState``; call
    ``dismiss()`` when the driver presses the dismiss key; draw ``status`` with
    ``draw_alert_overlay``. ``close()`` at shutdown.
    """

    def __init__(self, config: Optional[AlertConfig] = None, audio: Optional[AudioWorker] = None,
                 log: Optional[AlertLog] = None, echo: bool = True) -> None:
        self.config = config or AlertConfig()
        cfg = self.config
        self.audio = audio or AudioWorker(enabled=cfg.audio,
                                          speaker=Speaker(cfg.tts_backend if cfg.audio else "none"))
        self.log = log or AlertLog(cfg.log_path, echo=echo)
        self.level = NONE
        self.level_since = 0.0
        self.episode_since = math.nan
        self.dismissed_until = math.nan
        self.beeps = 0
        self.voices = 0
        self.last_beep_t = -math.inf
        self.last_voice_t = -math.inf
        self.status: Optional[AlertStatus] = None
        self.events: List[Tuple[float, str, int]] = []   # (t, event, level) - in memory for tests / summary
        self._dismiss_requested = False
        self._ever_updated = False
        self._busy_logged = False

    # -- public API --------------------------------------------------------------

    def dismiss(self) -> None:
        """Driver pressed the dismiss key. Applied on the next update() so that
        the decision and the log entry carry the pipeline's time base."""
        self._dismiss_requested = True

    def update(self, ts: TemporalState) -> AlertStatus:
        cfg = self.config
        t = ts.t
        if not self._ever_updated:
            self.level_since = t
            self._ever_updated = True

        # 1. the driver's dismissal
        if self._dismiss_requested:
            self._dismiss_requested = False
            if self.level > NONE:
                self.dismissed_until = t + cfg.dismiss_s
                self.audio.silence()
                self._log(ts, "DISMISSED", self.level, "audio muted {:.0f} s; visual stays".format(cfg.dismiss_s))
            else:
                self._log(ts, "DISMISS_IGNORED", self.level, "no active alert")

        # 2. the level the state calls for
        target = self._target_level(ts, t)

        # 3. transitions between levels
        if target != self.level:
            previous = self.level
            self.level, self.level_since = target, t
            if target == NONE:
                self.audio.silence()
                self._log(ts, "CLEARED", NONE, "episode {:.0f} s, {} beeps, {} voice".format(
                    t - self.episode_since if not math.isnan(self.episode_since) else 0.0, self.beeps, self.voices))
                self.episode_since = math.nan
                self.dismissed_until = math.nan
                self.beeps = self.voices = 0
                self.last_beep_t = self.last_voice_t = -math.inf
            elif previous == NONE:
                self.episode_since = t
                self._log(ts, "RAISED", target, "state {}".format(ts.state))
                if target == VISUAL and cfg.mild_beep_on_entry and self._audio_allowed(t):
                    if self.audio.play("beep", cfg.mild_beep_hz, cfg.mild_beep_ms):
                        self._log(ts, "BEEP", target, "soft entry beep {} Hz {} ms".format(
                            cfg.mild_beep_hz, cfg.mild_beep_ms))
            elif target > previous:
                self._log(ts, "ESCALATED", target, "{} -> {}".format(LEVEL_NAMES[previous], LEVEL_NAMES[target]))
                if not math.isnan(self.dismissed_until) and t < self.dismissed_until:
                    self.dismissed_until = math.nan
                    self._log(ts, "REARMED", target, "escalation overrides dismissal")
            else:
                self._log(ts, "DEESCALATED", target, "{} -> {}".format(LEVEL_NAMES[previous], LEVEL_NAMES[target]))

        # 4. audible actions for the current level, subject to cooldown / dismissal
        if self.level >= BEEP:
            if not self._audio_allowed(t):
                pass                                   # dismissed: log nothing per frame, the overlay shows it
            else:
                if not math.isnan(self.dismissed_until) and t >= self.dismissed_until:
                    self.dismissed_until = math.nan
                    self._log(ts, "REARMED", self.level, "dismissal expired, still {}".format(ts.state))
                if self.level == VOICE and t - self.last_voice_t >= cfg.voice_cooldown_s:
                    text = cfg.voice_text if self.voices == 0 else cfg.voice_repeat_text
                    if self.audio.play("voice", text):
                        self.voices += 1
                        self.last_voice_t = t
                        self.last_beep_t = t          # do not beep over the speech
                        self._busy_logged = False
                        self._log(ts, "VOICE", VOICE, '"{}"'.format(text))
                elif t - self.last_beep_t >= cfg.beep_interval_s:
                    if cfg.max_beeps_per_episode and self.beeps >= cfg.max_beeps_per_episode:
                        if self.beeps == cfg.max_beeps_per_episode:
                            self.beeps += 1           # log the cap once
                            self._log(ts, "SUPPRESSED", self.level,
                                      "beep cap {} per episode reached".format(cfg.max_beeps_per_episode))
                    elif self.audio.play("beep", cfg.beep_hz, cfg.beep_ms):
                        self.beeps += 1
                        self.last_beep_t = t
                        self._busy_logged = False
                        self._log(ts, "BEEP", self.level, "{} Hz {} ms, next in {:.0f} s".format(
                            cfg.beep_hz, cfg.beep_ms, cfg.beep_interval_s))
                    elif not self._busy_logged:              # retried every frame, logged once per stretch
                        self._busy_logged = True
                        self._log(ts, "SUPPRESSED", self.level, "audio thread still playing the previous sound")

        self.status = AlertStatus(
            t=t, level=self.level, state=ts.state, level_since=self.level_since, episode_since=self.episode_since,
            dismissed_until=self.dismissed_until, beeps_this_episode=self.beeps, voices_this_episode=self.voices,
            message=self._message(ts), reasons=list(ts.reasons))
        return self.status

    def close(self) -> None:
        self.audio.close()
        self.log.close()

    # -- internals -------------------------------------------------------------------

    def _target_level(self, ts: TemporalState, t: float) -> int:
        cfg = self.config
        if ts.state == ALERT:
            return NONE
        if ts.state == MILD:
            return VISUAL
        # DROWSY
        drowsy_for = t - ts.state_since
        if ts.closed_now and ts.closure_now_s >= cfg.voice_closure_s:
            return VOICE
        if drowsy_for >= cfg.voice_after_s:
            return VOICE
        return BEEP if self.level < VOICE else VOICE   # once at VOICE, stay there for the DROWSY episode

    def _audio_allowed(self, t: float) -> bool:
        return math.isnan(self.dismissed_until) or t >= self.dismissed_until

    def _message(self, ts: TemporalState) -> str:
        if self.level == NONE:
            return ""
        if self.level == VISUAL:
            return "MILD DROWSINESS - take a break soon"
        if self.level == BEEP:
            return "DROWSY - WAKE UP"
        return "DROWSY - WAKE UP - PULL OVER"

    def _log(self, ts: TemporalState, event: str, level: int, detail: str) -> None:
        if event in ("BEEP", "VOICE") and not self.audio.enabled:
            detail += " [audio off]"
        self.events.append((ts.t, event, level))
        self.log.write(ts.t, event, level, ts.state, detail, ts.perclos, ts.closure_now_s, ts.reasons)


# --- display ------------------------------------------------------------------------

LEVEL_COLORS = {VISUAL: (0, 200, 255), BEEP: (0, 0, 255), VOICE: (0, 0, 255)}


def draw_alert_overlay(frame: np.ndarray, status: Optional[AlertStatus], flash_hz: float = 2.0) -> None:
    """Banner across the lower-middle of the frame (clear of the HUD, the eye
    boxes and the crop panel) and, for DROWSY levels, a flashing red border.
    Pure drawing - a few rectangles and two strings."""
    if status is None or status.level == NONE:
        return
    import cv2

    h, w = frame.shape[:2]
    color = LEVEL_COLORS[status.level]
    flashing = status.level >= BEEP
    phase_on = (int(status.t * flash_hz * 2) % 2 == 0) if flashing else True

    if flashing and phase_on:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 14)

    band_y0, band_y1 = int(h * 0.57) - 32, int(h * 0.57) + 32     # below the face centre, above the crop panel
    overlay_color = color if phase_on else tuple(int(c * 0.45) for c in color)
    cv2.rectangle(frame, (0, band_y0), (w, band_y1), overlay_color, -1)
    text = status.message
    scale = 1.0 if status.level >= BEEP else 0.8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cv2.putText(frame, text, ((w - tw) // 2, (band_y0 + band_y1 + th) // 2), cv2.FONT_HERSHEY_DUPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)

    sub = "level {} {}".format(status.level, status.level_name)
    if status.dismissed:
        sub += "  |  audio dismissed, {:.0f} s left".format(status.dismissed_until - status.t)
    elif status.level >= BEEP:
        sub += "  |  beeps {}  voice {}  |  'd' silences audio for a while".format(
            status.beeps_this_episode, status.voices_this_episode)
    else:
        sub += "  |  'd' dismiss"
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, sub, ((w - sw) // 2, band_y1 + sh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(frame, sub, ((w - sw) // 2, band_y1 + sh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)


# --- replay of a recorded session ---------------------------------------------------

def replay(path: Path, config: AlertConfig, temporal_kwargs: Optional[dict] = None) -> int:
    """Drive Stage 9 from a --record CSV and Stage 10 from its output, exactly
    as the live loop does, and print the resulting alert timeline."""
    from src.temporal import TemporalConfig, TemporalEngine, observations_from_csv

    observations = observations_from_csv(path)
    if not observations:
        print("[replay] no rows in {}".format(path), file=sys.stderr)
        return 1
    engine = TemporalEngine(TemporalConfig(**(temporal_kwargs or {})))
    manager = AlertManager(config)
    t0 = observations[0].t
    print("[replay] {} frames, {:.1f} s, audio {}, tts {} | beep every {:.0f} s, voice after {:.0f} s in DROWSY "
          "or {:.1f} s closure, voice cooldown {:.0f} s".format(
              len(observations), observations[-1].t - t0, "on" if config.audio else "off",
              manager.audio.speaker.backend, config.beep_interval_s, config.voice_after_s,
              config.voice_closure_s, config.voice_cooldown_s))
    wall_start = time.perf_counter()
    levels_time: Dict[int, float] = {NONE: 0.0, VISUAL: 0.0, BEEP: 0.0, VOICE: 0.0}
    last_t = t0
    update_times: List[float] = []
    for obs in observations:
        ts = engine.update(obs)
        if config.audio:                                   # real time so the sounds land where they would live
            wall = time.perf_counter() - wall_start
            if obs.t - t0 > wall:
                time.sleep(obs.t - t0 - wall)
        tick = time.perf_counter()
        status = manager.update(ts)
        update_times.append((time.perf_counter() - tick) * 1000.0)
        levels_time[status.level] += obs.t - last_t
        last_t = obs.t
    if config.audio:
        deadline = time.time() + 10
        while manager.audio.busy and time.time() < deadline:
            time.sleep(0.05)
    manager.close()
    counts: Dict[str, int] = {}
    for _, event, _ in manager.events:
        counts[event] = counts.get(event, 0) + 1
    print("[replay] time per level: " + ", ".join(
        "{} {:.1f} s".format(LEVEL_NAMES[k], v) for k, v in levels_time.items()))
    print("[replay] events: " + (", ".join("{} x{}".format(k, v) for k, v in counts.items()) or "none"))
    print("[replay] update() cost: median {:.3f} ms, max {:.3f} ms over {} frames".format(
        float(np.median(update_times)), max(update_times), len(update_times)))
    if manager.audio.errors:
        print("[replay] audio errors: {}".format(manager.audio.errors))
    print("[replay] log: {} ({} rows)".format(manager.log.path, manager.log.rows))
    return 0


# --- self-test ------------------------------------------------------------------------

def _ts(t: float, state: str, since: float, closed_now: bool = False, closure_now: float = 0.0,
        perclos: float = 0.0, reasons: Optional[List[str]] = None) -> TemporalState:
    return TemporalState(t=t, state=state, state_since=since, reasons=reasons or [], sufficient=True,
                         window_fill_s=60.0, frames=1200, valid_frames=1100, invalid_rate=0.08, perclos=perclos,
                         closed_now=closed_now, closure_now_s=closure_now, longest_closure_s=closure_now,
                         blink_count=10, blink_rate_per_min=10.0, mean_blink_s=0.2, yawn_count=0,
                         yawn_rate_per_min=0.0, nod_count=0, pitch_baseline_deg=-10.0, fusion="fused")


class _FakeBeeper:
    """Records calls; optionally blocks to prove the pipeline thread never waits."""

    def __init__(self, block_s: float = 0.0) -> None:
        self.block_s = block_s
        self.calls: List[Tuple[int, int]] = []

    def __call__(self, hz: int, ms: int) -> None:
        self.calls.append((hz, ms))
        if self.block_s:
            time.sleep(self.block_s)


def _silent_manager(config: AlertConfig, beeper: Optional[_FakeBeeper] = None, tmp: Optional[Path] = None
                    ) -> Tuple[AlertManager, _FakeBeeper]:
    beeper = beeper or _FakeBeeper()
    audio = AudioWorker(beeper=beeper, speaker=Speaker("mute"), enabled=True)
    log = AlertLog(tmp or Path("logs/self_test_alerts.csv"), echo=False)
    return AlertManager(config, audio=audio, log=log), beeper


def _drive(manager: AlertManager, t0: float, seconds: float, state: str, since: float, fps: float = 20.0,
           closed: bool = False, closure_start: Optional[float] = None) -> AlertStatus:
    status = None
    n = int(seconds * fps)
    for i in range(n):
        t = t0 + i / fps
        closure = (t - closure_start) if (closed and closure_start is not None) else 0.0
        status = manager.update(_ts(t, state, since, closed_now=closed, closure_now=closure,
                                    perclos=0.35 if state == DROWSY else 0.18 if state == MILD else 0.02))
        # Simulated time runs thousands of times faster than the wall clock, so give the
        # (instant) fake audio thread a chance to drain before the next simulated frame;
        # otherwise the GIL switch interval alone makes sounds look 'still playing'.
        deadline = time.perf_counter() + 0.5
        while manager.audio.busy and time.perf_counter() < deadline:
            time.sleep(0.0005)
    return status


def self_test() -> int:
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="alert_selftest_"))
    cfg = AlertConfig(beep_interval_s=3.0, voice_after_s=6.0, voice_closure_s=2.0, voice_cooldown_s=15.0,
                      dismiss_s=30.0, mild_beep_on_entry=True)

    # 1. ALERT -> nothing; MILD -> VISUAL with one soft beep; back to ALERT -> CLEARED.
    m, beeper = _silent_manager(cfg, tmp=tmpdir / "t1.csv")
    s = _drive(m, 0.0, 5.0, ALERT, 0.0)
    assert s.level == NONE and not m.events, m.events
    s = _drive(m, 5.0, 10.0, MILD, 5.0)
    time.sleep(0.05)
    assert s.level == VISUAL and s.message.startswith("MILD"), s
    assert [e for _, e, _ in m.events] == ["RAISED", "BEEP"], m.events
    assert beeper.calls == [(cfg.mild_beep_hz, cfg.mild_beep_ms)], beeper.calls
    s = _drive(m, 15.0, 5.0, ALERT, 15.0)
    assert s.level == NONE and m.events[-1][1] == "CLEARED" and math.isnan(s.episode_since), m.events[-1]
    m.close()
    print("[self-test] MILD -> VISUAL banner + one soft beep, ALERT -> CLEARED: ok")

    # 2. DROWSY -> BEEP level: beeps at the cooldown, never faster; VOICE after 6 s in DROWSY;
    #    voice repeats at its own cooldown; beeps pause for the utterance.
    m, beeper = _silent_manager(cfg, tmp=tmpdir / "t2.csv")
    _drive(m, 0.0, 2.0, ALERT, 0.0)
    s = _drive(m, 2.0, 5.9, DROWSY, 2.0)                  # 2.0 .. 7.9 s: BEEP level
    time.sleep(0.05)
    assert s.level == BEEP, s.level_name
    beep_times = [t for t, e, _ in m.events if e == "BEEP"]
    assert np.allclose(beep_times, [2.0, 5.0], atol=0.06), beep_times   # entry, +3 s; 8.0 not yet
    gaps = np.diff(beep_times)
    assert np.all(gaps >= cfg.beep_interval_s - 1e-9), gaps
    s = _drive(m, 7.9, 20.0, DROWSY, 2.0)                  # 7.9 .. 27.9 s: VOICE from t=8.0
    time.sleep(0.05)
    assert s.level == VOICE and s.voices_this_episode == 2, (s.level_name, s.voices_this_episode)
    voice_times = [t for t, e, _ in m.events if e == "VOICE"]
    assert np.allclose(voice_times, [8.0, 23.0], atol=0.06), voice_times  # escalate at 8 s, repeat after 15 s
    all_beeps = [t for t, e, _ in m.events if e == "BEEP"]
    assert np.all(np.diff(all_beeps) >= cfg.beep_interval_s - 1e-9), all_beeps
    assert np.all(np.diff(voice_times) >= cfg.voice_cooldown_s - 1e-9), voice_times
    for v in voice_times:                                  # no beep on top of an utterance
        assert not any(v <= b < v + cfg.beep_interval_s for b in all_beeps), (v, all_beeps)
    assert len(all_beeps) == 7, all_beeps                  # 2, 5 | voice 8 | 11, 14, 17, 20 | voice 23 | 26
    assert "ESCALATED" in [e for _, e, _ in m.events]
    m.close()
    print("[self-test] DROWSY: beeps every {:.0f} s ({}), voice at {} s, repeat at {} s: ok".format(
        cfg.beep_interval_s, beep_times, voice_times[0], voice_times[1]))

    # 3. eyes closed for >= 2 s while DROWSY -> VOICE immediately (before 6 s in DROWSY).
    m, beeper = _silent_manager(cfg, tmp=tmpdir / "t3.csv")
    _drive(m, 0.0, 1.0, ALERT, 0.0)
    s = _drive(m, 1.0, 1.9, DROWSY, 1.0, closed=True, closure_start=0.5)   # closure reaches 2.0 s at t=2.5
    assert s.level == VOICE and s.t < 1.0 + cfg.voice_after_s, (s.level_name, s.t)
    first_voice = [t for t, e, _ in m.events if e == "VOICE"][0]
    assert abs(first_voice - 2.5) < 0.06, first_voice
    m.close()
    print("[self-test] closed eyes 2 s in DROWSY -> voice at {:.2f} s, not after 6 s: ok".format(first_voice))

    # 4. dismissal: audio stops for 30 s, visual stays, escalation re-arms it, expiry re-arms it.
    m, beeper = _silent_manager(cfg, tmp=tmpdir / "t4.csv")
    _drive(m, 0.0, 1.0, ALERT, 0.0)
    _drive(m, 1.0, 4.0, DROWSY, 1.0)                       # beeps at 1.0, 4.0
    m.dismiss()
    s = _drive(m, 5.0, 2.0, DROWSY, 1.0)                    # 5.0 .. 7.0: dismissed
    assert s.dismissed and s.level == BEEP and s.message, s
    n_before = len([e for _, e, _ in m.events if e in ("BEEP", "VOICE")])
    s = _drive(m, 7.0, 20.0, DROWSY, 1.0)                   # 7.0 .. 27.0: voice would be due at 7.0 -> escalation re-arms
    ev = [(t, e) for t, e, _ in m.events]
    assert ("DISMISSED" in [e for _, e in ev]) and ("REARMED" in [e for _, e in ev]), ev
    rearm_t = [t for t, e in ev if e == "REARMED"][0]
    assert abs(rearm_t - 7.0) < 0.06 and not s.dismissed, (rearm_t, s.dismissed)
    assert len([e for _, e, _ in m.events if e in ("BEEP", "VOICE")]) > n_before
    m.close()
    # 4b. dismissal that simply expires while still DROWSY at BEEP level (voice_after huge).
    cfg_b = AlertConfig(beep_interval_s=3.0, voice_after_s=1e9, voice_closure_s=1e9, dismiss_s=10.0)
    m, beeper = _silent_manager(cfg_b, tmp=tmpdir / "t4b.csv")
    _drive(m, 0.0, 1.0, ALERT, 0.0)
    _drive(m, 1.0, 2.0, DROWSY, 1.0)
    m.dismiss()
    _drive(m, 3.0, 12.0, DROWSY, 1.0)                      # dismissed 3.0 .. 13.0, beeps resume at 13.0
    beeps = [t for t, e, _ in m.events if e == "BEEP"]
    assert all(not (3.0 <= b < 13.0) for b in beeps) and any(b >= 13.0 for b in beeps), beeps
    assert "REARMED" in [e for _, e, _ in m.events]
    # dismiss with no alert is ignored (logged, nothing muted)
    _drive(m, 15.0, 1.0, ALERT, 15.0)
    m.dismiss()
    s = _drive(m, 16.0, 0.5, ALERT, 15.0)
    assert m.events[-1][1] == "DISMISS_IGNORED" and math.isnan(s.dismissed_until)
    m.close()
    print("[self-test] dismissal mutes audio {:.0f} s, keeps the banner, re-arms on escalation / expiry: ok".format(
        cfg.dismiss_s))

    # 5. beep cap per episode.
    cfg_c = AlertConfig(beep_interval_s=1.0, voice_after_s=1e9, voice_closure_s=1e9, max_beeps_per_episode=3)
    m, beeper = _silent_manager(cfg_c, tmp=tmpdir / "t5.csv")
    _drive(m, 0.0, 1.0, ALERT, 0.0)
    _drive(m, 1.0, 10.0, DROWSY, 1.0)
    time.sleep(0.05)
    beeps = [t for t, e, _ in m.events if e == "BEEP"]
    assert len(beeps) == 3 and "SUPPRESSED" in [e for _, e, _ in m.events], (beeps, m.events)
    m.close()
    print("[self-test] beep cap 3 per episode -> 3 beeps then SUPPRESSED: ok")

    # 6. the pipeline thread never blocks: a beeper that sleeps 1 s per call must not slow update().
    slow = _FakeBeeper(block_s=1.0)
    m, _ = _silent_manager(AlertConfig(beep_interval_s=0.5, voice_after_s=1e9, voice_closure_s=1e9), beeper=slow,
                           tmp=tmpdir / "t6.csv")
    _drive(m, 0.0, 1.0, ALERT, 0.0)
    costs = []
    for i in range(60):                                  # 3 s of DROWSY at 20 FPS
        t = 1.0 + i / 20.0
        tick = time.perf_counter()
        m.update(_ts(t, DROWSY, 1.0, perclos=0.35))
        costs.append(time.perf_counter() - tick)
    worst_ms = max(costs) * 1000.0
    assert worst_ms < 20.0, "update() blocked for {:.1f} ms".format(worst_ms)
    suppressed = [e for _, e, _ in m.events if e == "SUPPRESSED"]
    assert suppressed, "a busy audio thread must be reported as SUPPRESSED, not waited for"
    m.close(); time.sleep(0.05)
    print("[self-test] audio thread blocked 1 s per beep; update() worst case {:.2f} ms, {} suppressed: ok".format(
        worst_ms, len(suppressed)))

    # 7. the log file has the documented columns and one row per event.
    rows = list(csv.DictReader(open(tmpdir / "t2.csv", encoding="utf-8")))
    assert list(rows[0].keys()) == LOG_FIELDS, rows[0].keys()
    assert all(datetime.fromisoformat(r["timestamp"]) for r in rows)
    assert [r["event"] for r in rows][:3] == ["RAISED", "BEEP", "BEEP"], [r["event"] for r in rows][:5]
    print("[self-test] alert log: {} rows, columns {}: ok".format(len(rows), ", ".join(LOG_FIELDS)))

    # 8. overlay draws for every level without touching frame size, and is cheap.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2  # noqa: F401  (import cost must not count as drawing cost)
    draw_alert_overlay(frame, AlertStatus(t=0.0, level=BEEP, state=DROWSY, level_since=0.0, episode_since=0.0,
                                          dismissed_until=math.nan, beeps_this_episode=0, voices_this_episode=0,
                                          message="warm-up"))
    for level in (NONE, VISUAL, BEEP, VOICE):
        st = AlertStatus(t=1.3, level=level, state=DROWSY, level_since=0.0, episode_since=0.0, dismissed_until=5.0,
                         beeps_this_episode=2, voices_this_episode=1, message="DROWSY - WAKE UP")
        tick = time.perf_counter()
        draw_alert_overlay(frame, st)
        cost_ms = (time.perf_counter() - tick) * 1000.0
        assert cost_ms < 20.0 and frame.shape == (480, 640, 3), cost_ms
    assert frame.any(), "overlay drew nothing"
    print("[self-test] overlay for all levels < 20 ms each: ok")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("[self-test] ALL PASSED")
    return 0


def test_sounds(config: AlertConfig) -> int:
    """Play each audible level once, in order, and report the real backends and timings."""
    audio = AudioWorker(enabled=True, speaker=Speaker(config.tts_backend))
    print("[alert] beep backend: {} | tts backend: {}".format(audio.beeper_name, audio.speaker.backend))
    steps = [("soft MILD beep", "beep", (config.mild_beep_hz, config.mild_beep_ms)),
             ("DROWSY beep", "beep", (config.beep_hz, config.beep_ms)),
             ("voice", "voice", (config.voice_text,))]
    for label, kind, args in steps:
        t0 = time.perf_counter()
        audio.play(kind, *args)
        while audio.busy and time.perf_counter() - t0 < 20:
            time.sleep(0.02)
        print("[alert] {:<15} done in {:.2f} s (main thread was free the whole time)".format(
            label, time.perf_counter() - t0))
        time.sleep(0.4)
    audio.close()
    if audio.errors:
        print("[alert] errors: {}".format(audio.errors))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 10: escalating laptop alerts.")
    parser.add_argument("--self-test", action="store_true", help="Policy / cooldown / dismissal / threading checks")
    parser.add_argument("--test-sounds", action="store_true", help="Play the mild beep, drowsy beep and voice once")
    parser.add_argument("--replay", type=Path, help="Replay a --record CSV through Stage 9 + Stage 10")
    parser.add_argument("--audio", action="store_true", help="With --replay: play the sounds in real time")
    parser.add_argument("--log", type=Path, default=None, help="Alert log CSV (default logs/alerts_<time>.csv)")
    parser.add_argument("--tts", choices=("auto", "pyttsx3", "powershell", "none", "mute"), default="auto")
    parser.add_argument("--beep-interval", type=float, default=3.0)
    parser.add_argument("--voice-after", type=float, default=6.0)
    parser.add_argument("--voice-closure", type=float, default=2.0)
    parser.add_argument("--voice-cooldown", type=float, default=15.0)
    parser.add_argument("--dismiss", type=float, default=30.0)
    parser.add_argument("--max-beeps", type=int, default=0)
    parser.add_argument("--no-mild-beep", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace, audio: bool) -> AlertConfig:
    return AlertConfig(beep_interval_s=args.beep_interval, voice_after_s=args.voice_after,
                       voice_closure_s=args.voice_closure, voice_cooldown_s=args.voice_cooldown,
                       dismiss_s=args.dismiss, max_beeps_per_episode=args.max_beeps,
                       mild_beep_on_entry=not args.no_mild_beep, tts_backend=args.tts, audio=audio,
                       log_path=args.log)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.test_sounds:
        return test_sounds(config_from_args(args, audio=True))
    if args.replay:
        return replay(args.replay, config_from_args(args, audio=args.audio))
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
