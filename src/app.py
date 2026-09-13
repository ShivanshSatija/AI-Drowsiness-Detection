"""Streamlit dashboard (Stage 12).

    python -m streamlit run src/app.py            # then open http://localhost:8501

The page never touches the camera itself. ``PipelineProcess`` (src/pipeline.py)
runs camera -> MediaPipe -> EAR/MAR -> head pose -> validity -> eye CNN ->
temporal window -> state machine -> alerts -> ESP32 -> SQLite log on a daemon
*separate process* at the camera's own frame rate; this script polls the
latest ``Snapshot`` a few times a second and draws it. The process boundary
matters: measured on the development laptop, a Python thread hogging the
interpreter lock starved MediaPipe (10 frames in 6 s against 32 FPS alone), and
Streamlit's reruns are that kind of load - the first in-process dashboard ran
at 6 FPS. Across a process boundary the page cannot slow the detector.

The core system stays usable without Streamlit: ``python -m src.features`` is
the OpenCV tool, ``python -m src.pipeline`` the headless runner; all three share
``DrowsinessPipeline``.
"""

from __future__ import annotations

import atexit
import csv
import io
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                     # `streamlit run src/app.py` puts src/ on the path, not the repo root
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.pipeline import DEFAULT_SETTINGS, PipelineProcess, Snapshot  # noqa: E402
from src.session_log import DEFAULT_DB, get_session, list_sessions, read_events, read_metrics  # noqa: E402

STATE_COLORS = {"ALERT": "#1f9d55", "MILD": "#d99a00", "DROWSY": "#d62828"}
LEVEL_COLORS = {"NONE": "#3a3a3a", "VISUAL": "#d99a00", "BEEP": "#d62828", "VOICE": "#9d0208", "OFF": "#3a3a3a"}
STATE_LEVEL = {"ALERT": 0, "MILD": 1, "DROWSY": 2}


@st.cache_resource
def get_worker() -> PipelineProcess:
    worker = PipelineProcess()          # separate process: the page can never slow the detector
    atexit.register(worker.stop)
    return worker


# Set in main(). Module level stays free of Streamlit calls so that the detector's
# spawned child process (which re-imports this file on Windows) does nothing.
worker: Optional[PipelineProcess] = None
settings: Dict[str, Any] = {}


# --- helpers -------------------------------------------------------------------------

def fmt(value: Optional[float], spec: str = "{:.3f}", none: str = "–") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return none
    return spec.format(value)


def badge(text: str, color: str, size: str = "2.2rem") -> str:
    return ("<div style='background:{c};color:white;border-radius:10px;padding:0.35em 0.8em;text-align:center;"
            "font-size:{s};font-weight:700;letter-spacing:0.04em'>{t}</div>").format(c=color, s=size, t=text)


def serial_port_options() -> List[str]:
    options = ["auto", "off"]
    try:
        from src.hardware import list_serial_ports
        for p in list_serial_ports():
            if not p.is_bluetooth:
                options.append(p.device)
    except Exception:
        pass
    return options


def events_frame(events: List[Dict[str, Any]], from_db: bool = False) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["time", "t (s)", "kind", "event", "detail", "state", "PERCLOS"])
    rows = []
    for e in reversed(events):
        if from_db:
            when = (e.get("wall_ts") or "")[11:23]
            event = e["event"] if e["kind"] != "alert" else "{} ({})".format(e["event"], e.get("level") or "")
        else:
            when = datetime.fromtimestamp(e["wall"]).strftime("%H:%M:%S.%f")[:-3]
            event = e["event"] if e["kind"] != "alert" else "{} ({})".format(e["event"], e.get("level") or "")
        rows.append({"time": when, "t (s)": e.get("t") if e.get("t") is not None else e.get("t_s"),
                     "kind": e["kind"], "event": event, "detail": e.get("detail") or "", "state": e.get("state") or "",
                     "PERCLOS": fmt(e.get("perclos"), "{:.1%}", "")})
    return pd.DataFrame(rows)


def summary_frame(s: Dict[str, Any]) -> pd.DataFrame:
    tis, sis = s.get("time_in_state", {}), s.get("seconds_in_state", {})
    totals = s.get("event_totals", {})
    rows = [
        ("Duration", "{:.1f} s".format(s.get("duration_s", 0.0))),
        ("Frames", "{}".format(s.get("frames", 0))),
        ("FPS (end to end)", "{:.1f}".format(s.get("fps_end_to_end", 0.0))),
        ("Face detected", "{:.1%} of frames".format(s.get("face_rate", 0.0))),
        ("Landmark inference (median)", "{:.1f} ms".format(s.get("inference_median_ms", 0.0))),
        ("Invalid frames (session)", "{:.1%}".format(s.get("invalid_session_rate", 0.0))),
        ("Time ALERT", "{:.1f} s ({:.0%})".format(sis.get("ALERT", 0.0), tis.get("ALERT", 0.0))),
        ("Time MILD", "{:.1f} s ({:.0%})".format(sis.get("MILD", 0.0), tis.get("MILD", 0.0))),
        ("Time DROWSY", "{:.1f} s ({:.0%})".format(sis.get("DROWSY", 0.0), tis.get("DROWSY", 0.0))),
        ("State transitions", "{}".format(s.get("transitions", 0))),
        ("Blinks / long closures (whole session)", "{} / {}".format(totals.get("blink", 0), totals.get("closure", 0))),
        ("Yawns / nods (whole session)", "{} / {}".format(totals.get("yawn", 0), totals.get("nod", 0))),
        ("Alert events", "{}".format(s.get("alert_events", "–"))),
        ("Final state", "{} (PERCLOS {})".format(s.get("final_state", "–"), fmt(s.get("final_perclos"), "{:.1%}"))),
    ]
    for key, label in (("ear", "EAR median (p5–p95)"), ("mar", "MAR median (p5–p95)"),
                       ("yaw_deg", "Yaw median (p5–p95)"), ("pitch_deg", "Pitch median (p5–p95)")):
        if key in s:
            v = s[key]
            rows.append((label, "{:.3f} ({:.3f}–{:.3f})".format(v["median"], v["p5"], v["p95"])))
    if "cnn" in s:
        rows.append(("Eye CNN", "left {} | right {} | {} ms".format(s["cnn"]["left"], s["cnn"]["right"],
                                                                     s["cnn"]["median_ms"])))
    if "esp32" in s:
        e = s["esp32"]
        rows.append(("ESP32", "{} | sent {} | acks {} | errors {}".format(e.get("port") or "not connected",
                                                                          e["sent"], e["acks"], e["errors"])))
    if s.get("session_id") is not None:
        rows.append(("Session log", "#{} in {}".format(s["session_id"], s["session_db"])))
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


def csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


# --- sidebar: settings + start/stop --------------------------------------------------------

def sidebar():
    running = worker.running
    with st.sidebar:
        st.title("🚗 Drowsiness Detection")
        st.caption("Stage 12 dashboard. The detector runs in its own process; this page only displays.")
        c1, c2 = st.columns(2)
        start = c1.button("▶ Start", type="primary", disabled=running, width="stretch")
        stop = c2.button("⏹ Stop", disabled=not running, width="stretch")
        if running:
            st.success("Running" + (" — session #{}".format(worker.snapshot().session_id)
                                    if worker.snapshot() and worker.snapshot().session_id else ""))
        st.caption("Settings lock while running. Stop, change, Start.")

        st.subheader("Camera")
        device = st.text_input("Device: index or video path", "0", disabled=running)
        resolution = st.selectbox("Resolution", ["640x480", "1280x720", "320x240"], disabled=running)
        mirror = st.toggle("Mirror preview", True, disabled=running)
        draw_landmarks = st.toggle("Draw landmarks", True, disabled=running)

        st.subheader("Detection")
        use_cnn = st.toggle("Eye-state CNN (Stage 8)", True, disabled=running)
        fusion = st.selectbox("Eye-closure source", ["fused", "cnn", "ear"], disabled=running,
                              help="fused = CNN probability and EAR combined (default); cnn / ear isolate one source")
        perclos_mild = st.slider("PERCLOS → MILD", 0.00, 0.50, float(DEFAULT_SETTINGS["perclos_mild"]), 0.01,
                                 disabled=running,
                                 help="Initial value, untuned. Exit threshold is 2/3 of this. "
                                      "0.00 forces MILD after the 2 s dwell (debug / demo of the alert chain).")
        perclos_drowsy = st.slider("PERCLOS → DROWSY", 0.10, 0.80, float(DEFAULT_SETTINGS["perclos_drowsy"]), 0.01,
                                   disabled=running, help="Initial value, untuned. Exit threshold is 0.73 × this.")
        microsleep = st.slider("Microsleep closure (s)", 0.5, 3.0, float(DEFAULT_SETTINGS["microsleep"]), 0.1,
                               disabled=running, help="A closure this long enters DROWSY immediately.")
        yawn_mar = st.slider("Yawn MAR threshold", 0.30, 1.00, float(DEFAULT_SETTINGS["yawn_mar"]), 0.05, disabled=running)
        with st.expander("Validity gate (Stage 4)"):
            max_yaw = st.number_input("Max |yaw| (°)", 10, 60, int(DEFAULT_SETTINGS["max_yaw"]), 1, disabled=running)
            min_face_px = st.number_input("Min face width (px)", 30, 300, int(DEFAULT_SETTINGS["min_face_px"]), 5,
                                          disabled=running, help="Stage 4 gate; 80 px ≈ a driver at 1.5 m")
            min_eye_px = st.number_input("Min eye width (px)", 5, 60, int(DEFAULT_SETTINGS["min_eye_px"]), 1,
                                         disabled=running)

        st.subheader("Alerts (Stage 10)")
        alerts = st.toggle("Laptop alerts", True, disabled=running)
        mute = st.toggle("Mute laptop audio", False, disabled=running, help="Alerts still decided, drawn and logged")
        tts = st.selectbox("Voice backend", ["auto", "pyttsx3", "powershell", "none"], disabled=running)

        st.subheader("ESP32 buzzer (Stage 11)")
        serial = st.selectbox("Serial port", serial_port_options(), disabled=running,
                              help="auto = first USB-serial bridge; keeps looking every 3 s if none is plugged in")

        st.subheader("Session log (Stage 12)")
        db_on = st.toggle("Log to SQLite", True, disabled=running)
        db_path = st.text_input("Database file", str(DEFAULT_DB), disabled=running)
        metrics_interval = st.number_input("Metrics row every (s)", 0.2, 10.0, 1.0, 0.2, disabled=running)

    width, height = (int(v) for v in resolution.split("x"))
    settings = {
        "device": device.strip() or "0", "width": width, "height": height, "mirror": mirror,
        "draw_landmarks": draw_landmarks, "no_cnn": not use_cnn, "fusion": fusion,
        "perclos_mild": perclos_mild, "perclos_drowsy": perclos_drowsy, "microsleep": microsleep, "yawn_mar": yawn_mar,
        "max_yaw": max_yaw, "min_face_px": min_face_px, "min_eye_px": min_eye_px,
        "alerts": alerts, "mute": mute, "tts": tts,
        "serial": None if serial == "off" else serial,
        "db": db_path if db_on else None, "metrics_interval": metrics_interval,
    }
    return start, stop, settings, db_path


def main() -> None:
    global worker, settings
    st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗", layout="wide",
                       initial_sidebar_state="expanded")
    worker = get_worker()
    start_clicked, stop_clicked, settings, db_path = sidebar()
    if start_clicked:
        worker.start(settings)
        time.sleep(0.3)
        st.rerun()
    if stop_clicked:
        worker.stop()
        st.rerun()


    # --- header --------------------------------------------------------------------------------

    st.markdown("## AI Driver Drowsiness Detection — live dashboard")
    if worker.error:
        st.error("The detector stopped with an error:\n\n```\n{}\n```".format(worker.error))
    elif not worker.running and worker.stop_reason:
        st.info("Detector stopped: {}. Press **Start** in the sidebar to run again.".format(worker.stop_reason))
    elif not worker.running:
        st.info("Press **Start** in the sidebar. Use device **0** for the laptop webcam, or a video path such as "
                "`data/two_faces.avi`. Close `python -m src.features` first — only one program can hold the camera.")


    # --- live view: frame + state + metrics ----------------------------------------------------------

    LIVE_EVERY_S = 0.33      # the detector thread shares the interpreter with this page: poll gently
    SLOW_EVERY_S = 2.0


    @st.fragment(run_every=LIVE_EVERY_S if worker.running else None)
    def live_view() -> None:
        snap: Optional[Snapshot] = worker.snapshot()
        col_video, col_state = st.columns([3, 2], gap="medium")

        with col_video:
            if snap is not None and snap.jpeg:
                st.image(snap.jpeg, width="stretch",
                         caption="frame {} · {:.1f} s · {:.1f} FPS · landmarks {:.1f} ms{}".format(
                             snap.frame_index, snap.elapsed_s, snap.fps, snap.inference_ms,
                             " · CNN {:.1f} ms".format(snap.cnn_ms) if snap.cnn_enabled else ""))
            else:
                st.markdown(badge("no video", "#3a3a3a", "1.4rem"), unsafe_allow_html=True)
                for line in worker.banner:
                    st.caption(line)

        with col_state:
            if snap is None:
                st.markdown(badge("—", "#3a3a3a"), unsafe_allow_html=True)
                return
            color = STATE_COLORS[snap.state] if snap.sufficient else "#6c757d"
            st.markdown(badge("{}  ·  {:.0f} s".format(snap.state, snap.state_since_s), color), unsafe_allow_html=True)
            st.caption("; ".join(snap.reasons) if snap.reasons else "no supporting evidence in the window")
            alert_color = LEVEL_COLORS.get(snap.alert_level, "#3a3a3a")
            st.markdown(badge("alert level {}{}".format(snap.alert_level,
                                                         " · " + snap.alert_message if snap.alert_message else ""),
                              alert_color, "1.05rem"), unsafe_allow_html=True)
            if snap.alert_dismissed:
                st.caption("laptop audio muted, {:.0f} s left · buzzer CLEAR".format(snap.alert_dismiss_left_s))
            elif snap.alert_level not in ("NONE", "OFF"):
                st.caption("beeps {} · voice {} this episode".format(snap.beeps, snap.voices))
            b1, b2 = st.columns(2)
            if b1.button("🔇 Mute audio (d)", width="stretch", disabled=not worker.running,
                         help="Silences the laptop audio and the buzzer for 30 s; the banner stays"):
                worker.dismiss()
            if b2.button("↺ Reset alert (r)", width="stretch", disabled=not worker.running,
                         help="Clears the alert and returns the state machine to ALERT with an empty window"):
                worker.reset()
            st.caption(("🟢 " if snap.esp32_connected else "⚪ ") + snap.esp32_text
                       + (" · word {}".format(snap.buzzer_word) if snap.buzzer_word else ""))
            st.caption("frame {} · {}".format(
                "VALID" if snap.valid else "INVALID: " + ", ".join(snap.invalid_reasons),
                "face found" if snap.face_found else "no face") + (
                " · {} faces in view".format(snap.faces_in_frame) if snap.faces_in_frame > 1 else ""))

        if snap is None:
            return
        m = st.columns(6)
        m[0].metric("EAR (mean)", fmt(snap.ear), help="Eye aspect ratio; ~0.3 open, < 0.2 closed (person-dependent)")
        m[1].metric("MAR", fmt(snap.mar), help="Mouth aspect ratio; yawn ≥ {}".format(settings["yawn_mar"]))
        m[2].metric("PERCLOS (60 s)", fmt(snap.perclos, "{:.1%}"),
                    help="Share of valid frames with closed eyes in the last 60 s")
        m[3].metric("Blink duration", fmt(snap.mean_blink_s * 1000 if not math.isnan(snap.mean_blink_s) else None,
                                          "{:.0f} ms"), delta="{:.1f} / min".format(snap.blink_rate_per_min),
                    delta_color="off", help="Mean of blinks completed in the window, and the blink rate")
        m[4].metric("Yawn rate", "{:.1f} / min".format(snap.yawn_rate_per_min), delta="{} in window".format(snap.yawn_count),
                    delta_color="off")
        m[5].metric("Nod count (60 s)", "{}".format(snap.nod_count))
        m = st.columns(6)
        m[0].metric("Head yaw", fmt(snap.yaw_deg, "{:+.0f}°"), help="+ = subject's left ({})".format(snap.pose_method))
        m[1].metric("Head pitch", fmt(snap.pitch_deg, "{:+.0f}°"), delta="roll {}".format(fmt(snap.roll_deg, "{:+.0f}°")),
                    delta_color="off")
        m[2].metric("FPS", "{:.1f}".format(snap.fps), delta="{:.1f} ms landmarks".format(snap.inference_ms), delta_color="off")
        m[3].metric("Invalid frames (60 s)", "{:.0%}".format(snap.invalid_rate_window),
                    delta="{:.0%} session".format(snap.invalid_rate_session), delta_color="off")
        m[4].metric("Eye closure now", "{:.1f} s".format(snap.closure_now_s),
                    delta="longest {:.1f} s".format(snap.longest_closure_s), delta_color="off")
        if snap.cnn_enabled:
            left = "{} {:.2f}".format(*snap.cnn_left) if snap.cnn_left else "–"
            right = "{} {:.2f}".format(*snap.cnn_right) if snap.cnn_right else "–"
            m[5].metric("Eye CNN L / R", "{} / {}".format(left.split()[0], right.split()[0]),
                        delta="P(closed) {}".format(fmt(snap.cnn_closed_prob, "{:.2f}")), delta_color="off")
        else:
            m[5].metric("Eye CNN", "off")


    live_view()

    with st.expander("Detector configuration in force (start-up banner)", expanded=False):
        for line in worker.banner or ["(appears once the detector has started)"]:
            st.text(line)


    # --- charts, event log, session summary -------------------------------------------------------------

    @st.fragment(run_every=SLOW_EVERY_S if worker.running else None)
    def analytics() -> None:
        tab_charts, tab_events, tab_summary = st.tabs(["📈 Live charts", "📋 Event log", "🧾 Session summary"])
        with tab_charts:
            rows = worker.history_rows(300)
            if rows:
                df = pd.DataFrame(rows, columns=["t (s)", "EAR", "MAR", "PERCLOS", "state", "closure (s)"])
                df["state level"] = df["state"].map(STATE_LEVEL)
                c1, c2 = st.columns(2)
                c1.caption("EAR and MAR (valid frames)")
                c1.line_chart(df, x="t (s)", y=["EAR", "MAR"], height=220)
                c2.caption("PERCLOS (share of closed frames in the 60 s window) and current closure (s)")
                c2.line_chart(df, x="t (s)", y=["PERCLOS", "closure (s)"], height=220)
                st.caption("State: 0 ALERT · 1 MILD · 2 DROWSY")
                st.line_chart(df, x="t (s)", y="state level", height=120)
            else:
                st.caption("Charts appear once the detector is running.")
        with tab_events:
            events = worker.events(300)
            st.caption("Transitions (Stage 9), alerts (Stage 10), driver actions, ESP32 link — newest first. "
                       "The same rows are in the SQLite `events` table.")
            st.dataframe(events_frame(events), width="stretch", height=320, hide_index=True)
        with tab_summary:
            summary = worker.summary()
            if summary:
                c1, c2 = st.columns([2, 3])
                c1.dataframe(summary_frame(summary), width="stretch", hide_index=True, height=560)
                with c2:
                    st.caption("State transitions")
                    tl = summary.get("transition_list") or []
                    st.dataframe(pd.DataFrame(tl, columns=["t (s)", "from", "to", "reason"]) if tl
                                 else pd.DataFrame(columns=["t (s)", "from", "to", "reason"]), width="stretch", hide_index=True)
                    st.caption("Alert events by type")
                    ac = summary.get("alert_counts") or {}
                    st.dataframe(pd.DataFrame(sorted(ac.items()), columns=["event", "count"]) if ac
                                 else pd.DataFrame(columns=["event", "count"]), width="stretch", hide_index=True)
                    ir = summary.get("invalid_reasons") or {}
                    if ir:
                        st.caption("Invalid-frame reasons")
                        st.dataframe(pd.DataFrame(sorted(ir.items(), key=lambda kv: -kv[1]), columns=["reason", "frames"]),
                                     width="stretch", hide_index=True)
            else:
                st.caption("The summary fills in while the detector runs and stays after Stop.")


    analytics()


    # --- past sessions from the SQLite log ---------------------------------------------------------------------

    st.divider()
    st.subheader("Past sessions")
    sessions = list_sessions(Path(db_path)) if db_path else []
    if not sessions:
        st.caption("No sessions in `{}` yet. Every run of the dashboard or of `python -m src.features` adds one.".format(db_path))
    else:
        labels = {"#{} · {} · {:.0f} s · {} frames · final {}".format(
            s["id"], (s.get("started_at") or "")[:19].replace("T", " "), s.get("duration_s") or 0.0, s.get("frames") or 0,
            (s.get("summary") or {}).get("final_state", "?")): s["id"] for s in sessions}
        choice = st.selectbox("Session", list(labels.keys()))
        sid = labels[choice]
        session = get_session(Path(db_path), sid) or {}
        events = read_events(Path(db_path), sid)
        metrics = read_metrics(Path(db_path), sid)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.caption("Summary written at the end of the session")
            summary = session.get("summary") or {}
            if summary:
                st.dataframe(summary_frame(summary), width="stretch", hide_index=True, height=520)
            else:
                st.caption("No summary: the session is still running or ended abnormally.")
            with st.expander("Thresholds in force (config JSON)"):
                st.json(session.get("config") or {})
        with c2:
            if metrics:
                mdf = pd.DataFrame(metrics)
                mdf["state level"] = mdf["state"].map(STATE_LEVEL)
                st.caption("Metrics table ({} rows, one every {:.1f} s)".format(
                    len(mdf), (mdf["t_s"].iloc[-1] / max(len(mdf) - 1, 1)) if len(mdf) > 1 else 0.0))
                st.line_chart(mdf, x="t_s", y=["ear", "perclos"], height=200)
                st.line_chart(mdf, x="t_s", y="state level", height=110)
            st.caption("Events ({})".format(len(events)))
            st.dataframe(events_frame(events, from_db=True), width="stretch", height=260, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇ events CSV", csv_bytes(events), "session{}_events.csv".format(sid), "text/csv",
                               width="stretch", disabled=not events)
            d2.download_button("⬇ metrics CSV", csv_bytes(metrics), "session{}_metrics.csv".format(sid), "text/csv",
                               width="stretch", disabled=not metrics)


if __name__ == "__main__":       # Streamlit runs the page as __main__; a spawned child imports it as __mp_main__
    main()
