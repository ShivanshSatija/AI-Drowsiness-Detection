"""Stage 13 camera check: identify, measure, refocus and test the NoIR / IR camera.

Four jobs, one per sub-command. Nothing here modifies the detection code; the
measurement runs the real pipeline (src.pipeline.DrowsinessPipeline) exactly
as the live tools do, with alerts, buzzer and session log switched off.

    identify   Which cameras exist (Windows device names + OpenCV indices), what
               each one delivers (resolution, FPS, pixel format, which exposure /
               gain / white-balance controls it exposes) and a first frame's
               colour balance. Answers "is OpenCV seeing the USB webcam, and
               which index is it?"

    measure    Records a labelled JSON under evaluation/results/stage13/ with the
               capture path (FPS, latency, drops), the image (brightness, B/G/R
               balance, focus score, saturation, 3x3 evenness map) AND the
               detector's performance on this camera (face rate, valid-frame
               rate, landmark inference time, EAR / eye-size / pose statistics,
               invalid-frame reasons) plus a bright-spot measurement for the
               TV-remote IR test. Run it at every step of the conversion with
               a different --label and compare afterwards.

    live       A preview window with the numbers that matter while you work:
               focus score with the best-so-far marker (for refocusing after
               reassembly), saturation, 3x3 brightness map and uniformity ratio
               (illumination evenness), B/G/R means (IR leakage), and a bright-
               spot detector (the TV-remote test). Keys: q quit, s snapshot,
               g grey display, f reset best focus, r write a JSON record.

    compare    Side-by-side table of two measure JSONs with deltas.

Usage::

    python evaluation/stage13_camera_check.py identify
    python evaluation/stage13_camera_check.py measure --device 1 --label usb-unmodified-room --notes "desk lamp, 60 cm"
    python evaluation/stage13_camera_check.py live --device 1
    python evaluation/stage13_camera_check.py compare usb-unmodified-room noir-daylight
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from camera_baseline import frame_statistics  # noqa: E402  (same folder: the Stage 1 image statistics)
from src.capture import CameraConfig, CameraError, WebcamSource, save_snapshot  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results" / "stage13"

# OpenCV property ids worth knowing for IR work: a camera whose auto-exposure
# cannot be switched off will fight the IR illuminator at night.
CAMERA_PROPS = {
    "fps": cv2.CAP_PROP_FPS, "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "exposure": cv2.CAP_PROP_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN, "brightness": cv2.CAP_PROP_BRIGHTNESS, "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION, "sharpness": cv2.CAP_PROP_SHARPNESS, "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE, "backlight": cv2.CAP_PROP_BACKLIGHT,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS, "focus": cv2.CAP_PROP_FOCUS, "zoom": cv2.CAP_PROP_ZOOM,
}
BRIGHT_SPOT_THRESHOLD = 240          # grey level counted as "saturated" for the IR-remote spot test
BRIGHT_SPOT_MIN_AREA = 12            # pixels; smaller blobs are sensor noise / specular glints


# --- helpers ------------------------------------------------------------------------

def fourcc_to_str(value: float) -> str:
    v = int(value)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip() if v > 0 else "?"


def read_props(cap: cv2.VideoCapture) -> Dict[str, Any]:
    props = {}
    for name, pid in CAMERA_PROPS.items():
        try:
            props[name] = round(float(cap.get(pid)), 3)
        except Exception:
            props[name] = None
    props["fourcc"] = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    return props


def pnp_cameras() -> List[Dict[str, Any]]:
    """Windows device names for the cameras present (PowerShell PnP). Empty on
    other platforms or if PowerShell is unavailable."""
    if platform.system() != "Windows":
        return []
    cmd = ("Get-PnpDevice -Class Camera,Image -PresentOnly -ErrorAction SilentlyContinue | "
           "Select-Object FriendlyName, Status, Class, InstanceId | ConvertTo-Json")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return []
    if not out:
        return []
    data = json.loads(out)
    if isinstance(data, dict):
        data = [data]
    cams = []
    for d in data:
        inst = d.get("InstanceId", "") or ""
        vid = pid = None
        if "VID_" in inst and "PID_" in inst:
            try:
                vid = inst.split("VID_")[1][:4]
                pid = inst.split("PID_")[1][:4]
            except IndexError:
                pass
        name = d.get("FriendlyName", "") or ""
        cams.append({"name": name, "status": d.get("Status"), "class": d.get("Class"), "instance_id": inst,
                     "vid": vid, "pid": pid, "usb": inst.upper().startswith("USB"),
                     "looks_integrated": any(k in name.lower() for k in ("integrated", "built-in", "internal"))})
    return cams


def bright_spot(gray: np.ndarray, threshold: int = BRIGHT_SPOT_THRESHOLD) -> Dict[str, Any]:
    """Largest saturated blob: the TV-remote / IR-LED test. Before the filter is
    removed the remote's LED is dim or invisible; after removal it is a bright
    white/pink spot. Lamps and windows saturate too - compare remote off vs on."""
    mask = (gray >= threshold).astype(np.uint8)
    frac = float(mask.mean())
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_area, best_xy = 0, None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area, best_xy = area, (int(centroids[i][0]), int(centroids[i][1]))
    return {"saturated_fraction": round(frac, 5), "largest_blob_px": best_area if best_area >= BRIGHT_SPOT_MIN_AREA else 0,
            "largest_blob_xy": best_xy if best_area >= BRIGHT_SPOT_MIN_AREA else None, "threshold": threshold}


def channel_spread(stats: Dict[str, Any]) -> float:
    c = stats["channel_means"]
    return round(max(c.values()) - min(c.values()), 2)


def percentiles(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {"median": round(float(np.median(arr)), 4), "p5": round(float(np.percentile(arr, 5)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4), "std": round(float(arr.std()), 4), "n": int(arr.size)}


def result_path(label: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    return RESULTS_DIR / "stage13_{}_{:%Y%m%d_%H%M%S}.json".format(safe, datetime.now())


# --- identify -----------------------------------------------------------------------------

def probe_index(index: int, width: int, height: int, read_frames: int = 6) -> Optional[Dict[str, Any]]:
    from src.capture import _BACKEND_NAMES, _BACKEND_ORDER  # private, but this is the project's own module
    for backend in _BACKEND_ORDER:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        t0 = time.perf_counter()
        for _ in range(read_frames):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        dt = (time.perf_counter() - t0) / max(read_frames, 1)
        info = {"index": index, "backend": _BACKEND_NAMES.get(backend, str(backend)),
                "resolution": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
                "props": read_props(cap), "delivers_frames": frame is not None,
                "read_ms_avg": round(dt * 1000.0, 1)}
        if frame is not None:
            st = frame_statistics(frame)
            info["first_frame"] = {"mean_brightness": st["mean_brightness"], "channel_means": st["channel_means"],
                                   "channel_spread": channel_spread(st), "focus_laplacian_var": st["focus_laplacian_var"]}
        cap.release()
        return info
    return None


def cmd_identify(args: argparse.Namespace) -> int:
    print("Stage 13 camera identification  ({})".format(datetime.now().isoformat(timespec="seconds")))
    print("\n[1] Cameras Windows knows about (PnP, present only)")
    cams = pnp_cameras()
    if not cams:
        print("  (none listed, or not Windows / PowerShell unavailable)")
    for c in cams:
        print("  - {:<40} status {:<8} VID:PID {}:{}  {}{}".format(
            c["name"][:40], c["status"], c["vid"], c["pid"], "USB " if c["usb"] else "",
            "<- looks like the laptop's built-in camera" if c["looks_integrated"] else ""))
    print("\n[2] OpenCV indices 0..{} (each is opened, {}x{} requested, a few frames read)".format(
        args.max_index, args.width, args.height))
    found = []
    for index in range(args.max_index + 1):
        info = probe_index(index, args.width, args.height)
        if info is None:
            print("  index {}: nothing".format(index))
            continue
        found.append(info)
        p = info["props"]
        print("  index {}: backend {} | {}x{} | fourcc {} | driver fps {} | frames {} ({} ms/read)".format(
            index, info["backend"], info["resolution"][0], info["resolution"][1], p["fourcc"], p["fps"],
            "yes" if info["delivers_frames"] else "NO", info["read_ms_avg"]))
        print("           controls: auto_exposure {} | exposure {} | gain {} | auto_wb {} | wb_temp {} | "
              "autofocus {} | focus {} | brightness {} | backlight {}".format(
                  p["auto_exposure"], p["exposure"], p["gain"], p["auto_wb"], p["wb_temperature"], p["autofocus"],
                  p["focus"], p["brightness"], p["backlight"]))
        if "first_frame" in info:
            ff = info["first_frame"]
            print("           first frame: brightness {} | B {blue} G {green} R {red} (spread {}) | focus {}".format(
                ff["mean_brightness"], ff["channel_spread"], ff["focus_laplacian_var"], **ff["channel_means"]))
    print("\n[3] Reading the result")
    print("  - Each PnP camera normally maps to one OpenCV index. With the USB webcam plugged in you should see")
    print("    TWO PnP entries and TWO indices. Unplug it and run again: the index that disappears is the USB webcam.")
    print("  - 'controls' shows what the driver exposes. exposure/gain/auto_exposure values of -1 mean 'not")
    print("    reported' - later, in darkness with the IR illuminator, we may need manual exposure; note it now.")
    print("  - 'spread' is the gap between the B/G/R channel means. A colour camera with its IR-cut filter in")
    print("    place shows a clear spread on a normal scene; after the filter comes out the channels converge.")
    out = {"timestamp": datetime.now().isoformat(timespec="seconds"), "pnp_cameras": cams, "opencv_indices": found,
           "environment": {"platform": platform.platform(), "opencv": cv2.__version__}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "stage13_identify_{:%Y%m%d_%H%M%S}.json".format(datetime.now())
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved: {}".format(path.relative_to(PROJECT_ROOT)))
    return 0 if found else 1


# --- measure --------------------------------------------------------------------------------

def cmd_measure(args: argparse.Namespace) -> int:
    from src.eye_cnn import EyePreprocessConfig
    from src.features import ValidityConfig
    from src.landmarks import FaceLandmarkDetector, LandmarkConfig
    from src.pipeline import DrowsinessPipeline, load_classifier
    from src.temporal import TemporalConfig

    print("Stage 13 measurement  label={!r}  device={}".format(args.label, args.device))
    if args.notes:
        print("notes: {}".format(args.notes))
    device = int(args.device) if str(args.device).isdigit() else args.device
    source = WebcamSource(CameraConfig(device=device, width=args.width, height=args.height, flip_horizontal=False))
    t0 = time.perf_counter()
    try:
        source.open()
    except CameraError as exc:
        print("ERROR: {}".format(exc))
        return 1
    open_s = time.perf_counter() - t0
    props = read_props(source._cap)  # noqa: SLF001 - the project's own capture layer
    print("opened {} in {:.2f} s | fourcc {} | driver fps {}".format(source.description, open_s, props["fourcc"],
                                                                     props["fps"]))
    print("settling auto-exposure for {:.0f} s ...".format(args.settle))
    t_end = time.time() + args.settle
    while time.time() < t_end:
        source.read()

    eye = EyePreprocessConfig(min_eye_width_px=args.min_eye_px)
    classifier, cnn_status = (None, "disabled (--no-cnn)") if args.no_cnn else load_classifier({"model": str(args.model)}, eye)
    detector = FaceLandmarkDetector(LandmarkConfig())
    pipeline = DrowsinessPipeline(detector, classifier=classifier, eye_config=eye,
                                  validity_config=ValidityConfig(min_face_width_px=args.min_face_px,
                                                                 min_eye_width_px=args.min_eye_px),
                                  temporal_config=TemporalConfig(), alerts=False, serial_port=None, session_log=None,
                                  cnn_status=cnn_status, echo=False)
    print("CNN: {}".format("loaded" if classifier is not None else cnn_status))
    print("capturing {} frames through the full detection pipeline ...".format(args.frames))

    latencies, dropped, brightness_series = [], 0, []
    ears, eye_widths, yaws, pitches, inference, cnn_probs = [], [], [], [], [], []
    stats_frames: List[Dict[str, Any]] = []
    spots: List[Dict[str, Any]] = []
    last = None
    with detector:
        t_start = time.perf_counter()
        for i in range(args.frames):
            t = time.perf_counter()
            frame = source.read()
            latencies.append((time.perf_counter() - t) * 1000.0)
            if frame is None:
                dropped += 1
                continue
            last = frame
            r = pipeline.process(frame)
            inference.append(r.inference_ms)
            gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            brightness_series.append(gray_mean)
            if r.feats is not None and r.assessment.valid:
                ears.append(r.feats.ear_mean)
            for crop in r.crops:
                if crop is not None:
                    eye_widths.append(crop.eye_width_px)
            if r.pose is not None and r.pose.ok:
                yaws.append(r.pose.yaw_deg)
                pitches.append(r.pose.pitch_deg)
            if r.observation.cnn_closed_prob is not None:
                cnn_probs.append(r.observation.cnn_closed_prob)
            if i % max(1, args.frames // 10) == 0:            # image statistics on ~10 frames spread over the run
                stats_frames.append(frame_statistics(frame))
                spots.append(bright_spot(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        elapsed = time.perf_counter() - t_start
    summary = pipeline.close()
    source.release()

    captured = args.frames - dropped
    if captured == 0 or last is None:
        print("ERROR: no frames captured")
        return 1
    lat = sorted(latencies)
    pct = lambda p: round(lat[min(len(lat) - 1, int(len(lat) * p))], 2)  # noqa: E731

    def avg(key, sub=None):
        vals = [(s[key][sub] if sub else s[key]) for s in stats_frames]
        return round(float(np.mean(vals)), 3)

    image = {
        "mean_brightness": avg("mean_brightness"), "std_brightness": avg("std_brightness"),
        "channel_means": {c: avg("channel_means", c) for c in ("blue", "green", "red")},
        "focus_laplacian_var": avg("focus_laplacian_var"), "saturated_fraction": avg("saturated_fraction"),
        "dark_fraction": avg("dark_fraction"), "grid_uniformity_ratio": avg("grid_uniformity_ratio"),
        "grid_brightness_last": stats_frames[-1]["grid_brightness"],
        "brightness_over_time_std": round(float(np.std(brightness_series)), 3),
    }
    image["channel_spread"] = channel_spread(image)
    spot = max(spots, key=lambda s: s["largest_blob_px"])
    detection = {
        "frames": captured, "fps_end_to_end": round(captured / elapsed, 2) if elapsed else 0.0,
        "inference_ms": percentiles(inference), "face_rate": summary.get("face_rate"),
        "valid_rate": round(1.0 - summary.get("invalid_session_rate", 0.0), 4),
        "invalid_reasons": summary.get("invalid_reasons", {}),
        "ear_valid_frames": percentiles(ears), "eye_width_px": percentiles(eye_widths),
        "yaw_deg": percentiles(yaws), "pitch_deg": percentiles(pitches),
        "cnn_closed_prob": percentiles(cnn_probs) if cnn_probs else None, "cnn": "loaded" if classifier else cnn_status,
    }
    out = {
        "label": args.label, "notes": args.notes, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": {"index": args.device, "description": source.description, "open_s": round(open_s, 3), "props": props,
                   "requested": [args.width, args.height]},
        "capture": {"measured_fps": round(captured / elapsed, 2), "dropped": dropped,
                    "drop_rate": round(dropped / args.frames, 4),
                    "latency_ms": {"median": pct(0.5), "p95": pct(0.95), "max": round(lat[-1], 2)}},
        "image": image, "bright_spot": spot, "detection": detection,
        "environment": {"platform": platform.platform(), "opencv": cv2.__version__, "python": platform.python_version()},
    }
    print("\n--- capture ---   {:.1f} FPS | dropped {} | read latency median {} ms p95 {} ms".format(
        out["capture"]["measured_fps"], dropped, pct(0.5), pct(0.95)))
    print("--- image -----   brightness {} (std {}) | B {blue} G {green} R {red} spread {} | focus {} | saturated {:.2%} | "
          "dark {:.2%} | uniformity {}".format(image["mean_brightness"], image["std_brightness"], image["channel_spread"],
                                              image["focus_laplacian_var"], image["saturated_fraction"],
                                              image["dark_fraction"], image["grid_uniformity_ratio"],
                                              **image["channel_means"]))
    print("                 3x3 map (last frame): " + " / ".join(
        " ".join("{:5.1f}".format(v) for v in row) for row in image["grid_brightness_last"]))
    print("--- IR spot ---   saturated {:.2%} | largest bright blob {} px{}".format(
        spot["saturated_fraction"], spot["largest_blob_px"],
        " at {}".format(spot["largest_blob_xy"]) if spot["largest_blob_xy"] else ""))
    d = detection
    print("--- detector --   {:.1f} FPS end to end | landmarks median {} ms | face in {:.0%} of frames | valid {:.0%} | "
          "invalid reasons {}".format(d["fps_end_to_end"], d["inference_ms"]["median"] if d["inference_ms"] else "?",
                                      d["face_rate"] or 0.0, d["valid_rate"], d["invalid_reasons"] or "none"))
    if d["ear_valid_frames"]:
        print("                 EAR (valid frames) median {median} p5 {p5} p95 {p95} std {std} n {n}".format(**d["ear_valid_frames"]))
    if d["eye_width_px"]:
        print("                 eye width px median {median} (p5 {p5}, p95 {p95})".format(**d["eye_width_px"]))
    if d["yaw_deg"]:
        print("                 yaw median {:+.1f} | pitch median {:+.1f}".format(d["yaw_deg"]["median"], d["pitch_deg"]["median"]))
    if d["cnn_closed_prob"]:
        print("                 CNN P(closed) median {median} (p5 {p5}, p95 {p95})".format(**d["cnn_closed_prob"]))
    if args.save_frame:
        path = save_snapshot(last)
        out["sample_frame"] = str(path.relative_to(PROJECT_ROOT))
        print("sample frame saved: {} (git-ignored)".format(path))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(args.label)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved: {}".format(path.relative_to(PROJECT_ROOT)))
    return 0


# --- compare -----------------------------------------------------------------------------------

def resolve(label_or_path: str) -> Path:
    p = Path(label_or_path)
    if p.exists():
        return p
    matches = sorted(RESULTS_DIR.glob("stage13_{}_*.json".format(label_or_path)))
    if not matches:
        raise SystemExit("no result for {!r} (looked in {})".format(label_or_path, RESULTS_DIR))
    return matches[-1]                                    # newest run of that label


def dig(d: Dict[str, Any], keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


COMPARE_ROWS = [
    ("capture FPS", ("capture", "measured_fps")), ("drop rate", ("capture", "drop_rate")),
    ("latency median ms", ("capture", "latency_ms", "median")),
    ("mean brightness", ("image", "mean_brightness")), ("std brightness", ("image", "std_brightness")),
    ("blue mean", ("image", "channel_means", "blue")), ("green mean", ("image", "channel_means", "green")),
    ("red mean", ("image", "channel_means", "red")), ("channel spread", ("image", "channel_spread")),
    ("focus (Laplacian)", ("image", "focus_laplacian_var")), ("saturated fraction", ("image", "saturated_fraction")),
    ("dark fraction", ("image", "dark_fraction")), ("uniformity ratio", ("image", "grid_uniformity_ratio")),
    ("largest bright blob px", ("bright_spot", "largest_blob_px")),
    ("detector FPS", ("detection", "fps_end_to_end")), ("landmarks ms median", ("detection", "inference_ms", "median")),
    ("face rate", ("detection", "face_rate")), ("valid rate", ("detection", "valid_rate")),
    ("EAR median", ("detection", "ear_valid_frames", "median")), ("EAR std", ("detection", "ear_valid_frames", "std")),
    ("eye width px median", ("detection", "eye_width_px", "median")),
    ("CNN P(closed) median", ("detection", "cnn_closed_prob", "median")),
]


def cmd_compare(args: argparse.Namespace) -> int:
    pa, pb = resolve(args.a), resolve(args.b)
    a, b = json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8"))
    print("A: {:<28} {}   {}".format(a["label"], a["timestamp"], a.get("notes", "")))
    print("B: {:<28} {}   {}".format(b["label"], b["timestamp"], b.get("notes", "")))
    print("\n{:<24} {:>12} {:>12} {:>12}".format("metric", "A", "B", "B - A"))
    print("-" * 64)
    for title, keys in COMPARE_ROWS:
        va, vb = dig(a, keys), dig(b, keys)
        if va is None and vb is None:
            continue
        delta = "{:+.3f}".format(vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else ""
        print("{:<24} {:>12} {:>12} {:>12}".format(title, "-" if va is None else va, "-" if vb is None else vb, delta))
    print("-" * 64)
    print("What to look for is written in hardware/noir_conversion.md section 8.")
    return 0


# --- live helper ------------------------------------------------------------------------------------

def cmd_live(args: argparse.Namespace) -> int:
    device = int(args.device) if str(args.device).isdigit() else args.device
    source = WebcamSource(CameraConfig(device=device, width=args.width, height=args.height, flip_horizontal=False))
    try:
        source.open()
    except CameraError as exc:
        print("ERROR: {}".format(exc))
        return 1
    print("live helper on {} | q quit | s snapshot | g grey | f reset best focus | r record JSON ({})".format(
        source.description, args.label))
    window = "Stage 13 - camera helper"
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    best_focus, best_focus_centre = 0.0, 0.0
    grey_display = False
    history: List[Dict[str, Any]] = []
    frames = 0
    t_last_print = 0.0
    while True:
        frame = source.read()
        if frame is None:
            print("no frame - camera lost")
            break
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        st = frame_statistics(frame)
        h, w = gray.shape
        centre = gray[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        focus_centre = float(cv2.Laplacian(centre, cv2.CV_64F).var())
        best_focus = max(best_focus, st["focus_laplacian_var"])
        best_focus_centre = max(best_focus_centre, focus_centre)
        spot = bright_spot(gray)
        record = {"t": time.time(), "mean_brightness": st["mean_brightness"], "channel_means": st["channel_means"],
                  "channel_spread": channel_spread(st), "focus": st["focus_laplacian_var"], "focus_centre": round(focus_centre, 1),
                  "saturated_fraction": st["saturated_fraction"], "uniformity": st["grid_uniformity_ratio"],
                  "grid": st["grid_brightness"], "blob_px": spot["largest_blob_px"], "blob_xy": spot["largest_blob_xy"]}
        history.append(record)
        if len(history) > 60:
            history.pop(0)

        if args.headless:
            if time.time() - t_last_print >= 1.0:
                t_last_print = time.time()
                print("brightness {:6.1f} | spread {:5.1f} | focus {:7.1f} (centre {:7.1f}, best {:7.1f}) | saturated {:.2%} | "
                      "uniformity {:.3f} | blob {} px".format(st["mean_brightness"], record["channel_spread"], st["focus_laplacian_var"],
                                                              focus_centre, best_focus_centre, st["saturated_fraction"],
                                                              st["grid_uniformity_ratio"], spot["largest_blob_px"]))
            if args.max_frames and frames >= args.max_frames:
                break
            continue

        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if grey_display else frame.copy()
        # centre box used for the focus score, 3x3 grid lines and per-cell brightness
        cv2.rectangle(display, (w // 3, h // 3), (2 * w // 3, 2 * h // 3), (0, 255, 255), 1)
        for k in (1, 2):
            cv2.line(display, (k * w // 3, 0), (k * w // 3, h), (90, 90, 90), 1)
            cv2.line(display, (0, k * h // 3), (w, k * h // 3), (90, 90, 90), 1)
        for row in range(3):
            for col in range(3):
                cv2.putText(display, "{:.0f}".format(st["grid_brightness"][row][col]),
                            (col * w // 3 + 6, row * h // 3 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, "{:.0f}".format(st["grid_brightness"][row][col]),
                            (col * w // 3 + 6, row * h // 3 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if spot["largest_blob_xy"]:
            cv2.circle(display, spot["largest_blob_xy"], 18, (255, 0, 255), 2)
        focus_color = (0, 255, 0) if focus_centre >= 0.95 * best_focus_centre else (0, 200, 255)
        lines = [
            ("FOCUS centre {:6.0f}   best {:6.0f}   (turn the lens until this peaks; f resets best)".format(
                focus_centre, best_focus_centre), focus_color, 0.6),
            ("brightness {:5.1f}   saturated {:5.2%}   dark {:5.2%}   uniformity {:.3f}".format(
                st["mean_brightness"], st["saturated_fraction"], st["dark_fraction"], st["grid_uniformity_ratio"]),
             (255, 255, 255), 0.5),
            ("B {blue:5.1f}  G {green:5.1f}  R {red:5.1f}   spread {spread:5.1f}   (IR leak -> channels converge)".format(
                spread=record["channel_spread"], **st["channel_means"]), (255, 255, 255), 0.5),
            ("IR spot: {}".format("YES  {} px at {}".format(spot["largest_blob_px"], spot["largest_blob_xy"])
                                  if spot["largest_blob_px"] else "none (>= {} grey)".format(BRIGHT_SPOT_THRESHOLD)),
             (255, 0, 255) if spot["largest_blob_px"] else (160, 160, 160), 0.5),
            ("q quit | s snapshot | g grey | f reset focus | r record '{}'".format(args.label), (160, 160, 160), 0.45),
        ]
        y = h - 12 - 22 * (len(lines) - 1)
        for text, color, scale in lines:
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 22
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("g"):
            grey_display = not grey_display
        if key == ord("f"):
            best_focus = best_focus_centre = 0.0
        if key == ord("s"):
            print("snapshot: {}".format(save_snapshot(display)))
        if key == ord("r"):
            recent = history[-30:]
            out = {"label": args.label, "notes": args.notes, "timestamp": datetime.now().isoformat(timespec="seconds"),
                   "device": {"index": args.device, "description": source.description},
                   "averaged_over_frames": len(recent),
                   "image": {k: round(float(np.mean([r[k] for r in recent])), 3)
                             for k in ("mean_brightness", "channel_spread", "focus", "focus_centre", "saturated_fraction", "uniformity")},
                   "channel_means": {c: round(float(np.mean([r["channel_means"][c] for r in recent])), 2) for c in ("blue", "green", "red")},
                   "grid_last": recent[-1]["grid"], "best_focus_centre": round(best_focus_centre, 1),
                   "bright_spot_px_max": max(r["blob_px"] for r in recent)}
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            path = result_path("live-" + args.label)
            path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print("recorded {}".format(path.relative_to(PROJECT_ROOT)))
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
    source.release()
    if not args.headless:
        cv2.destroyAllWindows()
    return 0


# --- CLI ---------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 13: identify, measure, refocus and test the NoIR/IR camera.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("identify", help="List cameras (Windows names + OpenCV indices) and their controls")
    s.add_argument("--max-index", type=int, default=3)
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)

    s = sub.add_parser("measure", help="Record capture + image + detector performance to a labelled JSON")
    s.add_argument("--device", default="0", help="Camera index (see identify)")
    s.add_argument("--label", required=True, help="e.g. usb-unmodified-room, noir-daylight, noir-dark-ir-on")
    s.add_argument("--notes", default="", help="lighting, distance, IR on/off, remote on/off - anything needed to repeat it")
    s.add_argument("--frames", type=int, default=300)
    s.add_argument("--settle", type=float, default=3.0, help="Seconds to let auto-exposure settle before measuring")
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--min-face-px", type=int, default=80)
    s.add_argument("--min-eye-px", type=float, default=15.0)
    s.add_argument("--no-cnn", action="store_true")
    s.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "eye_cnn.pt")
    s.add_argument("--save-frame", action="store_true", help="Keep one frame in data/snapshots/ (git-ignored)")

    s = sub.add_parser("live", help="Preview with focus / saturation / evenness / IR-spot readouts")
    s.add_argument("--device", default="0")
    s.add_argument("--label", default="live", help="Label used by the r key")
    s.add_argument("--notes", default="")
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)
    s.add_argument("--headless", action="store_true", help="No window: print the readouts once a second (testing)")
    s.add_argument("--max-frames", type=int, default=0, help="With --headless: stop after N frames")

    s = sub.add_parser("compare", help="Two measure JSONs side by side (labels or paths)")
    s.add_argument("a")
    s.add_argument("b")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"identify": cmd_identify, "measure": cmd_measure, "live": cmd_live, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
