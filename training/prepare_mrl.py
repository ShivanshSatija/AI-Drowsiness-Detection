"""Prepare the MRL Eye Dataset for eye-state CNN training (Stage 6).

Written against the archive's real structure, verified with
training/inspect_mrl.py on 2026-09-13:

    mrlEyes_2018_01.zip
      mrlEyes_2018_01/annotation.txt          field codings (reproduced below)
      mrlEyes_2018_01/stats_2018_01.ods
      mrlEyes_2018_01/s0001 ... s0037/        one folder per subject
          sXXXX_YYYYY_G_GL_ST_RF_LT_SN.png    84,898 8-bit grayscale PNGs, 56-278 px square

Filename fields, as documented in the archive's annotation.txt:

    sXXXX  subject id            YYYYY  image number
    G      gender      0 male, 1 female
    GL     glasses     0 no, 1 yes
    ST     eye state   0 closed, 1 open
    RF     reflections 0 none, 1 low, 2 high
    LT     lighting / image quality  0 bad, 1 good
    SN     sensor      01 RealSense SR300 640x480, 02 IDS 1280x1024, 03 Aptina 752x480

What this script does
---------------------
1. Parses every filename into a manifest (path + all fields).
2. Builds a SUBJECT-INDEPENDENT split: whole subjects go to train, validation
   or test, never images. Because subjects differ enormously in size (382 to
   10,257 images) and class balance (s0004: 1,069 closed / 0 open; s0028:
   13 / 723), a plain shuffle can hand a split a near single-class population.
   Instead, a seeded search over many random subject partitions keeps the one
   with the lowest cost, where the cost measures deviation from the image
   fractions (70 / 15 / 15), from the global closed ratio and glasses ratio in
   validation and test, and penalises a validation or test split without a
   female subject, without the second sensor, or with fewer than a minimum
   number of images of either class. Every term is printed for the chosen
   split, the subject lists are written to training/splits/ and committed, and
   the disjointness of the three subject sets is asserted before anything is
   saved.
3. Preprocesses every image with ``src.eye_cnn.preprocess_eye_image`` - the
   same function the live system uses - and packs each split into an .npz
   (uint8 size x size images, labels, subjects and the other fields) for fast
   loading in Colab. Standardisation happens at load time with the same
   ``normalize_eye``.
4. Re-opens the .npz files and verifies shapes, dtypes, label values and
   subject disjointness, then prints a per-split summary.

Usage::

    python training/prepare_mrl.py --source data/mrl/mrlEyes_2018_01.zip --out data/mrl_prepared
    python training/prepare_mrl.py --source data/mrl/mrlEyes_2018_01.zip --out data/mrl_prepared --preview

Nothing here is fabricated; the split is reproducible from --seed and the
committed subject lists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eye_cnn import EyePreprocessConfig, preprocess_eye_image  # noqa: E402

FILENAME = re.compile(r"^s(\d{4})_(\d{5})_([01])_([01])_([01])_([012])_([01])_(\d{2})$")
SPLITS = ("train", "val", "test")
SPLITS_DIR = PROJECT_ROOT / "training" / "splits"


@dataclass
class Record:
    path: str
    subject: int
    image_number: int
    gender: int
    glasses: int
    eye_state: int
    reflections: int
    lighting: int
    sensor: int


# --- enumeration -------------------------------------------------------------

class Source:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.zip: Optional[zipfile.ZipFile] = zipfile.ZipFile(path) if path.is_file() else None
        if self.zip is not None:
            self.names = [n for n in self.zip.namelist() if n.lower().endswith(".png")]
        else:
            self.names = [str(p.relative_to(path)).replace("\\", "/") for p in path.rglob("*.png")]

    def read_gray(self, name: str) -> Optional[np.ndarray]:
        data = self.zip.read(name) if self.zip is not None else (self.path / name).read_bytes()
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    def sha256(self) -> str:
        if self.zip is None:
            return ""
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()


def parse_records(names: List[str]) -> Tuple[List[Record], int]:
    records, skipped = [], 0
    for name in names:
        m = FILENAME.match(Path(name).stem)
        if not m:
            skipped += 1
            continue
        g = m.groups()
        records.append(Record(name, int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]),
                              int(g[5]), int(g[6]), int(g[7])))
    return records, skipped


# --- split search -------------------------------------------------------------

def subject_table(records: List[Record]) -> Dict[int, Dict[str, int]]:
    table: Dict[int, Counter] = defaultdict(Counter)
    for r in records:
        c = table[r.subject]
        c["n"] += 1
        c["closed"] += r.eye_state == 0
        c["open"] += r.eye_state == 1
        c["glasses"] += r.glasses == 1
        c["female"] += r.gender == 1
        c["sensor2"] += r.sensor == 2
        c["sensor3"] += r.sensor == 3
        c["good_light"] += r.lighting == 1
        c["reflections"] += r.reflections > 0
    return {s: dict(c) for s, c in table.items()}


def split_stats(assign: Dict[int, str], table: Dict[int, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    total = sum(t["n"] for t in table.values())
    for split in SPLITS:
        subjects = [s for s, sp in assign.items() if sp == split]
        agg = Counter()
        for s in subjects:
            agg.update(table[s])
        n = agg["n"]
        stats[split] = {
            "subjects": len(subjects), "images": n, "fraction": n / total if total else 0.0,
            "closed": agg["closed"], "open": agg["open"],
            "closed_ratio": agg["closed"] / n if n else 0.0,
            "glasses": agg["glasses"], "glasses_ratio": agg["glasses"] / n if n else 0.0,
            "female_subjects": sum(1 for s in subjects if table[s]["female"] > 0),
            "sensor2_images": agg["sensor2"], "sensor3_images": agg["sensor3"],
            "good_light_ratio": agg["good_light"] / n if n else 0.0,
            "reflection_ratio": agg["reflections"] / n if n else 0.0,
        }
    return stats


def split_cost(stats: Dict[str, Dict[str, float]], targets: Dict[str, float], global_closed: float,
               global_glasses: float, min_class_images: int) -> Tuple[float, Dict[str, float]]:
    """Lower is better. Each term is reported so the chosen split is auditable."""
    terms: Dict[str, float] = {}
    terms["fraction"] = sum(abs(stats[s]["fraction"] - targets[s]) for s in SPLITS) * 2.0
    terms["closed_ratio"] = sum(abs(stats[s]["closed_ratio"] - global_closed) for s in ("val", "test")) * 3.0
    terms["glasses_ratio"] = sum(abs(stats[s]["glasses_ratio"] - global_glasses) for s in ("val", "test")) * 2.0
    terms["female_missing"] = sum(0.2 for s in ("val", "test") if stats[s]["female_subjects"] == 0)
    terms["sensor2_missing"] = sum(0.1 for s in ("val", "test") if stats[s]["sensor2_images"] == 0)
    terms["min_class"] = sum(1.0 for s in ("val", "test")
                             if min(stats[s]["closed"], stats[s]["open"]) < min_class_images)
    return sum(terms.values()), terms


def search_split(table: Dict[int, Dict[str, int]], seed: int, trials: int, val_frac: float, test_frac: float,
                 min_class_images: int) -> Tuple[Dict[int, str], Dict[str, Dict[str, float]], Dict[str, float], int]:
    subjects = sorted(table)
    total = sum(t["n"] for t in table.values())
    global_closed = sum(t["closed"] for t in table.values()) / total
    global_glasses = sum(t["glasses"] for t in table.values()) / total
    targets = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}

    best: Tuple[float, Dict[int, str], Dict, Dict, int] = (float("inf"), {}, {}, {}, -1)
    for trial in range(trials):
        rng = np.random.default_rng([seed, trial])
        order = list(rng.permutation(subjects))
        # Greedy fill: each subject goes to the split furthest below its image target,
        # with a little randomness so trials differ in more than the order.
        counts = {s: 0 for s in SPLITS}
        assign: Dict[int, str] = {}
        for subject in order:
            deficits = {s: targets[s] - counts[s] / total for s in SPLITS}
            noisy = {s: d + rng.normal(0.0, 0.02) for s, d in deficits.items()}
            chosen = max(noisy, key=noisy.get)
            assign[int(subject)] = chosen
            counts[chosen] += table[int(subject)]["n"]
        stats = split_stats(assign, table)
        cost, terms = split_cost(stats, targets, global_closed, global_glasses, min_class_images)
        if cost < best[0]:
            best = (cost, assign, stats, terms, trial)
    _, assign, stats, terms, trial = best
    return assign, stats, terms, trial


# --- main ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the subject-independent MRL split and .npz packs.")
    parser.add_argument("--source", type=Path, required=True, help="mrlEyes_2018_01.zip or the extracted folder")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "mrl_prepared")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                        help="Where the committed subject lists and split_stats.json go")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=20000, help="Random subject partitions to evaluate")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--min-class-images", type=int, default=2000,
                        help="Val and test must each hold at least this many images of BOTH classes")
    parser.add_argument("--size", type=int, default=64, help="Must equal EyePreprocessConfig.size used live")
    parser.add_argument("--limit", type=int, default=0, help="Debug: only preprocess the first N images per split")
    parser.add_argument("--preview", action="store_true", help="Write preview montages of preprocessed images")
    parser.add_argument("--split-only", action="store_true", help="Compute and write the split, skip image packing")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    source = Source(args.source)
    records, skipped = parse_records(source.names)
    if not records:
        raise SystemExit("No files matched the MRL filename pattern under {}".format(args.source))
    print("[prepare] {:,} images parsed, {} files skipped (not matching the pattern)".format(len(records), skipped))
    table = subject_table(records)
    total = len(records)
    print("[prepare] {} subjects; closed {:,} ({:.1%}), open {:,}; glasses {:,} ({:.1%}); female subjects {}".format(
        len(table), sum(t["closed"] for t in table.values()), sum(t["closed"] for t in table.values()) / total,
        sum(t["open"] for t in table.values()), sum(t["glasses"] for t in table.values()),
        sum(t["glasses"] for t in table.values()) / total, sum(1 for t in table.values() if t["female"] > 0)))

    # --- split ----------------------------------------------------------------
    print("[prepare] searching {:,} seeded subject partitions ...".format(args.trials))
    assign, stats, terms, trial = search_split(table, args.seed, args.trials, args.val_frac, args.test_frac,
                                               args.min_class_images)
    by_split = {s: sorted(sub for sub, sp in assign.items() if sp == s) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not set(by_split[a]) & set(by_split[b]), "subject leakage between {} and {}".format(a, b)
    assert sum(len(v) for v in by_split.values()) == len(table), "every subject must be assigned exactly once"
    print("[prepare] best partition: trial {} of seed {}, cost {:.4f} = {}".format(
        trial, args.seed, sum(terms.values()), ", ".join("{} {:.3f}".format(k, v) for k, v in terms.items())))
    for s in SPLITS:
        st = stats[s]
        print("[prepare] {:<5} {:>2} subjects {:>6,} images ({:.1%})  closed {:>6,} ({:.1%})  open {:>6,}  "
              "glasses {:.1%}  female subjects {}  sensor2 {:,}  sensor3 {:,}".format(
                  s, st["subjects"], st["images"], st["fraction"], st["closed"], st["closed_ratio"], st["open"],
                  st["glasses_ratio"], st["female_subjects"], st["sensor2_images"], st["sensor3_images"]))
        print("          subjects: " + " ".join("s{:04d}".format(x) for x in by_split[s]))

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (args.splits_dir / "{}_subjects.txt".format(s)).write_text(
            "\n".join("s{:04d}".format(x) for x in by_split[s]) + "\n", encoding="utf-8")
    split_doc = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": args.source.name, "source_sha256": source.sha256(),
        "images_total": total, "subjects_total": len(table),
        "method": "seeded search over random subject partitions; greedy fill towards image-fraction targets; "
                  "lowest cost kept; whole subjects only",
        "seed": args.seed, "trials": args.trials, "best_trial": trial,
        "targets": {"train": 1 - args.val_frac - args.test_frac, "val": args.val_frac, "test": args.test_frac},
        "cost_terms": terms, "min_class_images": args.min_class_images,
        "splits": {s: {"subjects": ["s{:04d}".format(x) for x in by_split[s]], **stats[s]} for s in SPLITS},
        "per_subject": {"s{:04d}".format(s): table[s] for s in sorted(table)},
        "field_codings": {"gender": {"0": "male", "1": "female"}, "glasses": {"0": "no", "1": "yes"},
                          "eye_state": {"0": "closed", "1": "open"},
                          "reflections": {"0": "none", "1": "low", "2": "high"},
                          "lighting": {"0": "bad", "1": "good"},
                          "sensor": {"1": "RealSense SR300 640x480", "2": "IDS 1280x1024", "3": "Aptina 752x480"}},
    }
    (args.splits_dir / "split_stats.json").write_text(json.dumps(split_doc, indent=2), encoding="utf-8")
    print("[prepare] wrote subject lists + split_stats.json to {}".format(args.splits_dir))
    if args.split_only:
        return 0

    # --- preprocess + pack ------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    config = EyePreprocessConfig(size=args.size)
    per_split: Dict[str, List[Record]] = {s: [] for s in SPLITS}
    for r in records:
        per_split[assign[r.subject]].append(r)
    if args.limit:
        per_split = {s: v[:args.limit] for s, v in per_split.items()}

    manifest_rows = []
    size_hist: Counter = Counter()
    for s in SPLITS:
        recs = per_split[s]
        n = len(recs)
        images = np.zeros((n, args.size, args.size), dtype=np.uint8)
        fields = {k: np.zeros(n, dtype=np.uint8) for k in ("labels", "glasses", "gender", "reflections", "lighting", "sensor")}
        subjects = np.zeros(n, dtype=np.int32)
        orig = np.zeros((n, 2), dtype=np.uint16)
        t_split = time.perf_counter()
        for i, r in enumerate(recs):
            img = source.read_gray(r.path)
            if img is None:
                raise SystemExit("could not decode {}".format(r.path))
            gray, _ = preprocess_eye_image(img, config)
            images[i] = gray
            fields["labels"][i] = r.eye_state
            fields["glasses"][i] = r.glasses
            fields["gender"][i] = r.gender
            fields["reflections"][i] = r.reflections
            fields["lighting"][i] = r.lighting
            fields["sensor"][i] = r.sensor
            subjects[i] = r.subject
            orig[i] = (img.shape[1], img.shape[0])
            size_hist[img.shape[0]] += 1
            manifest_rows.append([s, r.path, r.subject, r.image_number, r.gender, r.glasses, r.eye_state,
                                  r.reflections, r.lighting, r.sensor, img.shape[1], img.shape[0]])
            if (i + 1) % 10000 == 0:
                print("[prepare]   {} {:>6,}/{:,} images ({:.0f}s)".format(s, i + 1, n, time.perf_counter() - t_split))
        np.savez_compressed(args.out / "{}.npz".format(s), images=images, subjects=subjects,
                            original_size=orig, **fields)
        print("[prepare] {:<5} packed {:,} images -> {} ({:.1f} MB, {:.0f}s)".format(
            s, n, (args.out / "{}.npz".format(s)).name, (args.out / "{}.npz".format(s)).stat().st_size / 1e6,
            time.perf_counter() - t_split))

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "path", "subject", "image_number", "gender", "glasses", "eye_state", "reflections",
                    "lighting", "sensor", "width", "height"])
        w.writerows(manifest_rows)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "source": args.source.name,
            "source_sha256": split_doc["source_sha256"], "preprocess": asdict(config),
            "preprocess_function": "src.eye_cnn.preprocess_eye_image (standardisation applied at load time by "
                                   "src.eye_cnn.normalize_eye)",
            "label_coding": {"0": "CLOSED", "1": "OPEN"}, "splits": {s: len(per_split[s]) for s in SPLITS},
            "original_size_px": {"min": min(size_hist), "max": max(size_hist),
                                 "median": int(np.median(np.repeat(list(size_hist.keys()), list(size_hist.values()))))},
            "seed": args.seed, "split_stats": str(args.splits_dir / "split_stats.json")}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- verify -------------------------------------------------------------------
    print("[verify ] re-opening the packs")
    seen: Dict[str, set] = {}
    for s in SPLITS:
        with np.load(args.out / "{}.npz".format(s)) as z:
            images, labels, subjects = z["images"], z["labels"], z["subjects"]
            assert images.dtype == np.uint8 and images.shape[1:] == (args.size, args.size), images.shape
            assert set(np.unique(labels).tolist()) <= {0, 1}, np.unique(labels)
            assert len(images) == len(labels) == len(subjects)
            seen[s] = set(subjects.tolist())
            n_closed = int((labels == 0).sum())
            print("[verify ] {:<5} images {} uint8 | closed {:,} open {:,} | subjects {} | glasses {:.1%} | "
                  "mean intensity {:.1f}".format(s, images.shape, n_closed, len(labels) - n_closed,
                                                  len(seen[s]), float((z["glasses"] == 1).mean()),
                                                  float(images.mean())))
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not (seen[a] & seen[b]), "LEAKAGE: subjects shared by {} and {}".format(a, b)
    print("[verify ] subject sets are pairwise disjoint: {}".format(
        " | ".join("{} {}".format(s, len(seen[s])) for s in SPLITS)))

    if args.preview:
        rng = np.random.default_rng(args.seed)
        for s in SPLITS:
            with np.load(args.out / "{}.npz".format(s)) as z:
                idx = rng.choice(len(z["labels"]), size=min(32, len(z["labels"])), replace=False)
                tiles = z["images"][idx]
                labels = z["labels"][idx]
            canvas = np.full((4 * 80, 8 * 68, 3), 30, np.uint8)
            for k, (t, l) in enumerate(zip(tiles, labels)):
                r, c = divmod(k, 8)
                y, x = r * 80 + 12, c * 68 + 2
                canvas[y:y + 64, x:x + 64] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
                cv2.putText(canvas, "CLOSED" if l == 0 else "OPEN", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 200, 255) if l == 0 else (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.out / "preview_{}.png".format(s)), canvas)
        print("[prepare] preview montages written to {}".format(args.out))

    print("[prepare] done in {:.1f} min -> {}".format((time.perf_counter() - t0) / 60.0, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
