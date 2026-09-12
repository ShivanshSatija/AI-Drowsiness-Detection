"""Inspect the MRL Eye Dataset as it actually is - a zip file or an extracted
folder - without assuming anything about its layout.

Reports: directory depth and top-level entries, file-extension counts, sample
filenames, an automatic analysis of the filename token structure (how many
underscore-separated fields, how many distinct values each field takes), image
sizes and colour modes on a random sample, and any README/annotation files
found inside. If the filenames match the annotation pattern the MRL authors
document, the per-field value counts are printed with those documented
meanings marked "to verify".

Usage (locally or in Colab; needs only the standard library plus Pillow for
image properties)::

    python training/inspect_mrl.py path/to/mrlEyes_2018_01.zip
    python training/inspect_mrl.py path/to/extracted_folder --sample 300

Paste the whole output into the chat; prepare_mrl.py is written from it.
"""

from __future__ import annotations

import argparse
import io
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is normally present
    Image = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".csv", ".md", ".json", ".xml", ".yaml", ".yml"}

# The annotation pattern documented by the dataset authors (MRL, VSB-TUO):
#   sXXXX_YYYYY_G_GL_ST_RF_LT_SN.png
# It is only used to LABEL the analysis; the analysis itself never relies on it.
DOCUMENTED_PATTERN = re.compile(
    r"^s(\d{4})_(\d{5})_([01])_([01])_([01])_([012])_([01])_(\d{2})$")
DOCUMENTED_FIELDS = [
    ("subject", None),
    ("image_number", None),
    ("gender", {"0": "man", "1": "woman"}),
    ("glasses", {"0": "no", "1": "yes"}),
    ("eye_state", {"0": "closed", "1": "open"}),
    ("reflections", {"0": "none", "1": "low", "2": "high"}),
    ("lighting", {"0": "bad", "1": "good"}),
    ("sensor", {"01": "RealSense SR300 640x480", "02": "IDS 1280x1024", "03": "Aptina 752x480"}),
]  # wording taken from the archive's own annotation.txt (verified 2026-09-13)


class Source:
    """Uniform view over a zip archive or a directory: relative paths + bytes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.zip: Optional[zipfile.ZipFile] = None
        if path.is_file() and zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
            self.files = [n for n in self.zip.namelist() if not n.endswith("/")]
            self.dirs = sorted({n for n in self.zip.namelist() if n.endswith("/")}
                               | {str(Path(f).parent).replace("\\", "/") + "/" for f in self.files
                                  if str(Path(f).parent) != "."})
        elif path.is_dir():
            self.files, self.dirs = [], []
            for root, dirnames, filenames in os.walk(path):
                rel_root = Path(root).relative_to(path)
                for d in dirnames:
                    self.dirs.append(str(rel_root / d).replace("\\", "/") + "/")
                for f in filenames:
                    self.files.append(str(rel_root / f).replace("\\", "/"))
            self.files = [f[2:] if f.startswith("./") else f for f in self.files]
            self.dirs = sorted(d[2:] if d.startswith("./") else d for d in self.dirs)
        else:
            raise SystemExit("Not a zip file or a directory: {}".format(path))

    def read(self, rel: str) -> bytes:
        if self.zip is not None:
            return self.zip.read(rel)
        return (self.path / rel).read_bytes()


def depth(rel: str) -> int:
    return rel.count("/")


def analyse_filename_tokens(stems: List[str]) -> Tuple[int, Dict[int, List[Counter]]]:
    """How many '_' separated fields do the stems have, and how many distinct
    values does each field take? Returns (most common token count, per-count
    list of per-field Counters)."""
    by_count: Dict[int, List[Counter]] = defaultdict(list)
    counts = Counter(len(s.split("_")) for s in stems)
    for stem in stems:
        parts = stem.split("_")
        n = len(parts)
        while len(by_count[n]) < n:
            by_count[n].append(Counter())
        for i, part in enumerate(parts):
            by_count[n][i][part] += 1
    return (counts.most_common(1)[0][0] if counts else 0), by_count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report the real structure of the MRL Eye Dataset.")
    parser.add_argument("path", type=Path, help="mrlEyes_*.zip or the extracted folder")
    parser.add_argument("--sample", type=int, default=200, help="Images to open for size/mode statistics")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    src = Source(args.path)
    rng = random.Random(args.seed)
    print("=" * 72)
    print("SOURCE      : {} ({})".format(args.path, "zip archive" if src.zip else "directory"))
    if src.zip:
        print("ARCHIVE SIZE: {:.1f} MB".format(args.path.stat().st_size / 1e6))
    print("FILES       : {:,}".format(len(src.files)))
    print("DIRECTORIES : {:,}".format(len(src.dirs)))
    depths = Counter(depth(f) for f in src.files)
    print("FILE DEPTHS : " + ", ".join("depth {} -> {:,} files".format(d, n) for d, n in sorted(depths.items())))

    # --- extensions ---------------------------------------------------------
    ext_counts = Counter(Path(f).suffix.lower() for f in src.files)
    print("\nEXTENSIONS  : " + ", ".join("{} x{:,}".format(e or "(none)", n) for e, n in ext_counts.most_common()))

    # --- top-level layout ---------------------------------------------------
    top = Counter((f.split("/")[0] if "/" in f else "(root)") for f in src.files)
    print("\nTOP-LEVEL ENTRIES ({}):".format(len(top)))
    for name, n in list(top.most_common())[:12]:
        print("  {:<40} {:>8,} files".format(name, n))
    if len(top) > 12:
        print("  ... {} more".format(len(top) - 12))

    # second level, if the top level is a single wrapper folder
    if len(top) == 1 and "(root)" not in top:
        second = Counter(f.split("/")[1] if f.count("/") >= 2 else "(files)" for f in src.files)
        print("\nSECOND-LEVEL ENTRIES under '{}' ({}):".format(next(iter(top)), len(second)))
        for name, n in sorted(second.items())[:45]:
            print("  {:<40} {:>8,} files".format(name, n))
        if len(second) > 45:
            print("  ... {} more".format(len(second) - 45))

    # --- text / annotation files ------------------------------------------
    text_files = [f for f in src.files if Path(f).suffix.lower() in TEXT_EXTENSIONS]
    print("\nTEXT / ANNOTATION FILES: {}".format(len(text_files)))
    for f in text_files[:10]:
        print("  {}".format(f))
        if src.zip or True:
            try:
                head = src.read(f)[:600].decode("utf-8", errors="replace")
                for line in head.splitlines()[:8]:
                    print("      | " + line)
            except Exception as exc:  # noqa: BLE001
                print("      (could not read: {})".format(exc))

    # --- filenames ----------------------------------------------------------
    images = [f for f in src.files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
    print("\nIMAGE FILES : {:,}".format(len(images)))
    if not images:
        print("No image files found - is this the right path?")
        return 1
    print("SAMPLE FILENAMES (first 5, then 5 random):")
    for f in images[:5] + rng.sample(images, min(5, len(images))):
        print("  {}".format(f))

    stems = [Path(f).stem for f in images]
    common_count, by_count = analyse_filename_tokens(stems)
    print("\nFILENAME TOKEN STRUCTURE (split on '_'):")
    for n, fields in sorted(by_count.items()):
        total = sum(fields[0].values()) if fields else 0
        print("  {:,} files have {} fields".format(total, n))
    if common_count:
        print("  Field-by-field for the {}-field files:".format(common_count))
        for i, counter in enumerate(by_count[common_count]):
            values = counter.most_common()
            preview = ", ".join("{}({:,})".format(v, c) for v, c in values[:6])
            print("    field {}: {:>6,} distinct values  e.g. {}{}".format(
                i + 1, len(values), preview, " ..." if len(values) > 6 else ""))

    # --- documented pattern check -------------------------------------------
    matched = [DOCUMENTED_PATTERN.match(s) for s in stems]
    n_match = sum(1 for m in matched if m)
    print("\nDOCUMENTED MRL PATTERN sXXXX_YYYYY_G_GL_ST_RF_LT_SN: {:,} / {:,} files match ({:.1%})".format(
        n_match, len(stems), n_match / len(stems)))
    if n_match / len(stems) > 0.95:
        per_field = [Counter() for _ in DOCUMENTED_FIELDS]
        subjects_images: Dict[str, Counter] = defaultdict(Counter)
        for m in matched:
            if not m:
                continue
            for i in range(len(DOCUMENTED_FIELDS)):
                per_field[i][m.group(i + 1)] += 1
            subjects_images[m.group(1)][m.group(5)] += 1
        print("  Per-field counts (documented meanings - TO VERIFY against the dataset's own README):")
        for (name, meaning), counter in zip(DOCUMENTED_FIELDS, per_field):
            if name in ("subject", "image_number"):
                print("    {:<13} {:,} distinct values".format(name, len(counter)))
                continue
            parts = []
            for value, count in sorted(counter.items()):
                label = meaning.get(value, "?") if meaning else ""
                parts.append("{}={}: {:,}".format(value, label, count))
            print("    {:<13} {}".format(name, "  ".join(parts)))
        print("  Images per subject (field 5 = documented eye state; 0/1 counts shown):")
        for subject in sorted(subjects_images):
            c = subjects_images[subject]
            print("    s{}  total {:>6,}   state0 {:>6,}   state1 {:>6,}".format(
                subject, sum(c.values()), c.get("0", 0), c.get("1", 0)))
    else:
        print("  -> filenames do NOT follow the documented pattern; subject IDs may be missing."
              " Report this - a subject-independent split needs them.")

    # --- image properties ---------------------------------------------------
    if Image is None:
        print("\nIMAGE PROPERTIES: Pillow not installed - skipped (pip install pillow)")
        return 0
    sample = rng.sample(images, min(args.sample, len(images)))
    sizes, modes, failures = Counter(), Counter(), 0
    widths, heights = [], []
    for f in sample:
        try:
            with Image.open(io.BytesIO(src.read(f))) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
                widths.append(im.size[0])
                heights.append(im.size[1])
        except Exception:  # noqa: BLE001
            failures += 1
    print("\nIMAGE PROPERTIES (random sample of {}):".format(len(sample)))
    print("  modes : " + ", ".join("{} x{}".format(m, n) for m, n in modes.most_common())
          + "   (L = 8-bit grayscale)")
    if widths:
        widths.sort()
        heights.sort()
        print("  width : min {}  median {}  max {}".format(widths[0], widths[len(widths) // 2], widths[-1]))
        print("  height: min {}  median {}  max {}".format(heights[0], heights[len(heights) // 2], heights[-1]))
        print("  most common sizes: " + ", ".join("{}x{} x{}".format(w, h, n) for (w, h), n in sizes.most_common(6)))
    if failures:
        print("  unreadable files in sample: {}".format(failures))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
