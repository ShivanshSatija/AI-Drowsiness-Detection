# training/

Everything needed to build the eye-state CNN's training data (Stage 6) and to
train it in Google Colab (Stage 7). The live system never imports from here;
the shared preprocessing lives in `src/eye_cnn.py` and is imported by both.

## Stage 6 — MRL Eye Dataset

### 1. Obtain the dataset

The MRL Eye Dataset (Media Research Lab, VŠB – Technical University of
Ostrava) is a set of low-resolution infrared eye crops from 37 people, with
open/closed labels and glasses / reflection / lighting / sensor annotations.
It is distributed by the authors as a single zip archive:

- Direct download (verified 2026-09-13 by HTTP HEAD: `application/zip`,
  341,866,898 bytes ≈ **342 MB**):
  `https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip`
- The former landing page `mrl.cs.vsb.cz/eyedataset` currently serves the
  lab's homepage rather than a dataset description, so the archive link above
  is the reliable entry point. The annotation scheme documented by the
  authors (subject, image number, gender, glasses, eye state, reflections,
  lighting, sensor — encoded in each filename) is what the inspection script
  checks the real files against.
- Cite it in the report: Fusek, R. *Pupil localization using geodesic
  distance.* ISVC 2018, and the MRL Eye Dataset by name with the URL above.

**Prefer the official archive over Kaggle mirrors.** Several Kaggle
re-uploads reorganise the images into `Open_Eyes/` and `Closed_Eyes/` folders
or rename files; if the subject ID is lost in that process a
subject-independent split becomes impossible and any accuracy figure would be
inflated by the same people appearing in train and test. If only a mirror is
available, the inspection below will show whether subject IDs survived.

### 2. Where to put it

| Location | Purpose |
|---|---|
| `data/mrl/mrlEyes_2018_01.zip` (this repo, git-ignored) | local inspection and split preparation |
| `MyDrive/AI-Drowsiness-Detection/mrl/` on Google Drive | Colab training (Stage 7) |

Do not commit the dataset. `data/` is git-ignored for exactly this reason.

### 3. Inspect the real structure — before anything is written against it

```bat
python training\inspect_mrl.py data\mrl\mrlEyes_2018_01.zip
```

Works on the zip directly (no need to extract 85 k files first) or on an
extracted folder, and in Colab. It reports the folder layout, file counts,
sample filenames, how the filenames are tokenised and how many distinct
values each field takes, image sizes and colour modes on a random sample,
and any README inside the archive. It **does not assume** the layout; where
the filenames happen to match the pattern the authors document
(`sXXXX_YYYYY_G_GL_ST_RF_LT_SN.png`) it labels the fields with the documented
meanings, marked "to verify".

Paste the complete output into the chat. `prepare_mrl.py` is written from
that output, not from assumptions.

### 4. What `prepare_mrl.py` does (built from the inspected structure, 2026-09-13)

```bat
python training\prepare_mrl.py --source data\mrl\mrlEyes_2018_01.zip --out data\mrl_prepared --preview
```

1. Parses every filename (all 84,898 matched the pattern documented in the
   archive's `annotation.txt`; 0 skipped) into a manifest with subject, eye
   state, gender, glasses, reflections, lighting, sensor and original size.
2. **Subject-independent split by seeded search.** Whole subjects go to
   train / val / test. Because subjects range from 382 to 10,257 images and
   several are almost single-class, 20,000 seeded random partitions are
   scored and the lowest-cost one kept (image fractions near 70 / 15 / 15,
   val/test closed ratio and glasses ratio near the global values, at least
   one female subject and some sensor-02 images in val and test, at least
   2,000 images of each class in val and test). Disjointness is asserted
   before writing. Result (seed 0, trial 14,152): train 25 subjects / 59,012
   images, val 4 / 12,779, test 8 / 13,107; closed 49.3 / 49.6 / 49.8 %. The
   subject lists and all statistics are in `training/splits/` and are
   committed.
3. Preprocesses every image with `src.eye_cnn.preprocess_eye_image` and packs
   each split into `.npz` (uint8 64 × 64 images + all fields). Standardisation
   happens at load time via `src.eye_cnn.normalize_eye`.
4. Re-opens the packs and verifies shapes, dtypes, label values and
   disjointness; `--preview` writes 32-image montages per split.

Copy `data/mrl_prepared/{train,val,test}.npz` to
`MyDrive/AI-Drowsiness-Detection/mrl_prepared/` for the Colab notebook.

Augmentation (training-time only, in `src.eye_cnn.augment_eye`): horizontal
flip (the dataset does not label eye side), rotation / scale / shift jitter,
contrast / brightness / gamma, a low-light branch (darkening then sensor
noise, before standardisation), blur, specular spots and cutouts for glasses.
