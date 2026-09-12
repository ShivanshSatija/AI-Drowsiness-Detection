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

### 4. What `prepare_mrl.py` will do (plan — confirmed only after inspection)

1. Walk the dataset, parse the annotation fields from each filename, and
   write a manifest CSV: path, subject, eye state, glasses, reflections,
   lighting, sensor, image size.
2. **Subject-independent split.** Subjects (not images) are assigned to
   train / validation / test with a fixed seed; no subject appears in more
   than one split, and the script asserts this before writing anything. The
   assignment is chosen so that each split contains both eye states, both
   glasses conditions and, as far as 37 subjects allow, every sensor. The
   exact subject lists are written to `training/splits/*.txt` and committed,
   so the split is reproducible and auditable.
3. Preprocess every image with `src.eye_cnn.preprocess_eye_image` — the same
   function the live system uses — and pack each split into a compact
   `.npz` (uint8 64 × 64 images, labels, subject IDs, annotations) for fast
   loading in Colab. Standardisation is applied at load time by the same
   `normalize_eye` function, so training and inference tensors are produced
   by identical code.
4. Report the class balance, glasses balance and image-size statistics per
   split, and place a few MRL samples next to the eye crops saved from the
   live camera (`data/eye_crops/`) to check that the live `crop_scale`
   produces comparable framing before any training happens.

Augmentation (training-time only, Stage 7): horizontal flip (the dataset does
not label eye side, so the classifier must be side-agnostic), small rotation
and scale/shift jitter (to absorb crop-geometry differences), brightness /
contrast / gamma jitter and mild blur or noise (low-light and IR rehearsal).
Augmentation is applied to the uint8 image *before* standardisation.
