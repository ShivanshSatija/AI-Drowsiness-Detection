# models/

Model files needed at runtime. Small enough to commit, so a fresh clone runs
offline.

| File | Origin | Size | SHA-256 |
|---|---|---|---|
| `face_landmarker.task` | Official MediaPipe Face Landmarker model (`float16/1`), downloaded from `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` on 2026-09-13. Apache-2.0 licensed by Google. | 3,758,596 bytes | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| `eye_cnn.pt` | Eye-state CNN trained in this project (Stage 7) with `training/train_eye_cnn.py --export`. Self-describing checkpoint (`format: eye_cnn_v1`): weights, architecture, the exact preprocessing config, class names `("CLOSED", "OPEN")` and test metrics as metadata. **Current file: run 01, epoch 5 (interrupted run), test accuracy 0.9373 on 8 unseen subjects** — see `evaluation/results/eye_cnn_run01_epoch5/`. Replaced when the resumed run finishes. | 2,350,181 bytes | (embedded in the checkpoint metadata) |

`src/landmarks.py` checks the hash of `face_landmarker.task` when it loads and
prints a warning if the file differs from the version the project was
developed against, so a silently substituted model cannot go unnoticed.

Training checkpoints (`models/checkpoints/`) are git-ignored; only the final
model that the live system loads is committed.
