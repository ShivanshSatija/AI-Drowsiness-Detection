"""AI Driver Drowsiness Detection System.

Modules are added stage by stage:

    capture.py    Stage 1  - camera abstraction + live preview   [done]
    landmarks.py  Stage 2  - MediaPipe Face Landmarker           [done]
    features.py   Stage 3  - EAR / MAR, Stage 4 - frame validity  [done]
    headpose.py   Stage 4  - head pose: yaw / pitch / roll        [done]
    eye_cnn.py    Stage 5  - eye crop preprocessing (shared with training) [done]
                  Stage 7  - CNN, augmentation, checkpoint I/O, classifier [code done, untrained]
    temporal.py   Stage 9  - 60 s window: PERCLOS / blinks / yawns / nods, FSM + hysteresis [done, untuned]
    alert.py      Stage 10 - escalating laptop alerts
    hardware.py   Stage 11 - ESP32 serial link
    app.py        Stage 12 - Streamlit dashboard
"""

__version__ = "0.1.0"
