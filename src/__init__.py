"""AI Driver Drowsiness Detection System.

Modules are added stage by stage:

    capture.py    Stage 1  - camera abstraction + live preview   [done]
    landmarks.py  Stage 2  - MediaPipe Face Mesh
    features.py   Stage 3  - EAR / MAR, Stage 4 - head pose
    eye_cnn.py    Stage 7  - eye-state CNN loading + inference
    temporal.py   Stage 9  - PERCLOS / blink / yawn / nod, state machine
    alert.py      Stage 10 - escalating laptop alerts
    hardware.py   Stage 11 - ESP32 serial link
    app.py        Stage 12 - Streamlit dashboard
"""

__version__ = "0.1.0"
