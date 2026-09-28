# Face Expression Reader

Points your webcam at your face and guesses a rough expression: neutral,
smiling, surprised, or eyes closed. Runs entirely on your machine using
mediapipe's face mesh -- no cloud API, no account, no data leaves your PC.

## Setup

    pip install -r requirements.txt
    python main.py

Grant webcam access if your OS prompts you. Close the window or Ctrl+C in
the terminal to release the camera.

## How it works

`expression_logic.py` measures a few distances between face landmarks
(eye openness, mouth width/height, eyebrow height) and classifies based on
simple thresholds. It's a heuristic, not a trained model -- lighting and
camera angle affect accuracy.
