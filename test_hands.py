from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = Path(__file__).parent / "models" / "hand_landmarker.task"
if not model_path.is_file():
    raise FileNotFoundError(f"Modèle introuvable : {model_path}")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(model_path)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera.release()
    raise RuntimeError("Impossible d'ouvrir la webcam à l'index 0.")

try:
    with vision.HandLandmarker.create_from_options(options) as detector:
        frame_number = 0

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Impossible de lire une image de la webcam.")

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = frame_number * 33
            frame_number += 1
            result = detector.detect_for_video(image, timestamp_ms)

            height, width, _ = frame.shape
            for hand in result.hand_landmarks:
                for point in hand:
                    x = int(point.x * width)
                    y = int(point.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"Mains detectees : {len(result.hand_landmarks)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Test MediaPipe - appuie sur Q pour quitter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()