import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HANDPONG_DIR = Path(__file__).parent
MODEL_PATH = HANDPONG_DIR / "models" / "hand_landmarker.task"
DATA_PATH = HANDPONG_DIR / "data" / "gestures.csv"
CAMERA_INDEX = 0  # Ta webcam externe

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")


def extract_features(hand):
    wrist = hand[0]
    scale = math.hypot(hand[9].x - wrist.x, hand[9].y - wrist.y)

    if scale < 1e-6:
        return None

    features = []
    for point in hand:
        features.extend((
            (point.x - wrist.x) / scale,
            (point.y - wrist.y) / scale,
            (point.z - wrist.z) / scale,
        ))

    return features


options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    camera.release()
    raise RuntimeError(f"Impossible d'ouvrir la caméra à l'index {CAMERA_INDEX}.")

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
write_header = not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0
header = ["label"] + [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]

try:
    with DATA_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(header)

        with vision.HandLandmarker.create_from_options(options) as detector:
            label = None
            saved = {"poing": 0, "paume": 0}
            frame_number = 0

            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Impossible de lire une image de la webcam.")

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                result = detector.detect_for_video(image, frame_number * 33)
                frame_number += 1

                hands = result.hand_landmarks
                height, width, _ = frame.shape

                for hand in hands:
                    for point in hand:
                        x = int(point.x * width)
                        y = int(point.y * height)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                label_text = label if label else "aucun"
                message = (
                    f"Geste: {label_text} | Mains: {len(hands)} | "
                    f"Enregistrees: poing {saved['poing']}, paume {saved['paume']}"
                )
                cv2.putText(
                    frame,
                    message,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "F=poing, P=paume, ESPACE=exemple, Q=quitter",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Collecte des gestes", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("f"):
                    label = "poing"
                elif key == ord("p"):
                    label = "paume"
                elif key == ord(" ") and label is not None and len(hands) == 1:
                    features = extract_features(hands[0])
                    if features is not None:
                        writer.writerow([label, *features])
                        csv_file.flush()
                        saved[label] += 1

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Exemples ajoutés dans : {DATA_PATH}")