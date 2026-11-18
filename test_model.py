import math
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_DIR = Path(__file__).parent
LANDMARKER_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = PROJECT_DIR / "models" / "gesture_classifier.joblib"
CAMERA_INDEX = 0  # Remplace par l’index qui fonctionne chez toi
model = joblib.load(CLASSIFIER_PATH)
options = vision.HandLandmarkerOptions(
   base_options=python.BaseOptions(
       model_asset_path=str(LANDMARKER_PATH)
   ),
   running_mode=vision.RunningMode.VIDEO,
   num_hands=2,
)
def extract_features(hand):
   wrist = hand[0]
   scale = math.hypot(hand[9].x - wrist.x, hand[9].y - wrist.y)
   if scale < 1e-6:
       return None
   features = []
   for point in hand:
       features.extend([
           (point.x - wrist.x) / scale,
           (point.y - wrist.y) / scale,
           (point.z - wrist.z) / scale,
       ])
   return features

camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
   raise RuntimeError(f"Impossible d’ouvrir la caméra à l’index {CAMERA_INDEX}.")
try:
   with vision.HandLandmarker.create_from_options(options) as detector:
       frame_number = 0
       while True:
           success, frame = camera.read()
           if not success:
               print("Impossible de lire l’image de la caméra.")
               break
           frame = cv2.flip(frame, 1)
           rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           image = mp.Image(
               image_format=mp.ImageFormat.SRGB,
               data=rgb_frame,
           )
           result = detector.detect_for_video(
               image,
               frame_number * 33,
           )
           frame_number += 1
           height, width, _ = frame.shape
           for hand in result.hand_landmarks:
               features = extract_features(hand)
               if features is None:
                   continue
               probabilities = model.predict_proba([features])[0]
               index = probabilities.argmax()
               label = model.classes_[index]
               confidence = probabilities[index]
               x = int(hand[0].x * width)
               y = int(hand[0].y * height)
               cv2.putText(
                   frame,
                   f"{label} ({confidence:.0%})",
                   (x, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   (0, 255, 0),
                   2,
               )
           cv2.imshow("Test du modele — Q pour quitter", frame)
           if cv2.waitKey(1) & 0xFF == ord("q"):
               break
finally:
   camera.release()
   cv2.destroyAllWindows()