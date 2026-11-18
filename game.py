import math
from pathlib import Path
import cv2
import joblib
import mediapipe as mp
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_DIR = Path(__file__).parent
LANDMARKER_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = PROJECT_DIR / "models" / "gesture_classifier.joblib"
CAMERA_INDEX = 0  # Mets ici l’index de caméra qui fonctionne chez toi
WIDTH, HEIGHT = 900, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 16, 110
BALL_RADIUS = 10
CONFIDENCE_MIN = 0.60
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

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("HandPong AI")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 42)
small_font = pygame.font.Font(None, 26)
camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
   pygame.quit()
   raise RuntimeError(f"Impossible d'ouvrir la caméra à l'index {CAMERA_INDEX}.")
left_y = HEIGHT // 2
right_y = HEIGHT // 2
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_vx = 9
ball_vy = 6
score_left = 0
score_right = 0
frame_number = 0
running = True
try:
   with vision.HandLandmarker.create_from_options(options) as detector:
       while running:
           for event in pygame.event.get():
               if event.type == pygame.QUIT:
                   running = False
           success, frame = camera.read()
           if success:
               frame = cv2.flip(frame, 1)
               rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               image = mp.Image(
                   image_format=mp.ImageFormat.SRGB,
                   data=rgb_frame,
               )
               result = detector.detect_for_video(image, frame_number * 33)
               frame_number += 1
               for hand in result.hand_landmarks:
                   features = extract_features(hand)
                   if features is None:
                       continue
                   probabilities = model.predict_proba([features])[0]
                   best_index = probabilities.argmax()
                   label = model.classes_[best_index]
                   confidence = probabilities[best_index]
                   if confidence < CONFIDENCE_MIN:
                       continue
                   hand_y = int(hand[0].y * HEIGHT)
                   if label == "poing":
                       left_y = hand_y
                   elif label == "paume":
                       right_y = hand_y
           left_y = max(PADDLE_HEIGHT // 2, min(HEIGHT - PADDLE_HEIGHT // 2, left_y))
           right_y = max(PADDLE_HEIGHT // 2, min(HEIGHT - PADDLE_HEIGHT // 2, right_y))
           left_paddle = pygame.Rect(
               35, left_y - PADDLE_HEIGHT // 2,
               PADDLE_WIDTH, PADDLE_HEIGHT,
           )
           right_paddle = pygame.Rect(
               WIDTH - 35 - PADDLE_WIDTH, right_y - PADDLE_HEIGHT // 2,
               PADDLE_WIDTH, PADDLE_HEIGHT,
           )
           ball = pygame.Rect(
               int(ball_x - BALL_RADIUS),
               int(ball_y - BALL_RADIUS),
               BALL_RADIUS * 2,
               BALL_RADIUS * 2,
           )
           ball_x += ball_vx
           ball_y += ball_vy
           if ball_y <= BALL_RADIUS or ball_y >= HEIGHT - BALL_RADIUS:
               ball_vy *= -1
           ball = pygame.Rect(
               int(ball_x - BALL_RADIUS),
               int(ball_y - BALL_RADIUS),
               BALL_RADIUS * 2,
               BALL_RADIUS * 2,
           )
           if ball.colliderect(left_paddle) and ball_vx < 0:
               ball_vx *= -1
               ball_vy += (ball_y - left_y) * 0.04
           if ball.colliderect(right_paddle) and ball_vx > 0:
               ball_vx *= -1
               ball_vy += (ball_y - right_y) * 0.04
           if ball_x < 0:
               score_right += 1
               ball_x, ball_y = WIDTH // 2, HEIGHT // 2
               ball_vx, ball_vy = 6, 4
           elif ball_x > WIDTH:
               score_left += 1
               ball_x, ball_y = WIDTH // 2, HEIGHT // 2
               ball_vx, ball_vy = -6, 4
           screen.fill((18, 24, 38))
           pygame.draw.line(
               screen, (90, 100, 120),
               (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2,
           )
           pygame.draw.rect(screen, (70, 190, 255), left_paddle, border_radius=6)
           pygame.draw.rect(screen, (255, 170, 70), right_paddle, border_radius=6)
           pygame.draw.circle(screen, (245, 245, 245), (int(ball_x), int(ball_y)), BALL_RADIUS)
           score_text = font.render(f"{score_left}     {score_right}", True, (255, 255, 255))
           screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 25))
           instructions = small_font.render(
               "Poing : joueur gauche  |  Paume : joueur droit  |  Fermer la fenetre pour quitter",
               True,
               (190, 200, 215),
           )
           screen.blit(instructions, (20, HEIGHT - 35))
           pygame.display.flip()
           clock.tick(60)
finally:
   camera.release()
   pygame.quit()