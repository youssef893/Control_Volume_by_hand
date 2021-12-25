import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hilostic = mp.solutions.holistic
hiolistic = mp_hilostic.Holistic()
mp_draw = mp.solutions.drawing_utils
drawing_specs = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hiolistic.process(imgRgb)

    mp_draw.draw_landmarks(img, results.face_landmarks, mp_hilostic.FACEMESH_CONTOURS, drawing_specs, drawing_specs)
    mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp_hilostic.HAND_CONNECTIONS, drawing_specs, drawing_specs)
    mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp_hilostic.HAND_CONNECTIONS, drawing_specs, drawing_specs)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_hilostic.POSE_CONNECTIONS, drawing_specs, drawing_specs)

    cv2.imshow('image', img)

    if cv2.waitKey(1) == 13:
        break
