import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)
video.set(3, 1280)
video.set(4, 720)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    imgFlip = cv2.flip(img, 1)
    cv2.imshow("Img", imgFlip)
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
