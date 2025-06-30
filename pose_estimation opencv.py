import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture("video5.mp4")

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    # draw fonksiyonu çağrılmalıdır
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, color = img.shape

            cx, cy = int(lm.x*width), int(lm.y*height)
            if id == 4:
                cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)


    cv2.imshow("img", img)
    cv2.waitKey(25)


