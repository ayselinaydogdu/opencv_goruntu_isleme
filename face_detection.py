import cv2
import mediapipe as mp

cap = cv2.VideoCapture("yüz1.mp4")
mpFaceDetection = mp.solutions.face_detection   # face_detection: MediaPipe'in yuz algilama moduludur
faceDetection = mpFaceDetection.FaceDetection(0.20)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # MediaPipe RGB formatinda, OpenCV ise BGR formatinda.


    results = faceDetection.process(imgRGB)   # imgRGB goruntusundeki yuzler process() ile algilanir.
    #print(results.detections)

    if results.detections:    # eger herhangi bir yuz algilandiysa
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            #print(bboxC)

            h, w, _ = img.shape
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)

            cv2.rectangle(img, bbox, (0, 255, 255), 2)   #bbox korrdinatlarina gore bir sari dikdortgen cizilir

    cv2.imshow("img", img)
    cv2.waitKey(10)


