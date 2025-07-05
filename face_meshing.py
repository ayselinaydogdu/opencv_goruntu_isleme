import cv2
import time
import mediapipe as mp   # insan vucudu, eller, yuz gibi yapilari tespit etmek icin kullanilan bir kutuphane.

cap = cv2.VideoCapture("face_mesh1.mp4")
mpFaceMesh = mp.solutions.face_mesh   # MediaPipe'in face_mesh modulunu alir
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)   # sadece bir yüz algilanacak
mpDraw = mp.solutions.drawing_utils   # mpDraw yuz noktalarini(landmark: yuzun X, Y,, Z koordinatları) cizmek icindir
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)  # noktalarin çizim kalinligini ve daire yaricapini belirler.
previous_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    print(results.multi_face_landmarks)

    if results.multi_face_landmarks:  # eger yuz bulunduysa

        # görselleştir
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec)

        # x y koordinatları
        for id, lm in enumerate(faceLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print([id, cx, cy])


    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, "FPS:"+ str(int(fps)), (10,65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(50)