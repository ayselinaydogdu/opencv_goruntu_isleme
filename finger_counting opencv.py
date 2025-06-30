import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_RGB)

    lm_list = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                height, width, color = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                lm_list.append([id, cx, cy])

                if id == 8:
                    cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED)

                if id == 6:
                    cv2.circle(img, (cx, cy), 9, (0, 0, 255), cv2.FILLED)


    if len(lm_list) != 0:
        fingers = []

        el_yonu = results.multi_handedness[0].classification[0].label
        if el_yonu == "Right":

            if lm_list[tipIds[0]][1] < lm_list[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        elif el_yonu == "Left":
            if lm_list[tipIds[0]][1] > lm_list[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        for id in range(1, 5):

            if lm_list[tipIds[id]][2] < lm_list[tipIds[id] - 2][2]:
                fingers.append(1)

            else:
                fingers.append(0)

        #print(fingers)

        total_acik_parmak_sayisi = fingers.count(1)    
        #print(total_acik_parmak_sayisi)

        cv2.putText(img, str(total_acik_parmak_sayisi), (30,125), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 8)

    #print(lm_list)


    cv2.imshow("img", img)
    cv2.waitKey(1)
