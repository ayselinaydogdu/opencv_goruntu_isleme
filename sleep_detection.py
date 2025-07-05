import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector   # yuzdeki 468 noktayi bulur
from cvzone.PlotModule import LivePlot  # canli grafik

cap = cv2.VideoCapture("sleep2.mp4")
detector = FaceMeshDetector()
plotY = LivePlot(540, 360, [10, 60])

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
color = (0, 0, 255)
ratioList = []
counter = 0
blinkCounter = 0

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:   # eger en az bir yuz algilanmissa
        face = faces[0]    # ilk yuz seciliyor
        for id in idList:  # Secilen yuzun idList icinde tanimli olan gozle ilgili landmark noktalarina
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 255, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 255, 0), 3)

        ratio = int((lengthVer / lengthHor)*100)   # gozun ne kadar acik oldugunu gosteren oran, kuculurse goz kapalidir
        ratioList.append(ratio)

        if len(ratioList) > 3:  # son 3 oran saklanir
            ratioList.pop(0)

        ratioAverage = sum(ratioList) / len(ratioList)
        print(ratioAverage)

        if ratioAverage < 35 and counter == 0:
            blinkCounter += 1
            color = (0, 255, 0)
            counter += 1

        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (0, 0, 255)
        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), colorR=color)


        imgPlot = plotY.update(ratioAverage, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)



    cv2.imshow("img", imgStack)
    cv2.waitKey(25)

    