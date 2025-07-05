import cv2
import pickle    # daha onceden isaretlenen park yeri koordinatlarini dosyadan okur
import numpy as np

def checkParkSpace(imgg):
    spaceCounter = 0    # bos park yeri sayisini tutmak icin
    for pos in posList:
        x, y = pos

        img_crop = imgg[y: y + 15, x: x + 27]
        count = cv2.countNonZero(img_crop)    # ilgili korrdinatlardaki pikseller sayilir

        #print("count: ", count)

        if count < 150:
            color = (0, 255, 0)    #yeşil: boş
            spaceCounter += 1

        else:   # çok piksel varsa
            color = (0, 0, 255)   # kırmızı: dolu
        cv2.rectangle(img, pos, (pos[0] + 27, pos[1] + 15), color, 2)


    cv2.putText(img, f"Empty Space: {spaceCounter} / {len(posList)}", (15,24), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)


cap = cv2.VideoCapture("parking_space.mp4")

with open("CarParkPos", "rb") as f:
    posList = pickle.load(f)

while True:

    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # siyah beyaza cevrildi

    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imMedian = cv2.medianBlur(imgThreshold,5)
    imgDilate = cv2.dilate(imMedian, np.ones((3, 3)), iterations=1)   # kalınlaştırır

    checkParkSpace(imgDilate)   # fonksiyonu cagir


    cv2.imshow("img", img)
    #cv2.imshow("imgGray", imgGray)
    #cv2.imshow("imgBlur", imgBlur)
    #cv2.imshow("imgThreshold", imgThreshold)
    #cv2.imshow("imMedian", imMedian)
    #cv2.imshow("imgDilate", imgDilate)
    cv2.waitKey(200)