### import bibliotek ###

import cv2
import numpy as np

### zmienne globalne ###

strumien_video = cv2.VideoCapture(0)

detektor_twarzy = cv2.CascadeClassifier('kaskady/haarcascade_frontalface_default.xml')
detektor_oczu = cv2.CascadeClassifier('kaskady/haarcascade_eye.xml')

### pętla główna ###

while True:
    _, obrazek = strumien_video.read()

    wyszarzony_obrazek = cv2.cvtColor(obrazek, cv2.COLOR_BGR2GRAY)

    ### wykrywanie twarzy i oczu ###

    twarze = detektor_twarzy.detectMultiScale(wyszarzony_obrazek, scaleFactor=1.05, minNeighbors=15,
                                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in twarze:
        cv2.rectangle(obrazek, (x, y), (x + w, y + h), (0, 0, 255), 2)

        roi_gray = wyszarzony_obrazek[y:y + h, x:x + w]
        roi_color = obrazek[y:y + h, x:x + w]

        oczy = detektor_oczu.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=20,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (ex, ey, ew, eh) in oczy:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        ### określanie stopnia pochylenia twarzy ###

        if 2 == len(oczy):
            lewe_oko = oczy[0]
            prawe_oko = oczy[1]

            if (lewe_oko[0] > prawe_oko[0]):
                lewe_oko, prawe_oko = prawe_oko, lewe_oko

            if lewe_oko[1] > prawe_oko[1]:
                cv2.putText(obrazek, "w lewo", (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2, cv2.LINE_AA)
            else:
                cv2.putText(obrazek, "w prawo", (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2, cv2.LINE_AA)

            cv2.putText(obrazek, str(lewe_oko), (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2, cv2.LINE_AA)
            cv2.putText(obrazek, str(prawe_oko), (40, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2,
                        cv2.LINE_AA)

    ### pokazywanie rezultatu i czytanie klawiatury ###

    cv2.imshow("Oczy", obrazek)

    if cv2.waitKey(16) & 255 == ord('q'):
        break