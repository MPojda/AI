import cv2
import numpy as np
import mediapipe as mp
import time


mp_draws = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
film = cv2.VideoCapture(0)
previousTime = 0
currentTime = 0


### funkcje ###

def vec(x, y): return [x, y]
def dot(a, b): return a[0]*b[0] + a[1]*b[1]
def len(a): return np.sqrt(dot(a, a))
def nor(a): return a / len(a)


### pętla wyświetlania filmu ###

while True:
    _, obrazek = film.read()

    results = hands.process(obrazek)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draws.draw_landmarks(obrazek, hand_lms, mp_hands.HAND_CONNECTIONS)

            v1 = vec(hand_lms.landmark[4].x - hand_lms.landmark[0].x,
                      hand_lms.landmark[4].y - hand_lms.landmark[0].y)

            v2 = vec(hand_lms.landmark[17].x - hand_lms.landmark[0].x,
                      hand_lms.landmark[17].y - hand_lms.landmark[0].y)

            v1 = nor(v1)
            v2 = nor(v2)
            kat = dot(v1, v2)

            napis = f"({v1[0]:.2f}, {v1[1]:.2f})"
            cv2.putText(obrazek, napis, (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2, cv2.LINE_AA)

            napis = f"({v2[0]:.2f}, {v2[1]:.2f})"
            cv2.putText(obrazek, napis, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2, cv2.LINE_AA)

            if kat < 0.25 and dot(vec(0, -1), v1) > 0.8:
                napis = "OK"
            elif kat < 0.25 and dot(vec(0, 1), v1) > 0.8:
                napis = "NO OK"
            elif kat < 0.25 and dot(vec(1, 1), v1) > 0.8:
                napis = "LEWO"
            elif kat < 0.25 and dot(vec(-1, 1), v1) > 0.8:
                napis = "PRAWO"
            else:
                napis = ""

            cv2.putText(obrazek, napis, (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 130, 230), 2, cv2.LINE_AA)

    cv2.imshow("Testy gestów", obrazek)

    if cv2.waitKey(16) & 255 == ord('q'):
        break
