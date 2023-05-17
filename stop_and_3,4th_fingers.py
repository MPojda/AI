import cv2
import numpy as np
import mediapipe as mp



mp_draws = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
film = cv2.VideoCapture(0)
#funkcje
def vec(x, y): return (x, y)

def dot(a, b): return a[0]*b[0] + a[1]*b[1]

def len(a): return np.sqrt(dot(a, a))

def nor(a): return a / len(a)

#wyświetlanie filmu

while True:
    _, obrazek = film.read()

    results = hands.process(obrazek)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draws.draw_landmarks(obrazek, hand_lms, mp_hands.HAND_CONNECTIONS)

            # vektory od kciuka dalej

            v1 = vec(hand_lms.landmark[1].x - hand_lms.landmark[4].x,
                     hand_lms.landmark[1].y - hand_lms.landmark[4].y)
            v2 = vec(hand_lms.landmark[5].x - hand_lms.landmark[8].x,
                     hand_lms.landmark[5].y - hand_lms.landmark[8].y)
            v3 = vec(hand_lms.landmark[9].x - hand_lms.landmark[12].x,
                     hand_lms.landmark[9].y - hand_lms.landmark[12].y)
            v4 = vec(hand_lms.landmark[13].x - hand_lms.landmark[16].x,
                     hand_lms.landmark[13].y - hand_lms.landmark[16].y)
            v5 = vec(hand_lms.landmark[17].x - hand_lms.landmark[20].x,
                     hand_lms.landmark[17].y - hand_lms.landmark[20].y)

            #normalizowanie vektorów

            v1 = nor(v1)
            v2 = nor(v2)
            v3 = nor(v3)
            v4 = nor(v4)
            v5 = nor(v5)

            kat_1 = dot(v1, v2)
            kat_2 = dot(v2, v3)
            kat_3 = dot(v3, v4)
            kat_4 = dot(v4, v5)

            #wypisywanie kątów

            kat_wypisz = f"({kat_1:.2f},{kat_2:.2f},{kat_3:.2f},{kat_4:.2f})"
            cv2.putText(obrazek, kat_wypisz, (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (250, 10, 50), 2, cv2.LINE_AA)



            '''napis = f"({v1[0]:.2f}, {v1[1]:.2f})"
            cv2.putText(obrazek, napis, (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (250, 130, 50), 2, cv2.LINE_AA)

            napis = f"({v2[0]:.2f}, {v2[1]:.2f})"
            cv2.putText(obrazek, napis, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (250, 130, 50), 2, cv2.LINE_AA)'''

            if kat_3 > 0.99 and kat_2 < 0.97 and kat_4 < 0.97 and kat_1 < 0.75:
                napis = "3 i 4 palec zlaczony"
            elif kat_1 > 0.95 and kat_2 < 1.0 and kat_3 < 1.0 and kat_4 < 1.0:
                napis = "stop"
            else:
                napis = ""

            cv2.putText(obrazek, napis, (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 130, 230), 2, cv2.LINE_AA)

    cv2.imshow("Testy gestów", obrazek)

    if cv2.waitKey(16) & 255 == ord('q'):
        break




