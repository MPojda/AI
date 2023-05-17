import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

def vec(x, y): return (x, y)

def dot(a, b): return a[0]*b[0] + a[1]*b[1]

def len(a): return np.sqrt(dot(a, a))

def nor(a): return a / len(a)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Przetwarzanie każdej klatki obrazu
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        psl = results.pose_landmarks

        if psl is not None:
            landmark_point = psl.landmark[11] # wybieramy punkt kluczowy o numerze 0
            landmark_x = landmark_point.x
            landmark_y = landmark_point.y
            landmark_z = landmark_point.z
            #print("Koordynaty punktu kluczowego 0: x={:.2f}, y={:.2f}, z={:.2f}".format(landmark_x, landmark_y, landmark_z))

            v1 = vec(psl.landmark[7].x - psl.landmark[11].x,
                 psl.landmark[7].y - psl.landmark[11].y)
            v2 = vec(psl.landmark[8].x - psl.landmark[12].x,
                 psl.landmark[8].y - psl.landmark[12].y)

            v1 = nor(v1)
            v2 = nor(v2)

            napis = f"({v1[0]:.2f},{v1[1]:.2f})"
            cv2.putText(cap, napis, (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2, cv2.LINE_AA)
            napis = f"({v2[0]:.2f},{v2[1]:.2f})"
            cv2.putText(cap, napis, (40, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (200, 130, 30), 2, cv2.LINE_AA)


        # Rysowanie pozy na klatce obrazu
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
