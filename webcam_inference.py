import cv2
import os
import sys
import matplotlib.pyplot as plt
import inference as faceid

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()

    if status:

        sim = faceid.main(frame)

        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame, (400,0), (510, 128), (0,255,0), 3)
        frame = cv2.putText(frame, "YouKnowYunHo", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("test", frame)

    if cv2.waitKey(1) == 32:
        break

webcam.release()
cv2.destroyAllWindows()