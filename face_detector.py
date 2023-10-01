import cv2
import os
import cv2
import numpy as np
from time import time

# 임계값 설정
EAR_THRESHOLD = 0.2 

def predict_drowsiness_with_dlib(predictor, faces, frame, gray):

    def eye_aspect_ratio(eye):
        # 눈의 수직 거리
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # 눈의 수평 거리
        C = np.linalg.norm(eye[0] - eye[3])
        # 눈의 개방 비율 계산
        ear = (A + B) / (2.0 * C)
        return ear
    
    drowsiness = 0
    for face in faces:
        landmarks = predictor(gray, face)

        # 왼쪽 눈과 오른쪽 눈의 랜드마크 좌표 추출
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            drowsiness += 1

    if drowsiness >= len(faces)/2:  return True, drowsiness/len(faces)
    else:  return False, 0


def detect_faces_with_dlib(detector, frame, gray):
    # img = cv2.resize(img, (2400, 1200))
    # print(img.shape)
    
    start = time()

    faces = detector(gray)
    
    if not faces:
        return frame, [], faces

    cropped_faces = []
    for idx, rect in enumerate(faces):
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cropped_face = cv2.resize(frame[y:y+h, x:x+w, :], (112, 112))
        cropped_faces.append(cropped_face)
        # cv2.imwrite(os.path.join(face_path, f'{idx}.jpg'), cropped_face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    print(f'dlib: {time()-start}')
    # cv2.imwrite('save-directory/dlib_detection.png', img)
    # cv2.imshow('dlib Face Detection', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return frame, cropped_faces, faces

