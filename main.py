# -*- coding: utf-8 -*-
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import cv2
import dlib
import torch
import numpy as np
from emotracker.run_inference import initialize_emotracker, preprocess_images
from face_extractor.face_extractor import initialize_face_extractor, preprocss_for_extractor, get_face_vector_from_extractor
from face_detector import detect_faces_with_dlib
import piheaan as heaan
from heaan_utils import Heaan
import pandas as pd

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


he = Heaan()
ctxt1, ctxt2 = he.heaan_initilize()

# face detector initialization.
face_detector = dlib.get_frontal_face_detector()

# face vector extractor initialization.
face_extractor = initialize_face_extractor(DEVICE)

# emotracker initialization.
emotracker = initialize_emotracker(DEVICE)

# Set the threshold values
cos_thres = 0.75572848

register_feat = np.array([])
face_verified = False
smile_verified = False
verification_start = False


if __name__ == '__main__':
    
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        # Read the frame from the webcam
        success, frame = webcam.read()
        if success:
            # Extract design features from the frame
            h, w, = frame.shape[:2]
            font_state = (int(w/2), int(h*0.8))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame, cropped_faces, faces = detect_faces_with_dlib(face_detector, frame, gray) # bounding box 그려줌

            if len(cropped_faces) == 1:
            
                # User registeration part
                if register_feat.shape[0] <= 4:
                    # Display user registration instructions on the frame
                    (text_width, text_height) = cv2.getTextSize("User Registration Phase. Press space", cv2.FONT_HERSHEY_PLAIN, 3, 3)[0]
                    text_offset_x = (w - text_width) // 2
                    text_offset_y = (h + text_height) // 8
                    frame = cv2.putText(frame, "User Registration Phase. Press space", (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)

                    # Display specific instructions based on the number of registered features
                    if register_feat.shape[0] == 0:
                        frame = cv2.putText(frame, "Look front!", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
                    elif register_feat.shape[0] == 1:
                        frame = cv2.putText(frame, "Head up!", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
                    elif register_feat.shape[0] == 2:
                        frame = cv2.putText(frame, "Eyes on left!", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
                    elif register_feat.shape[0] == 3:
                        frame = cv2.putText(frame, "Eyes on right!", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
                    elif register_feat.shape[0] == 4: 
                        frame = cv2.putText(frame, "Look front again!", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)

                    face_arr = preprocss_for_extractor(cropped_faces)
                    face_vector = get_face_vector_from_extractor(face_extractor, face_arr, DEVICE)

                    # Register the feature if it is an array and the space bar is pressed
                    if isinstance(face_vector, np.ndarray) and cv2.waitKey(100) == 32:
                        frame = cv2.putText(frame, "registered", (h-20, w-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3, cv2.LINE_AA)
                        if register_feat.size == 0:
                            register_feat = face_vector
                        else:
                            register_feat = np.concatenate([register_feat, face_vector]) # stack
                            if register_feat.shape[0] == 5:
                                avg_feat = np.mean(register_feat, axis=0)
                                msg1 = he.feat_msg_generate(avg_feat)
                                he.encrypt(msg1, ctxt1)
                
                # After Registration
                else:
                    if not verification_start:
                        print('Before')
                        frame = cv2.putText(frame, "Click space if you want to verify", (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
                        if cv2.waitKey(100) == 32:
                            verification_start = True
                    else:
                        face_arr = preprocss_for_extractor(cropped_faces)
                        face_vector = get_face_vector_from_extractor(face_extractor, face_arr, DEVICE)
                        
                        msg2 = he.feat_msg_generate(np.squeeze(face_vector))
                        he.encrypt(msg2, ctxt2)
                        
                        # 1) cosine similarity measurement
                        res_ctxt = he.cosin_sim(ctxt1, ctxt2)
                        face_verified = True if he.compare('cosine', cos_thres, res_ctxt) == 'unlock' else False
                        print('verificaiton', face_verified)
                        
                        # Unverified !!
                        if not face_verified:
                            frame = cv2.putText(frame, "Lock", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)

                        if face_verified and not smile_verified:
                            frame = cv2.putText(frame, "Smile !!", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
                            sample = preprocess_images(cropped_faces, emotracker)
                            estimates, outputs = emotracker.forward(input_image = sample['image'])
                            smile_verified = True if int(estimates['EXPR']) == 4 else False

                        # Verified !!       
                        if face_verified and smile_verified:
                            frame = cv2.putText(frame, "Unlock", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)                    
            
            # Too many faces or No faces...
            else:
                face_verified = False

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) == 27:
            break
    
    webcam.release()
    cv2.destroyAllWindows()