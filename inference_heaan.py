# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel
import dlib
import piheaan as heaan
from heaan_utils import Heaan
import pandas as pd
import statistics



class FeatureProcessing:
    def __init__(self) -> None:
        # Initialize the face detector from dlib
        self.face_detector = dlib.get_frontal_face_detector()
        self.detects = None

    def get_features(self, model, frame, device):
        # Preprocess the frame to obtain faces and compute features
        pre_res = self.preproc(frame)
        if type(pre_res) != int:
            # Convert preprocessed data to torch tensor and move it to the specified device
            data = torch.from_numpy(pre_res)
            data = data.to(device)
            # Extract features using the model
            feat = model(data)
            feat = feat.detach().numpy()
            # Split the features into two halves
            fe_1 = feat[::2]
            fe_2 = feat[1::2]
            # Concatenate the feature halves
            feature = np.hstack((fe_1, fe_2))
            return feature
        else:
            if pre_res == 0:    return 0 # No face
            else:   return -1 # Many face

    def preproc(self, frame):
        # Detect faces in the frame using the face detector
        dets = self.face_detector(frame, 0)
        # when face not detected
        if len(dets) == 0:
            return 0
        # when face detected
        elif len(dets) == 1:
            self.detects = dets[0]
            # Extract the face region from the frame and convert it to grayscale
            face = frame[self.detects.top():self.detects.bottom(), self.detects.left():self.detects.right()]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # Resize the face to a fixed size
            face = cv2.resize(face, (125, 125), interpolation=cv2.INTER_CUBIC)
            # Create a stack of the face and its horizontally flipped version
            faces = np.dstack((face, np.fliplr(face))).transpose((2, 0, 1))
            # Add an extra dimension to match the expected input shape of the model
            faces = faces[:, np.newaxis, :, :]
            # Normalize the face data
            faces = faces.astype(np.float32, copy=False)
            faces -= 127.5
            faces /= 127.5
            return faces
        else:
            return -1


if __name__ == '__main__':

    # log_slots is used for the number of slots per ciphertext
    # It depends on the parameter used (ParameterPreset)
    # The number '15' is the value for maximum number of slots,
    # but you can also use a smaller number (ex. 2, 3, 5, 7 ...)
    # The actual number of slots in the ciphertext is calculated as below.
    he = Heaan()
    ctxt1, ctxt2 = he.heaan_initilize()

    # Select the appropriate model based on the specified backbone
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    cpu = torch.device("cpu")
    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path, map_location=torch.device('cpu')))
    model.to(cpu)
    model.eval()

    fe_proc = FeatureProcessing()
    
    # Set the threshold values
    cos_thres = opt.cosine_thres
    euc_thres = opt.euc_thres
    man_thres = opt.man_thres

    register_feat = np.array([])
    
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    # List for storing execution time
    time_spent = []
    
    while webcam.isOpened():
        # Read the frame from the webcam
        status, frame = webcam.read()
        if status:
            # Extract features from the frame
            feature = fe_proc.get_features(model, frame, cpu)
            h, w = frame.shape[:2]
            font_state = (int(w/2), int(h*0.8))
            
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

                # Register the feature if it is an array and the space bar is pressed
                if isinstance(feature, np.ndarray) and cv2.waitKey(100) == 32:
                    frame = cv2.putText(frame, "registered", (h-20, w-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3, cv2.LINE_AA)
                    if register_feat.size == 0:
                        register_feat = feature
                    else:
                        register_feat = np.concatenate([register_feat, feature]) # stack
                        if register_feat.shape[0] == 5:
                            avg_feat = np.mean(register_feat, axis=0)
                            msg1 = he.feat_msg_generate(avg_feat)
                            he.encrypt(msg1, ctxt1)
            
            else:
                # Start time of inference
                start = time.time()
                if isinstance(feature, np.ndarray):
                    input_feat = np.squeeze(feature)
                    msg2 = he.feat_msg_generate(input_feat)
                    he.encrypt(msg2, ctxt2)
                    
                    # 1) cosine similarity measurement
                    res_ctxt = he.cosin_sim(ctxt1, ctxt2)
                    result = he.compare('cosine', cos_thres, res_ctxt)
                    
                    # # 2) euclidean distance measurement
                    # res_ctxt = he.euclidean_distance(ctxt1, ctxt2)
                    # result = he.compare('euclidean', euc_thres, res_ctxt)
                    
                    # # 3) manhattan distance measurement
                    # res_ctxt = he.manhattan_distance(ctxt1, ctxt2)
                    # result = he.compare('manhattan', man_thres, res_ctxt)
                    
                    #print similarity
                    # print(he.similarity_calc(res_ctxt), result)
                    
                    # Display the result on the frame
                    if result == "unlock":
                        face = fe_proc.detects
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        frame = cv2.putText(frame, "Unlock", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3, cv2.LINE_AA)
                    else:
                        frame = cv2.putText(frame, "Lock", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
                    
                elif isinstance(feature, int) and feature == -1:
                    # Many face
                    frame = cv2.putText(frame, "Too many face! ", font_state, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    # No face
                    pass
                
                # End time of inference
                end = time.time()
                # Calculate execution time
                # Unit of execution time is second
                end_to_start = end - start
                # print(f"{end_to_start} sec")
                # Store execution time per frame
                time_spent.append(end_to_start)
                
                # # Used 10000 frames to obtain of average and standard deviation execution time
                # if len(time_spent) == 10000:
                #     average = statistics.mean(time_spent)
                #     standard_deviation = statistics.stdev(time_spent)
                #     print("Average : ", average)
                #     print("Standard Deviation : ", standard_deviation)
                #     break;
                
            
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) == 27:
            break
    
    webcam.release()
    cv2.destroyAllWindows()