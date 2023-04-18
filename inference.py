# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
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

class FaceVerifying:
    def __init__(self, distance_method) -> None:
        self.distance_method = distance_method
        self.fe_dict = self.call_feature_dict()

    def call_feature_dict(self):
        fe_dict = {}
        # with open(): set feature value.
        return fe_dict

    def cosin_metric(self, x1, x2):
        # x1 = x1.transpose((1,0))
        x2 = x2.transpose((1,0))
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
    def cal_accuracy(self, y_score, y_true):
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        best_acc = 0
        best_th = 0
        for i in range(len(y_score)):
            th = y_score[i]
            y_test = (y_score >= th)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th
        return (best_acc, best_th)
    
    def verify_id(self, input, input2):
        # fe = self.fe_dict[id]
        fe = input2
        sim = 0 # init similarity
        if self.distance_method == 'cosine':
            sim = self.cosin_metric(input, fe)
            print(f'{sim}')
        elif self.distance_method == 'manhattan':
            pass
        elif self.distance_method == 'euclidean':
            pass

        # acc, th = self.cal_accuracy(sim) # 어떻게 구현할지 이야기해봐야 할 듯. 
        return sim


class FeatureProcessing:
    def __init__(self) -> None:
        self.face_detector = dlib.get_frontal_face_detector()

    def preproc(self, frame):
        # 얼굴이 잡히지 않을 경우도 가정해야 함.
        dets = self.face_detector(frame, 0)
        if len(dets) == 0:
            return 0
        elif len(dets) == 1:
            d = dets[0]
            face = frame[d.top():d.bottom(), d.left():d.right()] # face 차원 확인하기
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (125, 125), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('tmp.jpg', face)
            faces = np.dstack((face, np.fliplr(face))).transpose((2, 0, 1))
            faces = faces[:, np.newaxis, :, :]
            faces = faces.astype(np.float32, copy=False)
            faces -= 127.5
            faces /= 127.5
            return faces
        else:
            return -1

    def get_features(self, model, frame, opt, device):
        pre_res = self.preproc(frame)
        if type(pre_res) != int:
            data = torch.from_numpy(pre_res)
            data = data.to(device)
            feat = model(data)
            feat = feat.detach().numpy()
            fe_1 = feat[::2]
            fe_2 = feat[1::2]
            feature = np.hstack((fe_1, fe_2))
            # import sys
            # np.set_printoptions(threshold=sys.maxsize)
            # print(feature.tolist())
            return feature
        else:
            if pre_res == 0:
                # No face
                return 0
            else:
                # Many face
                return -1


def lfw_test(model, frame, opt, device):
    fe_proc = FeatureProcessing()
    fa_verify = FaceVerifying('cosine')

    feature1 = fe_proc.get_features(model, frame, opt, device)

    frame2 = cv2.imread('sum_face.jpg')
    feature2 = fe_proc.get_features(model, frame2, opt, device)
    if type(feature1) != int:
        sim = fa_verify.verify_id(feature1, feature2)
        # print('lfw face verification accuracy: ', acc, 'threshold: ', th)
        return sim
    else:
        if feature1 == 0:
            # No face
            return '0'
        else:
            # Many face
            return '-1'
    

# def main(frame):
    
#     return 


if __name__ == '__main__':

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

    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()
        if status:
            res = lfw_test(model, frame, opt, cpu)
            if res not in ['0', '-1']:
                # Verify result
                # frame = cv2.flip(frame, 1)
                # frame = cv2.rectangle(frame, (400,0), (510, 128), (0,255,0), 3)
                frame = cv2.putText(frame, "Unlock", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
            elif res == '-1':
                # Many face
                frame = cv2.putText(frame, "Too many face! ", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                # No face
                pass
            
            cv2.imshow("test", frame)

        if cv2.waitKey(1) == 32:
            break

    webcam.release()
    cv2.destroyAllWindows()


