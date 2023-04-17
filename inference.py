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
    
    def verify_id(self, input):
        fe = self.fe_dict[id]
        sim = 0 # init similarity
        if self.distance_method == 'cosine':
            sim = self.cosin_metric(input, fe)
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
        d = self.face_detector(frame, 0)
        face = frame[d.top():d.bottom(), d.left():d.right()] # face 차원 확인하기
        face = cv2.resize(face, (250, 250), interpolation=cv2.INTER_CUBIC)
        face = face[np.newaxis, :, :]
        face = face.astype(np.float32, copy=False)
        face -= 127.5
        face /= 127.5
        return face

    def get_features(self, model, frame, opt, device):
        img = self.preproc(frame)
        data = torch.from_numpy(img)
        data = data.to(device)
        output = model(data)
        # output = output.data.cpu().numpy()
        # check the fe_1, fe_2 output.
        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))
        return feature


def lfw_test(model, frame, opt, device):
    fe_proc = FeatureProcessing()
    fa_verify = FaceVerifying('cosine')

    feature = fe_proc.get_features(model, frame, opt, device)
    sim = fa_verify.verify_id(feature) 
    # print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return sim


def main(frame):
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

    return lfw_test(model, frame, opt, cpu)




