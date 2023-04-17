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


# def get_lfw_list(pair_list):
#     with open(pair_list, 'r') as fd:
#         pairs = fd.readlines()
#     data_list = []
#     for pair in pairs:
#         splits = pair.split()

#         if splits[0] not in data_list:
#             data_list.append(splits[0])

#         if splits[1] not in data_list:
#             data_list.append(splits[1])
#     return data_list

# def load_model(model, model_path):
#     model_dict = model.state_dict()
#     pretrained_dict = torch.load(model_path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
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


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def load_image(img_path):
    img = cv2.imread(img_path, 0) # 컬러는 상관 없어서 ?
    if img is None:
        raise 'read {} error'.format(img_path)
    img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
    img = np.dstack((img, np.fliplr(img)))
    img = img.transpose((2, 0, 1))
    img = img[:, np.newaxis, :, :]
    img = img.astype(np.float32, copy=False)
    img -= 127.5
    img /= 127.5
    return img


def get_features(model, img_path, opt, device):

    img = load_image(img_path)
    cnt += 1

    data = torch.from_numpy(img)
    data = data.to(device)
    output = model(data)
    # output = output.data.cpu().numpy()

    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))

    return feature


def lfw_test(model, img_path, opt, device):
    s = time.time()
    features = get_features(model, img_paths, opt, device)
    print(features.shape)
    fe_dict = get_feature_dict(identity_list, features)
    # acc, th = test_performance(fe_dict, compair_list)
    # print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return fe_dict


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

    img_paths = [os.path.join(opt.lfw_root, each) for each in os.listdir('face_img')]

    cnt = 0
    for i, img_path in enumerate(img_paths):
        lfw_test(model, img_path, opt, cpu)




