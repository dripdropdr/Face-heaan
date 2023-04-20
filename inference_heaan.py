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

import piheaan as heaan
from piheaan.math import sort
from piheaan.math import approx # for piheaan math function
import math
import pandas as pd

class FaceVerifying:
    def __init__(self, distance_method) -> None:
        self.distance_method = distance_method


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
    
    

class FeatureProcessing:
    def __init__(self) -> None:
        self.face_detector = dlib.get_frontal_face_detector()

    def preproc(self, frame):
        dets = self.face_detector(frame, 0)
        img_dir_path = 'face_image/face_images'
        # 얼굴이 탐지되지 않을 경우 전처리 (평균 얼굴 이미지를 위해서)
        if len(dets) == 0:
            # return 0
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (125, 125), interpolation=cv2.INTER_CUBIC)
            faces = np.dstack((face, np.fliplr(face))).transpose((2, 0, 1))
            faces = faces[:, np.newaxis, :, :]
            faces = faces.astype(np.float32, copy=False)
            faces -= 127.5
            faces /= 127.5
            return faces
        # 얼굴이 탐지되었을 경우 전처리
        elif len(dets) == 1:
            d = dets[0]
            face = frame[d.top():d.bottom(), d.left():d.right()] # face 차원 확인하기
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (125, 125), interpolation=cv2.INTER_CUBIC)
            
            # 이미지 파일 경로 생성
            img_path = os.path.join(img_dir_path, "{}.jpg".format(time.time()))
            # 스페이스바 누르면 이미지 파일 저장
            if cv2.waitKey(1) == 32:
                cv2.imwrite(img_path, face)
                
            cv2.imwrite('tmp.jpg', face)
            faces = np.dstack((face, np.fliplr(face))).transpose((2, 0, 1))
            faces = faces[:, np.newaxis, :, :]
            faces = faces.astype(np.float32, copy=False)
            faces -= 127.5
            faces /= 127.5
            return faces
        else:
            return -1

    def get_features(self, model, frame, device):
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
                        
            # feature : (1,1024) shape
            # print("feature함수", feature[0])
            return feature[0]
        else:
            if pre_res == 0:
                # No face
                return 0
            else:
                # Many face
                return -1
            
            


def call_feature_dict():
        fe_dict = {}
        # with open(): set feature value.
        return fe_dict

def lfw_test(model, frame, device):
    feature = fe_proc.get_features(model, frame, device)
    if type(feature) != int:
        feature = np.squeeze(feature)
        # print('lfw face verification accuracy: ', acc, 'threshold: ', th)
        return feature
    else:
        if feature == 0:
            # No face
            return '0'
        else:
            # Many face
            return '-1'



## 현재 : 뺀 값을 approx sign을 돌림
## compare 함수쓰면 바로 가능
def cosin_sim(a,b,eval,enc,dec,sk,pk,log_slots,num_slots,context):
    
  # denominator
  msg1 = heaan.Message(log_slots)
  msg2 = heaan.Message(log_slots)
  for i in range(num_slots):
    msg1[i] = a[i]
    msg2[i] = b[i]

  # mult 
  ctxt1 = heaan.Ciphertext(context)
  ctxt2 = heaan.Ciphertext(context)
  ctxt3 = heaan.Ciphertext(context)

  enc.encrypt(msg1, pk, ctxt1)
  enc.encrypt(msg2, pk, ctxt2)
  eval.mult(ctxt1, ctxt2, ctxt3)

  # sigma
  denom_ctxt = heaan.Ciphertext(context)
  eval.left_rotate_reduce(ctxt3,1,num_slots,denom_ctxt)
 
  # numerator
  
  ## how to calculate without sqrt?

  # square
  ctxt1_sqr = heaan.Ciphertext(context)
  eval.square(ctxt1, ctxt1_sqr)

  ctxt2_sqr = heaan.Ciphertext(context)
  eval.square(ctxt2, ctxt2_sqr)

  # sigma
  ctxt1_rot = heaan.Ciphertext(context)
  eval.left_rotate_reduce(ctxt1_sqr,1,num_slots,ctxt1_rot)

  ctxt2_rot = heaan.Ciphertext(context)
  eval.left_rotate_reduce(ctxt2_sqr,1,num_slots,ctxt2_rot)

  # sqrt
  ## sigma 결과값 범위 : 대략 10 ~ 30
  ## divide by 100 and mult 10 to later result value
  ## input range : 2^-18 ≤ x ≤ 2

  hun_msg = heaan.Message(log_slots)
  for i in range(num_slots):
    hun_msg[i] = 0.01

  eval.mult(ctxt1_rot,hun_msg,ctxt1_rot)

  eval.mult(ctxt2_rot,hun_msg,ctxt2_rot)

  ctxt1_sqrt = heaan.Ciphertext(context)
  approx.sqrt(eval,ctxt1_rot,ctxt1_sqrt)

  ctxt2_sqrt = heaan.Ciphertext(context)
  approx.sqrt(eval,ctxt2_rot,ctxt2_sqrt)

  # mult and inverse 

  ## inverse range : 1 ≤ x ≤ 2^22 or 2^-10 ≤ x ≤ 1
  num_ctxt = heaan.Ciphertext(context)
  eval.mult(ctxt1_sqrt, ctxt2_sqrt, num_ctxt)

  eval.mult(num_ctxt,1000,num_ctxt)

  num_inverse = heaan.Ciphertext(context)
  approx.inverse(eval,num_ctxt,num_inverse)

  eval.mult(num_inverse,10, num_inverse)

  eval.bootstrap(num_inverse, num_inverse)

  # cosine similarity

  # mult denominator & numberator^-1
  res_ctxt = heaan.Ciphertext(context)
  eval.mult(num_inverse,denom_ctxt,res_ctxt)

  return res_ctxt


def compare(type,thres,comp_ctxt,eval,enc,dec,sk,pk,log_slots,num_slots,context):
  thres_list = []
  thres_list.append(thres)

  thres_list += (num_slots-len(thres_list))*[0]

  thres_msg = heaan.Message(log_slots)
  for i in range(num_slots):
    thres_msg[i] = thres_list[i]

  sub_ctxt = heaan.Ciphertext(context)
  if type == 'cosine':
    eval.sub(comp_ctxt,thres_msg,sub_ctxt)
  elif type == 'euclidean' or 'manhattan':
    thres_ctxt = heaan.Ciphertext(context)
    enc.encrypt(thres_msg, pk, thres_ctxt)
    eval.sub(thres_ctxt,comp_ctxt,sub_ctxt)
  ## cos_similarity - threshold > 0 이면 그 값을 1로 (그냥 그 뺀값도 숨겨버려려)

  sign_ctxt = heaan.Ciphertext(context)
  approx.sign(eval, sub_ctxt, sign_ctxt)

  res = heaan.Message(log_slots)
  dec.decrypt(sign_ctxt, sk, res)

  real = res[0].real
  if -0.0001 < 1-real < 0.0001:
    res = 'unlock'
  else:
    res = 'lock'

  return res
    
    



if __name__ == '__main__':

    # set parameter
    params = heaan.ParameterPreset.FGb
    context = heaan.make_context(params) # context has paramter information
    heaan.make_bootstrappable(context) # make parameter bootstrapable

    # create and save keys
    key_file_path = "./keys"
    sk = heaan.SecretKey(context) # create secret key
    os.makedirs(key_file_path, mode=0o775, exist_ok=True)
    sk.save(key_file_path+"/secretkey.bin") # save secret key

    key_generator = heaan.KeyGenerator(context, sk) # create public key
    key_generator.gen_common_keys()
    key_generator.save(key_file_path+"/") # save public key
    
    # load secret key and public key
    # When a key is created, it can be used again to save a new key without creating a new one
    key_file_path = "./keys"

    sk = heaan.SecretKey(context,key_file_path+"/secretkey.bin") # load secret key
    pk = heaan.KeyPack(context, key_file_path+"/") # load public key
    pk.load_enc_key()
    pk.load_mult_key()

    eval = heaan.HomEvaluator(context,pk) # to load piheaan basic function
    dec = heaan.Decryptor(context) # for decrypt
    enc = heaan.Encryptor(context) # for encrypt
    
    # log_slots is used for the number of slots per ciphertext
    # It depends on the parameter used (ParameterPreset)
    # The number '15' is the value for maximum number of slots,
    # but you can also use a smaller number (ex. 2, 3, 5, 7 ...)
    # The actual number of slots in the ciphertext is calculated as below.
    log_slots = 15
    num_slots = 2**log_slots


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
    fa_verify = FaceVerifying('cosine')

    # DB사진 : 얼굴 이미지 평균 registration 전 이미지
    frame2 = cv2.imread('face_image/average_image/default_image.jpg')
    feature2 = fe_proc.get_features(model, frame2, cpu)
    feature2 = np.squeeze(feature2)
    
    
    
    # (1024,)
    # print(feature2.shape)
    
    # DB 사진
    b = feature2.tolist()
    b = b + (num_slots-len(b))*[0]
    
    avg_dir_path = 'face_image/average_image'
    img_dir_path = 'face_image/face_images'
    
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        status, frame = webcam.read()
        if status:
            res = lfw_test(model, frame, cpu)
            
            if len(os.listdir(img_dir_path)) < 5: 
                if len(os.listdir(img_dir_path)) == 0:
                    frame = cv2.putText(frame, "User Registration!", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                    frame = cv2.putText(frame, "Look front!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                elif len(os.listdir(img_dir_path)) == 1:
                    frame = cv2.putText(frame, "Head up!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                elif len(os.listdir(img_dir_path)) == 2:
                    frame = cv2.putText(frame, "Eyes on left!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                elif len(os.listdir(img_dir_path)) == 3:
                    frame = cv2.putText(frame, "Eyes on right!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                elif len(os.listdir(img_dir_path)) == 4:
                    frame = cv2.putText(frame, "Look front again!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                    
            
            
            elif res not in ['0', '-1']:
                
                img_paths = [os.path.join(img_dir_path, filename) for filename in os.listdir(img_dir_path)]
                
                images = []
                for img_path in img_paths:
                    image = cv2.imread(img_path)
                    if image is not None:  # 이미지가 제대로 읽혔을 때만 리스트에 추가
                        image_resized = cv2.resize(image, (125, 125))  # 이미지를 500x500 크기로 리사이즈
                        images.append(image_resized)

                # 이미지들의 평균 계산
                average_image = np.mean(images, axis=0).astype(np.uint8)

                # 평균 이미지를 파일로 저장
                cv2.imwrite('face_image/average_image/average.jpg', average_image)
                
                frame2 = cv2.imread('face_image/average_image/average.jpg')
                feature2 = fe_proc.get_features(model, frame2, cpu)
                feature2 = np.squeeze(feature2)   
                
                b = feature2.tolist()
                b = b + (num_slots-len(b))*[0]
                
                # res => 리스트
                a = res.tolist()
                a = a + (num_slots-len(a))*[0]
                res_ctxt = cosin_sim(a,b,eval,enc,dec,sk,pk,log_slots,num_slots,context)

                sim = heaan.Message(log_slots)
                dec.decrypt(res_ctxt, sk, sim)
                # 얼굴 인식될때마다 cosine similarity 출력
                # print("cosine similarity : ", sim)
            
                # threshold
                thres = 0.2
                # text 선언하면 에러뜸! ex) type = 'cosine'
                result = compare('cosine',thres,res_ctxt,eval,enc,dec,sk,pk,log_slots,num_slots,context)
                if result == "unlock":
                    frame = cv2.putText(frame, "Unlock", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                
            elif res == '-1':
                # Many face
                frame = cv2.putText(frame, "Too many face! ", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                # No face
                pass
            
            cv2.imshow("test", frame)

        # if cv2.waitKey(1) == 32:
        #     break
    


    webcam.release()
    cv2.destroyAllWindows()

