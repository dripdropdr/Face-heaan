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
import glob


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
        # when face not detected
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
        # when face detected
        elif len(dets) == 1:
            d = dets[0]
            face = frame[d.top():d.bottom(), d.left():d.right()] # face 차원 확인하기
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (125, 125), interpolation=cv2.INTER_CUBIC)
            # generating image path
            img_path = os.path.join(img_dir_path, "{}.jpg".format(time.time()))
            # storing image when pressing space keyboard
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



def cosin_sim(ctxt_path,ctxt2,eval,enc,dec,sk,pk,log_slots,num_slots,context):

    # denominator
    ctxt1 = heaan.Ciphertext(context)
    ctxt1.load(ctxt_path)
    # mult 
    ctxt3 = heaan.Ciphertext(context)
    eval.mult(ctxt1, ctxt2, ctxt3)

    # sigma
    denom_ctxt = heaan.Ciphertext(context)
    eval.left_rotate_reduce(ctxt3,1,num_slots,denom_ctxt)

    # numerator

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
    ## sigma output range : about 10 ~ 30
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

def euclidean_distance(ctxt_path,ctxt2,eval,enc,dec,sk,pk,log_slots,num_slots,context):

    # sub
    ctxt1 = heaan.Ciphertext(context)
    ctxt1.load(ctxt_path)

    ctxt3 = heaan.Ciphertext(context)
    eval.sub(ctxt1, ctxt2, ctxt3)

    # square
    ctxt_square = heaan.Ciphertext(context)
    eval.square(ctxt3, ctxt_square)

    # sigma
    ctxt_sig = heaan.Ciphertext(context)
    eval.left_rotate_reduce(ctxt_square,1,num_slots,ctxt_sig)

    # sqrt
    ## ctxt_sig is bigger than 2
    ## input range : 2^-18 ≤ x ≤ 2

    eval.mult(ctxt_sig,0.01,ctxt_sig)
    ctxt_sqrt = heaan.Ciphertext(context)
    approx.sqrt(eval,ctxt_sig,ctxt_sqrt)
    eval.mult(ctxt_sqrt,10,ctxt_sqrt)

    return ctxt_sqrt

def manhattan_distance(ctxt_path,ctxt2,eval,enc,dec,sk,pk,log_slots,num_slots,context):
  
    small_tmp_ctxt= heaan.Ciphertext(context)
    small_ctxt = heaan.Ciphertext(context)
    big_tmp_ctxt = heaan.Ciphertext(context)
    big_ctxt = heaan.Ciphertext(context)
    abs_ctxt = heaan.Ciphertext(context)
    res_ctxt = heaan.Ciphertext(context)
    ctxt3 = heaan.Ciphertext(context)
    ctxt1 = heaan.Ciphertext(context)
    ctxt1.load(ctxt_path)


    ## if ctxt1 < ctxt2 -> 0
    comp_ctxt = heaan.Ciphertext(context)
    approx.compare(eval, ctxt1, ctxt2, comp_ctxt)

    ## discrete equal zero 
    ## input range : |x| ≤ 54 (x : int)
    discrete_ctxt = heaan.Ciphertext(context)
    two_msg = heaan.Message(log_slots)
    for i in range(num_slots):
        two_msg[i] = 2
    two_ctxt = heaan.Ciphertext(context)
    enc.encrypt(two_msg,pk,two_ctxt)

    comp_tmp_ctxt = heaan.Ciphertext(context)
    eval.mult(two_ctxt,comp_ctxt,comp_tmp_ctxt)
    approx.discrete_equal_zero(eval, comp_tmp_ctxt, discrete_ctxt)

    # sub
    eval.sub(ctxt1, ctxt2, ctxt3)

    # small_tmp_ctxt = remain only minus values
    eval.mult(ctxt3,discrete_ctxt,small_tmp_ctxt)
    # small_ctxt = - to +
    eval.negate(small_tmp_ctxt,small_ctxt)

    one_msg = heaan.Message(log_slots)
    for i in range(num_slots):
        one_msg[i] = 1
    one_ctxt = heaan.Ciphertext(context)
    enc.encrypt(one_msg, pk, one_ctxt)

    eval.sub(one_ctxt,discrete_ctxt,big_tmp_ctxt)
    eval.mult(big_tmp_ctxt,ctxt3,big_ctxt)
    eval.add(big_ctxt,small_ctxt,abs_ctxt)

    ## sigma
    eval.left_rotate_reduce(abs_ctxt,1,num_slots,res_ctxt)

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
    
  ## cos_similarity - threshold > 0 ==> 1

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

    # DB image : before registration. default image
    frame2 = cv2.imread('face_image/average_image/default_image.jpg')
    feature2 = fe_proc.get_features(model, frame2, cpu)
    feature2 = np.squeeze(feature2)
    
    avg_dir_path = 'face_image/average_image'
    img_dir_path = 'face_image/face_images'
    ctxt_path = 'face_image/average_ctxt/ctxt1.ctxt'
    
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
                    if image is not None:  # adding to list when image is recognized properly
                        image_resized = cv2.resize(image, (125, 125))  # image 125*125 resize
                        images.append(image_resized)

                # average of images
                average_image = np.mean(images, axis=0).astype(np.uint8)

                # storing average face image
                cv2.imwrite('face_image/average_image/average.jpg', average_image)
                
                frame2 = cv2.imread('face_image/average_image/average.jpg')
                feature2 = fe_proc.get_features(model, frame2, cpu)
                feature2 = np.squeeze(feature2)   
                
                # avg : average face
                avg = feature2.tolist()
                avg = avg + (num_slots-len(avg))*[0]
                # real : real time face
                real = res.tolist()
                real = real + (num_slots-len(real))*[0]
                
                msg2 = heaan.Message(log_slots)
                msg1 = heaan.Message(log_slots)
                for i in range(num_slots):
                    msg1[i] = avg[i]
                    msg2[i] = real[i]
                    
                ctxt1 = heaan.Ciphertext(context)
                ctxt2 = heaan.Ciphertext(context)
                
                enc.encrypt(msg1, pk, ctxt1)
                enc.encrypt(msg2, pk, ctxt2)
                ctxt1.save('face_image/average_ctxt/ctxt1.ctxt')
                
                # threshold
                thres = 0.5
                
                # 1) cosine similarity measurement
                res_ctxt = cosin_sim(ctxt_path,ctxt2,eval,enc,dec,sk,pk,log_slots,num_slots,context)
                result = compare('cosine',thres,res_ctxt,eval,enc,dec,sk,pk,log_slots,num_slots,context)
                
                # # 2) euclidean distance measurement
                # res_ctxt = euclidean_distance(ctxt_path,ctxt2,eval,enc,dec,sk,pk,log_slots,num_slots,context)
                # result = compare('euclidean',thres,res_ctxt,eval,enc,dec,sk,pk,log_slots,num_slots,context)
                
                # # 3) manhattan distance measurement
                # res_ctxt = manhattan_distance(ctxt_path,ctxt2,eval,enc,dec,sk,pk,log_slots,num_slots,context)
                # result = compare('manhattan',thres,res_ctxt,eval,enc,dec,sk,pk,log_slots,num_slots,context)
                
                # similarity
                sim = heaan.Message(log_slots)
                dec.decrypt(res_ctxt, sk, sim)
                print(sim)
            
                
                if result == "unlock":
                    frame = cv2.putText(frame, "Unlock", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    frame = cv2.putText(frame, "Lock", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                
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
