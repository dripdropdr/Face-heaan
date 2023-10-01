from face_extractor.models.resnet import resnet_face18
from face_extractor.config import Config
from torch.nn import DataParallel
import torch
import cv2
import numpy as np


def initialize_face_extractor(DEVICE):
    opt = Config()
    resnet_face = resnet_face18(opt.use_se)
    resnet_face = DataParallel(resnet_face)
    resnet_face.load_state_dict(torch.load(opt.test_model_path, map_location=DEVICE))
    resnet_face.eval().to(DEVICE)
    
    return resnet_face

def preprocss_for_extractor(cropped_faces):
    face_img = cropped_faces[0]
    # Extract the face region from the frame and convert it to grayscale
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (125, 125), interpolation=cv2.INTER_CUBIC)
    # Create a stack of the face and its horizontally flipped version
    face_arr = np.dstack((face_img, np.fliplr(face_img))).transpose((2, 0, 1))
    # Add an extra dimension to match the expected input shape of the model
    face_arr = face_arr[:, np.newaxis, :, :]
    # Normalize the face data
    face_arr = face_arr.astype(np.float32, copy=False)
    face_arr -= 127.5
    face_arr /= 127.5

    return face_arr

def get_face_vector_from_extractor(model, face_arr, DEVICE):
        # Convert preprocessed data to torch tensor and move it to the specified device
        data = torch.from_numpy(face_arr)
        data = data.to(DEVICE)
        # Extract features using the model
        feat = model(data)
        feat = feat.detach().numpy()
        # Split the features into two halves
        fe_1 = feat[::2]
        fe_2 = feat[1::2]
        # Concatenate the feature halves
        face_vector = np.hstack((fe_1, fe_2))

        return face_vector
    