# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55
Modified on 23-5-19

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

# Reads the LFW pair list file and returns a list of image paths.
def get_lfw_list(pair_list):
    # Open the LFW pair list file in read mode
    with open(pair_list, 'r') as fd:
        # Read all the lines in the file
        pairs = fd.readlines()
    # Initialize an empty list to store unique data entries
    data_list = []
    # Iterate over each line (pair) in the pair list
    for pair in pairs:
        # Split the line into individual elements
        splits = pair.split()

        # Check if the first element of the pair is not already in the data list
        if splits[0] not in data_list:
            # Add the first element to the data list
            data_list.append(splits[0])

        # Check if the second element of the pair is not already in the data list
        if splits[1] not in data_list:
            # Add the second element to the data list
            data_list.append(splits[1])
    # Return the list of unique data entries
    return data_list


# Loads and preprocesses an image from the given image path.
def load_image(img_path):
    # Read the image from the given image path using OpenCV (grayscale mode)
    image = cv2.imread(img_path, 0)
    # Check if the image is not loaded successfully
    if image is None:
        return None
    # Stack the original image and the flipped image along the depth axis to create a 3-channel image
    image = np.dstack((image, np.fliplr(image)))
    # Transpose the image dimensions to match the expected input format of the model (channels-first)
    image = image.transpose((2, 0, 1))
    # Add an additional dimension to represent the batch size (set to 1)
    image = image[:, np.newaxis, :, :]
    # Convert the image data type to float32 and perform in-place normalization
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    # Return the preprocessed image
    return image


#  Extracts features from a list of images using the given model.
def get_features(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        # Load the image from the image path
        image = load_image(img_path)
        # Check if the image loading was unsuccessful
        if image is None:
            print('read {} error'.format(img_path))

        # Concatenate the loaded image with the existing images
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        # Check if the batch size condition is met or if it is the last image
        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            # Convert the image data to a PyTorch tensor and move it to the CPU device
            data = torch.from_numpy(images)
            data = data.to(torch.device("cpu"))
            
            # Perform forward pass on the model to obtain the output features
            output = model(data)
            output = output.data.numpy()

            # Split the output features into two halves
            fe_1 = output[::2]
            fe_2 = output[1::2]
            
            # Concatenate the two halves of features horizontally
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            # Concatenate the extracted feature with the existing features
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            # Reset the images variable for the next batch
            images = None

    return features, cnt


# Loads the pretrained weights of the model from the given path.
def load_model(model, model_path):
    # Get the state dictionary of the model
    model_dict = model.state_dict()
    # Load the pretrained weights from the specified model path
    pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Filter the pretrained weights dictionary to match the keys in the model state dictionary
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # Update the model state dictionary with the filtered pretrained weights
    model_dict.update(pretrained_dict)
    # Load the updated state dictionary into the model
    model.load_state_dict(model_dict)


# Creates a dictionary mapping image paths to their corresponding features.
def get_feature_dict(test_list, features):
    # Initialize an empty dictionary
    fe_dict = {}
    # Iterate over the test_list and features in parallel using enumerate
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        # Assign the i-th feature to the corresponding image path in the dictionary
        fe_dict[each] = features[i]
    # Return the dictionary mapping image paths to features
    return fe_dict


# Calculates the cosine similarity between two vectors.
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# Calculates the Euclidean distance between two vectors.
def euclidean_metric(x1, x2):
    return np.linalg.norm(x1 - x2)

# Calculates the Manhattan distance between two vectors.
def manhattan_metric(x1, x2):
    return np.sum(np.abs(x1 - x2))


# Calculates the accuracy and threshold for a given set of scores and ground truth labels.
def cal_accuracy(y_score, y_true):
    # Convert the scores and labels to NumPy arrays
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    # Initialize variables to store the best accuracy and corresponding threshold
    best_acc = 0
    best_th = 0
    # Iterate over the scores
    for i in range(len(y_score)):
        # Get the threshold value for the current iteration
        th = y_score[i]
        # Generate binary predictions based on the threshold
        y_test = (y_score >= th)
        # Calculate the accuracy by comparing the predictions with the ground truth labels
        acc = np.mean((y_test == y_true).astype(int))
        # Check if the current accuracy is better than the previous best accuracy
        if acc > best_acc:
            best_acc = acc
            best_th = th

    # Return the best accuracy and corresponding threshold as a tuple
    return (best_acc, best_th)


# Tests the performance of the cosine similarity metric on the LFW dataset.
def test_performance_cos(fe_dict, pair_list):
    # Open the pair list file
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    # Initialize lists to store similarity scores and labels
    sims = []
    labels = []
    
    # Iterate over pairs in the pair list
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]] # Get the feature vector for the first image
        fe_2 = fe_dict[splits[1]] # Get the feature vector for the second image
        label = int(splits[2]) # Get the ground truth label
        sim = cosin_metric(fe_1, fe_2) # Calculate the cosine similarity between the feature vectors

        # Append the similarity score and label to the respective lists
        sims.append(sim)
        labels.append(label)

    # Calculate the accuracy and threshold using the similarity scores and labels
    acc, th = cal_accuracy(sims, labels)
    # Return the accuracy and threshold as a tuple
    return acc, th


# Tests the performance of the Euclidean distance metric on the LFW dataset.
def test_performance_euc(fe_dict, pair_list):
    # Open the pair list file
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    # Initialize lists to store similarity scores and labels
    sims = []
    labels = []
    
    # Iterate over pairs in the pair list
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]] # Get the feature vector for the first image
        fe_2 = fe_dict[splits[1]] # Get the feature vector for the second image
        label = int(splits[2]) # Get the ground truth label
        sim = euclidean_metric(fe_1, fe_2) # Calculate the euclidean similarity between the feature vectors

        # Append the similarity score and label to the respective lists
        sims.append(sim)
        labels.append(label)
        
    # Calculate the accuracy and threshold using the similarity scores and labels
    acc, th = cal_accuracy(sims, labels)
    # Return the accuracy and threshold as a tuple
    return acc, th


# Tests the performance of the Manhattan distance metric on the LFW dataset.
def test_performance_man(fe_dict, pair_list):
    # Open the pair list file
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    # Initialize lists to store similarity scores and labels
    sims = []
    labels = []
    
    # Iterate over pairs in the pair list
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]] # Get the feature vector for the first image
        fe_2 = fe_dict[splits[1]] # Get the feature vector for the second image
        label = int(splits[2]) # Get the ground truth label
        sim = manhattan_metric(fe_1, fe_2) # Calculate the manhattan similarity between the feature vectors

        # Append the similarity score and label to the respective lists
        sims.append(sim)
        labels.append(label)
        
    # Calculate the accuracy and threshold using the similarity scores and labels
    acc, th = cal_accuracy(sims, labels)
    # Return the accuracy and threshold as a tuple
    return acc, th


# Performs LFW face verification using the cosine similarity metric.
def lfw_test_cos(model, img_paths, identity_list, compare_list, batch_size):
    s = time.time()
    features, cnt = get_features(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance_cos(fe_dict, compare_list)
    print('lfw face verification accuracy:', acc, 'threshold:', th)
    return acc, th


# Performs LFW face verification using the Euclidean distance metric.
def lfw_test_euc(model, img_paths, identity_list, compare_list, batch_size):
    s = time.time()
    features, cnt = get_features(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance_euc(fe_dict, compare_list)
    print('lfw face verification accuracy:', acc, 'threshold:', th)
    return acc, th


# Performs LFW face verification using the Manhattan distance metric.
def lfw_test_man(model, img_paths, identity_list, compare_list, batch_size):
    s = time.time()
    features, cnt = get_features(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance_man(fe_dict, compare_list)
    print('lfw face verification accuracy:', acc, 'threshold:', th)
    return acc, th





if __name__ == '__main__':
    # Create an instance of the Config class to store configuration parameters.
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    # Based on the specified backbone in the configuration, create an instance of the corresponding face recognition model.
    # Wrap the model with DataParallel for parallel processing (if supported).
    model = DataParallel(model)
    # Load the pretrained weights of the model from the specified model path.
    load_model(model, opt.test_model_path)
    # Move the model to the CPU device.
    model.to(torch.device("cpu"))

    # Get the list of identity names from the LFW pair list.
    identity_list = get_lfw_list(opt.lfw_test_list)
    # Generate the list of image paths by concatenating the LFW root path with each identity name.
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]
    
    # Set the model to evaluation mode.
    model.eval()
    # Perform LFW face verification using the cosine similarity metric and print the accuracy and threshold.
    cos_acc, cos_th = lfw_test_cos(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    # Perform LFW face verification using the Euclidean distance metric and print the accuracy and threshold.
    euc_acc, euc_th = lfw_test_euc(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    # Perform LFW face verification using the Manhattan distance metric and print the accuracy and threshold.
    man_acc, man_th = lfw_test_man(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    
    # Print the thresholds for each distance metric.
    print("Cosine Threshold : ", cos_th)
    print("Euclidean Threshold : ", euc_th)
    print("Manhanttan Threshold : ", man_th)