import sys
import argparse
import os
import torch
import os.path
import numpy as np
from tqdm import tqdm
import sys
import time
from PIL import Image
from glob import glob

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='Implement the pretrained model on your own data')
    parser.add_argument('--image_dir', type=str, default = '../src/proj',
                        help='a directory containing a sequence of cropped and aligned face images')
    parser.add_argument('--seq_len', type=int, default = 8, choices = [32, 16, 8], help='sequence length when the model type is CNN-RNN')
    parser.add_argument('--image_ext', default = ['.jpg', '.bmp', '.png'], help='image extentions')
    parser.add_argument('--eval_with_students', action='store_true', help='whether to predict with student models')
    parser.add_argument('--AU_label_size', type=int, default = 8, help='# of AUs')
    parser.add_argument('--EXPR_label_size', type=int, default = 7, help='# of EXpressions')
    parser.add_argument('--VA_label_size', type=int, default = 2, help='# of VA ')
    parser.add_argument('--digitize_num', type=int, default= 20, choices = [1, 20], help='1 means no digitization,\
                                                    20 means to digitize continuous label to 20 one hot vector ')
    parser.add_argument('--hidden_size', type=int, default = 128, help='the embedding size of each output head' )
    parser.add_argument('--image_size', type=int, default= 112, help='input image size')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--tasks', type=str, default = ['EXPR'],nargs="+") # ['EXPR','AU','VA']
    parser.add_argument('--save_dir', type=str, default='../save-directory', help='where to save the predictions')
    parser.add_argument('--pretrained_dataset', type=str, default='ferplus',
                                    choices = ['ferplus', 'sfew','imagenet'], 
                                    help="the pretrained_dataset of the face feature extractor, choices:['ferplus', 'sfew','imagenet']")
    opt = parser.parse_args()
    return opt


def run_tracker(opt, model, sample):

    # init track result
    track_val = {}
    for task in opt.tasks:
        track_val[task] = {'outputs':[], 'estimates':[], 'frames_ids':[]}
    
    estimates, outputs = model.forward(input_image = sample['image'])
    #store the predictions and labels
    for task in opt.tasks:
        track_val[task]['outputs'].append(outputs[task])
        track_val[task]['frames_ids'].append(np.array(sample['frames_ids']))
        track_val[task]['estimates'].append(estimates[task])
        for key in track_val[task].keys():
            try:
                track_val[task][key] = np.concatenate(track_val[task][key], axis=0)
            except ValueError as e:
                track_val[task][key] = np.array([0])

    return track_val


def preprocess_images(images, val_transforms, index):
    images_transform = []
    frames_ids = list([index] * len(images))
    for image in images:
        image = Image.fromarray(image)
        image = val_transforms(image)
        images_transform.append(image)
    # pack data
    images_tensor = torch.stack(images_transform, dim=0)
    sample = {'image': images_tensor,
                'index': index,
                'frames_ids': frames_ids
                }

    return sample

if __name__ == '__main__':

    images = [Image.open(path) for path in glob(os.path.join('../src/proj', '*.jpg'))[:32]]
    # images = Image.open(os.path.join(opt.image_dir, '1.jpg'))
    # main(images=images, index=1)


