import os
import torch
import os.path
from PIL import Image
import random
import numpy as np
import glob
import torchvision.transforms as transforms
import numbers
import pandas as pd


class Image_dataset(object):
    def __init__(self, opt, transform = None):
        self._opt = opt
        assert transform is not None
        self._transform = transform
        # read dataset
        self._read_dataset()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        if 'RNN' in self._opt.model_type:
            images = []
            labels = []
            img_paths = []
            frames_ids = []
            df = self.sample_seqs[index]
            for i, row in df.iterrows():
                img_path = row['path']
                image = Image.open(img_path).convert('RGB')
                image = self._transform(image)
                frame_id = row['frames_ids']
                images.append(image)
                img_paths.append(img_path)
                frames_ids.append(frame_id)
            # pack data
            sample = {'image': torch.stack(images,dim=0),
                      'path': img_paths,
                      'index': index,
                      'frames_ids':frames_ids
                      }
        else:
            image = None
            label = None
            img_path = self._data['path'][index]
            image = Image.open( img_path).convert('RGB')
            frame_ids = self._data['frames_ids'][index]
            # transform data
            image = self._transform(image)
            # pack data
            sample = {'image': image,
                      'path': img_path,
                      'index': index,
                      'frames_ids': frame_ids
                      }
        return sample
    def _read_dataset(self):
        #sample them 
        seq_len = self._opt.seq_len
        model_type = self._opt.model_type
        frames_paths = glob.glob(os.path.join(self._opt.image_dir, '*'))
        frames_paths = [x for x in frames_paths if any([ext in x for ext in self._opt.image_ext])]
        frames_paths = sorted(frames_paths)
        self._data = {'path': frames_paths, 'frames_ids': np.arange(len(frames_paths))} # dataframe are easier for indexing
        if 'RNN' in self._opt.model_type:
            self._data = pd.DataFrame.from_dict(self._data)
            self.sample_seqs = []
            N = seq_len
            for i in range(len(self._data['path'])//N + 1):
                start, end = i*N, i*N + seq_len
                if end >= len(self._data):
                    start, end = len(self._data) - seq_len, len(self._data)
                new_df = self._data.iloc[start:end]
                if not len(new_df) == seq_len:
                    assert len(new_df) < seq_len
                    count = seq_len - len(new_df)
                    for _ in range(count):
                        new_df = new_df.append(new_df.iloc[-1])
                assert len(new_df) == seq_len
                self.sample_seqs.append(new_df)
            self._ids = np.arange(len(self.sample_seqs)) 
            self._dataset_size = len(self._ids)
        else:
            self._ids = np.arange(len(self._data['path'])) 
            self._dataset_size = len(self._ids) 

    def __len__(self):
        return self._dataset_size


def sigmoid(x):
    return 1/(1+np.exp(-x))

def compose_transforms(meta, center_crop=True, new_imageSize = None,
                      override_meta_imsize=False):
    """Compose preprocessing transforms for model

    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.

    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `new_meta`
           to select the image input size, 
    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    if override_meta_imsize:
        im_size = new_imageSize
    assert im_size[0] == im_size[1], 'expected square image size'

    if center_crop:
        transform_list = [transforms.Resize(int(im_size[0]*1.2)),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    
    return transforms.Compose(transform_list)

def augment_transforms(meta, random_crop=True, new_imageSize = None,
                      override_meta_imsize=False):
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    if override_meta_imsize:
        im_size = new_imageSize
    assert im_size[0] == im_size[1], 'expected square image size'
    if random_crop:
        v = random.random()
        transform_list = [transforms.Resize(int(im_size[0]*1.2)),
                          RandomCrop(im_size[0], v),
                          RandomHorizontalFlip(v)]
    else:
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)

    return transforms.Compose(transform_list) 

class RandomCrop(object):
    def __init__(self, size, v):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.v = v
    def __call__(self, img):

        w, h = img.size
        th, tw = self.size
        x1 = int(( w - tw)*self.v)
        y1 = int(( h - th)*self.v)
        #print("print x, y:", x1, y1)
        assert(img.size[0] == w and img.size[1] == h)
        if w == tw and h == th:
            out_image = img
        else:
            out_image = img.crop((x1, y1, x1 + tw, y1 + th)) #same cropping method for all images in the same group
        return out_image

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, v):
        self.v = v
        return
    def __call__(self, img):
        if self.v < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) 
        #print ("horiontal flip: ",self.v)
        return img