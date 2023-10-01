import argparse
import os
import sys
import torch
import os.path
import numpy as np
from tqdm import tqdm
import sys
import time

from dataset import Image_dataset
from models import ResNet50, ResNet50_GRU
from postprocess import save_to_file, save_result

sys.path.append('/Users/hongsumin/workspace/capston/emotracker')

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


parser = argparse.ArgumentParser(description='Implement the pretrained model on your own data')
parser.add_argument('--image_dir', type=str, 
                    help='a directory containing a sequence of cropped and aligned face images')
parser.add_argument('--model_type', type=str, default='CNN', choices= ['CNN', 'CNN-RNN'],
                    help='By default, the CNN pretrained models are stored in "Multitask-CNN", and the CNN-RNN \
                    pretrained models are stored in "Multitask-CNN-RNN"')
parser.add_argument('--seq_len', type=int, default = 32, choices = [32, 16, 8], help='sequence length when the model type is CNN-RNN')
parser.add_argument('--image_ext', default = ['.jpg', '.bmp', '.png'], help='image extentions')
parser.add_argument('--eval_with_teacher', action='store_true', help='whether to predict with teacher model')
parser.add_argument('--eval_with_students', action='store_true', help='whether to predict with student models')
parser.add_argument('--ensemble', action='store_true', help='whether to merge the student predictions')
parser.add_argument('--AU_label_size', type=int, default = 8, help='# of AUs')
parser.add_argument('--EXPR_label_size', type=int, default = 7, help='# of EXpressions')
parser.add_argument('--VA_label_size', type=int, default = 2, help='# of VA ')
parser.add_argument('--digitize_num', type=int, default= 20, choices = [1, 20], help='1 means no digitization,\
                                                 20 means to digitize continuous label to 20 one hot vector ')
parser.add_argument('--hidden_size', type=int, default = 128, help='the embedding size of each output head' )
parser.add_argument('--image_size', type=int, default= 112, help='input image size')
parser.add_argument('--batch_size', type=int, default= 20, help='input batch size per task')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--tasks', type=str, default = ['EXPR','AU','VA'],nargs="+")
parser.add_argument('--save_dir', type=str, help='where to save the predictions')
parser.add_argument('--pretrained_dataset', type=str, default='ferplus',
                                  choices = ['ferplus', 'sfew','imagenet'], 
                                  help="the pretrained_dataset of the face feature extractor, choices:['ferplus', 'sfew','imagenet']")
opt = parser.parse_args()

def test_one_video(model, data_loader):
    track_val = {}
    for task in opt.tasks:
        track_val[task] = {'outputs':[], 'estimates':[], 'frames_ids':[]}
    for i_val_batch, val_batch in tqdm(enumerate(data_loader), total = len(data_loader)):
        estimates, outputs = model.forward( input_image = val_batch['image'])
        #store the predictions and labels
        for task in opt.tasks:
            if 'RNN' in opt.model_type:
                B, N, C = outputs[task].shape
                track_val[task]['outputs'].append(outputs[task].reshape(B*N, C))
                track_val[task]['frames_ids'].append(np.array([np.array(x) for x in val_batch['frames_ids']]).reshape(B*N, -1).squeeze())
                track_val[task]['estimates'].append(estimates[task].reshape(B*N, -1).squeeze())
            else:
                track_val[task]['outputs'].append(outputs[task])
                track_val[task]['frames_ids'].append(np.array(val_batch['frames_ids']))
                track_val[task]['estimates'].append(estimates[task])
        # if i_val_batch >5:
        #     break
    for task in opt.tasks:
        for key in track_val[task].keys():
            track_val[task][key] = np.concatenate(track_val[task][key], axis=0)
    #assert len(track_val['frames_ids']) -1 == track_val['frames_ids'][-1]
    return track_val

def main():
    if opt.model_type == 'CNN':
        model = ResNet50(opt)
        val_transforms = model.resnet50.backbone.compose_transforms

    elif opt.model_type == 'CNN-RNN':
        model = ResNet50_GRU(opt)
        val_transforms = model.resnet50_GRU.backbone.backbone.compose_transforms

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    outputs_record = {}
    estimates_record = {}
    frames_ids_record = {}

    model.set_eval()
    model_path = os.path.join('pretrained_models', opt.model_type, '0.pth')
    assert os.path.exists(model_path)

    model.load(model_path)
    print('finish model load ...')

    start = time.time()
    
    dataset =  Image_dataset(opt, transform=val_transforms)
    dataloader = torch.utils.data.DataLoader(
                                            dataset,
                                            batch_size=opt.batch_size,
                                            shuffle= False,
                                            num_workers=opt.workers,
                                            drop_last=False
                                            )
    
    track = test_one_video(model, dataloader)
    torch.cuda.empty_cache()

    print(time.time() - start)

    for task in opt.tasks:
        outputs_record[task] = track[task]['outputs']
        estimates_record[task] = track[task]['estimates']
        frames_ids_record[task] = track[task]['frames_ids']

        save_path = '{}/{}.txt'.format(opt.save_dir, task)
        save_to_file(track[task]['frames_ids'], track[task]['estimates'], save_path, task=task)

    save_result(opt, outputs_record, frames_ids_record)

if __name__ == '__main__':
    main()


