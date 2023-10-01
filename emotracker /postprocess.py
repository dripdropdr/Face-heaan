import os
import os.path
import numpy as np
import torch.nn.functional as F
import torch
from collections import Counter
from scipy.special import softmax
from emotracker.dataset import sigmoid


CATEGORIES = {'AU': ['AU1', 'AU2', 'AU4', 'AU6', 'AU12', 'AU15', 'AU20', 'AU25'],
                            'EXPR':['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
                            'VA':['valence', 'arousal']}
Best_AU_Thresholds = {'CNN': [0.1448537, 0.03918985, 0.13766725, 0.02652811, 0.40589422, 0.15572545,0.04808964, 0.10848708],
                      'CNN-RNN': {32: [0.4253935, 0.02641966, 0.1119782, 0.02978198, 0.17256933, 0.06369855, 0.07433069, 0.13828614],
                                  16: [0.30485213, 0.09509478, 0.59577084, 0.4417419, 0.4396544, 0.0452404,0.05204154, 0.0633798 ],
                                  8: [0.4365209 ,0.10177602, 0.2649502,  0.22586018, 0.3772219,  0.07532539, 0.07667687, 0.04306327]}}


def soft_voting(predictions):
    '''
        - Input: Tensor of expr, au, va predictions. 
                 (A number of subjects , A number of predictions labels)

                This function integrates predictions by EXPR dimension and vote the highest two labels.

        - Output: Array of Top 2 EXPR label indexds.
    '''
    predictions = F.softmax(predictions['EXPR'], dim=-1).mean(dim=0).numpy()
    expr = predictions.argsort()[-1:]
    strength = predictions[expr]
    return expr, strength
    

def hard_voting(predictions):
    '''
        - Input: Tensor of expr, au, va predictions. 
                 (A number of subjects , A number of predictions labels)

                This function 

        - Output: Array of Top 2 EXPR labels.
    '''
    predictions = F.softmax(predictions['EXPR'], dim=-1).argmax(-1).numpy()
    counter = Counter(predictions)
    top_2_pred = [item[0] for item in counter.most_common(2)]
    return top_2_pred

def save_to_file(frames_ids, predictions, save_path, task= 'AU'):
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    categories = CATEGORIES[task]
    print(f'SAVE FILE: {save_path} {categories}')
    #filtered out repeated frames
    mask = np.zeros_like(frames_ids, dtype=bool)
    mask[np.unique(frames_ids, return_index=True)[1]] = True
    frames_ids = frames_ids[mask]
    predictions = predictions[mask]
    assert len(frames_ids) == len(predictions)

    with open(save_path, 'w') as f:
        f.write(",".join(categories)+"\n")
        for i, line in enumerate(predictions):
            if isinstance(line, np.ndarray):
                digits = []
                for x in line:
                    if isinstance(x, float) or isinstance(x, np.float32) or isinstance(x, np.float64):
                        digits.append("{:.4f}".format(x))
                    elif isinstance(x, np.int64):
                        digits.append(str(x))
                line = ','.join(digits)+'\n'
            elif isinstance(line, np.int64):
                line = str(line)+'\n'
            if i == len(predictions)-1:
                line = line[:-1]
            f.write(line)

def save_result(opt, outputs_record, frames_ids_record):
#merge the raw outputs && save them with raw_outputs
    for task in opt.tasks:
        preds = []
        for model_id in outputs_record.keys():
            if 'student' in model_id:
                preds.append(outputs_record[model_id][task])
        preds = np.array(preds)
        #assert frames_ids_record[0][task][video] == frames_ids_record[1][task][video]
        video_frames_ids = frames_ids_record[model_id][task]

        if task == 'AU':
            merged_preds = sigmoid(preds)
            merged_preds = np.mean(merged_preds, axis=0)
            save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged_raw', task)
            save_to_file(video_frames_ids, merged_preds, save_path, task='AU')
            best_thresholds_over_models = Best_AU_Thresholds[opt.model_type]
            if 'RNN' in opt.model_type:
                best_thresholds_over_models = best_thresholds_over_models[opt.seq_len]
            #print("The best AU thresholds over models: {}".format(best_thresholds_over_models))
            merged_preds = merged_preds > (np.ones_like(merged_preds)*best_thresholds_over_models)
            merged_preds = merged_preds.astype(np.int64)
            save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged', task)
            save_to_file(video_frames_ids, merged_preds, save_path, task='AU')

        elif task == 'EXPR':
            merged_preds = softmax(preds, axis=-1).mean(0)
            save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged_raw', task)
            save_to_file(video_frames_ids, merged_preds, save_path, task='EXPR')
            merged_preds = merged_preds.argmax(-1).astype(np.int64).squeeze()
            save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged', task)
            save_to_file(video_frames_ids, merged_preds, save_path, task='EXPR')

        else:
            N = opt.digitize_num
            v = softmax(preds[:, :, :N], axis=-1)
            a = softmax(preds[:, :, N:], axis=-1)
            bins = np.linspace(-1, 1, num=opt.digitize_num)
            v = (bins * v).sum(-1)
            a = (bins * a).sum(-1)
            merged_preds = np.stack([v.mean(0), a.mean(0)], axis = 1).squeeze() 
            save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged', task)
            save_to_file(video_frames_ids, merged_preds, save_path, task='VA') 
            save_path = '{}/{}/{}.txt'.format(opt.save_dir, 'merged_raw', task)
            save_to_file(video_frames_ids, merged_preds, save_path, task='VA') 