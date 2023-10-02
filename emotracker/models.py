import os
import sys
import torch
import torch.nn as nn
import os.path
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import six

from emotracker.dataset import augment_transforms, compose_transforms

MODEL_DIR = './emotracker/'+'pytorch-benchmarks/models/'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_module_2or3(model_name, model_def_path):
    """Load model definition module in a manner that is compatible with
    both Python2 and Python3

    Args:
        model_name: The name of the model to be loaded
        model_def_path: The filepath of the module containing the definition

    Return:
        The loaded python module."""
    if six.PY3:
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        import importlib
        dirname = os.path.dirname(model_def_path)
        sys.path.insert(0, dirname)
        module_name = os.path.splitext(os.path.basename(model_def_path))[0]
        mod = importlib.import_module(module_name)
    return mod

def load_model(model_name, MODEL_DIR):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = os.path.join(MODEL_DIR, model_name + '.py')
    weights_path = os.path.join(MODEL_DIR, model_name + '.pth')
    mod = load_module_2or3(model_name, model_def_path)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net
      
class ResNet50():
    def __init__(self, opt):
        self._opt = opt
        self._name = 'ResNet50'
        self._output_size_per_task = {'AU': self._opt.AU_label_size, 'EXPR': self._opt.EXPR_label_size, 'VA': self._opt.VA_label_size * self._opt.digitize_num}
        # create networks
        self._init_create_networks()

    def _init_create_networks(self):
        """
        init current model according to sofar tasks
        """
        backbone = BackBone(self._opt)
        output_sizes = [self._output_size_per_task[x] for x in self._opt.tasks]
        output_feature_dim = backbone.output_feature_dim
        classifiers = [Head(output_feature_dim, self._opt.hidden_size, output_sizes[i]) for i in range(len(self._opt.tasks))]
        classifiers = nn.ModuleList(classifiers)
        self.resnet50 = Model(backbone, classifiers, self._opt.tasks)
        if len(self._opt.gpu_ids) > 1:
            self.resnet50 = torch.nn.DataParallel(self.resnet50, device_ids=self._opt.gpu_ids)
        # self.resnet50.cuda()
        
    def load(self, model_path, DEVICE):
        self.resnet50.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)  

    def set_eval(self):
        self.resnet50.eval()
        self._is_train = False

    def forward(self, input_image = None):
        assert self._is_train is False, "Model must be in eval mode"
        with torch.no_grad():
            input_image = Variable(input_image)
            # if not input_image.is_cuda:
            #     input_image = input_image.cuda()
            output = self.resnet50(input_image)
            out_dict = self._format_estimates(output['output'])
            out_dict_raw = dict([(key,output['output'][key].cpu()) for key in output['output'].keys()])
        return out_dict, out_dict_raw
    
    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                o = (torch.sigmoid(output['AU'].cpu())>0.5).type(torch.LongTensor)
                estimates['AU'] = o.numpy()
            elif task == 'EXPR':
                o = F.softmax(output['EXPR'].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                N = self._opt.digitize_num
                v = F.softmax(output['VA'][:, :N].cpu(), dim=-1).numpy()
                a = F.softmax(output['VA'][:, N:].cpu(), dim=-1).numpy()
                bins = np.linspace(-1, 1, num=self._opt.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                estimates['VA'] = np.stack([v, a], axis = 1)
        return estimates
    

class ResNet50_GRU():
    def __init__(self, opt):
        self._opt = opt
        self._name = 'ResNet50_GRU'
        self._output_size_per_task = {'AU': self._opt.AU_label_size, 'EXPR': self._opt.EXPR_label_size, 'VA': self._opt.VA_label_size * self._opt.digitize_num}
        # create networks
        self._init_create_networks()

    def _init_create_networks(self):
        """
        init current model according to sofar tasks
        """
        backbone = BackBone(self._opt)
        output_sizes = [self._output_size_per_task[x] for x in self._opt.tasks]
        output_feature_dim = backbone.output_feature_dim
        classifiers = [Head(output_feature_dim, self._opt.hidden_size, output_sizes[i]) for i in range(len(self._opt.tasks))]
        classifiers = nn.ModuleList(classifiers)
        resnet50 = Model(backbone, classifiers, self._opt.tasks)
        # create GRUs 
        GRU_classifiers = [GRU_Head(self._opt.hidden_size, self._opt.hidden_size//2, output_sizes[i]) for i in range(len(self._opt.tasks))]
        GRU_classifiers = nn.ModuleList(GRU_classifiers)
        self.resnet50_GRU = Seq_Model(resnet50, GRU_classifiers, self._opt.tasks)

        if len(self._opt.gpu_ids) > 1:
            self.resnet50_GRU = torch.nn.DataParallel(self.resnet50_GRU, device_ids=self._opt.gpu_ids)
        self.resnet50_GRU.to(DEVICE)

    def load(self, model_path):
        self.resnet50_GRU.load_state_dict(torch.load(model_path, map_location=DEVICE))

    def set_eval(self):
        self.resnet50_GRU.eval()
        self._is_train = False

    def forward(self, input_image = None):
        assert self._is_train is False, "Model must be in eval mode"
        with torch.no_grad():
            input_image = Variable(input_image)
            # if not input_image.is_cuda:
            #     input_image = input_image.cuda()
            output = self.resnet50_GRU(input_image)
            out_dict = self._format_estimates(output['output'])
            out_dict_raw = dict([(key, output['output'][key].cpu().numpy()) for key in output['output'].keys()])
        return out_dict, out_dict_raw
    
    def _format_estimates(self, output):
        estimates = {}
        for task in output.keys():
            if task == 'AU':
                o = (torch.sigmoid(output['AU'].cpu())>0.5).type(torch.LongTensor)
                estimates['AU'] = o.numpy()
            elif task == 'EXPR':
                o = F.softmax(output['EXPR'].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates['EXPR'] = o.numpy()
            elif task == 'VA':
                N = self._opt.digitize_num
                v = F.softmax(output['VA'][:,:, :N].cpu(), dim=-1).numpy()
                a = F.softmax(output['VA'][:,:, N:].cpu(), dim=-1).numpy()
                bins = np.linspace(-1, 1, num=self._opt.digitize_num)
                v = (bins * v).sum(-1)
                a = (bins * a).sum(-1)
                estimates['VA'] = np.stack([v, a], axis = -1)
        return estimates


class Model(nn.Module):
    def __init__(self, backbone, classifier, sofar_task):
        super(Model, self).__init__()
        self._name = 'Model'
        self.backbone = backbone
        self.classifier = classifier
        self.sofar_task = sofar_task
    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)
        features = {'cross_task': f}
        outputs = {}
        for i,m in enumerate(self.classifier):
            task = self.sofar_task[i] 
            o = m(f)
            outputs[task] = o['output']
            features[task] = o['feature']
        return {'output':outputs, 'feature':features}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BackBone(nn.Module):
    def __init__(self, opt):
        super(BackBone, self).__init__()
        self._name = 'BackBone'
        self._opt = opt
        self.model = self._init_create_networks()

    def _init_create_networks(self):
        # the feature extractor
        # different models have different input sizes, different mean and std
        if self._opt.pretrained_dataset == 'ferplus' or self._opt.pretrained_dataset == 'sfew':
            if self._opt.pretrained_dataset == 'ferplus':
                model_name = 'resnet50_ferplus_dag'
                model_dir = os.path.join(MODEL_DIR, 'fer+')
            else:
                model_name = 'resnet50_face_sfew_dag'
                model_dir = os.path.join(MODEL_DIR, 'sfew')
            feature_extractor = load_model(model_name, model_dir)
            meta = feature_extractor.meta
            if not meta['imageSize'][0] == self._opt.image_size:
                new_imageSize = [self._opt.image_size, self._opt.image_size, 3]
                override_meta_imsize = True
            else:
                new_imageSize = None
                override_meta_imsize = False
            setattr(self, 'augment_transforms', augment_transforms(meta, new_imageSize=new_imageSize, override_meta_imsize=override_meta_imsize))
            setattr(self, 'compose_transforms', compose_transforms(meta, new_imageSize=new_imageSize, override_meta_imsize=override_meta_imsize))
        else:
            raise ValueError("Pretrained dataset %s not recognized." % self._opt.pretrained_dataset)
        
        setattr(feature_extractor, 'name', model_name)
        
        # reform the final layer of feature extrator, turn it into a Identity module
        last_layer_name, last_module = list(feature_extractor.named_modules())[-1]
        try:
            in_channels, out_channels = last_module.in_features, last_module.out_features
            last_linear = True
        except:
            in_channels, out_channels = last_module.in_channels, last_module.out_channels
            last_linear = False

        setattr(feature_extractor, '{}'.format(last_layer_name), Identity()) # the second last layer has 512 dimensions
        setattr(self, 'output_feature_dim', in_channels)

        # orginal input size is 224, if the image size is different from 224, change the pool5 layer to adaptive avgpool2d
        if not meta['imageSize'][0] == self._opt.image_size:
            pool_layer_name, pool_layer = list(feature_extractor.named_modules())[-2]
            setattr(feature_extractor, '{}'.format(pool_layer_name), nn.AdaptiveAvgPool2d((1, 1)))
        return feature_extractor

    def forward(self, x):
        return self.model(x) 

class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class = 8):
        super(Head, self).__init__()
        self._name = 'Head'
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc_0 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, n_class)
    def forward(self, x):
        x = self.bn0(x)
        f0 = self.bn1(F.relu(self.fc_0(x)))
        output = self.fc_1(f0)
        return {'output':output, 'feature':f0}

class GRU_Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class = 8):
        super(GRU_Head, self).__init__()
        self._name = 'Head'
        self.GRU_layer = nn.GRU(input_dim, hidden_dim, batch_first= True, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_dim*2, n_class)
    def forward(self, x):
        B, N, C = x.size()
        self.GRU_layer.flatten_parameters()
        f0 = F.relu(self.GRU_layer(x)[0])
        output = self.fc_1(f0)
        return {'output':output, 'feature':f0}

class Seq_Model(nn.Module):
    def __init__(self, backbone, classifier, sofar_task):
        super(Seq_Model, self).__init__()
        self._name = 'Seq_Model'
        self.backbone = backbone
        self.classifier = classifier
        self.sofar_task = sofar_task
    def forward(self, x):
        B, N, C, W, H = x.size()
        x = x.view(B*N, C, W, H)
        out_backbone = self.backbone(x)
        outputs = {}
        features = {}
        for i,m in enumerate(self.classifier):
            task = self.sofar_task[i] 
            feature = out_backbone['feature'][task]
            feature = feature.view(B, N ,-1)
            o = m(feature)
            outputs[task] = o['output']
            features[task] = feature
        return {'output':outputs, 'feature':features}