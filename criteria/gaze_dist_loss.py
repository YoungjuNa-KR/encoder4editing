import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
from criteria.gaze_feature_extractor import GazeModel

import pickle
import PIL
import numpy as np

# from criteria.gaze_option import args
from options.train_options import TrainOptions
from criteria.getGazeLoss import computeGazeLoss
'''
    PureGaze에서는 Gaze에 필요 없는 정보를 제거하는 Adversarial Network를 고안하여 시선 추정을 개선했지만,
    우리 모델에서는 Encoder를 사용하여 시선 추정에 관한 정보는 더욱 분명하게 만들고, 그 외적인 정보는 제거한다.
'''

class GazeDistortionLoss(nn.Module):
    def __init__(self):       
        super(GazeDistortionLoss, self).__init__()
        print('Loading Gaze Feature Extractor')
        self.opts = TrainOptions().parse()
        
        '''resnet -> feature map '''
        self.gazenet = GazeModel(self.opts)
        print('Loading gaze model from {}'.format(model_paths['gaze']))
        
        self.gazenet.load_state_dict(
            torch.load(model_paths['gaze']),
            strict=True)
        print("Complete loading Gaze estimation model weight")
        self.gazenet.eval()
        
        for param in self.gazenet.parameters():
            param.requires_grad = False

    
    def extract_feats(self, x):
        x_feats = self.gazenet(x)
        return x_feats

    def forward(self, y_hat, y, x, labels):
        # print("gaze dist loss!")
        n_samples = x.shape[0]  # BATCH SIZE
        
        y_hat_feats = self.extract_feats(y_hat)
        # y_feats = y_feats.detach()
    
        loss = 0
        
        labels_theta = labels[0]
        labels_pi = labels[1]
        
        labels = torch.stack([labels_theta, labels_pi], dim=1)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(torch.device("cuda"))


        # print("label size:", labels.size())
        # print("img   size:", y_hat_feats.size())
        # print("labels:")
        # print(labels)
        # print("imgs")
        # print(y_hat_feats)
        # sim_improvement = 0
        gd_logs = []
        count = 0
        for i in range(n_samples):
            gaze_loss, angular_error = computeGazeLoss(y_hat_feats, labels)
            gd_logs.append({'gaze_target': float(angular_error)})
            
            loss += angular_error
            # id_diff = float(diff_target) - float(diff_views)
            # sim_improvement += id_diff
            count += 1

        return loss / count, gd_logs