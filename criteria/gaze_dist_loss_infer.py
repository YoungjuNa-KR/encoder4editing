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

    def forward(self, x, img_no_gd, img_with_gd, labels):
        # print("gaze dist loss!")
        n_samples = x.shape[0]  # BATCH SIZE
        
        x_feats = self.extract_feats(x)
        img_no_gd_feats = self.extract_feats(img_no_gd)
        img_with_gd_feats = self.extract_feats(img_with_gd)
        
        # y_feats = y_feats.detach()
    
        loss = 0
        
        labels_theta = labels[0]
        labels_pi = labels[1]
        
        labels = torch.stack([labels_theta, labels_pi], dim=1)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(torch.device("cuda"))

        gd_logs = []
        count = 0
        for i in range(n_samples):
            gaze_loss_x, angular_error_x = computeGazeLoss(x_feats, labels)
            gaze_loss_no_gd, angular_error_no_gd = computeGazeLoss(img_no_gd_feats, labels)
            gaze_loss_with_gd, angular_error_with_gd = computeGazeLoss(img_with_gd_feats, labels)
            
            
            gd_logs.append({'angular_x': float(angular_error_x)})
            gd_logs.append({'angular_no_gd': float(angular_error_no_gd)})
            gd_logs.append({'angular_with_gd': float(angular_error_with_gd)})

            loss_x += gaze_loss_x
            loss_no_gd += gaze_loss_no_gd
            loss_with_gd += gaze_loss_with_gd
            
            loss_x_ang += angular_error_x
            loss_no_gd_ang += angular_error_no_gd
            loss_with_gd_ang += angular_error_with_gd
            
            # id_diff = float(diff_target) - float(diff_views)
            # sim_improvement += id_diff
            count += 1

        loss_gaze = {
            'x' : loss_x,
            'no_gd' : loss_no_gd,
            'with_gd' : loss_with_gd
        }
        
        loss_ang = {
            'x_ang' : angular_error_x,
            'no_gd_ang' : angular_error_no_gd,
            'with_gd_ang' : angular_error_with_gd 
        }

        return loss_gaze, loss_ang, gd_logs