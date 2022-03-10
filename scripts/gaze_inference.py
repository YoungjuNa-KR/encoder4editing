import os
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, moco_loss, gaze_dist_loss_infer
from configs import data_configs
from datasets.images_dataset_inf import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger
from criteria.gaze_feature_extractor import GazeModel
import os
import json
import math
import sys
import pprint
import torch
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach


def main():
    opts = TrainOptions().parse()

    coach = Coach(opts)
    x_avg_gaze, no_gd_avg_ang, with_gd_avg_gaze = coach.validate()
    
    print("x avg gaze: ", x_avg_gaze)
    print("no gd avg ang: ", no_gd_avg_ang)
    print("with gd avg ang: ", with_gd_avg_gaze)

class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device
    
        self.net = GazeModel(self.opts).to(self.device)

        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)
        
        self.gd_loss = gaze_dist_loss_infer.GazeDistortionLoss().to(self.device).eval()
        
        self.total_gaze = 0
        self.total_ang = 0
        
        
    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
    
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        
        self.test_len = len(test_dataset)
        
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset
    
    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        x_total_gaze = 0
        no_gd_total_gaze = 0
        with_gd_total_gaze = 0
        
        for batch_idx, batch in enumerate(self.test_dataloader): # batch : img_no_gd, img_with_gd, label
            # img_names = batch[-1]
            # batch = batch[:-1]
            # cur_loss_dict = {}
            x, img_no_gd, img_with_gd, labels = batch
            
            
            
            with torch.no_grad():
                loss_gaze, loss_ang = self.calc_loss(x, img_no_gd, img_with_gd, labels)
            
            x_total_gaze = loss_gaze['x']
            no_gd_total_gaze = loss_gaze['no_gd']
            with_gd_total_gaze = loss_gaze['with_gd']
            
        x_avg_gaze = x_total_gaze / len(self.test_len)
        no_gd_avg_gaze = no_gd_total_gaze / len(self.test_len)
        with_gd_avg_gaze = with_gd_total_gaze / len(self.test_len)
    
        return x_avg_gaze, no_gd_avg_gaze, with_gd_avg_gaze
    
    def calc_loss(self, x, img_no_gd, img_with_gd, labels):
        loss_dict = {}
        loss_gaze = 0.0
        id_logs = None
        
        loss_gaze, loss_ang, gd_logs = self.gd_loss(x, img_no_gd, img_with_gd, labels)

        
        return loss_gaze, loss_ang

            