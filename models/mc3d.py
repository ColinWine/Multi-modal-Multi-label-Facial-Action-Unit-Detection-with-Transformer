"""
Code from
https://github.com/jiangxingxun/DFEW/blob/main/DataDistributedParallel/makemodel/i3d.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mc3_18
import pdb
from torch.autograd import Variable
from .loss import CCCLoss,AULoss,FocalLoss_Ori
import numpy as np
from collections import OrderedDict

import numpy as np

import os
import sys
from collections import OrderedDict

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input

def load_pretrain(model,weight_path):
    print('Loading former weight')
    pretrained_dict = torch.load(weight_path)
    new_state_dict=OrderedDict()
    for k,v in pretrained_dict.items():
        new_name = k.replace('module.','')
        new_state_dict[new_name]=v
    model.load_state_dict(new_state_dict,strict=False)

class VisualMC3DModel(nn.Module):
    def __init__(self, modality='A;V;M', video_pretrained = True, task='EX'):
        super(VisualMC3DModel, self).__init__()

        self.video_model = mc3_18()
        if video_pretrained:
            load_pretrain(self.video_model,r'K:\ABAW2022\models\pretrain\mc3_18_fold5_epo074_UAR48.55_WAR58.78.pth')
        assert 'V' in modality and 'M' not in modality
        self.task = task
        self.modes = ["clip"]
        self.fc = nn.Sequential(
        nn.Linear(in_features=512,out_features=256),
        nn.BatchNorm1d(256),
        nn.Linear(in_features=256,out_features=12+7+2)
        )
        self.video_model.fc = Dummy()
        self.loss_EX = nn.CrossEntropyLoss(ignore_index=7)
        #weight=torch.tensor([2.62, 26.5, 45, 40, 4.0, 5.87, 1.0])
        #self.loss_EX = FocalLoss_Ori(num_class=7, gamma=2.0, ignore_index=7, reduction='mean')
        self.loss_AU = AULoss()
        self.loss_VA = CCCLoss() #nn.MSELoss() 

    def forward(self, x):
        clip = x['clip']
        video_model_features = self.video_model(clip)
        out = self.fc(video_model_features)
        # out = self.fc_video(video_model_features)
        return out

    def get_ex_loss(self, y_pred, y_true):
        y_pred = y_pred[:, 12:19]
        y_true = y_true.view(-1)
        loss = self.loss_EX(y_pred, y_true)
        return loss

    def get_au_loss(self, y_pred, y_true):
        #y_pred = torch.sigmoid(y_pred[:, :12])
        loss = self.loss_AU(y_pred[:, :12], y_true)
        return loss

    def get_va_loss(self, y_pred, y_true):
        y_pred_v = torch.tanh(y_pred[:, 19])
        y_pred_a = torch.tanh(y_pred[:, 20])
        #print(y_pred_v)
        #print(y_true[:, 0])
        loss = 2*self.loss_VA(y_pred_v, y_true[:, 0]) + self.loss_VA(y_pred_a, y_true[:, 1])
        return loss