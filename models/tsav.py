"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch.nn as nn
import torch
from torchvision import models
from .loss import CCCLoss,AULoss
from torch.functional import F
import numpy as np

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class VideoModel(nn.Module):
    def __init__(self, num_channels=3):
        super(VideoModel, self).__init__()
        self.r2plus1d = models.video.r2plus1d_18(pretrained=True)
        self.r2plus1d.fc = nn.Sequential(nn.Dropout(0.2),
                                         nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=22))
        if num_channels == 4:
            new_first_layer = nn.Conv3d(in_channels=4,
                                        out_channels=self.r2plus1d.stem[0].out_channels,
                                        kernel_size=self.r2plus1d.stem[0].kernel_size,
                                        stride=self.r2plus1d.stem[0].stride,
                                        padding=self.r2plus1d.stem[0].padding,
                                        bias=False)
            # copy pre-trained weights for first 3 channels
            new_first_layer.weight.data[:, 0:3] = self.r2plus1d.stem[0].weight.data
            self.r2plus1d.stem[0] = new_first_layer

        self.modes = ["clip"]

    def forward(self, x):
        return self.r2plus1d(x)


class AudioModel(nn.Module):
    def __init__(self, pretrained=False):
        super(AudioModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.0),
                                       nn.Linear(in_features=self.resnet.fc.in_features, out_features=22))

        old_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, out_channels=self.resnet.conv1.out_channels,
                                      kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if pretrained == True:
            self.resnet.conv1.weight.data.copy_(
                torch.mean(old_layer.weight.img_data, dim=1, keepdim=True))  # mean channel

        self.modes = ["audio_features"]

    def forward(self, x):
        return self.resnet(x)


class TwoStreamAuralVisualModel(nn.Module):
    def __init__(self, num_channels=4, audio_pretrained=False, task='EX'):
        super(TwoStreamAuralVisualModel, self).__init__()
        self.audio_model = AudioModel(pretrained=audio_pretrained)
        self.video_model = VideoModel(num_channels=num_channels)
        self.task = task
        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=self.audio_model.resnet.fc._modules['1'].in_features +
                                                      self.video_model.r2plus1d.fc._modules['1'].in_features,
                                          out_features=12 + 8 + 2))
        # self.fc_video = nn.Sequential(nn.Dropout(0.0),
        #                               nn.Linear(in_features=self.video_model.r2plus1d.fc._modules['1'].in_features,
        #                                         out_features=12 + 8 + 2))
        self.modes = ['clip', 'audio_features']
        self.audio_model.resnet.fc = Dummy()
        self.video_model.r2plus1d.fc = Dummy()
        self.loss_EX = nn.CrossEntropyLoss(ignore_index=7,weight=torch.tensor([2.62, 26.5, 45, 40, 4.0, 5.87, 1.0, 0]))
        #self.loss_EX = FocalLoss_Ori(num_class=7, gamma=2.0, ignore_index=7, reduction='mean')
        self.loss_AU = AULoss()
        self.loss_VA = CCCLoss() #nn.MSELoss()

    def forward(self, x):
        audio = x['audio_features']
        clip = x['clip']

        audio_model_features = self.audio_model(audio)
        video_model_features = self.video_model(clip)

        features = torch.cat([audio_model_features, video_model_features], dim=1)
        out = self.fc(features)
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
        loss = self.loss_VA(y_pred_v, y_true[:, 0]) + self.loss_VA(y_pred_a, y_true[:, 1])
        return loss
    
    def get_mt_loss(self,y_pred, y_true, normalize = False):  #multi-task loss
        loss_ex = self.get_ex_loss(y_pred,y_true['EX'])
        loss_au = self.get_au_loss(y_pred, y_true['AU'])
        loss_va = self.get_va_loss(y_pred, y_true['VA'])
        if normalize:
            valid_ex_label_num = np.sum(y_true['EX'].detach().cpu().numpy() != 7)
            if valid_ex_label_num != 0:
                loss_ex = loss_ex/valid_ex_label_num
            else:
                device = y_true.device
                loss_ex = torch.tensor(0.0, requires_grad=True).to(device)
            
            valid_au_label_num = np.sum((y_true['AU'].detach().cpu().numpy() != -1))
            if valid_au_label_num != 0:
                loss_au = loss_au/valid_au_label_num
            else:
                device = y_true.device
                loss_au = torch.tensor(0.0, requires_grad=True).to(device)
            
            valid_va_label_num = np.sum(y_true['VA'].detach().cpu().numpy() != -5.0)
            if valid_va_label_num != 0:
                loss_va = loss_va/valid_va_label_num
            else:
                device = y_true.device
                loss_va = torch.tensor(0.0, requires_grad=True).to(device)

        return [loss_ex,loss_au,loss_va]