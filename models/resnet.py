import torch.nn as nn
import torch
from torchvision import models
from .loss import CCCLoss
from torch.functional import F


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class ImageResNetModel(nn.Module):
    def __init__(self, modality='V;M',task='EX'):
        super(ImageResNetModel, self).__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.task = task
        self.fc = nn.Sequential(nn.Dropout(0.0),
                                nn.Linear(in_features=self.base_model.fc.in_features,
                                          out_features=12 + 7 + 2))

        self.modes = ['clip']
        self.base_model.fc = Dummy()
        self.loss_EX = nn.CrossEntropyLoss(ignore_index=7)
        self.loss_AU = nn.BCELoss()
        self.loss_VA = CCCLoss()
        self.num_channels = 3
        self.config_modality(modality)

    def forward(self, x):
        clip = x['clip']  # bx3x1x112x112
        clip = clip[:,-self.num_channels:,:,:,:]
        assert clip.size(2) == 1
        clip = clip.squeeze(2)

        features = self.base_model(clip)
        out = self.fc(features)

        return out

    def config_modality(self, modality='V;M'):
        #if only RGB, dont need to change input channel
        #if use mask
        if 'M' in modality: 
            if 'V' in modality: #Both Mask and RGB, concat RGB with Mask, change input channel to 4
                self.num_channels = 4
            else: #only Mask, dont need to change input channel
                self.num_channels = 1

            new_first_layer = nn.Conv2d(in_channels=self.num_channels,
                                        out_channels=self.base_model.conv1.out_channels,
                                        kernel_size=self.base_model.conv1.kernel_size,
                                        stride=self.base_model.conv1.stride,
                                        padding=self.base_model.conv1.padding,
                                        bias=False)
            # copy pre-trained weights for first 3 channels
            if 'V' in modality:
                new_first_layer.weight.data[:, 0:3] = self.base_model.conv1.weight.data
            self.base_model.conv1 = new_first_layer

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
        loss = self.loss_VA(y_pred_v, y_true[:, 0]) + self.loss_VA(y_pred_a, y_true[:, 1])
        return loss
