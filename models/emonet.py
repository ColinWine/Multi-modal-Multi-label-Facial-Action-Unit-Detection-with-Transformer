import torch.nn as nn
import torch
from torchvision import models
from .loss import CCCLoss,AULoss,SmoothAULoss
from torch.functional import F
import math
import numpy as np
from .heads import AU_former

nn.InstanceNorm2d = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.InstanceNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest') 

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class EmoNet(nn.Module):
    def __init__(self, num_modules=2, n_expression=8, n_reg=2, n_blocks=4, attention=True, temporal_smoothing=False):
        super(EmoNet, self).__init__()
        self.num_modules = num_modules
        self.n_expression = n_expression
        self.n_reg = n_reg
        self.attention = attention
        self.temporal_smoothing = temporal_smoothing
        self.init_smoothing = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.InstanceNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))
        #Do not optimize the FAN
        for p in self.parameters():
            p.requires_grad = False


        if self.attention:
            n_in_features = 256*(num_modules+1) #Heatmap is applied hence no need to have it
        else:
            n_in_features = 256*(num_modules+1)+68 #68 for the heatmap
        
        n_features = [(256, 256)]*(n_blocks)

        self.emo_convs = []
        self.conv1x1_input_emo_2 =nn.Conv2d(n_in_features, 256, kernel_size=1, stride=1, padding=0)
        for in_f, out_f in n_features:
            self.emo_convs.append(ConvBlock(in_f, out_f))
            self.emo_convs.append(nn.MaxPool2d(2,2))
        self.emo_net_2 = nn.Sequential(*self.emo_convs)
        self.avg_pool_2 = nn.AvgPool2d(4)
        self.emo_fc_2 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, self.n_expression + n_reg))
        self.au_head = AU_former(input_dim=256)

    def forward(self, x, reset_smoothing=False):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg_features = []

        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

            hg_features.append(ll)

        hg_features_cat = torch.cat(tuple(hg_features), dim=1)

        if self.attention:
            mask = torch.sum(tmp_out, dim=1, keepdim=True)
            hg_features_cat *= mask
            emo_feat = torch.cat((x, hg_features_cat), dim=1)
        else:
            emo_feat = torch.cat([x, hg_features_cat, tmp_out], dim=1)
        
        emo_feat_conv1D = self.conv1x1_input_emo_2(emo_feat)
        final_features = self.emo_net_2(emo_feat_conv1D)
        final_features = self.avg_pool_2(final_features)
        batch_size = final_features.shape[0]
        final_features = final_features.view(batch_size, final_features.shape[1])
        final_predict = self.emo_fc_2(final_features)
        #empty_au = torch.zeros(batch_size,12)
        au_out = self.au_head(final_features)
        return {'heatmap': tmp_out, 'expression': final_predict[:,:-2], 'valence_arousal': final_predict[:,-2:],'action_unit':au_out}
        #return final_features

  
    def eval(self):
        for module in self.children():
            module.eval()

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class ImageEmoNetModel(nn.Module):
    def __init__(self, modality='V;M',task='EX'):
        super(ImageEmoNetModel, self).__init__()

        self.base_model = EmoNet()
        self.load_pretrain_weight(r'K:\ABAW2022\models\pretrain\emonet_8.pth')
        self.task = task

        self.modes = ['clip']
        #self.base_model.fc = Dummy()
        self.loss_EX = nn.CrossEntropyLoss(ignore_index=7)
        self.loss_AU = SmoothAULoss()
        self.loss_VA = CCCLoss()
        self.num_channels = 3
        self.config_modality(modality)

    def forward(self, x):
        clip = x['clip']  # bx3x1x112x112
        clip = clip[:,-self.num_channels:,:,:,:]
        assert clip.size(2) == 1
        clip = clip.squeeze(2)

        features = self.base_model(clip)

        return features

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
    
    def load_pretrain_weight(self,state_dict_path):
        print(f'Loading the model from {state_dict_path}.')
        state_dict = torch.load(str(state_dict_path), map_location='cpu')
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        self.base_model.load_state_dict(state_dict, strict=False)

    def get_ex_loss(self, y_pred, y_true):
        y_pred = y_pred['expression']
        y_true = y_true.view(-1)
        loss = self.loss_EX(y_pred, y_true)
        return loss

    def get_au_loss(self, y_pred, y_true):
        #y_pred = torch.sigmoid(y_pred[:, :12])
        loss = self.loss_AU(y_pred['action_unit'], y_true)
        return loss

    def get_va_loss(self, y_pred, y_true):
        va = y_pred['valence_arousal']
        y_pred_v = torch.tanh(va[:, -2])
        y_pred_a = torch.tanh(va[:, -1])
        loss = self.loss_VA(y_pred_v, y_true[:, 0]) + self.loss_VA(y_pred_a, y_true[:, 1])
        return loss
