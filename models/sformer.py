"""
Code from
https://github.com/zengqunzhao/Former-DFER
"""
from einops import rearrange, repeat
from torch import nn, einsum
import math
import torch
from torchvision import models
from .loss import *
from torch.functional import F
import numpy as np
from collections import OrderedDict
from torch.nn import SmoothL1Loss
from .heads import AU_former,VA_former

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_IBN, self).__init__()
        norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = IBN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def load_pretrain(model,weight_path):
    print('Loading former weight')
    pretrained_dict = torch.load(weight_path)['state_dict']
    new_state_dict=OrderedDict()
    for k,v in pretrained_dict.items():
        new_name = k.replace('module.','')
        new_state_dict[new_name]=v
    model.load_state_dict(new_state_dict,strict=False)

class ResFormer(nn.Module):   # S-Former after stage3

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 num_patches=7*7, dim=256, depth=1, heads=8, mlp_dim=512, dim_head=32, dropout=0.0):
        super(ResFormer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or"
                             " a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        #torch.Size([2, 16, 3, 112, 112])
        b,t,c,h,w = x.shape
        x = x.contiguous().view(-1, c, h, w)
        #torch.Size([32, 3, 112, 112])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # torch.Size([1, 64, 28, 28])
        x = self.layer1(x)     # torch.Size([1, 64, 28, 28])
        x = self.layer2(x)     # torch.Size([1, 128, 14, 14])
        x = self.layer3(x)     # torch.Size([1, 256, 7, 7])
        b_l, c, h, w = x.shape
        #torch.Size([32, 256, 7, 7])
        x = x.reshape((b_l, c, h*w))
        #torch.Size([32, 256, 49])
        x = x.permute(0, 2, 1)
        b, n, _ = x.shape
        #x shape: torch.Size([32, 49, 256])
        #pos_embedding shape torch.Size([1, 49, 256])
        x = x + self.pos_embedding[:, :n]
        #torch.Size([32, 49, 256])
        x = self.spatial_transformer(x)
        #torch.Size([32, 49, 256])
        x = x.permute(0, 2, 1)
        #torch.Size([32, 256, 49])
        x = x.reshape((b, c, h, w))
        #torch.Size([32, 256, 7, 7])
        x = self.layer4(x)     
        #torch.Size([32, 512, 4, 4])
        x = self.avgpool(x)
        #torch.Size([32, 512, 1, 1])
        x = torch.flatten(x, 1)
        #torch.Size([32, 512])

        return x

class SpatialFormer(nn.Module):
    def __init__(self, modality='A;V;M', video_pretrained = True, task='EX'):
        super(SpatialFormer, self).__init__()

        self.base_model = ResFormer(BasicBlock, [2, 2, 2, 2],dropout=0.2)
        '''
        if video_pretrained:
            load_pretrain(self.base_model,r'K:\ABAW2022\models\pretrain\Former-DFER-Pretrained-on-DFEW.pth')
        '''
        self.num_channels = 3
        self.config_modality(modality)
        self.task = task
        self.modes = ["clip"]
        self.fc = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Linear(in_features=512,out_features=256),
        nn.BatchNorm1d(256),
        nn.Linear(in_features=256,out_features=12+7+2)
        )
        self.au_head = AU_former(dropout=0.2)
        self.va_head = VA_former(dropout=0.2)
        self.loss_EX = nn.CrossEntropyLoss(ignore_index=7)
        #weight=torch.tensor([2.62, 26.5, 45, 40, 4.0, 5.87, 1.0])
        #self.loss_EX = FocalLoss_Ori(num_class=7, gamma=2.0, ignore_index=7, reduction='mean')
        self.loss_AU = DiceAULoss() #AULoss() #SmoothAULoss()
        self.loss_VA = CCCLoss()

    def forward(self, x):
        clip = x['clip']
        clip = clip[:,-self.num_channels:,:,:,:]
        assert clip.size(2) == 1
        clip = clip.permute(0,2,1,3,4)
        #clip = clip.squeeze(2)


        features = self.base_model(clip)
        out = self.fc(features)
        if self.task == 'AU':
            au_out,_ = self.au_head(features)
            out[:,:12] = au_out
        if self.task == 'VA':
            va_out,_ = self.va_head(features)
            out[:,-2:] = va_out
        
        return out
    
    def config_modality(self, modality='A;V;M'):
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