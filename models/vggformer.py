"""
Code from
https://github.com/zengqunzhao/Former-DFER
"""
from einops import rearrange, repeat
from torch import nn, einsum
import math
import torch
from torchvision import models
from .loss import CCCLoss,AULoss,FocalLoss_Ori,FocalLoss_TOPK
from torch.functional import F
import numpy as np
from collections import OrderedDict
import pickle

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class VGGFace2_extractor(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(VGGFace2_extractor, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        block = Bottleneck
        layers = [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

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
    
class VGGCONV(nn.Module):   # S-Former after stage3

    def __init__(self):
        super(VGGCONV, self).__init__()
        self.VGG_model = VGGFace2_extractor()
        for param in self.VGG_model.parameters():
            param.requires_grad = False
        self.conv = conv1x1(2048,512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def load_pretrain_vgg(self,path):
        with open(path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        self.VGG_model.load_state_dict(weights,strict=False)

    def forward(self, x):
        #torch.Size([2, 16, 3, 112, 112])
        b,t,c,h,w = x.shape
        x = x.contiguous().view(-1, c, h, w)
        #torch.Size([32, 3, 112, 112])
        x = self.VGG_model(x)
        x = self.conv(x)
        b_l, c, h, w = x.shape
        #torch.Size([32, 256, 7, 7])
        x = self.avgpool(x)
        #torch.Size([32, 512, 1, 1])
        x = torch.flatten(x, 1)
        #torch.Size([32, 512])

        return x

class VGGFormer(nn.Module):   # S-Former after stage3

    def __init__(self,num_patches=7*7, dim=512, depth=1, heads=8, mlp_dim=512, dim_head=32, dropout=0.0):
        super(VGGFormer, self).__init__()
        self.VGG_model = VGGFace2_extractor()
        for param in self.VGG_model.parameters():
            param.requires_grad = False
        self.conv = conv1x1(2048,512)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def load_pretrain_vgg(self,path):
        with open(path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        self.VGG_model.load_state_dict(weights,strict=False)

    def forward(self, x):
        #torch.Size([2, 16, 3, 112, 112])
        b,t,c,h,w = x.shape
        x = x.contiguous().view(-1, c, h, w)
        #torch.Size([32, 3, 112, 112])
        x = self.VGG_model(x)
        x = self.conv(x)
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
        x = self.avgpool(x)
        #torch.Size([32, 512, 1, 1])
        x = torch.flatten(x, 1)
        #torch.Size([32, 512])

        return x

class TFormer(nn.Module):
    def __init__(self, num_patches=16, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0):
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.dim) #-1, 16, 512
        #torch.Size([2, 16, 512])
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        #torch.Size([2, 1, 512])
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n+1)]
        #torch.Size([2, 17, 512])
        x = self.spatial_transformer(x)
        #torch.Size([2, 17, 512])
        x = x[:, 0]
        #torch.Size([2, 1, 512])

        return x

class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()
        self.s_former = VGGFormer()
        #self.s_former = VGGCONV()
        self.s_former.load_pretrain_vgg(r'K:\ABAW2022\models\pretrain\resnet50_ft_weight.pkl')
        self.t_former = TFormer(num_patches=16)
        self.fc = nn.Linear(in_features=512, out_features=7)
        self.num_channels = 3
        self.modes = ["clip"]

    def forward(self, x):
        #b C T W H
        x = x[:,-self.num_channels:,:,:,:]
        x = x.permute(0,2,1,3,4)
        #([2, 16, 3, 112, 112]) b T C W H
        x = self.s_former(x)
        x = self.t_former(x)
        x = self.fc(x)
        return x

    def config_modality(self, modality='V;M'):
        #if only RGB, dont need to change input channel
        #if use mask
        if 'M' in modality: 
            if 'V' in modality: #Both Mask and RGB, concat RGB with Mask, change input channel to 4
                self.num_channels = 4
            else: #only Mask, dont need to change input channel
                self.num_channels = 1

            new_first_layer = nn.Conv2d(in_channels=self.num_channels,
                                        out_channels=self.s_former.conv1.out_channels,
                                        kernel_size=self.s_former.conv1.kernel_size,
                                        stride=self.s_former.conv1.stride,
                                        padding=self.s_former.conv1.padding,
                                        bias=False)
            # copy pre-trained weights for first 3 channels
            if 'V' in modality:
                new_first_layer.weight.data[:, 0:3] = self.s_former.conv1.weight.data
            self.s_former.VGG_model.conv1 = new_first_layer

class VGGVisualFormer(nn.Module):
    def __init__(self, modality='A;V;M', video_pretrained = True, task='EX'):
        super(VGGVisualFormer, self).__init__()

        self.video_model = VideoModel()
        self.video_model.config_modality(modality)
        '''
        if modality == 4:
            new_first_layer = nn.Conv2d(in_channels=4,
                                        out_channels=self.video_model.s_former.conv1.out_channels,
                                        kernel_size=self.video_model.s_former.conv1.kernel_size,
                                        stride=self.video_model.s_former.conv1.stride,
                                        padding=self.video_model.s_former.conv1.padding,
                                        bias=False)
            # copy pre-trained weights for first 3 channels
            new_first_layer.weight.data[:, 0:3] = self.video_model.s_former.conv1.weight.data
            self.video_model.s_former.conv1 = new_first_layer
        '''
        self.task = task
        self.modes = ["clip"]
        self.fc = nn.Sequential(
        nn.Linear(in_features=self.video_model.fc.in_features,out_features=256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace = True),
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