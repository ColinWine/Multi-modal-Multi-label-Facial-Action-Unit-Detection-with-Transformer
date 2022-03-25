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
from .heads import AU_former

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
        self.s_former = ResFormer(BasicBlock, [2, 2, 2, 2])
        self.au_head = AU_former(dropout=0.2)
        self.t_former = TFormer(dim=128*12)
        self.fc = nn.Linear(in_features=128*12, out_features=7)
        self.num_channels = 3

    def forward(self, x):
        x = x[:,-self.num_channels:,:,:,:]
        #b C T W H
        x = x.permute(0,2,1,3,4)
        #([2, 16, 3, 112, 112]) b T C W H
        x = self.s_former(x)
        _,transformer_out= self.au_head(x)
        bs = transformer_out.shape[0]
        transformer_out = transformer_out.view(bs,-1)
        x = self.t_former(transformer_out)
        x = self.fc(x)
        return x
    
    def config_modality(self, modality='A;V;M'):
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
            self.s_former.conv1 = new_first_layer
        
def load_pretrain(model,weight_path):
    print('Loading former weight')
    pretrained_dict = torch.load(weight_path)['state_dict']
    new_state_dict=OrderedDict()
    for k,v in pretrained_dict.items():
        new_name = k.replace('module.','')
        new_name = k.replace('base_model.','s_former.')
        new_state_dict[new_name]=v
    print(new_state_dict.keys())
    model.load_state_dict(new_state_dict,strict=False)

def load_pretrain_sformer(model,weight_path):
    print('Loading former weight')
    pretrained_dict = torch.load(weight_path)
    new_state_dict=OrderedDict()
    for k,v in pretrained_dict.items():
        new_name = k.replace('base_model.','s_former.')
        new_state_dict[new_name]=v
    print(new_state_dict.keys())
    print(model.load_state_dict(new_state_dict,strict=False))
    '''
    for p in model.s_former.parameters():
            p.requires_grad = False
    '''
class tformer_AU_head(nn.Module):
    def __init__(self, emb_dim = 128, dropout=0.0):
        super(tformer_AU_head, self).__init__()
         #AU branch
        self.pos_embedding = nn.Parameter(torch.randn(1, 12, emb_dim))
        self.corr_transformer = Transformer(emb_dim, depth=3, heads=8, mlp_dim=256, dim_head=32, dropout=dropout)
        self.AU_linear_last1 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last2 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last3 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last4 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last5 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last6 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last7 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last8 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last9 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last10 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last11 = nn.Linear(emb_dim, 1, bias=False)
        self.AU_linear_last12 = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, input):
        bs = input.shape[0]
        AU_inter_out = input.view(bs, 12, -1)
        bs, n, _ = AU_inter_out.shape
        #x shape: torch.Size([32, 49, 256])
        #pos_embedding shape torch.Size([1, 49, 256])
        transformer_input = AU_inter_out + self.pos_embedding[:, :n]
        transformer_out = self.corr_transformer(transformer_input)
        x1 = self.AU_linear_last1(transformer_out[:,0,:])
        x2 = self.AU_linear_last2(transformer_out[:,1,:])
        x3 = self.AU_linear_last3(transformer_out[:,2,:])
        x4 = self.AU_linear_last4(transformer_out[:,3,:])
        x5 = self.AU_linear_last5(transformer_out[:,4,:])
        x6 = self.AU_linear_last6(transformer_out[:,5,:])
        x7 = self.AU_linear_last7(transformer_out[:,6,:])
        x8 = self.AU_linear_last8(transformer_out[:,7,:])
        x9 = self.AU_linear_last9(transformer_out[:,8,:])
        x10 = self.AU_linear_last10(transformer_out[:,9,:])
        x11 = self.AU_linear_last11(transformer_out[:,10,:])
        x12 = self.AU_linear_last12(transformer_out[:,11,:])
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        
        return AU_out

class SpatialTemporalFormer(nn.Module):
    def __init__(self, modality='A;V;M', video_pretrained = True, task='EX'):
        super(SpatialTemporalFormer, self).__init__()

        self.video_model = VideoModel()
        if video_pretrained:
            load_pretrain_sformer(self.video_model,r'K:\ABAW2022\models\pretrain\sformer.pth')
        self.video_model.config_modality(modality)
        self.task = task
        self.modes = ["clip"]
        self.au_head = tformer_AU_head(dropout=0.2)
        self.fc = nn.Sequential(
        nn.BatchNorm1d(self.video_model.fc.in_features),
        nn.Linear(in_features=self.video_model.fc.in_features,out_features=256),
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
        au_out = self.au_head(video_model_features)
        out[:,:12] = au_out
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
