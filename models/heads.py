#code from https://github.com/FuxiVirtualHuman/ABAW2_Competition/blob/main/models/multi_model_series.py

import torch.nn as nn
import torch
from torch import einsum
from torch.functional import F
import numpy as np
from einops import rearrange, repeat
import math

class AU_multihead(nn.Module):
    def __init__(self, input_dim=512,emb_dim = 16, inter=False):
        super(AU_multihead, self).__init__()
        self.inter = inter
         #AU branch
        self.emb_dim = input_dim
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, emb_dim)
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
        if self.inter:
            self.AU_inter = nn.Linear(emb_dim * 12,64)

    def forward(self, emb):
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        
        AU_inter_out = self.AU_inter(AU_inter_out)
        if not self.inter:
            return AU_out
        else:
            return AU_out,AU_inter_out

class EXP_head(nn.Module):
    def __init__(self, input_dim=512,inter=False):
        super(EXP_head, self).__init__()
        self.inter = inter
        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, 64)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        if self.inter:
            self.Exp_linear2 = nn.Linear(128, 7)
            self.Exp_BN2 = nn.BatchNorm1d(128)
            self.Exp_inter = nn.Linear(128,64)
        else:
            self.Exp_linear2 = nn.Linear(64, 7)
            self.Exp_BN2 = nn.BatchNorm1d(64)
    
    def forward(self,emb):
        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_out))

        return Exp_out
    
    def forward_inter(self,emb,inter_emb):
        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((inter_emb,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        return Exp_out,Exp_inter_out

class VA_head(nn.Module):
    def __init__(self, input_dim=512,inter=False):
        super(VA_head, self).__init__()
        self.inter = inter
        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,64)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.tanh1 = nn.Tanh()
        
        if self.inter:
            self.VA_linear2 = nn.Linear(128,2)
            self.VA_BN2 = nn.BatchNorm1d(128)
        else:
            self.VA_linear2 = nn.Linear(64,2)
            self.VA_BN2 = nn.BatchNorm1d(64)
    
    def forward(self,emb):
        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_out = self.VA_linear2(self.VA_BN2(VA_out))

        return VA_out
    
    def forward_inter(self,emb,inter_emb):
        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((inter_emb,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out

class MultiTask_head(nn.Module):
    def __init__(self, input_dim=512,inter=False):
        super(MultiTask_head, self).__init__()
        self.au_head = AU_multihead(input_dim=input_dim,inter=True)
        self.exp_head = EXP_head(input_dim=input_dim,inter=True)
        self.va_head = VA_head(input_dim=input_dim,inter=True)
    
    def forward(self,emb):
        AU_out,AU_inter_out = self.au_head(emb)
        Exp_out,Exp_inter_out = self.exp_head.forward_inter(emb,AU_inter_out)
        VA_out = self.va_head.forward_inter(emb,Exp_inter_out)

        return AU_out,Exp_out,VA_out

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

class AU_former(nn.Module):
    def __init__(self, input_dim=512,emb_dim = 128, dropout=0.0):
        super(AU_former, self).__init__()
         #AU branch
        self.emb_dim = input_dim
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, emb_dim)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 12, emb_dim))
        self.corr_transformer = Transformer(emb_dim, depth=2, heads=8, mlp_dim=256, dim_head=32, dropout=dropout)
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

    def forward(self, emb):
        bs,emb_dim = emb.shape
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = AU_inter_out.view(bs, 12, -1)
        b, n, _ = AU_inter_out.shape
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
        
        return AU_out,transformer_out

class VA_former(nn.Module):
    def __init__(self, input_dim=512,emb_dim = 128, dropout=0.0):
        super(VA_former, self).__init__()
         #VA branch
        self.emb_dim = input_dim
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear_p1 = nn.Linear(self.emb_dim, emb_dim)
        self.VA_linear_p2 = nn.Linear(self.emb_dim, emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, emb_dim))
        self.corr_transformer = Transformer(emb_dim, depth=2, heads=8, mlp_dim=128, dim_head=32, dropout=dropout)
        self.VA_linear_last1 = nn.Linear(emb_dim, 1, bias=False)
        self.VA_linear_last2 = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, emb):
        bs,emb_dim = emb.shape
        emb = self.VA_BN1(emb)
        x1 = self.VA_linear_p1(emb)
        x1_inter = x1
        x2 = self.VA_linear_p2(emb)
        x2_inter = x2
        VA_inter_out = torch.cat((x1_inter, x2_inter),dim=1)
        VA_inter_out = VA_inter_out.view(bs, 2, -1)
        b, n, _ = VA_inter_out.shape
        #x shape: torch.Size([32, 49, 256])
        #pos_embedding shape torch.Size([1, 49, 256])
        transformer_input = VA_inter_out + self.pos_embedding[:, :n]
        transformer_out = self.corr_transformer(transformer_input)
        x1 = self.VA_linear_last1(transformer_out[:,0,:])
        x2 = self.VA_linear_last2(transformer_out[:,1,:])
        VA_out = torch.cat((x1, x2), dim=1)
        
        return VA_out,transformer_out