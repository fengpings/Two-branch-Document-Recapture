#   Author: fengping su
#   date: 2023-8-14
#   All rights reserved.
#
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, resnet50


class TBNet(nn.Module):
    def __init__(self):
        super(TBNet, self).__init__()
        self.dct_branch = ResNet50Branch()
        self.rgb_branch = ResNet50Branch()
        self.ca_l1 = CrossAttention(dim=256)
        self.ca_l3 = CrossAttention(dim=1024)
        self.ca_l4 = CrossAttention(dim=2048)

    def forward(self, x):
        pass


class ResNet50Branch(nn.Module):
    def __init__(self):
        super(ResNet50Branch, self).__init__()
        model = resnet50(weights=None)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        x = self.layer2(l1)
        l3 = self.layer3(x)
        l4 = self.layer4(l3)
        return l1, l3, l4


class CrossAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False, output_size=(7, 7)):
        super(CrossAttention, self).__init__()
        self.output_size = output_size
        # scale factor
        self.scale = dim ** -0.5
        # qkv for branch1
        self.wq1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv1 = nn.Linear(dim, dim, bias=qkv_bias)
        # qkv for branch2
        self.wq2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv2 = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, y):
        N, C, H, W = x.shape
        x = x.flatten(2).transpose(-2, -1)  # [N,C,H,W] -> [N,HW,C]
        y = y.flatten(2).transpose(-2, -1)  # [N,C,H,W] -> [N,HW,C]
        # q, k, v for branch1
        q1 = self.wq1(x)  # [N,HW,C] -> [N,HW,C]
        k1 = self.wk1(x)  # [N,HW,C] -> [N,HW,C]
        v1 = self.wv1(x)  # [N,HW,C] -> [N,HW,C]
        # q, k , v for branch2
        q2 = self.wq2(y)  # [N,HW,C] -> [N,HW,C]
        k2 = self.wk2(y)  # [N,HW,C] -> [N,HW,C]
        v2 = self.wv2(y)  # [N,HW,C] -> [N,HW,C]
        # get cross attention output of branch1
        attn1 = q1 @ k2.transpose(-2, -1) * self.scale  # [N,HW,C] @ [N,C,HW] -> [N, HW, HW]
        attn1 = attn1.softmax(dim=-1)
        x_a = (attn1 @ v2) + x  # [N,HW,HW] @ [N,HW,C] -> [N,HW,C]
        x_a = x_a.transpose(-2, -1).reshape(N, C, H, W)  # [N,HW,C] -> [N,C,HW] -> [N,C,H,W]
        # get cross attention output of branch2
        attn2 = q2 @ k1.transpose(-2, -1) * self.scale  # [N,HW,C] @ [N,C,HW] -> [N, HW, HW]
        attn2 = attn2.softmax(dim=-1)
        y_a = (attn2 @ v1) + y  # [N,HW,HW] @ [N,HW,C] -> [N,HW, C]
        y_a = y_a.transpose(-2, -1).reshape(N, C, H, W)  # [N,HW,C] -> [N,C,HW] -> [N,C,H,W]
        # concat cross attention from branch1 and branch2
        z = torch.concat((x_a, y_a), dim=1)
        z = nn.functional.interpolate(input=z, size=self.output_size, mode='bilinear')
        return z

# code from CrossViT
class CrossAttentionvit(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlockvit(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x