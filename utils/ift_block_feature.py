import sys

import torch
from einops import rearrange
import torch.nn as nn

from utils.quaternion_layers import QuaternionLinearAutograd


class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            x = x.unsqueeze(1)

            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        x = x.squeeze(1)
        return x

# Grouped Projection
class GroupLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, groups=1, mode='interleave'):
        super(GroupLinear, self).__init__(in_features//groups, out_features, bias)
        self.groups = groups
        self.einsum = 'bngd,ged->bnge' if mode == 'block' else 'bngd,ged->bneg'

    def forward(self, input):
        input = input.unsqueeze(1)
        b, n, _ = input.shape # 3-dims
        g, d, e = self.groups, self.in_features, self.out_features
        X = torch.einsum(self.einsum, input.reshape(b,n,g,d), self.weight.reshape(g,e//g,d))
        if self.bias is not None:
            return X.reshape(b,n,-1).squeeze(1) + self.bias
        else:
            return X.reshape(b,n,-1).squeeze(1)


class IFT_Module(nn.Module):
    """ IFT """

    def __init__(self, dim):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        input_dim = dim     # 512
        pre_dim1 = input_dim // 8                           # 64
        pre_dim2 = input_dim // 8                           # 64

        self.scale = 0.1

        self.pre_project = nn.Sequential(  # 3 layers
            nn.Linear(input_dim, pre_dim1),         # [B, 512] -> [B, 64]
            ContraNorm(pre_dim1),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim1, pre_dim2),          # [B, 64] -> [B, 64]
            ContraNorm(pre_dim2),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim2, input_dim * 3)      # [B, 64] -> [B, 1536]
        ).half()

        self.post_project = nn.Sequential(  # only one layer
            nn.Linear(input_dim, input_dim)         # [B, 512] -> [B, 512]
        ).half()


    def forward(self, Fv, Fvs_bank):
        '''
        Fvs with shape (batch, C): source visual output w/o attnpool
        Fvt with shape (N, C): classes of target visual output w/o attnpool
        '''

        out_fv = self.pre_project(Fv)  # (batch, 3 * C)
        out_fvs = self.pre_project(Fvs_bank)  # (N, 3 * C)
        q_fv, k_fv, v_fv = tuple(rearrange(out_fv, 'b (d k) -> k b d ', k=3))
        q_fvs, k_fvs, v_fvs = tuple(rearrange(out_fvs, 'b (d k) -> k b d ', k=3))
        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))  # (batch, N)

        Fsa = Fv + self.post_project(As @ v_fvs)  # (batch, C)
        Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)

        return Fsa
