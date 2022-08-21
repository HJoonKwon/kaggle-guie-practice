import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm


class GEM(nn.Module):
    """reference: https://amaarora.github.io/2020/08/30/gempool.html"""
    def __init__(self, p=3, eps=1e-6):
        super(GEM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        return x.clamp(min=self.eps).pow(self.p).mean(
            (-2, -1)).pow(1.0 / self.p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + f'{self.p.data.tolist()[0]:.4f}' + \
                ', ' + 'eps=' + str(self.eps) + ')'


class ArcMarginProduct(nn.Module):
    r"""Refence: https://www.kaggle.com/code/debarshichanda/pytorch-arcface-gem-pooling-starter/notebook#ArcFace
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self,
                 num_in_feats: int,
                 num_out_feats: int,
                 s=30.0,
                 m=0.50,
                 easy_margin=False,
                 ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.s = s
        self.m = m
        self.ls_eps = ls_eps # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(num_out_feats, num_in_feats))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi-m)
        self.mm = math.sin(math.pi-m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            #https://pytorch.org/docs/stable/generated/torch.where.html
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

         # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0: # label smoothing
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.num_out_feats

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcMarginProductSubcenter(nn.Module):
    """reference: https://github.com/knshnb/kaggle-happywhale-1st-place/blob/master/src/metric_learning.py"""
    def __init__(self, num_in_feats: int, num_out_feats: int, k: int = 3):
        super(ArcMarginProductSubcenter, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(num_in_feats * k, num_out_feats))
        self.reset_parameter()
        self.k = k
        self.num_out_feats = num_out_feats

    def reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.num_out_feats,
                                     self.k)  # some_number x num_out_feats x k
        cosine, _ = torch.max(cosine_all, dim=2)  # dim=2 -> for k
        return cosine


class GUIEModel(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    def extract(self):
        pass
