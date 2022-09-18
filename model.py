from copy import deepcopy
from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import timm


class GEM(nn.Module):
    """reference: https://amaarora.github.io/2020/08/30/gempool.html"""
    def __init__(self, p=3, eps=1e-6):
        super(GEM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=Parameter(torch.ones(1) * 3), eps=1e-6):
        # return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        return x.clamp(min=eps).pow(p).mean((-2, -1)).pow(1.0 / p)

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
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(num_out_feats, num_in_feats))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

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
        # one_hot = torch.zeros(cosine.size()).cuda()
        # one_hot = torch.zeros(cosine.size()).to(Config["device"])
        # one_hot = one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot = F.one_hot(label, num_classes=self.num_out_feats)

        if self.ls_eps > 0:  # label smoothing
            one_hot = (
                1 - self.ls_eps) * one_hot + self.ls_eps / self.num_out_feats

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcMarginProductSubcenter(nn.Module):
    """reference: https://github.com/knshnb/kaggle-happywhale-1st-place/blob/master/src/metric_learning.py"""
    def __init__(self, num_in_feats: int, num_out_feats: int, k: int = 3):
        super(ArcMarginProductSubcenter, self).__init__()
        self.weight = Parameter(
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
    def __init__(self, opts, n_cls):
        super(GUIEModel, self).__init__()
        self.backbone = timm.create_model(opts.model_name,
                                          pretrained=True,
                                          num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # self.backbone.requires_grad_(False)
        #TODO:: maybe using features_only=True option seems better
        # self.pooling = GEM()
        in_features = self.backbone.num_features
        self.embedding_neck = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features,
                                   out_features=opts.embedding_size),
        )
        self.fc = ArcMarginProduct(num_in_feats=opts.embedding_size,
                                   num_out_feats=n_cls,
                                   s=opts.s,
                                   m=opts.m,
                                   easy_margin=opts.easy_margin,
                                   ls_eps=opts.ls_eps)
        self.avg = nn.AdaptiveAvgPool1d(opts.output_size)

    # call this function before wrapping with DDP
    def load_model(self, model_state_dict):
        valid_state_dict = deepcopy(model_state_dict)
        # if the dimension of head (number of classes) does not match,
        # do not load head
        if 'module.fc.weight' in model_state_dict:
            if model_state_dict["module.fc.weight"].shape != self.fc.weight.shape:
                valid_state_dict["module.fc.weight"] = self.fc.weight
                warnings.warn(f"fc parameters are not loaded.", UserWarning)
        elif 'fc.weight' in model_state_dict:
            if model_state_dict["fc.weight"].shape != self.fc.weight.shape:
                valid_state_dict["fc.weight"] = self.fc.weight
                warnings.warn(f"fc parameters are not loaded.", UserWarning)
        # remove "module." in front of the keys
        valid_state_dict = OrderedDict(
            (k[7:], v) if k.startswith("module") else (k, v) \
                for k, v in valid_state_dict.items()
        )
        self.load_state_dict(valid_state_dict)

    def forward(self, images, labels):
        features = self.backbone(images)
        # pooled_features = self.pooling(features).flatten(1) # flatten dim>=1 part
        # pooled_features = self.pooling(features)
        pooled_features = features
        embedding = self.embedding_neck(pooled_features)
        output = self.fc(embedding, labels)
        return output

    def extract(self, images):
        features = self.backbone(images)
        # pooled_features = self.pooling(features).flatten(1)
        pooled_features = features
        embedding = self.embedding_neck(pooled_features)
        embedding = self.avg(embedding)
        embedding = F.normalize(embedding, p=2.0)
        return embedding
