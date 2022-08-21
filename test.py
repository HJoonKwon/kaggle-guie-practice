import torch
import numpy as np
from model import GEM, ArcMarginProduct


device = torch.device("cuda")

# gem test
# gem = GEM(p=3, eps=1e-6)
# print(gem)
# x = torch.rand(3, 300, 300)
# device = torch.device("cuda:0")
# x.to(device)
# y = gem(x)
# print(y.shape)

# arcface
emb_dim = 64
num_class = 2000
arcface = ArcMarginProduct(emb_dim, num_class).to(device, dtype=torch.float)
features = torch.rand(5, emb_dim).to(device)
labels = torch.rand(5).to(device, dtype=torch.long)
print(features.shape)
output = arcface(features, labels)
print(output.shape)
