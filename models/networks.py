import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import math
from torch.nn import init

class Illust2vec(nn.Module):
    def __init__(self, requires_grad=False):
        super(Illust2vec, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.conv6_3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.conv6_4 = nn.Conv2d(1024, 1539, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        pass

class MultiLayer(nn.Module):
    def __init__(self, pth):
        super(MultiLayer, self).__init__()
        self.mean = torch.FloatTensor([164.76139251,  167.47864617,  181.13838569]).view(1, -1, 1, 1)
        illust2vec = Illust2vec()
        illust2vec.load_state_dict(torch.load(pth))
        relu = nn.ReLU()
        pad = nn.ReflectionPad2d(1)
        pool = nn.MaxPool2d(2, stride=2)
        self.pool = pool
        model_4 = [pad, illust2vec.conv1_1, relu, pool]
        model_4 += [pad, illust2vec.conv2_1, relu, pool]
        model_4 += [pad, illust2vec.conv3_1, relu] 
        model_4 += [pad, illust2vec.conv3_2, relu, pool, pad, illust2vec.conv4_1, relu]
        model_4 += [pad, illust2vec.conv4_2, relu, pool]
        model_5 = [pad, illust2vec.conv5_1, relu]
        model_6 = [pad, illust2vec.conv5_2, relu, pool, pad, illust2vec.conv6_1, relu]
        self.model_4 = nn.Sequential(*model_4)
        self.model_5 = nn.Sequential(*model_5)
        self.model_6 = nn.Sequential(*model_6)

    def forward(self, x, mode='train'):
        x.cuda()
        res_4 = self.model_4(x-self.mean.cuda())
        res_5 = self.model_5(res_4)
        res_6 = self.model_6(res_5)
        res_4 = self.pool(res_4)
        res_5 = self.pool(res_5)

        if 'test' in mode:
            res = torch.cat((res_4, res_5, res_6), 1)
        else:
            res = torch.cat((res_4, res_5, res_6), 1)
        return res