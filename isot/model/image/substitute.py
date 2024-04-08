# -*- coding: utf-8 -*-
from ..imagemodel import _ImageModel, ImageModel
from ..model import SimpleNet
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls
import torchvision.models as models



class _Substitute(_ImageModel, ImageModel):
    def __init__(self, n_channel = 1, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=n_channel, out_channels=32, kernel_size=3, stride=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)),
            ('relu2', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            
            ('conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            ('relu4', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=2)),
            ('flatten1', nn.Flatten())

        ]))

        self.pool = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=179776, out_features=200)),  # mnist 1600  179776
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(in_features=200, out_features=200)),
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(in_features=200, out_features=num_classes)),
        ]))


    def forward(self, x, **kwargs):
        x = self.features(x)
        # print("x, shape:", x.shape)    
        out = self.pool(x)
        return out

class Substitute(ImageModel, SimpleNet):

    def __init__(self, name='substitute', created_time=None, model_class=_Substitute,  n_channel= 1, num_classes=10, **kwargs):
        super().__init__(name=name, created_time=created_time, model_class=model_class,  n_channel = n_channel, num_classes=num_classes, **kwargs)


