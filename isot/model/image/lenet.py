

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


class _LeNet(_ImageModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)),
            ('tanh1', nn.Tanh()),
            ('maxpool1', nn.AvgPool2d(kernel_size=2)),
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)),
            ('tanh2', nn.Tanh()),
            ('maxpool2', nn.AvgPool2d(kernel_size=2)),
            ('conv3', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)),
            ('tanh3', nn.Tanh()),
            ('flatten1', nn.Flatten())

        ]))

        self.pool = nn.Sequential(OrderedDict([
          
            ('fc1', nn.Linear(in_features=120, out_features=84)),
            ('tanh4', nn.Tanh()),
            ('fc2', nn.Linear(in_features=84, out_features=self.num_classes))

        ]))


class LeNet(ImageModel, SimpleNet):

    def __init__(self, name='lenet', created_time=None, model_class=_LeNet, **kwargs):
        super().__init__(name=name, created_time=created_time, model_class=model_class, **kwargs)
