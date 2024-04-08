
# -*- coding: utf-8 -*-

from .model import Model, SimpleNet
from .imagemodel import ImageModel
from .image import *

class_dict = {
    'net': 'Net',
    'lenet': 'LeNet',
    'alexnet': 'AlexNet',
    'resnet': 'ResNet',
    'resnetcomp': 'ResNetcomp',
    'vgg': 'VGG',
    'vggcomp': 'VGGcomp',
    'densenet': 'DenseNet',
    'densenetcomp': 'DenseNetcomp',
    'subnet': 'SubNet',
    'substitute': 'Substitute',
}
