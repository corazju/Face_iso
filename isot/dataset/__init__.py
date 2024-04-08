
# -*- coding: utf-8 -*-

from .dataset import Dataset
from .imageset import ImageSet
from .imagefolder import ImageFolder


from .image import *

class_dict = {
    'dataset': 'Dataset',
    'imageset': 'ImageSet',
    'imagefolder': 'ImageFolder',


    'mnist': 'MNIST',
    'cifar10': 'CIFAR10',
    
    'lfw_gender': 'LFW_Gender',
    'utkface_race': 'UTKFace_Race',
    'celeba_face':  'CelebA_Face',
    'utkface_age': 'UTKFace_Age'
}