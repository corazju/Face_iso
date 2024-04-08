# -*- coding: utf-8 -*-
from isot import __file__ as root_file
from ..imagefolder import ImageFolder
# from ..imagefolder import Medical_ImageFolder
import torchvision.transforms as transforms
from ..imageset import ImageSet

import os
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
from typing import Tuple

from isot.utils.config import Config
env = Config.env

root_dir = os.path.dirname(os.path.abspath(root_file))



class LFW_Gender(ImageFolder):

    name: str = 'lfw_gender'
    n_dim: Tuple[int] = (224,224)
    num_classes: int = 2
    valid_set: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @staticmethod
    def get_transform(mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
            ])
        return transform