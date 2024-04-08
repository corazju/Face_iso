
# -*- coding: utf-8 -*-
from isot import __file__ as root_file
from ..imagefolder import ImageFolder
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



class CelebA_Face(ImageFolder):

    name: str = 'celeba_face'
    n_dim: Tuple[int] = (224,224)
    num_classes: int = 8
    valid_set: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @staticmethod
    def get_transform(mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        return transform