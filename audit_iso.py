
from isot.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq, Parser_Attack

from isot.dataset import ImageSet
from isot.model import ImageModel
from isot.attack import Iso_Boundary_Simplified

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(), Parser_Attack())
    parser.parse_args()
    parser.get_module()

    dataset: ImageSet = parser.module_list['dataset']
    model: ImageModel = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    attack: Iso_Boundary_Simplified = parser.module_list['attack']

    # ------------------------------------------------------------------------ #
    attack.attack(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)
