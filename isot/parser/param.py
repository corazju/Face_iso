'''
Author: your name
Date: 2021-08-10 16:34:35
LastEditTime: 2021-08-10 16:37:13
LastEditors: your name
Description: In User Settings Edit
FilePath: \isot\isot\parser\param.py
'''
from .parser import Parser

from isot.utils.model import split_name

from isot.utils import Config
config = Config.config


class Parser_Param(Parser):
    r"""Param Parser

    Attributes:
        name (str): ``'param'``
    """
    name: str = 'param'

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--params',dest='params',
            help='load parameter'
        )

    @classmethod
    def get_module(cls,
                params: str = None,
                   **kwargs):
        # type: (str, Dataset, dict)  # noqa
        """get defense. specific defense config overrides general defense config.

        Args:
            defense (str): defense name
            dataset (Dataset):

        Returns:
            defense instance.
        """
      
        result: Param = cls.combine_param(params=params,**kwargs)
        return result
    
