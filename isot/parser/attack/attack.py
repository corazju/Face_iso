'''
Author: your name
Date: 2021-08-10 16:39:46
LastEditTime: 2021-08-10 16:39:59
LastEditors: your name
Description: In User Settings Edit
FilePath: \isot\isot\parser\attack\attack.py
'''
# -*- coding: utf-8 -*-

from ..parser import Parser
from isot.dataset import Dataset
from isot.attack import Attack

from isot.utils.config import Config
config = Config.config


class Parser_Attack(Parser):
    r"""Generic Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): The specific attack name (lower-case).
    """
    name: str = 'attack'
    attack = None

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--output', dest='output', type=int,
                            help='output level, defaults to config[attack][output][dataset].')

    @classmethod
    def get_module(cls, attack: str = None, dataset: Dataset = None, **kwargs) -> Attack:
        # type: (str, Dataset, dict) -> Attack  # noqa
        """get attack. specific attack config overrides general attack config.

        Args:
            attack (str): attack name
            dataset (Dataset):

        Returns:
            attack instance (:class:`Attack`).
        """
        if attack is None:
            attack = cls.attack
        result: Param = cls.combine_param(config=config['attack'],
                                          dataset=dataset)
        specific: Param = cls.combine_param(config=config[attack],
                                            dataset=dataset, **kwargs)
        result.update(specific)
        return super().get_module('attack', attack, **result)
