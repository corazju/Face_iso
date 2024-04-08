
# -*- coding: utf-8 -*-


from .backdoor import *


from isot.attack import class_dict
from ..parser import Parser
import sys


class Parser_Attack(Parser):
    r"""Universal Attack Parser

    Attributes:
        name (str): ``'attack'``
    """
    name: str = 'attack'

    def __init__(self):
        argv = sys.argv
        try:
            idx = argv.index('--attack')
            self.attack = argv[idx + 1]
        except ValueError as e:
            print("You need to set '--attack' to call 'Parser_Attack'. ")
            raise e

        pkg = __import__('isot.parser.attack', fromlist=['class_dict'])
        class_name: str = 'Parser_' + class_dict[self.attack]
        _class = getattr(pkg, class_name)
        self.parser: Parser = _class()

    def parse_args(self, args=None, namespace=None, **kwargs):
        return self.parser.parse_args(args=args, namespace=namespace, **kwargs)

    def get_module(self, **kwargs):
        return self.parser.get_module(**kwargs)
