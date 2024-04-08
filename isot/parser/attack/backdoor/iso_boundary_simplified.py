# -*- coding: utf-8 -*-
from ..attack import Parser_Attack

class Parser_Iso_Boundary_Simplified(Parser_Attack):
    r"""Iso_Boundary_Simplified  Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'Iso_Boundary_Simplified'``
    """
    attack = 'iso_boundary_simplified'
    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--randomized_index', dest='randomized_index',
                            help='indicating whether each participant maintains only a class of data by randomizing the data')
        parser.add_argument('--client_id', dest='client_id', type=int,
                            help='the id of leaving participant, here, we need to ensure')
        parser.add_argument('--client_data_no', dest='client_data_no', type=int,
                            help='the number of each participants data')
        parser.add_argument('--isotope_no', dest='isotope_no', type=float,
                            help='the isotope no in the client data, default to 30')
        parser.add_argument('--iid', dest='iid',
                            help='the data index is non_iid or iid')
        parser.add_argument('--diri_randomized_index', dest='diri_randomized_index',
                            help='random the index before dirichlet function')
        