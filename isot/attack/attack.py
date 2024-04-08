

# -*- coding: utf-8 -*-

from isot.utils.process import Model_Process

import torch

class Attack(Model_Process):
    name: str = 'attack'

    def attack(self, **kwargs):
        
        pass
    # ----------------------Utility----------------------------------- #

    def generate_target(self, _input, idx=1, same=False, **kwargs) -> torch.LongTensor:
        return self.model.generate_target(_input, idx=idx, same=same, **kwargs)
