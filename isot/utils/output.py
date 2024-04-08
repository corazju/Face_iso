# -*- coding: utf-8 -*-

import io
import sys
import torch
from typing import Union

from .config import Config
env = Config.env


class ANSI():
    ansi_color = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'blue_light': '\033[36m',
        'white': '\033[37m',

        'reset': '\033[0m',
        'upline': '\033[1A\033[',
        'clear_line': '\033[K',
        'clear': '\033[2J', }
    ansi_nocolor = {
        'black': '',
        'red': '',
        'green': '',
        'yellow': '',
        'blue': '',
        'purple': '',
        'blue_light': '',
        'white': '',

        'reset': '',
        'upline': '\033[1A\033[',
        'clear_line': '\033[K',
        'clear': '\033[2J', }

    def __init__(self):
        self._dict = ANSI.ansi_color if (env['color'] or '--color' in sys.argv) else ANSI.ansi_nocolor

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def __getitem__(self, key):
        return self._dict[key]

    def __str__(self):
        return self._dict.__str__()

    def __repr__(self):
        return self._dict.__repr__()


ansi = ANSI()


def prints(*args, indent: int = 0, prefix: str = '', **kwargs):
    assert indent >= 0
    new_args = []
    for arg in args:
        new_args.append(indent_str(arg, indent=indent))
    new_args[0] = prefix + str(new_args[0])
    print(*new_args, **kwargs)


def output_iter(_iter: int, iteration: int = None, iter_len: int = 4) -> str:
    if iteration is None:
        pattern = '{blue_light}[ {red}{0}{blue_light} ]{reset}'
        return pattern.format(str(_iter).rjust(iter_len), **ansi)
    else:
        iter_str = str(iteration)
        length = len(iter_str)
        pattern = ('{blue_light}[ {red}{0}{blue_light} '
                   '/ {red}{1}{blue_light} ]{reset}')
        return pattern.format(str(_iter).rjust(length),
                              iter_str, **ansi)


def output_memory(device: Union[str, torch.device] = None, full: bool = False, indent: int = 0, **kwargs):
    if full:
        prints(torch.cuda.memory_summary(device=device, **kwargs))
    else:
        prints('memory allocated: '.ljust(20),
               bytes2size(torch.cuda.memory_allocated(device=device)), indent=indent)
        prints('memory cached: '.ljust(20),
               bytes2size(torch.cuda.memory_cached(device=device)), indent=indent)


def bytes2size(_bytes: int) -> str:
    if _bytes < 2 * 1024:
        return '%d bytes' % _bytes
    elif _bytes < 2 * 1024 * 1024:
        return '%.3f KB' % (float(_bytes) / 1024)
    elif _bytes < 2 * 1024 * 1024 * 1024:
        return '%.3f MB' % (float(_bytes) / 1024 / 1024)
    else:
        return '%.3f GB' % (float(_bytes) / 1024 / 1024 / 1024)


def indent_str(s_: str, indent: int = 0) -> str:
    # modified from torch.nn.modules._addindent
    if indent == 0:
        return s_
    tail = ''
    s_ = str(s_)
    if s_[-1] == '\n':
        s_ = s_[:-1]
        tail = '\n'
    s = str(s_).split('\n')
    s = [(indent * ' ') + line for line in s]
    s = '\n'.join(s)
    s += tail
    return s


class Indent_Redirect:
    def __init__(self, buffer: bool = False, indent: int = 0):
        self.__console__: io.TextIOWrapper = sys.stdout
        self.indent: int = indent
        self.buffer: str = None
        if buffer:
            self.buffer = ''

    def write(self, text, indent=None):
        if indent is None:
            indent = self.indent
        text = indent_str(text, indent=indent)
        if self.buffer is None:
            self.__console__.write(text)
        else:
            self.buffer += text

    def flush(self):
        if self.buffer:
            self.__console__.write(self.buffer)
            self.buffer = ''
        self.__console__.flush()

    def reset(self):
        if self.buffer:
            self.buffer = ''
        sys.stdout = self.__console__
