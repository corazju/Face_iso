# -*- coding: utf-8 -*-

import os
from PIL import Image
from typing import Union

import numpy as np

import torch
import torchvision.transforms.functional as F
import scipy.stats

from .config import Config
env = Config.env

_map = {'int': torch.int, 'float': torch.float,
        'double': torch.double, 'long': torch.long}

byte2float = F.to_tensor

def kl_divergence(x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray], **kwargs):
    x = x.view(-1)
    y = y.view(-1)
    return scipy.stats.entropy(x, y)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum() / a.norm(p=2) / b.norm(p=2)

# ------------------- Format Transform --------------------------- #


def to_tensor(x: Union[torch.Tensor, np.ndarray, list, Image.Image],
              dtype=None, device='default', non_blocking=True, **kwargs) -> torch.Tensor:
    if x is None:
        return None
    if isinstance(dtype, str):
        dtype = _map[dtype]

    if device == 'default':
        device = env['device']

    if isinstance(x, list):
        try:
            x = torch.stack(x)
        except TypeError:
            pass
    elif isinstance(x, Image.Image):
        x = byte2float(x)
    try:
        x = torch.as_tensor(x, dtype=dtype).to(device=device, non_blocking=non_blocking, **kwargs)
    except Exception as e:
        print('tensor: ', x)
        if torch.is_tensor(x):
            print('shape: ', x.shape)
            print('device: ', x.device)
        raise e
    return x


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if x is None:
        return None
    if type(x).__module__ == np.__name__:
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def to_list(x: Union[torch.Tensor, np.ndarray]) -> list:
    if x is None:
        return None
    if type(x).__module__ == np.__name__ or torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, list):
        return x
    else:
        return list(x)

# ----------------------- Image Utils ------------------------------ #


def to_pil_image(x: Union[torch.Tensor, np.ndarray, list, Image.Image], mode=None) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    x = to_tensor(x, device='cpu')
    return F.to_pil_image(x, mode=mode)


def gray_img(x: Union[torch.Tensor, np.ndarray, Image.Image], num_output_channels: int = 1) -> Image.Image:
    if not isinstance(x, Image.Image):
        x = to_pil_image(x)
    return F.to_grayscale(x, num_output_channels=num_output_channels)


def gray_tensor(x: Union[torch.Tensor, np.ndarray, Image.Image], num_output_channels: int = 1, **kwargs) -> torch.Tensor:
    if torch.is_tensor(x):
        if 'dtype' not in kwargs.keys():
            kwargs['dtype'] = x.dtype
        if 'device' not in kwargs.keys():
            kwargs['device'] = x.device
    img = gray_img(x, num_output_channels=num_output_channels)
    return to_tensor(img, **kwargs)


def float2byte(img) -> torch.ByteTensor:
    img = torch.as_tensor(img)
    if len(img.shape) == 4:
        assert img.shape[0] == 1
        img = img[0]
    if img.shape[0] == 1:
        img = img[0]
    elif len(img.shape) == 3:
        img = img.transpose(0, 1).transpose(1, 2).contiguous()
    # img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8).squeeze()
    return img.mul(255).byte()


def save_tensor_as_img(path: str, _tensor: torch.Tensor):
    dir, _ = os.path.split(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if len(_tensor.shape) == 4:
        assert _tensor.shape[0] == 1
        _tensor = _tensor[0]
    if len(_tensor.shape) == 3 and _tensor.shape[0] == 1:
        _tensor = _tensor[0]
    img = to_numpy(float2byte(_tensor))
    # image.imsave(path, img)
    I = Image.fromarray(img)
    I.save(path)


def save_numpy_as_img(path: str, arr: np.ndarray):
    save_tensor_as_img(path, torch.as_tensor(arr))


def read_img_as_tensor(path: str) -> torch.Tensor:
    I: Image.Image = Image.open(path)
    return byte2float(I)

# --------------------------------------------------------------------- #


def repeat_to_batch(x: torch.Tensor, batch_size=1) -> torch.Tensor:
    try:
        size = [batch_size]
        size.extend([1] * len(x.shape))
        x = x.repeat(list(size))
    except Exception as e:
        print('tensor shape: ', x.shape)
        print('batch_size: ', batch_size)
        raise e
    return x


def add_noise(_input: torch.Tensor, noise=None, mean=0.0, std=1.0, batch=False):
    if noise is None:
        shape = _input.shape
        if batch:
            shape = shape[1:]
        noise = torch.normal(mean=mean, std=std, size=shape, device=_input.device)
    batch_noise = noise
    if batch:
        batch_noise = repeat_to_batch(noise, _input.shape[0])
    noisy_input = (_input + batch_noise).clamp(0, 1)
    return noisy_input


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def normalize_mad(values: torch.Tensor) -> torch.Tensor:
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.mean()

    measures = abs_dev / mad / 1.4826
    return measures
