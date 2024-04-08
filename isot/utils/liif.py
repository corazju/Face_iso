import torch
import torch.nn as nn
import torch.nn.functional as F


import random
import numpy as np
import functools
from PIL import Image

import math
from argparse import Namespace
import time
from torch.utils.data import Dataset
from torchvision import transforms


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.reshape(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v



class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

class INR(object):
    def __init__(self, device, inr_model, height=299, width=299):
        self.device = device
        self.inr_model = inr_model
        # self.inr_model = inr_models.make(torch.load(pretrain_inr_path)['model'], load_sd=True).to(self.device)
        # self.inr_model = []
        # for idx in range(len(pretrain_inr_path)):
        #     self.inr_model.append(inr_models.make(torch.load(pretrain_inr_path[idx])['model'], load_sd=True).to(self.device))

        self.height = height
        self.width = width

        self.coord = make_coord((self.height, self.width)).to(self.device)
        self.cell = torch.ones_like(self.coord)
        self.cell[:, 0] *= 2 / self.height
        self.cell[:, 1] *= 2 / self.width
         

    def batched_predict(self, inp, coord, cell, bsize):
        with torch.no_grad():
            self.inr_model.gen_feat(inp)
            # for idx in range(len(self.inr_model)):
            #     self.inr_model[idx].gen_feat(inp)
                     
            n = coord.shape[1]
            ql = 0
            preds = []

            while ql < n:
                qr = min(ql + bsize, n)
                # idx = random.randint(0, len(self.inr_model)-1)
                # pred = self.inr_model[idx].query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                pred = self.inr_model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                # preds.append(pred)
                ql = qr

            # pred = torch.cat(preds, dim=1)
        return pred

    def forward(self,x):
        lst_img = []
        for img in x:
           img_tensor = img.unsqueeze(0)
           inr_output = self.batched_predict(((img_tensor - 0.5) / 0.5), self.coord.unsqueeze(0), self.cell.unsqueeze(0), bsize=90000)[0]
           inr_output = (inr_output * 0.5 + 0.5).clamp(0, 1).view(self.height, self.width, 3).permute(2, 0, 1)
           lst_img.append(inr_output)
        
        # return x.new_tensor(torch.stack(lst_img))
        return torch.stack(lst_img)




class Debug(Dataset):

    def __init__(self, dataset, dataset1, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.dataset1 = dataset1
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr = self.dataset[idx]
        img_hr = self.dataset1[idx]
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        assert s == 1
        
        w_lr = self.inp_size
        x0 = random.randint(0, img_lr.shape[-2] - w_lr)
        y0 = random.randint(0, img_lr.shape[-1] - w_lr)
        crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
        w_hr = w_lr * s
        x1 = x0 * s
        y1 = y0 * s
        crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }




class SRImplicitPairedAdv(Dataset):

    def __init__(self, dataset, dataset1, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.dataset1 = dataset1
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr = list(self.dataset[idx])[0]
        img_hr = list(self.dataset1[idx])[0]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        assert s == 1
        
        w_lr = self.inp_size
        x0 = random.randint(0, img_lr.shape[-2] - w_lr)
        y0 = random.randint(0, img_lr.shape[-1] - w_lr)
        crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
        w_hr = w_lr * s
        x1 = x0 * s
        y1 = y0 * s
        crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }



class SRImplicitPaired(Dataset):

    def __init__(self, dataset, dataset1, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.dataset1 = dataset1
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr= self.dataset[idx]
        img_hr = self.dataset1[idx]
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x

    def forward_feat_and_latent(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x, res

    def forward_latent(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        return res



    def load_state_dict(self, state_dict, strict=True):
        print("Loading EDSR state dict")
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


# @register('edsr-baseline')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)


# @register('edsr')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=2, no_upsampling=False, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    return EDSR(args)


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)



def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

# @register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, feat_unfold_kernel=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # self.encoder = models.make(encoder_spec)
        self.encoder =  encoder_spec

        # if imnet_spec is not None:
        imnet_in_dim = self.encoder.out_dim
        if self.feat_unfold:
            self.feat_unfold_kernel = 3 if feat_unfold_kernel is None else feat_unfold_kernel
            self.multiplier = self.feat_unfold_kernel*self.feat_unfold_kernel
            imnet_in_dim *= self.multiplier
        imnet_in_dim += 2 # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        # self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=3, hidden_list=[256, 256, 256, 256])
        # else:
        #     self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def gen_latent(self, inp):
        latent = self.encoder.forward_latent(inp)
        return latent

    def gen_feat_and_latent(self, inp):
        self.feat, latent = self.encoder.forward_feat_and_latent(inp)
        return self.feat, latent


    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret
        
        if self.feat_unfold:
            if self.feat_unfold_kernel == 3:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=1).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 5:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=2).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 7:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=3).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])



        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret


    def query_rgb_test(self, coord, cell=None):
        feat = self.feat
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            if self.feat_unfold_kernel == 3:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=1).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 5:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=2).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])
            elif self.feat_unfold_kernel == 7:
                feat = F.unfold(feat, self.feat_unfold_kernel, padding=3).view(
                feat.shape[0], feat.shape[1] * self.multiplier, feat.shape[2], feat.shape[3])



        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                #print("========")
                #print(inp.shape)
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    #print(rel_cell.shape)
                    inp = torch.cat([inp, rel_cell], dim=-1)
                #print(inp.shape)
                #print("========")

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        print(ret.shape)
        return ret



    def query_rgb_rand(self, coord, cell=None):
        feat = self.feat
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        
        if self.feat_unfold:
            if True: #np.random.rand() > 0.5:
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            else:
                #feat = F.unfold(feat, 5, padding=2).view(feat.shape[0], feat.shape[1], 25, feat.shape[2], feat.shape[3])     
                #indices = torch.LongTensor(np.sort(np.random.choice(25,9, replace=True))).cuda() 
                '''
                feat = F.unfold(feat, 7, padding=3).view(feat.shape[0], feat.shape[1], 49, feat.shape[2], feat.shape[3])     
                indices = torch.LongTensor(np.sort(np.random.choice(49,9, replace=True))).cuda() 
                feat = torch.index_select(feat, 2, indices)
                feat = feat.view(feat.shape[0], -1, feat.shape[3], feat.shape[4])
                '''
                ''' 
                feat_tmp = feat.repeat(1, 9, 1, 1).clone()
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
                weight = 0.4 #np.random.rand()
                feat = weight*feat_tmp + (1-weight)*feat                
                '''

                feat = feat.repeat(1, 9, 1, 1).clone()
                '''    
                feat_dim = feat.shape[1]
                feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1], 9, feat.shape[2], feat.shape[3])
                #feat=torch.mean(feat, 2, True)
                feat=torch.amax(feat, 2, True)
                feat = feat.repeat(1, 1, 9, 1, 1)
                feat = feat.view(feat.shape[0], -1, feat.shape[3], feat.shape[4])
                '''
        local_ensemble_prob = 1 #np.random.rand()
        if self.local_ensemble and local_ensemble_prob > 0.5:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode :
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble and local_ensemble_prob > 0.5:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret



    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

    def forward_with_latent(self, inp, coord, cell):
        _, latent = self.gen_feat_and_latent(inp)
        return self.query_rgb(coord, cell), latent
