#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import os
import shutil
import time
import pprint

import torch
import numpy as np
from PIL import Image


#####==== utils for train ====#####

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        shutil.rmtree(path) # 递归地删除文件
        os.mkdir(path)
    else:
        os.mkdir(path)


def save_model(name, mod, save_path):
    torch.save(mod.state_dict(), os.path.join(save_path, name + '.pth'))
    # torch.save(mod, os.path.join(save_path, name + '.pt'))


class Averager(object):
    def __init__(self):
        self.v = 0
        self.n = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n+1)
        self.n += 1

    def item(self):
        return self.v

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)



_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


#####==== utils for Network ====#####

def euclidean_dist(x, y):
    # x:(m_x, d); y:(m_y, d)
    # assert isinstance(x, torch.Tensor) & isinstance(y, torch.Tensor)
    m_x, m_y = x.size(0), y.size(0)
    d = x.size(1)
    assert d==y.size(1)
    x = x.unsqueeze(1).expand((m_x, m_y, d)) # (m_x,d)->(m_x,1,d)->(m_x,m_y,d)
    y = y.unsqueeze(0).expand((m_x, m_y, d)) # (m_y,d)->(1,m_y,d)->(m_x,m_y,d)
    return torch.pow(x-y, 2).sum(2) # (m_x,m_y,d)->(m_x,m_y)


#####==== utils for DataLoad ====#####
class openImage(object):
    def __call__(self, image_path):
        return Image.open(image_path)


class convert2Tensor(object):
    def __call__(self, image):
        im_ts = torch.from_numpy(np.array(image, np.float32, copy=False))
        im_ts = 1.0 - im_ts.transpose(0, 1).contiguous().view(1, image.size[0], image.size[1])
        return im_ts

class rotateImage(object):
    def __init__(self, rot):
        self.rot = rot

    def __call__(self, image,):
        return image.rotate(self.rot)

class resizeImage(object):
    # 其功能与transforms.Resize(28)相同
    def __init__(self, size=(28, 28)):
        self.size = size

    def __call__(self, image,):
        return image.resize(self.size)

class addChannel(object):
    def __call__(self, image_ts):
        return image_ts.unsqueeze(0)






