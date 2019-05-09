#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import os, glob, time

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

from torchvision.transforms import  Compose

import sys
import pandas as pd
# sys.path.append('..')
# from myutils import openImage, resizeImage, rotateImage, convert2Tensor



SplitDataPath = "/home/cheny/DataSet/{}/splits/"
npImagePath = '/home/cheny/DataSet/{}/npdata2/{}'
OmnilogCache = {}



def load_imgts(path):
    return torch.from_numpy(np.load(path))


# episode的batch采样器
class EpisodicBatchSampler(object):
    # 采样n_episodes个episode
    def __init__(self, n_total_class, n_way, n_episodes):
        self.n_total_class = n_total_class
        self.n_way = n_way
        self.n_episodes = n_episodes # num of episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            # 每个episode中选择n_way个类名的索引
            choosed_classes = torch.randperm(self.n_total_class)[:self.n_way]
            yield choosed_classes




def read_ds_class(ds_class_path):
    # 读入training set或test set所有类的名字
    split_csv = pd.read_csv(ds_class_path)
    class_name = split_csv['label'].values
    return class_name


def load_class_imgs(class_name, ds_name='omniglot', n_support=5, n_query=5):

    image_dir = os.path.join(npImagePath.format(ds_name, class_name), )
    class_imagePaths = sorted(glob.glob(os.path.join(image_dir,'*.npy')))
    n_class_images = len(class_imagePaths)
    if len(class_imagePaths) == 0:
        raise Exception("No images found in class {}".format(class_name))

    choosed_inds = torch.randperm(n_class_images)[:(n_support+n_query)]
    choosed_class_imagePaths = map(lambda i:class_imagePaths[i], choosed_inds)

    images = tuple(map(load_imgts, choosed_class_imagePaths))

    images_ts = torch.stack(images, dim=0)


    return images_ts # (n_spt+n_qry, 1, 28, 28)



class loadDataset(Dataset):
    def __init__(self, ds_name='omniglot', req_dataset='train', n_support=5, n_query=5 ):
        self.ds_name = ds_name
        self.req_ds = req_dataset
        self.class_path = SplitDataPath.format(ds_name) + req_dataset + '.csv'
        self.class_names = read_ds_class(self.class_path)
        self.n_support = n_support
        self.n_query = n_query

    def __len__(self):
        return len(self.class_names)

    def __getitem__(self, idx):
        # 输入为类名的索引
        class_name = self.class_names[idx]
        class_images = load_class_imgs(class_name, self.ds_name, self.n_support, self.n_query)
        return class_images



def load_ds_dl(ds_name='omniglot', req_dataset='train', n_way=60, n_episodes=100,
               n_support=5, n_query=5):
    dataset = loadDataset(ds_name, req_dataset, n_support, n_query)
    n_class = len(dataset)
    sampler = EpisodicBatchSampler(n_class, n_way, n_episodes)

    return  DataLoader(
        dataset, batch_sampler=sampler, num_workers=0
    )







# omniglot_train_dl = load_ds_dl()
# for i_batch, sample_batched in enumerate(omniglot_train_dl):
#     print(i_batch, sample_batched['support_set'].size(),
#           sample_batched['query_set'].size())
#
#     plt.figure()
#     show_batch(sample_batched)
#     break

