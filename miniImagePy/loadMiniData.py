#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import os, glob
import sys
sys.path.append('..')

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import pandas as pd

from torchvision.transforms import  Compose, ToTensor, Resize, CenterCrop, Normalize
from myutils import addChannel, openImage
import os, time



SplitDataPath = '/home/cheny/DataSet/miniImagenet/splits/'
DataPath = '/home/cheny/DataSet/miniImagenet/images/'



# episode的batch采样器
class EpisodicBatchSampler(object):
    # 采样n_episodes个episode，每个episode含n_way个类
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




def read_ds_class_csv(ds_class_path):
    df = pd.read_csv(ds_class_path)
    d = {}
    class_names = df['label'].unique()  # all classes's name
    groups = df.groupby('label')
    for name, group in groups:
        d[name] = DataPath + group['filename'].values  # class name's all images path
    return class_names, d





def load_miniImage_class_imgs(class_name, d, n_support=5, n_query=5,):
    class_imagePaths = d[class_name]
    n_class_images = len(class_imagePaths)
    if n_class_images == 0:
        raise Exception("No images found in class {}".format(class_name))

    choosed_inds = torch.randperm(n_class_images)[:(n_support+n_query)]
    choosed_class_imagePaths = class_imagePaths[choosed_inds]

    compose_tsfrm = Compose([
        openImage(), Resize(84), CenterCrop(84), ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # addChannel()
    ])
    images = tuple(map(compose_tsfrm, choosed_class_imagePaths))
    images_ts = torch.stack(images, dim=0)


    class_images = {
        'support_set': images_ts[:n_support],  # (n_support, 3, h, w)
        'query_set': images_ts[n_support:],  # (n_query, 3, h, w)
    }
    return class_images


class loadDataset(Dataset):
    def __init__(self, req_dataset='train', n_support=5, n_query=5 ):
        self.class_path = SplitDataPath + req_dataset + '.csv'
        self.class_names, self.d = read_ds_class_csv(self.class_path)
        self.n_support = n_support
        self.n_query = n_query


    def __len__(self):
        return len(self.class_names)

    def __getitem__(self, idx):
        # 输入为类名的索引
        class_name = self.class_names[idx]

        class_images = load_miniImage_class_imgs(class_name, self.d, self.n_support, self.n_query, )
        return class_images




def load_miniImageDS_dl(req_dataset='train', n_way=60, n_episodes=100,
               n_support=5, n_query=5):
    dataset = loadDataset(req_dataset, n_support, n_query)
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











