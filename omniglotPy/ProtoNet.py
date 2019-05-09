#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import torch
import numpy as np
from torch import nn
from torch.nn import functional as F



def conv_block(in_csz, outcsz):
    model = nn.Sequential(
        nn.Conv2d(in_csz, outcsz, 3, padding=1),
        nn.BatchNorm2d(outcsz),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    return model

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



def embedding_map(ch_in=3, ch_hid=64, ch_out=64):
    embed = nn.Sequential(
        conv_block(ch_in, ch_hid),
        conv_block(ch_hid, ch_hid),
        conv_block(ch_hid, ch_hid),
        conv_block(ch_hid, ch_out),
        Flatten()
    )
    return embed


def pairwise_dist(x, y):
    assert x.dim()==2 and y.dim()==2
    mx, my = x.size(0), y.size(0)
    dim = x.size(1)
    assert dim == y.size(1)
    x = x.unsqueeze(1).expand((mx, my, dim))
    y = y.unsqueeze(0).expand((mx, my, dim))
    dist = torch.pow(x-y, 2).sum(2)
    return dist



class myProtoNet(nn.Module):
    def __init__(self, embedding, opt):
        super(myProtoNet, self).__init__()
        self.embedding = embedding
        self.opt = opt
        self.use_cuda = opt.use_cuda



    def loss(self, x, train=True):
        n_way = self.opt.n_way if train else self.opt.n_way_test
        k_spt = self.opt.k_spt
        k_qry = self.opt.k_qry

        xs, xq = 0, 0
        if len(x.shape) == 5:
            xs, xq = x[:, :k_spt, ], x[:, k_spt:, ]
            xs_shape = xs.shape
            xq_shape = xq.shape
            n_way = xs_shape[0]

            x = torch.cat(
                (xs.contiguous().view(-1, *xs_shape[-3:]), xq.contiguous().view(-1, *xq_shape[-3:]))
                , dim=0
            )

        x = x.cuda() if self.use_cuda else x

        y = torch.arange(n_way).view(n_way, 1, 1).expand(n_way, k_qry, 1)
        y = y.long()
        y = y.cuda() if self.use_cuda else y

        z = self.embedding(x)
        z_proto = z[:n_way*k_spt].view(n_way, k_spt, -1).mean(1)
        z_q = z[n_way*k_spt:]

        dist = pairwise_dist(z_q, z_proto)
        logits = F.log_softmax(-dist, 1).view(n_way, k_qry, n_way)
        loss = -logits.gather(dim=2, index=y).mean()

        y_hat = torch.argmax(logits, dim=2)
        acc = (y.squeeze()==y_hat).float().mean()

        del z, z_proto, z_q, dist, logits, y_hat, xs, xq
        return loss, acc.item()


def get_model(opt=None, ch_in=3, cuda=False):
    embedMap = embedding_map(ch_in)
    model = myProtoNet(embedMap, opt)
    if cuda:
        model.cuda()
    return model

