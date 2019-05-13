#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import os
import torch
import numpy as np
# from nWay_kShot import loadDL
from loadData_sampler1 import load_ds_dl
from ProtoNet import embedding_map, myProtoNet, get_model
from torch import optim
import argparse


class Averager(object):
    def __init__(self):
        self.v = 0
        self.n = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n+1)
        self.n += 1

    def item(self):
        return self.v


def run(args):

    max_epoch = args.max_epoch
    n_episodes = args.n_episodes
    n_episodes_test = args.n_episodes_test
    n_way = args.n_way
    n_way_test = args.n_way_test
    k_spt = args.k_spt
    k_qry = args.k_qry
    patience = args.patience
    use_cuda = args.use_cuda
    save_path = args.save_path
    lr = args.learning_rate
    step_size = args.step_size

    if use_cuda:
        torch.cuda.manual_seed(0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = get_model(opt=args, ch_in=3, cuda=use_cuda)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    trlog = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'min_loss': np.inf,
        'max_acc': 0.0
    }

    train_dl = load_ds_dl('miniImagenet','train',n_way, n_episodes, k_spt, k_qry, )
    val_dl = load_ds_dl('miniImagenet','val', n_way_test, n_episodes_test, k_spt, k_qry)

    state = True
    epoch = 0
    wait = 0

    while epoch < max_epoch and state:
        model.train()
        lr_sche.step()
        tl_a, tc_a = Averager(), Averager()
        for i, bx in enumerate(train_dl, 1):

            optimizer.zero_grad()
            loss, acc = model.loss(bx)
            tl_a.add(loss.item());  tc_a.add(acc)
            loss.backward()
            optimizer.step()
            print('epoch {}, on train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, n_episodes, loss.item(), acc))
            del loss
        trlog['train_loss'].append(tl_a.item())
        trlog['train_acc'].append(tc_a.item())
        print('----a batch training episodes end----\n')


        model.eval()
        vl_a, vc_a = Averager(), Averager()
        for i, bx in enumerate(val_dl, 1):
            loss, acc = model.loss(bx, train=False)
            vl_a.add(loss.item());  vc_a.add(acc)
            del loss
        vl_a, vc_a = vl_a.item(), vc_a.item()
        print('epoch {}, on val, loss={:.4f} acc={:.4f}\n'.format(epoch, vl_a, vc_a))
        trlog['val_loss'].append(vl_a)
        trlog['val_acc'].append(vc_a)


        if vl_a < trlog['min_loss']:
            trlog['min_loss'] = vl_a
            print("==> best loss model (loss = {:0.6f}), saving model...\n".format(trlog['min_loss']))
            if use_cuda:
                model.cpu()
            torch.save(model.state_dict(), os.path.join(save_path, 'min-loss' + '.pth'))
            if use_cuda:
                model.cuda()
            wait = 0
        elif vc_a > trlog['max_acc']:
            trlog['max_acc'] = vc_a
            print("==> best accurate model (acc = {:0.2f}), saving model...\n".format(trlog['max_acc']))
            if use_cuda:
                model.cpu()
            torch.save(model.state_dict(), os.path.join(save_path, name_+'max-acc' + '.pth'))
            if use_cuda:
                model.cuda()
            wait = 0
        else:
            wait += 1

        torch.save(trlog, os.path.join(save_path, 'trlog'))

        if wait > patience:
            print("==> patience {:d} exceeded\n".format(patience))
            state = False
            if use_cuda:
                model.cpu()
            torch.save(model.state_dict(), os.path.join(save_path, 'epoch-last' + '.pth'))

        epoch += 1


if __name__ == '__main__':

    torch.manual_seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=1000)
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--n-episodes-test', type=int, default=200)
    parser.add_argument('--n-way', type=int, default=30)
    parser.add_argument('--n-way-test', type=int, default=5)
    parser.add_argument('--k-spt', type=int, default=5)
    parser.add_argument('--k-qry', type=int, default=15)
    parser.add_argument('--patience', type=int, default=120)
    parser.add_argument('--use-cuda', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--save-path', type=str, default='./saved_train_mini/')


    args = parser.parse_args()
    print(vars(args), '\n\n')

    run(args)




