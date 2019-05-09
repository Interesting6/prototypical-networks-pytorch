#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy
import sys
sys.path.append('..')

from loadMiniData import load_miniImageDS_dl
from protoNet import encoding_map, myProtoNet
from myutils import Averager, save_model, ensure_path
#from plot_lib import show_batch
import torch
from torch import optim
import numpy as np
import os



torch.manual_seed(1234)
miniImage_train_dl = load_miniImageDS_dl('train', 30, 100, 1, 15)
miniImage_val_dl = load_miniImageDS_dl('val', 5, 400, 1, 15)

EmcodingMap = encoding_map(3, 64, 64)
model = myProtoNet(EmcodingMap)


optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

max_epoch = 600
epoch = 0
best_loss = np.inf
wait = 0
train_patience = 150
state = True
savePath = '../save/mini-models'
ensure_path(savePath)

trlog = {
    'train_loss': [], 'train_acc': [],
    'val_loss':[], 'val_acc': [],
    'min_loss': best_loss
}


while epoch < max_epoch and state:

    model.train()
    lr_sche.step()

    tl_a, tc_a = Averager(), Averager()
    for i, batch_sample in enumerate(miniImage_train_dl):
        optimizer.zero_grad()
        loss_, outs = model.loss(batch_sample)
        loss_.backward()
        optimizer.step()
        tl_a.add(outs['loss']); tc_a.add(outs['accuracy'])
        print('epoch {}'.format(epoch) + ' batch {}:'.format(i) +
              ' loss:{:0.6f}'.format(outs['loss']) +
              '; accuracy:{:0.4f}'.format(outs['accuracy']))
    tl_a, tc_a = tl_a.item(), tc_a.item()
    trlog['train_loss'].append(tl_a)
    trlog['train_acc'].append(tc_a)


    model.eval()
    vl_a, vc_a = Averager(), Averager()
    for i, batch_sample in enumerate(miniImage_val_dl, 1):
        loss_, outs = model.loss(batch_sample)

        vl_a.add(outs['loss']); vc_a.add(outs['accuracy'])
    vl_a, vc_a = vl_a.item(), vc_a.item()
    print('\nepoch {}, on val, loss={:.4f} acc={:.4f}\n'.format(epoch, vl_a, vc_a))
    trlog['val_loss'].append(vl_a)
    trlog['val_acc'].append(vc_a)



    if vl_a < trlog['min_loss']:
        trlog['min_loss'] = vl_a
        print("==> best loss model (loss = {:0.6f}), saving model...\n".format(trlog['min_loss']))
        save_model('min-loss', model, savePath)
        wait = 0
    else:
        wait += 1

    torch.save(trlog, os.path.join(savePath, 'trlog'))

    if wait > train_patience:
        print("==> patience {:d} exceeded\n".format(train_patience))
        state = False
        save_model('epoch-last', model, savePath)


    epoch += 1
