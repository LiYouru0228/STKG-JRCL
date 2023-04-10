from __future__ import absolute_import
from __future__ import print_function

from model import JRCL

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from load_data import Dataset

import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import copy
from tqdm import tqdm
from sklearn import metrics


def Bootstrap(y_true,y_pred, test_ret):
    N = len(y_true)
    N_idx = np.arange(N)
    K = 1000

    acc = []
    minpse = []
    auroc = []
    auprc = []
    
    for i in range(K):
        boot_idx = np.random.choice(N_idx, N, replace=True)
        boot_true = np.array(y_true)[boot_idx]
        boot_pred = y_pred[boot_idx, :]
        test_ret = print_metrics_binary(boot_true, boot_pred, verbose=0)
        acc.append(test_ret['acc'])
        auroc.append(test_ret['auroc'])
        auprc.append(test_ret['auprc'])
        minpse.append(test_ret['minpse'])

    print('acc: %.4f(%.4f)'%(np.mean(acc), np.std(acc)))
    print('auroc: %.4f(%.4f)'%(np.mean(auroc), np.std(auroc)))
    print('auprc: %.4f(%.4f)'%(np.mean(auprc), np.std(auprc)))
    print('min(+P, Se): %.4f(%.4f)'%(np.mean(minpse), np.std(minpse)))

    
def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    
    if verbose:
        print("acc= {}".format(acc))
        print("auroc= {}".format(auroc))
        print("auprc= {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        
    return {"acc": acc,
            "auroc": auroc,
            "auprc":auprc,
            "minpse":minpse
            }

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED) #numpy
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED) # cpu
torch.cuda.manual_seed(RANDOM_SEED) #gpu
torch.backends.cudnn.deterministic=True # cudnn

train_loader = pickle.load(open("./dataset.pkl", "rb"))['train_raw']
valid_loader = pickle.load(open("./dataset.pkl", "rb"))['val_raw']
test_loader = pickle.load(open("./dataset.pkl", "rb"))['test_raw']
dkg_path2vec = pickle.load(open("./path2vec.pkl", "rb"))

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')


def train(args):

    max_iters = args.max_epoch
    d = args.d
    sampleSize = args.K
    model = JRCL(input_dim = sampleSize, hidden_dim = d, d_model = d,  MHD_num_head = args.NH, d_ff = 256, output_dim = 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_roc = 0
    max_prc = 0
    train_loss = []
    train_model_loss = []
    valid_loss = []
    valid_model_loss = []
    history = []
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    for each_epoch in range(max_iters):
        batch_loss = []
        model_batch_loss = []

        model.train()

        for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):   
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            d_paths_emb = {}
            for slice_id in dkg:
                paths_emb = []
                path2vec = dkg_path2vec[slice_id][0]
                for name in batch_name:
                    path_emb = path2vec[name + ':' + str(0)].reshape(1,d)
                    for index in range(sampleSize-1): 
                        path_emb = np.concatenate((path_emb, path2vec[name + ':' + str(index + 1)].reshape(1,d)), axis=0) 
                    path_emb = torch.tensor(path_emb, dtype=torch.float32)
                    paths_emb.append(path_emb)
                d_paths_emb[slice_id] = paths_emb
            paths_emb = torch.stack([torch.stack(d_paths_emb[i]) for i in range(12)],dim=3).to(device)

            d_lbs_emb = {}
            for slice_id in dkg:
                lbs_embs = []
                lbs2vec = dkg_path2vec[slice_id][1]
                for name in batch_name:
                    lbs_emb = lbs2vec[name + ':' + str(0)].reshape(1,d)
                    for index in range(sampleSize-1): 
                        lbs_emb = np.concatenate((lbs_emb, lbs2vec[name + ':' + str(index + 1)].reshape(1,d)), axis=0) 
                    lbs_emb = torch.tensor(lbs_emb, dtype=torch.float32)
                    lbs_embs.append(lbs_emb)
                d_lbs_emb[slice_id] = lbs_embs
            lbs_embs = torch.stack([torch.stack(d_lbs_emb[i]) for i in range(12)],dim=3).to(device)
            output = model(batch_x, paths_emb, lbs_embs, each_epoch, step)
            model_loss = get_loss(output, batch_y.unsqueeze(-1))
            loss = model_loss 

            batch_loss.append(loss.cpu().detach().numpy())
            model_batch_loss.append(model_loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f'%(each_epoch, step, np.mean(np.array(batch_loss))))
                print('Model Loss = %.4f'%(np.mean(np.array(model_batch_loss))))
        train_loss.append(np.mean(np.array(batch_loss)))
        train_model_loss.append(np.mean(np.array(model_batch_loss)))

        batch_loss = []
        model_batch_loss = []

        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                d_paths_emb = {}
                for slice_id in dkg:
                    paths_emb = []
                    path2vec = dkg_path2vec[slice_id][0]
                    for name in batch_name:
                        path_emb = path2vec[name + ':' + str(0)].reshape(1,d)
                        for index in range(sampleSize-1): 
                            path_emb = np.concatenate((path_emb, path2vec[name + ':' + str(index + 1)].reshape(1,d)), axis=0) 
                        path_emb = torch.tensor(path_emb, dtype=torch.float32)
                        paths_emb.append(path_emb)
                    d_paths_emb[slice_id] = paths_emb
                paths_emb = torch.stack([torch.stack(d_paths_emb[i]) for i in range(12)],dim=3).to(device)

                d_lbs_emb = {}
                for slice_id in dkg:
                    lbs_embs = []
                    lbs2vec = dkg_path2vec[slice_id][1]
                    for name in batch_name:
                        lbs_emb = lbs2vec[name + ':' + str(0)].reshape(1,d)
                        for index in range(sampleSize-1): 
                            lbs_emb = np.concatenate((lbs_emb, lbs2vec[name + ':' + str(index + 1)].reshape(1,d)), axis=0) 
                        lbs_emb = torch.tensor(lbs_emb, dtype=torch.float32)
                        lbs_embs.append(lbs_emb)
                    d_lbs_emb[slice_id] = lbs_embs
                lbs_embs = torch.stack([torch.stack(d_lbs_emb[i]) for i in range(12)],dim=3).to(device)
                output = model(batch_x, paths_emb, lbs_embs, each_epoch, step)
                model_loss = get_loss(output, batch_y.unsqueeze(-1))

                loss = model_loss
                batch_loss.append(loss.cpu().detach().numpy())
                model_batch_loss.append(model_loss.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        valid_loss.append(np.mean(np.array(batch_loss)))
        valid_model_loss.append(np.mean(np.array(model_batch_loss)))

        print("\n==>Predicting on validation")
        print('Valid Loss = %.4f'%(valid_loss[-1]))
        print('valid_model Loss = %.4f'%(valid_model_loss[-1]))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        ret = print_metrics_binary(y_true, y_pred)
        history.append(ret)
        print()

        #-------------------- test -----------------------
        batch_loss = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                d_paths_emb = {}
                for slice_id in dkg:
                    paths_emb = []
                    path2vec = dkg_path2vec[slice_id][0]
                    for name in batch_name:
                        path_emb = path2vec[name + ':' + str(0)].reshape(1,d)
                        for index in range(sampleSize-1): 
                            path_emb = np.concatenate((path_emb, path2vec[name + ':' + str(index + 1)].reshape(1,d)), axis=0) 
                        path_emb = torch.tensor(path_emb, dtype=torch.float32)
                        paths_emb.append(path_emb)
                    d_paths_emb[slice_id] = paths_emb
                paths_emb = torch.stack([torch.stack(d_paths_emb[i]) for i in range(12)],dim=3).to(device)

                d_lbs_emb = {}
                for slice_id in dkg:
                    lbs_embs = []
                    lbs2vec = dkg_path2vec[slice_id][1]
                    for name in batch_name:
                        lbs_emb = lbs2vec[name + ':' + str(0)].reshape(1,d)
                        for index in range(sampleSize-1): 
                            lbs_emb = np.concatenate((lbs_emb, lbs2vec[name + ':' + str(index + 1)].reshape(1,d)), axis=0) 
                        lbs_emb = torch.tensor(lbs_emb, dtype=torch.float32)
                        lbs_embs.append(lbs_emb)
                    d_lbs_emb[slice_id] = lbs_embs
                lbs_embs = torch.stack([torch.stack(d_lbs_emb[i]) for i in range(12)],dim=3).to(device)            
                output = model(batch_x, paths_emb, lbs_embs, each_epoch, step)

                loss = get_loss(output, batch_y.unsqueeze(-1))
                batch_loss.append(loss.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())

        print("\n==>Predicting on test")
        print('Test Loss = %.4f'%(np.mean(np.array(batch_loss))))
        y_pred = np.array(y_pred)
        print(y_pred.shape)
        print(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        test_res = print_metrics_binary(y_true, y_pred)

        cur_auroc = test_res['auroc']
        print ('testing experimental report:')
        Bootstrap(y_true,y_pred,test_res)
    
    print('=====DONE=====')

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=20, help='paths sample size')
parser.add_argument('--d', type=int, default=64, help='embeddings size')
parser.add_argument('--NH', type=int, default=2, help='head size of MHA')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=50, help='the number of epochs')

def main():
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
