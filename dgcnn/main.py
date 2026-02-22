#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

from __future__ import print_function
import os
import sys
import ast
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from model import HierarchicalCanonicalNet, HierarchicalSpectralNet
import wandb


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'hierarchical_canonical':
        model = HierarchicalCanonicalNet(
            sampling=args.sampling,
            k=args.k,
            dropout=args.dropout,
            patch_mlps=args.patch_mlps
        ).to(device)
    elif args.model == 'hierarchical_spectral':
        model = HierarchicalSpectralNet(
            sampling=args.sampling,
            k=args.k,
            dropout=args.dropout,
            patch_mlps=args.patch_mlps,
            sigma_kernel=args.sigma_kernel
        ).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # --- THE CLEAN OPTIMIZER AND SCHEDULER LOGIC ---
    if args.use_sgd:
        print("Use SGD")
        start_lr = args.lr * 100
        opt = optim.SGD(model.parameters(), lr=start_lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        start_lr = args.lr
        opt = optim.Adam(model.parameters(), lr=start_lr, weight_decay=1e-4)

    # Automatically set the minimum LR to 1% of the starting LR
    min_lr = start_lr * 0.001
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=min_lr)
    # -----------------------------------------------

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for data, label in train_bar:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)

            if torch.isnan(loss):
                print(f"\nFATAL ERROR: NaN loss detected at Epoch {epoch}!")
                print("Killing the Slurm job to save cluster compute time.")
                wandb.run.summary["status"] = "failed_due_to_nan"
                sys.exit(1)

            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss * 1.0 / count,
            "train/acc": metrics.accuracy_score(train_true, train_pred),
            "train/avg_acc": metrics.balanced_accuracy_score(train_true, train_pred)
        })

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.pt' % args.exp_name)

        wandb.log({
            "epoch": epoch,
            "test/loss": test_loss * 1.0 / count,
            "test/acc": test_acc,
            "test/avg_acc": avg_per_class_acc,
            "test/best_acc": best_test_acc,
            "lr": opt.param_groups[0]['lr']
        })


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'hierarchical_canonical':
        model = HierarchicalCanonicalNet(
            sampling=args.sampling,
            k=args.k,
            dropout=args.dropout,
            patch_mlps=args.patch_mlps
        ).to(device)
    elif args.model == 'hierarchical_spectral':
        model = HierarchicalSpectralNet(
            sampling=args.sampling,
            k=args.k,
            dropout=args.dropout,
            patch_mlps=args.patch_mlps,
            sigma_kernel=args.sigma_kernel
        ).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn', 'canonical_mlp', 'hybrid_dgcnn', 'hierarchical_canonical',
                                 'hierarchical_spectral'],
                        help='Model to use')
    parser.add_argument('--sigma_kernel', type=float, default=1.0,
                        help='Sigma for the Gaussian kernel in the spectral Fiedler computation')
    parser.add_argument('--sampling', type=ast.literal_eval, default=[512, 128, 32],
                        help='Hierarchical downsampling steps')
    parser.add_argument('--patch_mlp_dims', type=int, nargs='+', default=[64, 64],
                        help='Dimensions for the initial patch MLP layers')
    parser.add_argument('--patch_mlps', type=ast.literal_eval,
                        default=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
                        help='Nested list of MLP dimensions for the hierarchical stages')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')

    # Booleans are now proper flags.
    parser.add_argument('--use_sgd', action='store_true',
                        help='Use SGD (Default is Adam)')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--no_cuda', action='store_true',
                        help='enables CUDA training')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    wandb.init(project="modelnet40-canon", name=args.exp_name, config=vars(args))

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)