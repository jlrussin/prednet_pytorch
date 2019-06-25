import os
import json
import numpy as np
import time
from math import ceil
from random import Random

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist

from data import *
from custom_losses import *
from PredNet import *
from ConvLSTM import *

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    def __init__(self, dataset, world_size):
        self.dataset = dataset
        self.partitions = []

        # Partition data into world_size partitions
        rng = Random()
        rng.seed(1234) # ensures data is shuffled the same way in each process
        data_len = len(dataset)
        ids = [i for i in range(data_len)]
        rng.shuffle(ids)

        for i in range(world_size):
            part_len = int(data_len/world_size)
            self.partitions.append(ids[0:part_len])
            ids = ids[part_len:]

    def get_partition(self, rank):
        partition = Partition(self.dataset, self.partitions[rank])
        return partition

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def train(rank, world_size, args):
    # Info
    pid = os.getpid()
    print("Started process on PID: ", pid)

    # Model
    device = 'cpu' # cpu only
    torch.manual_seed(args.seed) # all processes start with the same model
    if args.model_type == 'PredNet':
        model = PredNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                        args.A_kernel_sizes,args.Ahat_kernel_sizes,
                        args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                        args.satlu_act,args.error_act,args.LSTM_act,
                        args.LSTM_c_act,args.bias,args.use_1x1_out,args.FC,
                        device)
    elif args.model_type == 'ConvLSTM':
        model = ConvLSTM(args.in_channels,args.hidden_channels,args.kernel_size,
                         args.LSTM_act,args.LSTM_c_act,args.out_act,
                         args.bias,args.FC,device)

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.train()

    # Data
    if args.dataset == 'KITTI':
        dataset = KITTI(args.train_data_path,args.train_sources_path,
                           args.seq_len)
    elif args.dataset == 'CCN':
        dataset = CCN(args.train_data_path,args.seq_len)
    partitioner = DataPartitioner(dataset, world_size)
    partition = partitioner.get_partition(rank)
    train_loader = DataLoader(partition, args.batch_size,
                              shuffle=True,num_workers=1,pin_memory=False)

    # Loss function
    loss_fn = get_loss_fn(args.loss,args.layer_lambdas)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)
    lrs_step_size = args.num_iters // (args.lr_steps+1)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=lrs_step_size,
                                          gamma=0.1)

    # Stats
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    results_fn = 'r%d_' % rank + args.out_data_file
    results_path = os.path.join(args.results_dir,results_fn)
    loss_data = [] # records loss every args.record_loss_every iters

    # Training loop:
    iter = 0
    epoch_count = 0
    while iter < args.num_iters:
        epoch_count += 1
        for X in train_loader:
            iter += 1
            optimizer.zero_grad()
            # Forward
            start_t = time.time()
            if args.model_type == 'PredNet':
                preds,errors = model(X)
            else:
                preds = model(X)
            # Compute loss
            if args.loss == 'E':
                loss = loss_fn(errors)
            else:
                X_no_t0 = X[:,1:,:,:,:]
                loss = loss_fn(preds,X_no_t0)
            # Backward pass
            loss.backward()
            average_gradients(model) # average gradients across all models
            optimizer.step()
            scheduler.step()
            # Record loss
            if iter % args.record_loss_every == 0:
                loss_datapoint = loss.data.item()
                print('PID:', pid,
                      'Epoch:', epoch_count,
                      'Iter:', iter,
                      'Loss:', loss_datapoint,
                      'lr:', scheduler.get_lr(),
                      'time: ', time.time() - start_t)
                loss_data.append(loss_datapoint)
            if iter >= args.num_iters:
                break
        # Write stats file
        stats = {'loss_data':loss_data}
        with open(results_path, 'w') as f:
            json.dump(stats, f)
        if rank == 0 and args.checkpoint_path is not None:
            torch.save(model.state_dict(),
                       args.checkpoint_path)

def test(rank,args): # Don't need grad_reduce_fn
    # Info
    pid = os.getpid()
    print("Started process on PID: ", pid)

    # Model
    device = 'cpu' # cpu only
    if args.model_type == 'PredNet':
        model = PredNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                        args.A_kernel_sizes,args.Ahat_kernel_sizes,
                        args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                        args.satlu_act,args.error_act,args.LSTM_act,
                        args.LSTM_c_act,args.bias,args.use_1x1_out,args.FC,
                        device)
    elif args.model_type == 'ConvLSTM':
        model = ConvLSTM(args.in_channels,args.hidden_channels,args.kernel_size,
                         args.LSTM_act,args.LSTM_c_act,args.out_act,
                         args.bias,args.FC,device)
    # Load from checkpoint
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
    else:
        print("Testing not possible because model_weights were not saved")
    model.eval()

    # Data
    if args.dataset == 'KITTI':
        dataset = KITTI(args.test_data_path,args.test_sources_path,args.seq_len)
    elif args.dataset == 'CCN':
        dataset = CCN(args.test_data_path,args.seq_len)
    partitioner = DataPartitioner(dataset, world_size)
    partition = partitioner.get_partition(rank)
    test_loader = DataLoader(partition, args.batch_size,
                             shuffle=True,num_workers=1,pin_memory=False)

    # Loss function: always use mse for testing
    mse_loss = nn.MSELoss()

    # Test model on partition
    with torch.no_grad():
        losses = []
        for X in test_loader:
            # Forward
            X = X.to(device)
            if args.model_type == 'PredNet':
                preds,errors = model(X)
            else:
                preds = model(X)
            # Compute loss
            X_no_t0 = X[:,1:,:,:,:]
            loss = mse_loss(preds,X_no_t0)
            # Record loss
            loss_datapoint = loss.data.item()
            losses.append(loss_datapoint)
    print("Average MSE: ", np.mean(losses))
