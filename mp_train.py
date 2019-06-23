import os
import json
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import *
from custom_losses import *

def train(rank, args, model, device, dataloader_kwargs):
    # Manage cpus
    pid = os.getpid()
    print("Started process on PID: ", pid)

    model.train()

    # Data loader
    torch.manual_seed(args.seed + rank)
    train_loader = get_dataloader(args,'train',dataloader_kwargs)

    # Loss function
    loss_fn = get_loss_fn(args.loss,args.layer_lambdas)
    loss_fn = loss_fn.to(device)

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
            X = X.to(device)
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
        with open(results_file_name, 'w') as f:
            json.dump(stats, f)

def test(args,model,device,dataloader_kwargs):
    val_loader = get_dataloader(args,'val',dataloader_kwargs)
    mse_loss = nn.MSELoss() # always use mse for testing
    model.eval()
    with torch.no_grad():
        losses = []
        for X in val_loader:
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
