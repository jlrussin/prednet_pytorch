import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import *
from activations import *
from PredNet import *
from ConvLSTM import *
from custom_losses import *
from utils import *

parser = argparse.ArgumentParser()
# Training data
parser.add_argument('--train_data_hkl',
                    default='../data/kitti_data/X_train.hkl',
                    help='Path to training images hkl file')
parser.add_argument('--train_sources_hkl',
                    default='../data/kitti_data/sources_train.hkl',
                    help='Path to training sources hkl file')
parser.add_argument('--val_data_hkl',
                    default='../data/kitti_data/X_val.hkl',
                    help='Path to validation images hkl file')
parser.add_argument('--val_sources_hkl',
                    default='../data/kitti_data/sources_val.hkl',
                    help='Path to validation sources hkl file')
parser.add_argument('--test_data_hkl',
                    default='../data/kitti_data/X_test.hkl',
                    help='Path to test images hkl file')
parser.add_argument('--test_sources_hkl',
                    default='../data/kitti_data/sources_test.hkl',
                    help='Path to test sources hkl file')
parser.add_argument('--seq_len',type=int,default=10,
                    help='Number of images in each kitti sequence')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Samples per batch')
parser.add_argument('--num_iters', type=int, default=75000,
                    help='Number of optimizer steps before stopping')

# Models
parser.add_argument('--model_type', choices=['PredNet','ConvLSTM'],
                    default='PredNet', help='Type of model to use.')
# Hyperparameters for PredNet
parser.add_argument('--stack_sizes', type=int, nargs='+', default=[3,48,96,192],
                    help='number of channels in targets (A) and ' +
                         'predictions (Ahat) in each layer. ' +
                         'Length should be equal to number of layers')
parser.add_argument('--R_stack_sizes', type=int, nargs='+',
                    default=[3,48,96,192],
                    help='Number of channels in R modules. ' +
                         'Length should be equal to number of layers')
parser.add_argument('--A_kernel_sizes', type=int, nargs='+', default=[3,3,3],
                    help='Kernel sizes for each A module. ' +
                         'Length should be equal to (number of layers - 1)')
parser.add_argument('--Ahat_kernel_sizes', type=int, nargs='+',
                    default=[3,3,3,3], help='Kernel sizes for each Ahat' +
                    'module. Length should be equal to number of layers')
parser.add_argument('--R_kernel_sizes', type=int, nargs='+', default=[3,3,3,3],
                    help='Kernel sizes for each Ahat module' +
                         'Length should be equal to number of layers')
parser.add_argument('--use_satlu', type=str2bool, default=True,
                    help='Boolean indicating whether to use SatLU in Ahat.')
parser.add_argument('--satlu_act', default='hardtanh',
                    choices=['hardtanh','logsigmoid'],
                    help='Type of activation to use for SatLU in Ahat.')
parser.add_argument('--pixel_max', type=float, default=255.0,
                    help='Maximum output value for Ahat if using SatLU.')
parser.add_argument('--error_act', default='relu',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation to use in E modules.')
parser.add_argument('--use_1x1_out', type=str2bool, default=False,
                    help='Boolean indicating whether to use 1x1 conv layer' +
                         'for output of ConvLSTM cells')
# Hyperparameters for ConvLSTM
parser.add_argument('--hidden_channels', type=int, default=192,
                    help='Number of channels in hidden states of ConvLSTM')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='Kernel size in ConvLSTM')
parser.add_argument('--out_act', default='relu',
                    help='Activation for output layer of ConvLSTM cell')
# Hyperparameters shared by PredNet and ConvLSTM
parser.add_argument('--in_channels', type=int, default=3,
                    help='Number of channels in input images')
parser.add_argument('--LSTM_act', default='tanh',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation to use in ConvLSTM.')
parser.add_argument('--LSTM_c_act', default='hardsigmoid',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation for inner ConvLSTM (C_t).')
parser.add_argument('--bias', type=str2bool, default=True,
                    help='Boolean indicating whether to use bias units')
parser.add_argument('--FC', type=str2bool, default=False,
                    help='Boolean indicating whether to use fully connected' +
                         'convolutional LSTM cell')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')

# Optimization
parser.add_argument('--loss', default='E',choices=['E','MSE','L1'])
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')
parser.add_argument('--lr_steps', type=int, default=1,
                    help='num times to decrease learning rate by factor of 0.1')
parser.add_argument('--layer_lambdas', type=float,
                    nargs='+', default=[1.0,0.0,0.0,0.0],
                    help='Weight of loss on error of each layer' +
                         'Length should be equal to number of layers')

# Output options
parser.add_argument('--results_dir', default='../results',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='results.json',
                    help='Name of output data file with training loss data')
parser.add_argument('--checkpoint_path',default=None,
                    help='Path to output saved weights.')
parser.add_argument('--checkpoint_every', type=int, default=5,
                    help='Epochs before evaluating model and saving weights')
parser.add_argument('--record_loss_every', type=int, default=20,
                    help='iters before printing and recording loss')

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Data
    train_data = KITTI(args.train_data_hkl,args.train_sources_hkl,args.seq_len)
    val_data = KITTI(args.val_data_hkl,args.val_sources_hkl,args.seq_len)
    test_data = KITTI(args.test_data_hkl,args.test_sources_hkl,args.seq_len)
    train_loader = DataLoader(train_data,args.batch_size,shuffle=True)
    val_loader = DataLoader(val_data,args.batch_size,shuffle=True)
    test_loader = DataLoader(test_data,args.batch_size,shuffle=True)

    # Model
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
    model.to(device)
    model.train()

    # Select loss function
    loss_fn = get_loss_fn(args.loss,args.layer_lambdas)
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)
    lrs_step_size = args.num_iters // (args.lr_steps+1)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=lrs_step_size,
                                          gamma=0.1)

    # Training loop:
    iter = 0
    epoch_count = 0
    loss_data = [] # records loss every args.record_loss_every iters
    train_losses = [] # records mean training loss every checkpoint
    val_losses = [] # records mean validation loss every checkpoint
    test_losses = [] # records mean test loss every checkpoint
    best_val_loss = float("inf") # will only save best weights
    while iter < args.num_iters:
        epoch_count += 1
        for X in train_loader:
            iter += 1
            optimizer.zero_grad()
            # Forward
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
                print('Epoch:', epoch_count,
                      'Iter:', iter,
                      'Loss:', loss_datapoint,
                      'lr:', scheduler.get_lr())
                loss_data.append(loss_datapoint)
        # Checkpoint
        last_epoch = (iter >= args.num_iters)
        if epoch_count % args.checkpoint_every == 0 or last_epoch:
            # Train
            print("Checking training loss...")
            train_loss = checkpoint(train_loader, model, device, args)
            print("Training loss is ", train_loss)
            train_losses.append(train_loss)
            # Validation
            print("Checking validation loss...")
            val_loss = checkpoint(val_loader, model, device, args)
            print("Validation loss is ", val_loss)
            val_losses.append(val_loss)
            # Test
            print("Checking test loss...")
            test_loss = checkpoint(test_loader, model, device, args)
            print("Test loss is ", test_loss)
            test_losses.append(test_loss)
            # Write stats file
            if not os.path.isdir(args.results_dir):
                os.mkdir(args.results_dir)
            stats = {'loss_data':loss_data,
                     'train_mse_losses':train_losses,
                     'val_mse_losses':val_losses,
                     'test_mse_losses':test_losses}
            results_file_name = '%s/%s' % (args.results_dir,args.out_data_file)
            with open(results_file_name, 'w') as f:
                json.dump(stats, f)
            # Save model weights
            if val_loss < best_val_loss: # use val (not test) to decide to save
                best_val_loss = val_loss
                if args.checkpoint_path is not None:
                    torch.save(model.state_dict(),
                               args.checkpoint_path)


def checkpoint(dataloader, model, device, args):
    # Always use MSE loss for checkpointing:
    mse_loss = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        losses = []
        for X in dataloader:
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

    model.train()
    return np.mean(losses)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
