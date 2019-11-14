import os
import time
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import *
from PredNet import *
from ConvLSTM import *
from Ladder import *
from StackedConvLSTM import *
from custom_losses import *
from utils import *

parser = argparse.ArgumentParser()
# Training data
parser.add_argument('--dataset',choices=['KITTI','CCN'],default='KITTI',
                    help='Dataset to use')
parser.add_argument('--train_data_path',
                    default='../data/kitti_data/X_train.hkl',
                    help='Path to training images hkl file')
parser.add_argument('--train_sources_path',
                    default='../data/kitti_data/sources_train.hkl',
                    help='Path to training sources hkl file')
parser.add_argument('--val_data_path',
                    default='../data/kitti_data/X_val.hkl',
                    help='Path to validation images hkl file')
parser.add_argument('--val_sources_path',
                    default='../data/kitti_data/sources_val.hkl',
                    help='Path to validation sources hkl file')
parser.add_argument('--test_data_path',
                    default='../data/kitti_data/X_test.hkl',
                    help='Path to test images hkl file')
parser.add_argument('--test_sources_path',
                    default='../data/kitti_data/sources_test.hkl',
                    help='Path to test sources hkl file')
parser.add_argument('--seq_len',type=int,default=10,
                    help='Number of images in each kitti sequence')
parser.add_argument('--downsample_size',type=int,default=128,
                    help='Height and width of downsampled CCN inputs.')
parser.add_argument('--last_only',type=str2bool,default=False,
                    help='Train on sequences of static (final) images.')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Samples per batch')
parser.add_argument('--num_iters', type=int, default=75000,
                    help='Number of optimizer steps before stopping')

# Models
parser.add_argument('--model_type', choices=['PredNet','ConvLSTM',
                                             'MultiConvLSTM','LadderNet',
                                             'StackedConvLSTM'],
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
parser.add_argument('--A_act', default='relu',
                    choices=['relu','lrelu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation for output of Ahat cell.')
parser.add_argument('--Ahat_act', default='relu',
                    choices=['relu','lrelu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation for output of Ahat cell.')
parser.add_argument('--use_satlu', type=str2bool, default=True,
                    help='Boolean indicating whether to use SatLU in Ahat.')
parser.add_argument('--satlu_act', default='hardtanh',
                    choices=['hardtanh','logsigmoid','sigmoid'],
                    help='Type of activation to use for SatLU in Ahat.')
parser.add_argument('--pixel_max', type=float, default=1.0,
                    help='Maximum output value for Ahat if using SatLU.')
parser.add_argument('--error_act', default='relu',
                    choices=['relu','sigmoid','tanh','hardsigmoid'],
                    help='Type of activation to use in E modules.')
parser.add_argument('--use_1x1_out', type=str2bool, default=False,
                    help='Boolean indicating whether to use 1x1 conv layer' +
                         'for output of ConvLSTM cells')
parser.add_argument('--send_acts', type=str2bool, default=False,
                    help='Boolean indicating whether to send activities' +
                         'rather than errors')
parser.add_argument('--no_ER', type=str2bool, default=False,
                    help='Boolean indicating whether to ablate connection' +
                         'between E_l and R_l on all but last layer')
parser.add_argument('--RAhat', type=str2bool, default=False,
                    help='Boolean indicating whether to add connection' +
                         'between R_lp1 and Ahat_l')
parser.add_argument('--no_A_conv', type=str2bool, default=False,
                    help='No convolutional layer in A cells')
parser.add_argument('--higher_satlu', type=str2bool, default=False,
                    help='Use satlu in higher layers')
parser.add_argument('--local_grad', type=str2bool, default=False,
                    help='Boolean indicating whether to restrict gradients ' +
                         'to flow locally (within each layer)')
parser.add_argument('--forward_conv',type=str2bool,default=False,
                    help='Boolean indicating whether to use conv2d rather' +
                         'than convLSTM in forward path of StackedConvLSTM')
parser.add_argument('--conv_dilation', type=int, default=1,
                    help='Dilation for convolution in ACells')
parser.add_argument('--use_BN', type=str2bool, default=False,
                    help='Boolean indicating whether to use batch ' +
                         'normalization on inputs in all A and Ahat cells')
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
parser.add_argument('--dropout_p', type=float, default=0.0,
                    help='Proportion dropout for inputs to R cells')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')
# Hyperparameters unique to LadderNet
parser.add_argument('--no_R0', type=str2bool, default=True,
                    help='Boolean indicating whether not to include' +
                         'ConvLSTM in first layer of LadderNet')
parser.add_argument('--no_skip0', type=str2bool, default=True,
                    help='Boolean indicating whether not to include' +
                         'skip connection in first layer of LadderNet')

# Optimization
parser.add_argument('--loss', default='E',choices=['E','MSE','L1','BCE'])
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')
parser.add_argument('--lr_steps', type=int, default=1,
                    help='num times to decrease learning rate by factor of 0.1')
parser.add_argument('--layer_lambdas', type=float,
                    nargs='+', default=[1.0,0.0,0.0,0.0],
                    help='Weight of loss on error of each layer' +
                         'Length should be equal to number of layers')
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay')

# Output options
parser.add_argument('--record_E', default=False,
                    help='Record E for each layer')
parser.add_argument('--record_corr', default=False,
                    help='Record correlation throughout training')
parser.add_argument('--results_dir', default='../results/train_results',
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
    if args.dataset == 'KITTI':
        train_data = KITTI(args.train_data_path,args.train_sources_path,
                           args.seq_len)
        val_data = KITTI(args.val_data_path,args.val_sources_path,
                         args.seq_len)
        test_data = KITTI(args.test_data_path,args.test_sources_path,
                          args.seq_len)
    elif args.dataset == 'CCN':
        downsample_size = (args.downsample_size,args.downsample_size)
        train_data = CCN(args.train_data_path,args.seq_len,
                         downsample_size=downsample_size,
                         last_only=args.last_only)
        val_data = CCN(args.val_data_path,args.seq_len,
                       downsample_size=downsample_size,
                       last_only=args.last_only)
        test_data = CCN(args.test_data_path,args.seq_len,
                        downsample_size=downsample_size,
                        last_only=args.last_only)
    train_loader = DataLoader(train_data,args.batch_size,shuffle=True)
    val_loader = DataLoader(val_data,args.batch_size,shuffle=True)
    test_loader = DataLoader(test_data,args.batch_size,shuffle=True)

    # Model
    model_out = 'error' if args.loss == 'E' else 'pred'
    if args.model_type == 'PredNet':
        model = PredNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                        args.A_kernel_sizes,args.Ahat_kernel_sizes,
                        args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                        args.Ahat_act,args.satlu_act,args.error_act,
                        args.LSTM_act,args.LSTM_c_act,args.bias,
                        args.use_1x1_out,args.FC,args.dropout_p,
                        args.send_acts,args.no_ER,args.RAhat,args.no_A_conv,
                        args.higher_satlu,args.local_grad,args.conv_dilation,
                        args.use_BN,model_out,device)
    elif args.model_type == 'MultiConvLSTM':
        model = MultiConvLSTM(args.in_channels,args.R_stack_sizes,
                              args.R_kernel_sizes,args.use_satlu,args.pixel_max,
                              args.Ahat_act,args.satlu_act,args.error_act,
                              args.LSTM_act,args.LSTM_c_act,args.bias,
                              args.use_1x1_out,args.FC,args.local_grad,
                              model_out,device)
    elif args.model_type == 'ConvLSTM':
        model = ConvLSTM(args.in_channels,args.hidden_channels,args.kernel_size,
                         args.LSTM_act,args.LSTM_c_act,args.out_act,
                         args.bias,args.FC,device)
    elif args.model_type == 'LadderNet':
        model = LadderNet(args.in_channels,args.stack_sizes,args.R_stack_sizes,
                          args.A_kernel_sizes,args.Ahat_kernel_sizes,
                          args.R_kernel_sizes,args.conv_dilation,args.use_BN,
                          args.use_satlu,args.pixel_max,args.A_act,
                          args.Ahat_act,args.satlu_act,args.error_act,
                          args.LSTM_act,args.LSTM_c_act,args.bias,
                          args.use_1x1_out,args.FC,args.no_R0,args.no_skip0,
                          args.no_A_conv,args.higher_satlu,args.local_grad,
                          model_out,device)
    elif args.model_type == 'StackedConvLSTM':
        model = StackedConvLSTM(args.in_channels,args.R_stack_sizes,
                                args.R_kernel_sizes,args.use_1x1_out,
                                args.FC,args.local_grad,args.forward_conv,
                                model_out,device)
    print(model)
    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)
    model.train()

    # Select loss function
    loss_fn = get_loss_fn(args.loss,args.layer_lambdas)
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate,weight_decay=args.wd)
    lrs_step_size = args.num_iters // (args.lr_steps+1)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=lrs_step_size,
                                          gamma=0.1)

    # Stats
    ave_time = 0.0
    loss_data = [] # records loss every args.record_loss_every iters
    corr_data = [] # records correlation every args.record_loss_every iters
    train_losses = [] # records mean training loss every checkpoint
    train_corrs = [] # records mean training  correlation every checkpoint
    val_losses = [] # records mean validation loss every checkpoint
    val_corrs = [] # records mean validation correlation every checkpoint
    test_losses = [] # records mean test loss every checkpoint
    test_corrs = [] # records mean test correlation every checkpoint
    best_val_loss = float("inf") # will only save best weights
    if args.record_E:
        E_data = {'layer%d' % i:[] for i in range(model.nb_layers)}
        train_Es = {'layer%d' % i:[] for i in range(model.nb_layers)}
        val_Es = {'layer%d' % i:[] for i in range(model.nb_layers)}
        test_Es = {'layer%d' % i:[] for i in range(model.nb_layers)}
    # Training loop
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
            output = model(X)
            # Compute loss
            if args.loss == 'E':
                loss = loss_fn(output)
            else:
                X_no_t0 = X[:,1:,:,:,:]
                loss = loss_fn(output,X_no_t0)
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Record loss
            iter_time = time.time() - start_t
            ave_time = (ave_time*(iter-1) + iter_time)/iter
            if iter % args.record_loss_every == 0:
                loss_datapoint = loss.data.item()
                print('Epoch:', epoch_count,
                      'Iter:', iter,
                      'Loss:', loss_datapoint,
                      'lr:', scheduler.get_lr(),
                      'ave time: ', ave_time)
                loss_data.append(loss_datapoint)
                if args.record_E:
                    E_means = torch.mean(output.detach(),dim=0)
                    for l in range(model.nb_layers):
                        E_datapoint = E_means[l].data.item()
                        E_data['layer%d' % l].append(E_datapoint)
                if args.record_corr:
                    model_output = model.output
                    model.output = 'pred'
                    output = model(X)
                    X_no_t0 = X[:,1:,:,:,:]
                    corr = correlation(output,X_no_t0)
                    corr_data.append(corr.data.item())
                    model.output = model_output
            if iter >= args.num_iters:
                break
        # Checkpoint
        last_epoch = (iter >= args.num_iters)
        if epoch_count % args.checkpoint_every == 0 or last_epoch:
            # Train
            print("Checking training loss...")
            train_checkpoint = checkpoint(train_loader,model,device,args)
            if args.record_E:
                train_loss,train_corr,train_E = train_checkpoint
                for l in range(model.nb_layers):
                    train_Es['layer%d' % l].append(train_E[l])
            else:
                train_loss,train_corr = train_checkpoint
            print("Training loss is ", train_loss)
            print("Training average correlation is ", train_corr)
            train_losses.append(train_loss)
            train_corrs.append(train_corr)
            # Validation
            print("Checking validation loss...")
            val_checkpoint = checkpoint(val_loader,model,device,args)
            if args.record_E:
                val_loss,val_corr,val_E = val_checkpoint
                for l in range(model.nb_layers):
                    val_Es['layer%d' % l].append(val_E[l])
            else:
                val_loss,val_corr = val_checkpoint
            print("Validation loss is ", val_loss)
            print("Validation average correlation is ",val_corr)
            val_losses.append(val_loss)
            val_corrs.append(val_corr)
            # Test
            print("Checking test loss...")
            test_checkpoint = checkpoint(test_loader,model,device,args)
            if args.record_E:
                test_loss,test_corr,test_E = test_checkpoint
                for l in range(model.nb_layers):
                    test_Es['layer%d' % l].append(test_E[l])
            else:
                test_loss,test_corr = test_checkpoint
            print("Test loss is ", test_loss)
            print("Test average correlation is ", test_corr)
            test_losses.append(test_loss)
            test_corrs.append(test_corr)
            # Write stats file
            if not os.path.isdir(args.results_dir):
                os.mkdir(args.results_dir)
            stats = {'loss_data':loss_data,
                     'corr_data':corr_data,
                     'train_mse_losses':train_losses,
                     'train_corrs':train_corrs,
                     'val_mse_losses':val_losses,
                     'val_corrs':val_corrs,
                     'test_mse_losses':test_losses,
                     'test_corrs':test_corrs}
            if args.record_E:
                stats['E_data'] = E_data
                stats['train_Es'] = train_Es
                stats['val_Es'] = val_Es
                stats['test_Es'] = test_Es
            results_file_name = '%s/%s' % (args.results_dir,args.out_data_file)
            with open(results_file_name, 'w') as f:
                json.dump(stats, f)
            # Save model weights
            if val_loss < best_val_loss: # use val (not test) to decide to save
                best_val_loss = val_loss
                if args.checkpoint_path is not None:
                    torch.save(model.state_dict(),
                               args.checkpoint_path)


def correlation(X,Y):
    batch_size = X.shape[0]
    X = X.view(batch_size,-1)
    Y = Y.view(batch_size,-1)
    X_bar = torch.mean(X,dim=1,keepdim=True)
    Y_bar = torch.mean(Y,dim=1,keepdim=True)
    X_c = X - X_bar
    Y_c = Y - Y_bar
    X_norm = torch.norm(X_c,dim=1,keepdim=True)
    Y_norm = torch.norm(Y_c,dim=1,keepdim=True)
    X_n = X_c/X_norm
    Y_n = Y_c/Y_norm
    corr = torch.sum(X_n*Y_n,dim=1)
    ave_corr = torch.mean(corr,dim=0)
    return ave_corr

def checkpoint(dataloader, model, device, args):
    # Always use MSE loss for checkpointing:
    mse_loss = nn.MSELoss()
    model.eval()
    model_output = model.output # Save model output type to undo after done
    model.output = 'pred' # model output is pred for mse loss
    with torch.no_grad():
        losses = []
        corrs = []
        if args.record_E:
            Es = [[] for l in range(model.nb_layers)]
        for X in dataloader:
            # Forward
            X = X.to(device)
            output = model(X)
            # Compute loss
            X_no_t0 = X[:,1:,:,:,:]
            loss = mse_loss(output,X_no_t0)
            corr = correlation(output,X_no_t0)
            # Record loss
            loss_datapoint = loss.data.item()
            corr_datapoint = corr.data.item()
            losses.append(loss_datapoint)
            corrs.append(corr_datapoint)
            # record E
            if args.record_E:
                model.output = 'error'
                errors = model(X)
                E_means = torch.mean(errors.detach(),dim=0)
                for l in range(model.nb_layers):
                    Es[l].append(E_means[l].data.item())
                model.output = 'pred'

    model.train()
    model.output = model_output # Undo model output change to resume training
    if args.record_E:
        mean_Es = [np.mean(Es[l]) for l in range(model.nb_layers)]
        return np.mean(losses),np.mean(corrs),mean_Es
    else:
        return np.mean(losses),np.mean(corrs)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    start_train_time = time.time()
    print("MKL is available: ", torch.backends.mkl.is_available())
    print("MKL DNN is available: ", torch._C.has_mkldnn)
    if args.record_E:
        msg = "Must be using PredNet with E loss to record E"
        model_has_E = args.model_type in ['PredNet','MultiConvLSTM',
                                          'LadderNet','StackedConvLSTM']
        assert model_has_E and args.loss == 'E', msg
    if args.model_type in ['PredNet','LadderNet']:
        if args.local_grad and not args.no_A_conv:
            print("WARNING: TRAINING WITH LOCAL GRADIENTS DOES NOT MAKE SENSE "
                   "WHEN THERE ARE CONVOLUTIONAL LAYERS IN A CELLS. DON'T GIVE "
                   "LAYERS CONTROL OVER THEIR OWN TARGETS!")
    main(args)
    print("Total training time: ", time.time() - start_train_time)
